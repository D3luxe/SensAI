"""
Rocket League ML Bot - Comprehensive Gradio Management Dashboard.
Provides real-time training controls, dynamic reward tuning, live metric charts, console stream, and match replay visualizer.
"""

from __future__ import annotations
import os
import sys
import glob
import time
import json
import yaml
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.process_manager import TrainingProcessManager
from utils.visualizer import simulate_match
from utils.diagnostics import (
    extract_rolling_telemetry,
    render_action_biases_plot,
    render_positional_biases_plot,
    generate_ai_coach_diagnostics
)


def load_yaml_config(path: str = "config/default_config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


def save_yaml_config(cfg: dict, path: str = "config/default_config.yaml"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def get_available_checkpoints() -> list:
    pts = glob.glob("checkpoints/*.pt")
    if not pts:
        return ["checkpoints/latest_model.pt (none saved yet)"]
    return sorted(pts, key=os.path.getmtime, reverse=True)


def create_ui():
    mgr = TrainingProcessManager.get_instance()
    default_cfg = load_yaml_config("config/default_config.yaml")

    hp_cfg = default_cfg.get("hyperparameters", {})
    env_cfg = default_cfg.get("environment", {})
    rew_cfg = default_cfg.get("rewards", {})
    log_cfg = default_cfg.get("logging", {})

    # Overlay latest live config values so they persist across reloads
    if os.path.exists("config/live_config.json"):
        try:
            with open("config/live_config.json", "r") as f:
                live_data = json.load(f)
                if "rewards" in live_data and isinstance(live_data["rewards"], dict):
                    rew_cfg.update(live_data["rewards"])
                if "learning_rate" in live_data:
                    hp_cfg["learning_rate"] = float(live_data["learning_rate"])
                if "ent_coef" in live_data:
                    hp_cfg["ent_coef"] = float(live_data["ent_coef"])
                if "clip_range" in live_data:
                    hp_cfg["clip_range"] = float(live_data["clip_range"])
        except Exception:
            pass

CUSTOM_CSS = """
/* Futuristic Cyber / Rocket League Dark Theme */
.gradio-container {
    max-width: 1440px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

.hero-status-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 22px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    margin-bottom: 12px;
}

.status-badge-running {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid #22c55e;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.9em;
    letter-spacing: 0.5px;
}

.status-badge-paused {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    background: rgba(234, 179, 8, 0.15);
    color: #facc15;
    border: 1px solid #eab308;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.9em;
    letter-spacing: 0.5px;
}

.status-badge-stopped {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    background: rgba(148, 163, 184, 0.15);
    color: #94a3b8;
    border: 1px solid #64748b;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.9em;
    letter-spacing: 0.5px;
}

button.primary-btn {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.action-bar-row {
    align-items: center !important;
    gap: 10px !important;
}
"""


def format_elapsed_time(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def build_status_card_html(status_info: dict, feedback_msg: str = "") -> str:
    running = status_info.get("running", False)
    paused = status_info.get("paused", False)
    pid = status_info.get("pid")
    elapsed = status_info.get("elapsed_seconds", 0)
    metrics = status_info.get("metrics", {})

    elapsed_str = format_elapsed_time(elapsed)
    iter_num = metrics.get("iteration", 0)
    step_num = metrics.get("global_step", 0)
    sps = metrics.get("sps", 0)
    rew = metrics.get("mean_reward", 0.0)

    if running and not paused:
        badge = '<span class="status-badge-running">● RUNNING</span>'
    elif running and paused:
        badge = '<span class="status-badge-paused">❚❚ PAUSED</span>'
    else:
        badge = '<span class="status-badge-stopped">○ STOPPED</span>'

    html = f"""
    <div class="hero-status-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 16px;">
                {badge}
                <span style="color: #94a3b8; font-size: 0.95em;">PID: <b style="color: #f1f5f9;">{pid if pid else 'None'}</b></span>
                <span style="color: #94a3b8; font-size: 0.95em;">Elapsed: <b style="color: #f1f5f9;">{elapsed_str}</b></span>
            </div>
            <div style="display: flex; align-items: center; gap: 20px; color: #cbd5e1; font-size: 0.95em;">
                <span>Iteration: <b style="color: #38bdf8;">{iter_num:,}</b></span>
                <span>Global Steps: <b style="color: #818cf8;">{step_num:,}</b></span>
                <span>Speed: <b style="color: #34d399;">{sps:,} SPS</b></span>
                <span>Mean Reward: <b style="color: {'#4ade80' if rew >= 0 else '#f87171'};">{rew:+.2f}</b></span>
            </div>
        </div>
        {f'<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #334155; color: #60a5fa; font-size: 0.9em;">{feedback_msg}</div>' if feedback_msg else ''}
    </div>
    """
    return html


def build_full_diagnostic_export() -> tuple[str, str]:
    """
    Assembles a comprehensive, single-source-of-truth diagnostic summary of the entire bot training system.
    Returns:
        (formatted_overview_markdown, copy_paste_export_string)
    """
    mgr = TrainingProcessManager.get_instance()
    status = mgr.get_status_info()
    running = status.get("running", False)
    paused = status.get("paused", False)
    pid = status.get("pid", "None")
    elapsed = status.get("elapsed_seconds", 0)
    elapsed_str = format_elapsed_time(elapsed)
    metrics = status.get("metrics", {})

    # 1. System & Engine
    try:
        from env.physics_engine import ROCKETSIM_AVAILABLE
    except Exception:
        ROCKETSIM_AVAILABLE = False
    
    engine_str = "C++ RocketSim (High Speed Bullet Physics ~5000+ SPS)" if ROCKETSIM_AVAILABLE else "Pure-Python Fallback (~1100 SPS)"

    # 2. Configs
    default_cfg = load_yaml_config("config/default_config.yaml")
    hp = default_cfg.get("hyperparameters", {})
    env = default_cfg.get("environment", {})
    rew = default_cfg.get("rewards", {})
    
    # Overlay live config
    if os.path.exists("config/live_config.json"):
        try:
            with open("config/live_config.json", "r") as f:
                ld = json.load(f)
                if "rewards" in ld and isinstance(ld["rewards"], dict):
                    rew.update(ld["rewards"])
                if "learning_rate" in ld:
                    hp["learning_rate"] = float(ld["learning_rate"])
                if "ent_coef" in ld:
                    hp["ent_coef"] = float(ld["ent_coef"])
                if "clip_range" in ld:
                    hp["clip_range"] = float(ld["clip_range"])
        except Exception:
            pass

    # 3. Telemetry & AI Coach
    telem = extract_rolling_telemetry("logs/history.jsonl", window=8)
    coach_report = generate_ai_coach_diagnostics(telem, active_rewards=rew)

    # 4. Checkpoint Files
    ckpts = glob.glob("checkpoints/*.pt")
    ckpt_info = []
    for c in sorted(ckpts, key=os.path.getmtime, reverse=True)[:6]:
        size_mb = os.path.getsize(c) / (1024 * 1024)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(c)))
        ckpt_info.append(f"- `{os.path.basename(c)}` ({size_mb:.1f} MB, modified {mtime})")
    ckpts_str = "\n".join(ckpt_info) if ckpt_info else "*(No checkpoint files found)*"

    # 5. Recent Logs Tail
    log_tail = mgr.get_logs(max_lines=20)
    log_tail_str = "".join(log_tail).strip() if log_tail else "*(No console output recorded yet)*"

    # Formatted Markdown Overview for UI Display
    run_state_badge = "🟢 RUNNING" if (running and not paused) else ("⏸️ PAUSED" if (running and paused) else "🛑 STOPPED")
    
    overview_md = f"""
### 📊 SensAI Live Diagnostic Dashboard

| Metric / Parameter | Current Value | Metric / Parameter | Current Value |
| :--- | :--- | :--- | :--- |
| **Process Status** | `{run_state_badge}` | **Training Speed** | `{metrics.get('sps', 0):,} SPS` |
| **Process PID** | `{pid}` | **Elapsed Runtime** | `{elapsed_str}` |
| **Physics Engine** | `{engine_str}` | **Learning Rate** | `{hp.get('learning_rate', 3e-4)}` |
| **Current Iteration** | `{metrics.get('iteration', 0):,}` | **Mean Reward** | `{metrics.get('mean_reward', 0.0):+.2f} pts` |
| **Global Timesteps** | `{metrics.get('global_step', 0):,}` | **Policy Entropy** | `{metrics.get('entropy', 0.0):.4f}` |
| **Policy Loss** | `{metrics.get('policy_loss', 0.0):.5f}` | **Value Loss** | `{metrics.get('value_loss', 0.0):.4f}` |
| **Ball Touches / Rollout** | `{metrics.get('ball_touches', 0.0):.1f}` | **Goals / Rollout** | `{metrics.get('goals', 0)}` |

---

### 🧠 AI Coach Behavioral Diagnosis
{coach_report}

---

### 💾 Available Checkpoints
{ckpts_str}
"""

    # Comprehensive Snapshot for AI Assistant (Copy-Paste string)
    snapshot = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "process_status": {
            "running": running,
            "paused": paused,
            "pid": pid,
            "elapsed_seconds": elapsed,
            "elapsed_formatted": elapsed_str,
            "physics_engine": engine_str,
        },
        "convergence_metrics": {
            "iteration": metrics.get("iteration", 0),
            "global_step": metrics.get("global_step", 0),
            "sps": metrics.get("sps", 0),
            "mean_reward": metrics.get("mean_reward", 0.0),
            "policy_loss": metrics.get("policy_loss", 0.0),
            "value_loss": metrics.get("value_loss", 0.0),
            "entropy": metrics.get("entropy", 0.0),
            "ball_touches_per_rollout": metrics.get("ball_touches", 0.0),
            "goals_per_rollout": metrics.get("goals", 0),
        },
        "hyperparameters": hp,
        "environment": env,
        "active_reward_weights": rew,
        "behavioral_telemetry": telem,
        "recent_checkpoints": [os.path.basename(c) for c in ckpts[:6]],
        "recent_console_tail": log_tail_str.split("\n")[-15:] if log_tail_str else []
    }

    export_json = json.dumps(snapshot, indent=2)
    export_box = f"```json\n{export_json}\n```"

    return overview_md, export_box


def create_ui():
    mgr = TrainingProcessManager.get_instance()
    default_cfg = load_yaml_config("config/default_config.yaml")

    hp_cfg = default_cfg.get("hyperparameters", {})
    env_cfg = default_cfg.get("environment", {})
    rew_cfg = default_cfg.get("rewards", {})
    log_cfg = default_cfg.get("logging", {})

    # Overlay latest live config values so they persist across reloads
    if os.path.exists("config/live_config.json"):
        try:
            with open("config/live_config.json", "r") as f:
                live_data = json.load(f)
                if "rewards" in live_data and isinstance(live_data["rewards"], dict):
                    rew_cfg.update(live_data["rewards"])
                if "learning_rate" in live_data:
                    hp_cfg["learning_rate"] = float(live_data["learning_rate"])
                if "ent_coef" in live_data:
                    hp_cfg["ent_coef"] = float(live_data["ent_coef"])
                if "clip_range" in live_data:
                    hp_cfg["clip_range"] = float(live_data["clip_range"])
        except Exception:
            pass

    init_status = mgr.get_status_info()

    with gr.Blocks(title="SensAI - Rocket League ML Studio", css=CUSTOM_CSS) as demo:
        gr.Markdown(
            """
            # 🏎️⚽ SensAI - Rocket League ML Studio
            ### High-Performance Headless Reinforcement Learning with Vectorized PPO & Live Tuning
            """
        )

        # -------------------------------------------------------------
        # TOP STATUS HERO BANNER
        # -------------------------------------------------------------
        status_card = gr.HTML(build_status_card_html(init_status))

        # -------------------------------------------------------------
        # STREAMLINED ACTION CONTROLS
        # -------------------------------------------------------------
        with gr.Row(elem_classes=["action-bar-row"]):
            with gr.Column(scale=2):
                resume_chk = gr.Checkbox(
                    label="Auto-Resume Latest Checkpoint",
                    value=True,
                    info="Resumes from checkpoints/latest_model.pt. Uncheck for fresh run."
                )
            with gr.Column(scale=5):
                with gr.Row():
                    start_btn = gr.Button(
                        "🚀 Start Training" if not init_status["running"] else "🟢 Training Active",
                        variant="primary" if not init_status["running"] else "secondary",
                        interactive=not init_status["running"]
                    )
                    pause_btn = gr.Button(
                        "⏸️ Pause Training" if not init_status.get("paused", False) else "▶️ Resume Training",
                        variant="primary" if init_status.get("paused", False) else "secondary",
                        interactive=init_status["running"]
                    )
                    stop_btn = gr.Button(
                        "🛑 Stop Training",
                        variant="stop" if init_status["running"] else "secondary",
                        interactive=init_status["running"]
                    )
                    ckpt_btn = gr.Button("💾 Save Checkpoint", variant="secondary")
                    tb_btn = gr.Button("📊 TensorBoard", variant="secondary")

        # -------------------------------------------------------------
        # MAIN TAB INTERFACE
        # -------------------------------------------------------------
        with gr.Tabs():

            # ---------------------------------------------------------
            # TAB 1: LIVE REWARD WEIGHTS & MULTIPLIERS
            # ---------------------------------------------------------
            with gr.TabItem("🎛️ Live Reward Weights"):
                gr.Markdown(
                    """
                    > **💡 Standardized Reward Architecture:** 
                    > * **🏆 Flat Macro Events:** High-impact point payouts (50–100+ pts) awarded immediately when executing game-winning actions (Goals, Saves, First Touch).
                    > * **⚡ Tactical Multipliers:** Dynamically amplify base event rewards (e.g. up to 2.5x for jump/flip strikes or supersonic shots).
                    > * **🎯 Continuous Guidance (Per-Step):** Micro-scaled (~0.01–0.08 pts/step) so 500 steps of passive driving never overshadows scoring a goal.
                    """
                )

                # SECTION 1: MACRO EVENTS
                gr.Markdown("### 🏆 1. Match-Winning & Macro Game Events (Flat Point Bonuses)")
                with gr.Row():
                    with gr.Column():
                        goal_slider = gr.Slider(
                            0.0, 300.0, value=rew_cfg.get("goal_weight", 100.0), step=5.0,
                            label="Goal Scored Reward (Base Flat Bonus)",
                            info="Major flat reward granted when the ball enters the opponent goal. The primary objective of the match."
                        )
                        concede_slider = gr.Slider(
                            -300.0, 0.0, value=rew_cfg.get("concede_weight", -100.0), step=5.0,
                            label="Goal Conceded Penalty (Flat Deduction)",
                            info="Penalty applied when the opponent scores in your net. Teaches defensive urgency and shot blocking."
                        )
                        save_slider = gr.Slider(
                            0.0, 150.0, value=rew_cfg.get("save_weight", 50.0), step=5.0,
                            label="Defensive Save & Goal-Line Clear",
                            info="Flat bonus for touching/clearing the ball when it is within the defensive danger zone in front of net."
                        )
                        aligned_shot_slider = gr.Slider(
                            0.0, 100.0, value=rew_cfg.get("aligned_shot_weight", 25.0), step=5.0,
                            label="Shot on Target (Goal-Bound Hit)",
                            info="Major flat event bounty granted when striking the ball directly on-target into the opponent net (would score if unsaved)."
                        )
                        ground_to_air_setup_slider = gr.Slider(
                            0.0, 30.0, value=rew_cfg.get("ground_to_air_setup_weight", 8.0), step=1.0,
                            label="Ground-to-Air Setup Pop Bounty",
                            info="Flat bounty awarded when popping a ground ball upward with high vertical velocity into a self-pass."
                        )
                        wall_aerial_launch_slider = gr.Slider(
                            0.0, 30.0, value=rew_cfg.get("wall_aerial_launch_weight", 12.0), step=1.0,
                            label="Wall-to-Air Launch Bounty",
                            info="Flat bounty awarded for popping the ball off the sidewall and jumping off the wall into mid-air pursuit."
                        )
                    with gr.Column():
                        touch_ball_slider = gr.Slider(
                            0.0, 50.0, value=rew_cfg.get("touch_ball_weight", 10.0), step=1.0,
                            label="Ball Contact Base Hit Reward",
                            info="Flat reward granted every time the car makes physical contact with the ball."
                        )
                        kickoff_first_touch_slider = gr.Slider(
                            0.0, 100.0, value=rew_cfg.get("kickoff_first_touch_bonus", 35.0), step=5.0,
                            label="Kickoff First-Touch Bounty",
                            info="Massive flat bounty granted to the first bot that impacts the ball on kickoff. Heavily drives kickoff aggression."
                        )
                        demo_bump_slider = gr.Slider(
                            0.0, 50.0, value=rew_cfg.get("demo_bump_weight", 15.0), step=1.0,
                            label="Demolitions & Heavy Bumps",
                            info="Flat bonus for high-speed bumps and demolitions against opponent cars. Promotes physical awareness."
                        )
                        boost_steal_slider = gr.Slider(
                            0.0, 50.0, value=rew_cfg.get("boost_steal_weight", 10.0), step=1.0,
                            label="Opponent Big Boost Steal",
                            info="Flat bonus for collecting 100 boost pads in the opponent half to starve their boost reserves."
                        )
                        small_pad_slider = gr.Slider(
                            0.0, 10.0, value=rew_cfg.get("small_pad_weight", 2.0), step=0.5,
                            label="Small Boost Pad Pickup (+12 Boost)",
                            info="Flat bounty granted immediately upon running over a small boost pad on rotation."
                        )
                        big_pad_slider = gr.Slider(
                            0.0, 25.0, value=rew_cfg.get("big_pad_weight", 5.0), step=1.0,
                            label="Big Boost Orb Pickup (+100 Boost)",
                            info="Flat bounty granted immediately upon collecting a full 100-boost orb."
                        )

                # SECTION 2: ACTION MULTIPLIERS
                gr.Markdown("### ⚡ 2. Tactical Action Multipliers (Scales Base Events)")
                with gr.Row():
                    with gr.Column():
                        goal_speed_multi_slider = gr.Slider(
                            0.0, 3.0, value=rew_cfg.get("goal_speed_multi", 1.5), step=0.1,
                            label="Goal Shot Speed Multiplier",
                            info="🔗 Connected to [Goal Scored]: Multiplies goal reward by up to (1 + multiplier)x for high-speed supersonic laser shots."
                        )
                        touch_aerial_flip_multi_slider = gr.Slider(
                            0.0, 5.0, value=rew_cfg.get("touch_aerial_flip_multi", 2.5), step=0.1,
                            label="Aerial & Flip Strike Multiplier",
                            info="🔗 Connected to [Ball Contact]: Multiplies touch reward when jumping, aerial dodging, or flipping into the ball."
                        )
                    with gr.Column():
                        dodge_rush_multi_slider = gr.Slider(
                            0.0, 3.0, value=rew_cfg.get("dodge_rush_multi", 1.5), step=0.1,
                            label="Dodge Rush Velocity Multiplier",
                            info="🔗 Connected to [Speed Toward Ball]: Multiplies closing speed reward when speed-flipping/dodging forward towards the ball."
                        )

                # SECTION 3: CONTINUOUS GUIDANCE REWARDS (PER-STEP)
                gr.Markdown("### 🎯 3. Continuous Guidance Rewards (Micro-Scaled Per-Step)")
                with gr.Row():
                    with gr.Column():
                        ball_vel_toward_goal_slider = gr.Slider(
                            0.0, 0.5, value=rew_cfg.get("ball_vel_toward_goal_weight", 0.08), step=0.01,
                            label="Ball Velocity Toward Opponent Goal (Per-Step)",
                            info="Rewards propelling the ball toward the opponent net. Encourages offensive pressure."
                        )
                        speed_toward_ball_slider = gr.Slider(
                            0.0, 0.5, value=rew_cfg.get("speed_toward_ball_weight", 0.05), step=0.01,
                            label="Speed Toward Ball Through Front Bumper (Per-Step)",
                            info="Rewards closing distance directly toward the ball through the car's nose."
                        )
                        kickoff_slider = gr.Slider(
                            0.0, 0.5, value=rew_cfg.get("kickoff_weight", 0.05), step=0.01,
                            label="Kickoff Speed Rush (Per-Step)",
                            info="Early kickoff acceleration bonus to teach fast kickoffs."
                        )
                        face_ball_slider = gr.Slider(
                            0.0, 0.2, value=rew_cfg.get("face_ball_weight", 0.02), step=0.005,
                            label="Facing / Tracking Ball Alignment (Per-Step)",
                            info="Rewards aligning car nose to face the ball when actively moving."
                        )
                        aerial_height_slider = gr.Slider(
                            0.0, 0.5, value=rew_cfg.get("aerial_height_weight", 0.05), step=0.01,
                            label="Airborne Ball Intercept Height (Per-Step)",
                            info="Rewards challenging elevated balls in the air."
                        )
                        air_dribble_carry_slider = gr.Slider(
                            0.0, 0.2, value=rew_cfg.get("air_dribble_carry_weight", 0.06), step=0.005,
                            label="Air-Dribble Velocity Matching & Carry (Per-Step)",
                            info="Rewards continuous speed-matching and guidance toward the opponent net while airborne with the ball."
                        )
                        velocity_slider = gr.Slider(
                            0.0, 0.2, value=rew_cfg.get("velocity_weight", 0.02), step=0.005,
                            label="General Driving Forward Speed (Per-Step)",
                            info="Encourages positive forward momentum, penalizes reversing away."
                        )
                    with gr.Column():
                        behind_ball_slider = gr.Slider(
                            0.0, 0.2, value=rew_cfg.get("behind_ball_weight", 0.03), step=0.005,
                            label="Goal-Side Rotation & Positioning (Per-Step)",
                            info="Rewards staying between the ball and your defending net. Teaches proper defensive rotations."
                        )
                        possession_slider = gr.Slider(
                            0.0, 0.2, value=rew_cfg.get("possession_weight", 0.04), step=0.005,
                            label="Tactical Space Dominance (Per-Step)",
                            info="Rewards maintaining uncontested field control when reaching the ball before the opponent."
                        )
                        dribble_slider = gr.Slider(
                            0.0, 0.2, value=rew_cfg.get("dribble_weight", 0.04), step=0.005,
                            label="Roof Carry & Close Bumper Dribble (Per-Step)",
                            info="Rewards balancing the ball atop the car roof or pushing with precision speed-matching."
                        )
                        defensive_pos_slider = gr.Slider(
                            0.0, 0.2, value=rew_cfg.get("defensive_position_weight", 0.03), step=0.005,
                            label="Defensive Third Coverage (Per-Step)",
                            info="Rewards shadowing the ball in the defensive third to prevent breakaways."
                        )
                        save_boost_slider = gr.Slider(
                            0.0, 0.2, value=rew_cfg.get("save_boost_weight", 0.02), step=0.005,
                            label="Boost Tank Retention (Per-Step)",
                            info="Rewards preserving boost reserves using a concave square root curve."
                        )
                        inactivity_penalty_slider = gr.Slider(
                            0.0, 0.5, value=rew_cfg.get("inactivity_penalty_weight", 0.05), step=0.01,
                            label="Inactivity & Standstill Penalty (Per-Step Deduction)",
                            info="Penalizes sitting stationary or wiggling in place for >1s. Eliminates midfield staring and mutual standstills."
                        )

                with gr.Row():
                    apply_rewards_btn = gr.Button("⚡ Apply Live Reward Weights", variant="primary")
                    reset_rewards_btn = gr.Button("🔄 Reset to Standard Balanced Weights", variant="secondary")

                reward_apply_msg = gr.Markdown("")

            # ---------------------------------------------------------
            # TAB 2: HYPERPARAMETERS & ENVIRONMENT
            # ---------------------------------------------------------
            with gr.TabItem("⚙️ Hyperparameters & Environment"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🧠 PPO Hyperparameters")
                        lr_input = gr.Number(
                            value=hp_cfg.get("learning_rate", 3e-4),
                            label="Learning Rate (Live Tunable)",
                            info="PPO Policy and Value network step size. Can be changed on the fly."
                        )
                        live_lr_btn = gr.Button("⚡ Apply Live Learning Rate", variant="primary")

                        ent_coef_slider = gr.Slider(
                            0.0, 0.05, value=hp_cfg.get("ent_coef", 0.01), step=0.001,
                            label="Entropy Coefficient (Live Tunable)",
                            info="Controls exploration bonus. Higher values encourage discovering new mechanics."
                        )
                        clip_range_slider = gr.Slider(
                            0.05, 0.4, value=hp_cfg.get("clip_range", 0.2), step=0.01,
                            label="PPO Clip Range (Live Tunable)",
                            info="Surrogate clipping bounds (epsilon)."
                        )
                        gamma_slider = gr.Slider(
                            0.9, 0.999, value=hp_cfg.get("gamma", 0.99), step=0.001,
                            label="Discount Factor (Gamma)",
                            info="How much future rewards are valued vs immediate points."
                        )
                        gae_lambda_slider = gr.Slider(
                            0.8, 1.0, value=hp_cfg.get("gae_lambda", 0.95), step=0.01,
                            label="GAE Lambda",
                            info="Generalized Advantage Estimation variance vs bias trade-off."
                        )
                        batch_size_input = gr.Number(
                            value=hp_cfg.get("batch_size", 8192), precision=0,
                            label="Rollout Buffer Batch Size",
                            info="Total steps collected across all environments per iteration."
                        )
                        mini_batch_input = gr.Number(
                            value=hp_cfg.get("mini_batch_size", 512), precision=0,
                            label="Mini-Batch Size",
                            info="Gradient update chunk size."
                        )
                        n_epochs_input = gr.Number(
                            value=hp_cfg.get("n_epochs", 10), precision=0,
                            label="Epochs per Iteration",
                            info="Number of optimization passes per rollout."
                        )

                    with gr.Column():
                        gr.Markdown("### 🏟️ Environment & Simulation")
                        num_envs_slider = gr.Slider(
                            1, 128, value=env_cfg.get("num_envs", 64), step=1,
                            label="Vectorized Environments",
                            info="Parallel Rocket League arena instances simulated simultaneously."
                        )
                        tick_skip_slider = gr.Slider(
                            1, 8, value=env_cfg.get("tick_skip", 8), step=1,
                            label="Tick Skip (Action Repeat)",
                            info="120Hz physics substeps per agent decision (8 skip ≈ 15 decisions/sec)."
                        )
                        max_steps_input = gr.Number(
                            value=env_cfg.get("max_episode_steps", 750), precision=0,
                            label="Max Episode Steps (per Arena)",
                            info="Maximum length before resetting if no goal is scored (750 steps ≈ 50s game time)."
                        )
                        game_mode_dropdown = gr.Dropdown(
                            ["1v1", "2v2", "3v3"], value=env_cfg.get("game_mode", "1v1"),
                            label="Game Mode",
                            info="Match format: 1v1 (Duels), 2v2 (Doubles), or 3v3 (Standard team play)."
                        )
                        checkpoint_interval_input = gr.Number(
                            value=log_cfg.get("checkpoint_interval", 20), precision=0,
                            label="Checkpoint Save Interval (Iterations)",
                            info="How often to save a new checkpoint to disk (every N iterations)."
                        )
                        max_checkpoints_input = gr.Number(
                            value=log_cfg.get("max_checkpoints_to_keep", 5), precision=0,
                            label="Max Checkpoints to Keep (Rolling Retention)",
                            info="Number of recent checkpoint files to preserve. Older numbered checkpoints are automatically pruned to prevent disk bloat."
                        )
                        save_cfg_btn = gr.Button("💾 Save Configuration to YAML", variant="secondary")

                cfg_save_msg = gr.Markdown("")

            # ---------------------------------------------------------
            # TAB 3: REAL-TIME METRICS & MONITORING
            # ---------------------------------------------------------
            with gr.TabItem("📈 Real-Time Metrics"):
                with gr.Row():
                    refresh_metrics_btn = gr.Button("🔄 Refresh Metrics", variant="primary")
                    auto_refresh_chk = gr.Checkbox(label="Auto Refresh (Every 3s)", value=False)

                with gr.Row():
                    kpi_iteration = gr.Textbox(label="Iteration", value="0", interactive=False)
                    kpi_step = gr.Textbox(label="Global Timestep", value="0", interactive=False)
                    kpi_reward = gr.Textbox(label="Mean Episode Reward", value="0.00", interactive=False)
                    kpi_loss = gr.Textbox(label="Policy / Value Loss", value="0.000 / 0.000", interactive=False)
                    kpi_sps = gr.Textbox(label="Steps Per Sec (SPS)", value="0", interactive=False)

                with gr.Row():
                    chart_reward = gr.LinePlot(
                        x="iteration",
                        y="mean_reward",
                        title="Mean Episode Reward per Iteration",
                        tooltip=["iteration", "mean_reward", "global_step"],
                        height=280
                    )
                    chart_losses = gr.LinePlot(
                        x="iteration",
                        y="policy_loss",
                        title="Policy Loss over Iterations",
                        tooltip=["iteration", "policy_loss", "value_loss"],
                        height=280
                    )

                with gr.Row():
                    chart_touches = gr.LinePlot(
                        x="iteration",
                        y="ball_touches",
                        title="Ball Touches per Episode",
                        tooltip=["iteration", "ball_touches", "goals"],
                        height=280
                    )
                    chart_entropy = gr.LinePlot(
                        x="iteration",
                        y="entropy",
                        title="Policy Entropy (Exploration Decay)",
                        tooltip=["iteration", "entropy"],
                        height=280
                    )

            # ---------------------------------------------------------
            # TAB 4: 🧠 POLICY BIASES & BEHAVIORAL DIAGNOSTICS
            # ---------------------------------------------------------
            with gr.TabItem("🧠 Training Biases & Habit Radar"):
                gr.Markdown(
                    """
                    ### 🧠 Policy Biases & Behavioral Radar
                    Analyzes the bot's action distributions and pitch spatial positioning averaged across recent iterations.
                    Catches bad habits early (corner trapping, donut spinning, jump reluctance, boost starvation) with actionable tuning advice.
                    """
                )
                with gr.Row():
                    diag_window_slider = gr.Slider(
                        1, 25, value=8, step=1,
                        label="Rolling Average Window (Iterations)",
                        info="Number of recent iterations to average across."
                    )
                    refresh_diag_btn = gr.Button("🔄 Refresh Diagnostics", variant="primary")

                with gr.Row():
                    with gr.Column(scale=1):
                        diag_coach_report = gr.Markdown(value="*Click 'Refresh Diagnostics' or run training to view live AI coach analysis.*")
                    with gr.Column(scale=1):
                        diag_action_plot = gr.Plot(label="Action & Control Distributions")
                        diag_position_plot = gr.Plot(label="Pitch Positioning & Vehicle State")

            # ---------------------------------------------------------
            # TAB 5: CONSOLE LOG STREAM
            # ---------------------------------------------------------
            with gr.TabItem("📜 Live Console Logs"):
                with gr.Row():
                    refresh_logs_btn = gr.Button("🔄 Refresh Logs")
                    clear_logs_btn = gr.Button("🧹 Clear Display")
                console_output = gr.TextArea(
                    label="Training Process Output (stdout / stderr)",
                    lines=18,
                    max_lines=25,
                    interactive=False,
                    autoscroll=True
                )

            # ---------------------------------------------------------
            # TAB 6: BOT MATCH VISUALIZER & EVALUATION
            # ---------------------------------------------------------
            with gr.TabItem("🎮 Bot Match Visualizer & Evaluation"):
                gr.Markdown(
                    """
                    ### 🎯 Headless Simulation Match Replay
                    Select a trained checkpoint model and simulate a full match on the 2D pitch with trajectory tracing.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        ckpt_dropdown = gr.Dropdown(
                            choices=get_available_checkpoints(),
                            value=get_available_checkpoints()[0],
                            label="Select Blue Team Checkpoint",
                            info="Select a trained PyTorch model checkpoint (.pt) for the Blue Team."
                        )
                        opponent_mode = gr.Radio(
                            ["Self-Play (Bot vs Itself)", "Baseline Bot (Chase Ball Heuristic)", "Another Checkpoint"],
                            value="Self-Play (Bot vs Itself)",
                            label="Opponent Matchup Type",
                            info="Choose who the bot plays against in the visualizer."
                        )
                        orange_ckpt_dropdown = gr.Dropdown(
                            choices=get_available_checkpoints(),
                            value=get_available_checkpoints()[0],
                            label="Select Orange Team Checkpoint",
                            visible=False,
                            info="Select a different checkpoint for the Orange Team."
                        )
                        refresh_ckpts_btn = gr.Button("🔄 Scan Checkpoint Directory")
                        sim_steps_slider = gr.Slider(
                            100, 1000, value=400, step=50,
                            label="Simulation Steps",
                            info="Duration of match simulation (e.g. 400 steps ≈ 26 seconds of gameplay)."
                        )
                        run_sim_btn = gr.Button("🕹️ Simulate Match & Render Replay", variant="primary")
                        sim_stats_box = gr.Markdown("#### Match Results: Click 'Simulate Match' to evaluate.")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("🗺️ 2D Pitch Trajectories"):
                                visualizer_plot = gr.Plot(label="Top-Down Rocket League Field Trajectory")
                            with gr.TabItem("📊 Match Reward Breakdown"):
                                reward_breakdown_plot = gr.Plot(label="Points Earned by Reward Category")

            # ---------------------------------------------------------
            # TAB 7: 🔬 DIAGNOSTICS & AI SNAPSHOT EXPORT
            # ---------------------------------------------------------
            with gr.TabItem("🔬 Diagnostics & AI Export"):
                gr.Markdown(
                    """
                    ### 🔬 Comprehensive Diagnostics & AI Assistant Export
                    Provides a unified single-pane-of-glass snapshot containing all process info, active reward weights, hyperparameters, convergence metrics, behavioral telemetry, and console output.
                    
                    **💡 How to use:** Click **'🔄 Refresh Diagnostic Snapshot'**, then click the copy icon on the text box below to paste directly into your conversation with Antigravity!
                    """
                )
                with gr.Row():
                    refresh_snapshot_btn = gr.Button("🔄 Refresh Diagnostic Snapshot", variant="primary")

                with gr.Row():
                    with gr.Column(scale=1):
                        diag_overview_md = gr.Markdown(value="*Click 'Refresh Diagnostic Snapshot' to generate live overview.*")
                    with gr.Column(scale=1):
                        diag_export_raw = gr.Code(
                            label="📋 Complete Diagnostic Snapshot (Copy & Paste to Assistant)",
                            language="markdown",
                            lines=22,
                            interactive=False
                        )

        # -------------------------------------------------------------
        # EVENT HANDLERS & CALLBACKS
        # -------------------------------------------------------------

        # Dynamic State Synchronizer
        def sync_ui_state(feedback_msg: str = ""):
            status = mgr.get_status_info()
            card_html = build_status_card_html(status, feedback_msg)
            running = status.get("running", False)
            paused = status.get("paused", False)

            start_btn_update = gr.Button(
                value="🚀 Start Training" if not running else "🟢 Training Active",
                variant="primary" if not running else "secondary",
                interactive=not running
            )
            pause_btn_update = gr.Button(
                value="▶️ Resume Training" if paused else "⏸️ Pause Training",
                variant="primary" if paused else "secondary",
                interactive=running
            )
            stop_btn_update = gr.Button(
                value="🛑 Stop Training",
                variant="stop" if running else "secondary",
                interactive=running
            )
            return card_html, start_btn_update, pause_btn_update, stop_btn_update

        # Training Controls
        def on_start(resume_latest):
            ckpt = "checkpoints/latest_model.pt" if (resume_latest and os.path.exists("checkpoints/latest_model.pt")) else None
            success, msg = mgr.start_training(checkpoint_path=ckpt)
            time.sleep(0.3)
            return sync_ui_state(f"{'✅' if success else '❌'} {msg}")

        def on_stop():
            success, msg = mgr.stop_training()
            time.sleep(0.3)
            return sync_ui_state(f"{'🛑' if success else '⚠️'} {msg}")

        def on_pause():
            success, msg = mgr.toggle_pause()
            return sync_ui_state(f"ℹ️ {msg}")

        def on_save_checkpoint():
            success, msg = mgr.trigger_save_checkpoint()
            return sync_ui_state(f"{'💾' if success else '⚠️'} {msg}")

        def on_tensorboard():
            success, msg = mgr.start_tensorboard()
            return sync_ui_state(f"{'📊' if success else '⚠️'} {msg}")

        control_outputs = [status_card, start_btn, pause_btn, stop_btn]

        start_btn.click(fn=on_start, inputs=[resume_chk], outputs=control_outputs)
        stop_btn.click(fn=on_stop, outputs=control_outputs)
        pause_btn.click(fn=on_pause, outputs=control_outputs)
        ckpt_btn.click(fn=on_save_checkpoint, outputs=control_outputs)
        tb_btn.click(fn=on_tensorboard, outputs=control_outputs)

        # Apply Live Rewards
        def on_apply_rewards(
            g_w, c_w, sv_w, as_w, g2a_w, wal_w, t_w, kft_b, db_w, bs_w, sp_w, bp_w,
            g_spd, t_flip, d_rush,
            bvg_w, s_w, ko_w, f_w, a_w, adc_w,
            bb_w, p_w, dr_w, dp_w, sb_w, v_w, inact_w
        ):
            rewards = {
                # Macro Flat Events
                "goal_weight": float(g_w),
                "concede_weight": float(c_w),
                "save_weight": float(sv_w),
                "aligned_shot_weight": float(as_w),
                "ground_to_air_setup_weight": float(g2a_w),
                "wall_aerial_launch_weight": float(wal_w),
                "touch_ball_weight": float(t_w),
                "kickoff_first_touch_bonus": float(kft_b),
                "demo_bump_weight": float(db_w),
                "boost_steal_weight": float(bs_w),
                "small_pad_weight": float(sp_w),
                "big_pad_weight": float(bp_w),

                # Action Multipliers
                "goal_speed_multi": float(g_spd),
                "touch_aerial_flip_multi": float(t_flip),
                "dodge_rush_multi": float(d_rush),

                # Micro Guidance
                "ball_vel_toward_goal_weight": float(bvg_w),
                "speed_toward_ball_weight": float(s_w),
                "kickoff_weight": float(ko_w),
                "face_ball_weight": float(f_w),
                "aerial_height_weight": float(a_w),
                "air_dribble_carry_weight": float(adc_w),
                "behind_ball_weight": float(bb_w),
                "possession_weight": float(p_w),
                "dribble_weight": float(dr_w),
                "defensive_position_weight": float(dp_w),
                "save_boost_weight": float(sb_w),
                "velocity_weight": float(v_w),
                "inactivity_penalty_weight": float(inact_w),
            }
            mgr.update_live_config({"rewards": rewards})
            # Also persist to default_config.yaml so they reload permanently
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                base_cfg["rewards"] = rewards
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            return f"✅ **Live reward weights applied and saved at {time.strftime('%H:%M:%S')}!** Settings will now persist across reloads."

        apply_rewards_btn.click(
            fn=on_apply_rewards,
            inputs=[
                # Flat Events
                goal_slider, concede_slider, save_slider, aligned_shot_slider, ground_to_air_setup_slider, wall_aerial_launch_slider, touch_ball_slider, kickoff_first_touch_slider, demo_bump_slider, boost_steal_slider, small_pad_slider, big_pad_slider,
                # Multipliers
                goal_speed_multi_slider, touch_aerial_flip_multi_slider, dodge_rush_multi_slider,
                # Guidance
                ball_vel_toward_goal_slider, speed_toward_ball_slider, kickoff_slider, face_ball_slider, aerial_height_slider, air_dribble_carry_slider,
                behind_ball_slider, possession_slider, dribble_slider, defensive_pos_slider, save_boost_slider, velocity_slider, inactivity_penalty_slider
            ],
            outputs=[reward_apply_msg]
        )

        def on_reset_rewards():
            # Standardized defaults:
            return (
                100.0, -100.0, 50.0, 25.0, 8.0, 12.0, 10.0, 35.0, 15.0, 10.0, 2.0, 5.0,
                1.5, 2.5, 1.5,
                0.08, 0.05, 0.05, 0.02, 0.05, 0.06,
                0.03, 0.04, 0.04, 0.03, 0.02, 0.02, 0.05
            )

        reset_rewards_btn.click(
            fn=on_reset_rewards,
            outputs=[
                goal_slider, concede_slider, save_slider, aligned_shot_slider, ground_to_air_setup_slider, wall_aerial_launch_slider, touch_ball_slider, kickoff_first_touch_slider, demo_bump_slider, boost_steal_slider, small_pad_slider, big_pad_slider,
                goal_speed_multi_slider, touch_aerial_flip_multi_slider, dodge_rush_multi_slider,
                ball_vel_toward_goal_slider, speed_toward_ball_slider, kickoff_slider, face_ball_slider, aerial_height_slider, air_dribble_carry_slider,
                behind_ball_slider, possession_slider, dribble_slider, defensive_pos_slider, save_boost_slider, velocity_slider, inactivity_penalty_slider
            ]
        )

        # Apply Live LR
        def on_live_lr(lr_val):
            mgr.update_live_config({"learning_rate": float(lr_val)})
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                if "hyperparameters" not in base_cfg:
                    base_cfg["hyperparameters"] = {}
                base_cfg["hyperparameters"]["learning_rate"] = float(lr_val)
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            return f"✅ **Learning rate set to {lr_val} live and saved permanently.**"

        live_lr_btn.click(fn=on_live_lr, inputs=[lr_input], outputs=[cfg_save_msg])

        # Save Config YAML
        def on_save_yaml(lr, ent, clip, gamma, gae, bs, mbs, n_ep, n_env, t_skip, m_steps, g_mode, ckpt_int, max_ckpts):
            base_cfg = load_yaml_config("config/default_config.yaml")
            base_cfg["hyperparameters"] = {
                "learning_rate": float(lr),
                "ent_coef": float(ent),
                "clip_range": float(clip),
                "gamma": float(gamma),
                "gae_lambda": float(gae),
                "batch_size": int(bs),
                "mini_batch_size": int(mbs),
                "n_epochs": int(n_ep),
            }
            base_cfg["environment"] = {
                "num_envs": int(n_env),
                "tick_skip": int(t_skip),
                "max_episode_steps": int(m_steps),
                "game_mode": str(g_mode),
            }
            if "logging" not in base_cfg:
                base_cfg["logging"] = {}
            base_cfg["logging"]["tensorboard"] = True
            base_cfg["logging"]["save_dir"] = "checkpoints"
            base_cfg["logging"]["log_dir"] = "logs"
            base_cfg["logging"]["checkpoint_interval"] = int(ckpt_int)
            base_cfg["logging"]["max_checkpoints_to_keep"] = int(max_ckpts)

            save_yaml_config(base_cfg)
            return f"✅ **Saved configuration to config/default_config.yaml**"

        save_cfg_btn.click(
            fn=on_save_yaml,
            inputs=[
                lr_input, ent_coef_slider, clip_range_slider, gamma_slider,
                gae_lambda_slider, batch_size_input, mini_batch_input, n_epochs_input,
                num_envs_slider, tick_skip_slider, max_steps_input, game_mode_dropdown,
                checkpoint_interval_input, max_checkpoints_input
            ],
            outputs=[cfg_save_msg]
        )

        # Logs Handler
        def on_get_logs():
            return mgr.get_logs()

        refresh_logs_btn.click(fn=on_get_logs, outputs=[console_output])
        clear_logs_btn.click(fn=lambda: "", outputs=[console_output])

        # Metrics & Charts Handler
        def on_refresh_metrics():
            card_html, start_u, pause_u, stop_u = sync_ui_state()
            status = mgr.get_status_info()
            metrics = status.get("metrics", {})

            iter_val = str(metrics.get("iteration", 0))
            step_val = f"{metrics.get('global_step', 0):,}"
            rew_val = f"{metrics.get('mean_reward', 0.0):+.2f}"
            loss_val = f"{metrics.get('policy_loss', 0.0):.4f} / {metrics.get('value_loss', 0.0):.4f}"
            sps_val = str(metrics.get("sps", 0))

            # Load history dataframe with downsampling to prevent browser lag
            history_file = os.path.join("logs", "history.jsonl")
            if os.path.exists(history_file):
                records = []
                try:
                    with open(history_file, "r") as f:
                        for line in f:
                            if line.strip():
                                records.append(json.loads(line.strip()))
                    if records:
                        df = pd.DataFrame(records)
                        # Downsample to ~300 smooth points so browser charts render instantaneously
                        if len(df) > 300:
                            stride = max(1, len(df) // 300)
                            df_sampled = df.iloc[::stride].copy()
                            if df.index[-1] not in df_sampled.index:
                                df_sampled = pd.concat([df_sampled, df.iloc[[-1]]])
                            df = df_sampled
                    else:
                        df = pd.DataFrame({"iteration": [0], "mean_reward": [0.0], "policy_loss": [0.0], "value_loss": [0.0], "ball_touches": [0.0], "entropy": [0.0]})
                except Exception:
                    df = pd.DataFrame({"iteration": [0], "mean_reward": [0.0], "policy_loss": [0.0], "value_loss": [0.0], "ball_touches": [0.0], "entropy": [0.0]})
            else:
                df = pd.DataFrame({"iteration": [0], "mean_reward": [0.0], "policy_loss": [0.0], "value_loss": [0.0], "ball_touches": [0.0], "entropy": [0.0]})

            return (
                card_html, start_u, pause_u, stop_u,
                iter_val, step_val, rew_val, loss_val, sps_val,
                df, df, df, df
            )

        refresh_metrics_btn.click(
            fn=on_refresh_metrics,
            outputs=[
                status_card, start_btn, pause_btn, stop_btn,
                kpi_iteration, kpi_step, kpi_reward, kpi_loss, kpi_sps,
                chart_reward, chart_losses, chart_touches, chart_entropy
            ]
        )

        # Periodic timer if auto-refresh is active
        timer = gr.Timer(3.0, active=True)
        auto_refresh_chk.change(fn=lambda v: gr.Timer(active=v), inputs=[auto_refresh_chk], outputs=[timer])
        timer.tick(
            fn=on_refresh_metrics,
            outputs=[
                status_card, start_btn, pause_btn, stop_btn,
                kpi_iteration, kpi_step, kpi_reward, kpi_loss, kpi_sps,
                chart_reward, chart_losses, chart_touches, chart_entropy
            ]
        )

        # Initialize UI on page load
        demo.load(fn=sync_ui_state, outputs=[status_card, start_btn, pause_btn, stop_btn])

        # Checkpoints Scanner
        def on_scan_checkpoints():
            ckpts = get_available_checkpoints()
            return gr.Dropdown(choices=ckpts, value=ckpts[0] if ckpts else None), gr.Dropdown(choices=ckpts, value=ckpts[0] if ckpts else None)

        refresh_ckpts_btn.click(fn=on_scan_checkpoints, outputs=[ckpt_dropdown, orange_ckpt_dropdown])

        def on_opp_mode_change(mode):
            return gr.Dropdown(visible=(mode == "Another Checkpoint"))

        opponent_mode.change(fn=on_opp_mode_change, inputs=[opponent_mode], outputs=[orange_ckpt_dropdown])

        # Match Simulator
        def on_run_simulation(blue_choice, opp_mode, orange_choice, steps):
            blue_path = blue_choice.split(" ")[0] if blue_choice else None

            if opp_mode == "Self-Play (Bot vs Itself)":
                orange_path = "same_as_blue"
            elif opp_mode == "Baseline Bot (Chase Ball Heuristic)":
                orange_path = "baseline"
            else:
                orange_path = orange_choice.split(" ")[0] if orange_choice else "baseline"

            pitch_fig, reward_fig, stats = simulate_match(
                blue_model_path=blue_path,
                orange_model_path=orange_path,
                max_steps=int(steps)
            )

            bg = stats["blue_goals"]
            og = stats["orange_goals"]
            bt = stats["blue_touches"]
            ot = stats["orange_touches"]
            b_rew = stats.get("blue_total_reward", 0.0)
            o_rew = stats.get("orange_total_reward", 0.0)
            m_type = stats.get("match_type", "Match")

            result = "🏆 **BLUE BOT WON!**" if bg > og else ("⚠️ **ORANGE WON!**" if og > bg else "🤝 **DRAW MATCH**")

            summary_md = f"""
            ### {result}
            * **Matchup:** {m_type}
            * **Score:** Blue **{bg}** - **{og}** Orange
            * **Reward Points:** Blue **{b_rew:+.1f} pts** | Orange **{o_rew:+.1f} pts**
            * **Ball Touches:** Blue **{bt}** | Orange **{ot}**
            * **Simulated Duration:** {steps} steps ({int(steps / 15)}s game time)
            """
            return pitch_fig, reward_fig, summary_md

        run_sim_btn.click(
            fn=on_run_simulation,
            inputs=[ckpt_dropdown, opponent_mode, orange_ckpt_dropdown, sim_steps_slider],
            outputs=[visualizer_plot, reward_breakdown_plot, sim_stats_box]
        )

        # Behavioral Diagnostics Callbacks
        def on_refresh_diagnostics(window_size):
            telem = extract_rolling_telemetry("logs/history.jsonl", window=int(window_size))
            act_fig = render_action_biases_plot(telem)
            pos_fig = render_positional_biases_plot(telem)
            report = generate_ai_coach_diagnostics(telem, active_rewards={})
            return report, act_fig, pos_fig

        refresh_diag_btn.click(
            fn=on_refresh_diagnostics,
            inputs=[diag_window_slider],
            outputs=[diag_coach_report, diag_action_plot, diag_position_plot]
        )
        diag_window_slider.change(
            fn=on_refresh_diagnostics,
            inputs=[diag_window_slider],
            outputs=[diag_coach_report, diag_action_plot, diag_position_plot]
        )

        # Comprehensive Diagnostics Export Callback
        def on_refresh_full_diagnostics():
            overview_md, export_box = build_full_diagnostic_export()
            return overview_md, export_box

        refresh_snapshot_btn.click(
            fn=on_refresh_full_diagnostics,
            outputs=[diag_overview_md, diag_export_raw]
        )

    return demo
