"""
Rocket League ML Bot - Comprehensive Gradio Management Dashboard.
Provides real-time training controls, dynamic reward tuning, live metric charts, console stream, and match replay visualizer.
"""

from __future__ import annotations
import os
import sys
import glob
import time
import math
import json
import yaml
import torch
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.process_manager import TrainingProcessManager
from utils.visualizer import simulate_match
from utils.replay_parser import ReplayParser, DEFAULT_DEMO_DIR, get_default_demo_dir
from agent.pretrainer import BehavioralCloningTrainer
from utils.test_runner import run_all_unit_tests, get_cached_or_run_tests, format_test_results_markdown
from utils.diagnostics import (
    extract_rolling_telemetry,
    render_action_biases_plot,
    render_positional_biases_plot,
    generate_ai_coach_diagnostics
)
from utils.scenario_manager import (
    ScenarioManager,
    render_scenario_visual_guide,
    simulate_custom_scenario,
    DEFAULT_CUSTOM_SCENARIOS
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
    pts = glob.glob("checkpoints/*.pt") + glob.glob("checkpoints/**/*.pt")
    # Normalize paths and eliminate duplicates
    pts = list(dict.fromkeys([os.path.normpath(p).replace("\\", "/") for p in pts]))
    if not pts:
        return ["checkpoints/latest_model.pt (none saved yet)"]
    return sorted(pts, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0.0, reverse=True)


def get_available_opponent_options() -> list:
    base_options = ["Heuristic Chaser (Rule-Based Aggressive Bot)"]
    ckpts = get_available_checkpoints()
    return base_options + ckpts


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

.cyber-panel {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
}

.status-callout-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-left: 4px solid #38bdf8;
    border-radius: 6px;
    padding: 10px 16px;
    margin-top: 10px;
    font-size: 0.95em;
    color: #f1f5f9;
    min-height: 44px;
    display: flex;
    align-items: center;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.4);
}
"""


def format_elapsed_time(seconds: Union[int, float]) -> str:
    sec = int(seconds or 0)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
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

    # 1. System & Physics Engine
    try:
        from env.physics_engine import ROCKETSIM_AVAILABLE
    except Exception:
        ROCKETSIM_AVAILABLE = False
    
    engine_str = "C++ RocketSim (High Speed Bullet Physics ~4000+ SPS)" if ROCKETSIM_AVAILABLE else "Pure-Python Fallback (~1100 SPS)"

    # 2. Hyperparameters & Environment Config
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

    # 3. Model Weights & Exploration Health
    model_health: Dict[str, Any] = {}
    latest_ckpt_path = "checkpoints/latest_model.pt"
    if os.path.exists(latest_ckpt_path):
        try:
            ckpt = torch.load(latest_ckpt_path, map_location="cpu")
            sd = ckpt.get("model_state_dict", {})
            obs_dim = ckpt.get("obs_dim", 74)
            act_dim = ckpt.get("act_dim", 8)
            continuous = ckpt.get("continuous_actions", True)

            # Exploration standard deviations
            log_std = sd.get("actor_log_std", None)
            if log_std is not None:
                clamped_std = torch.clamp(log_std, min=-3.0, max=0.0)
                stds = torch.exp(clamped_std).squeeze().tolist()
                channel_names = ["Throttle", "Steer", "Pitch", "Yaw", "Roll", "Jump", "Boost", "Handbrake"]
                exploration_sigmas = {channel_names[i]: round(stds[i], 3) for i in range(min(len(channel_names), len(stds)))}
            else:
                exploration_sigmas = {}

            # Weight norms for stability check
            w_actor = sd.get("actor_mean.weight", None)
            w_critic = sd.get("critic.0.weight", None)
            actor_norm = round(float(torch.norm(w_actor)), 3) if w_actor is not None else None
            critic_norm = round(float(torch.norm(w_critic)), 3) if w_critic is not None else None

            model_health = {
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "continuous_actions": continuous,
                "checkpoint_iteration": ckpt.get("iteration", 0),
                "checkpoint_global_step": ckpt.get("global_step", 0),
                "actor_weight_norm": actor_norm,
                "critic_weight_norm": critic_norm,
                "exploration_std_by_channel": exploration_sigmas
            }
        except Exception as e:
            model_health = {"error": f"Failed to parse checkpoint: {e}"}

    # 4. Telemetry & AI Coach
    telem = extract_rolling_telemetry("logs/history.jsonl", window=10)
    coach_report = generate_ai_coach_diagnostics(telem, active_rewards=rew)

    # 5. Checkpoint Files
    ckpts = glob.glob("checkpoints/*.pt")
    ckpt_info = []
    for c in sorted(ckpts, key=os.path.getmtime, reverse=True)[:6]:
        size_mb = os.path.getsize(c) / (1024 * 1024)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(c)))
        ckpt_info.append(f"- `{os.path.basename(c)}` ({size_mb:.1f} MB, modified {mtime})")
    ckpts_str = "\n".join(ckpt_info) if ckpt_info else "*(No checkpoint files found)*"

    # 6. Recent Logs Tail
    log_tail = mgr.get_logs(max_lines=20)
    log_tail_str = "".join(log_tail).strip() if log_tail else "*(No console output recorded yet)*"

    # 7. Automated Unit Test Suite Diagnostics
    test_diag = get_cached_or_run_tests()
    test_md = format_test_results_markdown(test_diag)

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
| **Observation Dimensions** | `{model_health.get('obs_dim', 74)} features` | **Action Dimensions** | `{model_health.get('act_dim', 8)} channels` |

---

{test_md}

---

### 🧠 AI Coach Behavioral Diagnosis
{coach_report}

---

### 🎛️ Active Macro Potential-Based Reward Weights
* **⚽ Match Macro:** Goal Bounty: `{rew.get('goal_weight', 10.0):+.1f}` | Concede Penalty: `{rew.get('concede_weight', -10.0):+.1f}` | Save / Clear: `{rew.get('save_weight', 3.0):+.1f}`
* **🎯 Field Progression:** Ball-to-Goal Velocity: `{rew.get('ball_to_goal_weight', 1.5):.2f}` | Player-to-Ball Closing Speed: `{rew.get('player_to_ball_weight', 0.8):.2f}`
* **💥 Touch Quality:** Directional Touch: `{rew.get('touch_weight', 1.2):.2f}`
* **⚡ Boost Potential (Necto):** Pad Collection Gain: `{rew.get('boost_gain_weight', 0.6):.2f}` | Ground Waste Penalty: `{rew.get('boost_lose_weight', 0.3):.2f}`

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
        "model_health": model_health,
        "test_suite_diagnostics": {
            "all_passed": test_diag.get("all_passed", False),
            "passed_count": test_diag.get("passed", 0),
            "total_tests": test_diag.get("total_tests", 0),
            "pass_rate_pct": test_diag.get("pass_rate_pct", 0.0),
            "duration_seconds": test_diag.get("duration_seconds", 0.0),
            "subsystems": test_diag.get("subsystems", [])
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
    sc_cfg = default_cfg.get("scenarios", {})

    # Overlay latest live config values so they persist across reloads
    if os.path.exists("config/live_config.json"):
        try:
            with open("config/live_config.json", "r") as f:
                live_data = json.load(f)
                if "rewards" in live_data and isinstance(live_data["rewards"], dict):
                    rew_cfg.update(live_data["rewards"])
                if "scenarios" in live_data and isinstance(live_data["scenarios"], dict):
                    sc_cfg.update(live_data["scenarios"])
                if "learning_rate" in live_data:
                    hp_cfg["learning_rate"] = float(live_data["learning_rate"])
                if "ent_coef" in live_data:
                    hp_cfg["ent_coef"] = float(live_data["ent_coef"])
                if "clip_range" in live_data:
                    hp_cfg["clip_range"] = float(live_data["clip_range"])
                if "baseline_opponent_ratio" in live_data:
                    env_cfg["baseline_opponent_ratio"] = float(live_data["baseline_opponent_ratio"])
                if "baseline_opponent_type" in live_data:
                    env_cfg["baseline_opponent_type"] = str(live_data["baseline_opponent_type"])
                elif "baseline_opponent_model" in live_data:
                    env_cfg["baseline_opponent_type"] = str(live_data["baseline_opponent_model"])
        except Exception:
            pass

    init_status = mgr.get_status_info()

    with gr.Blocks(title="SensAI - Rocket League ML Studio") as demo:
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
                    > **🏆 Macro Potential-Based Reward Architecture (Nexto & Necto Standard):**
                    > * **⚽ Match Macro:** Zero-sum win/loss outcome (Goals `+10.0`, Concedes `-10.0`, Saves `+3.0`).
                    > * **🎯 Ball-to-Goal Progression:** Smooth continuous potential for moving the ball toward opponent net.
                    > * **🏎️ Player-to-Ball Pursuit:** Continuous closing speed towards the ball from anywhere on the pitch.
                    > * **💥 Touch Quality:** Atomic strike reward scaled by touch power and forward goal alignment.
                    > * **⚡ Sqrt-Boost Potential:** Necto square-root boost conservation with ground waste penalty (free aerial flight).
                    """
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🥅 Match Macro Outcomes")
                        goal_slider = gr.Slider(0.0, 30.0, value=float(rew_cfg.get("goal_weight", 20.0)), step=1.0, label="Goal Scored Bounty (+pts)", info="Primary zero-sum win payout.")
                        concede_slider = gr.Slider(-30.0, 0.0, value=float(rew_cfg.get("concede_weight", -20.0)), step=1.0, label="Goal Conceded Penalty (-pts)", info="Defensive urgency deduction.")
                        save_slider = gr.Slider(0.0, 15.0, value=float(rew_cfg.get("save_weight", 3.0)), step=0.5, label="Goal-Line Save & Clear Bounty (+pts)", info="Clearing dangerous shots off defending goal line.")

                    with gr.Column():
                        gr.Markdown("### 🎯 Field Progression & Mechanics")
                        ball_to_goal_slider = gr.Slider(0.0, 5.0, value=float(rew_cfg.get("ball_to_goal_weight", 1.5)), step=0.1, label="Ball-to-Goal Velocity Weight", info="Continuous field progression toward opponent net.")
                        player_to_ball_slider = gr.Slider(0.0, 3.0, value=float(rew_cfg.get("player_to_ball_weight", 0.6)), step=0.1, label="Player-to-Ball Approach & Control Weight", info="Distance-gated speed rush downfield with strike-zone pacing.")
                        jump_bridge_slider = gr.Slider(0.0, 1.0, value=float(rew_cfg.get("jump_bridge_weight", 0.35)), step=0.05, label="Jump & Aerial Takeoff Incentive", info="Takeoff & speed-flip transition bounty (2.0x on elevated aerials).")
                        air_roll_recovery_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("air_roll_recovery_weight", 0.10)), step=0.05, label="Air-Roll & Landing Recovery Weight", info="Rewards wheels-down recovery on descent and aerial alignment.")
                        powerslide_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("powerslide_weight", 0.20)), step=0.05, label="Powerslide & Drift Cut Bounty (+pts)", info="Rewards handbrake powerslides on sharp ground recovery turns.")
                        touch_slider = gr.Slider(0.0, 5.0, value=float(rew_cfg.get("touch_weight", 1.2)), step=0.1, label="Directional Ball Strike Quality", info="Touch impact scaled by speed & goal alignment.")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(r"### ⚡ Boost Potential Engine (Necto $\sqrt{\text{boost}}$)")
                        boost_gain_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("boost_gain_weight", 0.6)), step=0.05, label="Boost Pickup Gain Weight (Sqrt Curve)", info="Scales heavily when empty to encourage pad pickups.")
                    with gr.Column():
                        gr.Markdown("### 🛡️ Ground Conservation Gate")
                        boost_lose_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("boost_lose_weight", 0.3)), step=0.05, label="Ground Boost Waste Penalty Weight", info="Penalizes burning boost on ground (airborne flight is exempt).")

                gr.Markdown("---")
                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=4):
                            gr.Markdown("### 🎲 Dynamic Scenario Setter Distribution (Normalized 100% Group)")
                            gr.Markdown("*Move any slider — the group dynamically rebalances and snaps to 0.01 so the total always equals 100%.*")
                        with gr.Column(scale=1):
                            scenario_total_badge = gr.HTML(
                                """
                                <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                                    <span class="status-badge-running" style="font-size: 1.0em; padding: 6px 16px;">● Total Mix: 100%</span>
                                </div>
                                """
                            )
                    with gr.Row():
                        with gr.Column():
                            kickoff_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("kickoff_prob", 0.28)), step=0.01, label="Kickoff Scenario Probability", info="Standard 1v1 kickoff formations (diagonal, off-center, straight).")
                            replay_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("replay_prob", 0.22)), step=0.01, label="Human Replay Scenario Probability", info="Authentic match situations sampled from ingested replays.")
                            aerial_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("aerial_prob", 0.12)), step=0.01, label="High Aerial Scenario Probability", info="Floating & rising balls (z: 600-1500) for aerial training.")
                            custom_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("custom_prob", 0.10)), step=0.01, label="🎯 Custom Scenarios Probability", info="User-designed custom situations (Opposing 1/3rd powershots, dribbles, custom drills).")

                        with gr.Column():
                            turnaround_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("turnaround_prob", 0.08)), step=0.01, label="Turnaround Recovery Probability", info="Fast downfield spawns moving away from ball to force 180° cuts.")
                            wall_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("wall_prob", 0.10)), step=0.01, label="Wall Play Scenario Probability", info="Sidewall rolling and backboard rebound situations.")
                            save_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("save_prob", 0.10)), step=0.01, label="Goalie Save Scenario Probability", info="Fast opponent shots heading on target into defending net.")

                    custom_sc_count = len(ScenarioManager.get_instance().get_active_scenarios())
                    gr.HTML(
                        f"""
                        <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid #334155; border-radius: 8px; padding: 9px 16px; margin-top: 8px; font-size: 0.9em; display: flex; justify-content: space-between; align-items: center; box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);">
                            <span>📦 <b>Custom Scenarios Distribution Pool:</b> <b style="color: #38bdf8;">{custom_sc_count} Active Scenarios</b> enabled in training rotation.</span>
                            <span style="color: #94a3b8;">Design & test scenarios in the <b>🎯 Custom Scenario Generator</b> tab.</span>
                        </div>
                        """
                    )

                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 👥 Opponent Bot Matchup & Mixup Training")
                            with gr.Row():
                                initial_opp = env_cfg.get("baseline_opponent_type", "heuristic")
                                opp_choices = get_available_opponent_options()
                                default_opp_val = opp_choices[0]
                                for opt in opp_choices:
                                    if initial_opp.lower() in opt.lower() or opt.replace("\\", "/").endswith(initial_opp.replace("\\", "/")):
                                        default_opp_val = opt
                                        break
                                
                                opponent_bot_dropdown = gr.Dropdown(
                                    choices=opp_choices,
                                    value=default_opp_val,
                                    label="Opponent Bot Model / Checkpoint",
                                    info="Select the opponent model to spar against (Necto, Nexto, ActorCritic Checkpoints, or Heuristic Chaser).",
                                    scale=3
                                )
                                refresh_opponent_btn = gr.Button("🔄 Scan", scale=1)

                            baseline_opp_slider = gr.Slider(
                                0.0, 1.0,
                                value=float(env_cfg.get("baseline_opponent_ratio", 0.25)),
                                step=0.01,
                                label="Opponent Bot Matchup Ratio (Mixup Proportion)",
                                info="Proportion of match environments paired against selected opponent bot vs self-play (0% = Pure Self-Play, 100% = Pure Opponent Sparring)."
                            )
                        with gr.Column():
                            gr.Markdown("### 👤 Human Replay Guidance (BC)")
                            bc_weight_slider = gr.Slider(0.0, 1.0, value=float(hp_cfg.get("bc_regularization_weight", 0.10)), step=0.01, label="Replay Guidance Weight (Continuous Trajectory)", info="Nudges vehicle steering, throttle pacing, and aerial orientation from human replays.")
                            bc_decay_input = gr.Number(value=int(hp_cfg.get("bc_decay_steps", 150000000)), precision=0, label="Replay Guidance Decay Horizon (Global Steps)", info="Step threshold over which replay guidance smoothly decays to 0.0.")

                with gr.Row():
                    apply_rewards_btn = gr.Button("⚡ Apply Live Training Dials", variant="primary")
                    reset_rewards_btn = gr.Button("🔄 Reset to Balanced Standard Dials", variant="secondary")

                reward_apply_msg = gr.Markdown("")

            # ---------------------------------------------------------
            # TAB 2: 🎯 CUSTOM SCENARIO GENERATOR & BUILDER
            # ---------------------------------------------------------
            with gr.TabItem("🎯 Custom Scenario Generator"):
                gr.Markdown(
                    """
                    > **🎯 Interactive RocketSim Scenario Generator & Visual Builder:**
                    > * **🗺️ Visual Guide Preview:** Live 2D pitch showing vehicle positions, yaw heading, velocity momentum arrows (green), and ball height & trajectory vectors (orange).
                    > * **⚡ One-Click Presets:** Instant loading for **Opposing 1/3rd Bouncing Powershots / Dribbles**, Breakaway Sprints, Air Dribble setups, and Shadow Defense.
                    > * **💾 Dynamic Persistence:** Saved scenarios automatically join the **Custom Scenarios** pool in the Dynamic Scenario Setter Distribution.
                    """
                )
                sc_mgr = ScenarioManager.get_instance()
                all_scenarios = sc_mgr.get_all_scenarios()
                initial_sc = all_scenarios[0] if all_scenarios else DEFAULT_CUSTOM_SCENARIOS[0]

                with gr.Row():
                    # Left Column: 2D Visual Guide Preview & Simulation Rollout
                    with gr.Column(scale=5):
                        gr.Markdown("### 🗺️ Live 2D Pitch Visual Guide")
                        sc_preview_plot = gr.Plot(
                            value=render_scenario_visual_guide(initial_sc),
                            label="Interactive 2D Pitch Preview"
                        )
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                choices=["(Select Template Preset...)"] + [sc["name"] for sc in DEFAULT_CUSTOM_SCENARIOS],
                                value="(Select Template Preset...)",
                                label="⚡ Quick Template Presets",
                                scale=3
                            )
                            refresh_preview_btn = gr.Button("🔄 Refresh Guide", scale=1)

                        with gr.Accordion("🧪 2-Second Trajectory Rollout Simulation", open=False):
                            gr.Markdown("*Runs 150 steps in RocketSim from this custom scenario with active bot policy to preview physics response.*")
                            sim_scenario_btn = gr.Button("🚀 Simulate Scenario Physics (2s Rollout)", variant="primary")
                            sc_sim_plot = gr.Plot(label="Trajectory Rollout Plot")
                            sc_sim_stats = gr.JSON(label="Rollout Diagnostics")

                    # Right Column: Interactive Parameter Controls
                    with gr.Column(scale=6):
                        with gr.Group():
                            gr.Markdown("### 📝 Scenario Metadata")
                            with gr.Row():
                                sc_id_input = gr.Textbox(label="Scenario ID (Unique Key)", value=initial_sc.get("id", "opposing_third_bouncing_ball"), scale=2)
                                sc_name_input = gr.Textbox(label="Scenario Name", value=initial_sc.get("name", "Opposing 1/3rd Bouncing Powershot / Dribble"), scale=3)
                                sc_enabled_cb = gr.Checkbox(label="Active in Training Pool", value=initial_sc.get("enabled", True), scale=1)
                            sc_desc_input = gr.Textbox(
                                label="Tactical Intent / Description",
                                value=initial_sc.get("description", "Bot spawns in opposing 1/3rd behind bouncing ball."),
                                lines=2
                            )

                        with gr.Group():
                            gr.Markdown("### 🏎️ Bot State (Car 0 / Blue)")
                            with gr.Row():
                                car_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc["car"]["pos"][0]), step=25.0, label="Pos X (Left / Right)")
                                car_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc["car"]["pos"][1]), step=25.0, label="Pos Y (Goal to Goal)")
                                car_pos_z = gr.Slider(17.0, 1600.0, value=float(initial_sc["car"]["pos"][2]), step=10.0, label="Pos Z (Altitude)")
                            with gr.Row():
                                car_yaw = gr.Slider(-180.0, 180.0, value=float(initial_sc["car"].get("yaw", 90.0)), step=5.0, label="Heading / Yaw (deg: 90° = +Y, -90° = -Y)")
                                car_speed = gr.Slider(0.0, 2300.0, value=float(math.hypot(initial_sc["car"]["vel"][0], initial_sc["car"]["vel"][1])), step=25.0, label="Forward Velocity Speed (uu/s)")
                                car_boost = gr.Slider(0.0, 100.0, value=float(initial_sc["car"].get("boost", 50.0)), step=5.0, label="Starting Boost Amount (%)")

                        with gr.Group():
                            gr.Markdown("### ⚽ Ball State")
                            with gr.Row():
                                ball_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc["ball"]["pos"][0]), step=25.0, label="Ball Pos X")
                                ball_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc["ball"]["pos"][1]), step=25.0, label="Ball Pos Y")
                                ball_pos_z = gr.Slider(93.15, 1800.0, value=float(initial_sc["ball"]["pos"][2]), step=10.0, label="Ball Pos Z (Height)")
                            with gr.Row():
                                ball_vel_x = gr.Slider(-2500.0, 2500.0, value=float(initial_sc["ball"]["vel"][0]), step=25.0, label="Ball Vel X (uu/s)")
                                ball_vel_y = gr.Slider(-2500.0, 2500.0, value=float(initial_sc["ball"]["vel"][1]), step=25.0, label="Ball Vel Y (uu/s)")
                                ball_vel_z = gr.Slider(-1500.0, 1500.0, value=float(initial_sc["ball"]["vel"][2]), step=25.0, label="Ball Vel Z (uu/s)")

                        with gr.Group():
                            gr.Markdown("### 👤 Opponent State (Car 1 / Orange)")
                            with gr.Row():
                                opp_mode_radio = gr.Radio(["goalie", "shadow", "custom", "none"], value=initial_sc.get("opponent", {}).get("mode", "goalie"), label="Opponent Placement Mode")
                                opp_boost = gr.Slider(0.0, 100.0, value=float(initial_sc.get("opponent", {}).get("boost", 60.0)), step=5.0, label="Opponent Boost (%)")
                            with gr.Row(visible=(initial_sc.get("opponent", {}).get("mode", "goalie") == "custom")) as opp_custom_row:
                                opp_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc.get("opponent", {}).get("pos", [0, 4800, 17])[0]), step=25.0, label="Custom Opponent Pos X")
                                opp_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc.get("opponent", {}).get("pos", [0, 4800, 17])[1]), step=25.0, label="Custom Opponent Pos Y")
                                opp_yaw = gr.Slider(-180.0, 180.0, value=float(initial_sc.get("opponent", {}).get("yaw", -90.0)), step=5.0, label="Custom Opponent Yaw (deg)")

                        with gr.Group():
                            gr.Markdown("### 🎲 Training Variance & Symmetry")
                            with gr.Row():
                                pos_jitter = gr.Slider(0.0, 300.0, value=float(initial_sc.get("variance", {}).get("pos_jitter", 80.0)), step=10.0, label="Positional Jitter (±uu)", info="Adds natural positional randomness each spawn.")
                                vel_jitter = gr.Slider(0.0, 300.0, value=float(initial_sc.get("variance", {}).get("vel_jitter", 60.0)), step=10.0, label="Velocity Jitter (±uu/s)", info="Adds velocity variance each spawn.")
                                mirror_symmetry = gr.Checkbox(value=bool(initial_sc.get("variance", {}).get("mirror_symmetry", True)), label="Left/Right Mirror Symmetry (50% Chance)", info="Mirrors scenario across X-axis so bot trains left & right sides equally.")

                        with gr.Row():
                            save_scenario_btn = gr.Button("💾 Save / Update Custom Scenario", variant="primary")
                            new_scenario_btn = gr.Button("➕ New / Clear Form", variant="secondary")
                            delete_scenario_btn = gr.Button("🗑️ Delete Scenario", variant="stop")

                        scenario_action_msg = gr.Markdown("")

                gr.Markdown("### 📚 Saved Custom Scenarios Library")
                def build_scenarios_table():
                    items = ScenarioManager.get_instance().get_all_scenarios()
                    rows = []
                    for s in items:
                        bp = s.get("ball", {}).get("pos", [0, 0, 93])
                        cp = s.get("car", {}).get("pos", [0, 0, 17])
                        rows.append([
                            s.get("id", ""),
                            s.get("name", ""),
                            s.get("enabled", True),
                            f"({bp[0]:.0f}, {bp[1]:.0f}, {bp[2]:.0f})",
                            f"({cp[0]:.0f}, {cp[1]:.0f}, {cp[2]:.0f})",
                            s.get("description", "")
                        ])
                    return pd.DataFrame(rows, columns=["ID", "Name", "Active", "Ball Pos", "Car Pos", "Description"]) if rows else pd.DataFrame(columns=["ID", "Name", "Active", "Ball Pos", "Car Pos", "Description"])

                saved_scenarios_table = gr.Dataframe(
                    value=build_scenarios_table(),
                    interactive=False,
                    label="Custom Scenarios Pool"
                )
                with gr.Row():
                    load_scenario_dropdown = gr.Dropdown(
                        choices=[f"{sc['name']} ({sc['id']})" for sc in all_scenarios],
                        value=f"{initial_sc['name']} ({initial_sc['id']})" if all_scenarios else None,
                        label="Select Scenario from Library to Load / Edit",
                        scale=3
                    )
                    load_scenario_btn = gr.Button("📥 Load Selected Scenario", scale=1)
                    refresh_library_btn = gr.Button("🔄 Refresh Library Table", scale=1)

            # ---------------------------------------------------------
            # TAB 3: HYPERPARAMETERS & ENVIRONMENT
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
            # TAB 3: 📁 REPLAY INGESTION & IMITATION PRETRAINER
            # ---------------------------------------------------------
            with gr.TabItem("📁 Replays & Imitation Pretrainer"):
                def build_replay_stats_md():
                    st = ReplayParser().get_pool_stats()
                    has_data = st['total_frames'] > 0
                    badge = '<span class="status-badge-running">● DATASET ACTIVE</span>' if has_data else '<span class="status-badge-stopped">○ EMPTY DATASET</span>'
                    return f"""
                    <div class="cyber-panel" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; margin-bottom: 16px;">
                        <div style="display: flex; align-items: center; gap: 14px;">
                            {badge}
                            <span style="font-size: 1.1em; font-weight: 700; color: #f8fafc;">Active Replay Dataset Pool</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 24px; font-size: 0.95em; color: #cbd5e1;">
                            <span>Active Frames: <b style="color: #38bdf8;">{st['total_frames']:,}</b></span>
                            <span>Estimated Matches: <b style="color: #818cf8;">{st['num_matches']}</b></span>
                            <span>Pool Storage: <b style="color: #34d399;">{st['file_size_mb']} MB</b></span>
                        </div>
                    </div>
                    """

                replay_stats_box = gr.HTML(value=build_replay_stats_md())

                # Section 1: Replay Ingestion Engine
                with gr.Group():
                    gr.Markdown("### 📂 Replay Scanner & Ingestion Engine")
                    with gr.Row():
                        # Left: Local Directory Scanner
                        with gr.Column(scale=1):
                            demo_dir_input = gr.Textbox(
                                value=DEFAULT_DEMO_DIR,
                                label="Local Rocket League Demos Directory",
                                info="Path to your local saved .replay files."
                            )
                            with gr.Row():
                                max_replays_slider = gr.Slider(
                                    10, 1000,
                                    value=int(sc_cfg.get("max_replays_to_ingest", 50)),
                                    step=10,
                                    label="Max Replays to Ingest",
                                    info="Limits batch size to prevent lag."
                                )
                                sort_mode_dropdown = gr.Dropdown(
                                    choices=["Newest First", "Random Sample", "Oldest First"],
                                    value="Newest First",
                                    label="Selection Mode"
                                )
                            scan_demos_btn = gr.Button("📂 Scan & Ingest Local Demos", variant="primary")

                        # Right: Direct Upload Box
                        with gr.Column(scale=1):
                            replay_upload_box = gr.File(
                                file_count="multiple",
                                file_types=[".replay", ".npz", ".json", ".zip"],
                                label="Upload .replay, .zip (containing replays), or .npz files directly"
                            )
                            with gr.Row():
                                upload_ingest_btn = gr.Button("📥 Ingest Uploaded Files", variant="secondary", scale=2)
                                clear_replays_btn = gr.Button("🗑️ Clear Pool", variant="stop", scale=1)

                    ingestion_status_box = gr.HTML(
                        """
                        <div class="status-callout-box">
                            <span style="color: #38bdf8; font-weight: 700; margin-right: 8px;">STATUS:</span>
                            <span>Ready to scan or ingest replays.</span>
                        </div>
                        """
                    )

                gr.Markdown("---")

                # Section 2: Supervised Imitation Pretrainer (Behavioral Cloning)
                with gr.Group():
                    gr.Markdown("### 🎓 Supervised Imitation Pretrainer (Behavioral Cloning)")
                    gr.Markdown(
                        "*Bootstrap your agent with **Grand Champion / SSL baseline mechanics** (kickoffs, speed-flips, aerial reads, powerslide cuts) directly from human replay datasets before PPO reinforcement learning.*"
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            pretrain_epochs_slider = gr.Slider(10, 500, value=100, step=10, label="Pretraining Epochs", info="Number of supervised training passes over the replay dataset.")
                            with gr.Row():
                                pretrain_lr_input = gr.Number(value=0.001, label="Pretrain Learning Rate", info="Adam learning rate for imitation loss.")
                                pretrain_batch_dropdown = gr.Dropdown(choices=[64, 128, 256, 512, 1024], value=256, label="Batch Size")
                            pretrain_base_dropdown = gr.Dropdown(
                                choices=["Initialize Fresh Model (Clean Baseline)"] + get_available_checkpoints(),
                                value="Initialize Fresh Model (Clean Baseline)",
                                label="Target Base Checkpoint"
                            )
                        with gr.Column(scale=1):
                            with gr.Row():
                                run_pretrain_btn = gr.Button("🚀 Run Supervised Imitation Pretraining", variant="primary", scale=2)
                                stop_pretrain_btn = gr.Button("⏹️ Stop Pretraining", variant="stop", scale=1)
                            pretrain_status_box = gr.HTML(
                                """
                                <div class="status-callout-box">
                                    <span style="color: #38bdf8; font-weight: 700; margin-right: 8px;">PRETRAIN STATUS:</span>
                                    <span>Ready to pretrain. Ingest replay files above first.</span>
                                </div>
                                """
                            )
                            pretrain_handoff_box = gr.HTML(
                                """
                                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; margin-top: 10px;">
                                    <span style="color: #94a3b8; font-size: 0.9rem;">
                                        💡 <b>Seamless Handoff to PPO:</b> Pretraining saves directly to <code>checkpoints/latest_model.pt</code> and <code>checkpoints/pretrained_baseline.pt</code>. Once finished, click <b>Start Training</b> in the Top Dashboard to immediately begin PPO reinforcement learning on your pretrained baseline!
                                    </span>
                                </div>
                                """
                            )

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
            # TAB 6: 🔬 DIAGNOSTIC & EVALUATION HUB (CONSOLIDATED)
            # ---------------------------------------------------------
            with gr.TabItem("🔬 Diagnostic & Evaluation Hub"):
                gr.Markdown(
                    """
                    ### 🔬 Unified Diagnostic & Evaluation Hub
                    Single-pane-of-glass workspace for full system health, automated unit test verification, behavioral biases, and 2D match simulation replays.
                    """
                )
                with gr.Tabs():

                    # Sub-Tab 1: System Snapshot & AI Assistant Export
                    with gr.TabItem("📋 System Snapshot & AI Export"):
                        gr.Markdown(
                            """
                            #### 📋 Comprehensive Diagnostic Snapshot & AI Assistant Export
                            Provides real-time process info, active reward weights, hyperparameters, model weights, unit test health, and console output.
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

                    # Sub-Tab 2: Automated Unit Tests & Subsystem Health
                    with gr.TabItem("🧪 Automated Unit Tests & Health"):
                        gr.Markdown(
                            """
                            #### 🧪 Subsystem Unit & Integration Test Verification
                            Run the full programmatic test suite (Physics, Neural Architecture, Scenarios, Replay Parser) to verify environment and bot integrity.
                            """
                        )
                        with gr.Row():
                            run_unit_tests_btn = gr.Button("🧪 Run All Unit Tests", variant="primary")
                        
                        unit_tests_overview_md = gr.Markdown(value=format_test_results_markdown(get_cached_or_run_tests()))
                        unit_tests_stdout = gr.Code(
                            label="Test Runner Output Stream",
                            language="markdown",
                            lines=10,
                            interactive=False
                        )

                    # Sub-Tab 3: Behavioral Biases & AI Coach
                    with gr.TabItem("🧠 Behavioral Biases & AI Coach"):
                        gr.Markdown(
                            """
                            #### 🧠 Action Distributions & Pitch Positional Radar
                            Analyzes controller distributions and spatial heatmap positioning averaged across recent iterations to diagnose bad habits.
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

                    # Sub-Tab 4: 2D Pitch Match Visualizer & Evaluation
                    with gr.TabItem("🎮 2D Pitch Match Visualizer"):
                        gr.Markdown(
                            """
                            #### 🎮 Headless Simulation Match Replay
                            Simulate full matches on the 2D pitch with real-time trajectory tracing and reward breakdown.
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

        # -------------------------------------------------------------
        # EVENT HANDLERS & CALLBACKS
        # -------------------------------------------------------------

        # Dynamic State Synchronizer
        def sync_ui_state(feedback_msg: str = ""):
            status = mgr.get_status_info()
            card_html = build_status_card_html(status, feedback_msg)
            running = status.get("running", False)
            paused = status.get("paused", False)

            start_btn_update = gr.update(
                value="🚀 Start Training" if not running else "🟢 Training Active",
                variant="primary" if not running else "secondary",
                interactive=not running
            )
            pause_btn_update = gr.update(
                value="▶️ Resume Training" if paused else "⏸️ Pause Training",
                variant="primary" if paused else "secondary",
                interactive=running
            )
            stop_btn_update = gr.update(
                value="🛑 Stop Training",
                variant="stop" if running else "secondary",
                interactive=running
            )
            return card_html, start_btn_update, pause_btn_update, stop_btn_update

        # Training Controls
        def on_start(resume_latest: bool = True):
            ckpt = None
            if resume_latest:
                # Find the highest iteration checkpoint available
                candidates = []
                if os.path.exists("checkpoints/latest_model.pt"):
                    candidates.append("checkpoints/latest_model.pt")
                candidates.extend(glob.glob("checkpoints/checkpoint_iter_*.pt"))
                candidates.extend(glob.glob("checkpoints/manual_checkpoint_step_*.pt"))
                
                best_ckpt = None
                best_iter = -1
                for c_path in candidates:
                    try:
                        data = torch.load(c_path, map_location="cpu", weights_only=False)
                        if isinstance(data, dict):
                            it = int(data.get("iteration", 0))
                            if it > best_iter:
                                best_iter = it
                                best_ckpt = c_path
                    except Exception:
                        pass
                ckpt = best_ckpt if best_ckpt else ("checkpoints/latest_model.pt" if os.path.exists("checkpoints/latest_model.pt") else None)

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

        control_outputs = [status_card, start_btn, pause_btn, stop_btn]

        start_btn.click(fn=on_start, inputs=[resume_chk], outputs=control_outputs)
        stop_btn.click(fn=on_stop, outputs=control_outputs)
        pause_btn.click(fn=on_pause, outputs=control_outputs)
        ckpt_btn.click(fn=on_save_checkpoint, outputs=control_outputs)

        # Apply Live Training Dials (Rewards, Scenarios, Opponents, BC Guidance)
        def on_apply_rewards(
            g_w, c_w, sv_w,
            b2g_w, p2b_w, jb_w, ar_w, pw_w, tch_w,
            bg_w, bl_w,
            k_p, r_p, a_p, tr_p, w_p, s_p, c_p,
            opp_bot, base_opp, bc_w, bc_dec
        ):
            rewards = {
                "goal_weight": float(g_w),
                "concede_weight": float(c_w),
                "save_weight": float(sv_w),
                "ball_to_goal_weight": float(b2g_w),
                "player_to_ball_weight": float(p2b_w),
                "jump_bridge_weight": float(jb_w),
                "air_roll_recovery_weight": float(ar_w),
                "powerslide_weight": float(pw_w),
                "touch_weight": float(tch_w),
                "boost_gain_weight": float(bg_w),
                "boost_lose_weight": float(bl_w)
            }
            scenarios = {
                "kickoff_prob": float(k_p),
                "replay_prob": float(r_p),
                "aerial_prob": float(a_p),
                "turnaround_prob": float(tr_p),
                "wall_prob": float(w_p),
                "save_prob": float(s_p),
                "custom_prob": float(c_p)
            }

            clean_opp_type = "heuristic"
            if opp_bot and not str(opp_bot).startswith("Heuristic"):
                clean_opp_type = str(opp_bot).strip()

            payload = {
                "rewards": rewards,
                "scenarios": scenarios,
                "baseline_opponent_type": clean_opp_type,
                "baseline_opponent_ratio": float(base_opp),
                "bc_regularization_weight": float(bc_w),
                "bc_decay_steps": int(bc_dec)
            }
            mgr.update_live_config(payload)
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                base_cfg["rewards"] = rewards
                base_cfg["scenarios"] = scenarios
                if "environment" not in base_cfg:
                    base_cfg["environment"] = {}
                base_cfg["environment"]["baseline_opponent_type"] = clean_opp_type
                base_cfg["environment"]["baseline_opponent_ratio"] = float(base_opp)
                if "hyperparameters" not in base_cfg:
                    base_cfg["hyperparameters"] = {}
                base_cfg["hyperparameters"]["bc_regularization_weight"] = float(bc_w)
                base_cfg["hyperparameters"]["bc_decay_steps"] = int(bc_dec)
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            return f"✅ **Live Training Dials applied and saved at {time.strftime('%H:%M:%S')}!** Opponent: `{os.path.basename(clean_opp_type)}` (Ratio: `{float(base_opp):.0%}`), Rewards & Scenarios updated."

        apply_rewards_btn.click(
            fn=on_apply_rewards,
            inputs=[
                goal_slider, concede_slider, save_slider,
                ball_to_goal_slider, player_to_ball_slider, jump_bridge_slider, air_roll_recovery_slider, powerslide_slider, touch_slider,
                boost_gain_slider, boost_lose_slider,
                kickoff_prob_slider, replay_prob_slider, aerial_prob_slider, turnaround_prob_slider, wall_prob_slider, save_prob_slider, custom_prob_slider,
                opponent_bot_dropdown, baseline_opp_slider, bc_weight_slider, bc_decay_input
            ],
            outputs=[reward_apply_msg]
        )

        refresh_opponent_btn.click(
            fn=lambda: gr.Dropdown(choices=get_available_opponent_options()),
            outputs=[opponent_bot_dropdown]
        )

        # Dynamic 100% Normalized Scenario Rebalancing Handler (7 Scenario Mix)
        def rebalance_scenarios_handler(changed_idx, new_val, k, r, a, tr, w, s, c):
            current_vals = [float(k), float(r), float(a), float(tr), float(w), float(s), float(c)]
            new_val = round(max(0.0, min(1.0, float(new_val))), 2)
            vals = list(current_vals)
            vals[changed_idx] = new_val

            rem = round(1.0 - new_val, 4)
            other_indices = [i for i in range(7) if i != changed_idx]
            other_sum = sum(vals[i] for i in other_indices)

            if other_sum > 1e-6:
                for i in other_indices:
                    vals[i] = round((vals[i] / other_sum) * rem, 2)
            else:
                eq = round(rem / len(other_indices), 2)
                for i in other_indices:
                    vals[i] = eq

            # Fix rounding drift to guarantee exact 1.00 (100%) sum
            drift = round(1.0 - sum(vals), 2)
            if abs(drift) > 1e-4:
                best_other = max(other_indices, key=lambda i: vals[i])
                vals[best_other] = round(vals[best_other] + drift, 2)

            out = [round(v, 2) for v in vals]
            tot_pct = int(round(sum(out) * 100))
            badge_html = f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                <span class="status-badge-running" style="font-size: 1.0em; padding: 6px 16px;">● Total Mix: {tot_pct}%</span>
            </div>
            """
            return out[0], out[1], out[2], out[3], out[4], out[5], out[6], badge_html

        scenario_sliders = [kickoff_prob_slider, replay_prob_slider, aerial_prob_slider, turnaround_prob_slider, wall_prob_slider, save_prob_slider, custom_prob_slider]
        scenario_outputs = [kickoff_prob_slider, replay_prob_slider, aerial_prob_slider, turnaround_prob_slider, wall_prob_slider, save_prob_slider, custom_prob_slider, scenario_total_badge]

        kickoff_prob_slider.release(fn=lambda v, k, r, a, tr, w, s, c: rebalance_scenarios_handler(0, v, k, r, a, tr, w, s, c), inputs=[kickoff_prob_slider] + scenario_sliders, outputs=scenario_outputs)
        replay_prob_slider.release(fn=lambda v, k, r, a, tr, w, s, c: rebalance_scenarios_handler(1, v, k, r, a, tr, w, s, c), inputs=[replay_prob_slider] + scenario_sliders, outputs=scenario_outputs)
        aerial_prob_slider.release(fn=lambda v, k, r, a, tr, w, s, c: rebalance_scenarios_handler(2, v, k, r, a, tr, w, s, c), inputs=[aerial_prob_slider] + scenario_sliders, outputs=scenario_outputs)
        turnaround_prob_slider.release(fn=lambda v, k, r, a, tr, w, s, c: rebalance_scenarios_handler(3, v, k, r, a, tr, w, s, c), inputs=[turnaround_prob_slider] + scenario_sliders, outputs=scenario_outputs)
        wall_prob_slider.release(fn=lambda v, k, r, a, tr, w, s, c: rebalance_scenarios_handler(4, v, k, r, a, tr, w, s, c), inputs=[wall_prob_slider] + scenario_sliders, outputs=scenario_outputs)
        save_prob_slider.release(fn=lambda v, k, r, a, tr, w, s, c: rebalance_scenarios_handler(5, v, k, r, a, tr, w, s, c), inputs=[save_prob_slider] + scenario_sliders, outputs=scenario_outputs)
        custom_prob_slider.release(fn=lambda v, k, r, a, tr, w, s, c: rebalance_scenarios_handler(6, v, k, r, a, tr, w, s, c), inputs=[custom_prob_slider] + scenario_sliders, outputs=scenario_outputs)

        def on_reset_rewards():
            return (
                30.0, -30.0, 12.0,
                1.5, 0.6, 0.35, 0.10, 0.20, 1.2,
                0.6, 0.3,
                0.28, 0.22, 0.12, 0.08, 0.10, 0.10, 0.10,
                0.05, 0.15, 150000000
            )

        reset_rewards_btn.click(
            fn=on_reset_rewards,
            outputs=[
                goal_slider, concede_slider, save_slider,
                ball_to_goal_slider, player_to_ball_slider, jump_bridge_slider, air_roll_recovery_slider, powerslide_slider, touch_slider,
                boost_gain_slider, boost_lose_slider,
                kickoff_prob_slider, replay_prob_slider, aerial_prob_slider, turnaround_prob_slider, wall_prob_slider, save_prob_slider, custom_prob_slider,
                baseline_opp_slider, bc_weight_slider, bc_decay_input
            ]
        )

        # -------------------------------------------------------------
        # CUSTOM SCENARIO GENERATOR EVENT HANDLERS & CALLBACKS
        # -------------------------------------------------------------
        def assemble_scenario_payload(
            sc_id, sc_name, sc_desc, sc_enabled,
            cx, cy, cz, cyaw, cspeed, cboost,
            bx, by, bz, bvx, bvy, bvz,
            opp_mode, opp_boost, ox, oy, oyaw,
            pos_jit, vel_jit, mirror
        ) -> Dict[str, Any]:
            yaw_rad = math.radians(float(cyaw))
            speed = float(cspeed)
            cvx = math.cos(yaw_rad) * speed
            cvy = math.sin(yaw_rad) * speed

            return {
                "id": str(sc_id).strip().lower().replace(" ", "_"),
                "name": str(sc_name).strip(),
                "description": str(sc_desc).strip(),
                "enabled": bool(sc_enabled),
                "car": {
                    "pos": [float(cx), float(cy), float(cz)],
                    "yaw": float(cyaw),
                    "pitch": 0.0,
                    "roll": 0.0,
                    "vel": [cvx, cvy, 0.0],
                    "boost": float(cboost)
                },
                "ball": {
                    "pos": [float(bx), float(by), float(bz)],
                    "vel": [float(bvx), float(bvy), float(bvz)]
                },
                "opponent": {
                    "mode": str(opp_mode),
                    "pos": [float(ox), float(oy), 17.0],
                    "yaw": float(oyaw),
                    "pitch": 0.0,
                    "roll": 0.0,
                    "vel": [0.0, 0.0, 0.0],
                    "boost": float(opp_boost)
                },
                "variance": {
                    "pos_jitter": float(pos_jit),
                    "vel_jitter": float(vel_jit),
                    "mirror_symmetry": bool(mirror)
                }
            }

        scenario_input_components = [
            sc_id_input, sc_name_input, sc_desc_input, sc_enabled_cb,
            car_pos_x, car_pos_y, car_pos_z, car_yaw, car_speed, car_boost,
            ball_pos_x, ball_pos_y, ball_pos_z, ball_vel_x, ball_vel_y, ball_vel_z,
            opp_mode_radio, opp_boost, opp_pos_x, opp_pos_y, opp_yaw,
            pos_jitter, vel_jitter, mirror_symmetry
        ]

        def on_update_visual_preview(*args):
            sc_dict = assemble_scenario_payload(*args)
            return render_scenario_visual_guide(sc_dict)

        # Refresh Guide on demand
        refresh_preview_btn.click(
            fn=on_update_visual_preview,
            inputs=scenario_input_components,
            outputs=[sc_preview_plot]
        )

        # Reactive preview on key sliders
        for comp in [car_pos_x, car_pos_y, car_pos_z, car_yaw, car_speed, car_boost, ball_pos_x, ball_pos_y, ball_pos_z, ball_vel_x, ball_vel_y, ball_vel_z, opp_mode_radio, opp_pos_x, opp_pos_y, opp_yaw]:
            comp.change(
                fn=on_update_visual_preview,
                inputs=scenario_input_components,
                outputs=[sc_preview_plot]
            )

        # Opponent Mode toggle visibility helper
        def on_opp_mode_change(mode):
            return gr.Row(visible=(mode == "custom"))
        opp_mode_radio.change(fn=on_opp_mode_change, inputs=[opp_mode_radio], outputs=[opp_custom_row])

        # Preset Loader Callback
        def on_select_preset_template(preset_name):
            if not preset_name or preset_name.startswith("("):
                return [gr.update()] * 24 + [gr.update()]
            target = None
            for sc in DEFAULT_CUSTOM_SCENARIOS:
                if sc["name"] == preset_name or sc["id"] == preset_name:
                    target = sc
                    break
            if not target:
                return [gr.update()] * 24 + [gr.update()]

            c_vel = target["car"]["vel"]
            speed = math.hypot(c_vel[0], c_vel[1])
            opp = target.get("opponent", {})
            var = target.get("variance", {})

            sc_plot = render_scenario_visual_guide(target)

            return (
                target["id"],
                target["name"],
                target["description"],
                target.get("enabled", True),
                float(target["car"]["pos"][0]),
                float(target["car"]["pos"][1]),
                float(target["car"]["pos"][2]),
                float(target["car"].get("yaw", 90.0)),
                float(speed),
                float(target["car"].get("boost", 50.0)),
                float(target["ball"]["pos"][0]),
                float(target["ball"]["pos"][1]),
                float(target["ball"]["pos"][2]),
                float(target["ball"]["vel"][0]),
                float(target["ball"]["vel"][1]),
                float(target["ball"]["vel"][2]),
                opp.get("mode", "goalie"),
                float(opp.get("boost", 60.0)),
                float(opp.get("pos", [0, 4800, 17])[0]),
                float(opp.get("pos", [0, 4800, 17])[1]),
                float(opp.get("yaw", -90.0)),
                float(var.get("pos_jitter", 80.0)),
                float(var.get("vel_jitter", 60.0)),
                bool(var.get("mirror_symmetry", True)),
                sc_plot
            )

        preset_dropdown.change(
            fn=on_select_preset_template,
            inputs=[preset_dropdown],
            outputs=scenario_input_components + [sc_preview_plot]
        )

        # Save Scenario Callback
        def on_save_custom_scenario(*args):
            sc_dict = assemble_scenario_payload(*args)
            success, msg = ScenarioManager.get_instance().save_scenario(sc_dict)
            table_df = build_scenarios_table()
            all_sc = ScenarioManager.get_instance().get_all_scenarios()
            choices = [f"{s['name']} ({s['id']})" for s in all_sc]
            status_text = f"✅ **{msg}**" if success else f"❌ **{msg}**"
            new_dropdown = gr.Dropdown(choices=choices, value=f"{sc_dict['name']} ({sc_dict['id']})")
            return status_text, table_df, new_dropdown

        save_scenario_btn.click(
            fn=on_save_custom_scenario,
            inputs=scenario_input_components,
            outputs=[scenario_action_msg, saved_scenarios_table, load_scenario_dropdown]
        )

        # Clear / New Scenario Callback
        def on_new_scenario_form():
            new_sc = {
                "id": f"custom_scenario_{int(time.time()) % 10000}",
                "name": "New Custom Training Drill",
                "description": "Custom training situation designed in Sensei ML Studio.",
                "enabled": True,
                "car": {"pos": [0.0, 0.0, 17.0], "yaw": 90.0, "vel": [0.0, 500.0, 0.0], "boost": 50.0},
                "ball": {"pos": [0.0, 1500.0, 93.15], "vel": [0.0, 0.0, 0.0]},
                "opponent": {"mode": "goalie", "pos": [0.0, 4800.0, 17.0], "yaw": -90.0, "boost": 60.0},
                "variance": {"pos_jitter": 50.0, "vel_jitter": 50.0, "mirror_symmetry": True}
            }
            plot = render_scenario_visual_guide(new_sc)
            return (
                new_sc["id"], new_sc["name"], new_sc["description"], True,
                0.0, 0.0, 17.0, 90.0, 500.0, 50.0,
                0.0, 1500.0, 93.15, 0.0, 0.0, 0.0,
                "goalie", 60.0, 0.0, 4800.0, -90.0,
                50.0, 50.0, True,
                plot, "ℹ️ Form cleared for new custom scenario creation."
            )

        new_scenario_btn.click(
            fn=on_new_scenario_form,
            outputs=scenario_input_components + [sc_preview_plot, scenario_action_msg]
        )

        # Delete Scenario Callback
        def on_delete_custom_scenario(sc_id):
            success, msg = ScenarioManager.get_instance().delete_scenario(str(sc_id).strip())
            table_df = build_scenarios_table()
            all_sc = ScenarioManager.get_instance().get_all_scenarios()
            choices = [f"{s['name']} ({s['id']})" for s in all_sc]
            status_text = f"🗑️ **{msg}**" if success else f"⚠️ **{msg}**"
            new_dropdown = gr.Dropdown(choices=choices, value=choices[0] if choices else None)
            return status_text, table_df, new_dropdown

        delete_scenario_btn.click(
            fn=on_delete_custom_scenario,
            inputs=[sc_id_input],
            outputs=[scenario_action_msg, saved_scenarios_table, load_scenario_dropdown]
        )

        # Load from Library Callback
        def on_load_scenario_from_library(selected_choice):
            if not selected_choice or "(" not in selected_choice:
                return [gr.update()] * 24 + [gr.update(), "⚠️ Please select a scenario from the library."]
            sc_id = selected_choice.split("(")[-1].rstrip(")")
            target = ScenarioManager.get_instance().get_scenario(sc_id)
            if not target:
                return [gr.update()] * 24 + [gr.update(), f"⚠️ Scenario '{sc_id}' not found."]

            c_vel = target.get("car", {}).get("vel", [0, 0, 0])
            speed = math.hypot(c_vel[0], c_vel[1])
            opp = target.get("opponent", {})
            var = target.get("variance", {})
            sc_plot = render_scenario_visual_guide(target)

            return (
                target["id"],
                target.get("name", target["id"]),
                target.get("description", ""),
                target.get("enabled", True),
                float(target["car"]["pos"][0]),
                float(target["car"]["pos"][1]),
                float(target["car"]["pos"][2]),
                float(target["car"].get("yaw", 90.0)),
                float(speed),
                float(target["car"].get("boost", 50.0)),
                float(target["ball"]["pos"][0]),
                float(target["ball"]["pos"][1]),
                float(target["ball"]["pos"][2]),
                float(target["ball"]["vel"][0]),
                float(target["ball"]["vel"][1]),
                float(target["ball"]["vel"][2]),
                opp.get("mode", "goalie"),
                float(opp.get("boost", 60.0)),
                float(opp.get("pos", [0, 4800, 17])[0]),
                float(opp.get("pos", [0, 4800, 17])[1]),
                float(opp.get("yaw", -90.0)),
                float(var.get("pos_jitter", 80.0)),
                float(var.get("vel_jitter", 60.0)),
                bool(var.get("mirror_symmetry", True)),
                sc_plot,
                f"📥 Loaded '{target.get('name', target['id'])}' into editor."
            )

        load_scenario_btn.click(
            fn=on_load_scenario_from_library,
            inputs=[load_scenario_dropdown],
            outputs=scenario_input_components + [sc_preview_plot, scenario_action_msg]
        )

        refresh_library_btn.click(
            fn=lambda: (build_scenarios_table(), gr.Dropdown(choices=[f"{s['name']} ({s['id']})" for s in ScenarioManager.get_instance().get_all_scenarios()])),
            outputs=[saved_scenarios_table, load_scenario_dropdown]
        )

        # Simulate Scenario Rollout Callback
        def on_run_scenario_simulation(*args):
            sc_dict = assemble_scenario_payload(*args)
            ckpts = get_available_checkpoints()
            model_path = ckpts[0] if ckpts and os.path.exists(ckpts[0]) else None
            sim_fig, stats = simulate_custom_scenario(sc_dict, model_path=model_path, num_steps=150)
            return sim_fig, stats

        sim_scenario_btn.click(
            fn=on_run_scenario_simulation,
            inputs=scenario_input_components,
            outputs=[sc_sim_plot, sc_sim_stats]
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

        # -------------------------------------------------------------
        # REAL-TIME BACKGROUND REFRESH & LOAD
        # -------------------------------------------------------------
        # Ultra-lightweight Real-Time Top Status Hero Banner Timer (every 2s across all tabs)
        status_timer = gr.Timer(2.0, active=True)
        status_timer.tick(
            fn=sync_ui_state,
            outputs=[status_card, start_btn, pause_btn, stop_btn]
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

        # Scenario & Replay Management Callbacks
        def on_scan_demos(demo_dir, max_replays, sort_mode):
            mode_map = {"Newest First": "newest", "Random Sample": "random", "Oldest First": "oldest"}
            mode = mode_map.get(sort_mode, "newest")
            
            p = ReplayParser()
            count, frames = p.ingest_directory(directory=demo_dir, max_replays=int(max_replays), sort_mode=mode)
            stats_html = build_replay_stats_md()
            
            if count > 0:
                msg = f"""
                <div class="status-callout-box" style="border-left-color: #4ade80;">
                    <span style="color: #4ade80; font-weight: 700; margin-right: 8px;">SUCCESS:</span>
                    <span>Successfully ingested <b>{count}</b> replays (<b style="color: #38bdf8;">{frames:,}</b> new game frames) into training pool!</span>
                </div>
                """
            else:
                msg = f"""
                <div class="status-callout-box" style="border-left-color: #facc15;">
                    <span style="color: #facc15; font-weight: 700; margin-right: 8px;">SCAN RESULT:</span>
                    <span>No valid <code>.replay</code> or dataset files found in <code>{demo_dir}</code>.</span>
                </div>
                """
            return stats_html, msg

        scan_demos_btn.click(
            fn=on_scan_demos,
            inputs=[demo_dir_input, max_replays_slider, sort_mode_dropdown],
            outputs=[replay_stats_box, ingestion_status_box]
        )

        def on_upload_ingest(uploaded_files):
            if not uploaded_files:
                msg = """
                <div class="status-callout-box" style="border-left-color: #facc15;">
                    <span style="color: #facc15; font-weight: 700; margin-right: 8px;">WARNING:</span>
                    <span>No files selected for upload. Drag and drop files first.</span>
                </div>
                """
                return build_replay_stats_md(), msg

            if not isinstance(uploaded_files, (list, tuple)):
                uploaded_files = [uploaded_files]

            p = ReplayParser()
            total_frames = 0
            file_count = 0
            errors = []

            for f in uploaded_files:
                # Robust path resolution for Gradio file objects
                if isinstance(f, str):
                    fpath = f
                elif isinstance(f, dict):
                    fpath = f.get("path") or f.get("name") or ""
                elif hasattr(f, "path") and f.path:
                    fpath = f.path
                elif hasattr(f, "name") and f.name:
                    fpath = f.name
                else:
                    fpath = str(f)

                if not os.path.exists(fpath):
                    errors.append(f"File not found: {fpath}")
                    continue

                ext = os.path.splitext(fpath)[1].lower()
                try:
                    if ext == ".zip":
                        count, frames_cnt = p.ingest_zip(fpath)
                        if count > 0:
                            file_count += count
                            total_frames += frames_cnt
                        else:
                            errors.append(f"No .replay files found inside {os.path.basename(fpath)}")
                    else:
                        frames = p._parse_file(fpath)
                        if frames and len(frames["ball_pos"]) > 0:
                            file_count += 1
                            total_frames += len(frames["ball_pos"])
                            if p.states_buffer is not None:
                                for k in p.states_buffer:
                                    p.states_buffer[k] = np.vstack([p.states_buffer[k], frames[k]])
                            else:
                                p.states_buffer = frames
                        else:
                            errors.append(f"Could not parse valid telemetry from {os.path.basename(fpath)}")
                except Exception as e:
                    errors.append(f"{os.path.basename(fpath)}: {e}")

            if file_count > 0:
                p.save_pool()
                stats_html = build_replay_stats_md()
                msg = f"""
                <div class="status-callout-box" style="border-left-color: #4ade80;">
                    <span style="color: #4ade80; font-weight: 700; margin-right: 8px;">INGESTION COMPLETE:</span>
                    <span>Successfully parsed & ingested <b>{file_count}</b> replays (<b style="color: #38bdf8;">{total_frames:,}</b> frames)!</span>
                </div>
                """
                return stats_html, msg

            stats_html = build_replay_stats_md()
            err_msg = " | ".join(errors) if errors else "No valid .replay, .zip, or .npz data extracted."
            msg = f"""
            <div class="status-callout-box" style="border-left-color: #f87171;">
                <span style="color: #f87171; font-weight: 700; margin-right: 8px;">ERROR:</span>
                <span>{err_msg}</span>
            </div>
            """
            return stats_html, msg

        upload_ingest_btn.click(
            fn=on_upload_ingest,
            inputs=[replay_upload_box],
            outputs=[replay_stats_box, ingestion_status_box]
        )

        def on_clear_replays():
            p = ReplayParser()
            p.clear_pool()
            stats_html = build_replay_stats_md()
            msg = """
            <div class="status-callout-box" style="border-left-color: #facc15;">
                <span style="color: #facc15; font-weight: 700; margin-right: 8px;">POOL CLEARED:</span>
                <span>Ingested replay dataset pool has been completely reset.</span>
            </div>
            """
            return stats_html, msg

        clear_replays_btn.click(
            fn=on_clear_replays,
            outputs=[replay_stats_box, ingestion_status_box]
        )

        # Pretrainer Callbacks
        bc_trainer = BehavioralCloningTrainer()

        def on_run_pretraining(epochs, lr, batch_size, base_ckpt):
            if bc_trainer.is_running():
                return """
                <div class="status-callout-box" style="border-left-color: #facc15;">
                    <span style="color: #facc15; font-weight: 700; margin-right: 8px;">BUSY:</span>
                    <span>Pretrainer is already running!</span>
                </div>
                """

            chosen_ckpt = None
            if base_ckpt and not base_ckpt.startswith("Initialize Fresh Model"):
                chosen_ckpt = base_ckpt.split(" ")[0]

            res = bc_trainer.train(
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                base_checkpoint=chosen_ckpt
            )
            raw_msg = res.get("message", "Pretraining finished.")
            success = res.get("success", True)
            color = "#4ade80" if success else "#f87171"
            title = "COMPLETED" if success else "FAILED"
            return f"""
            <div class="status-callout-box" style="border-left-color: {color};">
                <span style="color: {color}; font-weight: 700; margin-right: 8px;">{title}:</span>
                <span>{raw_msg}</span>
            </div>
            """

        def on_stop_pretraining():
            if bc_trainer.is_running():
                bc_trainer.request_stop()
                return """
                <div class="status-callout-box" style="border-left-color: #f87171;">
                    <span style="color: #f87171; font-weight: 700; margin-right: 8px;">STOPPED:</span>
                    <span>Imitation pretrainer stop requested.</span>
                </div>
                """
            return """
            <div class="status-callout-box" style="border-left-color: #94a3b8;">
                <span style="color: #94a3b8; font-weight: 700; margin-right: 8px;">IDLE:</span>
                <span>Pretrainer is not currently running.</span>
            </div>
            """

        run_pretrain_btn.click(
            fn=on_run_pretraining,
            inputs=[pretrain_epochs_slider, pretrain_lr_input, pretrain_batch_dropdown, pretrain_base_dropdown],
            outputs=[pretrain_status_box]
        )
        stop_pretrain_btn.click(
            fn=on_stop_pretraining,
            outputs=[pretrain_status_box]
        )

        # Unit Tests Runner Callback
        def on_run_unit_tests():
            res = run_all_unit_tests(verbose=True)
            res_md = format_test_results_markdown(res)
            return res_md, res.get("raw_output", "")

        run_unit_tests_btn.click(
            fn=on_run_unit_tests,
            outputs=[unit_tests_overview_md, unit_tests_stdout]
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
