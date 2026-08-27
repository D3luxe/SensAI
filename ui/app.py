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
import torch
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.process_manager import TrainingProcessManager
from utils.visualizer import simulate_match
from utils.replay_parser import ReplayParser, DEFAULT_DEMO_DIR
from utils.test_runner import run_all_unit_tests, get_cached_or_run_tests, format_test_results_markdown
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
                        goal_slider = gr.Slider(0.0, 30.0, value=float(rew_cfg.get("goal_weight", 10.0)), step=1.0, label="Goal Scored Bounty (+pts)", info="Primary zero-sum win payout.")
                        concede_slider = gr.Slider(-30.0, 0.0, value=float(rew_cfg.get("concede_weight", -10.0)), step=1.0, label="Goal Conceded Penalty (-pts)", info="Defensive urgency deduction.")
                        save_slider = gr.Slider(0.0, 15.0, value=float(rew_cfg.get("save_weight", 3.0)), step=0.5, label="Goal-Line Save & Clear Bounty (+pts)", info="Clearing dangerous shots off defending goal line.")

                    with gr.Column():
                        gr.Markdown("### 🎯 Field Progression & Pursuit")
                        ball_to_goal_slider = gr.Slider(0.0, 5.0, value=float(rew_cfg.get("ball_to_goal_weight", 1.5)), step=0.1, label="Ball-to-Goal Velocity Weight", info="Continuous field progression toward opponent net.")
                        player_to_ball_slider = gr.Slider(0.0, 3.0, value=float(rew_cfg.get("player_to_ball_weight", 0.8)), step=0.1, label="Player-to-Ball Closing Speed Weight", info="Continuous approach speed toward the ball.")
                        touch_slider = gr.Slider(0.0, 5.0, value=float(rew_cfg.get("touch_weight", 1.2)), step=0.1, label="Directional Ball Strike Quality", info="Touch impact scaled by speed & goal alignment.")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(r"### ⚡ Boost Potential Engine (Necto $\sqrt{\text{boost}}$)")
                        boost_gain_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("boost_gain_weight", 0.6)), step=0.05, label="Boost Pickup Gain Weight (Sqrt Curve)", info="Scales heavily when empty to encourage pad pickups.")
                    with gr.Column():
                        gr.Markdown("### 🛡️ Ground Conservation Gate")
                        boost_lose_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("boost_lose_weight", 0.3)), step=0.05, label="Ground Boost Waste Penalty Weight", info="Penalizes burning boost on ground (airborne flight is exempt).")

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
            # TAB 4: 🎯 SCENARIOS & REPLAY INGESTION
            # ---------------------------------------------------------
            with gr.TabItem("🎯 Scenarios & Replays"):
                gr.Markdown(
                    """
                    ### 🎯 Scenario State Setters & Replay Ingestion
                    Train advanced mechanics (Aerials, Wall Plays, Goalie Saves, Replays) from step 0 by injecting realistic game states directly into RocketSim training environments.
                    """
                )
                parser_inst = ReplayParser()
                def build_replay_stats_md():
                    st = parser_inst.get_pool_stats()
                    return f"""
                    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 12px 18px; margin-bottom: 12px;">
                        <h4 style="margin: 0 0 6px 0; color: #38bdf8;">📊 Active Replay Dataset Pool</h4>
                        <div style="display: flex; gap: 24px; font-size: 0.95rem;">
                            <span><b>Active Frames:</b> {st['total_frames']:,}</span>
                            <span><b>Estimated Matches:</b> {st['num_matches']}</span>
                            <span><b>Pool File Size:</b> {st['file_size_mb']} MB</span>
                        </div>
                    </div>
                    """

                replay_stats_box = gr.HTML(value=build_replay_stats_md())

                with gr.Row():
                    # Left Column: Replay Ingestion
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📁 Replay Scanner & Ingestion")
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
                                info="Limits batch size to prevent lag with thousands of replays."
                            )
                            sort_mode_dropdown = gr.Dropdown(
                                choices=["Newest First", "Random Sample", "Oldest First"],
                                value="Newest First",
                                label="Selection Mode"
                            )
                        scan_demos_btn = gr.Button("📂 Scan & Ingest Local Demos", variant="primary")
                        
                        gr.Markdown("##### 📤 Or Upload Replay Files Directly:")
                        replay_upload_box = gr.File(
                            file_count="multiple",
                            file_types=[".replay", ".npz", ".json"],
                            label="Drop .replay, .npz, or .json files here"
                        )
                        upload_ingest_btn = gr.Button("📥 Ingest Uploaded Files", variant="secondary")
                        ingestion_status_box = gr.Markdown("Ready to ingest.")

                    # Right Column: Scenario Probability Weights
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🎛️ Training Scenario Distribution")
                        gr.Markdown("*Configure the frequency of game scenarios generated during environment resets:*")
                        sc_kickoff_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("kickoff_prob", 0.35)), step=0.05, label="Kickoff Scenarios Ratio", info="Standard competitive kickoff formations.")
                        sc_replay_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("replay_prob", 0.25)), step=0.05, label="Human Replay States Ratio", info="Authentic match states sampled from ingested replays.")
                        sc_aerial_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("aerial_prob", 0.15)), step=0.05, label="High Aerial Shots Ratio", info="Floating & rising balls (z: 600-1500) for aerial training.")
                        sc_wall_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("wall_prob", 0.15)), step=0.05, label="Wall & Backboard Play Ratio", info="Sidewall rolling and backboard rebound situations.")
                        sc_save_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("save_prob", 0.10)), step=0.05, label="Goalie Save & Shadow Defense Ratio", info="High threat shots on net testing goal line defense.")
                        
                        apply_scenarios_btn = gr.Button("💾 Apply Scenario Distribution Live", variant="primary")
                        scenarios_feedback_box = gr.Markdown("")

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

        # Apply Live Rewards (Macro Potential Architecture)
        def on_apply_rewards(
            g_w, c_w, sv_w,
            b2g_w, p2b_w, tch_w,
            bg_w, bl_w
        ):
            rewards = {
                "goal_weight": float(g_w),
                "concede_weight": float(c_w),
                "save_weight": float(sv_w),
                "ball_to_goal_weight": float(b2g_w),
                "player_to_ball_weight": float(p2b_w),
                "touch_weight": float(tch_w),
                "boost_gain_weight": float(bg_w),
                "boost_lose_weight": float(bl_w)
            }
            mgr.update_live_config({"rewards": rewards})
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                base_cfg["rewards"] = rewards
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            return f"✅ **Live Macro Reward weights applied and saved at {time.strftime('%H:%M:%S')}!** Settings persist across reloads."

        apply_rewards_btn.click(
            fn=on_apply_rewards,
            inputs=[
                goal_slider, concede_slider, save_slider,
                ball_to_goal_slider, player_to_ball_slider, touch_slider,
                boost_gain_slider, boost_lose_slider
            ],
            outputs=[reward_apply_msg]
        )

        def on_reset_rewards():
            return (
                10.0, -10.0, 3.0,
                1.5, 0.8, 1.2,
                0.6, 0.3
            )

        reset_rewards_btn.click(
            fn=on_reset_rewards,
            outputs=[
                goal_slider, concede_slider, save_slider,
                ball_to_goal_slider, player_to_ball_slider, touch_slider,
                boost_gain_slider, boost_lose_slider
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

        # Scenario & Replay Management Callbacks
        def on_scan_demos(demo_dir, max_replays, sort_mode):
            mode_map = {"Newest First": "newest", "Random Sample": "random", "Oldest First": "oldest"}
            mode = mode_map.get(sort_mode, "newest")
            
            p = ReplayParser()
            count, frames = p.ingest_directory(directory=demo_dir, max_replays=int(max_replays), sort_mode=mode)
            stats_html = build_replay_stats_md()
            
            if count > 0:
                msg = f"✅ **Successfully ingested {count} replays ({frames:,} new game frames) into training pool!**"
            else:
                msg = f"⚠️ No valid `.replay` or dataset files found in `{demo_dir}`."
            return stats_html, msg

        scan_demos_btn.click(
            fn=on_scan_demos,
            inputs=[demo_dir_input, max_replays_slider, sort_mode_dropdown],
            outputs=[replay_stats_box, ingestion_status_box]
        )

        def on_upload_ingest(uploaded_files):
            if not uploaded_files:
                return build_replay_stats_md(), "⚠️ No files selected for upload."

            p = ReplayParser()
            total_frames = 0
            file_count = 0
            for f in uploaded_files:
                fpath = f.name if hasattr(f, "name") else str(f)
                frames = p._parse_file(fpath)
                if frames and len(frames["ball_pos"]) > 0:
                    file_count += 1
                    total_frames += len(frames["ball_pos"])
                    if p.states_buffer is not None:
                        for k in p.states_buffer:
                            p.states_buffer[k] = np.vstack([p.states_buffer[k], frames[k]])
                    else:
                        p.states_buffer = frames

            if file_count > 0:
                p.save_pool()
                stats_html = build_replay_stats_md()
                return stats_html, f"✅ **Successfully parsed & ingested {file_count} files ({total_frames:,} frames)!**"
            return build_replay_stats_md(), "❌ Failed to extract valid game frames from uploaded files."

        upload_ingest_btn.click(
            fn=on_upload_ingest,
            inputs=[replay_upload_box],
            outputs=[replay_stats_box, ingestion_status_box]
        )

        def on_apply_scenarios_weights(k_p, rep_p, aer_p, wall_p, save_p):
            sc_dict = {
                "kickoff_prob": float(k_p),
                "replay_prob": float(rep_p),
                "aerial_prob": float(aer_p),
                "wall_prob": float(wall_p),
                "save_prob": float(save_p),
            }
            # Update live_config.json
            live_path = "config/live_config.json"
            live_data = {}
            if os.path.exists(live_path):
                try:
                    with open(live_path, "r") as f:
                        live_data = json.load(f)
                except Exception:
                    pass
            live_data["scenarios"] = sc_dict
            with open(live_path, "w") as f:
                json.dump(live_data, f, indent=2)

            # Update default_config.yaml
            def_cfg = load_yaml_config("config/default_config.yaml")
            def_cfg["scenarios"] = def_cfg.get("scenarios", {})
            def_cfg["scenarios"].update(sc_dict)
            save_yaml_config(def_cfg, "config/default_config.yaml")

            tot = sum(sc_dict.values())
            return f"✅ **Scenario distribution saved & applied live! (Total weight: {tot:.2f})**"

        apply_scenarios_btn.click(
            fn=on_apply_scenarios_weights,
            inputs=[sc_kickoff_slider, sc_replay_slider, sc_aerial_slider, sc_wall_slider, sc_save_slider],
            outputs=[scenarios_feedback_box]
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
