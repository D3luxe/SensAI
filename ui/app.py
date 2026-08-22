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
                        "▶️ Resume Training" if init_status["paused"] else "⏸️ Pause Training",
                        variant="secondary",
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
                            1.0, 5.0, value=rew_cfg.get("touch_aerial_flip_multi", 2.5), step=0.1,
                            label="Jump / Flip / Aerial Strike Multiplier",
                            info="🔗 Connected to [Ball Contact]: Multiplies base hit reward when jumping, front-flipping, speed-flipping, or aerial striking ANY ball (ground or air)."
                        )
                    with gr.Column():
                        dodge_rush_multi_slider = gr.Slider(
                            1.0, 3.0, value=rew_cfg.get("dodge_rush_multi", 1.5), step=0.1,
                            label="Dodge & Speed-Flip Impulse Multiplier",
                            info="🔗 Connected to [Speed Toward Ball]: Multiplies closing speed reward when executing a front-flip or speed-flip directly toward the ball."
                        )

                # SECTION 3: DIRECTIONAL GUIDANCE
                gr.Markdown("### 🎯 3. Positional Flow & Directional Guidance (Micro-Scaled Per-Step Rate)")
                with gr.Row():
                    with gr.Column():
                        ball_vel_toward_goal_slider = gr.Slider(
                            0.0, 0.5, value=rew_cfg.get("ball_vel_toward_goal_weight", 0.08), step=0.01,
                            label="Ball Velocity Toward Opponent Goal",
                            info="Per-step reward for propelling the ball toward the opponent net (penalizes hitting toward own goal)."
                        )
                        speed_toward_ball_slider = gr.Slider(
                            0.0, 0.3, value=rew_cfg.get("speed_toward_ball_weight", 0.05), step=0.01,
                            label="Speed Toward Ball (Front Bumper)",
                            info="Per-step reward for closing distance to the ball with the nose pointing at the ball."
                        )
                        kickoff_slider = gr.Slider(
                            0.0, 0.3, value=rew_cfg.get("kickoff_weight", 0.05), step=0.01,
                            label="Kickoff Speed Rush Weight",
                            info="Per-step bonus for accelerating at max velocity during the initial kickoff rush."
                        )
                        face_ball_slider = gr.Slider(
                            0.0, 0.1, value=rew_cfg.get("face_ball_weight", 0.02), step=0.005,
                            label="Face Ball Alignment (Velocity-Gated)",
                            info="Per-step reward for aligning the nose with the ball while actively driving toward it (>350 uu/s)."
                        )
                        aerial_height_slider = gr.Slider(
                            0.0, 0.3, value=rew_cfg.get("aerial_height_weight", 0.05), step=0.01,
                            label="Airborne Ball Intercept Height",
                            info="Per-step bonus for aerial rising when challenging high airborne balls (Z > 140 uu)."
                        )

                    with gr.Column():
                        behind_ball_slider = gr.Slider(
                            0.0, 0.2, value=rew_cfg.get("behind_ball_weight", 0.03), step=0.005,
                            label="Goal-Side Rotation (Behind Ball)",
                            info="Per-step reward for staying between the ball and defending goal; stops defensive over-committing."
                        )
                        possession_slider = gr.Slider(
                            0.0, 0.2, value=rew_cfg.get("possession_weight", 0.04), step=0.005,
                            label="Possession & Dribble Control",
                            info="Per-step reward for carrying, dribbling, and matching ball speed (<350 uu distance)."
                        )
                        defensive_pos_slider = gr.Slider(
                            0.0, 0.2, value=rew_cfg.get("defensive_position_weight", 0.03), step=0.005,
                            label="Defensive Line Goalkeeping",
                            info="Per-step reward for positioning along the line between defending net and ball when defending."
                        )
                        save_boost_slider = gr.Slider(
                            0.0, 0.1, value=rew_cfg.get("save_boost_weight", 0.02), step=0.005,
                            label="Boost Tank Retention (SaveBoost sqrt Curve)",
                            info="Per-step concave reward: sqrt(boost / 100) encouraging maintaining healthy tank reserves without hoarding."
                        )
                        velocity_slider = gr.Slider(
                            0.0, 0.1, value=rew_cfg.get("velocity_weight", 0.02), step=0.005,
                            label="Forward Driving Speed",
                            info="Per-step reward for maintaining forward kinetic speed through the front bumper (penalizes reversing)."
                        )

                with gr.Row():
                    apply_rewards_btn = gr.Button("⚡ Apply Live Reward Weights to Active Run", variant="primary")
                    reset_rewards_btn = gr.Button("🔄 Reset to Standardized Recommended Defaults")

                reward_apply_msg = gr.Markdown("")

            # ---------------------------------------------------------
            # TAB 2: HYPERPARAMETERS & ENVIRONMENT
            # ---------------------------------------------------------
            with gr.TabItem("⚙️ Hyperparameters & Environment"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🧠 PPO Algorithm Hyperparameters")
                        lr_input = gr.Number(
                            value=hp_cfg.get("learning_rate", 0.0003),
                            label="Learning Rate",
                            info="Adam optimizer step size (e.g. 0.0003). Higher learns faster but may destabilize; lower converges steadily."
                        )
                        live_lr_btn = gr.Button("⚡ Apply Learning Rate Live")
                        ent_coef_slider = gr.Slider(
                            0.0, 0.1, value=hp_cfg.get("ent_coef", 0.01), step=0.001,
                            label="Entropy Coefficient (Exploration)",
                            info="Encourages action exploration. High values prevent premature convergence; decays as policy matures."
                        )
                        clip_range_slider = gr.Slider(
                            0.05, 0.4, value=hp_cfg.get("clip_range", 0.2), step=0.01,
                            label="PPO Clip Range (Epsilon)",
                            info="Limits policy ratio changes per update (e.g. 0.2). Prevents overly aggressive policy shifts."
                        )
                        gamma_slider = gr.Slider(
                            0.90, 0.999, value=hp_cfg.get("gamma", 0.99), step=0.001,
                            label="Discount Factor (Gamma)",
                            info="Discount factor (γ) for future rewards. High values (0.99) encourage long-term strategy."
                        )
                        gae_lambda_slider = gr.Slider(
                            0.80, 0.99, value=hp_cfg.get("gae_lambda", 0.95), step=0.01,
                            label="GAE Lambda",
                            info="Generalized Advantage Estimation smoothing factor (λ). 0.95 is standard gold standard for PPO."
                        )
                        batch_size_input = gr.Number(
                            value=hp_cfg.get("batch_size", 2048), precision=0,
                            label="Rollout Batch Size",
                            info="Total steps collected across all environments before triggering a PPO gradient update."
                        )
                        mini_batch_input = gr.Number(
                            value=hp_cfg.get("mini_batch_size", 256), precision=0,
                            label="Mini-Batch Size",
                            info="Sub-sample batch size for each gradient step. Should divide evenly into batch size."
                        )
                        n_epochs_input = gr.Number(
                            value=hp_cfg.get("n_epochs", 4), precision=0,
                            label="Optimization Epochs per Iteration",
                            info="Number of passes over the collected rollout buffer during each PPO update iteration."
                        )

                    with gr.Column():
                        gr.Markdown("#### 🏟️ Simulation & Environment Settings")
                        num_envs_slider = gr.Slider(
                            1, 64, value=env_cfg.get("num_envs", 16), step=1,
                            label="Parallel Vectorized Environments",
                            info="Number of matches running simultaneously in parallel. Higher values scale experience throughput."
                        )
                        tick_skip_slider = gr.Slider(
                            1, 15, value=env_cfg.get("tick_skip", 8), step=1,
                            label="Tick Skip (Physics sub-steps)",
                            info="Physics steps per bot decision (e.g. 8 ticks = ~15 actions/sec). Standard for Rocket League AI."
                        )
                        max_steps_input = gr.Number(
                            value=env_cfg.get("max_episode_steps", 1500), precision=0,
                            label="Max Steps per Episode",
                            info="Maximum episode length before forced kickoff reset (unless a goal ends it sooner)."
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
            # TAB 5: BOT MATCH VISUALIZER & ARENA REPLAY
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
            g_w, c_w, sv_w, as_w, t_w, kft_b, db_w, bs_w, sp_w, bp_w,
            g_spd, t_flip, d_rush,
            bvg_w, s_w, ko_w, f_w, a_w,
            bb_w, p_w, dp_w, sb_w, v_w
        ):
            rewards = {
                # Macro Flat Events
                "goal_weight": float(g_w),
                "concede_weight": float(c_w),
                "save_weight": float(sv_w),
                "aligned_shot_weight": float(as_w),
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
                "behind_ball_weight": float(bb_w),
                "possession_weight": float(p_w),
                "defensive_position_weight": float(dp_w),
                "save_boost_weight": float(sb_w),
                "velocity_weight": float(v_w),
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
                goal_slider, concede_slider, save_slider, aligned_shot_slider, touch_ball_slider, kickoff_first_touch_slider, demo_bump_slider, boost_steal_slider, small_pad_slider, big_pad_slider,
                # Multipliers
                goal_speed_multi_slider, touch_aerial_flip_multi_slider, dodge_rush_multi_slider,
                # Guidance
                ball_vel_toward_goal_slider, speed_toward_ball_slider, kickoff_slider, face_ball_slider, aerial_height_slider,
                behind_ball_slider, possession_slider, defensive_pos_slider, save_boost_slider, velocity_slider
            ],
            outputs=[reward_apply_msg]
        )

        def on_reset_rewards():
            # Standardized defaults:
            # 1. Flat: goal=100.0, concede=-100.0, save=50.0, shot_on_target=25.0, touch=10.0, kickoff_bounty=35.0, demo=15.0, boost_steal=10.0, small_pad=2.0, big_pad=5.0
            # 2. Multipliers: goal_spd=1.5, touch_flip=2.5, dodge_rush=1.5
            # 3. Guidance: bvg=0.08, speed=0.05, kickoff=0.05, face=0.02, aerial=0.05, behind=0.03, poss=0.04, def_pos=0.03, save_boost=0.02, vel=0.02
            return (
                100.0, -100.0, 50.0, 25.0, 10.0, 35.0, 15.0, 10.0, 2.0, 5.0,
                1.5, 2.5, 1.5,
                0.08, 0.05, 0.05, 0.02, 0.05,
                0.03, 0.04, 0.03, 0.02, 0.02
            )

        reset_rewards_btn.click(
            fn=on_reset_rewards,
            outputs=[
                goal_slider, concede_slider, save_slider, aligned_shot_slider, touch_ball_slider, kickoff_first_touch_slider, demo_bump_slider, boost_steal_slider, small_pad_slider, big_pad_slider,
                goal_speed_multi_slider, touch_aerial_flip_multi_slider, dodge_rush_multi_slider,
                ball_vel_toward_goal_slider, speed_toward_ball_slider, kickoff_slider, face_ball_slider, aerial_height_slider,
                behind_ball_slider, possession_slider, defensive_pos_slider, save_boost_slider, velocity_slider
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

    return demo
