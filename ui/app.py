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
from typing import Union

from utils.process_manager import TrainingProcessManager
from utils.visualizer import simulate_match
from utils.replay_parser import ReplayParser, DEFAULT_DEMO_DIR, get_default_demo_dir
from agent.pretrainer import BehavioralCloningTrainer
from utils.test_runner import run_all_unit_tests, get_cached_or_run_tests, format_test_results_markdown
from utils.diagnostics import (
    extract_rolling_telemetry,
    render_action_biases_plot,
    render_positional_biases_plot,
    generate_ai_coach_diagnostics,
    render_training_curves_plot,
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
    max-width: 1520px !important;
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

/* Clean Modern Tab Navigation */
.tabs > .tab-nav {
    border-bottom: 1px solid #334155 !important;
    gap: 8px !important;
    padding-bottom: 4px !important;
    margin-bottom: 16px !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
}

.tabs > .tab-nav > button {
    font-size: 0.98em !important;
    font-weight: 600 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 22px !important;
    transition: all 0.2s ease !important;
    color: #94a3b8 !important;
    white-space: nowrap !important;
}

.tabs > .tab-nav > button.selected {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: rgba(56, 189, 248, 0.08) !important;
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
                    hp["learning_rate"] = ld["learning_rate"]
                if "ent_coef" in ld:
                    hp["ent_coef"] = ld["ent_coef"]
                if "clip_range" in ld:
                    hp["clip_range"] = ld["clip_range"]
                if "baseline_opponent_type" in ld:
                    env["baseline_opponent_type"] = ld["baseline_opponent_type"]
                if "baseline_opponent_ratio" in ld:
                    env["baseline_opponent_ratio"] = ld["baseline_opponent_ratio"]
        except Exception:
            pass

    # 3. Model Architecture & Weights
    pts = get_available_checkpoints()
    latest_ckpt = pts[0] if pts else "None"
    model_details = "None loaded"
    if latest_ckpt != "None" and os.path.exists(latest_ckpt):
        try:
            sz = os.path.getsize(latest_ckpt) / (1024 * 1024)
            model_details = f"`{latest_ckpt}` ({sz:.2f} MB)"
        except Exception:
            model_details = f"`{latest_ckpt}`"

    # 4. Unit Tests Status
    test_results = get_cached_or_run_tests(force_refresh=False)
    tests_summary = f"{test_results.get('passed', 0)}/{test_results.get('total', 0)} Passed ({'ALL PASSING' if test_results.get('all_passed') else 'FAILURES DETECTED'})"

    # 5. Telemetry & Coach Analysis
    telem = extract_rolling_telemetry("logs/history.jsonl", window=10)
    coach_report = generate_ai_coach_diagnostics(telem)

    # 6. Recent Logs
    recent_logs = mgr.get_logs(max_lines=30)

    # Build Markdown Export
    export_text = f"""# 🏎️ SensAI Training & System State Snapshot
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Process Status:** {'RUNNING' if running and not paused else ('PAUSED' if paused else 'STOPPED')} (PID: {pid}, Elapsed: {elapsed_str})

## 1. System Health & Unit Tests
* **Physics Engine:** {engine_str}
* **Unit Tests Status:** {tests_summary}
* **Active Model Weights:** {model_details}

## 2. Live Training Metrics (Current Rollout)
* **Iteration:** {metrics.get('iteration', 0):,}
* **Global Steps:** {metrics.get('global_step', 0):,}
* **Throughput:** {metrics.get('sps', 0):,} Steps/Sec
* **Mean Reward:** {metrics.get('mean_reward', 0.0):+.3f}
* **Policy Loss:** {metrics.get('policy_loss', 0.0):.4f} | **Value Loss:** {metrics.get('value_loss', 0.0):.4f} | **Entropy:** {metrics.get('entropy', 0.0):.4f}

## 3. Active Hyperparameters & Opponent Mix
* **Learning Rate:** `{hp.get('learning_rate', 3e-4)}` | **Entropy Coef:** `{hp.get('ent_coef', 0.005)}` | **Clip Range:** `{hp.get('clip_range', 0.2)}`
* **Gamma:** `{hp.get('gamma', 0.99)}` | **GAE Lambda:** `{hp.get('gae_lambda', 0.95)}`
* **Batch Size:** `{hp.get('batch_size', 8192)}` | **Mini-Batch Size:** `{hp.get('mini_batch_size', 512)}` | **Epochs:** `{hp.get('n_epochs', 10)}`
* **Vectorized Envs:** `{env.get('num_envs', 64)}` | **Tick Skip:** `{env.get('tick_skip', 8)}`
* **Baseline Opponent:** `{env.get('baseline_opponent_type', 'heuristic')}` (Matchup Ratio: `{env.get('baseline_opponent_ratio', 0.25):.0%}`)

## 4. Active Reward Weights
```yaml
{yaml.dump(rew, default_flow_style=False).strip()}
```

## 5. AI Coach Behavioral Diagnosis
{coach_report}

## 6. Recent Process Output (Last 30 Lines)
```text
{recent_logs}
```
"""

    overview_md = f"""
### 📋 System Health Overview
* **Status:** `{'RUNNING' if running and not paused else ('PAUSED' if paused else 'STOPPED')}` (PID: `{pid}`)
* **Throughput:** `{metrics.get('sps', 0):,} SPS` | **Mean Reward:** `{metrics.get('mean_reward', 0.0):+.2f}`
* **Unit Tests:** `{tests_summary}`
* **Physics Engine:** `{engine_str}`
* **Active Weights:** {model_details}
* **Opponent Sparring:** `{os.path.basename(str(env.get('baseline_opponent_type', 'heuristic')))}` ({env.get('baseline_opponent_ratio', 0.25):.0%})

*Copy the raw Markdown on the right into your conversation with the AI assistant for instant debugging.*
"""
    return overview_md, export_text


def create_ui():
    mgr = TrainingProcessManager.get_instance()
    bc_trainer = BehavioralCloningTrainer()
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
                if "bc_regularization_weight" in live_data:
                    hp_cfg["bc_regularization_weight"] = float(live_data["bc_regularization_weight"])
                if "bc_decay_steps" in live_data:
                    hp_cfg["bc_decay_steps"] = int(live_data["bc_decay_steps"])
        except Exception:
            pass

    init_status = mgr.get_status_info()

    with gr.Blocks(title="SensAI - Rocket League ML Studio") as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
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
        # 4 STREAMLINED TOP-LEVEL TABS
        # -------------------------------------------------------------
        with gr.Tabs():

            # =========================================================
            # TAB 1: 🏠 LIVE COCKPIT (ALL-IN-ONE HOME DASHBOARD)
            # =========================================================
            with gr.TabItem("🏠 Live Cockpit"):
                gr.Markdown(
                    """
                    > **⚡ Real-Time Training Control Center:**
                    > Modify PPO hyperparameters, opponent bot sparring mix, and core reward weights dynamically on the fly while monitoring real-time loss/reward telemetry and process output.
                    """
                )
                with gr.Row():
                    # Left Column: Live Tuners & Dynamic Dials
                    with gr.Column(scale=5):
                        # Card 1: Live Hyperparameters
                        with gr.Group():
                            gr.Markdown("### 🧠 Live Hyperparameters")
                            with gr.Row():
                                lr_input = gr.Number(
                                    value=hp_cfg.get("learning_rate", 3e-4),
                                    label="Learning Rate (Live Tunable)",
                                    info="PPO Policy & Value step size.",
                                    scale=1
                                )
                                ent_coef_slider = gr.Slider(
                                    0.0, 0.05,
                                    value=hp_cfg.get("ent_coef", 0.005),
                                    step=0.001,
                                    label="Entropy Coef (Exploration Bonus)",
                                    info="Higher values encourage exploring new mechanics.",
                                    scale=2
                                )
                            clip_range_slider = gr.Slider(
                                0.05, 0.4,
                                value=hp_cfg.get("clip_range", 0.2),
                                step=0.01,
                                label="PPO Clip Range (Live Tunable)",
                                info="Surrogate clipping bounds (epsilon)."
                            )
                            live_hp_btn = gr.Button("⚡ Apply Live Hyperparameters", variant="primary")
                            live_hp_msg = gr.Markdown("")

                        # Card 2: Opponent Bot Matchup & Mixup
                        with gr.Group():
                            gr.Markdown("### 👥 Opponent Bot Matchup & Mixup")
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
                                    info="Select model to spar against (Checkpoints or Heuristic Chaser).",
                                    scale=3
                                )
                                refresh_opponent_btn = gr.Button("🔄 Scan", scale=1)

                            baseline_opp_slider = gr.Slider(
                                0.0, 1.0,
                                value=float(env_cfg.get("baseline_opponent_ratio", 0.25)),
                                step=0.01,
                                label="Opponent Matchup Ratio (Mixup Proportion)",
                                info="0% = Pure Self-Play, 100% = Pure Opponent Sparring."
                            )
                            apply_opp_btn = gr.Button("⚡ Apply Opponent Mix", variant="secondary")
                            opp_apply_msg = gr.Markdown("")

                        # Card 3: Quick Live Reward Weights
                        with gr.Group():
                            gr.Markdown("### 🎛️ Quick Live Reward Weights")
                            with gr.Row():
                                goal_slider = gr.Slider(0.0, 30.0, value=float(rew_cfg.get("goal_weight", 20.0)), step=1.0, label="Goal Scored (+pts)")
                                concede_slider = gr.Slider(-30.0, 0.0, value=float(rew_cfg.get("concede_weight", -20.0)), step=1.0, label="Goal Conceded (-pts)")
                                save_slider = gr.Slider(0.0, 15.0, value=float(rew_cfg.get("save_weight", 3.0)), step=0.5, label="Save & Clear (+pts)")
                            with gr.Row():
                                ball_to_goal_slider = gr.Slider(0.0, 5.0, value=float(rew_cfg.get("ball_to_goal_weight", 1.5)), step=0.1, label="Ball-to-Goal Velocity")
                                player_to_ball_slider = gr.Slider(0.0, 3.0, value=float(rew_cfg.get("player_to_ball_weight", 0.6)), step=0.1, label="Player-to-Ball Pursuit")
                                touch_slider = gr.Slider(0.0, 5.0, value=float(rew_cfg.get("touch_weight", 1.2)), step=0.1, label="Touch Quality Bounty")
                            with gr.Row():
                                boost_gain_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("boost_gain_weight", 0.6)), step=0.05, label="Boost Gain (Sqrt)")
                                boost_lose_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("boost_lose_weight", 0.3)), step=0.05, label="Ground Waste Penalty")
                            apply_live_rewards_btn = gr.Button("⚡ Apply Live Rewards", variant="primary")
                            live_rewards_msg = gr.Markdown("")
                            gr.Markdown("<span style='color: #94a3b8; font-size: 0.88em;'>💡 For high aerials, jump bridges, air-roll recoveries, and custom scenario probabilities, visit the <b>🎛️ Rewards & Curriculum</b> tab.</span>")

                    # Right Column: Auto-Updating Metrics Plot & Live Console Output
                    with gr.Column(scale=6):
                        with gr.Group():
                            with gr.Row():
                                gr.Markdown("### 📈 Live Training Progress & Telemetry")
                                refresh_metrics_btn = gr.Button("🔄 Refresh Curves", size="sm", scale=1)
                            live_metrics_plot = gr.Plot(
                                value=render_training_curves_plot(),
                                label="Telemetry Curves (Mean Reward, Losses, Entropy, SPS)"
                            )

                        with gr.Group():
                            with gr.Row():
                                gr.Markdown("### 📜 Real-Time Process Output Stream")
                                refresh_logs_btn = gr.Button("🔄 Refresh Logs", size="sm", scale=1)
                                clear_logs_btn = gr.Button("🧹 Clear", size="sm", scale=1)
                            console_output = gr.TextArea(
                                value=mgr.get_logs(),
                                label="Training Process Output (stdout / stderr)",
                                lines=13,
                                max_lines=18,
                                interactive=False,
                                autoscroll=True
                            )

            # =========================================================
            # TAB 2: 🎛️ REWARDS & CURRICULUM STUDIO
            # =========================================================
            with gr.TabItem("🎛️ Rewards & Curriculum"):
                gr.Markdown(
                    """
                    > **🏆 Advanced Reward Architecture & Dynamic Curriculum Studio:**
                    > Tune aerial jump bridge incentives, air-roll recoveries, powerslide drifts, normalized scenario probability distributions, and design custom situations.
                    """
                )

                with gr.Group():
                    gr.Markdown("### 🚀 Advanced Flight & Recovery Mechanics")
                    with gr.Row():
                        jump_bridge_slider = gr.Slider(0.0, 1.0, value=float(rew_cfg.get("jump_bridge_weight", 0.35)), step=0.05, label="Jump & Aerial Takeoff Incentive", info="Takeoff & speed-flip transition bounty (2.0x on elevated aerials).")
                        air_roll_recovery_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("air_roll_recovery_weight", 0.10)), step=0.05, label="Air-Roll & Landing Recovery", info="Rewards wheels-down recovery on descent.")
                        powerslide_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("powerslide_weight", 0.20)), step=0.05, label="Powerslide & Drift Cut Bounty", info="Rewards handbrake powerslides on sharp turns.")

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
                            kickoff_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("kickoff_prob", 0.21)), step=0.01, label="Kickoff Scenario Probability", info="Standard 1v1 kickoff formations.")
                            replay_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("replay_prob", 0.17)), step=0.01, label="Human Replay Scenario Probability", info="Authentic match situations sampled from replays.")
                            aerial_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("aerial_prob", 0.08)), step=0.01, label="High Aerial Scenario Probability", info="Floating & rising balls for aerial training.")
                            custom_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("custom_prob", 0.25)), step=0.01, label="🎯 Custom Scenarios Probability", info="User-designed custom situations.")

                        with gr.Column():
                            turnaround_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("turnaround_prob", 0.14)), step=0.01, label="Turnaround Recovery Probability", info="Fast downfield spawns moving away from ball.")
                            wall_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("wall_prob", 0.07)), step=0.01, label="Wall Play Scenario Probability", info="Sidewall rolling and backboard rebounds.")
                            save_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("save_prob", 0.07)), step=0.01, label="Goalie Save Scenario Probability", info="Fast opponent shots into defending net.")

                    custom_sc_count = len(ScenarioManager.get_instance().get_active_scenarios())
                    gr.HTML(
                        f"""
                        <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid #334155; border-radius: 8px; padding: 9px 16px; margin-top: 8px; font-size: 0.9em; display: flex; justify-content: space-between; align-items: center; box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);">
                            <span>📦 <b>Custom Scenarios Distribution Pool:</b> <b style="color: #38bdf8;">{custom_sc_count} Active Scenarios</b> enabled in training rotation.</span>
                            <span style="color: #94a3b8;">Design and test custom drills below.</span>
                        </div>
                        """
                    )

                with gr.Group():
                    gr.Markdown("### 👤 Human Replay Guidance (BC Regularization)")
                    with gr.Row():
                        bc_weight_slider = gr.Slider(0.0, 1.0, value=float(hp_cfg.get("bc_regularization_weight", 0.10)), step=0.01, label="Replay Guidance Weight", info="Nudges vehicle steering and throttle from human replays.")
                        bc_decay_input = gr.Number(value=int(hp_cfg.get("bc_decay_steps", 150000000)), precision=0, label="Replay Guidance Decay Horizon (Steps)", info="Threshold over which guidance decays to 0.0.")

                with gr.Row():
                    apply_all_curriculum_btn = gr.Button("⚡ Apply All Curriculum & Reward Dials", variant="primary")
                    reset_curriculum_btn = gr.Button("🔄 Reset to Balanced Standard Dials", variant="secondary")

                curriculum_apply_msg = gr.Markdown("")

                gr.Markdown("---")

                # Sub-Section: Custom Scenario Generator & Builder
                gr.Markdown("### 🎯 Custom Scenario Generator & Interactive Pitch Builder")
                sc_mgr = ScenarioManager.get_instance()
                all_scenarios = sc_mgr.get_all_scenarios()
                initial_sc = all_scenarios[0] if all_scenarios else DEFAULT_CUSTOM_SCENARIOS[0]

                with gr.Row():
                    # Left Column: 2D Visual Guide Preview & Simulation Rollout
                    with gr.Column(scale=5):
                        gr.Markdown("#### 🗺️ Live 2D Pitch Visual Guide")
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
                            gr.Markdown("#### 📝 Scenario Metadata")
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
                            gr.Markdown("#### 🏎️ Bot State (Car 0 / Blue)")
                            with gr.Row():
                                car_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc["car"]["pos"][0]), step=25.0, label="Pos X (Left / Right)")
                                car_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc["car"]["pos"][1]), step=25.0, label="Pos Y (Goal to Goal)")
                                car_pos_z = gr.Slider(17.0, 1600.0, value=float(initial_sc["car"]["pos"][2]), step=10.0, label="Pos Z (Altitude)")
                            with gr.Row():
                                car_yaw = gr.Slider(-180.0, 180.0, value=float(initial_sc["car"].get("yaw", 90.0)), step=5.0, label="Heading / Yaw (deg: 90° = +Y, -90° = -Y)")
                                car_speed = gr.Slider(0.0, 2300.0, value=float(math.hypot(initial_sc["car"]["vel"][0], initial_sc["car"]["vel"][1])), step=25.0, label="Forward Velocity Speed (uu/s)")
                                car_boost = gr.Slider(0.0, 100.0, value=float(initial_sc["car"].get("boost", 50.0)), step=5.0, label="Starting Boost Amount (%)")

                        with gr.Group():
                            gr.Markdown("#### ⚽ Ball State")
                            with gr.Row():
                                ball_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc["ball"]["pos"][0]), step=25.0, label="Ball Pos X")
                                ball_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc["ball"]["pos"][1]), step=25.0, label="Ball Pos Y")
                                ball_pos_z = gr.Slider(93.15, 1800.0, value=float(initial_sc["ball"]["pos"][2]), step=10.0, label="Ball Pos Z (Height)")
                            with gr.Row():
                                ball_vel_x = gr.Slider(-2500.0, 2500.0, value=float(initial_sc["ball"]["vel"][0]), step=25.0, label="Ball Vel X (uu/s)")
                                ball_vel_y = gr.Slider(-2500.0, 2500.0, value=float(initial_sc["ball"]["vel"][1]), step=25.0, label="Ball Vel Y (uu/s)")
                                ball_vel_z = gr.Slider(-1500.0, 1500.0, value=float(initial_sc["ball"]["vel"][2]), step=25.0, label="Ball Vel Z (uu/s)")

                        with gr.Group():
                            gr.Markdown("#### 👤 Opponent State (Car 1 / Orange)")
                            with gr.Row():
                                opp_mode_radio = gr.Radio(["goalie", "shadow", "custom", "none"], value=initial_sc.get("opponent", {}).get("mode", "goalie"), label="Opponent Placement Mode")
                                opp_boost = gr.Slider(0.0, 100.0, value=float(initial_sc.get("opponent", {}).get("boost", 60.0)), step=5.0, label="Opponent Boost (%)")
                            with gr.Row(visible=(initial_sc.get("opponent", {}).get("mode", "goalie") == "custom")) as opp_custom_row:
                                opp_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc.get("opponent", {}).get("pos", [0, 4800, 17])[0]), step=25.0, label="Custom Opponent Pos X")
                                opp_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc.get("opponent", {}).get("pos", [0, 4800, 17])[1]), step=25.0, label="Custom Opponent Pos Y")
                                opp_yaw = gr.Slider(-180.0, 180.0, value=float(initial_sc.get("opponent", {}).get("yaw", -90.0)), step=5.0, label="Custom Opponent Yaw (deg)")

                        with gr.Group():
                            gr.Markdown("#### 🎲 Training Variance & Symmetry")
                            with gr.Row():
                                pos_jitter = gr.Slider(0.0, 300.0, value=float(initial_sc.get("variance", {}).get("pos_jitter", 80.0)), step=10.0, label="Positional Jitter (±uu)", info="Adds natural positional randomness each spawn.")
                                vel_jitter = gr.Slider(0.0, 300.0, value=float(initial_sc.get("variance", {}).get("vel_jitter", 60.0)), step=10.0, label="Velocity Jitter (±uu/s)", info="Adds velocity variance each spawn.")
                                mirror_symmetry = gr.Checkbox(value=bool(initial_sc.get("variance", {}).get("mirror_symmetry", True)), label="Left/Right Mirror Symmetry (50% Chance)", info="Mirrors scenario across X-axis so bot trains both sides.")

                        with gr.Row():
                            save_scenario_btn = gr.Button("💾 Save / Update Custom Scenario", variant="primary")
                            new_scenario_btn = gr.Button("➕ New / Clear Form", variant="secondary")
                            delete_scenario_btn = gr.Button("🗑️ Delete Scenario", variant="stop")

                        scenario_action_msg = gr.Markdown("")

                gr.Markdown("#### 📚 Saved Custom Scenarios Library")
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

            # =========================================================
            # TAB 3: ⚙️ ENGINE CONFIG & PRETRAINER
            # =========================================================
            with gr.TabItem("⚙️ Config & Pretrainer"):
                gr.Markdown(
                    """
                    > **⚙️ Base Architecture Configuration & Behavioral Cloning Pretrainer:**
                    > Tune offline PPO hyperparameters, vectorized arena simulation settings, ingest human `.replay` match files, and run supervised imitation pretraining.
                    """
                )
                with gr.Row():
                    # Left Column: PPO Hyperparameters & Arena Settings
                    with gr.Column(scale=5):
                        with gr.Group():
                            gr.Markdown("### 🧠 Offline PPO Hyperparameters")
                            with gr.Row():
                                gamma_slider = gr.Slider(
                                    0.9, 0.999, value=hp_cfg.get("gamma", 0.99), step=0.001,
                                    label="Discount Factor (Gamma)",
                                    info="Future rewards discount value."
                                )
                                gae_lambda_slider = gr.Slider(
                                    0.8, 1.0, value=hp_cfg.get("gae_lambda", 0.95), step=0.01,
                                    label="GAE Lambda",
                                    info="GAE variance vs bias trade-off."
                                )
                            with gr.Row():
                                batch_size_input = gr.Number(
                                    value=hp_cfg.get("batch_size", 8192), precision=0,
                                    label="Rollout Buffer Batch Size",
                                    info="Total steps per iteration across arenas."
                                )
                                mini_batch_input = gr.Number(
                                    value=hp_cfg.get("mini_batch_size", 512), precision=0,
                                    label="Mini-Batch Size",
                                    info="Gradient update chunk size."
                                )
                                n_epochs_input = gr.Number(
                                    value=hp_cfg.get("n_epochs", 10), precision=0,
                                    label="Epochs per Iteration",
                                    info="Optimization passes per rollout."
                                )

                        with gr.Group():
                            gr.Markdown("### 🏟️ Simulation & Checkpointing")
                            with gr.Row():
                                num_envs_slider = gr.Slider(
                                    1, 128, value=env_cfg.get("num_envs", 64), step=1,
                                    label="Vectorized Arenas",
                                    info="Parallel RocketSim arena instances."
                                )
                                tick_skip_slider = gr.Slider(
                                    1, 8, value=env_cfg.get("tick_skip", 8), step=1,
                                    label="Tick Skip (Action Repeat)",
                                    info="8 skip ≈ 15 decisions/sec."
                                )
                            with gr.Row():
                                max_steps_input = gr.Number(
                                    value=env_cfg.get("max_episode_steps", 750), precision=0,
                                    label="Max Episode Steps",
                                    info="750 steps ≈ 50s match time."
                                )
                                game_mode_dropdown = gr.Dropdown(
                                    ["1v1", "2v2", "3v3"], value=env_cfg.get("game_mode", "1v1"),
                                    label="Game Mode",
                                    info="Match format (1v1, 2v2, 3v3)."
                                )
                            with gr.Row():
                                checkpoint_interval_input = gr.Number(
                                    value=log_cfg.get("checkpoint_interval", 20), precision=0,
                                    label="Checkpoint Interval (Iters)"
                                )
                                max_checkpoints_input = gr.Number(
                                    value=log_cfg.get("max_checkpoints_to_keep", 5), precision=0,
                                    label="Max Checkpoints to Keep (Pruning)"
                                )

                            save_cfg_btn = gr.Button("💾 Save Configuration to YAML", variant="primary")
                            cfg_save_msg = gr.Markdown("")

                    # Right Column: Human Replay Dataset & Imitation Pretrainer
                    with gr.Column(scale=6):
                        def build_replay_stats_md():
                            parser = ReplayParser()
                            st = parser.get_pool_stats()
                            total_frames = st.get('total_frames', 0)
                            num_matches = st.get('num_matches', 0)
                            file_size_mb = st.get('file_size_mb', 0.0)
                            est_game_time = (total_frames / 15.0) / 60.0
                            has_data = total_frames > 0
                            badge = '<span class="status-badge-running">● DATASET ACTIVE</span>' if has_data else '<span class="status-badge-stopped">○ EMPTY DATASET</span>'
                            return f"""
                            <div class="cyber-panel">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <h4 style="margin: 0; color: #f1f5f9;">📦 Human Replay Dataset Pool</h4>
                                    {badge}
                                </div>
                                <div style="display: flex; gap: 20px; color: #cbd5e1; font-size: 0.9em; flex-wrap: wrap;">
                                    <span>Replay Matches: <b style="color: #38bdf8;">{num_matches}</b></span>
                                    <span>Total Frames: <b style="color: #818cf8;">{total_frames:,}</b></span>
                                    <span>Est. Duration: <b style="color: #34d399;">{est_game_time:.1f} mins</b></span>
                                    <span>Pool Size: <b style="color: #facc15;">{file_size_mb:.2f} MB</b></span>
                                </div>
                            </div>
                            """

                        replay_stats_box = gr.HTML(build_replay_stats_md())

                        with gr.Group():
                            gr.Markdown("### 📂 Ingest from Directory")
                            with gr.Row():
                                demos_dir_input = gr.Textbox(
                                    value=get_default_demo_dir(),
                                    label="Replay Directory Path",
                                    info="Absolute or relative path where .replay files are stored."
                                )
                                scan_demos_btn = gr.Button("🔍 Scan Replays", scale=1)

                            with gr.Row():
                                max_replays_slider = gr.Slider(
                                    1, 100, value=20, step=1,
                                    label="Max Replays to Ingest",
                                    info="Cap the number of replays to parse into training buffer."
                                )
                                sort_replays_radio = gr.Radio(
                                    ["newest", "oldest"], value="newest",
                                    label="Sort Order"
                                )

                            demos_table = gr.Dataframe(
                                headers=["Filename", "Size (KB)", "Modified"],
                                datatype=["str", "number", "str"],
                                value=[],
                                label="Discovered Replays (Select or Ingest All)"
                            )

                            with gr.Row():
                                ingest_selected_btn = gr.Button("⚡ Ingest Discovered Replays", variant="primary")
                                ingest_all_btn = gr.Button("📥 Ingest ALL Replays in Directory", variant="secondary")
                                clear_pool_btn = gr.Button("🗑️ Clear Replay Dataset Pool", variant="stop")

                            replays_status_box = gr.Markdown("")

                        with gr.Group():
                            gr.Markdown("### 📤 Upload `.replay` Files Directly")
                            replay_uploader = gr.File(
                                file_count="multiple",
                                file_types=[".replay"],
                                label="Drop Rocket League .replay files here"
                            )
                            upload_status_box = gr.Markdown("")

                        with gr.Group():
                            gr.Markdown("### 🎓 Behavioral Cloning (Imitation Pretrainer)")
                            with gr.Row():
                                pretrain_epochs_slider = gr.Slider(1, 20, value=5, step=1, label="Pretraining Epochs")
                                pretrain_lr_input = gr.Number(value=0.0005, label="BC Learning Rate")
                            with gr.Row():
                                pretrain_batch_dropdown = gr.Dropdown([64, 128, 256, 512], value=256, label="BC Batch Size")
                                pretrain_base_dropdown = gr.Dropdown(
                                    choices=get_available_checkpoints(),
                                    value=get_available_checkpoints()[0],
                                    label="Base Checkpoint"
                                )
                            with gr.Row():
                                run_pretrain_btn = gr.Button("🚀 Run Imitation Pretraining", variant="primary")
                                stop_pretrain_btn = gr.Button("⏹️ Stop Pretrainer", variant="stop")
                            pretrain_status_box = gr.HTML(
                                """
                                <div class="status-callout-box" style="border-left-color: #64748b;">
                                    <span style="color: #94a3b8; font-weight: 700; margin-right: 8px;">IDLE:</span>
                                    <span>Ready to train initial policy weights on parsed replay data.</span>
                                </div>
                                """
                            )

            # =========================================================
            # TAB 4: 🔬 DIAGNOSTICS & EVALUATION (FLATTENED HUB)
            # =========================================================
            with gr.TabItem("🔬 Diagnostics & Evaluation"):
                gr.Markdown(
                    """
                    ### 🔬 Unified Diagnostic & Evaluation Hub
                    Single-pane-of-glass workspace for full system health, automated unit test verification, 2D match simulation replays, behavioral bias heatmaps, and AI assistant snapshot export.
                    """
                )

                # SECTION 1: Automated Unit Tests & Health
                with gr.Group():
                    with gr.Row():
                        gr.Markdown("### 🧪 Subsystem Unit Tests & Health Verification")
                        run_unit_tests_btn = gr.Button("🧪 Run All Unit Tests", variant="primary", scale=1)
                    with gr.Row():
                        with gr.Column(scale=1):
                            unit_tests_overview_md = gr.Markdown(value=format_test_results_markdown(get_cached_or_run_tests()))
                        with gr.Column(scale=1):
                            unit_tests_stdout = gr.Code(
                                label="Test Runner Output Stream",
                                language="markdown",
                                lines=10,
                                interactive=False
                            )

                gr.Markdown("---")

                # SECTION 2: 2D Pitch Match Visualizer & Simulation
                with gr.Group():
                    gr.Markdown("### 🎮 2D Pitch Match Visualizer & Simulation Replay")
                    with gr.Row():
                        with gr.Column(scale=4):
                            ckpt_dropdown = gr.Dropdown(
                                choices=get_available_checkpoints(),
                                value=get_available_checkpoints()[0],
                                label="Select Blue Team Checkpoint",
                                info="Trained PyTorch model checkpoint (.pt) for Blue Team."
                            )
                            opponent_mode = gr.Radio(
                                ["Self-Play (Bot vs Itself)", "Baseline Bot (Chase Ball Heuristic)", "Another Checkpoint"],
                                value="Self-Play (Bot vs Itself)",
                                label="Opponent Matchup Type"
                            )
                            orange_ckpt_dropdown = gr.Dropdown(
                                choices=get_available_checkpoints(),
                                value=get_available_checkpoints()[0],
                                label="Select Orange Team Checkpoint",
                                visible=False,
                                info="Select a different checkpoint for Orange Team."
                            )
                            refresh_ckpts_btn = gr.Button("🔄 Scan Checkpoints")
                            sim_steps_slider = gr.Slider(
                                100, 1000, value=400, step=50,
                                label="Simulation Steps",
                                info="Duration of match simulation (400 steps ≈ 26s)."
                            )
                            run_sim_btn = gr.Button("🕹️ Simulate Match & Render Replay", variant="primary")
                            sim_stats_box = gr.Markdown("#### Match Results: Click 'Simulate Match' to evaluate.")

                        with gr.Column(scale=7):
                            visualizer_plot = gr.Plot(label="🗺️ 2D Pitch Trajectories")
                            reward_breakdown_plot = gr.Plot(label="📊 Match Reward Breakdown")

                gr.Markdown("---")

                # SECTION 3: Behavioral Biases & AI Coach
                with gr.Group():
                    with gr.Row():
                        gr.Markdown("### 🧠 Behavioral Biases & AI Behavioral Coach")
                        diag_window_slider = gr.Slider(
                            1, 25, value=8, step=1,
                            label="Rolling Average Window (Iterations)",
                            scale=2
                        )
                        refresh_diag_btn = gr.Button("🔄 Refresh AI Coach Analysis", variant="primary", scale=1)
                    with gr.Row():
                        with gr.Column(scale=1):
                            diag_coach_report = gr.Markdown(value="*Click 'Refresh AI Coach Analysis' or run training to view live AI coach analysis.*")
                        with gr.Column(scale=1):
                            diag_action_plot = gr.Plot(label="Action & Control Distributions")
                            diag_position_plot = gr.Plot(label="Pitch Positioning & Vehicle State Radar")

                gr.Markdown("---")

                # SECTION 4: Comprehensive System Snapshot & AI Assistant Export
                with gr.Group():
                    with gr.Row():
                        gr.Markdown("### 📋 System Snapshot & AI Assistant Export")
                        refresh_snapshot_btn = gr.Button("🔄 Refresh Diagnostic Snapshot", variant="primary", scale=1)
                    with gr.Row():
                        with gr.Column(scale=1):
                            diag_overview_md = gr.Markdown(value="*Click 'Refresh Diagnostic Snapshot' to generate live overview.*")
                        with gr.Column(scale=1):
                            diag_export_raw = gr.Code(
                                label="📋 Complete Diagnostic Snapshot (Copy & Paste to Assistant)",
                                language="markdown",
                                lines=20,
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

        # -------------------------------------------------------------
        # LIVE HYPERPARAMETERS & DIALS (TAB 1)
        # -------------------------------------------------------------
        def on_apply_live_hyperparams(lr_val, ent_val, clip_val):
            payload = {
                "learning_rate": float(lr_val),
                "ent_coef": float(ent_val),
                "clip_range": float(clip_val),
            }
            mgr.update_live_config(payload)
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                if "hyperparameters" not in base_cfg:
                    base_cfg["hyperparameters"] = {}
                base_cfg["hyperparameters"]["learning_rate"] = float(lr_val)
                base_cfg["hyperparameters"]["ent_coef"] = float(ent_val)
                base_cfg["hyperparameters"]["clip_range"] = float(clip_val)
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            return f"✅ **Live Hyperparameters Applied:** LR=`{float(lr_val):.2e}`, Ent Coef=`{float(ent_val):.4f}`, Clip=`{float(clip_val):.2f}` at {time.strftime('%H:%M:%S')}"

        live_hp_btn.click(
            fn=on_apply_live_hyperparams,
            inputs=[lr_input, ent_coef_slider, clip_range_slider],
            outputs=[live_hp_msg]
        )

        def on_apply_opponent_mix(opp_bot, opp_ratio):
            clean_opp_type = "heuristic"
            if opp_bot and not str(opp_bot).startswith("Heuristic"):
                clean_opp_type = str(opp_bot).strip()
            payload = {
                "baseline_opponent_type": clean_opp_type,
                "baseline_opponent_ratio": float(opp_ratio),
            }
            mgr.update_live_config(payload)
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                if "environment" not in base_cfg:
                    base_cfg["environment"] = {}
                base_cfg["environment"]["baseline_opponent_type"] = clean_opp_type
                base_cfg["environment"]["baseline_opponent_ratio"] = float(opp_ratio)
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            return f"✅ **Opponent Mix Applied:** `{os.path.basename(clean_opp_type)}` ({float(opp_ratio):.0%}) at {time.strftime('%H:%M:%S')}"

        apply_opp_btn.click(
            fn=on_apply_opponent_mix,
            inputs=[opponent_bot_dropdown, baseline_opp_slider],
            outputs=[opp_apply_msg]
        )

        refresh_opponent_btn.click(
            fn=lambda: gr.Dropdown(choices=get_available_opponent_options()),
            outputs=[opponent_bot_dropdown]
        )

        # Quick Live Rewards (Tab 1)
        def on_apply_quick_rewards(g_w, c_w, sv_w, b2g_w, p2b_w, tch_w, bg_w, bl_w):
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
                if "rewards" not in base_cfg:
                    base_cfg["rewards"] = {}
                base_cfg["rewards"].update(rewards)
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            return f"✅ **Live Rewards Applied at {time.strftime('%H:%M:%S')}!**"

        apply_live_rewards_btn.click(
            fn=on_apply_quick_rewards,
            inputs=[
                goal_slider, concede_slider, save_slider,
                ball_to_goal_slider, player_to_ball_slider, touch_slider,
                boost_gain_slider, boost_lose_slider
            ],
            outputs=[live_rewards_msg]
        )

        # -------------------------------------------------------------
        # FULL CURRICULUM & REWARD DIALS (TAB 2)
        # -------------------------------------------------------------
        def on_apply_curriculum(
            g_w, c_w, sv_w,
            b2g_w, p2b_w, jb_w, ar_w, pw_w, tch_w,
            bg_w, bl_w,
            k_p, r_p, a_p, tr_p, w_p, s_p, c_p,
            bc_w, bc_dec
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

            payload = {
                "rewards": rewards,
                "scenarios": scenarios,
                "bc_regularization_weight": float(bc_w),
                "bc_decay_steps": int(bc_dec)
            }
            mgr.update_live_config(payload)
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                base_cfg["rewards"] = rewards
                base_cfg["scenarios"] = scenarios
                if "hyperparameters" not in base_cfg:
                    base_cfg["hyperparameters"] = {}
                base_cfg["hyperparameters"]["bc_regularization_weight"] = float(bc_w)
                base_cfg["hyperparameters"]["bc_decay_steps"] = int(bc_dec)
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            return f"✅ **All Curriculum & Reward Dials Applied at {time.strftime('%H:%M:%S')}!**"

        apply_all_curriculum_btn.click(
            fn=on_apply_curriculum,
            inputs=[
                goal_slider, concede_slider, save_slider,
                ball_to_goal_slider, player_to_ball_slider, jump_bridge_slider, air_roll_recovery_slider, powerslide_slider, touch_slider,
                boost_gain_slider, boost_lose_slider,
                kickoff_prob_slider, replay_prob_slider, aerial_prob_slider, turnaround_prob_slider, wall_prob_slider, save_prob_slider, custom_prob_slider,
                bc_weight_slider, bc_decay_input
            ],
            outputs=[curriculum_apply_msg]
        )

        # Dynamic 100% Normalized Scenario Rebalancing Handler (7 Scenario Mix)
        def rebalance_scenarios_handler(changed_idx, new_val, k, r, a, tr, w, s, c):
            current_vals = [float(k), float(r), float(a), float(tr), float(w), float(s), float(c)]
            new_val = round(max(0.0, min(1.0, float(new_val))), 2)
            vals = list(current_vals)
            vals[changed_idx] = new_val

            rem = round(1.0 - new_val, 4)
            other_indices = [i for i in range(7) if i != changed_idx]
            other_sum = sum(current_vals[i] for i in other_indices)

            if other_sum > 0.0001:
                scale = rem / other_sum
                for i in other_indices:
                    vals[i] = round(current_vals[i] * scale, 2)
            else:
                even = round(rem / len(other_indices), 2)
                for i in other_indices:
                    vals[i] = even

            # Snap rounding error to first available other index
            tot = sum(vals)
            diff = round(1.0 - tot, 2)
            if abs(diff) > 0.0001:
                for idx in other_indices:
                    if vals[idx] + diff >= 0:
                        vals[idx] = round(vals[idx] + diff, 2)
                        break

            pct_total = int(round(sum(vals) * 100))
            badge_html = f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                <span class="status-badge-running" style="font-size: 1.0em; padding: 6px 16px;">● Total Mix: {pct_total}%</span>
            </div>
            """
            return tuple(vals) + (badge_html,)

        scenario_sliders = [kickoff_prob_slider, replay_prob_slider, aerial_prob_slider, turnaround_prob_slider, wall_prob_slider, save_prob_slider, custom_prob_slider]
        rebalance_outputs = scenario_sliders + [scenario_total_badge]

        for i, sld in enumerate(scenario_sliders):
            sld.change(
                fn=lambda *args, idx=i: rebalance_scenarios_handler(idx, args[0], *args[1:]),
                inputs=[sld] + scenario_sliders,
                outputs=rebalance_outputs
            )

        # Reset Rewards to Balanced Defaults
        def on_reset_rewards():
            def_cfg = load_yaml_config("config/default_config.yaml")
            rew = def_cfg.get("rewards", {})
            sc = def_cfg.get("scenarios", {})
            badge_html = """
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                <span class="status-badge-running" style="font-size: 1.0em; padding: 6px 16px;">● Total Mix: 100%</span>
            </div>
            """
            return (
                rew.get("goal_weight", 20.0),
                rew.get("concede_weight", -20.0),
                rew.get("save_weight", 3.0),
                rew.get("ball_to_goal_weight", 1.5),
                rew.get("player_to_ball_weight", 0.6),
                rew.get("jump_bridge_weight", 0.35),
                rew.get("air_roll_recovery_weight", 0.10),
                rew.get("powerslide_weight", 0.20),
                rew.get("touch_weight", 1.2),
                rew.get("boost_gain_weight", 0.6),
                rew.get("boost_lose_weight", 0.3),
                sc.get("kickoff_prob", 0.21),
                sc.get("replay_prob", 0.17),
                sc.get("aerial_prob", 0.08),
                sc.get("turnaround_prob", 0.14),
                sc.get("wall_prob", 0.07),
                sc.get("save_prob", 0.07),
                sc.get("custom_prob", 0.25),
                badge_html,
                "🔄 **Reset dials to balanced standard configuration.**"
            )

        reset_curriculum_btn.click(
            fn=on_reset_rewards,
            outputs=[
                goal_slider, concede_slider, save_slider,
                ball_to_goal_slider, player_to_ball_slider, jump_bridge_slider, air_roll_recovery_slider, powerslide_slider, touch_slider,
                boost_gain_slider, boost_lose_slider,
                kickoff_prob_slider, replay_prob_slider, aerial_prob_slider, turnaround_prob_slider, wall_prob_slider, save_prob_slider, custom_prob_slider,
                scenario_total_badge,
                curriculum_apply_msg
            ]
        )

        # -------------------------------------------------------------
        # CUSTOM SCENARIO GENERATOR HANDLERS
        # -------------------------------------------------------------
        def assemble_scenario_payload(
            s_id, s_name, s_enabled, s_desc,
            c_x, c_y, c_z, c_yaw, c_spd, c_boost,
            b_x, b_y, b_z, b_vx, b_vy, b_vz,
            o_mode, o_boost, o_x, o_y, o_yaw,
            p_jit, v_jit, mirror
        ) -> dict:
            yaw_rad = math.radians(float(c_yaw))
            spd = float(c_spd)
            car_vel = [spd * math.cos(yaw_rad), spd * math.sin(yaw_rad), 0.0]

            opp_dict = {
                "mode": str(o_mode),
                "boost": float(o_boost)
            }
            if o_mode == "custom":
                opp_dict["pos"] = [float(o_x), float(o_y), 17.0]
                opp_dict["yaw"] = float(o_yaw)
                opp_dict["vel"] = [0.0, 0.0, 0.0]

            return {
                "id": str(s_id).strip(),
                "name": str(s_name).strip(),
                "enabled": bool(s_enabled),
                "description": str(s_desc).strip(),
                "car": {
                    "pos": [float(c_x), float(c_y), float(c_z)],
                    "vel": car_vel,
                    "yaw": float(c_yaw),
                    "boost": float(c_boost)
                },
                "ball": {
                    "pos": [float(b_x), float(b_y), float(b_z)],
                    "vel": [float(b_vx), float(b_vy), float(b_vz)]
                },
                "opponent": opp_dict,
                "variance": {
                    "pos_jitter": float(p_jit),
                    "vel_jitter": float(v_jit),
                    "mirror_symmetry": bool(mirror)
                }
            }

        def on_update_visual_preview(*args):
            sc = assemble_scenario_payload(*args)
            return render_scenario_visual_guide(sc)

        all_sc_inputs = [
            sc_id_input, sc_name_input, sc_enabled_cb, sc_desc_input,
            car_pos_x, car_pos_y, car_pos_z, car_yaw, car_speed, car_boost,
            ball_pos_x, ball_pos_y, ball_pos_z, ball_vel_x, ball_vel_y, ball_vel_z,
            opp_mode_radio, opp_boost, opp_pos_x, opp_pos_y, opp_yaw,
            pos_jitter, vel_jitter, mirror_symmetry
        ]

        for comp in [car_pos_x, car_pos_y, car_pos_z, car_yaw, car_speed, car_boost,
                     ball_pos_x, ball_pos_y, ball_pos_z, ball_vel_x, ball_vel_y, ball_vel_z,
                     opp_mode_radio, opp_pos_x, opp_pos_y, opp_yaw]:
            comp.change(fn=on_update_visual_preview, inputs=all_sc_inputs, outputs=[sc_preview_plot])

        refresh_preview_btn.click(fn=on_update_visual_preview, inputs=all_sc_inputs, outputs=[sc_preview_plot])

        def on_scenario_opp_mode_change(mode):
            return gr.Row(visible=(mode == "custom"))

        opp_mode_radio.change(fn=on_scenario_opp_mode_change, inputs=[opp_mode_radio], outputs=[opp_custom_row])

        # Preset Selector Callback
        def on_select_preset_template(preset_name):
            if not preset_name or preset_name == "(Select Template Preset...)":
                return (gr.update(),) * 23
            match = next((s for s in DEFAULT_CUSTOM_SCENARIOS if s["name"] == preset_name), None)
            if not match:
                return (gr.update(),) * 23

            c = match["car"]
            b = match["ball"]
            o = match.get("opponent", {})
            v = match.get("variance", {})

            spd = math.hypot(c["vel"][0], c["vel"][1])
            o_pos = o.get("pos", [0, 4800, 17])

            return (
                f"{match['id']}_{int(time.time()) % 1000}",
                f"{match['name']} (Custom)",
                True,
                match.get("description", ""),
                c["pos"][0], c["pos"][1], c["pos"][2],
                c.get("yaw", 90.0), spd, c.get("boost", 50.0),
                b["pos"][0], b["pos"][1], b["pos"][2],
                b["vel"][0], b["vel"][1], b["vel"][2],
                o.get("mode", "goalie"), o.get("boost", 60.0),
                o_pos[0], o_pos[1], o.get("yaw", -90.0),
                v.get("pos_jitter", 80.0), v.get("vel_jitter", 60.0), v.get("mirror_symmetry", True)
            )

        preset_dropdown.change(
            fn=on_select_preset_template,
            inputs=[preset_dropdown],
            outputs=all_sc_inputs
        )

        # Save Scenario Callback
        def on_save_custom_scenario(*args):
            sc = assemble_scenario_payload(*args)
            if not sc["id"]:
                return "❌ Error: Scenario ID cannot be empty.", build_scenarios_table(), gr.Dropdown()
            sc_mgr.save_scenario(sc)
            all_scs = sc_mgr.get_all_scenarios()
            choices = [f"{s['name']} ({s['id']})" for s in all_scs]
            sel = f"{sc['name']} ({sc['id']})"
            return (
                f"✅ **Saved custom scenario '{sc['name']}' ({sc['id']})!** Added to active training pool.",
                build_scenarios_table(),
                gr.Dropdown(choices=choices, value=sel)
            )

        save_scenario_btn.click(
            fn=on_save_custom_scenario,
            inputs=all_sc_inputs,
            outputs=[scenario_action_msg, saved_scenarios_table, load_scenario_dropdown]
        )

        # New Scenario Form Callback
        def on_new_scenario_form():
            nid = f"custom_drill_{int(time.time()) % 10000}"
            return (
                nid, "New Custom Drill", True, "User custom drill description.",
                0.0, -2500.0, 17.0, 90.0, 500.0, 50.0,
                0.0, 0.0, 93.15, 0.0, 0.0, 0.0,
                "goalie", 50.0, 0.0, 4800.0, -90.0,
                80.0, 60.0, True,
                "✨ Cleared form. Design your scenario and click **Save Custom Scenario**."
            )

        new_scenario_btn.click(
            fn=on_new_scenario_form,
            outputs=all_sc_inputs + [scenario_action_msg]
        )

        # Delete Scenario Callback
        def on_delete_custom_scenario(sc_id):
            if not sc_id:
                return "⚠️ No scenario selected to delete.", build_scenarios_table(), gr.Dropdown()
            success = sc_mgr.delete_scenario(str(sc_id).strip())
            all_scs = sc_mgr.get_all_scenarios()
            choices = [f"{s['name']} ({s['id']})" for s in all_scs]
            sel = choices[0] if choices else None
            msg = f"🗑️ **Deleted scenario '{sc_id}'.**" if success else f"⚠️ Scenario '{sc_id}' could not be deleted."
            return msg, build_scenarios_table(), gr.Dropdown(choices=choices, value=sel)

        delete_scenario_btn.click(
            fn=on_delete_custom_scenario,
            inputs=[sc_id_input],
            outputs=[scenario_action_msg, saved_scenarios_table, load_scenario_dropdown]
        )

        # Load Scenario from Library Callback
        def on_load_scenario_from_library(selected_choice):
            if not selected_choice:
                return (gr.update(),) * 23
            try:
                sc_id = selected_choice.split("(")[-1].rstrip(")").strip()
            except Exception:
                sc_id = selected_choice
            match = sc_mgr.get_scenario(sc_id)
            if not match:
                return (gr.update(),) * 23

            c = match["car"]
            b = match["ball"]
            o = match.get("opponent", {})
            v = match.get("variance", {})
            spd = math.hypot(c["vel"][0], c["vel"][1])
            o_pos = o.get("pos", [0, 4800, 17])

            return (
                match["id"],
                match["name"],
                match.get("enabled", True),
                match.get("description", ""),
                c["pos"][0], c["pos"][1], c["pos"][2],
                c.get("yaw", 90.0), spd, c.get("boost", 50.0),
                b["pos"][0], b["pos"][1], b["pos"][2],
                b["vel"][0], b["vel"][1], b["vel"][2],
                o.get("mode", "goalie"), o.get("boost", 60.0),
                o_pos[0], o_pos[1], o.get("yaw", -90.0),
                v.get("pos_jitter", 80.0), v.get("vel_jitter", 60.0), v.get("mirror_symmetry", True)
            )

        load_scenario_btn.click(
            fn=on_load_scenario_from_library,
            inputs=[load_scenario_dropdown],
            outputs=all_sc_inputs
        )

        refresh_library_btn.click(
            fn=lambda: (build_scenarios_table(), gr.Dropdown(choices=[f"{s['name']} ({s['id']})" for s in sc_mgr.get_all_scenarios()])),
            outputs=[saved_scenarios_table, load_scenario_dropdown]
        )

        # 2-Second Trajectory Rollout Simulation Callback
        def on_run_scenario_simulation(*args):
            sc = assemble_scenario_payload(*args)
            pts = get_available_checkpoints()
            active_ckpt = pts[0] if pts and not pts[0].startswith("checkpoints/latest_model.pt (none") else None
            res = simulate_custom_scenario(sc, checkpoint_path=active_ckpt, steps=150)
            return res.get("plot"), res.get("stats")

        sim_scenario_btn.click(
            fn=on_run_scenario_simulation,
            inputs=all_sc_inputs,
            outputs=[sc_sim_plot, sc_sim_stats]
        )

        # -------------------------------------------------------------
        # TAB 3 CONFIG & PRETRAINER HANDLERS
        # -------------------------------------------------------------
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

        # Replay Scanner Callbacks
        def on_scan_demos(demo_dir, max_replays, sort_mode):
            p = ReplayParser(demo_dir=str(demo_dir).strip())
            files = p.scan_demos(max_replays=int(max_replays), sort=str(sort_mode))
            rows = []
            for fp in files:
                try:
                    sz = round(os.path.getsize(fp) / 1024, 1)
                    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(fp)))
                    rows.append([os.path.basename(fp), sz, mtime])
                except Exception:
                    rows.append([os.path.basename(fp), 0.0, "Unknown"])
            status_txt = f"🔍 Discovered **{len(rows)}** `.replay` files in `{demo_dir}`."
            return rows, status_txt

        scan_demos_btn.click(
            fn=on_scan_demos,
            inputs=[demos_dir_input, max_replays_slider, sort_replays_radio],
            outputs=[demos_table, replays_status_box]
        )

        def on_ingest_replays(demo_dir, max_replays, sort_mode):
            p = ReplayParser(demo_dir=str(demo_dir).strip())
            res = p.ingest_directory(max_replays=int(max_replays), sort=str(sort_mode))
            stats_md = build_replay_stats_md()
            msg = f"⚡ Ingested **{res['parsed_files']}** replays ({res['total_frames']:,} frames) into dataset pool in {res['elapsed_seconds']:.2f}s."
            return stats_md, msg

        ingest_selected_btn.click(
            fn=on_ingest_replays,
            inputs=[demos_dir_input, max_replays_slider, sort_replays_radio],
            outputs=[replay_stats_box, replays_status_box]
        )

        def on_ingest_all_replays(demo_dir):
            p = ReplayParser(demo_dir=str(demo_dir).strip())
            res = p.ingest_directory(max_replays=999999, sort="newest")
            stats_md = build_replay_stats_md()
            msg = f"📥 Ingested ALL **{res['parsed_files']}** replays ({res['total_frames']:,} frames) in {res['elapsed_seconds']:.2f}s."
            return stats_md, msg

        ingest_all_btn.click(
            fn=on_ingest_all_replays,
            inputs=[demos_dir_input],
            outputs=[replay_stats_box, replays_status_box]
        )

        def on_clear_replays():
            p = ReplayParser()
            p.clear_pool()
            stats_md = build_replay_stats_md()
            return stats_md, "🗑️ Replay dataset pool cleared."

        clear_pool_btn.click(
            fn=on_clear_replays,
            outputs=[replay_stats_box, replays_status_box]
        )

        # Upload Ingest Callback
        def on_upload_ingest(uploaded_files):
            if not uploaded_files:
                return build_replay_stats_md(), "⚠️ No files uploaded."
            p = ReplayParser()
            total_added = 0
            file_paths = [f.name if hasattr(f, "name") else str(f) for f in uploaded_files]
            for fp in file_paths:
                dest = os.path.join(p.demo_dir, os.path.basename(fp))
                try:
                    import shutil
                    shutil.copy2(fp, dest)
                    total_added += 1
                except Exception:
                    pass
            res = p.ingest_directory(max_replays=total_added, sort="newest")
            stats_md = build_replay_stats_md()
            msg = f"📤 Uploaded & Ingested **{total_added}** replays ({res['total_frames']:,} frames) into dataset."
            return stats_md, msg

        replay_uploader.upload(
            fn=on_upload_ingest,
            inputs=[replay_uploader],
            outputs=[replay_stats_box, upload_status_box]
        )

        # BC Pretrainer Callbacks
        def on_run_pretraining(epochs, lr, batch_size, base_ckpt):
            chosen_ckpt = base_ckpt.split(" ")[0] if base_ckpt and not base_ckpt.startswith("checkpoints/latest_model.pt (none") else None
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

        # -------------------------------------------------------------
        # TAB 4: DIAGNOSTICS & EVALUATION HANDLERS
        # -------------------------------------------------------------
        def on_run_unit_tests():
            res = run_all_unit_tests(verbose=True)
            res_md = format_test_results_markdown(res)
            return res_md, res.get("raw_output", "")

        run_unit_tests_btn.click(
            fn=on_run_unit_tests,
            outputs=[unit_tests_overview_md, unit_tests_stdout]
        )

        def on_scan_checkpoints():
            ckpts = get_available_checkpoints()
            return gr.Dropdown(choices=ckpts, value=ckpts[0] if ckpts else None), gr.Dropdown(choices=ckpts, value=ckpts[0] if ckpts else None)

        refresh_ckpts_btn.click(fn=on_scan_checkpoints, outputs=[ckpt_dropdown, orange_ckpt_dropdown])

        def on_opp_mode_change(mode):
            return gr.Dropdown(visible=(mode == "Another Checkpoint"))

        opponent_mode.change(fn=on_opp_mode_change, inputs=[opponent_mode], outputs=[orange_ckpt_dropdown])

        def on_run_simulation(blue_choice, opp_mode, orange_choice, steps):
            blue_path = blue_choice.split(" ")[0] if blue_choice else None
            if not blue_path or not os.path.exists(blue_path):
                blue_path = "checkpoints/latest_model.pt" if os.path.exists("checkpoints/latest_model.pt") else None

            orange_path = None
            if opp_mode == "Self-Play (Bot vs Itself)":
                orange_path = blue_path
            elif opp_mode == "Another Checkpoint":
                orange_path = orange_choice.split(" ")[0] if orange_choice else None
                if not orange_path or not os.path.exists(orange_path):
                    orange_path = "checkpoints/latest_model.pt" if os.path.exists("checkpoints/latest_model.pt") else None
            elif opp_mode == "Baseline Bot (Chase Ball Heuristic)":
                orange_path = "baseline"

            res = simulate_match(
                blue_checkpoint=blue_path,
                orange_checkpoint=orange_path,
                steps=int(steps),
                render_field=True
            )

            stats = res["stats"]
            summary_md = f"""
            #### 📊 Headless Match Simulation Results
            * **Simulated Duration:** `{stats['total_steps']}` steps ({stats['total_steps']/15.0:.1f}s match time)
            * **Score:** Blue **{stats['goals_blue']}** - **{stats['goals_orange']}** Orange
            * **Blue Ball Touches:** **{stats['touches_blue']}** | **Orange Ball Touches:** **{stats['touches_orange']}**
            * **Blue Net Reward:** `{stats['rewards_blue']:+.2f}` | **Orange Net Reward:** `{stats['rewards_orange']:+.2f}`
            """
            return res["plot"], res["reward_plot"], summary_md

        run_sim_btn.click(
            fn=on_run_simulation,
            inputs=[ckpt_dropdown, opponent_mode, orange_ckpt_dropdown, sim_steps_slider],
            outputs=[visualizer_plot, reward_breakdown_plot, sim_stats_box]
        )

        def on_refresh_diagnostics(window_size):
            telem = extract_rolling_telemetry("logs/history.jsonl", window=int(window_size))
            coach_md = generate_ai_coach_diagnostics(telem)
            action_fig = render_action_biases_plot(telem)
            pos_fig = render_positional_biases_plot(telem)
            return coach_md, action_fig, pos_fig

        refresh_diag_btn.click(
            fn=on_refresh_diagnostics,
            inputs=[diag_window_slider],
            outputs=[diag_coach_report, diag_action_plot, diag_position_plot]
        )

        def on_refresh_full_diagnostics():
            overview_md, export_box = build_full_diagnostic_export()
            return overview_md, export_box

        refresh_snapshot_btn.click(
            fn=on_refresh_full_diagnostics,
            outputs=[diag_overview_md, diag_export_raw]
        )

        # -------------------------------------------------------------
        # REAL-TIME BACKGROUND REFRESH TIMER & INITIAL LOAD
        # -------------------------------------------------------------
        def on_timer_tick():
            status = mgr.get_status_info()
            card_html = build_status_card_html(status)
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
            logs = mgr.get_logs()
            plot = render_training_curves_plot()
            return card_html, start_btn_update, pause_btn_update, stop_btn_update, logs, plot

        refresh_metrics_btn.click(fn=render_training_curves_plot, outputs=[live_metrics_plot])
        refresh_logs_btn.click(fn=mgr.get_logs, outputs=[console_output])
        clear_logs_btn.click(fn=lambda: "", outputs=[console_output])

        status_timer = gr.Timer(3.0, active=True)
        status_timer.tick(
            fn=on_timer_tick,
            outputs=[status_card, start_btn, pause_btn, stop_btn, console_output, live_metrics_plot]
        )

        # Initialize UI on page load
        demo.load(
            fn=on_timer_tick,
            outputs=[status_card, start_btn, pause_btn, stop_btn, console_output, live_metrics_plot]
        )

    return demo
