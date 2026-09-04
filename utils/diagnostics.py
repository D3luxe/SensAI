"""
Behavioral Diagnostics and AI Training Coach.
Analyzes rolling telemetry from PPO rollouts and checkpoints to visualize action biases,
spatial tendencies, and automatically flag bad habits with actionable reward recommendations.
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_rolling_telemetry(history_file: str = "logs/history.jsonl", window: int = 8) -> Dict[str, Any]:
    """
    Reads the last window iterations from history.jsonl and calculates the mean behavioral telemetry.
    """
    if not os.path.exists(history_file):
        return {}

    records = []
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line.strip()))
                    except Exception:
                        pass
    except Exception:
        return {}

    if not records:
        return {}

    recent = records[-window:]
    telemetries = [r.get("telemetry", {}) for r in recent if "telemetry" in r]

    if not telemetries:
        telemetries = [r for r in recent if "jump_rate_pct" in r]

    if not telemetries:
        return {}

    avg_telemetry: Dict[str, float] = {}
    keys = telemetries[0].keys()
    for k in keys:
        vals = [t[k] for t in telemetries if k in t and isinstance(t[k], (int, float))]
        if vals:
            avg_telemetry[k] = float(np.mean(vals))

    avg_telemetry["sample_iterations"] = len(telemetries)
    avg_telemetry["latest_iteration"] = recent[-1].get("iteration", 0)
    avg_telemetry["latest_step"] = recent[-1].get("global_step", 0)
    return avg_telemetry


def render_action_biases_plot(telemetry: Dict[str, Any]) -> plt.Figure:
    """
    Renders a dark-themed horizontal bar chart of the 8 controller action distributions.
    """
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=100)
    fig.patch.set_facecolor("#1a202c")
    ax.set_facecolor("#2d3748")

    if not telemetry:
        ax.text(0.5, 0.5, "No Behavioral Telemetry Recorded Yet\n(Will appear as training iterations complete)",
                color="#a0aec0", fontsize=12, ha="center", va="center")
        ax.set_axis_off()
        plt.tight_layout()
        return fig

    items = [
        ("Forward Throttle (>20%)", telemetry.get("throttle_forward_pct", 0.0), "#48bb78"),
        ("Coast / Neutral Throttle", telemetry.get("throttle_coast_pct", 0.0), "#a0aec0"),
        ("Reverse / Braking (<-20%)", telemetry.get("throttle_reverse_pct", 0.0), "#f56565"),
        ("Left Steering (<-20%)", telemetry.get("steer_left_pct", 0.0), "#4299e1"),
        ("Straight Driving (|steer|<=20%)", telemetry.get("steer_straight_pct", 0.0), "#cbd5e0"),
        ("Right Steering (>20%)", telemetry.get("steer_right_pct", 0.0), "#ed8936"),
        ("Jump Activation Rate", telemetry.get("jump_rate_pct", 0.0), "#9f7aea"),
        ("Boost Usage Rate", telemetry.get("boost_rate_pct", 0.0), "#ecc94b"),
        ("Handbrake / Drift Rate", telemetry.get("handbrake_rate_pct", 0.0), "#ed64a6"),
    ]

    labels = [it[0] for it in items]
    vals = [it[1] for it in items]
    colors = [it[2] for it in items]
    y_pos = np.arange(len(items))

    bars = ax.barh(y_pos, vals, height=0.55, color=colors, alpha=0.9, edgecolor="#ffffff", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="#e2e8f0", fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Usage Frequency (% of Simulation Steps)", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax.set_title("Action & Control Biases (Rolling Average)", color="white", fontsize=12, fontweight="bold", pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a5568")
    ax.spines["bottom"].set_color("#4a5568")
    ax.grid(axis="x", linestyle=":", alpha=0.4, color="#718096")
    ax.tick_params(colors="#cbd5e0")

    for bar, val in zip(bars, vals):
        ax.annotate(f"{val:.1f}%", (val + 1.2, bar.get_y() + bar.get_height() / 2),
                    color="#ffffff", fontsize=9, va="center", fontweight="bold")

    plt.tight_layout()
    return fig


def render_positional_biases_plot(telemetry: Dict[str, Any]) -> plt.Figure:
    """
    Renders a dark-themed bar chart of pitch territorial distribution and vehicle physical state.
    """
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=100)
    fig.patch.set_facecolor("#1a202c")
    ax.set_facecolor("#2d3748")

    if not telemetry:
        ax.text(0.5, 0.5, "No Positional Telemetry Recorded Yet",
                color="#a0aec0", fontsize=12, ha="center", va="center")
        ax.set_axis_off()
        plt.tight_layout()
        return fig

    items = [
        ("Defensive Third (Y < -33%)", telemetry.get("defensive_third_pct", 0.0), "#3182ce"),
        ("Midfield Neutral Third", telemetry.get("midfield_third_pct", 0.0), "#38b2ac"),
        ("Offensive Third (Y > +33%)", telemetry.get("offensive_third_pct", 0.0), "#dd6b20"),
        ("Corner Dead-Zone Time", telemetry.get("corner_zone_pct", 0.0), "#e53e3e"),
        ("Airborne Time (Off Ground)", telemetry.get("air_time_pct", 0.0), "#805ad5"),
        ("Ground Driving Time", telemetry.get("ground_time_pct", 0.0), "#48bb78"),
        ("Empty Boost Time (<1% Boost)", telemetry.get("zero_boost_pct", 0.0), "#d69e2e"),
        ("Average Boost Tank Level", telemetry.get("mean_boost_tank", 0.0), "#319795"),
    ]

    labels = [it[0] for it in items]
    vals = [it[1] for it in items]
    colors = [it[2] for it in items]
    y_pos = np.arange(len(items))

    bars = ax.barh(y_pos, vals, height=0.55, color=colors, alpha=0.9, edgecolor="#ffffff", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="#e2e8f0", fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Time Spent (% of Steps) / Level", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax.set_title("Pitch Territorial & Physical State Biases", color="white", fontsize=12, fontweight="bold", pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a5568")
    ax.spines["bottom"].set_color("#4a5568")
    ax.grid(axis="x", linestyle=":", alpha=0.4, color="#718096")
    ax.tick_params(colors="#cbd5e0")

    for bar, val in zip(bars, vals):
        unit = "%" if "Level" not in labels[int(bar.get_y())] else " boost"
        ax.annotate(f"{val:.1f}{unit}", (val + 1.2, bar.get_y() + bar.get_height() / 2),
                    color="#ffffff", fontsize=9, va="center", fontweight="bold")

    plt.tight_layout()
    return fig


def generate_ai_coach_diagnostics(telemetry: Dict[str, Any], active_rewards: Optional[Dict[str, float]] = None) -> str:
    """
    Analyzes telemetry to produce automated alerts, bad habit warnings, and tuning tips.
    """
    if not telemetry:
        return "### ⏳ Waiting for Behavioral Telemetry Data...\nRun training for at least 1-2 iterations to generate rolling habit diagnostics."

    active_rewards = active_rewards or {}

    alerts = []
    healthy = []
    tips = []

    samples = telemetry.get("sample_iterations", 1)
    iter_num = telemetry.get("latest_iteration", 0)

    corner_pct = telemetry.get("corner_zone_pct", 0.0)
    if corner_pct > 25.0:
        cur_b2g = active_rewards.get("ball_to_goal_weight", 1.5)
        alerts.append(f"🚨 **High Corner Trapping ({corner_pct:.1f}%)**: The bot is spending over a quarter of the match trapped in corner dead-zones.")
        tips.append(f"🔧 **Fix:** Increase **Ball-to-Goal Velocity (`ball_to_goal_weight`)** (currently `{cur_b2g:.2f}`, recommended `1.8` – `2.2`) to incentivize centering and clearing the ball.")
    else:
        healthy.append(f"✅ **Good Field Spacing**: Corner trapping is low ({corner_pct:.1f}%).")

    jump_pct = telemetry.get("jump_rate_pct", 0.0)
    air_pct = telemetry.get("air_time_pct", 0.0)
    if jump_pct < 1.5 and air_pct < 3.0:
        alerts.append(f"🚨 **Grounded / Low Aerial Rate (Jump: {jump_pct:.1f}%, Air: {air_pct:.1f}%)**: The bot is staying glued to the floor.")
        tips.append("🔧 **Fix:** Increase **Aerial Scenario Probability (`aerial_prob`)** to `0.25` in Scenario Settings to train airborne challenges.")
    elif jump_pct > 65.0:
        cur_bl = active_rewards.get("boost_lose_weight", 0.3)
        alerts.append(f"⚠️ **Jump Spamming ({jump_pct:.1f}%)**: The bot is spamming jump constantly, losing ground steering traction.")
        tips.append(f"🔧 **Fix:** Increase **Ground Boost Waste Penalty (`boost_lose_weight`)** (currently `{cur_bl:.2f}`, recommended `0.5` – `0.8`) to encourage stable driving lines.")
    else:
        healthy.append(f"✅ **Active Aerial Play**: Jump rate is {jump_pct:.1f}% with {air_pct:.1f}% airtime.")

    left_pct = telemetry.get("steer_left_pct", 0.0)
    right_pct = telemetry.get("steer_right_pct", 0.0)
    steer_diff = abs(left_pct - right_pct)
    if steer_diff > 30.0:
        dominant = "Left" if left_pct > right_pct else "Right"
        cur_p2b = active_rewards.get("player_to_ball_weight", 0.6)
        alerts.append(f"🚨 **Steer Asymmetry / Donut Bias**: Turning {dominant} {max(left_pct, right_pct):.1f}% vs {min(left_pct, right_pct):.1f}%. The bot has developed a circular driving habit.")
        tips.append(f"🔧 **Fix:** Increase **Player-to-Ball Pursuit (`player_to_ball_weight`)** (currently `{cur_p2b:.2f}`, recommended `1.0` – `1.4`) and **Touch Quality (`touch_weight`)** to force direct approaches.")
    else:
        healthy.append(f"✅ **Balanced Steering**: Left ({left_pct:.1f}%) and Right ({right_pct:.1f}%) steering are well-balanced.")

    zero_boost = telemetry.get("zero_boost_pct", 0.0)
    mean_boost = telemetry.get("mean_boost_tank", 33.3)
    if zero_boost > 35.0 or mean_boost < 15.0:
        cur_bg = active_rewards.get("boost_gain_weight", 0.6)
        alerts.append(f"🚨 **Boost Starvation (Empty: {zero_boost:.1f}%, Avg: {mean_boost:.1f} boost)**: The bot is frequently driving on empty tanks.")
        tips.append(f"🔧 **Fix:** Increase **Boost Pickup Gain Weight (`boost_gain_weight`)** (currently `{cur_bg:.2f}`, recommended `0.8` – `1.2`) or raise **Ground Waste Penalty (`boost_lose_weight`)**.")
    else:
        healthy.append(f"✅ **Healthy Boost Reserves**: Average tank is {mean_boost:.1f} boost ({zero_boost:.1f}% empty).")

    rev_pct = telemetry.get("throttle_reverse_pct", 0.0)
    fwd_pct = telemetry.get("throttle_forward_pct", 0.0)
    if rev_pct > 30.0:
        cur_p2b = active_rewards.get("player_to_ball_weight", 0.6)
        alerts.append(f"⚠️ **Excessive Reversing ({rev_pct:.1f}%)**: The bot is spending significant time backing up rather than rotating forward.")
        tips.append(f"🔧 **Fix:** Increase **Player-to-Ball Pursuit (`player_to_ball_weight`)** (currently `{cur_p2b:.2f}`, recommended `1.0` – `1.4`).")
    else:
        healthy.append(f"✅ **Forward Aggression**: Driving forward {fwd_pct:.1f}% of the time.")

    report = f"### 🧠 AI Coach Behavioral Diagnosis (Averaged over last {samples} iters | Iter #{iter_num})\n\n#### 🚨 Detected Bad Habits & Alerts:\n"
    if alerts:
        for a in alerts:
            report += f"* {a}\n"
    else:
        report += "* *No critical bad habits detected! Bot behavior is well-balanced.*\n"

    report += "\n#### 🟢 Healthy Mechanics:\n"
    for h in healthy:
        report += f"* {h}\n"

    if tips:
        report += "\n#### 🎯 Recommended Reward Adjustments:\n"
        for t in tips:
            report += f"* {t}\n"

    return report


def render_training_curves_plot(history_file: str = "logs/history.jsonl", max_points: int = 100, mode: str = "recent") -> plt.Figure:
    """
    Renders a 4-panel dark-themed live training progress chart:
    1. Mean Reward Curve
    2. Policy & Value Losses
    3. Policy Entropy (Exploration)
    4. Throughput (SPS) & Ball Touches
    Uses O(1) seek-from-tail for recent iterations (<1ms) or byte-stepped downsampling for full run history.
    """
    plt.close("all")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.0), dpi=100)
    fig.patch.set_facecolor("#1a202c")
    for ax in axes.flat:
        ax.set_facecolor("#2d3748")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#4a5568")
        ax.spines["bottom"].set_color("#4a5568")
        ax.grid(True, linestyle=":", alpha=0.35, color="#718096")
        ax.tick_params(colors="#cbd5e0", labelsize=8)

    if not os.path.exists(history_file):
        for ax in axes.flat:
            ax.set_axis_off()
        axes[0, 0].set_axis_on()
        axes[0, 0].text(0.5, 0.5, "Waiting for Training Data...\n(Metrics will display once training is active)",
                        color="#a0aec0", fontsize=11, ha="center", va="center")
        plt.tight_layout()
        return fig

    records = []
    try:
        file_size = os.path.getsize(history_file)
        if file_size > 0:
            if mode == "full" and file_size > max_points * 1500:
                # Byte-stepped seek downsampling across massive files (O(1) memory, ~5ms)
                step = file_size / max_points
                with open(history_file, "rb") as f:
                    for i in range(max_points):
                        f.seek(int(i * step))
                        if i > 0:
                            f.readline()  # Skip partial line
                        line = f.readline().decode("utf-8", errors="ignore").strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except Exception:
                                pass
            else:
                # Seek from tail for recent window (O(1) time & memory, <1ms regardless of file size)
                buf_size = min(file_size, max(8192, max_points * 2500))
                with open(history_file, "rb") as f:
                    f.seek(file_size - buf_size)
                    raw = f.read().decode("utf-8", errors="ignore")
                    lines = raw.splitlines()
                    if file_size > buf_size and lines:
                        lines.pop(0)  # Discard partial initial line
                    lines = lines[-max_points:]
                    for l in lines:
                        l = l.strip()
                        if l:
                            try:
                                records.append(json.loads(l))
                            except Exception:
                                pass
    except Exception:
        pass

    if not records:
        for ax in axes.flat:
            ax.set_axis_off()
        axes[0, 0].set_axis_on()
        axes[0, 0].text(0.5, 0.5, "No Training Data Found", color="#a0aec0", fontsize=11, ha="center", va="center")
        plt.tight_layout()
        return fig

    iters = [r.get("iteration", i + 1) for i, r in enumerate(records)]
    rewards = [r.get("mean_reward", 0.0) for r in records]
    p_losses = [r.get("policy_loss", 0.0) for r in records]
    v_losses = [r.get("value_loss", 0.0) for r in records]
    entropies = [r.get("entropy", 0.0) for r in records]
    sps_vals = [r.get("sps", 0) for r in records]
    touches = [r.get("ball_touches", 0.0) for r in records]
    title_suffix = " (Full Run)" if mode == "full" else f" (Recent {len(iters)} Iters)"

    # Panel 1: Mean Reward
    ax_rew = axes[0, 0]
    ax_rew.plot(iters, rewards, color="#38bdf8", linewidth=1.8, label="Mean Reward")
    if len(rewards) >= 5:
        # 5-step rolling moving average
        kernel = np.ones(5) / 5.0
        smooth = np.convolve(rewards, kernel, mode="valid")
        smooth_iters = iters[len(iters) - len(smooth):]
        ax_rew.plot(smooth_iters, smooth, color="#0284c7", linewidth=2.5, linestyle="--", alpha=0.85, label="Trend (MA-5)")
    ax_rew.set_title(f"Mean Reward{title_suffix}", color="white", fontsize=10, fontweight="bold", pad=6)
    ax_rew.set_ylabel("Reward Points", color="#cbd5e0", fontsize=8)
    ax_rew.legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=7)

    # Panel 2: Losses
    ax_loss = axes[0, 1]
    ax_loss.plot(iters, p_losses, color="#f87171", linewidth=1.5, label="Policy Loss")
    ax_loss.set_title("Losses (Policy & Value)", color="white", fontsize=10, fontweight="bold", pad=6)
    ax_loss.set_ylabel("Policy Loss", color="#f87171", fontsize=8)
    ax_loss.tick_params(axis="y", labelcolor="#f87171")
    
    ax_vloss = ax_loss.twinx()
    ax_vloss.plot(iters, v_losses, color="#facc15", linewidth=1.5, linestyle=":", label="Value Loss")
    ax_vloss.set_ylabel("Value Loss", color="#facc15", fontsize=8)
    ax_vloss.tick_params(axis="y", labelcolor="#facc15", labelsize=8)
    ax_vloss.spines["top"].set_visible(False)
    ax_vloss.spines["left"].set_visible(False)
    ax_vloss.spines["right"].set_color("#4a5568")

    # Panel 3: Policy Entropy
    ax_ent = axes[1, 0]
    ax_ent.plot(iters, entropies, color="#c084fc", linewidth=1.8, label="Policy Entropy")
    ax_ent.set_title("Policy Entropy (Exploration)", color="white", fontsize=10, fontweight="bold", pad=6)
    ax_ent.set_xlabel("Training Iteration", color="#cbd5e0", fontsize=8)
    ax_ent.set_ylabel("Entropy", color="#cbd5e0", fontsize=8)

    # Panel 4: SPS Throughput & Ball Touches
    ax_sps = axes[1, 1]
    ax_sps.plot(iters, sps_vals, color="#4ade80", linewidth=1.5, label="Throughput (SPS)")
    ax_sps.set_title("Throughput (SPS) & Ball Touches", color="white", fontsize=10, fontweight="bold", pad=6)
    ax_sps.set_xlabel("Training Iteration", color="#cbd5e0", fontsize=8)
    ax_sps.set_ylabel("SPS (Steps/Sec)", color="#4ade80", fontsize=8)
    ax_sps.tick_params(axis="y", labelcolor="#4ade80")

    ax_tch = ax_sps.twinx()
    ax_tch.plot(iters, touches, color="#60a5fa", linewidth=1.5, linestyle="--", label="Touches")
    ax_tch.set_ylabel("Avg Touches", color="#60a5fa", fontsize=8)
    ax_tch.tick_params(axis="y", labelcolor="#60a5fa", labelsize=8)
    ax_tch.spines["top"].set_visible(False)
    ax_tch.spines["left"].set_visible(False)
    ax_tch.spines["right"].set_color("#4a5568")

    plt.tight_layout()
    return fig