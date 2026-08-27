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


def generate_ai_coach_diagnostics(telemetry: Dict[str, Any], active_rewards: Dict[str, float]) -> str:
    """
    Analyzes telemetry to produce automated alerts, bad habit warnings, and tuning tips.
    """
    if not telemetry:
        return "### ⏳ Waiting for Behavioral Telemetry Data...\nRun training for at least 1-2 iterations to generate rolling habit diagnostics."

    alerts = []
    healthy = []
    tips = []

    samples = telemetry.get("sample_iterations", 1)
    iter_num = telemetry.get("latest_iteration", 0)

    corner_pct = telemetry.get("corner_zone_pct", 0.0)
    if corner_pct > 25.0:
        alerts.append(f"🚨 **High Corner Trapping ({corner_pct:.1f}%)**: The bot is spending over a quarter of the match trapped in corner dead-zones.")
        tips.append("🔧 **Fix:** Lower **Goal-Side Rotation Weight (`behind_ball_weight`)** and increase **Navigation & Forward Speed (`speed_toward_ball_weight`)** to force center clears.")
    else:
        healthy.append(f"✅ **Good Field Spacing**: Corner trapping is low ({corner_pct:.1f}%).")

    jump_pct = telemetry.get("jump_rate_pct", 0.0)
    air_pct = telemetry.get("air_time_pct", 0.0)
    if jump_pct < 1.5 and air_pct < 3.0:
        alerts.append(f"🚨 **Grounded / Jump Paralysis (Jump: {jump_pct:.1f}%, Air: {air_pct:.1f}%)**: The bot is staying completely glued to the floor.")
        tips.append("🔧 **Fix:** Increase **Tactical Aerial Flight Climb (`aerial_height_weight`)** (e.g. `0.10` – `0.15`) and **High Aerial Strike Bounty (`high_aerial_bounty`)**.")
    elif jump_pct > 65.0:
        alerts.append(f"⚠️ **Jump Spamming ({jump_pct:.1f}%)**: The bot is spamming jump constantly, losing ground steering traction.")
        tips.append("🔧 **Fix:** Slightly reduce **Tactical Aerial Flight Climb (`aerial_height_weight`)**.")
    else:
        healthy.append(f"✅ **Active Aerial Play**: Jump rate is {jump_pct:.1f}% with {air_pct:.1f}% airtime.")

    left_pct = telemetry.get("steer_left_pct", 0.0)
    right_pct = telemetry.get("steer_right_pct", 0.0)
    steer_diff = abs(left_pct - right_pct)
    if steer_diff > 30.0:
        dominant = "Left" if left_pct > right_pct else "Right"
        alerts.append(f"🚨 **Steer Asymmetry / Donut Bias**: Turning {dominant} {max(left_pct, right_pct):.1f}% vs {min(left_pct, right_pct):.1f}%. The bot has developed a circular driving habit.")
        tips.append("🔧 **Fix:** Increase **Navigation & Forward Speed (`speed_toward_ball_weight`)** and **Ball Contact Base Hit (`touch_ball_weight`)** to break circular turning.")
    else:
        healthy.append(f"✅ **Balanced Steering**: Left ({left_pct:.1f}%) and Right ({right_pct:.1f}%) steering are well-balanced.")

    zero_boost = telemetry.get("zero_boost_pct", 0.0)
    mean_boost = telemetry.get("mean_boost_tank", 33.3)
    if zero_boost > 35.0 or mean_boost < 15.0:
        alerts.append(f"🚨 **Boost Starvation (Empty: {zero_boost:.1f}%, Avg: {mean_boost:.1f} boost)**: The bot is frequently driving on empty tanks.")
        tips.append("🔧 **Fix:** Increase **Small Boost Pad Weight (`small_pad_weight`)** (e.g. `6.0` – `10.0`) and **Big Orb Weight (`big_pad_weight`)** (`18.0`).")
    else:
        healthy.append(f"✅ **Healthy Boost Reserves**: Average tank is {mean_boost:.1f} boost ({zero_boost:.1f}% empty).")

    rev_pct = telemetry.get("throttle_reverse_pct", 0.0)
    fwd_pct = telemetry.get("throttle_forward_pct", 0.0)
    if rev_pct > 30.0:
        alerts.append(f"⚠️ **Excessive Reversing ({rev_pct:.1f}%)**: The bot is spending significant time backing up rather than rotating forward.")
        tips.append("🔧 **Fix:** Increase **Navigation & Forward Speed (`speed_toward_ball_weight`)** and verify forward drive bias.")
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