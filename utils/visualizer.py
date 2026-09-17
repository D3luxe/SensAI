"""
Rocket League Visualizer & Match Replay Generator.
Renders top-down 2D arena views, trajectories, and bot match evaluations.
"""

from __future__ import annotations
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from env.physics_engine import (
    RocketSimArena, CarState, BallState, BoostPad,
    ARENA_EXTENT_X, ARENA_EXTENT_Y, GOAL_HALF_WIDTH, GOAL_HEIGHT,
    CAR_LENGTH, CAR_WIDTH, BALL_RADIUS
)
from env.observations import DefaultObservationBuilder, OBS_MIRROR_MASK_NP, ACT_MIRROR_MASK_NP
from env.actions import ContinuousActionParser, DiscreteActionParser
from env.rewards import RewardManager
from env.baseline_agent import BaseOpponent, BaselineChaser, NectoNextoOpponentBot, create_opponent_bot
from agent.models import ActorCritic
from agent.checkpoint import load_policy, read_checkpoint
from utils.config import effective_reward_weights, effective_reward_weights_for_checkpoint


def draw_rocket_league_pitch(ax):
    """
    Draws standard Rocket League arena boundaries, boost pads, and goal nets on a matplotlib axes.
    """
    # Background
    ax.set_facecolor("#1a202c")

    # Arena outline
    arena_rect = patches.Rectangle(
        (-ARENA_EXTENT_X, -ARENA_EXTENT_Y),
        ARENA_EXTENT_X * 2,
        ARENA_EXTENT_Y * 2,
        linewidth=2,
        edgecolor="#4a5568",
        facecolor="#2d3748"
    )
    ax.add_patch(arena_rect)

    # Midfield line
    ax.plot([-ARENA_EXTENT_X, ARENA_EXTENT_X], [0, 0], color="#718096", linestyle="--", linewidth=1.5)
    # Center circle
    center_circle = patches.Circle((0, 0), 1000, color="#718096", fill=False, linestyle="--", linewidth=1.5)
    ax.add_patch(center_circle)

    # Blue Goal (Bottom, Y = -ARENA_EXTENT_Y)
    blue_goal = patches.Rectangle(
        (-GOAL_HALF_WIDTH, -ARENA_EXTENT_Y - 800),
        GOAL_HALF_WIDTH * 2,
        800,
        linewidth=2,
        edgecolor="#3182ce",
        facecolor="#2b6cb0",
        alpha=0.6
    )
    ax.add_patch(blue_goal)

    # Orange Goal (Top, Y = +ARENA_EXTENT_Y)
    orange_goal = patches.Rectangle(
        (-GOAL_HALF_WIDTH, ARENA_EXTENT_Y),
        GOAL_HALF_WIDTH * 2,
        800,
        linewidth=2,
        edgecolor="#dd6b20",
        facecolor="#c05621",
        alpha=0.6
    )
    ax.add_patch(orange_goal)

    # Boost pads
    pads = BoostPad.create_standard_pads()
    for pad in pads:
        if pad.is_big:
            pad_circle = patches.Circle((pad.pos[0], pad.pos[1]), 180, color="#ecc94b", alpha=0.8)
        else:
            pad_circle = patches.Circle((pad.pos[0], pad.pos[1]), 80, color="#d69e2e", alpha=0.5)
        ax.add_patch(pad_circle)

    ax.set_xlim(-ARENA_EXTENT_X - 1000, ARENA_EXTENT_X + 1000)
    ax.set_ylim(-ARENA_EXTENT_Y - 1200, ARENA_EXTENT_Y + 1200)
    ax.set_aspect("equal")
    ax.axis("off")


def load_model(model_path: Optional[str], device: str = "cpu") -> Optional[ActorCritic]:
    """The checkpoint's policy through the shared loader, or None if there is none to load."""
    if not model_path:
        return None
    norm_path = os.path.normpath(model_path.strip().strip('"').strip("'"))
    if not os.path.exists(norm_path):
        return None
    try:
        return load_policy(norm_path, device=device)[0]
    except Exception as e:
        print(f"[Visualizer] Could not load model {norm_path}: {e}")
        return None


def _reward_weights_for(model_path: Optional[str]) -> Dict[str, float]:
    """Reward weights as training applies them, at the given checkpoint's step when it has one."""
    if model_path and os.path.exists(model_path):
        try:
            return effective_reward_weights_for_checkpoint(read_checkpoint(model_path))
        except Exception:
            pass
    return effective_reward_weights()


# Canonical metadata for all reward components and regularization terms
REWARD_METADATA: Dict[str, Dict[str, Any]] = {
    "goal": {
        "label": "Goals, Concedes & Saves",
        "category": "Primary Objectives",
        "desc": "Scoring (+30) / Conceding (-30) / Saves",
        "priority": 1,
    },
    "ball_to_goal": {
        "label": "Ball to Goal Progression",
        "category": "Primary Objectives",
        "desc": "Proximity-gated ball velocity toward opponent goal",
        "priority": 2,
    },
    "touch": {
        "label": "Ball Touches & Power Strikes",
        "category": "Primary Objectives",
        "desc": "Velocity & target-aligned ball strikes",
        "priority": 3,
    },
    "player_to_ball": {
        "label": "Ball Pursuit & Strike Pacing",
        "category": "Movement & Positioning",
        "desc": "PBRS approach delta and strike zone pacing",
        "priority": 4,
    },
    "boost": {
        "label": "Boost Economy & Pad Transit",
        "category": "Movement & Positioning",
        "desc": "PBRS conservation & trajectory-aligned pad collection",
        "priority": 5,
    },
    "powerslide": {
        "label": "Powerslide Turnarounds",
        "category": "Movement & Positioning",
        "desc": "Ground turnarounds & momentum cuts at distance",
        "priority": 6,
    },
    "air_roll_recovery": {
        "label": "Air-Roll Landing Recovery",
        "category": "Movement & Positioning",
        "desc": "Upright orientation recovery and attitude damping",
        "priority": 7,
    },
    "jump_bridge": {
        "label": "Aerial Jump Bridges",
        "category": "Movement & Positioning",
        "desc": "Launch incentives for intercepting aerial balls",
        "priority": 8,
    },
    "retreat_flip": {
        "label": "Retreat Flip",
        "category": "Movement & Positioning",
        "desc": "Dodges that add speed toward the defensive recovery point when beaten upfield",
        "priority": 8,
    },
    "time_cost": {
        "label": "Time Cost (Dawdle Penalty)",
        "category": "Penalties & Costs",
        "desc": "Per-step living cost incentivizing decisive action",
        "priority": 9,
    },
    "spin_cost": {
        "label": "Spin & Tumbling Fee",
        "category": "Penalties & Costs",
        "desc": "Per-step fee on excessive airborne angular tumbling",
        "priority": 10,
    },
    "jump_cost": {
        "label": "Jump / Flip Action Cost",
        "category": "Penalties & Costs",
        "desc": "Cost per jump/dodge to discourage action spamming",
        "priority": 11,
    },
    "own_goal_threat": {
        "label": "Own-Goal Threat Penalty",
        "category": "Penalties & Costs",
        "desc": "Penalty for dangerous fast ball movement towards own net",
        "priority": 12,
    },
    "lateral_slip_penalty": {
        "label": "Strike-Zone Lateral Slip",
        "category": "Penalties & Costs",
        "desc": "Penalty for sliding sideways across the strike zone",
        "priority": 13,
    },
    "handbrake_penalty": {
        "label": "Handbrake Drag Penalty",
        "category": "Penalties & Costs",
        "desc": "Penalty for dragging handbrake during forward drives",
        "priority": 14,
    },
}

CORE_REWARD_KEYS = ["goal", "ball_to_goal", "touch", "player_to_ball", "boost"]


def render_reward_breakdown_plot(
    blue_rewards: Dict[str, float],
    orange_rewards: Optional[Dict[str, float]] = None,
    match_type: str = "Match Evaluation",
    blue_timeline: Optional[List[float]] = None,
    orange_timeline: Optional[List[float]] = None,
    goal_events: Optional[List[Dict[str, Any]]] = None,
) -> plt.Figure:
    """
    Renders a dark-themed match reward evaluation figure.
    - Left panel: Horizontal bar chart showing points earned/penalized by reward component.
    - Right panel (if timeline provided): Cumulative reward trajectory over match steps with goal markers.
    """
    # 1. Dynamically select all relevant active reward categories
    all_keys: List[str] = []
    for k in REWARD_METADATA:
        b_val = blue_rewards.get(k, 0.0)
        o_val = orange_rewards.get(k, 0.0) if orange_rewards else 0.0
        # Include core categories or any category with non-negligible activity
        if k in CORE_REWARD_KEYS or abs(b_val) >= 0.005 or abs(o_val) >= 0.005:
            all_keys.append(k)

    # Capture any custom / non-standard keys returned by RewardManager
    for k in list(blue_rewards.keys()) + (list(orange_rewards.keys()) if orange_rewards else []):
        if k not in all_keys:
            b_val = blue_rewards.get(k, 0.0)
            o_val = orange_rewards.get(k, 0.0) if orange_rewards else 0.0
            if abs(b_val) >= 0.005 or abs(o_val) >= 0.005:
                all_keys.append(k)

    # Order logically by defined priority (Objectives -> Mechanics -> Costs)
    all_keys.sort(key=lambda k: REWARD_METADATA.get(k, {}).get("priority", 999))

    labels = [REWARD_METADATA.get(k, {}).get("label", k.replace("_", " ").title()) for k in all_keys]
    blue_vals = [blue_rewards.get(k, 0.0) for k in all_keys]
    orange_vals = [orange_rewards.get(k, 0.0) if orange_rewards else 0.0 for k in all_keys]

    blue_total = sum(blue_rewards.values())
    orange_total = sum(orange_rewards.values()) if orange_rewards else 0.0

    has_timeline = bool(blue_timeline and len(blue_timeline) > 0)
    fig_height = max(6.0, 0.44 * len(all_keys))

    if has_timeline:
        fig, (ax_bar, ax_time) = plt.subplots(
            1, 2, figsize=(15.0, fig_height), dpi=100, gridspec_kw={"width_ratios": [1.18, 0.82]}
        )
    else:
        fig, ax_bar = plt.subplots(figsize=(9.5, fig_height), dpi=100)
        ax_time = None

    fig.patch.set_facecolor("#1a202c")
    ax_bar.set_facecolor("#2d3748")

    y_pos = np.arange(len(all_keys))
    bar_height = 0.38 if orange_rewards else 0.55

    # Plot horizontal bars
    if orange_rewards:
        b_bars = ax_bar.barh(
            y_pos - bar_height / 2, blue_vals, bar_height,
            label=f"Blue Team ({blue_total:+.1f})", color="#4299e1", alpha=0.9, edgecolor="#bee3f8"
        )
        o_bars = ax_bar.barh(
            y_pos + bar_height / 2, orange_vals, bar_height,
            label=f"Orange Team ({orange_total:+.1f})", color="#ed8936", alpha=0.9, edgecolor="#feebc8"
        )
        ax_bar.legend(loc="lower right", facecolor="#1a202c", edgecolor="#4a5568", labelcolor="white", fontsize=9)
    else:
        b_bars = ax_bar.barh(
            y_pos, blue_vals, bar_height,
            label=f"Blue Team ({blue_total:+.1f})", color="#4299e1", alpha=0.9, edgecolor="#bee3f8"
        )
        o_bars = []

    # Value annotations on bars (both Blue and Orange)
    for bar in b_bars:
        w = bar.get_width()
        if abs(w) >= 0.01:
            txt = f"{w:+.2f}" if abs(w) < 1.0 else f"{w:+.1f}"
            ax_bar.annotate(
                txt,
                xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(4 if w >= 0 else -4, 0),
                textcoords="offset points",
                va="center",
                ha="left" if w >= 0 else "right",
                color="#bee3f8",
                fontsize=8,
                fontweight="bold"
            )

    for bar in o_bars:
        w = bar.get_width()
        if abs(w) >= 0.01:
            txt = f"{w:+.2f}" if abs(w) < 1.0 else f"{w:+.1f}"
            ax_bar.annotate(
                txt,
                xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(4 if w >= 0 else -4, 0),
                textcoords="offset points",
                va="center",
                ha="left" if w >= 0 else "right",
                color="#feebc8",
                fontsize=8,
                fontweight="bold"
            )

    # Dynamic X limits with margin padding so value labels never clip
    all_vals = blue_vals + (orange_vals if orange_rewards else [])
    min_w = min(all_vals) if all_vals else 0.0
    max_w = max(all_vals) if all_vals else 0.0
    span = max(2.0, max_w - min_w)
    ax_bar.set_xlim(min(0.0, min_w) - 0.16 * span, max(0.0, max_w) + 0.18 * span)

    ax_bar.axvline(0, color="#718096", linestyle="--", linewidth=1.0)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, color="#e2e8f0", fontsize=9.5, fontweight="medium")
    ax_bar.invert_yaxis()  # Puts top priority (Objectives) at the top of the chart
    ax_bar.tick_params(colors="#cbd5e0")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color("#4a5568")
    ax_bar.spines["bottom"].set_color("#4a5568")
    ax_bar.grid(axis="x", linestyle=":", alpha=0.4, color="#718096")
    ax_bar.set_xlabel("Cumulative Points Earned", color="#e2e8f0", fontsize=10, fontweight="bold")

    title_text = f"Match Reward Breakdown ({match_type})\n"
    if orange_rewards:
        title_text += f"Blue Net: {blue_total:+.2f}  |  Orange Net: {orange_total:+.2f}"
    else:
        title_text += f"Blue Net: {blue_total:+.2f}"
    ax_bar.set_title(title_text, color="white", fontsize=11, fontweight="bold", pad=10)

    # 2. Cumulative Reward Timeline Plot (if timeline provided)
    if has_timeline and ax_time is not None:
        ax_time.set_facecolor("#2d3748")
        steps_arr = np.arange(len(blue_timeline))
        ax_time.plot(steps_arr, blue_timeline, color="#63b3ed", label=f"Blue ({blue_total:+.1f})", linewidth=2.0)
        if orange_timeline is not None and len(orange_timeline) == len(blue_timeline):
            ax_time.plot(steps_arr, orange_timeline, color="#f6ad55", label=f"Orange ({orange_total:+.1f})", linewidth=2.0)

        ax_time.axhline(0, color="#718096", linestyle="--", linewidth=1.0, alpha=0.7)

        # Mark goal events on timeline
        if goal_events:
            for ge in goal_events:
                g_step = ge.get("step", 0)
                g_team = ge.get("team", 0)
                g_color = "#63b3ed" if g_team == 0 else "#f6ad55"
                ax_time.axvline(g_step, color=g_color, linestyle=":", alpha=0.75, linewidth=1.5)
                y_val = blue_timeline[g_step] if g_team == 0 else (orange_timeline[g_step] if orange_timeline else 0.0)
                ax_time.scatter([g_step], [y_val], marker="*", s=140, color=g_color, edgecolor="white", zorder=5)

        ax_time.set_xlabel("Simulation Step (15 steps ≈ 1s)", color="#e2e8f0", fontsize=10, fontweight="bold")
        ax_time.set_ylabel("Cumulative Reward", color="#e2e8f0", fontsize=10, fontweight="bold")
        ax_time.set_title("Reward Trajectory Over Match", color="white", fontsize=11, fontweight="bold", pad=10)
        ax_time.legend(loc="best", facecolor="#1a202c", edgecolor="#4a5568", labelcolor="white", fontsize=9)
        ax_time.tick_params(colors="#cbd5e0", labelsize=9)
        ax_time.grid(True, linestyle=":", alpha=0.35, color="#718096")
        ax_time.spines["top"].set_visible(False)
        ax_time.spines["right"].set_visible(False)
        ax_time.spines["left"].set_color("#4a5568")
        ax_time.spines["bottom"].set_color("#4a5568")

    plt.tight_layout()
    return fig


def simulate_match(
    blue_model_path: Optional[str] = None,
    orange_model_path: Optional[str] = "same_as_blue",
    max_steps: int = 400,
    device: str = "cpu",
    **kwargs
) -> Tuple[plt.Figure, plt.Figure, Dict[str, Any]]:
    """
    Simulates a match between Blue and Orange agents and produces:
    1. 2D trajectory pitch plot.
    2. Detailed Reward Breakdown Bar Chart and Timeline.
    3. Match statistics dict.
    """
    if "blue_checkpoint" in kwargs and kwargs["blue_checkpoint"] is not None:
        blue_model_path = kwargs["blue_checkpoint"]
    if "orange_checkpoint" in kwargs and kwargs["orange_checkpoint"] is not None:
        orange_model_path = kwargs["orange_checkpoint"]
    if "steps" in kwargs and kwargs["steps"] is not None:
        max_steps = int(kwargs["steps"])

    arena = RocketSimArena(num_players=2, game_mode="1v1")
    # For full match replay visualization, always start from a standard competitive kickoff
    arena.reset(random_kickoff=False)

    obs_builder = DefaultObservationBuilder(symmetric=True)
    action_parser = ContinuousActionParser()

    # Reward weights exactly as training applies them (yaml + live + annealing, gamma), so the
    # breakdown charts show what the trainer would pay for this match
    active_rewards = _reward_weights_for(blue_model_path)

    # Use isolated reward managers for each team to ensure zero potential cross-talk
    blue_reward_mgr = RewardManager(active_rewards)
    blue_reward_mgr.reset(arena)
    orange_reward_mgr = RewardManager(active_rewards)
    orange_reward_mgr.reset(arena)

    blue_bot = create_opponent_bot(blue_model_path, device=device) if blue_model_path else BaselineChaser()
    
    if orange_model_path == "same_as_blue":
        orange_bot = blue_bot
        match_type = "Self-Play"
    elif orange_model_path == "baseline" or orange_model_path is None or str(orange_model_path).lower() in ("heuristic", "baseline"):
        orange_bot = BaselineChaser()
        match_type = "Bot vs Baseline"
    else:
        orange_bot = create_opponent_bot(orange_model_path, device=device)
        match_type = "Checkpoint / Model Comparison"

    # Necto/Nexto emit raw per-tick controls; like training and TrueSkill, they must bypass
    # SenseiBot's jump sequencer or their jumps/flips get mangled and they never reach the ball.
    bot_mask = [isinstance(blue_bot, NectoNextoOpponentBot), isinstance(orange_bot, NectoNextoOpponentBot)]

    blue_traj_x, blue_traj_y = [], []
    orange_traj_x, orange_traj_y = [], []
    ball_traj_x, ball_traj_y = [], []

    blue_rewards: Dict[str, float] = {}
    orange_rewards: Dict[str, float] = {}
    blue_timeline: List[float] = []
    orange_timeline: List[float] = []
    goal_events: List[Dict[str, Any]] = []

    blue_cum_rew = 0.0
    orange_cum_rew = 0.0

    blue_goals = 0
    orange_goals = 0

    for step in range(max_steps):
        blue_traj_x.append(arena.cars[0].pos[0])
        blue_traj_y.append(arena.cars[0].pos[1])
        orange_traj_x.append(arena.cars[1].pos[0])
        orange_traj_y.append(arena.cars[1].pos[1])
        ball_traj_x.append(arena.ball.pos[0])
        ball_traj_y.append(arena.ball.pos[1])

        # Blue & Orange Actions
        act0 = blue_bot.get_action(arena.cars[0], arena)
        act1 = orange_bot.get_action(arena.cars[1], arena)

        actions = [act0, act1]
        goal, scoring_team = arena.step(actions, dt=1.0 / 15.0, bot_mask=bot_mask)

        # Calculate reward breakdowns with isolated managers
        r0, b0 = blue_reward_mgr.get_reward(arena.cars[0], arena, act0, goal, scoring_team)
        r1, b1 = orange_reward_mgr.get_reward(arena.cars[1], arena, act1, goal, scoring_team)
        for k, v in b0.items():
            blue_rewards[k] = blue_rewards.get(k, 0.0) + v
        for k, v in b1.items():
            orange_rewards[k] = orange_rewards.get(k, 0.0) + v

        blue_cum_rew += r0
        orange_cum_rew += r1
        blue_timeline.append(blue_cum_rew)
        orange_timeline.append(orange_cum_rew)

        if goal:
            if scoring_team == 0:
                blue_goals += 1
            else:
                orange_goals += 1
            goal_events.append({"step": step, "team": scoring_team})
            arena.reset(random_kickoff=True)
            blue_reward_mgr.reset(arena)
            orange_reward_mgr.reset(arena)

    blue_touches = arena.cars[0].ball_touches
    orange_touches = arena.cars[1].ball_touches

    # 1. Pitch Trajectory Plot
    pitch_fig, ax = plt.subplots(figsize=(8, 10), dpi=100)
    draw_rocket_league_pitch(ax)

    blue_label = f"Blue: {os.path.basename(blue_model_path)}" if blue_model_path else "Blue (Baseline)"
    if orange_model_path == "same_as_blue":
        orange_label = "Orange: Self-Play"
    elif orange_model_path == "baseline" or orange_model_path is None or str(orange_model_path).lower() in ("heuristic", "baseline"):
        orange_label = "Orange: Baseline Chaser"
    else:
        orange_label = f"Orange: {os.path.basename(orange_model_path)}"

    ax.plot(blue_traj_x, blue_traj_y, color="#63b3ed", label=blue_label, linewidth=2.0, alpha=0.85)
    ax.plot(orange_traj_x, orange_traj_y, color="#f6ad55", label=orange_label, linewidth=2.0, alpha=0.85)
    ax.plot(ball_traj_x, ball_traj_y, color="#f7fafc", label="Ball Trajectory", linewidth=1.5, linestyle=":", alpha=0.9)

    ax.scatter([arena.cars[0].pos[0]], [arena.cars[0].pos[1]], color="#3182ce", s=200, edgecolor="white", zorder=5, label="Blue End")
    ax.scatter([arena.cars[1].pos[0]], [arena.cars[1].pos[1]], color="#dd6b20", s=200, edgecolor="white", zorder=5, label="Orange End")
    ax.scatter([arena.ball.pos[0]], [arena.ball.pos[1]], color="#ffffff", s=150, edgecolor="#4a5568", zorder=6, label="Ball End")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=True, facecolor="#2d3748", edgecolor="#4a5568", labelcolor="white")
    plt.tight_layout()

    # 2. Reward Breakdown Bar Chart & Timeline
    reward_fig = render_reward_breakdown_plot(
        blue_rewards=blue_rewards,
        orange_rewards=orange_rewards,
        match_type=match_type,
        blue_timeline=blue_timeline,
        orange_timeline=orange_timeline,
        goal_events=goal_events,
    )

    stats = {
        "blue_goals": blue_goals,
        "orange_goals": orange_goals,
        "goals_blue": blue_goals,
        "goals_orange": orange_goals,
        "blue_touches": blue_touches,
        "orange_touches": orange_touches,
        "touches_blue": blue_touches,
        "touches_orange": orange_touches,
        "blue_total_reward": round(sum(blue_rewards.values()), 2),
        "orange_total_reward": round(sum(orange_rewards.values()), 2),
        "rewards_blue": round(sum(blue_rewards.values()), 2),
        "rewards_orange": round(sum(orange_rewards.values()), 2),
        "simulation_steps": max_steps,
        "total_steps": max_steps,
        "match_type": match_type,
        "blue_breakdown": blue_rewards,
        "orange_breakdown": orange_rewards,
        "goal_events": goal_events,
    }
    return pitch_fig, reward_fig, stats
