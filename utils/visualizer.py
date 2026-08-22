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
from env.observations import DefaultObservationBuilder
from env.actions import ContinuousActionParser, DiscreteActionParser
from env.rewards import RewardManager
from agent.models import ActorCritic
import json


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
    if model_path and os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            obs_dim = ckpt.get("obs_dim", 64)
            continuous = ckpt.get("continuous_actions", True)
            act_dim = ckpt.get("act_dim", 8 if continuous else 19)
            model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, continuous_actions=continuous).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model
        except Exception as e:
            print(f"[Visualizer] Could not load model {model_path}: {e}")
    return None


def render_reward_breakdown_plot(blue_rewards: Dict[str, float], orange_rewards: Optional[Dict[str, float]] = None, match_type: str = "Match Evaluation") -> plt.Figure:
    """
    Renders a clean dark-themed horizontal bar chart showing points earned by reward category.
    """
    CATEGORY_NAMES = {
        "goal": "Goals & Power Bonus",
        "ball_vel_toward_goal": "Ball Velocity to Goal",
        "aligned_shot": "Shots on Target",
        "possession": "Tactical Space Dominance",
        "dribble": "Roof Carry & Dribble",
        "behind_ball": "Goal-Side Rotation",
        "defensive_position": "Defensive Position",
        "face_ball": "Facing / Tracking Ball",
        "save": "Saves & Clears",
        "touch_ball": "Ball Touches",
        "speed_toward_ball": "Speed Toward Ball",
        "kickoff": "Kickoff Speed Rush",
        "demo_bump": "Bumps & Demolitions",
        "boost_steal": "Opponent Boost Steals",
        "small_pad": "Small Boost Pads (+12)",
        "big_pad": "Big Boost Orbs (+100)",
        "save_boost": "Boost Tank Retention",
        "aerial_height": "Aerial Jump / Height",
        "velocity": "General Driving Speed",
        "inactivity_penalty": "Inactivity & Standstill Penalty",
    }

    # Dynamically capture all active reward categories
    all_keys = list(CATEGORY_NAMES.keys())
    for k in list(blue_rewards.keys()) + (list(orange_rewards.keys()) if orange_rewards else []):
        if k not in all_keys:
            all_keys.append(k)

    labels = [CATEGORY_NAMES.get(k, k.replace("_", " ").title()) for k in all_keys]
    blue_vals = [blue_rewards.get(k, 0.0) for k in all_keys]
    orange_vals = [orange_rewards.get(k, 0.0) if orange_rewards else 0.0 for k in all_keys]

    fig_height = max(6.0, 0.42 * len(all_keys))
    fig, ax = plt.subplots(figsize=(9.5, fig_height), dpi=100)
    fig.patch.set_facecolor("#1a202c")
    ax.set_facecolor("#2d3748")

    y_pos = np.arange(len(all_keys))
    bar_height = 0.38 if orange_rewards else 0.55

    if orange_rewards:
        b_bars = ax.barh(y_pos + bar_height / 2, blue_vals, bar_height, label="Blue Team", color="#4299e1", alpha=0.9, edgecolor="#bee3f8")
        o_bars = ax.barh(y_pos - bar_height / 2, orange_vals, bar_height, label="Orange Team", color="#ed8936", alpha=0.9, edgecolor="#feebc8")
        ax.legend(loc="lower right", facecolor="#1a202c", edgecolor="#4a5568", labelcolor="white")
    else:
        b_bars = ax.barh(y_pos, blue_vals, bar_height, label="Blue Team", color="#4299e1", alpha=0.9, edgecolor="#bee3f8")

    ax.axvline(0, color="#718096", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="#e2e8f0", fontsize=10)
    ax.tick_params(colors="#cbd5e0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a5568")
    ax.spines["bottom"].set_color("#4a5568")
    ax.grid(axis="x", linestyle=":", alpha=0.4, color="#718096")
    ax.set_xlabel("Cumulative Reward Earned During Match", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_title(f"Match Reward Breakdown ({match_type})", color="white", fontsize=13, fontweight="bold", pad=12)

    # Value annotations on bars
    for bar in b_bars:
        w = bar.get_width()
        if abs(w) > 0.05:
            offset = 0.5 if w >= 0 else -0.5
            ha = "left" if w >= 0 else "right"
            ax.annotate(f"{w:+.1f}", (w + offset, bar.get_y() + bar.get_height() / 2),
                        color="#bee3f8", fontsize=8, va="center", ha=ha)

    plt.tight_layout()
    return fig


def simulate_match(
    blue_model_path: Optional[str] = None,
    orange_model_path: Optional[str] = "same_as_blue",
    max_steps: int = 400,
    device: str = "cpu"
) -> Tuple[plt.Figure, plt.Figure, Dict[str, Any]]:
    """
    Simulates a match between Blue and Orange agents and produces:
    1. 2D trajectory pitch plot.
    2. Detailed Reward Breakdown Bar Chart.
    3. Match statistics dict.
    """
    arena = RocketSimArena(num_players=2, game_mode="1v1")
    arena.reset(random_kickoff=True)

    obs_builder = DefaultObservationBuilder(symmetric=True)
    action_parser = ContinuousActionParser()

    # Load active reward weights for breakdown calculation
    active_rewards = {}
    if os.path.exists("config/live_config.json"):
        try:
            with open("config/live_config.json", "r") as f:
                cfg_data = json.load(f)
                active_rewards = cfg_data.get("rewards", {})
        except Exception:
            pass
    reward_mgr = RewardManager(active_rewards)
    reward_mgr.reset(arena)

    blue_model = load_model(blue_model_path, device=device)
    if orange_model_path == "same_as_blue":
        orange_model = blue_model
        match_type = "Self-Play"
    elif orange_model_path == "baseline" or orange_model_path is None:
        orange_model = None
        match_type = "Bot vs Baseline"
    else:
        orange_model = load_model(orange_model_path, device=device)
        match_type = "Checkpoint Comparison"

    blue_traj_x, blue_traj_y = [], []
    orange_traj_x, orange_traj_y = [], []
    ball_traj_x, ball_traj_y = [], []

    blue_rewards: Dict[str, float] = {}
    orange_rewards: Dict[str, float] = {}

    blue_goals = 0
    orange_goals = 0

    for step in range(max_steps):
        blue_traj_x.append(arena.cars[0].pos[0])
        blue_traj_y.append(arena.cars[0].pos[1])
        orange_traj_x.append(arena.cars[1].pos[0])
        orange_traj_y.append(arena.cars[1].pos[1])
        ball_traj_x.append(arena.ball.pos[0])
        ball_traj_y.append(arena.ball.pos[1])

        # Blue Action
        if blue_model is not None:
            obs0 = obs_builder.build_obs(arena.cars[0], arena)
            with torch.no_grad():
                obs_t = torch.tensor(obs0, dtype=torch.float32, device=device).unsqueeze(0)
                act0, _, _, _ = blue_model.get_action_and_value(obs_t, deterministic=True)
                if blue_model.continuous_actions:
                    act0 = act0.squeeze(0).cpu().numpy()
                else:
                    act0_idx = int(act0.squeeze().cpu().item())
                    act0 = DiscreteActionParser().parse_actions(act0_idx)
        else:
            diff = arena.ball.pos - arena.cars[0].pos
            fwd = arena.cars[0].get_forward_vector()
            steer = np.clip(diff[0] * fwd[1] - diff[1] * fwd[0], -1.0, 1.0)
            act0 = np.array([1.0, steer, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # Orange Action
        if orange_model is not None:
            obs1 = obs_builder.build_obs(arena.cars[1], arena)
            with torch.no_grad():
                obs_t1 = torch.tensor(obs1, dtype=torch.float32, device=device).unsqueeze(0)
                act1, _, _, _ = orange_model.get_action_and_value(obs_t1, deterministic=True)
                if orange_model.continuous_actions:
                    act1 = act1.squeeze(0).cpu().numpy()
                else:
                    act1_idx = int(act1.squeeze().cpu().item())
                    act1 = DiscreteActionParser().parse_actions(act1_idx)
        else:
            diff_o = arena.ball.pos - arena.cars[1].pos
            fwd_o = arena.cars[1].get_forward_vector()
            steer_o = np.clip(diff_o[0] * fwd_o[1] - diff_o[1] * fwd_o[0], -1.0, 1.0)
            act1 = np.array([1.0, steer_o, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        actions = [act0, act1]
        goal, scoring_team = arena.step(actions, dt=1.0 / 15.0)

        # Calculate reward breakdowns
        r0, b0 = reward_mgr.get_reward(arena.cars[0], arena, act0, goal, scoring_team)
        r1, b1 = reward_mgr.get_reward(arena.cars[1], arena, act1, goal, scoring_team)
        for k, v in b0.items():
            blue_rewards[k] = blue_rewards.get(k, 0.0) + v
        for k, v in b1.items():
            orange_rewards[k] = orange_rewards.get(k, 0.0) + v

        if goal:
            if scoring_team == 0:
                blue_goals += 1
            else:
                orange_goals += 1
            arena.reset(random_kickoff=True)
            reward_mgr.reset(arena)

    blue_touches = arena.cars[0].ball_touches
    orange_touches = arena.cars[1].ball_touches

    # 1. Pitch Trajectory Plot
    pitch_fig, ax = plt.subplots(figsize=(8, 10), dpi=100)
    draw_rocket_league_pitch(ax)

    blue_label = "Blue Bot" if blue_model else "Blue (Baseline)"
    orange_label = "Orange Bot (Self-Play)" if (orange_model is blue_model and blue_model is not None) else ("Orange Bot" if orange_model else "Orange (Baseline)")

    ax.plot(blue_traj_x, blue_traj_y, color="#63b3ed", label=blue_label, linewidth=2.0, alpha=0.85)
    ax.plot(orange_traj_x, orange_traj_y, color="#f6ad55", label=orange_label, linewidth=2.0, alpha=0.85)
    ax.plot(ball_traj_x, ball_traj_y, color="#f7fafc", label="Ball Trajectory", linewidth=1.5, linestyle=":", alpha=0.9)

    ax.scatter([arena.cars[0].pos[0]], [arena.cars[0].pos[1]], color="#3182ce", s=200, edgecolor="white", zorder=5, label="Blue End")
    ax.scatter([arena.cars[1].pos[0]], [arena.cars[1].pos[1]], color="#dd6b20", s=200, edgecolor="white", zorder=5, label="Orange End")
    ax.scatter([arena.ball.pos[0]], [arena.ball.pos[1]], color="#ffffff", s=150, edgecolor="#4a5568", zorder=6, label="Ball End")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=True, facecolor="#2d3748", edgecolor="#4a5568", labelcolor="white")
    plt.tight_layout()

    # 2. Reward Breakdown Bar Chart
    reward_fig = render_reward_breakdown_plot(
        blue_rewards=blue_rewards,
        orange_rewards=orange_rewards if (orange_model is not None) else None,
        match_type=match_type
    )

    stats = {
        "blue_goals": blue_goals,
        "orange_goals": orange_goals,
        "blue_touches": blue_touches,
        "orange_touches": orange_touches,
        "blue_total_reward": round(sum(blue_rewards.values()), 1),
        "orange_total_reward": round(sum(orange_rewards.values()), 1) if orange_model else 0.0,
        "simulation_steps": max_steps,
        "match_type": match_type,
        "blue_breakdown": blue_rewards,
        "orange_breakdown": orange_rewards
    }
    return pitch_fig, reward_fig, stats
