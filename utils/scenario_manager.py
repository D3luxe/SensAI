"""
Custom Scenario Manager & Visual Guide Generator for SenseiBot.
Handles scenario persistence, validation, default presets, 2D visual guide rendering,
and standalone scenario trajectory rollouts.
"""

from __future__ import annotations
import os
import json
import math
import copy
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from env.physics_engine import (
    ARENA_EXTENT_X, ARENA_EXTENT_Y, ARENA_HEIGHT_Z,
    GOAL_HALF_WIDTH, GOAL_HEIGHT, CAR_LENGTH, CAR_WIDTH, BALL_RADIUS,
    BoostPad
)

SCENARIOS_CONFIG_PATH = "config/custom_scenarios.json"

DEFAULT_CUSTOM_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "opposing_third_bouncing_ball",
        "name": "Opposing 1/3rd Bouncing Powershot / Dribble",
        "description": "Bot spawns in the opposing 1/3rd directly behind a slow forward-bouncing ball to practice power strikes and ground dribble pickups.",
        "enabled": True,
        "car": {
            "pos": [0.0, 1800.0, 17.0],
            "yaw": 90.0,       # Facing +Y toward orange goal (degrees)
            "pitch": 0.0,
            "roll": 0.0,
            "vel": [0.0, 800.0, 0.0],  # Moving forward towards ball
            "boost": 50.0
        },
        "ball": {
            "pos": [0.0, 2600.0, 220.0],  # In opposing third, bouncing
            "vel": [0.0, 250.0, -150.0]   # Moving slowly forward and downward
        },
        "opponent": {
            "mode": "goalie",            # "goalie", "shadow", "custom", "none"
            "pos": [0.0, 4800.0, 17.0],
            "yaw": -90.0,
            "pitch": 0.0,
            "roll": 0.0,
            "vel": [0.0, 0.0, 0.0],
            "boost": 60.0
        },
        "variance": {
            "pos_jitter": 80.0,
            "vel_jitter": 60.0,
            "mirror_symmetry": True
        }
    },
    {
        "id": "fast_breakaway_open_net",
        "name": "Midfield Fast Breakaway Sprint",
        "description": "Bot starts near midfield chasing a fast rolling ball toward an open net to practice boost feathering and supersonic power finishes.",
        "enabled": True,
        "car": {
            "pos": [-600.0, -200.0, 17.0],
            "yaw": 75.0,
            "pitch": 0.0,
            "roll": 0.0,
            "vel": [300.0, 1200.0, 0.0],
            "boost": 40.0
        },
        "ball": {
            "pos": [-300.0, 900.0, 93.15],
            "vel": [150.0, 800.0, 0.0]
        },
        "opponent": {
            "mode": "shadow",
            "pos": [400.0, 2400.0, 17.0],
            "yaw": 90.0,
            "pitch": 0.0,
            "roll": 0.0,
            "vel": [0.0, 600.0, 0.0],
            "boost": 33.3
        },
        "variance": {
            "pos_jitter": 100.0,
            "vel_jitter": 80.0,
            "mirror_symmetry": True
        }
    },
    {
        "id": "air_dribble_pop_setup",
        "name": "Air Dribble & Aerial Pop Setup",
        "description": "Ball popped up floating at 700uu height with the car positioned underneath for aerial takeoff and carry practice.",
        "enabled": True,
        "car": {
            "pos": [400.0, 0.0, 17.0],
            "yaw": 85.0,
            "pitch": 0.0,
            "roll": 0.0,
            "vel": [100.0, 900.0, 0.0],
            "boost": 80.0
        },
        "ball": {
            "pos": [450.0, 600.0, 750.0],
            "vel": [50.0, 300.0, 200.0]
        },
        "opponent": {
            "mode": "goalie",
            "pos": [0.0, 4800.0, 17.0],
            "yaw": -90.0,
            "pitch": 0.0,
            "roll": 0.0,
            "vel": [0.0, 0.0, 0.0],
            "boost": 70.0
        },
        "variance": {
            "pos_jitter": 60.0,
            "vel_jitter": 40.0,
            "mirror_symmetry": True
        }
    },
    {
        "id": "shadow_defense_1v1",
        "name": "Shadow Defense & 1v1 Retreat",
        "description": "Bot retreating toward its own defending net while shadowing an incoming opponent dribble.",
        "enabled": True,
        "car": {
            "pos": [200.0, -2600.0, 17.0],
            "yaw": -90.0,       # Facing defending goal -Y
            "pitch": 0.0,
            "roll": 0.0,
            "vel": [0.0, -900.0, 0.0],
            "boost": 45.0
        },
        "ball": {
            "pos": [-150.0, -1200.0, 93.15],
            "vel": [50.0, -750.0, 0.0]
        },
        "opponent": {
            "mode": "custom",
            "pos": [-150.0, -600.0, 17.0],
            "yaw": -90.0,
            "pitch": 0.0,
            "roll": 0.0,
            "vel": [0.0, -950.0, 0.0],
            "boost": 50.0
        },
        "variance": {
            "pos_jitter": 80.0,
            "vel_jitter": 60.0,
            "mirror_symmetry": True
        }
    }
]


class ScenarioManager:
    """
    Manages custom scenarios collection: persistence, retrieval, addition, deletion,
    and sampling.
    """
    _instance: Optional[ScenarioManager] = None

    def __init__(self, config_path: str = SCENARIOS_CONFIG_PATH):
        self.config_path = config_path
        self.scenarios: Dict[str, Dict[str, Any]] = {}
        self.load()

    @classmethod
    def get_instance(cls) -> ScenarioManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> Dict[str, Dict[str, Any]]:
        """Loads scenarios from JSON file or initializes defaults if not found."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.scenarios = {sc["id"]: sc for sc in data if "id" in sc}
                    elif isinstance(data, dict):
                        self.scenarios = data
                    return self.scenarios
            except Exception as e:
                print(f"[ScenarioManager] Error loading {self.config_path}: {e}. Initializing defaults.")
        
        # Initialize defaults
        self.scenarios = {sc["id"]: copy.deepcopy(sc) for sc in DEFAULT_CUSTOM_SCENARIOS}
        self.save()
        return self.scenarios

    def save(self) -> bool:
        """Persists scenarios dictionary to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(list(self.scenarios.values()), f, indent=2)
            return True
        except Exception as e:
            print(f"[ScenarioManager] Error saving {self.config_path}: {e}")
            return False

    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        return list(self.scenarios.values())

    def get_active_scenarios(self) -> List[Dict[str, Any]]:
        return [sc for sc in self.scenarios.values() if sc.get("enabled", True)]

    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        return self.scenarios.get(scenario_id)

    def save_scenario(self, scenario_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates and saves/updates a custom scenario."""
        sc_id = str(scenario_dict.get("id", "")).strip().lower().replace(" ", "_")
        if not sc_id:
            return False, "Scenario ID cannot be empty."

        scenario_dict["id"] = sc_id
        if not scenario_dict.get("name"):
            scenario_dict["name"] = sc_id.replace("_", " ").title()

        # Ensure required nested structures exist
        scenario_dict.setdefault("enabled", True)
        scenario_dict.setdefault("car", {
            "pos": [0.0, 0.0, 17.0],
            "yaw": 90.0,
            "pitch": 0.0,
            "roll": 0.0,
            "vel": [0.0, 0.0, 0.0],
            "boost": 50.0
        })
        scenario_dict.setdefault("ball", {
            "pos": [0.0, 1000.0, 93.15],
            "vel": [0.0, 0.0, 0.0]
        })
        scenario_dict.setdefault("opponent", {
            "mode": "goalie",
            "pos": [0.0, 4800.0, 17.0],
            "yaw": -90.0,
            "pitch": 0.0,
            "roll": 0.0,
            "vel": [0.0, 0.0, 0.0],
            "boost": 60.0
        })
        scenario_dict.setdefault("variance", {
            "pos_jitter": 50.0,
            "vel_jitter": 50.0,
            "mirror_symmetry": True
        })

        self.scenarios[sc_id] = scenario_dict
        if self.save():
            return True, f"Scenario '{scenario_dict['name']}' ({sc_id}) saved successfully!"
        return False, "Failed to write scenario to disk."

    def delete_scenario(self, scenario_id: str) -> Tuple[bool, str]:
        if scenario_id in self.scenarios:
            name = self.scenarios[scenario_id].get("name", scenario_id)
            del self.scenarios[scenario_id]
            self.save()
            return True, f"Scenario '{name}' deleted."
        return False, f"Scenario ID '{scenario_id}' not found."

    def reset_to_defaults(self):
        self.scenarios = {sc["id"]: copy.deepcopy(sc) for sc in DEFAULT_CUSTOM_SCENARIOS}
        self.save()


def draw_pitch_zones(ax):
    """Draws tactical zone overlays (Defensive Third, Midfield, Opposing Third)."""
    # Defensive Third (Y: -5120 to -1706)
    def_rect = patches.Rectangle(
        (-ARENA_EXTENT_X, -ARENA_EXTENT_Y),
        ARENA_EXTENT_X * 2,
        ARENA_EXTENT_Y * 2 / 3,
        facecolor="#1e3a8a",
        alpha=0.08,
        edgecolor=None
    )
    ax.add_patch(def_rect)
    ax.text(
        ARENA_EXTENT_X - 250, -3413, "DEFENSIVE 1/3RD (BLUE ZONE)",
        color="#60a5fa", fontsize=8, fontweight="bold", alpha=0.5,
        ha="right", va="center"
    )

    # Midfield Zone (Y: -1706 to +1706)
    mid_rect = patches.Rectangle(
        (-ARENA_EXTENT_X, -ARENA_EXTENT_Y / 3),
        ARENA_EXTENT_X * 2,
        ARENA_EXTENT_Y * 2 / 3,
        facecolor="#334155",
        alpha=0.08,
        edgecolor=None
    )
    ax.add_patch(mid_rect)
    ax.text(
        ARENA_EXTENT_X - 250, 0, "MIDFIELD TRANSITION ZONE",
        color="#94a3b8", fontsize=8, fontweight="bold", alpha=0.5,
        ha="right", va="center"
    )

    # Opposing Third (Y: +1706 to +5120)
    att_rect = patches.Rectangle(
        (-ARENA_EXTENT_X, ARENA_EXTENT_Y / 3),
        ARENA_EXTENT_X * 2,
        ARENA_EXTENT_Y * 2 / 3,
        facecolor="#7c2d12",
        alpha=0.08,
        edgecolor=None
    )
    ax.add_patch(att_rect)
    ax.text(
        ARENA_EXTENT_X - 250, 3413, "OPPOSING 1/3RD (ATTACK ZONE)",
        color="#f97316", fontsize=8, fontweight="bold", alpha=0.5,
        ha="right", va="center"
    )

    # Zone Dividers (Third lines)
    ax.plot([-ARENA_EXTENT_X, ARENA_EXTENT_X], [-ARENA_EXTENT_Y / 3, -ARENA_EXTENT_Y / 3],
            color="#475569", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.plot([-ARENA_EXTENT_X, ARENA_EXTENT_X], [ARENA_EXTENT_Y / 3, ARENA_EXTENT_Y / 3],
            color="#475569", linestyle=":", linewidth=1.0, alpha=0.7)


def draw_car_marker(
    ax,
    pos: List[float],
    yaw_deg: float,
    vel: List[float],
    boost: float = 50.0,
    color: str = "#38bdf8",
    label: str = "Bot (Car 0)"
):
    """
    Renders car footprint rectangle rotated by yaw, forward heading indicator,
    and velocity/momentum vector arrow with speed badge.
    """
    cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
    yaw_rad = math.radians(yaw_deg)
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
    speed = math.hypot(vx, vy)

    # Dimensions in 2D
    l, w = CAR_LENGTH, CAR_WIDTH

    # Rotate bounding box corners
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    
    corners_local = [
        (l / 2, -w / 2),
        (l / 2, w / 2),
        (-l / 2, w / 2),
        (-l / 2, -w / 2)
    ]
    corners_world = []
    for lx, ly in corners_local:
        wx = cx + (lx * cos_y - ly * sin_y)
        wy = cy + (lx * sin_y + ly * cos_y)
        corners_world.append((wx, wy))

    # Car body polygon
    car_polygon = patches.Polygon(
        corners_world,
        closed=True,
        facecolor=color,
        edgecolor="#ffffff",
        linewidth=1.8,
        alpha=0.85,
        zorder=6
    )
    ax.add_patch(car_polygon)

    # Front Bumper Accent Line (shows heading clearly)
    front_r = corners_world[0]
    front_l = corners_world[1]
    ax.plot([front_r[0], front_l[0]], [front_r[1], front_l[1]], color="#facc15", linewidth=3.5, zorder=7)

    # Forward Heading Nose Pointer
    nose_len = 180.0
    nose_x = cx + math.cos(yaw_rad) * (l / 2 + nose_len)
    nose_y = cy + math.sin(yaw_rad) * (l / 2 + nose_len)
    ax.annotate(
        "",
        xy=(nose_x, nose_y),
        xytext=(cx + math.cos(yaw_rad) * (l / 2), cy + math.sin(yaw_rad) * (l / 2)),
        arrowprops=dict(arrowstyle="-|>", color="#facc15", lw=2.0, mutation_scale=14),
        zorder=8
    )

    # Velocity / Momentum Vector (scaled proportional to speed)
    if speed > 10.0:
        arrow_len = min(800.0, speed * 0.35)
        vel_angle = math.atan2(vy, vx)
        arrow_end_x = cx + math.cos(vel_angle) * arrow_len
        arrow_end_y = cy + math.sin(vel_angle) * arrow_len
        
        ax.annotate(
            "",
            xy=(arrow_end_x, arrow_end_y),
            xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#22c55e",
                lw=2.5,
                mutation_scale=18,
                linestyle="solid"
            ),
            zorder=9
        )
        ax.text(
            arrow_end_x + 60, arrow_end_y + 60,
            f"Vel: {int(speed)} uu/s",
            color="#4ade80",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#22c55e", alpha=0.85),
            zorder=10
        )

    # Label & Boost Badge
    z_str = f" | z:{int(cz)}" if cz > 25.0 else ""
    ax.text(
        cx, cy - 220,
        f"{label}\n[Boost: {int(boost)}%{z_str}]",
        color="#ffffff",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", edgecolor=color, alpha=0.9),
        zorder=10
    )


def draw_ball_marker(
    ax,
    pos: List[float],
    vel: List[float],
    color: str = "#ef4444"
):
    """
    Renders ball circle, Z-altitude ring/shadow indicator, and velocity vector arrow.
    """
    bx, by, bz = float(pos[0]), float(pos[1]), float(pos[2])
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
    speed = math.hypot(vx, vy)

    # Ground Shadow if elevated (Z > 93.15)
    if bz > 120.0:
        shadow_r = max(40.0, BALL_RADIUS * (1.0 - min(0.6, (bz - 93.15) / 2000.0)))
        shadow_circle = patches.Circle(
            (bx, by), shadow_r,
            facecolor="#000000", edgecolor="#64748b", linestyle="--", linewidth=1.2, alpha=0.5, zorder=5
        )
        ax.add_patch(shadow_circle)
        ax.plot([bx, bx], [by, by], color="#94a3b8", linestyle=":", linewidth=1.2, zorder=5)

    # Ball Circle
    ball_circle = patches.Circle(
        (bx, by), BALL_RADIUS,
        facecolor=color, edgecolor="#ffffff", linewidth=2.0, alpha=0.95, zorder=7
    )
    ax.add_patch(ball_circle)

    # Ball Height Indicator Ring if elevated
    if bz > 120.0:
        ring = patches.Circle(
            (bx, by), BALL_RADIUS + 35,
            fill=False, edgecolor="#fbbf24", linestyle="-", linewidth=1.8, alpha=0.9, zorder=8
        )
        ax.add_patch(ring)

    # Ball Velocity / Momentum Vector
    if speed > 10.0:
        arrow_len = min(800.0, speed * 0.35)
        vel_angle = math.atan2(vy, vx)
        arrow_end_x = bx + math.cos(vel_angle) * arrow_len
        arrow_end_y = by + math.sin(vel_angle) * arrow_len
        
        ax.annotate(
            "",
            xy=(arrow_end_x, arrow_end_y),
            xytext=(bx, by),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#f97316",
                lw=2.5,
                mutation_scale=18
            ),
            zorder=9
        )
        z_vel_str = f", vz:{int(vz)}" if abs(vz) > 10 else ""
        ax.text(
            arrow_end_x + 60, arrow_end_y + 60,
            f"Ball Vel: {int(speed)} uu/s{z_vel_str}",
            color="#fb923c",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#f97316", alpha=0.85),
            zorder=10
        )

    # Altitude & Ball Label
    alt_label = f" (Z: {int(bz)} uu)" if bz > 100.0 else " (Ground)"
    ax.text(
        bx, by + 180,
        f"Ball{alt_label}",
        color="#fbbf24",
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", edgecolor="#fbbf24", alpha=0.9),
        zorder=10
    )


def render_scenario_visual_guide(
    scenario_data: Dict[str, Any],
    title: Optional[str] = None
) -> plt.Figure:
    """
    Renders a comprehensive 2D Rocket League pitch visual guide representing
    the custom scenario's vehicle positions, yaw orientations, momentum vectors,
    and ball altitude/velocity.
    """
    from utils.visualizer import draw_rocket_league_pitch

    fig, ax = plt.subplots(figsize=(8.0, 9.5), dpi=100)
    fig.patch.set_facecolor("#0f172a")

    # 1. Base Pitch & Goal Nets
    draw_rocket_league_pitch(ax)

    # 2. Zone Overlays
    draw_pitch_zones(ax)

    # 3. Car 0 (Bot / Blue)
    car_cfg = scenario_data.get("car", {})
    c_pos = car_cfg.get("pos", [0.0, 0.0, 17.0])
    c_yaw = car_cfg.get("yaw", 90.0)
    c_vel = car_cfg.get("vel", [0.0, 0.0, 0.0])
    c_boost = car_cfg.get("boost", 50.0)
    draw_car_marker(ax, c_pos, c_yaw, c_vel, boost=c_boost, color="#38bdf8", label="Bot (Blue)")

    # 4. Opponent Car (Orange)
    opp_cfg = scenario_data.get("opponent", {})
    opp_mode = opp_cfg.get("mode", "goalie")
    if opp_mode != "none":
        if opp_mode == "goalie":
            o_pos = [0.0, 4800.0, 17.0]
            o_yaw = -90.0
            o_vel = [0.0, 0.0, 0.0]
            o_boost = opp_cfg.get("boost", 60.0)
            draw_car_marker(ax, o_pos, o_yaw, o_vel, boost=o_boost, color="#f97316", label="Opponent (Goalie)")
        elif opp_mode == "shadow":
            o_pos = opp_cfg.get("pos", [200.0, 2600.0, 17.0])
            o_yaw = opp_cfg.get("yaw", 90.0)
            o_vel = opp_cfg.get("vel", [0.0, 600.0, 0.0])
            o_boost = opp_cfg.get("boost", 40.0)
            draw_car_marker(ax, o_pos, o_yaw, o_vel, boost=o_boost, color="#f97316", label="Opponent (Shadow)")
        else: # custom
            o_pos = opp_cfg.get("pos", [0.0, 3000.0, 17.0])
            o_yaw = opp_cfg.get("yaw", -90.0)
            o_vel = opp_cfg.get("vel", [0.0, 0.0, 0.0])
            o_boost = opp_cfg.get("boost", 50.0)
            draw_car_marker(ax, o_pos, o_yaw, o_vel, boost=o_boost, color="#f97316", label="Opponent (Custom)")

    # 5. Ball
    b_cfg = scenario_data.get("ball", {})
    b_pos = b_cfg.get("pos", [0.0, 1000.0, 93.15])
    b_vel = b_cfg.get("vel", [0.0, 0.0, 0.0])
    draw_ball_marker(ax, b_pos, b_vel)

    # 6. Title & Details
    sc_name = title or scenario_data.get("name", "Custom Scenario Visual Guide")
    ax.set_title(f"Visual Guide: {sc_name}", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    return fig


def simulate_custom_scenario(
    scenario_data: Dict[str, Any],
    model_path: Optional[str] = None,
    num_steps: int = 150,
    device: str = "cpu"
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Rolls out a 2-second simulation from the custom scenario in RocketSim,
    rendering trajectory trails and returning summary diagnostics.
    """
    from env.physics_engine import RocketSimArena
    from env.observations import DefaultObservationBuilder
    from env.actions import ContinuousActionParser
    from env.baseline_agent import BaselineChaser, create_opponent_bot

    arena = RocketSimArena(num_players=2, game_mode="1v1")
    arena.reset(random_kickoff=False)

    # Apply custom scenario state to RocketSim
    rsim_arena = arena._rsim_arena
    if rsim_arena:
        import RocketSim as rsim
        # 1. Ball
        b_pos = scenario_data["ball"]["pos"]
        b_vel = scenario_data["ball"]["vel"]
        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(float(b_pos[0]), float(b_pos[1]), float(b_pos[2]))
        bs.vel = rsim.Vec(float(b_vel[0]), float(b_vel[1]), float(b_vel[2]))
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        # 2. Car 0
        cars = rsim_arena.get_cars()
        if len(cars) > 0:
            c_pos = scenario_data["car"]["pos"]
            c_yaw = math.radians(scenario_data["car"]["yaw"])
            c_pitch = math.radians(scenario_data["car"].get("pitch", 0.0))
            c_roll = math.radians(scenario_data["car"].get("roll", 0.0))
            c_vel = scenario_data["car"]["vel"]
            c_boost = float(scenario_data["car"].get("boost", 50.0))

            cs = cars[0].get_state()
            cs.pos = rsim.Vec(float(c_pos[0]), float(c_pos[1]), float(c_pos[2]))
            cs.rot_mat = rsim.Angle(pitch=c_pitch, yaw=c_yaw, roll=c_roll).as_rot_mat()
            cs.vel = rsim.Vec(float(c_vel[0]), float(c_vel[1]), float(c_vel[2]))
            cs.ang_vel = rsim.Vec(0, 0, 0)
            cs.boost = c_boost
            cars[0].set_state(cs)

        # 3. Car 1 (Opponent)
        if len(cars) > 1:
            opp_cfg = scenario_data.get("opponent", {})
            opp_mode = opp_cfg.get("mode", "goalie")
            cs1 = cars[1].get_state()
            if opp_mode == "goalie":
                cs1.pos = rsim.Vec(0, 4800, 17)
                cs1.rot_mat = rsim.Angle(pitch=0, yaw=-math.pi / 2, roll=0).as_rot_mat()
                cs1.vel = rsim.Vec(0, 0, 0)
            elif opp_mode == "shadow":
                o_pos = opp_cfg.get("pos", [200.0, 2600.0, 17.0])
                cs1.pos = rsim.Vec(float(o_pos[0]), float(o_pos[1]), float(o_pos[2]))
                cs1.rot_mat = rsim.Angle(pitch=0, yaw=math.pi / 2, roll=0).as_rot_mat()
                cs1.vel = rsim.Vec(0, 600, 0)
            else: # custom
                o_pos = opp_cfg.get("pos", [0.0, 3000.0, 17.0])
                o_yaw = math.radians(opp_cfg.get("yaw", -90.0))
                o_vel = opp_cfg.get("vel", [0.0, 0.0, 0.0])
                cs1.pos = rsim.Vec(float(o_pos[0]), float(o_pos[1]), float(o_pos[2]))
                cs1.rot_mat = rsim.Angle(pitch=0, yaw=o_yaw, roll=0).as_rot_mat()
                cs1.vel = rsim.Vec(float(o_vel[0]), float(o_vel[1]), float(o_vel[2]))
            cs1.ang_vel = rsim.Vec(0, 0, 0)
            cs1.boost = float(opp_cfg.get("boost", 50.0))
            cars[1].set_state(cs1)

        arena._sync_from_rsim()

    bot = create_opponent_bot(model_path, device=device) if model_path else BaselineChaser()
    opp_bot = BaselineChaser()

    blue_x, blue_y = [], []
    ball_x, ball_y = [], []
    opp_x, opp_y = [], []
    touches = 0

    obs_builder = DefaultObservationBuilder(symmetric=True)
    action_parser = ContinuousActionParser()

    for step in range(num_steps):
        blue_x.append(float(arena.cars[0].pos[0]))
        blue_y.append(float(arena.cars[0].pos[1]))
        ball_x.append(float(arena.ball.pos[0]))
        ball_y.append(float(arena.ball.pos[1]))
        if len(arena.cars) > 1:
            opp_x.append(float(arena.cars[1].pos[0]))
            opp_y.append(float(arena.cars[1].pos[1]))

        # Actions
        obs_blue = obs_builder.build_obs(arena.cars[0], arena, 0)
        act_blue = bot.act(obs_blue)
        act_blue_parsed = action_parser.parse_actions(np.array([act_blue]))[0]

        if len(arena.cars) > 1:
            obs_opp = obs_builder.build_obs(arena.cars[1], arena, 1)
            act_opp = opp_bot.act(obs_opp)
            act_opp_parsed = action_parser.parse_actions(np.array([act_opp]))[0]
            actions = [act_blue_parsed, act_opp_parsed]
        else:
            actions = [act_blue_parsed]

        _, _, done, _ = arena.step(actions)
        if arena.cars[0].ball_touches > touches:
            touches = arena.cars[0].ball_touches
        if done:
            break

    # Render Simulation Trajectory Plot
    from utils.visualizer import draw_rocket_league_pitch
    fig, ax = plt.subplots(figsize=(8.0, 9.5), dpi=100)
    fig.patch.set_facecolor("#0f172a")
    draw_rocket_league_pitch(ax)
    draw_pitch_zones(ax)

    # Draw trajectories
    if len(blue_x) > 1:
        ax.plot(blue_x, blue_y, color="#38bdf8", linewidth=2.5, label="Bot Trajectory", zorder=8)
        ax.scatter([blue_x[0]], [blue_y[0]], color="#38bdf8", s=90, marker="o", edgecolors="white", zorder=9, label="Bot Start")
        ax.scatter([blue_x[-1]], [blue_y[-1]], color="#0284c7", s=110, marker="X", edgecolors="white", zorder=9, label="Bot End")

    if len(ball_x) > 1:
        ax.plot(ball_x, ball_y, color="#fb923c", linewidth=2.5, linestyle="--", label="Ball Trajectory", zorder=7)
        ax.scatter([ball_x[0]], [ball_y[0]], color="#ef4444", s=90, marker="o", edgecolors="white", zorder=9, label="Ball Start")
        ax.scatter([ball_x[-1]], [ball_y[-1]], color="#dc2626", s=110, marker="X", edgecolors="white", zorder=9, label="Ball End")

    if len(opp_x) > 1:
        ax.plot(opp_x, opp_y, color="#a855f7", linewidth=2.0, linestyle=":", label="Opponent Trajectory", zorder=6)

    ax.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=8.5)
    ax.set_title(f"Trajectory Rollout (2s): {scenario_data.get('name', 'Scenario')}", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()

    stats = {
        "steps_simulated": len(blue_x),
        "bot_touches": touches,
        "final_ball_pos": [round(float(arena.ball.pos[i]), 1) for i in range(3)],
        "final_bot_pos": [round(float(arena.cars[0].pos[i]), 1) for i in range(3)]
    }
    return fig, stats
