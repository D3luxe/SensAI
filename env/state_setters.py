"""
Modular State Setters for RocketSim Reinforcement Learning.
Generates authentic match scenarios (Aerials, Wall Plays, Goalie Saves, Replays, and Kickoffs)
to accelerate mechanics acquisition and diversify training distributions.
"""

from __future__ import annotations
import math
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import RocketSim as rsim

from utils.replay_parser import ReplayParser


ARENA_EXTENT_X = 4096.0
ARENA_EXTENT_Y = 5120.0
ARENA_HEIGHT_Z = 2044.0
GOAL_HEIGHT = 642.775
GOAL_HALF_WIDTH = 892.755


def rotation_to_rot_mat(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Computes exact 3x3 orthonormal basis (Forward, Right, Up)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    fwd = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
    up = np.array([-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr], dtype=np.float32)
    right = np.array([
        fwd[1] * up[2] - fwd[2] * up[1],
        fwd[2] * up[0] - fwd[0] * up[2],
        fwd[0] * up[1] - fwd[1] * up[0]
    ], dtype=np.float32)
    return np.vstack([fwd, right, up]).astype(np.float32)


def apply_teammate_stagger(base_x: float, base_y: float, teammate_idx: int, sign: float = 1.0) -> Tuple[float, float]:
    """
    Staggers secondary teammates laterally and trailing to prevent collision overlap in 2v2/3v3.
    teammate_idx: 0 for primary player on the team, 1 for second teammate, 2 for third teammate.
    sign: attacking direction (+1 for Blue attacking +Y, -1 for Orange attacking -Y).
    """
    if teammate_idx == 0:
        return base_x, base_y
    side_mult = 1.0 if (teammate_idx % 2 == 1) else -1.0
    side_offset = side_mult * 350.0
    trailing_offset = -sign * 400.0 * ((teammate_idx + 1) // 2)
    new_x = float(np.clip(base_x + side_offset, -ARENA_EXTENT_X + 250.0, ARENA_EXTENT_X - 250.0))
    new_y = float(np.clip(base_y + trailing_offset, -ARENA_EXTENT_Y + 250.0, ARENA_EXTENT_Y - 250.0))
    return new_x, new_y


class BaseStateSetter:
    def reset(self, rsim_arena: Any, num_players: int) -> None:
        raise NotImplementedError


class KickoffSetter(BaseStateSetter):
    """
    Standard competitive Rocket League kickoff configurations (Diagonal, Off-Center, Goal-Line).
    """
    KICKOFF_LOCATIONS = [
        # (x, y, yaw) for Team 0 (Standard Rocket League 45°, 135°, and 90° spawns)
        (-2048.0, -2560.0, math.pi / 4),      # Left Diagonal (45°)
        (2048.0, -2560.0, 3 * math.pi / 4),   # Right Diagonal (135°)
        (-256.0, -3840.0, math.pi / 2),        # Left Center (90°)
        (256.0, -3840.0, math.pi / 2),         # Right Center (90°)
        (0.0, -4608.0, math.pi / 2),           # Goal Line Straight (90°)
    ]

    def reset(self, rsim_arena: Any, num_players: int) -> None:
        # Ball at center
        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(0, 0, 93.15)
        bs.vel = rsim.Vec(0, 0, 0)
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        loc_idx = random.randint(0, len(self.KICKOFF_LOCATIONS) - 1)
        k_pos = self.KICKOFF_LOCATIONS[loc_idx]

        for i, car in enumerate(rsim_arena.get_cars()):
            cs = rsim.CarState()
            cs.vel = rsim.Vec(0, 0, 0)
            cs.ang_vel = rsim.Vec(0, 0, 0)
            cs.boost = 33.3

            team = i % 2
            if team == 0:
                cs.pos = rsim.Vec(k_pos[0], k_pos[1], 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=k_pos[2], roll=0.0).as_rot_mat()
            else:
                cs.pos = rsim.Vec(-k_pos[0], -k_pos[1], 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=k_pos[2] + math.pi, roll=0.0).as_rot_mat()
            car.set_state(cs)


class AerialScenarioSetter(BaseStateSetter):
    """
    Spawns high flying, floating, and rising aerial setups with physically guaranteed hang time (>= 1.8s),
    intercept lead vector car aiming, and 3-tier structured curriculum (stationary float, rising popup, dynamic cross).
    """
    def reset(self, rsim_arena: Any, num_players: int) -> None:
        target_team = random.choice([0, 1])
        sign = 1.0 if target_team == 0 else -1.0
        mode = random.choices(["stationary_float", "rising_popup", "dynamic_intercept"], weights=[0.25, 0.35, 0.40])[0]

        if mode == "stationary_float":
            # Mode 1: High floating ball with zero horizontal velocity; ideal for discovering initial jump+boost liftoff
            bx = random.uniform(-1200.0, 1200.0)
            by = sign * random.uniform(600.0, 1800.0)
            bz = random.uniform(1200.0, 1400.0)
            bvx = 0.0
            bvy = 0.0
            bvz = random.uniform(60.0, 120.0)
            base_cx = bx + random.uniform(-150.0, 150.0)
            base_cy = by - sign * random.uniform(700.0, 1000.0)
            approach_speed = random.uniform(750.0, 900.0)

        elif mode == "rising_popup":
            # Mode 2: Vertically launched rising popup; reaches apex in ~1.0s, training parabolic read and climb
            bx = random.uniform(-1200.0, 1200.0)
            by = sign * random.uniform(600.0, 2000.0)
            bz = random.uniform(500.0, 750.0)
            bvx = random.uniform(-150.0, 150.0)
            bvy = sign * random.uniform(50.0, 250.0)
            bvz = random.uniform(550.0, 800.0)
            base_cx = bx + random.uniform(-250.0, 250.0)
            base_cy = by - sign * random.uniform(700.0, 1100.0)
            approach_speed = random.uniform(800.0, 950.0)

        else:
            # Mode 3: Dynamic aerial cross / shot; analytical vz ensures >= 1.85s hang time above Z=250
            bx = random.uniform(-1400.0, 1400.0)
            by = sign * random.uniform(500.0, 2000.0)
            bz = random.uniform(900.0, 1300.0)
            bvx = random.uniform(-250.0, 250.0)
            bvy = sign * random.uniform(150.0, 450.0)
            t_hang = random.uniform(1.85, 2.50)
            bvz = (250.0 - bz + 0.5 * 650.0 * (t_hang ** 2)) / t_hang
            # Ensure apex stays safely below ceiling
            max_apex = bz + (max(0.0, bvz) ** 2) / 1300.0
            if max_apex > 1650.0:
                bvz = math.sqrt(max(0.0, (1650.0 - bz) * 1300.0))
            base_cx = bx + random.uniform(-250.0, 250.0)
            base_cy = by - sign * random.uniform(800.0, 1200.0)
            approach_speed = random.uniform(800.0, 950.0)

        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(bx, by, bz)
        bs.vel = rsim.Vec(bvx, bvy, bvz)
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        # Compute intercept lead point for car orientation
        dist_ground = math.hypot(bx - base_cx, by - base_cy)
        tau = min(1.4, max(0.8, dist_ground / approach_speed))
        x_lead = bx + bvx * tau
        y_lead = by + bvy * tau

        base_def_x = random.uniform(-400.0, 400.0)
        base_def_y = sign * random.uniform(4200.0, 4700.0)

        for i, car in enumerate(rsim_arena.get_cars()):
            cs = rsim.CarState()
            cs.boost = random.uniform(75.0, 100.0)
            team = i % 2
            teammate_idx = i // 2

            if team == target_team:
                cx, cy = apply_teammate_stagger(base_cx, base_cy, teammate_idx, sign)
                yaw = math.atan2(y_lead - cy, x_lead - cx)
                cs.pos = rsim.Vec(cx, cy, 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(math.cos(yaw) * approach_speed, math.sin(yaw) * approach_speed, 0.0)
            else:
                def_x, def_y = apply_teammate_stagger(base_def_x, base_def_y, teammate_idx, -sign)
                cs.pos = rsim.Vec(def_x, def_y, 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=-sign * math.pi / 2, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(0, 0, 0)

            cs.ang_vel = rsim.Vec(0, 0, 0)
            car.set_state(cs)


class WallPlaySetter(BaseStateSetter):
    """
    Spawns ball rolling along arena sidewalls or climbing the wall.
    Generates balanced distribution between ground-to-wall ramp approaches and on-wall tracking.
    """
    def reset(self, rsim_arena: Any, num_players: int) -> None:
        side = random.choice([-1.0, 1.0])  # Left or Right Wall
        target_team = random.choice([0, 1])
        sign = 1.0 if target_team == 0 else -1.0
        spawn_on_wall = random.random() < 0.50

        # Ball climbing sidewall (offset 100 uu hugs the wall just beyond ball radius 93.15)
        bx = side * (ARENA_EXTENT_X - 100.0)
        by = random.uniform(-2500.0, 2500.0)
        bz = random.uniform(300.0, 1200.0)

        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(bx, by, bz)
        # Gentle inward velocity allows wall play interaction without blasting prematurely into infield
        bs.vel = rsim.Vec(side * -40.0, sign * random.uniform(700.0, 1300.0), random.uniform(100.0, 400.0))
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        base_wall_cy = by - sign * random.uniform(500.0, 900.0)
        base_wall_cz = min(1100.0, max(250.0, bz - random.uniform(100.0, 250.0)))
        base_ground_cx = side * random.uniform(2500.0, 3600.0)
        base_ground_cy = by - sign * random.uniform(600.0, 1400.0)
        base_def_x = random.uniform(-400.0, 400.0)
        base_def_y = sign * random.uniform(4000.0, 4400.0)

        for i, car in enumerate(rsim_arena.get_cars()):
            cs = rsim.CarState()
            cs.boost = random.uniform(50.0, 100.0)
            team = i % 2
            teammate_idx = i // 2

            if team == target_team:
                if spawn_on_wall:
                    # Offset 25.0 uu engages wheel suspension for immediate grip (is_on_ground = True)
                    cx = side * (ARENA_EXTENT_X - 25.0)
                    cy = base_wall_cy - sign * (teammate_idx * 450.0)
                    cz = min(1100.0, max(250.0, base_wall_cz - teammate_idx * 150.0))
                    cs.pos = rsim.Vec(cx, cy, cz)
                    wall_yaw = (math.pi / 2) if sign > 0 else (-math.pi / 2)
                    wall_roll = (math.pi / 2) if (side * sign) > 0 else (-math.pi / 2)
                    wall_pitch = random.uniform(0.0, 0.20)
                    cs.rot_mat = rsim.Angle(pitch=wall_pitch, yaw=wall_yaw, roll=wall_roll).as_rot_mat()
                    car_speed = random.uniform(800.0, 1400.0)
                    cs.vel = rsim.Vec(0.0, sign * car_speed * math.cos(wall_pitch), car_speed * math.sin(wall_pitch))
                else:
                    cx, cy = apply_teammate_stagger(base_ground_cx, base_ground_cy, teammate_idx, sign)
                    cs.pos = rsim.Vec(cx, cy, 17.0)
                    yaw = math.atan2(by - cy, bx - cx)
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(math.cos(yaw) * 1000.0, math.sin(yaw) * 1000.0, 0.0)
            else:
                def_x, def_y = apply_teammate_stagger(base_def_x, base_def_y, teammate_idx, -sign)
                cs.pos = rsim.Vec(def_x, def_y, 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=-sign * math.pi / 2, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(0.0, 0.0, 0.0)

            cs.ang_vel = rsim.Vec(0, 0, 0)
            car.set_state(cs)


class GoalieSaveSetter(BaseStateSetter):
    """
    Spawns high threat shots moving directly on target towards the defending net.
    """
    def reset(self, rsim_arena: Any, num_players: int) -> None:
        defending_team = random.choice([0, 1])
        defend_goal_y = -ARENA_EXTENT_Y if defending_team == 0 else ARENA_EXTENT_Y
        sign = 1.0 if defending_team == 0 else -1.0

        # Ball moving rapidly toward net
        target_x = random.uniform(-750, 750)
        target_z = random.uniform(100, 550)

        start_y = defend_goal_y + sign * random.uniform(2500, 4000)
        start_x = random.uniform(-1500, 1500)
        start_z = random.uniform(200, 700)

        # Calculate shot velocity
        flight_time = random.uniform(1.2, 2.5)
        vx = (target_x - start_x) / flight_time
        vy = (defend_goal_y - start_y) / flight_time
        vz = (target_z - start_z - 0.5 * (-650.0) * (flight_time ** 2)) / flight_time

        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(start_x, start_y, start_z)
        bs.vel = rsim.Vec(vx, vy, vz)
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        base_def_x = random.uniform(-400.0, 400.0)
        base_def_y = defend_goal_y + sign * random.uniform(400.0, 1500.0)
        is_shadow = random.random() < 0.50
        def_speed = random.uniform(300.0, 800.0)
        if is_shadow:
            yaw_def = -sign * math.pi / 2 + random.uniform(-0.25, 0.25)
        else:
            yaw_def = sign * math.pi / 2 + random.uniform(-0.25, 0.25)

        base_shoot_x = start_x
        base_shoot_y = start_y + sign * 800.0
        shoot_yaw = math.atan2(defend_goal_y - start_y, target_x - start_x)

        for i, car in enumerate(rsim_arena.get_cars()):
            cs = rsim.CarState()
            cs.boost = random.uniform(40.0, 100.0)
            team = i % 2
            teammate_idx = i // 2

            if team == defending_team:
                def_x, def_y = apply_teammate_stagger(base_def_x, base_def_y, teammate_idx, sign)
                cs.pos = rsim.Vec(def_x, def_y, 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw_def, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(math.cos(yaw_def) * def_speed, math.sin(yaw_def) * def_speed, 0.0)
            else:
                shoot_x, shoot_y = apply_teammate_stagger(base_shoot_x, base_shoot_y, teammate_idx, -sign)
                cs.pos = rsim.Vec(shoot_x, shoot_y, 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=shoot_yaw, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(math.cos(shoot_yaw) * 1200.0, math.sin(shoot_yaw) * 1200.0, 0.0)

            cs.ang_vel = rsim.Vec(0, 0, 0)
            car.set_state(cs)


class ReplayStateSetter(BaseStateSetter):
    """
    Samples authentic match states from the ingested replay pool.
    """
    def __init__(self, parser: Optional[ReplayParser] = None):
        self.parser = parser or ReplayParser()

    def reset(self, rsim_arena: Any, num_players: int) -> bool:
        sample = self.parser.sample_state(num_cars=num_players)
        if sample is None:
            return False

        # Set ball state with arena boundary safety clamp (avoiding net depth post-goal)
        bp = sample["ball_pos"]
        bx = float(np.clip(bp[0], -4000.0, 4000.0))
        by = float(np.clip(bp[1], -5050.0, 5050.0))
        bz = float(np.clip(bp[2], 93.0, 2000.0))
        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(bx, by, bz)
        bs.vel = rsim.Vec(*sample["ball_vel"])
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        # Set car states
        cars = rsim_arena.get_cars()
        for i in range(min(len(cars), num_players)):
            cs = rsim.CarState()
            cs.pos = rsim.Vec(*sample["car_pos"][i])
            cs.vel = rsim.Vec(*sample["car_vel"][i])
            rot = sample["car_rot"][i]
            cs.rot_mat = rsim.Angle(pitch=float(rot[0]), yaw=float(rot[1]), roll=float(rot[2])).as_rot_mat()
            cs.boost = float(sample["car_boost"][i])
            cs.ang_vel = rsim.Vec(0, 0, 0)
            cars[i].set_state(cs)

        return True


class TurnaroundRecoverySetter(BaseStateSetter):
    """
    Spawns turnaround recoveries, wrong-side ball scenarios, reverse half-flip turnarounds,
    and mid-flip inverted recovery curriculum states.
    Forces the agent to master half-flips, powerslide 180° hairpin cuts, tap-braking, and peel-aways.
    """
    def reset(self, rsim_arena: Any, num_players: int) -> None:
        target_team = random.choice([0, 1])
        sign = 1.0 if target_team == 0 else -1.0
        mode = random.choice(["turnaround_sprint", "wrong_side_dribble", "reverse_halfflip", "midflip_inverted", "downfield_speedflip_sprint", "rear_quarter_scramble"])

        if mode == "downfield_speedflip_sprint":
            # Scenario E: Downfield breakaway / speed-flip sprint
            bx = random.uniform(-1000.0, 1000.0)
            by = sign * random.uniform(1000.0, 2500.0)
            bz = 93.15

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            bs.vel = rsim.Vec(random.uniform(-100.0, 100.0), sign * random.uniform(600.0, 1300.0), 0.0)
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            base_cx = bx + random.uniform(-200.0, 200.0)
            base_cy = by - sign * random.uniform(2000.0, 3200.0)
            base_opp_x = random.uniform(-800.0, 800.0)
            base_opp_y = by + sign * random.uniform(1000.0, 1800.0)

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = rsim.CarState()
                cs.boost = random.uniform(20.0, 50.0)
                team = i % 2
                teammate_idx = i // 2

                if team == target_team:
                    act_cx, act_cy = apply_teammate_stagger(base_cx, base_cy, teammate_idx, sign)
                    cs.pos = rsim.Vec(act_cx, act_cy, 17.0)
                    yaw = sign * math.pi / 2 + random.uniform(-0.15, 0.15)
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    speed = random.uniform(500.0, 900.0)
                    cs.vel = rsim.Vec(math.cos(yaw) * speed, math.sin(yaw) * speed, 0.0)
                else:
                    opp_x, opp_y = apply_teammate_stagger(base_opp_x, base_opp_y, teammate_idx, -sign)
                    cs.pos = rsim.Vec(opp_x, opp_y, 17.0)
                    yaw = -sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0.0, -sign * 700.0, 0.0)

                cs.ang_vel = rsim.Vec(0, 0, 0)
                car.set_state(cs)

        elif mode == "wrong_side_dribble":
            # Scenario A: Bot is directly behind the ball facing its own goal
            bx = random.uniform(-1000.0, 1000.0)
            by = sign * random.uniform(0.0, 2000.0)
            bz = 93.15

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            bs.vel = rsim.Vec(random.uniform(-50.0, 50.0), -sign * random.uniform(100.0, 400.0), 0.0)
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            base_cx = bx + random.uniform(-40.0, 40.0)
            base_cy = by + sign * random.uniform(150.0, 300.0)
            base_opp_x = random.uniform(-800.0, 800.0)
            base_opp_y = by - sign * random.uniform(1200.0, 2000.0)

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = rsim.CarState()
                cs.boost = random.uniform(20.0, 70.0)
                team = i % 2
                teammate_idx = i // 2

                if team == target_team:
                    act_cx, act_cy = apply_teammate_stagger(base_cx, base_cy, teammate_idx, -sign)
                    cs.pos = rsim.Vec(act_cx, act_cy, 17.0)
                    yaw = -sign * math.pi / 2 + random.uniform(-0.1, 0.1)
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0.0, -sign * random.uniform(300.0, 700.0), 0.0)
                else:
                    opp_x, opp_y = apply_teammate_stagger(base_opp_x, base_opp_y, teammate_idx, sign)
                    cs.pos = rsim.Vec(opp_x, opp_y, 17.0)
                    yaw = sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0.0, sign * 500.0, 0.0)

                cs.ang_vel = rsim.Vec(0, 0, 0)
                car.set_state(cs)

        elif mode == "reverse_halfflip":
            # Scenario C: Car is stopped or reversing facing forward, ball is cleared over its head toward own net
            bx = random.uniform(-1200.0, 1200.0)
            by = -sign * random.uniform(1000.0, 2800.0)
            bz = random.uniform(93.15, 350.0)

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            bs.vel = rsim.Vec(random.uniform(-200.0, 200.0), -sign * random.uniform(800.0, 1500.0), random.uniform(0.0, 300.0))
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            base_cx = random.uniform(-800.0, 800.0)
            base_cy = sign * random.uniform(200.0, 1400.0)
            base_opp_x = random.uniform(-600.0, 600.0)
            base_opp_y = sign * random.uniform(1600.0, 2600.0)

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = rsim.CarState()
                cs.boost = random.uniform(35.0, 80.0)
                team = i % 2
                teammate_idx = i // 2

                if team == target_team:
                    act_cx, act_cy = apply_teammate_stagger(base_cx, base_cy, teammate_idx, sign)
                    cs.pos = rsim.Vec(act_cx, act_cy, 17.0)
                    yaw = sign * math.pi / 2 + random.uniform(-0.2, 0.2)
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    rev_speed = random.uniform(-400.0, 0.0)
                    cs.vel = rsim.Vec(0.0, sign * rev_speed, 0.0)
                else:
                    opp_x, opp_y = apply_teammate_stagger(base_opp_x, base_opp_y, teammate_idx, -sign)
                    cs.pos = rsim.Vec(opp_x, opp_y, 17.0)
                    yaw = -sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0.0, -sign * 900.0, 0.0)

                cs.ang_vel = rsim.Vec(0, 0, 0)
                car.set_state(cs)

        elif mode == "midflip_inverted":
            # Scenario D: Curriculum reset - car spawned mid-air upside down with backward momentum
            bx = random.uniform(-1000.0, 1000.0)
            by = -sign * random.uniform(1500.0, 3200.0)
            bz = 93.15

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            bs.vel = rsim.Vec(0.0, -sign * random.uniform(600.0, 1200.0), 0.0)
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            base_cx = random.uniform(-600.0, 600.0)
            base_cy = sign * random.uniform(400.0, 1200.0)
            base_cz = random.uniform(120.0, 220.0)
            base_opp_y = sign * 2500.0

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = rsim.CarState()
                cs.boost = random.uniform(40.0, 80.0)
                team = i % 2
                teammate_idx = i // 2

                if team == target_team:
                    act_cx, act_cy = apply_teammate_stagger(base_cx, base_cy, teammate_idx, sign)
                    cs.pos = rsim.Vec(act_cx, act_cy, base_cz)
                    yaw = sign * math.pi / 2
                    roll_val = math.pi if random.random() < 0.5 else -math.pi
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=roll_val).as_rot_mat()
                    cs.vel = rsim.Vec(0.0, -sign * random.uniform(900.0, 1300.0), random.uniform(-50.0, 50.0))
                else:
                    opp_x, opp_y = apply_teammate_stagger(0.0, base_opp_y, teammate_idx, -sign)
                    cs.pos = rsim.Vec(opp_x, opp_y, 17.0)
                    yaw = -sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0.0, -sign * 700.0, 0.0)

                cs.ang_vel = rsim.Vec(0, 0, 0)
                car.set_state(cs)

        elif mode == "rear_quarter_scramble":
            # Scenario F: Close-proximity rear quarter-panel / blindspot scramble
            cx = random.uniform(-1500.0, 1500.0)
            cy = sign * random.uniform(0.0, 2000.0)
            yaw = sign * math.pi / 2 + random.uniform(-0.25, 0.25)

            side_sign = random.choice([-1.0, 1.0])
            lat_off = side_sign * random.uniform(35.0, 75.0)
            long_off = -sign * random.uniform(50.0, 130.0)
            bx = cx + lat_off
            by = cy + long_off
            bz = random.uniform(93.15, 140.0)

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            bs.vel = rsim.Vec(random.uniform(-40.0, 40.0), random.uniform(-40.0, 40.0), random.uniform(0.0, 100.0))
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            base_opp_x = random.uniform(-1000.0, 1000.0)
            base_opp_y = by + sign * random.uniform(1500.0, 2800.0)

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = rsim.CarState()
                cs.boost = random.uniform(0.0, 40.0)
                team = i % 2
                teammate_idx = i // 2

                if team == target_team:
                    act_cx, act_cy = apply_teammate_stagger(cx, cy, teammate_idx, sign)
                    cs.pos = rsim.Vec(act_cx, act_cy, 17.0)
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    speed = random.uniform(0.0, 200.0)
                    cs.vel = rsim.Vec(math.cos(yaw) * speed, math.sin(yaw) * speed, 0.0)
                else:
                    opp_x, opp_y = apply_teammate_stagger(base_opp_x, base_opp_y, teammate_idx, -sign)
                    cs.pos = rsim.Vec(opp_x, opp_y, 17.0)
                    yaw_opp = -sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw_opp, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0.0, -sign * random.uniform(400.0, 800.0), 0.0)

                cs.ang_vel = rsim.Vec(0, 0, 0)
                car.set_state(cs)

        else:
            # Scenario B: Ball grounded behind midfield line, car sprinting downfield away from ball
            bx = random.uniform(-1200.0, 1200.0)
            by = -sign * random.uniform(800.0, 2400.0)
            bz = 93.15

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            bs.vel = rsim.Vec(random.uniform(-100.0, 100.0), -sign * random.uniform(0.0, 300.0), 0.0)
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            base_cx = random.uniform(-1000.0, 1000.0)
            base_cy = sign * random.uniform(200.0, 1800.0)
            base_opp_x = random.uniform(-800.0, 800.0)
            base_opp_y = by - sign * random.uniform(400.0, 1000.0)

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = rsim.CarState()
                cs.boost = random.uniform(30.0, 80.0)
                team = i % 2
                teammate_idx = i // 2

                if team == target_team:
                    act_cx, act_cy = apply_teammate_stagger(base_cx, base_cy, teammate_idx, sign)
                    cs.pos = rsim.Vec(act_cx, act_cy, 17.0)
                    yaw = sign * math.pi / 2 + random.uniform(-0.3, 0.3)
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    speed = random.uniform(1200.0, 1800.0)
                    cs.vel = rsim.Vec(math.cos(yaw) * speed, math.sin(yaw) * speed, 0.0)
                else:
                    opp_x, opp_y = apply_teammate_stagger(base_opp_x, base_opp_y, teammate_idx, sign)
                    cs.pos = rsim.Vec(opp_x, opp_y, 17.0)
                    yaw = sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0.0, sign * 600.0, 0.0)

                cs.ang_vel = rsim.Vec(0, 0, 0)
                car.set_state(cs)


class CustomScenarioSetter(BaseStateSetter):
    """
    Samples user-defined custom scenarios configured in the Scenario Generator.
    Supports random positional/velocity jitter and left-right pitch mirroring.
    """
    def __init__(self, scenario_manager: Optional[Any] = None):
        if scenario_manager is None:
            from utils.scenario_manager import ScenarioManager
            self.manager = ScenarioManager.get_instance()
        else:
            self.manager = scenario_manager

    def reset(self, rsim_arena: Any, num_players: int) -> bool:
        active_scenarios = self.manager.get_active_scenarios()
        if not active_scenarios:
            return False

        scenario = random.choice(active_scenarios)
        variance = scenario.get("variance", {})
        pos_jit = float(variance.get("pos_jitter", 0.0))
        vel_jit = float(variance.get("vel_jitter", 0.0))
        mirror = bool(variance.get("mirror_symmetry", True)) and (random.random() < 0.5)
        mirror_sign = -1.0 if mirror else 1.0

        # 1. Reset Ball State
        b_cfg = scenario.get("ball", {})
        bx = float(b_cfg.get("pos", [0, 0, 93.15])[0]) * mirror_sign + random.uniform(-pos_jit, pos_jit)
        by = float(b_cfg.get("pos", [0, 0, 93.15])[1]) + random.uniform(-pos_jit, pos_jit)
        bz = max(93.15, min(ARENA_HEIGHT_Z - 100.0, float(b_cfg.get("pos", [0, 0, 93.15])[2]) + random.uniform(-pos_jit * 0.5, pos_jit * 0.5)))

        bvx = float(b_cfg.get("vel", [0, 0, 0])[0]) * mirror_sign + random.uniform(-vel_jit, vel_jit)
        bvy = float(b_cfg.get("vel", [0, 0, 0])[1]) + random.uniform(-vel_jit, vel_jit)
        bvz = float(b_cfg.get("vel", [0, 0, 0])[2]) + random.uniform(-vel_jit * 0.5, vel_jit * 0.5)

        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(
            max(-ARENA_EXTENT_X + 150.0, min(ARENA_EXTENT_X - 150.0, bx)),
            max(-ARENA_EXTENT_Y + 150.0, min(ARENA_EXTENT_Y - 150.0, by)),
            bz
        )
        bs.vel = rsim.Vec(bvx, bvy, bvz)
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        # 2. Reset Cars
        cars = rsim_arena.get_cars()
        if len(cars) > 0:
            c_cfg = scenario.get("car", {})
            cx = float(c_cfg.get("pos", [0, 0, 17])[0]) * mirror_sign + random.uniform(-pos_jit, pos_jit)
            cy = float(c_cfg.get("pos", [0, 0, 17])[1]) + random.uniform(-pos_jit, pos_jit)
            cz = max(17.0, min(ARENA_HEIGHT_Z - 100.0, float(c_cfg.get("pos", [0, 0, 17])[2])))

            yaw_deg = float(c_cfg.get("yaw", 90.0))
            if mirror:
                yaw_deg = 180.0 - yaw_deg
            yaw_rad = math.radians(yaw_deg)
            pitch_rad = math.radians(float(c_cfg.get("pitch", 0.0)))
            roll_rad = math.radians(float(c_cfg.get("roll", 0.0)))

            cvx = float(c_cfg.get("vel", [0, 0, 0])[0]) * mirror_sign + random.uniform(-vel_jit, vel_jit)
            cvy = float(c_cfg.get("vel", [0, 0, 0])[1]) + random.uniform(-vel_jit, vel_jit)
            cvz = float(c_cfg.get("vel", [0, 0, 0])[2])

            cs = rsim.CarState()
            cs.pos = rsim.Vec(
                max(-ARENA_EXTENT_X + 150.0, min(ARENA_EXTENT_X - 150.0, cx)),
                max(-ARENA_EXTENT_Y + 150.0, min(ARENA_EXTENT_Y - 150.0, cy)),
                cz
            )
            cs.rot_mat = rsim.Angle(pitch=pitch_rad, yaw=yaw_rad, roll=roll_rad).as_rot_mat()
            cs.vel = rsim.Vec(cvx, cvy, cvz)
            cs.ang_vel = rsim.Vec(0, 0, 0)
            cs.boost = float(c_cfg.get("boost", 50.0))
            cars[0].set_state(cs)

        if len(cars) > 1:
            opp_cfg = scenario.get("opponent", {})
            opp_mode = opp_cfg.get("mode", "goalie")
            cs1 = rsim.CarState()
            if opp_mode == "goalie":
                cs1.pos = rsim.Vec(random.uniform(-400, 400), 4800.0, 17.0)
                cs1.rot_mat = rsim.Angle(pitch=0.0, yaw=-math.pi / 2, roll=0.0).as_rot_mat()
                cs1.vel = rsim.Vec(0, 0, 0)
            elif opp_mode == "shadow":
                ox = float(opp_cfg.get("pos", [200, 2600, 17])[0]) * mirror_sign + random.uniform(-pos_jit, pos_jit)
                oy = float(opp_cfg.get("pos", [200, 2600, 17])[1]) + random.uniform(-pos_jit, pos_jit)
                cs1.pos = rsim.Vec(ox, oy, 17.0)
                cs1.rot_mat = rsim.Angle(pitch=0.0, yaw=math.pi / 2, roll=0.0).as_rot_mat()
                cs1.vel = rsim.Vec(0, 600, 0)
            elif opp_mode == "custom":
                ox = float(opp_cfg.get("pos", [0, 3000, 17])[0]) * mirror_sign + random.uniform(-pos_jit, pos_jit)
                oy = float(opp_cfg.get("pos", [0, 3000, 17])[1]) + random.uniform(-pos_jit, pos_jit)
                oz = max(17.0, float(opp_cfg.get("pos", [0, 3000, 17])[2]))
                o_yaw_deg = float(opp_cfg.get("yaw", -90.0))
                if mirror:
                    o_yaw_deg = 180.0 - o_yaw_deg
                cs1.pos = rsim.Vec(ox, oy, oz)
                cs1.rot_mat = rsim.Angle(pitch=0.0, yaw=math.radians(o_yaw_deg), roll=0.0).as_rot_mat()
                cs1.vel = rsim.Vec(
                    float(opp_cfg.get("vel", [0, 0, 0])[0]) * mirror_sign,
                    float(opp_cfg.get("vel", [0, 0, 0])[1]),
                    float(opp_cfg.get("vel", [0, 0, 0])[2])
                )
            else: # none / fallback
                cs1.pos = rsim.Vec(0.0, 4800.0, 17.0)
                cs1.rot_mat = rsim.Angle(pitch=0.0, yaw=-math.pi / 2, roll=0.0).as_rot_mat()
                cs1.vel = rsim.Vec(0, 0, 0)

            cs1.ang_vel = rsim.Vec(0, 0, 0)
            cs1.boost = float(opp_cfg.get("boost", 50.0))
            cars[1].set_state(cs1)

        return True


class WallBounceReboundSetter(BaseStateSetter):
    """
    Spawns hard clears, wall passes, and backboard/ceiling bangs with physics-first inversion:
    analytically calculates velocity so the ball cleanly strikes the sidewall, backboard (above crossbar),
    or ceiling in 0.7 - 1.3s before rebounding into the field for read anticipation.
    """
    def reset(self, rsim_arena: Any, num_players: int) -> None:
        target_team = random.choice([0, 1])
        sign = 1.0 if target_team == 0 else -1.0
        bounce_type = random.choice(["sidewall", "backboard", "ceiling"])

        side = random.choice([-1.0, 1.0])
        if bounce_type == "sidewall":
            # Target impact point on sidewall (flush with ball radius 93.15)
            target_x = side * (ARENA_EXTENT_X - 93.15)
            target_y = sign * random.uniform(0.0, 3000.0)
            target_z = random.uniform(500.0, 1200.0)

            # Start infield within reach of the wall
            start_x = side * random.uniform(1400.0, 2500.0)
            start_y = target_y - sign * random.uniform(400.0, 1000.0)
            start_z = random.uniform(200.0, 500.0)

            flight_time = random.uniform(0.7, 1.2)
            vx = (target_x - start_x) / flight_time
            vy = (target_y - start_y) / flight_time
            vz = (target_z - start_z + 0.5 * 650.0 * (flight_time ** 2)) / flight_time

            # Car positioned midfield tracking toward the post-rebound trajectory
            rebound_x = target_x - side * 800.0
            rebound_y = target_y + vy * 0.5
            base_cx = side * random.uniform(800.0, 1800.0)
            base_cy = start_y - sign * random.uniform(400.0, 900.0)
            yaw = math.atan2(rebound_y - base_cy, rebound_x - base_cx)
            car_speed = random.uniform(700.0, 1100.0)

        elif bounce_type == "backboard":
            # Target impact strictly on backboard above the crossbar (Z > 643)
            target_x = random.uniform(-1400.0, 1400.0)
            target_y = sign * (ARENA_EXTENT_Y - 93.15)
            target_z = random.uniform(800.0, 1500.0)

            start_x = target_x + random.uniform(-600.0, 600.0)
            start_y = sign * random.uniform(2000.0, 3400.0)
            start_z = random.uniform(200.0, 600.0)

            flight_time = random.uniform(0.8, 1.3)
            vx = (target_x - start_x) / flight_time
            vy = (target_y - start_y) / flight_time
            vz = (target_z - start_z + 0.5 * 650.0 * (flight_time ** 2)) / flight_time

            # Car positioned downfield tracking toward the rebound coming off the backboard
            rebound_x = target_x
            rebound_y = sign * 3800.0
            base_cx = target_x + random.uniform(-400.0, 400.0)
            base_cy = start_y - sign * random.uniform(600.0, 1200.0)
            yaw = math.atan2(rebound_y - base_cy, rebound_x - base_cx)
            car_speed = random.uniform(700.0, 1100.0)

        else: # "ceiling"
            # Target impact on arena ceiling (Z = 2044 - 93.15 = 1950.85)
            target_x = random.uniform(-1200.0, 1200.0)
            target_y = sign * random.uniform(800.0, 2600.0)
            target_z = ARENA_HEIGHT_Z - 93.15

            start_x = target_x + random.uniform(-400.0, 400.0)
            start_y = target_y - sign * random.uniform(400.0, 800.0)
            start_z = random.uniform(300.0, 800.0)

            flight_time = random.uniform(0.7, 1.1)
            vx = (target_x - start_x) / flight_time
            vy = (target_y - start_y) / flight_time
            vz = (target_z - start_z + 0.5 * 650.0 * (flight_time ** 2)) / flight_time

            rebound_x = target_x + vx * 0.4
            rebound_y = target_y + vy * 0.4
            base_cx = target_x + random.uniform(-300.0, 300.0)
            base_cy = start_y - sign * random.uniform(600.0, 1200.0)
            yaw = math.atan2(rebound_y - base_cy, rebound_x - base_cx)
            car_speed = random.uniform(600.0, 1000.0)

        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(start_x, start_y, start_z)
        bs.vel = rsim.Vec(vx, vy, vz)
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        base_def_x = random.uniform(-400.0, 400.0)
        base_def_y = sign * (ARENA_EXTENT_Y - random.uniform(400.0, 1200.0))
        def_yaw = -sign * math.pi / 2
        def_vel_y = -sign * random.uniform(200.0, 600.0)

        cars = rsim_arena.get_cars()
        for i, car in enumerate(cars):
            cs = rsim.CarState()
            cs.boost = random.uniform(50.0, 100.0)
            team = i % 2
            teammate_idx = i // 2

            if team == target_team:
                cx, cy = apply_teammate_stagger(base_cx, base_cy, teammate_idx, sign)
                cs.pos = rsim.Vec(cx, cy, 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(math.cos(yaw) * car_speed, math.sin(yaw) * car_speed, 0.0)
            else:
                def_x, def_y = apply_teammate_stagger(base_def_x, base_def_y, teammate_idx, -sign)
                cs.pos = rsim.Vec(def_x, def_y, 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=def_yaw, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(0.0, def_vel_y, 0.0)

            cs.ang_vel = rsim.Vec(0, 0, 0)
            car.set_state(cs)


class DribbleFlickScenarioSetter(BaseStateSetter):
    """
    Dedicated Ground Dribble & Flick Scenario (Seer / Slater / Nexto Architecture).
    Spawns the attacking car moving downfield at speed (700 - 1200 uu/s) with the ball
    already settled on the roof / hood (Z ~ 140 - 152 uu, matched velocity).
    Spawns a defending opponent in net (goalie) or shadowing downfield to force
    the bot to execute a flick over or past the defender.
    """
    def reset(self, rsim_arena: Any, num_players: int) -> None:
        arena_wrapper = None
        if hasattr(rsim_arena, "_rsim_arena") and rsim_arena._rsim_arena is not None:
            arena_wrapper = rsim_arena
            rsim_arena = rsim_arena._rsim_arena

        target_team = random.choice([0, 1])
        sign = 1.0 if target_team == 0 else -1.0  # Team 0 attacks +Y, Team 1 attacks -Y

        # Car spawns in midfield / attacking half, pointed downfield toward opponent goal
        cx = random.uniform(-1000.0, 1000.0)
        cy = -sign * random.uniform(500.0, 2000.0)
        cz = 17.0
        yaw = sign * math.pi / 2 + random.uniform(-0.15, 0.15)  # Pointed downfield

        car_speed = random.uniform(700.0, 1200.0)
        vx = math.cos(yaw) * car_speed
        vy = math.sin(yaw) * car_speed
        vz = 0.0

        # Ball is resting on the roof (center of mass slightly forward or neutral: local_x in [-20, 25])
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        right = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)
        local_x_offset = random.uniform(-20.0, 25.0)
        local_y_offset = random.uniform(-15.0, 15.0)

        bx = cx + local_x_offset * fwd[0] + local_y_offset * right[0]
        by = cy + local_x_offset * fwd[1] + local_y_offset * right[1]
        bz = random.uniform(142.0, 152.0)  # Perfectly settled on roof

        # Ball velocity is matched with car velocity (smooth velcro carry)
        bvx = vx + random.uniform(-20.0, 20.0)
        bvy = vy + random.uniform(-20.0, 20.0)
        bvz = random.uniform(-10.0, 15.0)

        is_goalie = (random.random() < 0.60)
        if is_goalie:
            base_def_x = random.uniform(-GOAL_HALF_WIDTH * 0.5, GOAL_HALF_WIDTH * 0.5)
            base_def_y = sign * (ARENA_EXTENT_Y - random.uniform(200.0, 600.0))
            opp_yaw = -sign * math.pi / 2
            opp_vx = random.uniform(-200.0, 200.0)
            opp_vy = 0.0
        else:
            base_def_x = cx + random.uniform(-200.0, 200.0)
            base_def_y = cy + sign * random.uniform(1000.0, 1600.0)
            opp_yaw = sign * math.pi / 2
            opp_vx = 0.0
            opp_vy = sign * random.uniform(500.0, 900.0)

        # Pure-Python arena fallback check
        if not hasattr(rsim_arena.ball, "get_state"):
            rsim_arena.ball.pos = np.array([bx, by, bz], dtype=np.float32)
            rsim_arena.ball.vel = np.array([bvx, bvy, bvz], dtype=np.float32)
            rsim_arena.ball.ang_vel = np.zeros(3, dtype=np.float32)
            for i, car in enumerate(rsim_arena.cars):
                car.boost = random.uniform(35.0, 80.0)
                team = i % 2
                teammate_idx = i // 2
                if team == target_team:
                    act_cx, act_cy = apply_teammate_stagger(cx, cy, teammate_idx, sign)
                    car.pos = np.array([act_cx, act_cy, cz], dtype=np.float32)
                    car.rot = np.array([0.0, yaw, 0.0], dtype=np.float32)
                    car.vel = np.array([vx, vy, vz], dtype=np.float32)
                else:
                    def_x, def_y = apply_teammate_stagger(base_def_x, base_def_y, teammate_idx, -sign)
                    car.pos = np.array([def_x, def_y, 17.0], dtype=np.float32)
                    car.rot = np.array([0.0, opp_yaw, 0.0], dtype=np.float32)
                    car.vel = np.array([opp_vx, opp_vy, 0.0], dtype=np.float32)
            return

        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(bx, by, bz)
        bs.vel = rsim.Vec(bvx, bvy, bvz)
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        cars = rsim_arena.get_cars()
        for i, car in enumerate(cars):
            cs = rsim.CarState()
            cs.boost = random.uniform(35.0, 80.0)
            team = i % 2
            teammate_idx = i // 2

            if team == target_team:
                act_cx, act_cy = apply_teammate_stagger(cx, cy, teammate_idx, sign)
                cs.pos = rsim.Vec(act_cx, act_cy, cz)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(vx, vy, vz)
            else:
                def_x, def_y = apply_teammate_stagger(base_def_x, base_def_y, teammate_idx, -sign)
                cs.pos = rsim.Vec(def_x, def_y, 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=opp_yaw, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(opp_vx, opp_vy, 0.0)

            cs.ang_vel = rsim.Vec(0, 0, 0)
            car.set_state(cs)

            cs.ang_vel = rsim.Vec(0, 0, 0)
            car.set_state(cs)

        if arena_wrapper is not None:
            arena_wrapper._sync_from_rsim()


class WeightedScenarioSetter:
    """
    Composite Scenario Manager. Samples across kickoffs, replays, aerials, wall plays, saves, turnarounds,
    wall rebounds, dribble flick setups, and user custom scenarios according to live user-configured probability weights.
    """
    def __init__(
        self,
        kickoff_prob: float = 0.22,
        replay_prob: float = 0.15,
        aerial_prob: float = 0.10,
        wall_prob: float = 0.09,
        save_prob: float = 0.09,
        turnaround_prob: float = 0.08,
        wall_rebound_prob: float = 0.08,
        dribble_flick_prob: float = 0.09,
        custom_prob: float = 0.10,
        replay_parser: Optional[ReplayParser] = None
    ):
        self.kickoff_prob = kickoff_prob
        self.replay_prob = replay_prob
        self.aerial_prob = aerial_prob
        self.wall_prob = wall_prob
        self.save_prob = save_prob
        self.turnaround_prob = turnaround_prob
        self.wall_rebound_prob = wall_rebound_prob
        self.dribble_flick_prob = dribble_flick_prob
        self.custom_prob = custom_prob

        self.kickoff_setter = KickoffSetter()
        self.aerial_setter = AerialScenarioSetter()
        self.wall_setter = WallPlaySetter()
        self.goalie_setter = GoalieSaveSetter()
        self.turnaround_setter = TurnaroundRecoverySetter()
        self.wall_rebound_setter = WallBounceReboundSetter()
        self.dribble_flick_setter = DribbleFlickScenarioSetter()
        self.replay_setter = ReplayStateSetter(parser=replay_parser)
        self.custom_setter = CustomScenarioSetter()

    def update_weights(self, config_dict: Dict[str, Any]):
        """Dynamically updates scenario distribution from live config."""
        sc = config_dict.get("scenarios", config_dict)
        if "kickoff_prob" in sc: self.kickoff_prob = float(sc["kickoff_prob"])
        if "replay_prob" in sc: self.replay_prob = float(sc["replay_prob"])
        if "aerial_prob" in sc: self.aerial_prob = float(sc["aerial_prob"])
        if "wall_prob" in sc: self.wall_prob = float(sc["wall_prob"])
        if "save_prob" in sc: self.save_prob = float(sc["save_prob"])
        if "turnaround_prob" in sc: self.turnaround_prob = float(sc["turnaround_prob"])
        if "wall_rebound_prob" in sc: self.wall_rebound_prob = float(sc["wall_rebound_prob"])
        if "dribble_flick_prob" in sc: self.dribble_flick_prob = float(sc["dribble_flick_prob"])
        if "custom_prob" in sc: self.custom_prob = float(sc["custom_prob"])

    def reset(self, rsim_arena: Any, num_players: int) -> str:
        """
        Samples a scenario based on current distribution and resets the RocketSim arena.
        Returns the chosen scenario name.
        """
        weights = [
            self.kickoff_prob,
            self.replay_prob,
            self.aerial_prob,
            self.wall_prob,
            self.save_prob,
            self.turnaround_prob,
            self.wall_rebound_prob,
            self.dribble_flick_prob,
            self.custom_prob
        ]
        total = sum(weights)
        if total <= 1e-6:
            self.kickoff_setter.reset(rsim_arena, num_players)
            return "kickoff"

        r = random.uniform(0, total)
        cumulative = 0.0

        # 1. Kickoff
        cumulative += self.kickoff_prob
        if r <= cumulative:
            self.kickoff_setter.reset(rsim_arena, num_players)
            return "kickoff"

        # 2. Replay
        cumulative += self.replay_prob
        if r <= cumulative:
            if self.replay_setter.reset(rsim_arena, num_players):
                return "replay"
            self.kickoff_setter.reset(rsim_arena, num_players)
            return "kickoff"

        # 3. Aerial
        cumulative += self.aerial_prob
        if r <= cumulative:
            self.aerial_setter.reset(rsim_arena, num_players)
            return "aerial"

        # 4. Wall Play
        cumulative += self.wall_prob
        if r <= cumulative:
            self.wall_setter.reset(rsim_arena, num_players)
            return "wall_play"

        # 5. Wall Rebound / Bounce Interception
        cumulative += self.wall_rebound_prob
        if r <= cumulative:
            self.wall_rebound_setter.reset(rsim_arena, num_players)
            return "wall_rebound"

        # 6. Turnaround Recovery
        cumulative += self.turnaround_prob
        if r <= cumulative:
            self.turnaround_setter.reset(rsim_arena, num_players)
            return "turnaround"

        # 7. Dribble & Flick
        cumulative += self.dribble_flick_prob
        if r <= cumulative:
            self.dribble_flick_setter.reset(rsim_arena, num_players)
            return "dribble_flick"

        # 8. Custom Scenarios
        cumulative += self.custom_prob
        if r <= cumulative:
            if self.custom_setter.reset(rsim_arena, num_players):
                return "custom"
            self.kickoff_setter.reset(rsim_arena, num_players)
            return "kickoff"

        # 9. Goalie Save
        self.goalie_setter.reset(rsim_arena, num_players)
        return "goalie_save"

