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
            cs = car.get_state()
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
    Spawns high flying / floating balls (z: 600 - 1500) with cars oriented for fast aerial launches.
    """
    def reset(self, rsim_arena: Any, num_players: int) -> None:
        target_team = random.choice([0, 1])
        sign = 1.0 if target_team == 0 else -1.0

        # Ball in mid-air or rising arc
        bx = random.uniform(-1500, 1500)
        by = sign * random.uniform(500, 2500)
        bz = random.uniform(600, 1400)
        bvx = random.uniform(-500, 500)
        bvy = sign * random.uniform(200, 1000)
        bvz = random.uniform(-200, 600)

        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(bx, by, bz)
        bs.vel = rsim.Vec(bvx, bvy, bvz)
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        # Attacking car
        for i, car in enumerate(rsim_arena.get_cars()):
            cs = car.get_state()
            cs.boost = random.uniform(60.0, 100.0)
            team = i % 2

            if team == target_team:
                cx = bx + random.uniform(-600, 600)
                cy = by - sign * random.uniform(800, 1800)
                cs.pos = rsim.Vec(cx, cy, 17.0)
                yaw = math.atan2(by - cy, bx - cx)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(math.cos(yaw) * 800, math.sin(yaw) * 800, 0)
            else:
                # Defending car back in goal area
                cs.pos = rsim.Vec(random.uniform(-800, 800), sign * random.uniform(3800, 4800), 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=-sign * math.pi/2, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(0, 0, 0)

            cs.ang_vel = rsim.Vec(0, 0, 0)
            car.set_state(cs)


class WallPlaySetter(BaseStateSetter):
    """
    Spawns ball rolling along arena sidewalls or bouncing high off backboard.
    """
    def reset(self, rsim_arena: Any, num_players: int) -> None:
        side = random.choice([-1.0, 1.0])  # Left or Right Wall
        target_team = random.choice([0, 1])
        sign = 1.0 if target_team == 0 else -1.0

        # Ball climbing sidewall
        bx = side * (ARENA_EXTENT_X - 120.0)
        by = random.uniform(-2500, 2500)
        bz = random.uniform(300, 1200)

        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(bx, by, bz)
        bs.vel = rsim.Vec(side * -200, sign * random.uniform(600, 1400), random.uniform(100, 500))
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        for i, car in enumerate(rsim_arena.get_cars()):
            cs = car.get_state()
            cs.boost = random.uniform(50.0, 100.0)
            team = i % 2

            if team == target_team:
                cx = side * random.uniform(2500, 3600)
                cy = by - sign * random.uniform(600, 1400)
                cs.pos = rsim.Vec(cx, cy, 17.0)
                yaw = math.atan2(by - cy, bx - cx)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(math.cos(yaw) * 1000, math.sin(yaw) * 1000, 0)
            else:
                cs.pos = rsim.Vec(0, sign * 4200, 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=-sign * math.pi/2, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(0, 0, 0)

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

        for i, car in enumerate(rsim_arena.get_cars()):
            cs = car.get_state()
            cs.boost = random.uniform(40.0, 100.0)
            team = i % 2

            if team == defending_team:
                # Defender in shadow defense position or near goal line
                cs.pos = rsim.Vec(random.uniform(-600, 600), defend_goal_y + sign * random.uniform(400, 1500), 17.0)
                yaw = -sign * math.pi / 2 + random.uniform(-0.4, 0.4)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(0, sign * random.uniform(200, 800), 0)
            else:
                # Shooter trailing the shot
                cs.pos = rsim.Vec(start_x, start_y + sign * 800, 17.0)
                yaw = math.atan2(defend_goal_y - start_y, target_x - start_x)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(math.cos(yaw) * 1200, math.sin(yaw) * 1200, 0)

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

        # Set ball state
        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(*sample["ball_pos"])
        bs.vel = rsim.Vec(*sample["ball_vel"])
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        # Set car states
        cars = rsim_arena.get_cars()
        for i in range(min(len(cars), num_players)):
            cs = cars[i].get_state()
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
            # Ball rolling fast downfield toward opponent goal, car chasing from behind.
            # Directly trains forward dodges, diagonal speed-flips, and forward traversal acceleration.
            bx = random.uniform(-1000, 1000)
            by = sign * random.uniform(1000, 2500)
            bz = 93.15

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            bs.vel = rsim.Vec(random.uniform(-100, 100), sign * random.uniform(600, 1300), 0)
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = car.get_state()
                cs.boost = random.uniform(20.0, 50.0)
                team = i % 2

                if team == target_team:
                    # Car placed 2000-3500 uu behind ball facing directly toward attacking net
                    cx = bx + random.uniform(-200, 200)
                    cy = by - sign * random.uniform(2000, 3200)
                    cs.pos = rsim.Vec(cx, cy, 17.0)
                    yaw = sign * math.pi / 2 + random.uniform(-0.15, 0.15)
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    speed = random.uniform(500, 900)
                    cs.vel = rsim.Vec(math.cos(yaw) * speed, math.sin(yaw) * speed, 0)
                else:
                    # Opponent scrambling back on defense
                    cs.pos = rsim.Vec(random.uniform(-800, 800), by + sign * random.uniform(1000, 1800), 17.0)
                    yaw = -sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0, -sign * 700, 0)

                cs.ang_vel = rsim.Vec(0, 0, 0)
                car.set_state(cs)

        elif mode == "wrong_side_dribble":
            # Scenario A: Bot is directly behind the ball facing its own goal
            # Forces policy to brake / peel off rather than accelerating into own net
            bx = random.uniform(-1000, 1000)
            by = sign * random.uniform(0, 2000)
            bz = 93.15

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            # Ball rolling slowly towards target team's defending net (-sign Y direction)
            bs.vel = rsim.Vec(random.uniform(-50, 50), -sign * random.uniform(100, 400), 0)
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = car.get_state()
                cs.boost = random.uniform(20.0, 70.0)
                team = i % 2

                if team == target_team:
                    # Car placed slightly behind ball facing defending goal
                    cx = bx + random.uniform(-40, 40)
                    cy = by + sign * random.uniform(150, 300)
                    cs.pos = rsim.Vec(cx, cy, 17.0)
                    yaw = -sign * math.pi / 2 + random.uniform(-0.1, 0.1)
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0, -sign * random.uniform(300, 700), 0)
                else:
                    # Opponent further back or challenging
                    cs.pos = rsim.Vec(random.uniform(-800, 800), by - sign * random.uniform(1200, 2000), 17.0)
                    yaw = sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0, sign * 500, 0)

                cs.ang_vel = rsim.Vec(0, 0, 0)
                car.set_state(cs)

        elif mode == "reverse_halfflip":
            # Scenario C: Car is stopped or reversing facing forward, ball is cleared over its head toward own net
            # Optimal response is an immediate half-flip turnaround to chase down the ball
            bx = random.uniform(-1200, 1200)
            by = -sign * random.uniform(1000, 2800)
            bz = random.uniform(93.15, 350.0)

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            # Fast cleared ball heading toward defending net
            bs.vel = rsim.Vec(random.uniform(-200, 200), -sign * random.uniform(800, 1500), random.uniform(0, 300))
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = car.get_state()
                cs.boost = random.uniform(35.0, 80.0)
                team = i % 2

                if team == target_team:
                    # Car facing away from ball (facing opponent end +sign Y) at rest or reversing slowly
                    cx = random.uniform(-800, 800)
                    cy = sign * random.uniform(200, 1400)
                    cs.pos = rsim.Vec(cx, cy, 17.0)
                    yaw = sign * math.pi / 2 + random.uniform(-0.2, 0.2)
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    rev_speed = random.uniform(-400.0, 0.0)
                    cs.vel = rsim.Vec(0.0, sign * rev_speed, 0.0)
                else:
                    # Opponent chasing the clear
                    cs.pos = rsim.Vec(random.uniform(-600, 600), sign * random.uniform(1600, 2600), 17.0)
                    yaw = -sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0, -sign * 900, 0)

                cs.ang_vel = rsim.Vec(0, 0, 0)
                car.set_state(cs)

        elif mode == "midflip_inverted":
            # Scenario D: Curriculum reset - car spawned mid-air upside down with backward momentum
            # Isolates and trains the flip-cancel air-roll recovery landing
            bx = random.uniform(-1000, 1000)
            by = -sign * random.uniform(1500, 3200)
            bz = 93.15

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            bs.vel = rsim.Vec(0.0, -sign * random.uniform(600, 1200), 0.0)
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = car.get_state()
                cs.boost = random.uniform(40.0, 80.0)
                team = i % 2

                if team == target_team:
                    cx = random.uniform(-600, 600)
                    cy = sign * random.uniform(400, 1200)
                    cz = random.uniform(120.0, 220.0)
                    cs.pos = rsim.Vec(cx, cy, cz)
                    yaw = sign * math.pi / 2
                    # Inverted orientation (roll = +/- pi)
                    roll_val = math.pi if random.random() < 0.5 else -math.pi
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=roll_val).as_rot_mat()
                    # High backward velocity
                    cs.vel = rsim.Vec(0.0, -sign * random.uniform(900, 1300), random.uniform(-50, 50))
                else:
                    cs.pos = rsim.Vec(0.0, sign * 2500, 17.0)
                    yaw = -sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0, -sign * 700, 0)

                cs.ang_vel = rsim.Vec(0, 0, 0)
                car.set_state(cs)

        elif mode == "rear_quarter_scramble":
            # Scenario F: Close-proximity rear quarter-panel / blindspot scramble
            # Ball rests or bounces slowly at the rear bumper/quarter panel (80-160 uu).
            # Directly trains low-speed hook turns, powerslide 180 cuts, and reverse ball sweeps.
            cx = random.uniform(-1500, 1500)
            cy = sign * random.uniform(0, 2000)
            yaw = sign * math.pi / 2 + random.uniform(-0.25, 0.25)

            # Place ball right at the rear quarter panel (longitudinal offset in [-130, -50], lateral in [+-35, +-75])
            side_sign = random.choice([-1.0, 1.0])
            lat_off = side_sign * random.uniform(35.0, 75.0)
            long_off = -sign * random.uniform(50.0, 130.0)
            bx = cx + lat_off
            by = cy + long_off
            bz = random.uniform(93.15, 140.0)

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            bs.vel = rsim.Vec(random.uniform(-40, 40), random.uniform(-40, 40), random.uniform(0, 100))
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = car.get_state()
                cs.boost = random.uniform(0.0, 40.0)
                team = i % 2

                if team == target_team:
                    cs.pos = rsim.Vec(cx, cy, 17.0)
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    speed = random.uniform(0.0, 200.0)
                    cs.vel = rsim.Vec(math.cos(yaw) * speed, math.sin(yaw) * speed, 0.0)
                else:
                    cs.pos = rsim.Vec(random.uniform(-1000, 1000), by + sign * random.uniform(1500, 2800), 17.0)
                    yaw_opp = -sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw_opp, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0.0, -sign * random.uniform(400, 800), 0.0)

                cs.ang_vel = rsim.Vec(0, 0, 0)
                car.set_state(cs)

        else:
            # Scenario B: Ball grounded behind midfield line, car sprinting downfield away from ball
            bx = random.uniform(-1200, 1200)
            by = -sign * random.uniform(800, 2400)
            bz = 93.15

            bs = rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(bx, by, bz)
            bs.vel = rsim.Vec(random.uniform(-100, 100), -sign * random.uniform(0, 300), 0)
            bs.ang_vel = rsim.Vec(0, 0, 0)
            rsim_arena.ball.set_state(bs)

            for i, car in enumerate(rsim_arena.get_cars()):
                cs = car.get_state()
                cs.boost = random.uniform(30.0, 80.0)
                team = i % 2

                if team == target_team:
                    # Car moving fast away from ball towards opponent end
                    cx = random.uniform(-1000, 1000)
                    cy = sign * random.uniform(200, 1800)
                    cs.pos = rsim.Vec(cx, cy, 17.0)
                    yaw = sign * math.pi / 2 + random.uniform(-0.3, 0.3)
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    # Fast forward momentum (1200 - 1800 uu/s) moving away from ball
                    speed = random.uniform(1200, 1800)
                    cs.vel = rsim.Vec(math.cos(yaw) * speed, math.sin(yaw) * speed, 0)
                else:
                    # Opponent challenging or rotating back
                    cs.pos = rsim.Vec(random.uniform(-800, 800), by - sign * random.uniform(400, 1000), 17.0)
                    yaw = sign * math.pi / 2
                    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                    cs.vel = rsim.Vec(0, sign * 600, 0)

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

            cs = cars[0].get_state()
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
            cs1 = cars[1].get_state()
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
    Spawns hard clears, wall passes, and backboard bangs (1200 - 2000 uu/s)
    speeding toward the sidewalls, corners, or backboard to rebound into the field.
    The car is placed infield with speed and boost to practice bounce anticipation,
    rebound timing, and interception.
    """
    def reset(self, rsim_arena: Any, num_players: int) -> None:
        target_team = random.choice([0, 1])
        sign = 1.0 if target_team == 0 else -1.0
        bounce_type = random.choice(["sidewall", "backboard"])

        side = random.choice([-1.0, 1.0])
        if bounce_type == "sidewall":
            # Ball starts infield and shoots toward the sidewall
            start_x = side * random.uniform(500, 2000)
            start_y = sign * random.uniform(-1000, 2000)
            start_z = random.uniform(150, 600)

            # High velocity aimed toward the side wall (x = side * 4096)
            vx = side * random.uniform(1200, 1800)
            vy = sign * random.uniform(200, 800)
            vz = random.uniform(100, 400)

            # Car positioned midfield / infield tracking toward the expected rebound zone
            cx = side * random.uniform(400, 1500)
            cy = start_y - sign * random.uniform(400, 1200)
            cz = 17.0

            yaw = math.atan2(sign * 1.0, side * 0.3)
            car_speed = random.uniform(600, 1100)
        else:
            # Backboard bounce: Ball shoots toward opponent backboard (above goal)
            start_x = random.uniform(-1500, 1500)
            start_y = sign * random.uniform(1000, 2800)
            start_z = random.uniform(200, 600)

            vy = sign * random.uniform(1300, 1900)
            vx = random.uniform(-400, 400)
            vz = random.uniform(300, 700)

            # Car trailing downfield ready to read the backboard rebound
            cx = random.uniform(-800, 800)
            cy = start_y - sign * random.uniform(1200, 2000)
            cz = 17.0

            yaw = sign * math.pi / 2
            car_speed = random.uniform(700, 1200)

        bs = rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(start_x, start_y, start_z)
        bs.vel = rsim.Vec(vx, vy, vz)
        bs.ang_vel = rsim.Vec(0, 0, 0)
        rsim_arena.ball.set_state(bs)

        cars = rsim_arena.get_cars()
        for i, car in enumerate(cars):
            cs = car.get_state()
            cs.boost = random.uniform(40.0, 90.0)
            team = i % 2

            if team == target_team:
                cs.pos = rsim.Vec(cx, cy, cz)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(math.cos(yaw) * car_speed, math.sin(yaw) * car_speed, 0.0)
            else:
                # Opponent in defensive posture
                opp_y = sign * (ARENA_EXTENT_Y - random.uniform(400, 1200))
                cs.pos = rsim.Vec(random.uniform(-600, 600), opp_y, 17.0)
                cs.rot_mat = rsim.Angle(pitch=0.0, yaw=-sign * math.pi / 2, roll=0.0).as_rot_mat()
                cs.vel = rsim.Vec(0.0, -sign * random.uniform(200, 600), 0.0)

            cs.ang_vel = rsim.Vec(0, 0, 0)
            car.set_state(cs)


class WeightedScenarioSetter:
    """
    Composite Scenario Manager. Samples across kickoffs, replays, aerials, wall plays, saves, turnarounds,
    wall rebounds, and user custom scenarios according to live user-configured probability weights.
    """
    def __init__(
        self,
        kickoff_prob: float = 0.25,
        replay_prob: float = 0.20,
        aerial_prob: float = 0.11,
        wall_prob: float = 0.09,
        save_prob: float = 0.09,
        turnaround_prob: float = 0.08,
        wall_rebound_prob: float = 0.08,
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
        self.custom_prob = custom_prob

        self.kickoff_setter = KickoffSetter()
        self.aerial_setter = AerialScenarioSetter()
        self.wall_setter = WallPlaySetter()
        self.goalie_setter = GoalieSaveSetter()
        self.turnaround_setter = TurnaroundRecoverySetter()
        self.wall_rebound_setter = WallBounceReboundSetter()
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

        # 7. Custom Scenarios
        cumulative += self.custom_prob
        if r <= cumulative:
            if self.custom_setter.reset(rsim_arena, num_players):
                return "custom"
            self.kickoff_setter.reset(rsim_arena, num_players)
            return "kickoff"

        # 8. Goalie Save
        self.goalie_setter.reset(rsim_arena, num_players)
        return "goalie_save"

