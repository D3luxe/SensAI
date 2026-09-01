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
    """Computes exact 3x3 orthonormal basis (Forward, Right, Up) matching RocketSim."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    fwd = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
    right = np.array([-sy * cr + cy * sp * sr, cy * cr + sy * sp * sr, -cp * sr], dtype=np.float32)
    up = np.array([-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr], dtype=np.float32)
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


class WeightedScenarioSetter:
    """
    Composite Scenario Manager. Samples across kickoffs, replays, aerials, wall plays, and saves
    according to live user-configured probability weights.
    """
    def __init__(
        self,
        kickoff_prob: float = 0.35,
        replay_prob: float = 0.25,
        aerial_prob: float = 0.15,
        wall_prob: float = 0.15,
        save_prob: float = 0.10,
        replay_parser: Optional[ReplayParser] = None
    ):
        self.kickoff_prob = kickoff_prob
        self.replay_prob = replay_prob
        self.aerial_prob = aerial_prob
        self.wall_prob = wall_prob
        self.save_prob = save_prob

        self.kickoff_setter = KickoffSetter()
        self.aerial_setter = AerialScenarioSetter()
        self.wall_setter = WallPlaySetter()
        self.goalie_setter = GoalieSaveSetter()
        self.replay_setter = ReplayStateSetter(parser=replay_parser)

    def update_weights(self, config_dict: Dict[str, Any]):
        """Dynamically updates scenario distribution from live config."""
        sc = config_dict.get("scenarios", config_dict)
        if "kickoff_prob" in sc: self.kickoff_prob = float(sc["kickoff_prob"])
        if "replay_prob" in sc: self.replay_prob = float(sc["replay_prob"])
        if "aerial_prob" in sc: self.aerial_prob = float(sc["aerial_prob"])
        if "wall_prob" in sc: self.wall_prob = float(sc["wall_prob"])
        if "save_prob" in sc: self.save_prob = float(sc["save_prob"])

    def reset(self, rsim_arena: Any, num_players: int) -> str:
        """
        Samples a scenario based on current distribution and resets the RocketSim arena.
        Returns the chosen scenario name.
        """
        weights = [self.kickoff_prob, self.replay_prob, self.aerial_prob, self.wall_prob, self.save_prob]
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
            # Fallback to kickoff if no replays available
            self.kickoff_setter.reset(rsim_arena, num_players)
            return "kickoff"

        # 3. Aerial
        cumulative += self.aerial_prob
        if r <= cumulative:
            self.aerial_setter.reset(rsim_arena, num_players)
            return "aerial"

        # 4. Wall
        cumulative += self.wall_prob
        if r <= cumulative:
            self.wall_setter.reset(rsim_arena, num_players)
            return "wall_play"

        # 5. Goalie Save
        self.goalie_setter.reset(rsim_arena, num_players)
        return "goalie_save"
