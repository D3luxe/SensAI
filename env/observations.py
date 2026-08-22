"""
Observation Builder for Rocket League Agents.
Transforms 3D simulation state into normalized, symmetric feature vectors for neural network policies.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Any, Optional
from env.physics_engine import (
    CarState, BallState, RocketSimArena,
    ARENA_EXTENT_X, ARENA_EXTENT_Y, ARENA_HEIGHT_Z,
    CAR_MAX_SPEED, BALL_MAX_SPEED, GOAL_HEIGHT
)


class DefaultObservationBuilder:
    """
    Standard RLGym-style observation builder with local coordinate transformations and symmetric team inversion.
    """
    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric
        self.obs_dim = 64

    def build_obs(self, car: CarState, arena: RocketSimArena) -> np.ndarray:
        obs = []

        # Symmetry multiplier: if Orange (team 1) and symmetric is True, flip X and Y
        inv = -1.0 if (self.symmetric and car.team == 1) else 1.0

        # Car orientation vectors
        fwd = car.get_forward_vector()
        right = car.get_right_vector()
        up = car.get_up_vector()

        if inv == -1.0:
            fwd = np.array([-fwd[0], -fwd[1], fwd[2]], dtype=np.float32)
            right = np.array([-right[0], -right[1], right[2]], dtype=np.float32)
            up = np.array([-up[0], -up[1], up[2]], dtype=np.float32)

        # 1. Self Car State (22 features)
        car_pos_norm = np.array([
            (car.pos[0] * inv) / ARENA_EXTENT_X,
            (car.pos[1] * inv) / ARENA_EXTENT_Y,
            car.pos[2] / ARENA_HEIGHT_Z
        ], dtype=np.float32)

        car_vel_norm = np.array([
            (car.vel[0] * inv) / CAR_MAX_SPEED,
            (car.vel[1] * inv) / CAR_MAX_SPEED,
            car.vel[2] / CAR_MAX_SPEED
        ], dtype=np.float32)

        obs.extend(car_pos_norm)             # 3
        obs.extend(car_vel_norm)             # 3
        obs.extend(fwd)                      # 3
        obs.extend(right)                    # 3
        obs.extend(up)                       # 3
        obs.extend(car.ang_vel * 0.1)        # 3
        obs.append(car.boost / 100.0)        # 1
        obs.append(1.0 if car.on_ground else 0.0) # 1
        obs.append(1.0 if car.has_jump else 0.0)  # 1
        obs.append(1.0 if car.has_flip else 0.0)  # 1

        # 2. Ball State (9 features)
        ball_pos_norm = np.array([
            (arena.ball.pos[0] * inv) / ARENA_EXTENT_X,
            (arena.ball.pos[1] * inv) / ARENA_EXTENT_Y,
            arena.ball.pos[2] / ARENA_HEIGHT_Z
        ], dtype=np.float32)

        ball_vel_norm = np.array([
            (arena.ball.vel[0] * inv) / BALL_MAX_SPEED,
            (arena.ball.vel[1] * inv) / BALL_MAX_SPEED,
            arena.ball.vel[2] / BALL_MAX_SPEED
        ], dtype=np.float32)

        obs.extend(ball_pos_norm)            # 3
        obs.extend(ball_vel_norm)            # 3
        obs.extend(arena.ball.ang_vel * 0.1) # 3

        # 3. Relative Features in Car Local Frame (13 features)
        rel_ball_pos = (arena.ball.pos - car.pos) * np.array([inv, inv, 1.0], dtype=np.float32)
        rel_ball_vel = (arena.ball.vel - car.vel) * np.array([inv, inv, 1.0], dtype=np.float32)

        local_ball_pos = np.array([
            np.dot(rel_ball_pos, fwd) / 2000.0,
            np.dot(rel_ball_pos, right) / 2000.0,
            np.dot(rel_ball_pos, up) / 2000.0
        ], dtype=np.float32)

        local_ball_vel = np.array([
            np.dot(rel_ball_vel, fwd) / CAR_MAX_SPEED,
            np.dot(rel_ball_vel, right) / CAR_MAX_SPEED,
            np.dot(rel_ball_vel, up) / CAR_MAX_SPEED
        ], dtype=np.float32)

        dist_ball = float(np.linalg.norm(rel_ball_pos)) / 6000.0

        # Goal vectors relative to car in local frame
        target_goal_global = np.array([0.0, ARENA_EXTENT_Y, GOAL_HEIGHT * 0.5], dtype=np.float32)
        defend_goal_global = np.array([0.0, -ARENA_EXTENT_Y, GOAL_HEIGHT * 0.5], dtype=np.float32)

        rel_target = target_goal_global - (car.pos * np.array([inv, inv, 1.0], dtype=np.float32))
        rel_defend = defend_goal_global - (car.pos * np.array([inv, inv, 1.0], dtype=np.float32))

        norm_target = max(1e-4, np.linalg.norm(rel_target))
        norm_defend = max(1e-4, np.linalg.norm(rel_defend))

        unit_target = rel_target / norm_target
        unit_defend = rel_defend / norm_defend

        local_target_goal = np.array([
            np.dot(unit_target, fwd),
            np.dot(unit_target, right),
            np.dot(unit_target, up)
        ], dtype=np.float32)

        local_defend_goal = np.array([
            np.dot(unit_defend, fwd),
            np.dot(unit_defend, right),
            np.dot(unit_defend, up)
        ], dtype=np.float32)

        obs.extend(local_ball_pos)           # 3
        obs.extend(local_ball_vel)           # 3
        obs.append(dist_ball)                # 1
        obs.extend(local_target_goal)        # 3
        obs.extend(local_defend_goal)        # 3

        # 4. Opponents / Other Players (14 features for primary opponent)
        opponents = [c for c in arena.cars if c.team != car.team]
        if opponents:
            opp = opponents[0]
            opp_pos_norm = np.array([
                (opp.pos[0] * inv) / ARENA_EXTENT_X,
                (opp.pos[1] * inv) / ARENA_EXTENT_Y,
                opp.pos[2] / ARENA_HEIGHT_Z
            ], dtype=np.float32)

            opp_vel_norm = np.array([
                (opp.vel[0] * inv) / CAR_MAX_SPEED,
                (opp.vel[1] * inv) / CAR_MAX_SPEED,
                opp.vel[2] / CAR_MAX_SPEED
            ], dtype=np.float32)

            rel_opp_pos = (opp.pos - car.pos) * np.array([inv, inv, 1.0], dtype=np.float32)
            rel_opp_vel = (opp.vel - car.vel) * np.array([inv, inv, 1.0], dtype=np.float32)

            local_opp_pos = np.array([
                np.dot(rel_opp_pos, fwd) / 2000.0,
                np.dot(rel_opp_pos, right) / 2000.0,
                np.dot(rel_opp_pos, up) / 2000.0
            ], dtype=np.float32)

            local_opp_vel = np.array([
                np.dot(rel_opp_vel, fwd) / CAR_MAX_SPEED,
                np.dot(rel_opp_vel, right) / CAR_MAX_SPEED,
                np.dot(rel_opp_vel, up) / CAR_MAX_SPEED
            ], dtype=np.float32)

            obs.extend(opp_pos_norm)         # 3
            obs.extend(opp_vel_norm)         # 3
            obs.extend(local_opp_pos)        # 3
            obs.extend(local_opp_vel)        # 3
            obs.append(opp.boost / 100.0)    # 1
            obs.append(1.0 if opp.on_ground else 0.0) # 1
        else:
            obs.extend([0.0] * 14)

        # 5. Big Boost Pads (6 features)
        big_pads = [pad for pad in arena.boost_pads if pad.is_big][:6]
        for pad in big_pads:
            obs.append(1.0 if pad.is_active else 0.0)
        while len(big_pads) < 6:
            obs.append(1.0)

        res = np.array(obs, dtype=np.float32)
        return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
