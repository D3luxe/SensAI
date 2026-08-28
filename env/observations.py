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

# 74-Dimensional Left-Right (X -> -X) Observation Symmetry Reflection Mask
# Multiplies features by -1.0 for lateral X components, roll, yaw, and relative right offsets
OBS_MIRROR_MASK_NP = np.array([
    # 1. Self Car State (22 features)
    -1.0,  1.0,  1.0,   # car_pos (pos.x negated)
    -1.0,  1.0,  1.0,   # car_vel (vel.x negated)
    -1.0,  1.0,  1.0,   # fwd (fwd.x negated)
     1.0, -1.0,  1.0,   # right (right.y negated across X=0 sagittal mirror plane)
    -1.0,  1.0,  1.0,   # up (up.x negated)
    -1.0,  1.0, -1.0,   # ang_vel (roll.x, yaw.z negated)
     1.0,  1.0,  1.0,  1.0,  # boost, on_ground, has_jump, has_flip
    # 2. Ball State (9 features)
    -1.0,  1.0,  1.0,   # ball_pos (pos.x negated)
    -1.0,  1.0,  1.0,   # ball_vel (vel.x negated)
    -1.0,  1.0, -1.0,   # ball_ang_vel (roll.x, yaw.z negated)
    # 2b. Future Ball Trajectory Prediction (3 features)
    -1.0,  1.0,  1.0,   # future_ball_pos (pos.x negated)
    # 3. Relative Features in Car Local Frame (16 features)
     1.0, -1.0,  1.0,   # local_ball_pos (right offset negated)
     1.0, -1.0,  1.0,   # local_future_ball_pos (right offset negated)
     1.0, -1.0,  1.0,   # local_ball_vel (right vel negated)
     1.0,             # dist_ball
     1.0, -1.0,  1.0,   # local_target_goal (right offset negated)
     1.0, -1.0,  1.0,   # local_defend_goal (right offset negated)
    # 3b/3c. Sensors (4 features)
     1.0,  1.0,  1.0,  1.0,  # threat_intensity, threat_z, is_kickoff, is_first_touch
    # 4. Opponent State (14 features)
    -1.0,  1.0,  1.0,   # opp_pos (pos.x negated)
    -1.0,  1.0,  1.0,   # opp_vel (vel.x negated)
     1.0, -1.0,  1.0,   # local_opp_pos (right offset negated)
     1.0, -1.0,  1.0,   # local_opp_vel (right vel negated)
     1.0,  1.0,        # opp_boost, opp_on_ground
    # 5. Boost Pad Spatial Vectors (6 features)
     1.0, -1.0,  1.0,   # nearest small pad (fwd, right negated, dist)
     1.0, -1.0,  1.0    # nearest big orb (fwd, right negated, dist)
], dtype=np.float32)

# 8-Dimensional Action Reflection Mask: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
ACT_MIRROR_MASK_NP = np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    """Mirrors a single or batched numpy observation across the X=0 plane."""
    return obs * OBS_MIRROR_MASK_NP


def mirror_act(act: np.ndarray) -> np.ndarray:
    """Mirrors continuous action vectors (negates steer, yaw, and roll)."""
    return act * ACT_MIRROR_MASK_NP


class DefaultObservationBuilder:
    """
    Standard RLGym-style observation builder with local coordinate transformations and symmetric team inversion.
    """
    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric
        self.obs_dim = 74

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

        # 2b. Future Ball Trajectory Prediction (0.5s ahead = 60 ticks @ 120Hz)
        future_ball_pos = arena.get_predicted_ball_pos(60) if hasattr(arena, "get_predicted_ball_pos") else None
        if future_ball_pos is None:
            # Kinematic ballistic trajectory fallback
            dt = 0.5
            px = arena.ball.pos[0] + arena.ball.vel[0] * dt
            py = arena.ball.pos[1] + arena.ball.vel[1] * dt
            pz = max(93.0, arena.ball.pos[2] + arena.ball.vel[2] * dt + 0.5 * (-650.0) * (dt ** 2))
            if abs(px) > 4000.0:
                px = np.sign(px) * (4000.0 - (abs(px) - 4000.0) * 0.6)
            if abs(py) > 5000.0:
                py = np.sign(py) * (5000.0 - (abs(py) - 5000.0) * 0.6)
            future_ball_pos = np.array([px, py, pz], dtype=np.float32)

        future_ball_pos_norm = np.array([
            (future_ball_pos[0] * inv) / ARENA_EXTENT_X,
            (future_ball_pos[1] * inv) / ARENA_EXTENT_Y,
            future_ball_pos[2] / ARENA_HEIGHT_Z
        ], dtype=np.float32)

        obs.extend(future_ball_pos_norm)     # 3

        # 3. Relative Features in Car Local Frame (16 features)
        rel_ball_pos = (arena.ball.pos - car.pos) * np.array([inv, inv, 1.0], dtype=np.float32)
        rel_ball_vel = (arena.ball.vel - car.vel) * np.array([inv, inv, 1.0], dtype=np.float32)
        rel_future_ball_pos = (future_ball_pos - car.pos) * np.array([inv, inv, 1.0], dtype=np.float32)

        local_ball_pos = np.array([
            np.dot(rel_ball_pos, fwd) / 2000.0,
            np.dot(rel_ball_pos, right) / 2000.0,
            np.dot(rel_ball_pos, up) / 2000.0
        ], dtype=np.float32)

        local_future_ball_pos = np.array([
            np.dot(rel_future_ball_pos, fwd) / 2000.0,
            np.dot(rel_future_ball_pos, right) / 2000.0,
            np.dot(rel_future_ball_pos, up) / 2000.0
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
        obs.extend(local_future_ball_pos)    # 3
        obs.extend(local_ball_vel)           # 3
        obs.append(dist_ball)                # 1
        obs.extend(local_target_goal)        # 3
        obs.extend(local_defend_goal)        # 3

        # 3b. Defending Goal Threat Sensor (2 features: Threat Intensity, Threat Entry Height)
        is_threat, threat_intensity, threat_z = arena.get_shot_threat(car.team) if hasattr(arena, "get_shot_threat") else (False, 0.0, 0.0)
        obs.append(float(threat_intensity))  # 1
        obs.append(float(threat_z))          # 1

        # 3c. Explicit Kickoff Awareness Sensor (2 features: is_kickoff, is_first_touch_open)
        is_center_ball = bool(abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and float(np.linalg.norm(arena.ball.vel)) < 80.0)
        is_first_touch = bool(all(c.ball_touches == 0 for c in arena.cars))
        obs.append(1.0 if is_center_ball else 0.0)        # 1: Kickoff active flag
        obs.append(1.0 if is_first_touch else 0.0)       # 1: First-touch race open flag

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

        # 5. Active Boost Pad Spatial Vectors (6 features)
        # Vectorized SIMD relative vector (fwd, right, dist) to nearest active small pad & big orb
        if hasattr(arena, "_small_pad_pos_3d") and hasattr(arena, "_small_pad_active"):
            sm_act = arena._small_pad_active
            if sm_act.any():
                act_pos = arena._small_pad_pos_3d[sm_act]
                diff = act_pos[:, :2] - car.pos[:2]
                d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
                min_i = int(np.argmin(d2))
                rel_sm = (act_pos[min_i] - car.pos) * np.array([inv, inv, 1.0], dtype=np.float32)
                dist_sm = float(math.sqrt(d2[min_i]))
                obs.extend([
                    float(np.dot(rel_sm, fwd)) / 2000.0,
                    float(np.dot(rel_sm, right)) / 2000.0,
                    dist_sm / 4000.0
                ])
            else:
                obs.extend([0.0, 0.0, 1.0])

            bg_act = arena._big_pad_active
            if bg_act.any():
                act_pos = arena._big_pad_pos_3d[bg_act]
                diff = act_pos[:, :2] - car.pos[:2]
                d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
                min_i = int(np.argmin(d2))
                rel_bg = (act_pos[min_i] - car.pos) * np.array([inv, inv, 1.0], dtype=np.float32)
                dist_bg = float(math.sqrt(d2[min_i]))
                obs.extend([
                    float(np.dot(rel_bg, fwd)) / 3000.0,
                    float(np.dot(rel_bg, right)) / 3000.0,
                    dist_bg / 6000.0
                ])
            else:
                obs.extend([0.0, 0.0, 1.0])
        else:
            obs.extend([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

        res = np.array(obs, dtype=np.float32)
        return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
