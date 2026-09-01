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

    def build_obs(self, car: CarState, arena: RocketSimArena, out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = np.empty(74, dtype=np.float32)

        # Symmetry multiplier: if Orange (team 1) and symmetric is True, flip X and Y
        inv = -1.0 if (self.symmetric and car.team == 1) else 1.0

        # Car orientation vectors
        fwd = car.get_forward_vector()
        right = car.get_right_vector()
        up = car.get_up_vector()

        if inv == -1.0:
            fx, fy, fz = -fwd[0], -fwd[1], fwd[2]
            rx, ry, rz = -right[0], -right[1], right[2]
            ux, uy, uz = -up[0], -up[1], up[2]
        else:
            fx, fy, fz = fwd[0], fwd[1], fwd[2]
            rx, ry, rz = right[0], right[1], right[2]
            ux, uy, uz = up[0], up[1], up[2]

        # 1. Self Car State (22 features)
        out[0] = (car.pos[0] * inv) / ARENA_EXTENT_X
        out[1] = (car.pos[1] * inv) / ARENA_EXTENT_Y
        out[2] = car.pos[2] / ARENA_HEIGHT_Z
        out[3] = (car.vel[0] * inv) / CAR_MAX_SPEED
        out[4] = (car.vel[1] * inv) / CAR_MAX_SPEED
        out[5] = car.vel[2] / CAR_MAX_SPEED
        out[6], out[7], out[8] = fx, fy, fz
        out[9], out[10], out[11] = rx, ry, rz
        out[12], out[13], out[14] = ux, uy, uz
        out[15] = car.ang_vel[0] * 0.1
        out[16] = car.ang_vel[1] * 0.1
        out[17] = car.ang_vel[2] * 0.1
        out[18] = car.boost / 100.0
        out[19] = 1.0 if car.on_ground else 0.0
        out[20] = 1.0 if car.has_jump else 0.0
        out[21] = 1.0 if car.has_flip else 0.0

        # 2. Ball State (9 features)
        bx, by, bz = arena.ball.pos[0], arena.ball.pos[1], arena.ball.pos[2]
        bvx, bvy, bvz = arena.ball.vel[0], arena.ball.vel[1], arena.ball.vel[2]
        out[22] = (bx * inv) / ARENA_EXTENT_X
        out[23] = (by * inv) / ARENA_EXTENT_Y
        out[24] = bz / ARENA_HEIGHT_Z
        out[25] = (bvx * inv) / BALL_MAX_SPEED
        out[26] = (bvy * inv) / BALL_MAX_SPEED
        out[27] = bvz / BALL_MAX_SPEED
        out[28] = arena.ball.ang_vel[0] * 0.1
        out[29] = arena.ball.ang_vel[1] * 0.1
        out[30] = arena.ball.ang_vel[2] * 0.1

        # 2b. Future Ball Trajectory Prediction (0.5s ahead = 60 ticks @ 120Hz)
        future_ball_pos = arena.get_predicted_ball_pos(60) if hasattr(arena, "get_predicted_ball_pos") else None
        if future_ball_pos is None:
            dt = 0.5
            fpx = bx + bvx * dt
            fpy = by + bvy * dt
            fpz = max(93.0, bz + bvz * dt + 0.5 * (-650.0) * (dt ** 2))
            if abs(fpx) > 4000.0:
                fpx = math.copysign(4000.0 - (abs(fpx) - 4000.0) * 0.6, fpx)
            if abs(fpy) > 5000.0:
                fpy = math.copysign(5000.0 - (abs(fpy) - 5000.0) * 0.6, fpy)
        else:
            fpx, fpy, fpz = future_ball_pos[0], future_ball_pos[1], future_ball_pos[2]

        out[31] = (fpx * inv) / ARENA_EXTENT_X
        out[32] = (fpy * inv) / ARENA_EXTENT_Y
        out[33] = fpz / ARENA_HEIGHT_Z

        # 3. Relative Features in Car Local Frame (16 features)
        dx = (bx - car.pos[0]) * inv
        dy = (by - car.pos[1]) * inv
        dz = bz - car.pos[2]
        out[34] = (dx * fx + dy * fy + dz * fz) / 2000.0
        out[35] = (dx * rx + dy * ry + dz * rz) / 2000.0
        out[36] = (dx * ux + dy * uy + dz * uz) / 2000.0

        fdx = (fpx - car.pos[0]) * inv
        fdy = (fpy - car.pos[1]) * inv
        fdz = fpz - car.pos[2]
        out[37] = (fdx * fx + fdy * fy + fdz * fz) / 2000.0
        out[38] = (fdx * rx + fdy * ry + fdz * rz) / 2000.0
        out[39] = (fdx * ux + fdy * uy + fdz * uz) / 2000.0

        dvx = (bvx - car.vel[0]) * inv
        dvy = (bvy - car.vel[1]) * inv
        dvz = bvz - car.vel[2]
        out[40] = (dvx * fx + dvy * fy + dvz * fz) / CAR_MAX_SPEED
        out[41] = (dvx * rx + dvy * ry + dvz * rz) / CAR_MAX_SPEED
        out[42] = (dvx * ux + dvy * uy + dvz * uz) / CAR_MAX_SPEED
        out[43] = math.sqrt(dx * dx + dy * dy + dz * dz) / 6000.0

        # Goal vectors relative to car in local frame
        c_inv_x, c_inv_y = car.pos[0] * inv, car.pos[1] * inv
        tg_x, tg_y, tg_z = -c_inv_x, ARENA_EXTENT_Y - c_inv_y, (GOAL_HEIGHT * 0.5) - car.pos[2]
        norm_tg = max(1e-4, math.sqrt(tg_x * tg_x + tg_y * tg_y + tg_z * tg_z))
        tg_ux, tg_uy, tg_uz = tg_x / norm_tg, tg_y / norm_tg, tg_z / norm_tg
        out[44] = tg_ux * fx + tg_uy * fy + tg_uz * fz
        out[45] = tg_ux * rx + tg_uy * ry + tg_uz * rz
        out[46] = tg_ux * ux + tg_uy * uy + tg_uz * uz

        dg_x, dg_y, dg_z = -c_inv_x, -ARENA_EXTENT_Y - c_inv_y, (GOAL_HEIGHT * 0.5) - car.pos[2]
        norm_dg = max(1e-4, math.sqrt(dg_x * dg_x + dg_y * dg_y + dg_z * dg_z))
        dg_ux, dg_uy, dg_uz = dg_x / norm_dg, dg_y / norm_dg, dg_z / norm_dg
        out[47] = dg_ux * fx + dg_uy * fy + dg_uz * fz
        out[48] = dg_ux * rx + dg_uy * ry + dg_uz * rz
        out[49] = dg_ux * ux + dg_uy * uy + dg_uz * uz

        # 3b/3c. Threat and kickoff sensors (4 features)
        is_threat, threat_intensity, threat_z = arena.get_shot_threat(car.team) if hasattr(arena, "get_shot_threat") else (False, 0.0, 0.0)
        out[50] = float(threat_intensity)
        out[51] = float(threat_z)
        is_center_ball = bool(abs(bx) < 50.0 and abs(by) < 50.0 and (abs(bvx) + abs(bvy) + abs(bvz)) < 80.0)
        is_first_touch = bool(all(c.ball_touches == 0 for c in arena.cars))
        out[52] = 1.0 if is_center_ball else 0.0
        out[53] = 1.0 if is_first_touch else 0.0

        # 4. Opponents / Other Players (14 features)
        opponents = [c for c in arena.cars if c.team != car.team]
        if opponents:
            opp = opponents[0]
            out[54] = (opp.pos[0] * inv) / ARENA_EXTENT_X
            out[55] = (opp.pos[1] * inv) / ARENA_EXTENT_Y
            out[56] = opp.pos[2] / ARENA_HEIGHT_Z
            out[57] = (opp.vel[0] * inv) / CAR_MAX_SPEED
            out[58] = (opp.vel[1] * inv) / CAR_MAX_SPEED
            out[59] = opp.vel[2] / CAR_MAX_SPEED

            odx = (opp.pos[0] - car.pos[0]) * inv
            ody = (opp.pos[1] - car.pos[1]) * inv
            odz = opp.pos[2] - car.pos[2]
            out[60] = (odx * fx + ody * fy + odz * fz) / 2000.0
            out[61] = (odx * rx + ody * ry + odz * rz) / 2000.0
            out[62] = (odx * ux + ody * uy + odz * uz) / 2000.0

            odvx = (opp.vel[0] - car.vel[0]) * inv
            odvy = (opp.vel[1] - car.vel[1]) * inv
            odvz = opp.vel[2] - car.vel[2]
            out[63] = (odvx * fx + odvy * fy + odvz * fz) / CAR_MAX_SPEED
            out[64] = (odvx * rx + odvy * ry + odvz * rz) / CAR_MAX_SPEED
            out[65] = (odvx * ux + odvy * uy + odvz * uz) / CAR_MAX_SPEED
            out[66] = opp.boost / 100.0
            out[67] = 1.0 if opp.on_ground else 0.0
        else:
            out[54:68] = 0.0

        # 5. Active Boost Pad Spatial Vectors (6 features)
        if hasattr(arena, "_small_pad_pos_3d") and hasattr(arena, "_small_pad_active"):
            sm_act = arena._small_pad_active
            if sm_act.any():
                act_pos = arena._small_pad_pos_3d[sm_act]
                diff = act_pos[:, :2] - car.pos[:2]
                d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
                min_i = int(np.argmin(d2))
                sm_x = (act_pos[min_i, 0] - car.pos[0]) * inv
                sm_y = (act_pos[min_i, 1] - car.pos[1]) * inv
                sm_z = act_pos[min_i, 2] - car.pos[2]
                out[68] = (sm_x * fx + sm_y * fy + sm_z * fz) / 2000.0
                out[69] = (sm_x * rx + sm_y * ry + sm_z * rz) / 2000.0
                out[70] = math.sqrt(d2[min_i]) / 4000.0
            else:
                out[68], out[69], out[70] = 0.0, 0.0, 1.0

            bg_act = arena._big_pad_active
            if bg_act.any():
                act_pos = arena._big_pad_pos_3d[bg_act]
                diff = act_pos[:, :2] - car.pos[:2]
                d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
                min_i = int(np.argmin(d2))
                bg_x = (act_pos[min_i, 0] - car.pos[0]) * inv
                bg_y = (act_pos[min_i, 1] - car.pos[1]) * inv
                bg_z = act_pos[min_i, 2] - car.pos[2]
                out[71] = (bg_x * fx + bg_y * fy + bg_z * fz) / 3000.0
                out[72] = (bg_x * rx + bg_y * ry + bg_z * rz) / 3000.0
                out[73] = math.sqrt(d2[min_i]) / 6000.0
            else:
                out[71], out[72], out[73] = 0.0, 0.0, 1.0
        else:
            out[68:74] = 0.0

        return out
