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
     1.0, -1.0, -1.0,   # right
    -1.0,  1.0,  1.0,   # up (up.x negated)
     1.0, -1.0, -1.0,   # ang_vel
     1.0,  1.0,  1.0,  1.0,  # boost, on_ground, has_jump, has_flip
    # 2. Ball State (9 features)
    -1.0,  1.0,  1.0,   # ball_pos (pos.x negated)
    -1.0,  1.0,  1.0,   # ball_vel (vel.x negated)
     1.0, -1.0, -1.0,   # ball_ang_vel
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

# Legacy (pre-fix) mirror mask preserved for checkpoint backward compatibility
OBS_LEGACY_MIRROR_MASK_NP = np.array([
    # Copy of the OLD mask before corrections
    -1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,
     1.0, -1.0,  1.0,    # old right (index 11 was 1.0)
    -1.0,  1.0,  1.0,
    -1.0,  1.0, -1.0,    # old ang_vel (indices 15,16 were -1.0, 1.0)
     1.0,  1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,
    -1.0,  1.0, -1.0,    # old ball ang_vel (indices 28,29 were -1.0, 1.0)
    -1.0,  1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0,
     1.0, -1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0,  1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0,  1.0,
     1.0, -1.0,  1.0,
     1.0, -1.0,  1.0
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

        # Car orientation vectors directly from rotation basis without .copy() or cross-product overhead
        rm = car.rot_mat
        if rm is not None:
            fx, fy, fz = float(rm[0, 0]), float(rm[0, 1]), float(rm[0, 2])
            ux, uy, uz = float(rm[2, 0]), float(rm[2, 1]), float(rm[2, 2])
            # True Right vector from fwd x up (In C++ RocketSim basis, row 1 is Left / -Right)
            rx = fy * uz - fz * uy
            ry = fz * ux - fx * uz
            rz = fx * uy - fy * ux
        else:
            f = car.get_forward_vector()
            u = car.get_up_vector()
            r = car.get_right_vector()
            fx, fy, fz = float(f[0]), float(f[1]), float(f[2])
            rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
            ux, uy, uz = float(u[0]), float(u[1]), float(u[2])

        if inv == -1.0:
            fx, fy = -fx, -fy
            rx, ry = -rx, -ry
            ux, uy = -ux, -uy

        cp = car.pos
        cv = car.vel
        ca = car.ang_vel
        cpx, cpy, cpz = float(cp[0]), float(cp[1]), float(cp[2])

        # 1. Self Car State (22 features)
        out[0] = (cpx * inv) / ARENA_EXTENT_X
        out[1] = (cpy * inv) / ARENA_EXTENT_Y
        out[2] = cpz / ARENA_HEIGHT_Z
        out[3] = (float(cv[0]) * inv) / CAR_MAX_SPEED
        out[4] = (float(cv[1]) * inv) / CAR_MAX_SPEED
        out[5] = float(cv[2]) / CAR_MAX_SPEED
        out[6], out[7], out[8] = fx, fy, fz
        out[9], out[10], out[11] = rx, ry, rz
        out[12], out[13], out[14] = ux, uy, uz
        out[15] = float(ca[0]) * inv * 0.1
        out[16] = float(ca[1]) * inv * 0.1
        out[17] = float(ca[2]) * 0.1
        out[18] = car.boost * 0.01
        out[19] = 1.0 if car.on_ground else 0.0
        out[20] = 1.0 if car.has_jump else 0.0
        out[21] = 1.0 if car.has_flip else 0.0

        # 2. Ball State (9 features)
        bp = arena.ball.pos
        bv = arena.ball.vel
        ba = arena.ball.ang_vel
        bx, by, bz = float(bp[0]), float(bp[1]), float(bp[2])
        bvx, bvy, bvz = float(bv[0]), float(bv[1]), float(bv[2])
        out[22] = (bx * inv) / ARENA_EXTENT_X
        out[23] = (by * inv) / ARENA_EXTENT_Y
        out[24] = bz / ARENA_HEIGHT_Z
        out[25] = (bvx * inv) / BALL_MAX_SPEED
        out[26] = (bvy * inv) / BALL_MAX_SPEED
        out[27] = bvz / BALL_MAX_SPEED
        out[28] = float(ba[0]) * inv * 0.1
        out[29] = float(ba[1]) * inv * 0.1
        out[30] = float(ba[2]) * 0.1

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
            fpx, fpy, fpz = float(future_ball_pos[0]), float(future_ball_pos[1]), float(future_ball_pos[2])

        out[31] = (fpx * inv) / ARENA_EXTENT_X
        out[32] = (fpy * inv) / ARENA_EXTENT_Y
        out[33] = fpz / ARENA_HEIGHT_Z

        # 3. Relative Features in Car Local Frame (16 features)
        dx = (bx - cpx) * inv
        dy = (by - cpy) * inv
        dz = bz - cpz
        out[34] = (dx * fx + dy * fy + dz * fz) * 0.0005
        out[35] = (dx * rx + dy * ry + dz * rz) * 0.0005
        out[36] = (dx * ux + dy * uy + dz * uz) * 0.0005

        fdx = (fpx - cpx) * inv
        fdy = (fpy - cpy) * inv
        fdz = fpz - cpz
        out[37] = (fdx * fx + fdy * fy + fdz * fz) * 0.0005
        out[38] = (fdx * rx + fdy * ry + fdz * rz) * 0.0005
        out[39] = (fdx * ux + fdy * uy + fdz * uz) * 0.0005

        dvx = (bvx - float(cv[0])) * inv
        dvy = (bvy - float(cv[1])) * inv
        dvz = bvz - float(cv[2])
        out[40] = (dvx * fx + dvy * fy + dvz * fz) / CAR_MAX_SPEED
        out[41] = (dvx * rx + dvy * ry + dvz * rz) / CAR_MAX_SPEED
        out[42] = (dvx * ux + dvy * uy + dvz * uz) / CAR_MAX_SPEED
        out[43] = math.sqrt(dx * dx + dy * dy + dz * dz) / 6000.0

        # Goal vectors relative to car in local frame
        c_inv_x, c_inv_y = cpx * inv, cpy * inv
        tg_x, tg_y, tg_z = -c_inv_x, ARENA_EXTENT_Y - c_inv_y, (GOAL_HEIGHT * 0.5) - cpz
        norm_tg = 1.0 / max(1e-4, math.sqrt(tg_x * tg_x + tg_y * tg_y + tg_z * tg_z))
        tg_ux, tg_uy, tg_uz = tg_x * norm_tg, tg_y * norm_tg, tg_z * norm_tg
        out[44] = tg_ux * fx + tg_uy * fy + tg_uz * fz
        out[45] = tg_ux * rx + tg_uy * ry + tg_uz * rz
        out[46] = tg_ux * ux + tg_uy * uy + tg_uz * uz

        dg_x, dg_y, dg_z = -c_inv_x, -ARENA_EXTENT_Y - c_inv_y, (GOAL_HEIGHT * 0.5) - cpz
        norm_dg = 1.0 / max(1e-4, math.sqrt(dg_x * dg_x + dg_y * dg_y + dg_z * dg_z))
        dg_ux, dg_uy, dg_uz = dg_x * norm_dg, dg_y * norm_dg, dg_z * norm_dg
        out[47] = dg_ux * fx + dg_uy * fy + dg_uz * fz
        out[48] = dg_ux * rx + dg_uy * ry + dg_uz * rz
        out[49] = dg_ux * ux + dg_uy * uy + dg_uz * uz

        # 3b/3c. Threat and kickoff sensors (4 features)
        is_threat, threat_intensity, threat_z = arena.get_shot_threat(car.team) if hasattr(arena, "get_shot_threat") else (False, 0.0, 0.0)
        out[50] = float(threat_intensity)
        out[51] = float(threat_z)
        is_center_ball = bool(abs(bx) < 50.0 and abs(by) < 50.0 and (abs(bvx) + abs(bvy) + abs(bvz)) < 80.0)
        is_first_touch = bool(arena.cars[0].ball_touches == 0 and arena.cars[1].ball_touches == 0) if len(arena.cars) >= 2 else bool(arena.cars[0].ball_touches == 0)
        out[52] = 1.0 if (is_center_ball and is_first_touch) else 0.0
        out[53] = 1.0 if is_first_touch else 0.0

        # 4. Opponents / Other Players (14 features)
        opponents = [c for c in arena.cars if c.team != car.team]
        if opponents:
            opp = opponents[0]
            op = opp.pos
            ov = opp.vel
            opx, opy, opz = float(op[0]), float(op[1]), float(op[2])
            ovx, ovy, ovz = float(ov[0]), float(ov[1]), float(ov[2])
            out[54] = (opx * inv) / ARENA_EXTENT_X
            out[55] = (opy * inv) / ARENA_EXTENT_Y
            out[56] = opz / ARENA_HEIGHT_Z
            out[57] = (ovx * inv) / CAR_MAX_SPEED
            out[58] = (ovy * inv) / CAR_MAX_SPEED
            out[59] = ovz / CAR_MAX_SPEED

            odx = (opx - cpx) * inv
            ody = (opy - cpy) * inv
            odz = opz - cpz
            out[60] = (odx * fx + ody * fy + odz * fz) * 0.0005
            out[61] = (odx * rx + ody * ry + odz * rz) * 0.0005
            out[62] = (odx * ux + ody * uy + odz * uz) * 0.0005

            odvx = (ovx - float(cv[0])) * inv
            odvy = (ovy - float(cv[1])) * inv
            odvz = ovz - float(cv[2])
            out[63] = (odvx * fx + odvy * fy + odvz * fz) / CAR_MAX_SPEED
            out[64] = (odvx * rx + odvy * ry + odvz * rz) / CAR_MAX_SPEED
            out[65] = (odvx * ux + odvy * uy + odvz * uz) / CAR_MAX_SPEED
            out[66] = opp.boost * 0.01
            out[67] = 1.0 if opp.on_ground else 0.0
        else:
            out[54:68] = 0.0

        # 5. Fast Zero-Allocation Boost Pad Spatial Vectors (6 features)
        if hasattr(arena, "_small_pad_pos_3d") and hasattr(arena, "_small_pad_active"):
            sm_act = arena._small_pad_active
            sm_poses = arena._small_pad_pos_3d
            min_sm_d2 = 1e12
            min_sm_idx = -1
            for p_idx in range(len(sm_act)):
                if sm_act[p_idx]:
                    px = float(sm_poses[p_idx, 0])
                    py = float(sm_poses[p_idx, 1])
                    dx_p = px - cpx
                    dy_p = py - cpy
                    d2 = dx_p * dx_p + dy_p * dy_p
                    if d2 < min_sm_d2:
                        min_sm_d2 = d2
                        min_sm_idx = p_idx

            if min_sm_idx >= 0:
                sm_x = (float(sm_poses[min_sm_idx, 0]) - cpx) * inv
                sm_y = (float(sm_poses[min_sm_idx, 1]) - cpy) * inv
                sm_z = float(sm_poses[min_sm_idx, 2]) - cpz
                out[68] = (sm_x * fx + sm_y * fy + sm_z * fz) * 0.0005
                out[69] = (sm_x * rx + sm_y * ry + sm_z * rz) * 0.0005
                out[70] = math.sqrt(min_sm_d2) * 0.00025
            else:
                out[68], out[69], out[70] = 0.0, 0.0, 1.0

            bg_act = arena._big_pad_active
            bg_poses = arena._big_pad_pos_3d
            min_bg_d2 = 1e12
            min_bg_idx = -1
            for p_idx in range(len(bg_act)):
                if bg_act[p_idx]:
                    px = float(bg_poses[p_idx, 0])
                    py = float(bg_poses[p_idx, 1])
                    dx_p = px - cpx
                    dy_p = py - cpy
                    d2 = dx_p * dx_p + dy_p * dy_p
                    if d2 < min_bg_d2:
                        min_bg_d2 = d2
                        min_bg_idx = p_idx

            if min_bg_idx >= 0:
                bg_x = (float(bg_poses[min_bg_idx, 0]) - cpx) * inv
                bg_y = (float(bg_poses[min_bg_idx, 1]) - cpy) * inv
                bg_z = float(bg_poses[min_bg_idx, 2]) - cpz
                out[71] = (bg_x * fx + bg_y * fy + bg_z * fz) * (1.0 / 3000.0)
                out[72] = (bg_x * rx + bg_y * ry + bg_z * rz) * (1.0 / 3000.0)
                out[73] = math.sqrt(min_bg_d2) * (1.0 / 6000.0)
            else:
                out[71], out[72], out[73] = 0.0, 0.0, 1.0
        else:
            out[68:74] = 0.0

        return out
