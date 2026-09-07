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

OBS_DIM = 94

# 94-Dimensional Left-Right (X -> -X) Observation Symmetry Reflection Mask
# Multiplies features by -1.0 for lateral X components, roll, yaw, and relative right offsets
# Features 80..93 (pad active flags and cooldown timers) are strictly positive scalars (+1.0)
OBS_MIRROR_MASK_NP = np.array([
    # 1. Self Car State (22 features: 0..21)
    -1.0,  1.0,  1.0,   # car_pos (pos.x negated) [0..2]
    -1.0,  1.0,  1.0,   # car_vel (vel.x negated) [3..5]
    -1.0,  1.0,  1.0,   # fwd (fwd.x negated) [6..8]
     1.0, -1.0, -1.0,   # right [9..11]
    -1.0,  1.0,  1.0,   # up (up.x negated) [12..14]
     1.0, -1.0, -1.0,   # ang_vel [15..17]
     1.0,  1.0,  1.0,  1.0,  # boost, on_ground, has_jump, has_flip [18..21]
    # 2. Ball State (9 features: 22..30)
    -1.0,  1.0,  1.0,   # ball_pos (pos.x negated) [22..24]
    -1.0,  1.0,  1.0,   # ball_vel (vel.x negated) [25..27]
     1.0, -1.0, -1.0,   # ball_ang_vel [28..30]
    # 2b. Future Ball Trajectory Prediction (6 features: 31..36)
    -1.0,  1.0,  1.0,   # future_ball_pos_0_5s (pos.x negated) [31..33]
    -1.0,  1.0,  1.0,   # future_ball_pos_1_5s (pos.x negated) [34..36]
    # 3. Relative Features in Car Local Frame (19 features: 37..55)
     1.0, -1.0,  1.0,   # local_ball_pos (right offset negated) [37..39]
     1.0, -1.0,  1.0,   # local_future_ball_pos_0_5s (right offset negated) [40..42]
     1.0, -1.0,  1.0,   # local_future_ball_pos_1_5s (right offset negated) [43..45]
     1.0, -1.0,  1.0,   # local_ball_vel (right vel negated) [46..48]
     1.0,             # dist_ball [49]
     1.0, -1.0,  1.0,   # local_target_goal (right offset negated) [50..52]
     1.0, -1.0,  1.0,   # local_defend_goal (right offset negated) [53..55]
    # 3b/3c. Sensors (4 features: 56..59)
     1.0,  1.0,  1.0,  1.0,  # threat_intensity, threat_z, is_kickoff, is_first_touch [56..59]
    # 4. Opponent State (14 features: 60..73)
    -1.0,  1.0,  1.0,   # opp_pos (pos.x negated) [60..62]
    -1.0,  1.0,  1.0,   # opp_vel (vel.x negated) [63..65]
     1.0, -1.0,  1.0,   # local_opp_pos (right offset negated) [66..68]
     1.0, -1.0,  1.0,   # local_opp_vel (right vel negated) [69..71]
     1.0,  1.0,        # opp_boost, opp_on_ground [72..73]
    # 5. Boost Pad Spatial Vectors (6 features: 74..79)
     1.0, -1.0,  1.0,   # nearest small pad (fwd, right negated, dist) [74..76]
     1.0, -1.0,  1.0,   # nearest big orb (fwd, right negated, dist) [77..79]
    # 6. Nearest Pad Cooldown Timers (2 features: 80..81)
     1.0,                # nearest small pad cooldown timer [80]
     1.0,                # nearest big pad cooldown timer [81]
    # 7. Strategic 6 Big Orbs in Symmetric Team Perspective (12 features: 82..93)
     1.0,  1.0,         # Defending Left Corner Orb (is_active, cooldown) [82..83]
     1.0,  1.0,         # Defending Right Corner Orb (is_active, cooldown) [84..85]
     1.0,  1.0,         # Midfield Left Orb (is_active, cooldown) [86..87]
     1.0,  1.0,         # Midfield Right Orb (is_active, cooldown) [88..89]
     1.0,  1.0,         # Attacking Left Corner Orb (is_active, cooldown) [90..91]
     1.0,  1.0          # Attacking Right Corner Orb (is_active, cooldown) [92..93]
], dtype=np.float32)

# Bilateral Permutation Indices for Mirror Reflection across X=0:
# Swaps Left <-> Right orb pairs:
# Defending Left [82, 83] <-> Defending Right [84, 85]
# Midfield Left [86, 87] <-> Midfield Right [88, 89]
# Attacking Left [90, 91] <-> Attacking Right [92, 93]
OBS_MIRROR_INDICES_NP = np.arange(OBS_DIM, dtype=int)
OBS_MIRROR_INDICES_NP[[82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]] = [
    84, 85, 82, 83, 88, 89, 86, 87, 92, 93, 90, 91
]

# Legacy mirror mask extended to 94 dimensions for backward compatibility
OBS_LEGACY_MIRROR_MASK_NP = np.array([
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
    -1.0,  1.0,  1.0,    # future_ball_pos_0_5s
    -1.0,  1.0,  1.0,    # future_ball_pos_1_5s
     1.0, -1.0,  1.0,    # local_ball_pos
     1.0, -1.0,  1.0,    # local_future_ball_pos_0_5s
     1.0, -1.0,  1.0,    # local_future_ball_pos_1_5s
     1.0, -1.0,  1.0,    # local_ball_vel
     1.0,             # dist_ball
     1.0, -1.0,  1.0,    # local_target_goal
     1.0, -1.0,  1.0,    # local_defend_goal
     1.0,  1.0,  1.0,  1.0,  # sensors
    -1.0,  1.0,  1.0,    # opp_pos
    -1.0,  1.0,  1.0,    # opp_vel
     1.0, -1.0,  1.0,    # local_opp_pos
     1.0, -1.0,  1.0,    # local_opp_vel
     1.0,  1.0,        # opp flags
     1.0, -1.0,  1.0,    # small pad
     1.0, -1.0,  1.0,    # big pad
     1.0,  1.0,        # timers (80..81)
     1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0 # big pads (82..93)
], dtype=np.float32)

# 8-Dimensional Action Reflection Mask: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
ACT_MIRROR_MASK_NP = np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    """Mirrors a single or batched numpy observation across the X=0 plane with bilateral pair swaps."""
    if obs.shape[-1] == OBS_DIM:
        return (obs * OBS_MIRROR_MASK_NP)[..., OBS_MIRROR_INDICES_NP]
    else:
        return obs * OBS_MIRROR_MASK_NP[:obs.shape[-1]]


def mirror_act(act: np.ndarray) -> np.ndarray:
    """Mirrors continuous action vectors (negates steer, yaw, and roll)."""
    return act * ACT_MIRROR_MASK_NP


class DefaultObservationBuilder:
    """
    Standard RLGym-style observation builder with local coordinate transformations and symmetric team inversion.
    """
    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric
        self.obs_dim = OBS_DIM

    def build_obs(self, car: CarState, arena: RocketSimArena, out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = np.empty(self.obs_dim, dtype=np.float32)

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

        # 1. Self Car State (22 features: 0..21)
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

        # 2. Ball State (9 features: 22..30)
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

        # 2b. Short-Term Ball Trajectory Prediction (0.5s ahead = 60 ticks @ 120Hz)
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

        # 2c. Medium-Term Ball Trajectory Prediction (1.5s ahead = 180 ticks @ 120Hz)
        future_ball_pos_180 = arena.get_predicted_ball_pos(180) if hasattr(arena, "get_predicted_ball_pos") else None
        if future_ball_pos_180 is None:
            dt2 = 1.5
            fpx2 = bx + bvx * dt2
            fpy2 = by + bvy * dt2
            fpz2 = max(93.0, bz + bvz * dt2 + 0.5 * (-650.0) * (dt2 ** 2))
            if abs(fpx2) > 4000.0:
                fpx2 = math.copysign(4000.0 - (abs(fpx2) - 4000.0) * 0.6, fpx2)
            if abs(fpy2) > 5000.0:
                fpy2 = math.copysign(5000.0 - (abs(fpy2) - 5000.0) * 0.6, fpy2)
        else:
            fpx2, fpy2, fpz2 = float(future_ball_pos_180[0]), float(future_ball_pos_180[1]), float(future_ball_pos_180[2])

        out[34] = (fpx2 * inv) / ARENA_EXTENT_X
        out[35] = (fpy2 * inv) / ARENA_EXTENT_Y
        out[36] = fpz2 / ARENA_HEIGHT_Z

        # 3. Relative Features in Car Local Frame (19 features: 37..55)
        dx = (bx - cpx) * inv
        dy = (by - cpy) * inv
        dz = bz - cpz
        out[37] = (dx * fx + dy * fy + dz * fz) * 0.0005
        out[38] = (dx * rx + dy * ry + dz * rz) * 0.0005
        out[39] = (dx * ux + dy * uy + dz * uz) * 0.0005

        fdx = (fpx - cpx) * inv
        fdy = (fpy - cpy) * inv
        fdz = fpz - cpz
        out[40] = (fdx * fx + fdy * fy + fdz * fz) * 0.0005
        out[41] = (fdx * rx + fdy * ry + fdz * rz) * 0.0005
        out[42] = (fdx * ux + fdy * uy + fdz * uz) * 0.0005

        fdx2 = (fpx2 - cpx) * inv
        fdy2 = (fpy2 - cpy) * inv
        fdz2 = fpz2 - cpz
        out[43] = (fdx2 * fx + fdy2 * fy + fdz2 * fz) * 0.0005
        out[44] = (fdx2 * rx + fdy2 * ry + fdz2 * rz) * 0.0005
        out[45] = (fdx2 * ux + fdy2 * uy + fdz2 * uz) * 0.0005

        dvx = (bvx - float(cv[0])) * inv
        dvy = (bvy - float(cv[1])) * inv
        dvz = bvz - float(cv[2])
        out[46] = (dvx * fx + dvy * fy + dvz * fz) / CAR_MAX_SPEED
        out[47] = (dvx * rx + dvy * ry + dvz * rz) / CAR_MAX_SPEED
        out[48] = (dvx * ux + dvy * uy + dvz * uz) / CAR_MAX_SPEED
        out[49] = math.sqrt(dx * dx + dy * dy + dz * dz) / 6000.0

        # Goal vectors relative to car in local frame
        c_inv_x, c_inv_y = cpx * inv, cpy * inv
        tg_x, tg_y, tg_z = -c_inv_x, ARENA_EXTENT_Y - c_inv_y, (GOAL_HEIGHT * 0.5) - cpz
        norm_tg = 1.0 / max(1e-4, math.sqrt(tg_x * tg_x + tg_y * tg_y + tg_z * tg_z))
        tg_ux, tg_uy, tg_uz = tg_x * norm_tg, tg_y * norm_tg, tg_z * norm_tg
        out[50] = tg_ux * fx + tg_uy * fy + tg_uz * fz
        out[51] = tg_ux * rx + tg_uy * ry + tg_uz * rz
        out[52] = tg_ux * ux + tg_uy * uy + tg_uz * uz

        dg_x, dg_y, dg_z = -c_inv_x, -ARENA_EXTENT_Y - c_inv_y, (GOAL_HEIGHT * 0.5) - cpz
        norm_dg = 1.0 / max(1e-4, math.sqrt(dg_x * dg_x + dg_y * dg_y + dg_z * dg_z))
        dg_ux, dg_uy, dg_uz = dg_x * norm_dg, dg_y * norm_dg, dg_z * norm_dg
        out[53] = dg_ux * fx + dg_uy * fy + dg_uz * fz
        out[54] = dg_ux * rx + dg_uy * ry + dg_uz * rz
        out[55] = dg_ux * ux + dg_uy * uy + dg_uz * uz

        # 3b/3c. Threat and kickoff sensors (4 features: 56..59)
        is_threat, threat_intensity, threat_z = arena.get_shot_threat(car.team) if hasattr(arena, "get_shot_threat") else (False, 0.0, 0.0)
        out[56] = float(threat_intensity)
        out[57] = float(threat_z)
        is_center_ball = bool(abs(bx) < 50.0 and abs(by) < 50.0 and (abs(bvx) + abs(bvy) + abs(bvz)) < 80.0)
        is_first_touch = bool(arena.cars[0].ball_touches == 0 and arena.cars[1].ball_touches == 0) if len(arena.cars) >= 2 else bool(arena.cars[0].ball_touches == 0)
        out[58] = 1.0 if (is_center_ball and is_first_touch) else 0.0
        out[59] = 1.0 if is_first_touch else 0.0

        # 4. Opponents / Other Players (14 features: 60..73)
        opponents = [c for c in arena.cars if c.team != car.team]
        if opponents:
            opp = opponents[0]
            op = opp.pos
            ov = opp.vel
            opx, opy, opz = float(op[0]), float(op[1]), float(op[2])
            ovx, ovy, ovz = float(ov[0]), float(ov[1]), float(ov[2])
            out[60] = (opx * inv) / ARENA_EXTENT_X
            out[61] = (opy * inv) / ARENA_EXTENT_Y
            out[62] = opz / ARENA_HEIGHT_Z
            out[63] = (ovx * inv) / CAR_MAX_SPEED
            out[64] = (ovy * inv) / CAR_MAX_SPEED
            out[65] = ovz / CAR_MAX_SPEED

            odx = (opx - cpx) * inv
            ody = (opy - cpy) * inv
            odz = opz - cpz
            out[66] = (odx * fx + ody * fy + odz * fz) * 0.0005
            out[67] = (odx * rx + ody * ry + odz * rz) * 0.0005
            out[68] = (odx * ux + ody * uy + odz * uz) * 0.0005

            odvx = (ovx - float(cv[0])) * inv
            odvy = (ovy - float(cv[1])) * inv
            odvz = ovz - float(cv[2])
            out[69] = (odvx * fx + odvy * fy + odvz * fz) / CAR_MAX_SPEED
            out[70] = (odvx * rx + odvy * ry + odvz * rz) / CAR_MAX_SPEED
            out[71] = (odvx * ux + odvy * uy + odvz * uz) / CAR_MAX_SPEED
            out[72] = opp.boost * 0.01
            out[73] = 1.0 if opp.on_ground else 0.0
        else:
            out[60:74] = 0.0

        # 5. Fast Zero-Allocation Boost Pad Spatial Vectors (6 features: 74..79)
        min_sm_idx = -1
        min_bg_idx = -1
        if hasattr(arena, "_small_pad_pos_3d") and hasattr(arena, "_small_pad_active"):
            sm_act = arena._small_pad_active
            sm_poses = arena._small_pad_pos_3d
            min_sm_d2 = 1e12
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
                out[74] = (sm_x * fx + sm_y * fy + sm_z * fz) * 0.0005
                out[75] = (sm_x * rx + sm_y * ry + sm_z * rz) * 0.0005
                out[76] = math.sqrt(min_sm_d2) * 0.00025
            else:
                out[74], out[75], out[76] = 0.0, 0.0, 1.0

            bg_act = arena._big_pad_active
            bg_poses = arena._big_pad_pos_3d
            min_bg_d2 = 1e12
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
                out[77] = (bg_x * fx + bg_y * fy + bg_z * fz) * (1.0 / 3000.0)
                out[78] = (bg_x * rx + bg_y * ry + bg_z * rz) * (1.0 / 3000.0)
                out[79] = math.sqrt(min_bg_d2) * (1.0 / 6000.0)
            else:
                out[77], out[78], out[79] = 0.0, 0.0, 1.0
        else:
            out[74:80] = 0.0

        # 6. Nearest Pad Cooldown Timers (2 features: 80..81)
        pads = getattr(arena, "boost_pads", None)
        num_pads = len(pads) if pads is not None else 0

        if min_sm_idx >= 0 and hasattr(arena, "_sm_pad_indices") and min_sm_idx < len(arena._sm_pad_indices):
            sm_pad_idx = arena._sm_pad_indices[min_sm_idx]
            if sm_pad_idx < num_pads:
                out[80] = min(1.0, max(0.0, getattr(pads[sm_pad_idx], "cooldown_timer", 0.0) / 4.0))
            else:
                out[80] = 0.0
        else:
            out[80] = 0.0

        if min_bg_idx >= 0 and hasattr(arena, "_bg_pad_indices") and min_bg_idx < len(arena._bg_pad_indices):
            bg_pad_idx = arena._bg_pad_indices[min_bg_idx]
            if bg_pad_idx < num_pads:
                out[81] = min(1.0, max(0.0, getattr(pads[bg_pad_idx], "cooldown_timer", 0.0) / 10.0))
            else:
                out[81] = 0.0
        else:
            out[81] = 0.0

        # 7. Strategic 6 Big Orbs in Symmetric Team Perspective (12 features: 82..93)
        # Standard RocketSim indices: [3, 4, 15, 18, 29, 30]
        # Team 0 (Blue): [3: DefL, 4: DefR, 15: MidL, 18: MidR, 29: AttL, 30: AttR]
        # Team 1 (Orange inverted): [30: DefL, 29: DefR, 18: MidL, 15: MidR, 4: AttL, 3: AttR]
        if self.symmetric and car.team == 1:
            big_pad_slots = (30, 29, 18, 15, 4, 3)
        else:
            big_pad_slots = (3, 4, 15, 18, 29, 30)

        for k, p_idx in enumerate(big_pad_slots):
            base_idx = 82 + 2 * k
            if p_idx < num_pads:
                pad = pads[p_idx]
                out[base_idx] = 1.0 if pad.is_active else 0.0
                out[base_idx + 1] = min(1.0, max(0.0, getattr(pad, "cooldown_timer", 0.0) / 10.0))
            else:
                out[base_idx] = 1.0
                out[base_idx + 1] = 0.0

        return out
