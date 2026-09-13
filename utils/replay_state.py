"""
Game state a replay does not record, reconstructed from what it does.

A replay carries car and ball kinematics and boost amounts. The observation also reads jump and flip
availability, dodge phase, air and flip timers, supersonic, whether the ball has been touched since
the kickoff, and boost pad state. Left at constructor defaults, a behaviour-cloning sample shows a car
that always holds its jump and flip, an untouched ball and pads that never go down. Together with the
missing opponent, 47 of the 108 features were constant in the cloning set while varying in RocketSim,
so the cloned policy learned its actions under conditions it never meets.

Detectors were calibrated against RocketSim ground truth sampled at the replay rate (30 Hz, random
held controls with frequent jumps and dodges, 216k car-frames):

    pad active / cooldown        exact
    has_flipped (airborne)       85%   (constant baseline 45%)
    has_double_jumped            99%
    is_flipping                  92%
    air and flip timer MAE       ~0.2 s

has_jumped is taken to be "airborne". That was right on 99.6% of airborne frames, better than a
takeoff impulse test, because the jump impulse lands while the surface estimate still reads grounded.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np

from env.physics_engine import BoostPad
from utils.replay_parser import REPLAY_ANG_VEL_SCALE
from utils.surface_contact import is_on_surface

# A dodge shows as a jump in angular velocity or an impulse across the car's up axis within one frame.
DODGE_SPIN_DELTA = 2.0        # rad/s
DODGE_SIDE_IMPULSE = 200.0    # uu/s
DOUBLE_JUMP_IMPULSE = 150.0   # uu/s along up
FLIP_ANIMATION_S = 0.65
SUPERSONIC_SPEED = 2200.0
GRAVITY = np.array([0.0, 0.0, -650.0])

# Kickoff pause and release, detected the way the live bot detects them (bot.py get_output).
KICKOFF_CENTER_DIST = 50.0
KICKOFF_MAX_SPEED = 80.0
RELEASE_SPEED = 100.0
RELEASE_CENTER_DIST = 120.0

SMALL_PAD_COOLDOWN_S = 4.0
BIG_PAD_COOLDOWN_S = 10.0
PAD_PICKUP_RADIUS = 450.0     # 2D, from the pad centre to the midpoint of the car's step
MIN_PICKUP_INCREMENT = 5.0

SURFACE_CHUNK = 100_000


def up_vectors(rot: np.ndarray) -> np.ndarray:
    """Car up axes (M, 3) from (pitch, yaw, roll) rows, in the convention of CarState.get_up_vector."""
    p, y, r = rot[:, 0], rot[:, 1], rot[:, 2]
    cp, sp, cy, sy, cr, sr = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
    return np.stack([-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr], axis=1)


def _last_index(mask: np.ndarray) -> np.ndarray:
    """For every position, the index of the latest True at or before it (-1 if none)."""
    return np.maximum.accumulate(np.where(mask, np.arange(len(mask)), -1))


class ReplayStateReconstructor:
    """Per-frame state for a replay pool (the dict ReplayParser.states_buffer holds)."""

    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data
        n = len(data["ball_pos"])
        self.n = n
        seg = data["segment_id"] if "segment_id" in data else np.zeros(n, dtype=np.int32)
        self.frame_time = data["frame_time"] if "frame_time" in data else np.arange(n) / 30.0
        self.new_segment = np.r_[True, seg[1:] != seg[:-1]] if n else np.zeros(0, dtype=bool)
        self.seg_start = _last_index(self.new_segment)

        # Touched since kickoff: set at a kickoff pause, cleared once the ball leaves the centre.
        bp, bv = data["ball_pos"], data["ball_vel"]
        dist = np.linalg.norm(bp[:, :2], axis=1)
        speed = np.linalg.norm(bv, axis=1)
        pause = (dist < KICKOFF_CENTER_DIST) & (speed < KICKOFF_MAX_SPEED)
        release = (speed > RELEASE_SPEED) | (dist > RELEASE_CENTER_DIST)
        last_pause = _last_index(pause)
        last_release = _last_index(release)
        pause_in_segment = last_pause >= self.seg_start
        self.untouched = pause_in_segment & (last_pause > last_release)
        # Pads reset at every kickoff, so pad history never reaches back past one.
        self.pad_history_start = np.where(pause_in_segment, last_pause, self.seg_start)

        # A replay with one car stores a mirror image of it as the second (ReplayParser); that is
        # not an opponent. The mirrored yaw is c0 + pi unwrapped, which a replicated rotation never is.
        cp, cr = data["car_pos"], data["car_rot"]
        self.synthetic_opponent = (
            np.all(np.abs(cp[:, 1, :2] + cp[:, 0, :2]) < 1e-3, axis=1)
            & (np.abs(cp[:, 1, 2] - cp[:, 0, 2]) < 1e-3)
            & (np.abs(cr[:, 1, 1] - (cr[:, 0, 1] + np.pi)) < 1e-4)
        )

        pads = BoostPad.create_standard_pads()
        self.pad_xy = np.array([p.pos[:2] for p in pads], dtype=np.float64)
        self.pad_big = np.array([p.is_big for p in pads], dtype=bool)
        self._index_pickups()

        self.mechanics = [self._car_mechanics(c) for c in range(2)]

    # ── Pads ────────────────────────────────────────────────────────────────────────────────
    def _index_pickups(self):
        d = self.data
        boost, cp = d["car_boost"], d["car_pos"]
        same = ~self.new_segment[1:]
        frames, want_big, xy = [], [], []
        for c in range(2):
            inc = boost[1:, c] - boost[:-1, c]
            ok = same & (inc >= MIN_PICKUP_INCREMENT)
            if c == 1:
                ok &= ~self.synthetic_opponent[1:]
            k = np.nonzero(ok)[0] + 1
            frames.append(k)
            # A +12 step is a small pad; anything larger, or +12-ish that tops out at 100, is a big one.
            want_big.append((inc[k - 1] > 14.0) | ((boost[k, c] >= 99.5) & (inc[k - 1] > 12.5)))
            xy.append(0.5 * (cp[k, c, :2] + cp[k - 1, c, :2]))
        frames = np.concatenate(frames)
        order = np.argsort(frames, kind="stable")
        self.pickup_frame = frames[order]
        self.pickup_big = np.concatenate(want_big)[order]
        self.pickup_xy = np.concatenate(xy)[order].astype(np.float64)

    def pads_at(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """(is_active, cooldown seconds remaining) for the 34 pads in standard order at frame idx."""
        t_now = float(self.frame_time[idx])
        lo = int(np.searchsorted(self.pickup_frame, self.pad_history_start[idx], "left"))
        hi = int(np.searchsorted(self.pickup_frame, idx, "right"))
        expire = np.full(len(self.pad_xy), -np.inf)
        for e in range(lo, hi):
            t_e = float(self.frame_time[self.pickup_frame[e]])
            if t_e < t_now - BIG_PAD_COOLDOWN_S:
                continue
            cand = np.nonzero((expire <= t_e) & (self.pad_big == self.pickup_big[e]))[0]
            if len(cand) == 0:
                continue
            dist = np.linalg.norm(self.pad_xy[cand] - self.pickup_xy[e], axis=1)
            j = int(dist.argmin())
            if dist[j] < PAD_PICKUP_RADIUS:
                expire[cand[j]] = t_e + (BIG_PAD_COOLDOWN_S if self.pad_big[cand[j]] else SMALL_PAD_COOLDOWN_S)
        remaining = np.maximum(expire - t_now, 0.0)
        return remaining <= 0.0, remaining

    # ── Car mechanics ───────────────────────────────────────────────────────────────────────
    def _car_mechanics(self, c: int) -> Dict[str, np.ndarray]:
        d, n = self.data, self.n
        if "car_time" in d:
            fresh = np.abs(d["frame_time"] - d["car_time"][:, c]) < 1e-3
            t_car = d["car_time"][:, c]
        else:
            fresh = np.ones(n, dtype=bool)
            t_car = self.frame_time
        rows = np.nonzero(fresh)[0]
        m = len(rows)

        # Only frames where the car was replicated carry new kinematics; differences run across those.
        P = d["car_pos"][rows, c].astype(np.float64)
        V = d["car_vel"][rows, c].astype(np.float64)
        AV = d["car_ang_vel"][rows, c].astype(np.float64) / REPLAY_ANG_VEL_SCALE if "car_ang_vel" in d else np.zeros((m, 3))
        up = up_vectors(d["car_rot"][rows, c].astype(np.float64))
        T = t_car[rows].astype(np.float64)

        gnd = np.zeros(m, dtype=bool)
        for s in range(0, m, SURFACE_CHUNK):
            gnd[s:s + SURFACE_CHUNK] = is_on_surface(P[s:s + SURFACE_CHUNK], up[s:s + SURFACE_CHUNK])

        seg_rows = self.seg_start[rows]
        new_seg = np.r_[True, seg_rows[1:] != seg_rows[:-1]] if m else np.zeros(0, dtype=bool)
        # Touchdown and recording boundaries both end whatever the car was doing in the air.
        last_reset = _last_index(gnd | new_seg)
        prev_gnd = np.r_[True, gnd[:-1]]
        airborne_step = ~gnd & ~new_seg & ~prev_gnd

        dt = np.r_[0.0, np.diff(T)]
        up_prev = np.r_[up[:1], up[:-1]]
        dv = V - np.r_[V[:1], V[:-1]] - GRAVITY * dt[:, None]
        dv_up = np.sum(dv * up_prev, axis=1)
        dv_side = np.linalg.norm(dv - dv_up[:, None] * up_prev, axis=1)
        d_spin = np.linalg.norm(AV - np.r_[AV[:1], AV[:-1]], axis=1)

        dodge = airborne_step & ((d_spin > DODGE_SPIN_DELTA) | (dv_side > DODGE_SIDE_IMPULSE))
        double_jump = airborne_step & ~dodge & (dv_up > DOUBLE_JUMP_IMPULSE)
        event = dodge | double_jump
        # Only the first second-press after takeoff counts: a dodge or a double jump spends it.
        last_event_before = np.r_[-1, _last_index(event)[:-1]] if m else np.zeros(0, dtype=int)
        first = event & (last_event_before <= last_reset)
        first_dodge = _last_index(first & dodge)
        first_dj = _last_index(first & double_jump)

        has_flipped = ~gnd & (first_dodge > last_reset)
        has_double_jumped = ~gnd & (first_dj > last_reset)
        fd = np.maximum(first_dodge, 0)
        flip_timer = np.where(has_flipped, T - T[fd] + dt[fd], 0.0)
        air_timer = np.where(gnd, 0.0, T - T[last_reset])
        supersonic = np.linalg.norm(V, axis=1) >= SUPERSONIC_SPEED

        # Frames where the car was not replicated read its last replicated state.
        src = np.cumsum(fresh) - 1
        valid = src >= 0
        src = np.maximum(src, 0)

        def full(x, default):
            if m == 0:
                return np.full(n, default, dtype=np.asarray(default).dtype)
            out = x[src]
            out[~valid] = default
            return out

        return {
            "on_ground": full(gnd, True),
            "has_flipped": full(has_flipped, False),
            "has_double_jumped": full(has_double_jumped, False),
            "flip_timer": full(flip_timer, 0.0),
            "air_timer": full(air_timer, 0.0),
            "is_supersonic": full(supersonic, False),
        }

    def car_fields(self, idx: int, c: int) -> Dict[str, Any]:
        """CarState keyword arguments for car c at frame idx, with the environment's definitions."""
        mech = self.mechanics[c]
        on_ground = bool(mech["on_ground"][idx])
        flipped = bool(mech["has_flipped"][idx])
        double_jumped = bool(mech["has_double_jumped"][idx])
        flip_timer = float(mech["flip_timer"][idx])
        return {
            "on_ground": on_ground,
            # RocketSimArena._sync_from_rsim: has_jump = not has_jumped or on_ground, and airborne is taken as jumped.
            "has_jump": on_ground,
            "has_flip": (not on_ground) and not flipped and not double_jumped,
            "has_double_jumped": double_jumped,
            "is_dodging": flipped and flip_timer <= FLIP_ANIMATION_S,
            "flip_timer": flip_timer,
            "air_timer": float(mech["air_timer"][idx]),
            "is_supersonic": bool(mech["is_supersonic"][idx]),
        }
