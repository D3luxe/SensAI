"""
Reward v7's replay starts: prune dead frames, tag the rest by situation, sample by tag, mirror.

docs/reward_v7_replay_spec.md. v7's reward is v6's; what changes is which replay frames an episode
may start from. Up to v6 a replay start was a uniform draw over every frame in the pool, and 30% of
the pool is not live play (kickoff countdowns, the ball sitting in a net after a goal, cars whose
last replicated position is stale). v7 drops those frames, tags each kept frame with the one
situation it shows, draws a tag by weight and then a frame uniformly within it, and mirrors the
result.

  prune_mask(pool, prune)           True where a frame may be a start
  tag_frames(pool)                  the situation of each frame (TAGS), exclusive, in priority order
  build_index(pool, prune)          kept frame indices grouped by tag, and the count per tag
  load_or_build_index(...)          the same, cached beside the pool's memory-mapped store
  mirror_x / swap_teams             exact symmetries of the pitch, applied to a sampled state
  ReplaySampler                     draws a start state from the index
  pool_fingerprint / check_replay_pool
                                    the pool is part of v7's frozen identity (R1)

The tag thresholds live here, so they are part of v7's code identity. The prune thresholds, the tag
weights and the mirroring probabilities are settings (the snapshot's `replay_sampling` block).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

# Exclusive tags, highest priority first: a frame takes the first tag whose rule it matches.
TAGS: Tuple[str, ...] = ("carry_contested", "carry", "aerial", "challenge", "goal_threat", "open_play")
PRUNED = -1

# Tag rules (uu, uu/s). A car "carries" when the ball sits on its roof: horizontally within 150 of
# the car, 110-260 up, moving with it (relative speed under 400), and the car on the ground.
CARRY_XY = 150.0
CARRY_Z = (110.0, 260.0)
CARRY_REL_SPEED = 400.0
CARRY_CAR_Z = 50.0
CONTEST_DIST = 1000.0          # both cars this close to the ball: a challenge
AERIAL_BALL_Z = 300.0          # ball this high, with a car airborne (above AERIAL_CAR_Z) within AERIAL_DIST
AERIAL_CAR_Z = 200.0
AERIAL_DIST = 800.0
THIRD_Y = 1700.0               # final third: |y| beyond this
THREAT_SPEED = 500.0           # ...with the ball moving at that goal faster than this

DEFAULT_PRUNE: Dict[str, float] = {
    "countdown_radius": 5.0,       # ball within this of centre and slower than countdown_speed
    "countdown_speed": 1.0,
    "goal_line_y": 5120.0,         # ball beyond a goal line: the frames after a goal
    "stale_car_update_s": 0.5,     # a car's last replicated position older than this
    "spawn_radius": 1.0,           # a car at (0, 0): demolished, or not replicated yet
}

CHUNK = 500_000                # frames per vectorised pass, to bound temporary memory


def _prune_settings(prune: Optional[Mapping[str, float]]) -> Dict[str, float]:
    out = dict(DEFAULT_PRUNE)
    out.update({k: float(v) for k, v in (prune or {}).items()})
    return out


def prune_mask(pool: Mapping[str, np.ndarray], prune: Optional[Mapping[str, float]] = None,
               lo: int = 0, hi: Optional[int] = None) -> np.ndarray:
    """True for frames [lo, hi) that may be a start. Timing rules apply only to pools that carry timing."""
    p = _prune_settings(prune)
    hi = len(pool["ball_pos"]) if hi is None else hi
    bp = np.asarray(pool["ball_pos"][lo:hi], dtype=np.float32)
    bv = np.asarray(pool["ball_vel"][lo:hi], dtype=np.float32)
    cp = np.asarray(pool["car_pos"][lo:hi], dtype=np.float32)
    countdown = (np.linalg.norm(bp[:, :2], axis=1) < p["countdown_radius"]) & \
                (np.linalg.norm(bv, axis=1) < p["countdown_speed"])
    in_net = np.abs(bp[:, 1]) > p["goal_line_y"]
    spawn = (np.linalg.norm(cp[:, :, :2], axis=2) < p["spawn_radius"]).any(axis=1)
    drop = countdown | in_net | spawn
    if "frame_time" in pool and "car_time" in pool:
        lag = np.asarray(pool["frame_time"][lo:hi])[:, None] - np.asarray(pool["car_time"][lo:hi])
        drop |= (lag > p["stale_car_update_s"]).any(axis=1)
    return ~drop


def tag_frames(pool: Mapping[str, np.ndarray], lo: int = 0, hi: Optional[int] = None) -> np.ndarray:
    """The TAGS index of each frame in [lo, hi), ignoring pruning."""
    hi = len(pool["ball_pos"]) if hi is None else hi
    bp = np.asarray(pool["ball_pos"][lo:hi], dtype=np.float32)
    bv = np.asarray(pool["ball_vel"][lo:hi], dtype=np.float32)
    cp = np.asarray(pool["car_pos"][lo:hi], dtype=np.float32)
    cv = np.asarray(pool["car_vel"][lo:hi], dtype=np.float32)
    bz = bp[:, 2]
    dist = np.linalg.norm(cp - bp[:, None, :], axis=2)
    xy = np.linalg.norm(cp[:, :, :2] - bp[:, None, :2], axis=2)
    rel = np.linalg.norm(cv - bv[:, None, :], axis=2)
    carrying = ((xy < CARRY_XY) & (bz[:, None] > CARRY_Z[0]) & (bz[:, None] < CARRY_Z[1])
                & (rel < CARRY_REL_SPEED) & (cp[:, :, 2] < CARRY_CAR_Z)).any(axis=1)
    contested = (dist < CONTEST_DIST).all(axis=1)
    aerial = (bz > AERIAL_BALL_Z) & ((dist < AERIAL_DIST) & (cp[:, :, 2] > AERIAL_CAR_Z)).any(axis=1)
    threat = ((bp[:, 1] > THIRD_Y) & (bv[:, 1] > THREAT_SPEED)) | ((bp[:, 1] < -THIRD_Y) & (bv[:, 1] < -THREAT_SPEED))

    tags = np.full(len(bp), TAGS.index("open_play"), dtype=np.int8)
    # Lowest priority first, so each later rule overwrites: the first match in TAGS order wins
    for name, mask in (("goal_threat", threat), ("challenge", contested), ("aerial", aerial),
                       ("carry", carrying), ("carry_contested", carrying & contested)):
        tags[mask] = TAGS.index(name)
    return tags


def build_index(pool: Mapping[str, np.ndarray], prune: Optional[Mapping[str, float]] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    (order, counts): the kept frames' indices grouped by tag in TAGS order (int32, ascending within
    a tag), and how many frames each tag has. Tag t's frames are order[sum(counts[:t]):][:counts[t]].
    """
    n = len(pool["ball_pos"])
    tags = np.empty(n, dtype=np.int8)
    for lo in range(0, n, CHUNK):
        hi = min(n, lo + CHUNK)
        t = tag_frames(pool, lo, hi)
        t[~prune_mask(pool, prune, lo, hi)] = PRUNED
        tags[lo:hi] = t
    kept = np.flatnonzero(tags != PRUNED)
    order = kept[np.argsort(tags[kept], kind="stable")].astype(np.int32)
    counts = np.array([int(np.sum(tags == i)) for i in range(len(TAGS))], dtype=np.int64)
    return order, counts


def index_key(prune: Optional[Mapping[str, float]] = None) -> str:
    """Names a cached index: changes with the prune settings or this file's tag rules."""
    with open(os.path.abspath(__file__), "rb") as fh:
        code = fh.read().replace(b"\r\n", b"\n")
    blob = json.dumps(_prune_settings(prune), sort_keys=True).encode() + code
    return hashlib.sha256(blob).hexdigest()[:12]


def load_or_build_index(pool: Mapping[str, np.ndarray], store_dir: Optional[str],
                        prune: Optional[Mapping[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    build_index, cached in the pool's store directory when it has one. The order array is
    memory-mapped, so every process shares one copy. The counts file is written last and marks the
    cache complete; concurrent builders each write their own temporary files and the first rename wins.
    """
    if not store_dir:
        return build_index(pool, prune)
    key = index_key(prune)
    order_path = os.path.join(store_dir, f"v7_order_{key}.npy")
    counts_path = os.path.join(store_dir, f"v7_counts_{key}.npy")
    if os.path.exists(counts_path) and os.path.exists(order_path):
        return np.load(order_path, mmap_mode="r"), np.load(counts_path)
    order, counts = build_index(pool, prune)
    tag = uuid.uuid4().hex[:8]
    for path, arr in ((order_path, order), (counts_path, counts)):
        tmp = f"{path}.{tag}.tmp.npy"
        try:
            np.save(tmp, arr)
            os.replace(tmp, path)
        except OSError:
            # Another process got there first and has it mapped (Windows will not replace a mapped
            # file); its copy is identical, since the key covers everything that decides the index
            try:
                os.remove(tmp)
            except OSError:
                pass
    if os.path.exists(order_path):
        return np.load(order_path, mmap_mode="r"), counts
    return order, counts


# ---- Mirroring ------------------------------------------------------------------------------

def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def mirror_x(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Left-right reflection, x -> -x. A car's forward vector (cos p cos y, cos p sin y, sin p) maps to
    yaw' = pi - yaw at the same pitch; a reflection reverses handedness, so roll' = -roll.
    """
    out = {k: np.array(v, dtype=np.float32, copy=True) for k, v in state.items()}
    for k in ("ball_pos", "ball_vel"):
        out[k][0] = -out[k][0]
    for k in ("car_pos", "car_vel"):
        out[k][:, 0] = -out[k][:, 0]
    out["car_rot"][:, 1] = _wrap(math.pi - out["car_rot"][:, 1])
    out["car_rot"][:, 2] = -out["car_rot"][:, 2]
    return out


def swap_teams(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    The two sides exchanged: a half turn about the vertical through the centre spot (x, y -> -x, -y;
    yaw + pi; pitch and roll unchanged, since a rotation keeps handedness), then the cars swap
    places, so car 0 plays what the replay's orange player faced.
    """
    out = {k: np.array(v, dtype=np.float32, copy=True) for k, v in state.items()}
    for k in ("ball_pos", "ball_vel"):
        out[k][:2] = -out[k][:2]
    for k in ("car_pos", "car_vel"):
        out[k][:, :2] = -out[k][:, :2]
    out["car_rot"][:, 1] = _wrap(out["car_rot"][:, 1] + math.pi)
    for k in ("car_pos", "car_vel", "car_rot", "car_boost"):
        out[k] = out[k][::-1].copy()
    return out


# ---- Sampling -------------------------------------------------------------------------------

class ReplaySampler:
    """
    Draws replay start states for v7: a tag by weight (renormalised over the tags that have frames),
    a frame uniformly within it, then each mirror with its probability. Uses the `random` module, as
    every other state setter does.
    """

    def __init__(self, pool: Mapping[str, np.ndarray], order: np.ndarray, counts: np.ndarray,
                 settings: Mapping[str, Any]):
        self.pool = pool
        self.order = order
        self.counts = np.asarray(counts, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)[:-1]]).astype(np.int64)
        weights = dict(settings.get("tag_weights") or {})
        unknown = set(weights) - set(TAGS)
        if unknown:
            raise ValueError(f"unknown replay tags: {sorted(unknown)}")
        w = np.array([float(weights.get(t, 0.0)) if self.counts[i] > 0 else 0.0
                      for i, t in enumerate(TAGS)])
        self.available = w.sum() > 0
        self.cum = np.cumsum(w / w.sum()) if self.available else w
        self.flip_x_prob = float(settings.get("flip_x_prob", 0.0))
        self.team_swap_prob = float(settings.get("team_swap_prob", 0.0))

    @classmethod
    def from_parser(cls, parser, settings: Mapping[str, Any]) -> Optional["ReplaySampler"]:
        if parser.states_buffer is None and not parser.load_pool():
            return None
        pool = parser.states_buffer
        order, counts = load_or_build_index(pool, getattr(parser, "store_dir", None), settings.get("prune"))
        return cls(pool, order, counts, settings)

    def draw_index(self) -> int:
        t = int(np.searchsorted(self.cum, random.random(), side="right"))
        t = min(t, len(TAGS) - 1)
        return int(self.order[self.offsets[t] + random.randrange(int(self.counts[t]))])

    def sample(self, num_cars: int = 2) -> Optional[Dict[str, np.ndarray]]:
        if not self.available:
            return None
        idx = self.draw_index()
        state = {
            "ball_pos": np.array(self.pool["ball_pos"][idx], dtype=np.float32),
            "ball_vel": np.array(self.pool["ball_vel"][idx], dtype=np.float32),
            "car_pos": np.array(self.pool["car_pos"][idx], dtype=np.float32),
            "car_vel": np.array(self.pool["car_vel"][idx], dtype=np.float32),
            "car_rot": np.array(self.pool["car_rot"][idx], dtype=np.float32),
            "car_boost": np.array(self.pool["car_boost"][idx], dtype=np.float32),
        }
        if random.random() < self.flip_x_prob:
            state = mirror_x(state)
        if random.random() < self.team_swap_prob:
            state = swap_teams(state)
        from utils.replay_parser import fit_car_count
        return fit_car_count(state, num_cars)


# ---- Frozen pool ----------------------------------------------------------------------------

def pool_fingerprint(pool: Mapping[str, np.ndarray], order: np.ndarray, counts: np.ndarray,
                     prune: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    """What the snapshot records: the kept frames per tag, and a hash of the frames and the index."""
    h = hashlib.sha256()
    h.update(index_key(prune).encode())
    for k in ("ball_pos", "ball_vel", "car_pos", "car_vel", "car_rot", "car_boost"):
        arr = pool[k]
        for lo in range(0, len(arr), CHUNK):
            h.update(np.ascontiguousarray(arr[lo:lo + CHUNK], dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(order, dtype=np.int32).tobytes())
    return {
        "frames": int(len(pool["ball_pos"])),
        "kept_frames": int(np.sum(counts)),
        "tag_counts": {t: int(c) for t, c in zip(TAGS, counts)},
        "sha": h.hexdigest()[:16],
    }


def current_fingerprint(settings: Mapping[str, Any], pool_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    from utils.replay_parser import ReplayParser
    parser = ReplayParser(pool_path=pool_path) if pool_path else ReplayParser()
    if parser.states_buffer is None:
        return None
    order, counts = load_or_build_index(parser.states_buffer, parser.store_dir, settings.get("prune"))
    return pool_fingerprint(parser.states_buffer, order, counts, settings.get("prune"))


def check_replay_pool(snapshot: Mapping[str, Any], pool_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    For a version whose starts come from a frozen pool, raise unless the pool on disk is the one the
    snapshot froze. Versions without a `replay_sampling` block are not checked. Returns the
    fingerprint checked, or None.
    """
    settings = (snapshot.get("settings") or {}).get("replay_sampling")
    if not settings:
        return None
    version = snapshot.get("version", "?")
    frozen = snapshot.get("replay_pool")
    if not frozen:
        raise RuntimeError(f"reward {version} samples a frozen replay pool, and none is frozen yet: ingest "
                           f"the replays, then run scripts/freeze_replay_pool.py --version {version}")
    now = current_fingerprint(settings, pool_path)
    if now is None:
        raise RuntimeError(f"reward {version} samples a frozen replay pool, and there is no pool on disk")
    if now["sha"] != frozen.get("sha"):
        raise RuntimeError(f"the replay pool changed since reward {version} froze it "
                           f"({frozen.get('kept_frames')} kept frames, sha {frozen.get('sha')}; now "
                           f"{now['kept_frames']}, sha {now['sha']}). The pool is part of the version: "
                           f"restore it, or make the change a new version.")
    return now
