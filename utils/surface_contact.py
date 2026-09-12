"""
Surface contact inference for replay frames.

Replays do not record whether a car's wheels touch a surface. A height test (z < 25) calls every
car on a wall, ramp or corner airborne. This module models the arena as a convex polytope (floor,
ceiling, side and back walls, 45 degree corner bevels) with rounded edges, joined with the two goal
boxes, and calls a car grounded when it sits close to that surface with its wheels facing it.

Fitted and validated against RocketSim's own is_on_ground flag on 144,000 randomly driven 30 Hz
states (floor, wall and backboard starts), thresholds tuned on 70% of episodes:

    held-out accuracy 93.6%  (z < 25: 76.6%)
    grounded called airborne 2.4%  (z < 25: 34.9%)
    airborne called grounded 14.2%  (z < 25: 0.8%)

Per region: floor 96.7%, fillet and low wall 91.1%, wall 96.5%, corner 91.8%, goal 87.8%. The
ceiling is the weak spot at 64.7%. Acceleration along the surface normal was tried as an extra
feature and did not improve accuracy: jump impulses near the floor overlap with grounded frames.
"""

from __future__ import annotations
import numpy as np

_S2 = 1.0 / np.sqrt(2.0)

# Outward plane normals and offsets: a point is inside when normal . p <= offset.
_ARENA_NORMALS = np.array([
    [0, 0, -1], [0, 0, 1],                  # floor, ceiling
    [1, 0, 0], [-1, 0, 0],                  # side walls
    [0, 1, 0], [0, -1, 0],                  # back walls
    [_S2, _S2, 0], [_S2, -_S2, 0], [-_S2, _S2, 0], [-_S2, -_S2, 0],   # corner bevels
], dtype=np.float64)
_ARENA_OFFSETS = np.array([0.0, 2044.0, 4096.0, 4096.0, 5120.0, 5120.0] + [8064.0 * _S2] * 4)

# Goal interiors: |x| <= 893, 0 <= z <= 642, reaching 880 uu behind the back wall.
_GOAL_NORMALS = np.array([[0, 0, -1], [0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float64)
_GOAL_OFFSETS = (
    np.array([0.0, 642.0, 893.0, 893.0, 6000.0, -4920.0]),    # orange goal, +Y
    np.array([0.0, 642.0, 893.0, 893.0, -4920.0, 6000.0]),    # blue goal, -Y
)

# Edge rounding radius. Swept 96-512 uu; 128 gave the tightest grounded depth distribution.
FILLET_RADIUS = 128.0
GOAL_FILLET_RADIUS = 128.0

# A grounded car's centre sits within this distance of the modelled surface (median ~30 uu,
# 90th percentile ~60 uu, widened by the approximate fillet shape).
CONTACT_DEPTH = 80.0
# Minimum cosine between the car's up vector and the surface's inward normal.
CONTACT_ALIGN = 0.8


def _shrunk_signed_distance(p: np.ndarray, normals: np.ndarray, offsets: np.ndarray, r: float) -> np.ndarray:
    """Signed distance from points (M, 3) to the polytope shrunk by r (negative inside), exact at faces and edges."""
    s = p @ normals.T - (offsets - r)
    q = s.max(axis=1)
    for a in range(len(offsets)):
        for b in range(a + 1, len(offsets)):
            c = float(normals[a] @ normals[b])
            if abs(c) > 0.999:
                continue
            sa, sb = s[:, a], s[:, b]
            edge = (sa > 0) & (sb > 0) & (sb - c * sa > 0) & (sa - c * sb > 0)
            if edge.any():
                dist = np.sqrt(np.maximum(0.0, (sa * sa + sb * sb - 2.0 * c * sa * sb) / (1.0 - c * c)))
                q = np.where(edge, np.maximum(q, dist), q)
    return q


def _rounded_surface(p: np.ndarray, normals: np.ndarray, offsets: np.ndarray, r: float, h: float = 2.0):
    q = _shrunk_signed_distance(p, normals, offsets, r)
    grad = np.stack([
        (_shrunk_signed_distance(p + h * e, normals, offsets, r) - _shrunk_signed_distance(p - h * e, normals, offsets, r)) / (2.0 * h)
        for e in np.eye(3)
    ], axis=1)
    inward = -grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9)
    return r - q, inward


def arena_surface(pos: np.ndarray):
    """
    Distance from each position to the nearest arena surface, and that surface's inward normal.
    pos: (3,) or (M, 3). Returns (depth, normal) shaped () / (3,) or (M,) / (M, 3).
    Depth is positive inside the arena; inside a goal the goal box's surfaces are used.
    """
    p = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    candidates = [_rounded_surface(p, _ARENA_NORMALS, _ARENA_OFFSETS, FILLET_RADIUS)]
    candidates += [_rounded_surface(p, _GOAL_NORMALS, o, GOAL_FILLET_RADIUS) for o in _GOAL_OFFSETS]
    depths = np.stack([d for d, _ in candidates], axis=1)
    k = depths.argmax(axis=1)
    rows = np.arange(len(p))
    depth = depths[rows, k]
    normal = np.stack([n for _, n in candidates], axis=1)[rows, k]
    if np.ndim(pos) == 1:
        return float(depth[0]), normal[0]
    return depth, normal


def is_on_surface(pos: np.ndarray, up: np.ndarray):
    """
    Infers wheel contact from position and the car's up vector. pos, up: (3,) or (M, 3).
    Returns a bool, or a bool array for batched input.
    """
    depth, normal = arena_surface(pos)
    align = np.sum(np.atleast_2d(np.asarray(up, dtype=np.float64)) * np.atleast_2d(normal), axis=1)
    grounded = (np.atleast_1d(depth) < CONTACT_DEPTH) & (align > CONTACT_ALIGN)
    if np.ndim(pos) == 1:
        return bool(grounded[0])
    return grounded
