"""
Audit imputed replay actions bucketed by car height.

Samples consecutive replay frame pairs with the same 250 uu continuity filter the
pretrainer uses, runs InverseDynamicsSolver.solve_car_action, and reports mean abs
pitch / saturation per height band. See docs/bc_replay_pipeline_defects.md.

Usage: python scripts/audit_imputed_actions.py [--pool PATH] [--samples N] [--seed S]
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.replay_parser import ReplayParser
from utils.inverse_dynamics import InverseDynamicsSolver

BANDS = [
    ("floor, z < 25", lambda z: z < 25.0),
    ("z 25-200", lambda z: (z >= 25.0) & (z <= 200.0)),
    ("wall, z > 200", lambda z: z > 200.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/replays/replays_pool.npz")
    ap.add_argument("--samples", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    parser = ReplayParser(pool_path=args.pool)
    parser.load_pool()
    data = parser.states_buffer
    if data is None:
        sys.exit(f"could not load pool at {args.pool}")

    n = len(data["car_pos"])
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(n - 1, size=min(args.samples, n - 1), replace=False)

    zs, acts = [], []
    for idx in idxs:
        cp, cv, cr, cb = (data[k][idx] for k in ("car_pos", "car_vel", "car_rot", "car_boost"))
        np_, nv, nr, nb = (data[k][idx + 1] for k in ("car_pos", "car_vel", "car_rot", "car_boost"))
        num_cars = cp.shape[0] if cp.ndim > 1 else 1
        for ci in range(min(2, num_cars)):
            p = cp[ci] if cp.ndim > 1 else cp
            v = cv[ci] if cv.ndim > 1 else cv
            r = cr[ci] if cr.ndim > 1 else cr
            b = cb[ci] if cb.ndim > 0 else cb
            p2 = np_[ci] if np_.ndim > 1 else np_
            v2 = nv[ci] if nv.ndim > 1 else nv
            r2 = nr[ci] if nr.ndim > 1 else nr
            b2 = nb[ci] if nb.ndim > 0 else nb
            if float(np.linalg.norm(p2 - p)) >= 250.0:
                continue
            a = InverseDynamicsSolver.solve_car_action(
                p, v, r, np.zeros(3, dtype=np.float32), float(b), bool(p[2] < 25.0),
                p2, v2, r2, np.zeros(3, dtype=np.float32), float(b2), bool(p2[2] < 25.0),
                dt=1.0 / 30.0,
            )
            zs.append(float(p[2]))
            acts.append(a)

    zs = np.asarray(zs)
    acts = np.asarray(acts)
    print(f"{len(zs)} car-frames from {len(idxs)} sampled frame pairs")
    print(f"{'band':<16}{'n':>8}  {'pitch |mean|':>12} {'>0.9':>7} {'==+1':>7}  {'yaw |mean|':>10} {'roll |mean|':>11}")
    for name, cond in BANDS + [("all", lambda z: np.ones_like(z, dtype=bool))]:
        m = cond(zs)
        if not m.any():
            continue
        pitch = acts[m, 2]
        print(f"{name:<16}{m.sum():>8}  {np.abs(pitch).mean():>12.3f} {np.mean(np.abs(pitch) > 0.9):>7.1%} "
              f"{np.mean(pitch == 1.0):>7.1%}  {np.abs(acts[m, 3]).mean():>10.3f} {np.abs(acts[m, 4]).mean():>11.3f}")


if __name__ == "__main__":
    main()
