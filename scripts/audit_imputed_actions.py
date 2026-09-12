"""
Audit imputed replay actions bucketed by car height.

Samples replay frames, pairs each with the car's next genuine state update using the same
find_car_transition the pretrainer uses (real per-car dt on timed pools; 250 uu filter and the
legacy 10-frame spacing on old pools), runs InverseDynamicsSolver.solve_car_action, and reports
mean abs pitch / saturation per height band. See docs/bc_replay_pipeline_defects.md.

Usage: python scripts/audit_imputed_actions.py [--pool PATH] [--samples N] [--seed S] [--dt SECONDS]
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.replay_parser import ReplayParser, is_fresh_car_update, find_car_transition, _car_vec
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
    ap.add_argument("--dt", type=float, default=None, help="override the frame spacing passed to the solver")
    args = ap.parse_args()

    parser = ReplayParser(pool_path=args.pool)
    parser.load_pool()
    data = parser.states_buffer
    if data is None:
        sys.exit(f"could not load pool at {args.pool}")

    n = len(data["car_pos"])
    timed = "car_time" in data
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(n - 1, size=min(args.samples, n - 1), replace=False)
    num_cars = data["car_pos"].shape[1] if data["car_pos"].ndim > 2 else 1

    zs, acts, dts = [], [], []
    for idx in idxs:
        for ci in range(min(2, num_cars)):
            if not is_fresh_car_update(data, idx, ci):
                continue
            tr = find_car_transition(data, idx, ci)
            if tr is None:
                continue
            j, dt = tr
            if args.dt is not None:
                dt = args.dt
            g = lambda k, i: _car_vec(data[k], i, ci)
            boost = lambda i: float(data["car_boost"][i][ci] if data["car_boost"].ndim > 1 else data["car_boost"][i])
            p, p2 = g("car_pos", idx), g("car_pos", j)
            a = InverseDynamicsSolver.solve_car_action(
                p, g("car_vel", idx), g("car_rot", idx), np.zeros(3, dtype=np.float32), boost(idx), bool(p[2] < 25.0),
                p2, g("car_vel", j), g("car_rot", j), np.zeros(3, dtype=np.float32), boost(j), bool(p2[2] < 25.0),
                dt=dt,
            )
            zs.append(float(p[2]))
            acts.append(a)
            dts.append(dt)

    zs = np.asarray(zs)
    acts = np.asarray(acts)
    print(f"{'timed' if timed else 'legacy'} pool, {n} frames | {len(zs)} car transitions from {len(idxs)} sampled frames"
          f" | dt median {np.median(dts):.4f}s")
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
