"""
Steering jitter probe: how often a policy zig-zags its steering, by distance to the ball.

Runs checkpoints deterministically in RocketSim, the way the in-game bot runs them, with both
cars driven by the checkpoint under test. At every 15 Hz decision where the car is on the
ground it records the steer command and the distance to the ball, then counts reversals: a
decision where the steer moved by more than --swing in one direction and then by more than
--swing back. A smooth approach has almost none; a controller overcorrecting against its own
last output alternates every decision.

Why it exists (2026-09-21): in-game, v5 visibly jittered left/right in the last moments of an
approach. The simulator shows the same thing in both the v3 and v5 kings, so it is learned
rather than a deployment bug. A first ad-hoc run read v3 at 5.9% and v5 at 21.7% inside 400 uu
and called that six standard errors; it was not reproducible (Python's random was unseeded)
and its binomial error bars ignored that decisions within an episode are correlated. Re-run
at 100M and 150M to see whether v5 separates from v3 by more than the error printed here.

The +/- is a bootstrap over whole episodes, which is the honest unit of independence. The
inside-400 bin has the fewest decisions, so it is the noisiest -- raise --episodes before
reading much into a difference there.

    python scripts/steering_jitter_probe.py
    python scripts/steering_jitter_probe.py checkpoints/baselines/v3_iter198000.pt checkpoints/checkpoint_iter_204000.pt
    python scripts/steering_jitter_probe.py --episodes 60 --json logs/jitter_150M.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.checkpoint import load_policy  # noqa: E402
from env.observations import DefaultObservationBuilder  # noqa: E402
from env.rocket_env import RocketLeagueEnv  # noqa: E402

DEFAULT_CHECKPOINTS = [
    "checkpoints/baselines/v3_iter198000.pt",
    "checkpoints/baselines/v5_iter200200.pt",
]
BINS: List[Tuple[float, float]] = [(0, 400), (400, 800), (800, 1500), (1500, 3000), (3000, math.inf)]


def _act(model, obs: np.ndarray) -> np.ndarray:
    o = torch.from_numpy(obs).float().unsqueeze(0)
    with torch.no_grad():
        if hasattr(model, "get_action"):
            a = model.get_action(o, deterministic=True)
        else:
            a = model.get_action_and_value(o, deterministic=True)[0]
    return a.squeeze(0).cpu().numpy()


def probe(checkpoint: str, episodes: int = 30, steps: int = 450, seed: int = 0,
          swing: float = 0.3) -> List[Dict[Tuple[float, float], List[int]]]:
    """One {distance bin: [reversals, decisions scored]} per episode, for one checkpoint."""
    model, _ = load_policy(checkpoint, device="cpu")
    model.eval()
    env = RocketLeagueEnv(game_mode="1v1", max_episode_steps=10 ** 6)
    builder = DefaultObservationBuilder(symmetric=True)
    # All three generators: env/state_setters.py draws every reset from Python's random, so
    # seeding only numpy and torch left the episodes -- and the results -- different each run.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    per_episode: List[Dict[Tuple[float, float], List[int]]] = []
    for _ in range(episodes):
        counts = {b: [0, 0] for b in BINS}
        env.reset()
        trace: List[Optional[Tuple[float, float]]] = []
        for _ in range(steps):
            actions = [_act(model, builder.build_obs(car, env.arena)) for car in env.arena.cars]
            me = env.arena.cars[0]
            if me.on_ground:
                dist = float(np.linalg.norm(np.asarray(env.arena.ball.pos) - np.asarray(me.pos)))
                trace.append((dist, float(np.clip(actions[0][1], -1.0, 1.0))))
            else:
                trace.append(None)     # aerial steering is a different control problem
            env.step(np.stack(actions).astype(np.float32))

        for prev, cur, nxt in zip(trace, trace[1:], trace[2:]):
            if prev is None or cur is None or nxt is None:
                continue
            d1, d2 = cur[1] - prev[1], nxt[1] - cur[1]
            for lo, hi in BINS:
                if lo <= cur[0] < hi:
                    counts[(lo, hi)][1] += 1
                    if d1 * d2 < 0 and abs(d1) > swing and abs(d2) > swing:
                        counts[(lo, hi)][0] += 1
                    break
        per_episode.append(counts)
    return per_episode


def summarise(per_episode, boot: int = 2000, seed: int = 0):
    """
    {bin: (rate, standard error, reversals, decisions)}, with the error from resampling
    whole episodes. Decisions within an episode are not independent -- a car that is
    jittering keeps jittering -- so the binomial sqrt(p(1-p)/n) understates the error badly
    and made a 4-point difference look like six standard errors.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for b in BINS:
        n = np.array([ep[b][0] for ep in per_episode], dtype=np.float64)
        t = np.array([ep[b][1] for ep in per_episode], dtype=np.float64)
        rate = n.sum() / t.sum() if t.sum() else 0.0
        idx = rng.integers(0, len(n), size=(boot, len(n)))
        tb = t[idx].sum(axis=1)
        rb = np.where(tb > 0, n[idx].sum(axis=1) / np.maximum(tb, 1), np.nan)
        out[b] = (rate, float(np.nanstd(rb)), int(n.sum()), int(t.sum()))
    return out


def _label(lo: float, hi: float) -> str:
    return f"{int(lo)}-{'+' if math.isinf(hi) else int(hi)} uu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", nargs="*", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--steps", type=int, default=450, help="15 Hz decisions per episode (450 = 30 s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swing", type=float, default=0.3, help="minimum steer change counted as a swing")
    ap.add_argument("--json", help="also write the counts to this file")
    args = ap.parse_args()

    results = {}
    for ck in args.checkpoints:
        if not os.path.exists(ck):
            print(f"skipping {ck}: not found")
            continue
        stats = summarise(probe(ck, args.episodes, args.steps, args.seed, args.swing), seed=args.seed)
        results[ck] = {_label(lo, hi): {"rate": r, "se": se, "reversals": n, "decisions": tot}
                       for (lo, hi), (r, se, n, tot) in stats.items()}
        print(f"\n{os.path.basename(ck)}: share of grounded decisions reversing a >{args.swing} swing"
              f"  (+/- is an episode-bootstrap standard error)")
        for (lo, hi), (r, se, n, tot) in stats.items():
            print(f"   {_label(lo, hi):>14}  {100 * r:5.1f}% +/- {100 * se:4.1f}   ({n}/{tot})")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"episodes": args.episodes, "steps": args.steps, "seed": args.seed,
                       "swing": args.swing, "results": results}, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
