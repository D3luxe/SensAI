"""
Absolute progress curve: every checkpoint against every fixed reference, separately.

Why this exists
---------------
The TrueSkill ladder is pool-relative. It solves each checkpoint's rating against the
other checkpoints, so a gain shared by the whole population is invisible to it: if every
save improves by the same amount, every head-to-head stays near even and the ladder
reports nothing. On the live board the later checkpoint of an established pair wins 47.8%
of its series (median 0.500, ahead in 18 of 44 pairs), which is exactly what a uniformly
improving population looks like to a relative measure. Mean reward cannot fill the gap
either, because the reward function was rebased during the run.

Only a fixed opponent gives a number that is comparable across checkpoints.

Why every reference is reported separately
------------------------------------------
The reference set is not a ladder. Nexto beats Necto every series, loses to every
checkpoint measured, and Necto beats those same checkpoints -- a closed cycle, so no
scalar ordering holds all three legs and any single pinned value is wrong against
someone. Collapsing these into one "progress" number would therefore invent a
transitivity that does not exist. Four independent curves are the honest output:
agreement across them is evidence of general improvement, and divergence localises a
style shift instead of hiding it.

Why goal margin rather than series wins
---------------------------------------
Series win rate against these references is saturated at the rails -- checkpoints take
100% off the heuristic and the BC baseline and 0% off Necto -- so it has no gradient left
to show a change in either direction. Goal margin still moves while the series outcome is
pinned: losing 5-0 and losing 5-3 are different policies. Margin per episode is the
primary readout here; series wins are reported alongside it but are expected to be flat.

Usage
-----
    python scripts/reference_sweep.py                      # full 4 x 10 x 4 sweep
    python scripts/reference_sweep.py --series 2           # cheaper
    python scripts/reference_sweep.py --checkpoints a.pt b.pt
    python scripts/reference_sweep.py --out logs/sweep.json

Nothing here touches the leaderboard. No rating moves.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.baseline_agent import create_opponent_bot
from utils.trueskill_evaluator import (
    DEFAULT_EVAL_MAX_STEPS,
    DEFAULT_NO_TOUCH_STEPS,
    DEFAULT_SERIES_LENGTH,
    DEFAULT_SERIES_WINS_NEEDED,
    simulate_headless_series,
)

# The full fixed-reference set, weakest to strongest as far as any ordering holds.
# Kept as (label, spec) so the heuristic, which is not a file, sits in the same list.
REFERENCES: List[Tuple[str, str]] = [
    ("heuristic", "heuristic"),
    ("BC baseline", "checkpoints/pretrained_baseline.pt"),
    ("Necto", "checkpoints/necto-model.pt"),
    ("Nexto", "checkpoints/nexto-model.pt"),
]

_ITER_RE = re.compile(r"iter_(\d+)")


def checkpoint_iteration(path: str) -> Optional[int]:
    m = _ITER_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def pick_checkpoints(count: int, ckpt_dir: str = "checkpoints") -> List[str]:
    """
    An evenly spaced spine across the whole saved range, with the newest always included.

    Even spacing in iteration rather than the newest N, because the question is the shape
    of the curve over the run, and clustering the sample at one end cannot show a shape.
    """
    found: List[Tuple[int, str]] = []
    for name in os.listdir(ckpt_dir):
        if not name.endswith(".pt"):
            continue
        n = checkpoint_iteration(name)
        if n is not None:
            found.append((n, os.path.join(ckpt_dir, name).replace("\\", "/")))
    found.sort()
    if not found:
        return []
    if len(found) <= count:
        return [p for _, p in found]
    lo, hi = found[0][0], found[-1][0]
    targets = [lo + (hi - lo) * i / (count - 1) for i in range(count)]
    picked: List[str] = []
    for t in targets:
        _, path = min(found, key=lambda pair: abs(pair[0] - t))
        if path not in picked:
            picked.append(path)
    newest = found[-1][1]
    if newest not in picked:
        picked.append(newest)
    return picked


def play(subject: str, reference: str, series: int, series_length: int,
         wins_needed: int, max_steps: int, no_touch_steps: int) -> Dict[str, float]:
    """One checkpoint against one reference. The subject is always side a."""
    a = create_opponent_bot(subject, continuous_actions=True)
    b = create_opponent_bot(reference, continuous_actions=True)

    series_won = series_drawn = 0
    goals_for = goals_against = episodes = 0
    for _ in range(series):
        r = simulate_headless_series(
            a, b,
            series_length=series_length,
            wins_needed=wins_needed,
            max_steps=max_steps,
            no_touch_steps=no_touch_steps,
        )
        w = r.get("winner")
        series_won += 1 if w == "a" else 0
        series_drawn += 1 if w == "draw" else 0
        goals_for += int(r.get("a_score", 0))
        goals_against += int(r.get("b_score", 0))
        episodes += int(r.get("episodes_played", 0))

    return {
        "series": series,
        "series_won": series_won,
        "series_drawn": series_drawn,
        "series_win_rate": round(series_won / series, 3) if series else 0.0,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "episodes": episodes,
        # The primary readout. Positive means the subject outscored the reference.
        "margin_per_episode": round((goals_for - goals_against) / episodes, 3) if episodes else 0.0,
        "goal_share": round(goals_for / (goals_for + goals_against), 3) if (goals_for + goals_against) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="Explicit checkpoint paths. Default: an evenly spaced spine of --count.")
    ap.add_argument("--count", type=int, default=10, help="Checkpoints to sample when none are named.")
    ap.add_argument("--series", type=int, default=4, help="Series per checkpoint-reference pair.")
    ap.add_argument("--references", nargs="*", default=None,
                    help="Subset of reference labels, e.g. Necto Nexto. Default: all four.")
    ap.add_argument("--series-length", type=int, default=DEFAULT_SERIES_LENGTH)
    ap.add_argument("--wins-needed", type=int, default=DEFAULT_SERIES_WINS_NEEDED)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_EVAL_MAX_STEPS)
    ap.add_argument("--no-touch-steps", type=int, default=DEFAULT_NO_TOUCH_STEPS)
    ap.add_argument("--out", default="logs/reference_sweep.json")
    args = ap.parse_args()

    refs = [(lab, spec) for lab, spec in REFERENCES
            if args.references is None or lab in args.references]
    refs = [(lab, spec) for lab, spec in refs if spec == "heuristic" or os.path.exists(spec)]
    if not refs:
        print("No usable references found.")
        return 1

    ckpts = args.checkpoints or pick_checkpoints(args.count)
    ckpts = [c for c in ckpts if os.path.exists(c)]
    if not ckpts:
        print("No usable checkpoints found.")
        return 1

    total = len(ckpts) * len(refs)
    print(f"Reference sweep: {len(ckpts)} checkpoints x {len(refs)} references x {args.series} series "
          f"= {total} matchups")
    print(f"References: {', '.join(lab for lab, _ in refs)}")
    print("No rating is touched by this script.\n")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    started = time.time()
    done = 0
    for ckpt in ckpts:
        name = os.path.basename(ckpt)
        results[name] = {}
        for label, spec in refs:
            t0 = time.time()
            try:
                row = play(ckpt, spec, args.series, args.series_length,
                           args.wins_needed, args.max_steps, args.no_touch_steps)
            except Exception as e:
                print(f"  {name} vs {label}: FAILED ({e})")
                continue
            results[name][label] = row
            done += 1
            eta = (time.time() - started) / done * (total - done)
            print(f"  [{done:3d}/{total}] {name:28s} vs {label:12s} "
                  f"margin/ep {row['margin_per_episode']:+6.2f}  "
                  f"goals {row['goals_for']:3d}-{row['goals_against']:3d}  "
                  f"series {row['series_won']}/{row['series']}  "
                  f"({time.time() - t0:5.1f}s, eta {eta / 60:4.1f}m)")

    payload = {
        "generated_at": datetime.datetime.now().isoformat(),
        "series_per_pair": args.series,
        "series_length": args.series_length,
        "references": [lab for lab, _ in refs],
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {args.out}\n")
    print("Goal margin per episode, by reference. Each column is its own scale;")
    print("the references are intransitive, so columns are never combined.\n")
    labels = [lab for lab, _ in refs]
    header = f"{'checkpoint':>14s} " + " ".join(f"{lab:>13s}" for lab in labels)
    print(header)
    print("-" * len(header))
    for name in sorted(results, key=lambda n: checkpoint_iteration(n) or 0):
        it = checkpoint_iteration(name)
        cells = []
        for lab in labels:
            row = results[name].get(lab)
            cells.append(f"{row['margin_per_episode']:+13.2f}" if row else f"{'--':>13s}")
        print(f"{it if it is not None else name:>14} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
