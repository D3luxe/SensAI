"""
Pools a run's eval results at a decision point: the table every judgement in docs/ is read off.

The pooling itself lives in ui/eval_pooling.py, which the Evaluation tab renders as a table; this is
the command line over it. A single checkpoint's reading is not a decision
(docs/reward_v5_gamma_spec.md §4): head to head has a per-seed sd near 6.7, and adjacent checkpoints
have differed by 25 goals per 10 min.

    python scripts/pool_evals.py --version v7 --last 5
    python scripts/pool_evals.py --version v7 --around 200 --boost   # within +/-10M of 200M
    python scripts/pool_evals.py evals/v7_196M.json evals/v7_199M.json
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ui.eval_pooling import pick_results, text_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="result files to pool (otherwise --version picks them)")
    ap.add_argument("--version", default=None, help="reward version whose run to pool, e.g. v7")
    ap.add_argument("--last", type=int, default=None, help="the last N results of that run")
    ap.add_argument("--around", type=float, default=None, help="results within --window of this many M steps")
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--boost", action="store_true", help="also show the boost-economy columns")
    ap.add_argument("--eval-dir", default="evals")
    args = ap.parse_args()
    if not args.files and not args.version:
        ap.error("give result files, or --version")
    if not args.files and args.last is None and args.around is None:
        args.last = 5

    chosen = pick_results(version=args.version, eval_dir=args.eval_dir, last=args.last,
                          around=args.around, window=args.window, paths=args.files or None)
    title = " ".join(x for x in [args.version, f"last {args.last}" if args.last else None,
                                 f"around {args.around:g}M" if args.around is not None else None] if x) or "results"
    print(text_report(title, chosen, args.boost))
    return 0


if __name__ == "__main__":
    sys.exit(main())
