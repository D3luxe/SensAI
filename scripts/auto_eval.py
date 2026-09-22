"""
Evaluates every new checkpoint as training writes it, so a decision point pools a real series.

A full eval is ~45 s on 4 workers and a checkpoint lands every ~7 min, so the suite can simply keep
up with the run: ~11% of four cores, on a machine whose training uses 16 of 24. That replaces
picking three checkpoints by hand at each decision point, where head to head has swung by 10 goals
between adjacent points in three runs running (docs/reward_v6_align_spec.md §6).

The head-to-head reference defaults to the version's own start checkpoint, the one its spec judges
it against. Results are named by the eval suite itself (<version>_<M>M), and a checkpoint whose
iteration already has a result is skipped, so this can be stopped and restarted freely.

    python scripts/auto_eval.py                     # follow the run, every checkpoint
    python scripts/auto_eval.py --every 2           # every other one
    python scripts/auto_eval.py --backfill --once   # catch up on what is already on disk, then stop
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.reward_registry import active_version, load_snapshot  # noqa: E402
from utils.eval_results import list_results, load  # noqa: E402

# Windows: keep the eval below the trainer in the scheduler, so following the run costs it nothing
BELOW_NORMAL = 0x00004000 if os.name == "nt" else 0


def iteration_of(path: str) -> int:
    return int(re.search(r"(\d+)\.pt$", path).group(1))


def evaluated_iterations(eval_dir: str) -> set:
    return {e["iteration"] for e in list_results(eval_dir) if e["iteration"] is not None and not e["quick"]}


def head_to_head(path: str):
    """(mean, seeds) of the head-to-head goal difference in a finished result, if it has one."""
    try:
        s = (load(path).get("results") or {}).get("reference", {}).get("goal_diff_per_10min")
    except (OSError, ValueError):
        return None
    return (s["mean"], s.get("per_seed", [])) if s else None


def run_one(checkpoint: str, reference: str, workers: int, eval_dir: str) -> None:
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "eval_suite.py"), "--checkpoint", checkpoint,
           "--workers", str(workers)]
    if reference:
        cmd += ["--reference", reference]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, creationflags=BELOW_NORMAL) \
        if BELOW_NORMAL else subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  FAILED {os.path.basename(checkpoint)} ({proc.returncode})\n{proc.stdout[-1500:]}{proc.stderr[-1500:]}")
        return
    written = re.search(r"wrote (\S+)", proc.stdout)
    out = written.group(1) if written else "?"
    h2h = head_to_head(os.path.join(ROOT, out)) if written else None
    line = f"  {os.path.basename(checkpoint)} -> {os.path.basename(out)}  {time.time() - t0:4.0f}s"
    if h2h:
        line += f"   head to head {h2h[0]:+6.2f}  ({', '.join(f'{x:+.1f}' for x in h2h[1])})"
    print(line, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=1, help="evaluate every Nth checkpoint (by iteration order)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reference", default=None, help="head-to-head opponent (default: the version's start checkpoint)")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between checks for a new checkpoint")
    ap.add_argument("--backfill", action="store_true", help="also evaluate checkpoints already on disk")
    ap.add_argument("--once", action="store_true", help="evaluate what is pending, then stop")
    ap.add_argument("--min-iteration", type=int, default=0)
    ap.add_argument("--eval-dir", default="evals")
    args = ap.parse_args()

    version = active_version()
    reference = args.reference or load_snapshot(version).get("start_checkpoint")
    if reference and not os.path.exists(os.path.join(ROOT, reference)):
        print(f"reference {reference} not found; head to head will be skipped")
        reference = None
    print(f"auto-eval: reward {version}, every {args.every} checkpoint(s), {args.workers} workers, "
          f"head to head vs {os.path.basename(reference) if reference else 'nothing'}")

    seen_floor = args.min_iteration
    if not args.backfill:
        on_disk = [iteration_of(p) for p in glob.glob(os.path.join(ROOT, "checkpoints", "checkpoint_iter_*.pt"))]
        seen_floor = max([seen_floor] + on_disk)
        print(f"following from iteration {seen_floor} (pass --backfill to evaluate what is already saved)")

    while True:
        done = evaluated_iterations(args.eval_dir)
        pending = sorted(p for p in glob.glob(os.path.join(ROOT, "checkpoints", "checkpoint_iter_*.pt"))
                         if iteration_of(p) > seen_floor and iteration_of(p) not in done)
        if args.every > 1:
            pending = [p for p in pending if (iteration_of(p) // 200) % args.every == 0]
        for path in sorted(pending, key=iteration_of):
            run_one(path, reference, args.workers, args.eval_dir)
        if args.once:
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
