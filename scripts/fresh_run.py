"""
Archive the current training run so the next one starts clean.

Restarting from scratch means retiring five kinds of state that otherwise leak across runs:

  1. Numbered league checkpoints. The league scans these, so a stale pool grades a fresh
     policy against opponents from a different regime. When the observation width has
     changed they load through the zero-padding migration rather than failing, which is
     worse than failing: the old opponents quietly behave as if the new features do not
     exist.
  2. League state and the TrueSkill leaderboard, which reference those checkpoints by path.
  3. Per-iteration history and metrics, which the UI charts as one continuous series.
  4. TensorBoard event files, which otherwise overlay the new run on the old.
  5. latest_model.pt, the trainer's resume point. Left in place the "fresh" run resumes.

Nothing is deleted. Everything moves under archive/run_<timestamp>/ so a run can be brought
back by moving it out again.

Deliberately preserved:
  - checkpoints/necto-model.pt and nexto-model.pt, the external benchmarks. They use their
    own observation builder and stay valid across our changes.
  - checkpoints/pretrained_baseline.pt, the behavioral-cloning seed.
  - data/, config/, and logs/reference_sweep*.json.

Usage:
    python scripts/fresh_run.py                 # dry run, prints the plan and changes nothing
    python scripts/fresh_run.py --apply         # do it
    python scripts/fresh_run.py --apply --keep-seed checkpoints/my_bc_seed.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Never archived: external benchmarks and the behavioral-cloning seed.
PRESERVE_ALWAYS = {
    "checkpoints/necto-model.pt",
    "checkpoints/nexto-model.pt",
    "checkpoints/pretrained_baseline.pt",
}

# Logs that describe the tooling rather than the run.
PRESERVE_LOG_GLOBS = ("reference_sweep*.json",)


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def training_is_live() -> tuple:
    """(is_live, detail). Reads logs/train.pid and checks whether that process still exists."""
    pid_file = os.path.join(ROOT, "logs", "train.pid")
    if not os.path.exists(pid_file):
        return False, "no logs/train.pid"
    try:
        with open(pid_file, encoding="utf-8") as fh:
            pid = int(fh.read().strip())
    except Exception:
        return False, "logs/train.pid unreadable"
    try:
        import subprocess
        out = subprocess.run(
            ["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        if str(pid) in out and "python" in out.lower():
            return True, "pid %d is running" % pid
    except Exception:
        # Cannot tell. Treat an existing pid file as live rather than risk clobbering a run.
        return True, "pid %d present, could not verify (treating as live)" % pid
    return False, "pid %d is not running (stale pid file)" % pid


def collect(keep_seed: str | None) -> dict:
    preserve = set(PRESERVE_ALWAYS)
    if keep_seed:
        preserve.add(_rel(os.path.abspath(keep_seed)))

    plan: dict = {}

    ckpts = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*.pt")))
    plan["checkpoints"] = [p for p in ckpts if _rel(p) not in preserve]

    log_names = [
        "league_state.json", "trueskill_leaderboard.json",
        "history.jsonl", "metrics.json", "start_time.txt", "train.pid",
        "test_results.json",
    ]
    logs = [os.path.join(ROOT, "logs", n) for n in log_names]
    logs = [p for p in logs if os.path.exists(p)]
    logs += sorted(glob.glob(os.path.join(ROOT, "logs", "events.out.tfevents.*")))

    keep_logs = set()
    for pat in PRESERVE_LOG_GLOBS:
        keep_logs.update(glob.glob(os.path.join(ROOT, "logs", pat)))
    plan["logs"] = [p for p in logs if p not in keep_logs]

    plan["preserved"] = sorted(preserve)
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually move the files; without this it only prints the plan")
    ap.add_argument("--keep-seed", default=None,
                    help="an extra checkpoint to preserve, e.g. a behavioral-cloning seed")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if a training process appears to be running")
    args = ap.parse_args()

    live, detail = training_is_live()
    if live and not args.force:
        print("REFUSING: training appears to be running (%s)." % detail)
        print("Stop it first, or pass --force if you know the pid file is stale.")
        return 1
    print("training check: %s\n" % detail)

    plan = collect(args.keep_seed)
    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    dest = os.path.join(ROOT, "archive", stamp)

    total = len(plan["checkpoints"]) + len(plan["logs"])
    if total == 0:
        print("Nothing to archive; the tree already looks clean.")
        return 0

    print("archive destination: %s\n" % _rel(dest))
    print("PRESERVED (not touched):")
    for p in plan["preserved"]:
        mark = "" if os.path.exists(os.path.join(ROOT, p)) else "   [missing]"
        print("  %s%s" % (p, mark))

    for group in ("checkpoints", "logs"):
        if not plan[group]:
            continue
        size = sum(os.path.getsize(p) for p in plan[group]) / 1e6
        print("\nARCHIVE %s (%d files, %.1f MB):" % (group, len(plan[group]), size))
        for p in plan[group][:12]:
            print("  %s" % _rel(p))
        if len(plan[group]) > 12:
            print("  ... and %d more" % (len(plan[group]) - 12))

    if not args.apply:
        print("\nDRY RUN. Nothing moved. Re-run with --apply to proceed.")
        return 0

    moved = 0
    for group in ("checkpoints", "logs"):
        if not plan[group]:
            continue
        out = os.path.join(dest, group)
        os.makedirs(out, exist_ok=True)
        for p in plan[group]:
            try:
                shutil.move(p, os.path.join(out, os.path.basename(p)))
                moved += 1
            except Exception as e:
                print("  WARNING: could not move %s: %s" % (_rel(p), e))

    print("\nArchived %d files to %s" % (moved, _rel(dest)))
    print("\nNext run starts clean. Reminders:")
    print("  - the trainer will mint a fresh league pool from scratch")
    print("  - re-seed with behavioral cloning if you want a warm start")
    print("  - confirm config/live_config.json carries the weights you intend; it")
    print("    overrides config/default_config.yaml at runtime")
    return 0


if __name__ == "__main__":
    sys.exit(main())
