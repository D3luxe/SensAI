"""
Re-anchor reward-anneal clocks that were wrongly backdated on a resume.

The first cut of the per-target anneal clocks migrated the old shared
reward_anneal_start_step by seeding *every* configured target from it. Targets added in the same
release as the migration had therefore never been covered by that clock, but inherited it anyway,
arrived at 100% progress, and snapped to their final weights on the first iteration after the
resume. ppo.py now consults the checkpoint's own saved config instead, but a checkpoint written
between the two only carries the damaged clock dict, which the loader trusts.

This drops the clocks for the named targets so they re-anchor at the checkpoint's global_step on
the next resume, restarting their ramps from wherever the policy currently sits.

Stop training before running this: a live trainer holds the damaged clocks in memory and will
write them back over any repair.

    python scripts/repair_anneal_clocks.py [checkpoint ...] --targets a_weight,b_weight
"""
import argparse
import shutil
import sys

import torch


def repair(path: str, targets: list, dry_run: bool = False) -> bool:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    clocks = ck.get("reward_anneal_start_steps")
    if not isinstance(clocks, dict):
        print(f"{path}: no per-target clocks, nothing to repair")
        return False

    stale = [k for k in targets if k in clocks]
    if not stale:
        print(f"{path}: none of the named targets carry a clock, nothing to repair")
        return False

    step = ck.get("global_step", 0)
    print(f"{path}: global_step {step:,}")
    for k in stale:
        print(f"  dropping {k} (was anchored at {clocks[k]:,}) -> re-anchors at {step:,} on resume")
    for k in stale:
        del clocks[k]
    print(f"  clocks kept: {clocks}")

    if dry_run:
        print("  dry run, not written")
        return False

    backup = path + ".before_clock_repair"
    shutil.copy2(path, backup)
    ck["reward_anneal_start_steps"] = clocks
    torch.save(ck, path)
    print(f"  written, previous file kept at {backup}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", default=["checkpoints/latest_model.pt"])
    ap.add_argument("--targets", default="jump_bridge_weight,player_to_ball_weight",
                    help="comma-separated reward weight names whose clocks should be dropped")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        print("no targets given", file=sys.stderr)
        return 2

    changed = any(repair(p, targets, args.dry_run) for p in args.checkpoints)
    if changed:
        print("\nRestart training to pick up the re-anchored clocks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
