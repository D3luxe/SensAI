"""
Freezes the replay pool into a reward version's snapshot, for versions that sample one (v7).

The pool decides where a v7 episode starts, so it is part of the version (R1): the trainer refuses
to run v7 on any pool but the one frozen here. Ingest the replays first, then freeze, then start the
run. Prints the kept frames per situation tag, which is also the check that the pool is what you
meant to freeze.

    python scripts/freeze_replay_pool.py --version v7 --dry-run
    python scripts/freeze_replay_pool.py --version v7
    python scripts/freeze_replay_pool.py --version v7 --refreeze   # only before the run has started
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.reward_registry import SNAPSHOT_DIR, _resolve, load_snapshot  # noqa: E402
from env.replay_sampling_v7 import current_fingerprint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--pool", default=None, help="pool .npz (default: data/replays/replays_pool.npz)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refreeze", action="store_true",
                    help="replace an existing frozen pool; only before the version's run has started")
    args = ap.parse_args()

    snap = load_snapshot(args.version)
    settings = (snap.get("settings") or {}).get("replay_sampling")
    if not settings:
        print(f"reward {args.version} does not sample a frozen replay pool; nothing to freeze")
        return 1
    fp = current_fingerprint(settings, args.pool)
    if fp is None:
        print("no replay pool on disk")
        return 1

    print(f"{fp['frames']:,} frames in the pool, {fp['kept_frames']:,} kept after pruning "
          f"({fp['kept_frames'] / max(1, fp['frames']):.1%}); sha {fp['sha']}")
    weights = settings.get("tag_weights") or {}
    for tag, n in fp["tag_counts"].items():
        print(f"  {tag:16} {n:>11,}  {n / max(1, fp['kept_frames']):6.1%} of kept   weight {float(weights.get(tag, 0)):.3f}")

    old = snap.get("replay_pool")
    if old and old.get("sha") == fp["sha"]:
        print("already frozen with this pool")
        return 0
    if old and not args.refreeze:
        print(f"reward {args.version} already froze a different pool (sha {old.get('sha')}). If its run "
              f"has started, the pool is part of it: restore that pool. Before the run, pass --refreeze.")
        return 1
    if args.dry_run:
        return 0

    path = _resolve(os.path.join(SNAPSHOT_DIR, f"{args.version}.json"))
    with io.open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["replay_pool"] = fp
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(raw, f, indent=2)
        f.write("\n")
    print(f"frozen into {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
