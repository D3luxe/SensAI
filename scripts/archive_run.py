"""
Moves a finished run's checkpoints out of checkpoints/ before the next run reuses their numbers.

Checkpoint numbering continues from each run's start checkpoint, so the next run writes
checkpoint_iter_N.pt names the last run already used. This moves the finished run's files to
checkpoints/archive/<run>/, copies latest_model.pt there, and rewrites every reference to them:
the leaderboard (keys, paths, and display names, prefixed with the run so the table can tell two
checkpoint_iter_222200s apart), the league state, and the batch fit's results log.

A checkpoint belongs to the run if its iteration is above --after and it was written at or after
--since, so older runs' files that happen to share the range are left alone.

    python scripts/archive_run.py --run v5_run --after 198000 --since 2026-09-20T21:53 --dry-run
    python scripts/archive_run.py --run v5_run --after 198000 --since 2026-09-20T21:53
"""
from __future__ import annotations

import argparse
import datetime
import glob
import io
import json
import os
import re
import shutil
import sys

LEADERBOARD = "logs/trueskill_leaderboard.json"
RESULTS = "logs/trueskill_leaderboard.results.jsonl"
STATE = "logs/league_state.json"
PATTERN = re.compile(r'checkpoints(?:/|\\\\|\\)+checkpoint_iter_(\d+)\.pt')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="archive folder name, e.g. v5_run")
    ap.add_argument("--after", type=int, required=True, help="only iterations above this")
    ap.add_argument("--since", required=True, help="only files written at or after this ISO time")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    since = datetime.datetime.fromisoformat(args.since).timestamp()
    dest = os.path.join("checkpoints", "archive", args.run)
    if os.path.exists(dest) and not args.dry_run:
        print(f"{dest} already exists; refusing to archive into it twice")
        return 1

    moving = {}
    for p in glob.glob("checkpoints/checkpoint_iter_*.pt"):
        n = int(re.search(r"(\d+)\.pt$", p).group(1))
        if n > args.after and os.path.getmtime(p) >= since:
            moving[n] = p
    print(f"{len(moving)} checkpoints to move: {min(moving)}..{max(moving)}" if moving else "nothing to move")
    if not moving:
        return 1

    def repoint(text: str) -> str:
        return PATTERN.sub(lambda m: (f"checkpoints/archive/{args.run}/checkpoint_iter_{m.group(1)}.pt"
                                      if int(m.group(1)) in moving else m.group(0)), text)

    lb = json.load(io.open(LEADERBOARD, encoding="utf-8"))
    ratings = {}
    renamed = 0
    for k, r in lb.get("ratings", {}).items():
        nk = repoint(k)
        if nk != k:
            r = dict(r, path=repoint(r.get("path", k)), name=f"{args.run.split('_')[0]}/{r.get('name')}")
            renamed += 1
        ratings[nk] = r
    lb["ratings"] = ratings
    state_text = repoint(io.open(STATE, encoding="utf-8").read())
    results_text = repoint(io.open(RESULTS, encoding="utf-8").read()) if os.path.exists(RESULTS) else None
    print(f"leaderboard records repointed: {renamed}")
    print(f"league state references repointed: {len(PATTERN.findall(io.open(STATE, encoding='utf-8').read())) - len(PATTERN.findall(state_text))}")
    if results_text is not None:
        before = io.open(RESULTS, encoding="utf-8").read()
        print(f"results-log references repointed: {len(PATTERN.findall(before)) - len(PATTERN.findall(results_text))}")

    if args.dry_run:
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    for p in (LEADERBOARD, STATE, RESULTS):
        if os.path.exists(p):
            shutil.copyfile(p, f"{p}.before_{args.run}_{stamp}")
    os.makedirs(dest)
    for n, p in sorted(moving.items()):
        shutil.move(p, os.path.join(dest, os.path.basename(p)))
    if os.path.exists("checkpoints/latest_model.pt"):
        shutil.copyfile("checkpoints/latest_model.pt", os.path.join(dest, "latest_model.pt"))
    with io.open(LEADERBOARD, "w", encoding="utf-8") as f:
        json.dump(lb, f, indent=2)
    with io.open(STATE, "w", encoding="utf-8") as f:
        f.write(state_text)
    if results_text is not None:
        with io.open(RESULTS, "w", encoding="utf-8", newline="\n") as f:
            f.write(results_text)
    print(f"moved {len(moving)} files to {dest}; backups at *.before_{args.run}_{stamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
