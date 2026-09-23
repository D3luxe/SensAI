"""
Puts the league back to a saved snapshot, so a new run starts against its start checkpoint's league
instead of whatever the last run left behind.

The league keeps any rated checkpoint whose file still exists, and archive_run.py moves files rather
than retiring them. Without this, each run's King and elite pool (half of all training environments)
are the previous run's checkpoints: v9 opened against v8's collect-and-dump bot, and a run started
after v9 would open against v9's hoarders (docs/reward_v9_boost_spec.md §6, finding 7).

archive_run.py saves the three league files as *.before_<run>_<stamp> before it repoints them, so
that backup is the league exactly as the run ended, still naming the run's checkpoints by their
original checkpoints/checkpoint_iter_N.pt paths. This installs it as the live league with those
paths repointed into checkpoints/archive/<run>/, as archive_run.py would have. The current league
files are backed up first as *.before_restore_<stamp>.

    python scripts/restore_league.py --snapshot before_v5_run_202609211333 --run v5_run --dry-run
    python scripts/restore_league.py --snapshot before_v5_run_202609211333 --run v5_run
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import io
import json
import os
import re
import shutil
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEADERBOARD = "logs/trueskill_leaderboard.json"
RESULTS = "logs/trueskill_leaderboard.results.jsonl"
STATE = "logs/league_state.json"
PATTERN = re.compile(r'checkpoints(?:/|\\\\|\\)+checkpoint_iter_(\d+)\.pt')


def _p(rel: str) -> str:
    return os.path.join(ROOT, rel)


def _training_is_live() -> Tuple[bool, str]:
    spec = importlib.util.spec_from_file_location("fresh_run_liveness", os.path.join(ROOT, "scripts", "fresh_run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = ROOT
    return mod.training_is_live()


def plan(snapshot: str, run: str) -> Dict[str, Any]:
    """The restored leaderboard, state and results text, and every reference that did not resolve."""
    dest = f"checkpoints/archive/{run}"
    missing: List[str] = []

    def repoint(text: str) -> str:
        def one(m: "re.Match[str]") -> str:
            target = f"{dest}/checkpoint_iter_{m.group(1)}.pt"
            if os.path.exists(_p(target)):
                return target
            missing.append(m.group(0))
            return m.group(0)
        return PATTERN.sub(one, text)

    lb = json.load(io.open(_p(f"{LEADERBOARD}.{snapshot}"), encoding="utf-8"))
    ratings = {}
    renamed = 0
    for k, r in lb.get("ratings", {}).items():
        nk = repoint(k)
        if nk != k:
            r = dict(r, path=repoint(r.get("path", k)), name=f"{run.split('_')[0]}/{r.get('name')}")
            renamed += 1
        ratings[nk] = r
    lb["ratings"] = ratings
    lb["history"] = json.loads(repoint(json.dumps(lb.get("history", []))))
    state_text = repoint(io.open(_p(f"{STATE}.{snapshot}"), encoding="utf-8").read())
    results_path = _p(f"{RESULTS}.{snapshot}")
    results_text = repoint(io.open(results_path, encoding="utf-8").read()) if os.path.exists(results_path) else None

    # A rated checkpoint that is gone cannot be an opponent; the league would silently drop it.
    unrated_missing = sorted({k for k, r in ratings.items()
                              if not r.get("is_anchor") and k != "heuristic" and not os.path.exists(_p(k))})
    state = json.loads(state_text)
    return {"leaderboard": lb, "state_text": state_text, "results_text": results_text,
            "renamed": renamed, "missing_refs": sorted(set(missing)), "missing_ratings": unrated_missing,
            "king": state.get("king_of_the_hill"), "pool": state.get("elite_pool", [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="backup suffix, e.g. before_v5_run_202609211333")
    ap.add_argument("--run", required=True, help="archive folder the snapshot's checkpoints now live in, e.g. v5_run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for rel in (LEADERBOARD, STATE):
        if not os.path.exists(_p(f"{rel}.{args.snapshot}")):
            print(f"no {rel}.{args.snapshot}")
            return 1
    live, detail = _training_is_live()
    if live:
        print(f"training is live ({detail}); the trainer would overwrite the restored league")
        return 1

    p = plan(args.snapshot, args.run)
    print(f"ratings: {len(p['leaderboard']['ratings'])}, repointed into checkpoints/archive/{args.run}/: {p['renamed']}")
    print(f"king: {p['king']}")
    print(f"elite pool: {len(p['pool'])} -- {', '.join(os.path.basename(x) for x in p['pool'])}")
    if p["missing_refs"] or p["missing_ratings"]:
        print(f"unresolved references: {len(p['missing_refs'])}; rated checkpoints missing: {len(p['missing_ratings'])}")
        for m in (p["missing_ratings"] or p["missing_refs"])[:20]:
            print(f"  {m}")
        print("refusing: every rated checkpoint in the snapshot must exist")
        return 1
    if args.dry_run:
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    for rel in (LEADERBOARD, STATE, RESULTS):
        if os.path.exists(_p(rel)):
            shutil.copyfile(_p(rel), _p(f"{rel}.before_restore_{stamp}"))
    with io.open(_p(LEADERBOARD), "w", encoding="utf-8") as f:
        json.dump(p["leaderboard"], f, indent=2)
    with io.open(_p(STATE), "w", encoding="utf-8") as f:
        f.write(p["state_text"])
    if p["results_text"] is not None:
        with io.open(_p(RESULTS), "w", encoding="utf-8", newline="\n") as f:
            f.write(p["results_text"])
    elif os.path.exists(_p(RESULTS)):
        os.remove(_p(RESULTS))
    print(f"restored {args.snapshot}; previous league at *.before_restore_{stamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
