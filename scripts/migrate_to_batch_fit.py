"""
One-off migration of the live leaderboard from online TrueSkill to the batch fit (2026-09-21).

  1. backs up logs/trueskill_leaderboard.json and logs/league_state.json (*.before_batchfit.json)
  2. rebuilds the results log from what survives: the leaderboard's last 500 series, and the
     benchmark series against Necto and Nexto played since the v5 league was rebuilt
  3. resets every fitted rating to the prior, so nothing carries over from the drifted scale
  4. plays --cal-series calibration series for the king and each elite member against the
     pinned v3 king, which is what ties the fit to mu 25
  5. refits and prints the board

Tallies (W-L-D, goals, calibration counts) are kept: they are the raw record, not the scale.

    python scripts/migrate_to_batch_fit.py --dry-run     # steps 1-3 and a refit, no games
    python scripts/migrate_to_batch_fit.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import rating_fit  # noqa: E402
from utils.config import effective_config  # noqa: E402
from utils.trueskill_evaluator import (  # noqa: E402
    DEFAULT_LEADERBOARD_PATH, TrueSkillEvaluator, get_anchor_calibration,
)

STATE_PATH = "logs/league_state.json"
V3 = "checkpoints/baselines/v3_iter198000.pt"
LEAGUE_REBUILT_AT = "2026-09-20T21:41"      # the v5 league's rebuild; older names are reused
BENCH_NAMES = {"Necto (EARL TorchScript)": "checkpoints/necto-model.pt",
               "Nexto (EARL TorchScript)": "checkpoints/nexto-model.pt"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cal-series", type=int, default=6)
    args = ap.parse_args()

    ev = TrueSkillEvaluator(DEFAULT_LEADERBOARD_PATH)
    if os.path.exists(ev.results_path) and not args.dry_run:
        print(f"{ev.results_path} already exists; refusing to migrate twice")
        return 1
    for p in (DEFAULT_LEADERBOARD_PATH, STATE_PATH):
        backup = p.replace(".json", ".before_batchfit.json")
        if not os.path.exists(backup):
            shutil.copyfile(p, backup)
            print(f"backed up {p} -> {backup}")

    state = json.load(open(STATE_PATH, encoding="utf-8"))
    by_name = {r.name: k for k, r in ev.ratings.items()}
    by_name.update(BENCH_NAMES)

    rows = []
    history = json.load(open(DEFAULT_LEADERBOARD_PATH, encoding="utf-8")).get("history", [])
    for h in history:
        a, b = by_name.get(h.get("model_a")), by_name.get(h.get("model_b"))
        if a and b:
            rows.append((a, b, h["a_score"], h["b_score"], "migrated"))
    for event in state.get("benchmark_history", []):
        if event.get("at", "") < LEAGUE_REBUILT_AT:
            continue
        subject = by_name.get(event.get("subject"))
        for opp_name, r in event.get("results", {}).items():
            opp = by_name.get(opp_name)
            if not subject or not opp:
                continue
            won, n = int(r.get("series_won", 0)), int(r.get("series", 0))
            rows += [(subject, opp, 5, 0, "benchmark")] * won
            rows += [(subject, opp, 0, 5, "benchmark")] * (n - won)
    print(f"{len(rows)} series recovered ({sum(1 for r in rows if r[4] == 'benchmark')} benchmark)")

    ev.get_or_create_rating(V3, is_anchor=True)
    for r in ev.ratings.values():
        if get_anchor_calibration(r.path) is None:
            r.mu, r.sigma, r.prior_mu, r.rating_locked = 25.0, 25.0 / 3.0, 25.0, False

    if args.dry_run:
        ev.results_path = os.path.join(os.path.dirname(ev.results_path), "_dryrun.results.jsonl")
        if os.path.exists(ev.results_path):
            os.remove(ev.results_path)
    at = datetime.datetime.now().isoformat()
    for a, b, sa, sb, kind in rows:
        ev.results.append(rating_fit.append_result(ev.results_path, a, b, sa, sb, kind, at))
    ev.refit()

    if not args.dry_run:
        ev.save_leaderboard()
        league = effective_config().get("league", {})
        targets = [state.get("king_of_the_hill")] + list(state.get("elite_pool", []))
        for path in dict.fromkeys(t for t in targets if t and os.path.exists(t)):
            ev.evaluate_pairing(
                path, V3, series_per_pair=args.cal_series,
                max_steps=league.get("eval_max_steps", 3000),
                no_touch_steps=league.get("eval_no_touch_steps", 500),
                series_length=league.get("series_length", 9),
                wins_needed=league.get("series_wins_needed", 5),
                calibration=True)
            r = ev.ratings[ev._key(path)]
            print(f"  calibrated {r.name}: mu {r.mu:.2f} +/- {r.sigma:.2f}", flush=True)

    fitted = sorted((r for r in ev.ratings.values() if r.sigma < 8.0 or r.is_anchor),
                    key=lambda r: -r.mu)
    print(f"\n{'model':32}{'mu':>8}{'sigma':>8}{'vs v3':>8}")
    for r in fitted[:25]:
        print(f"{r.name[:32]:32}{r.mu:8.2f}{r.sigma:8.2f}{r.mu - 25.0:+8.2f}")
    if args.dry_run and os.path.exists(ev.results_path):
        os.remove(ev.results_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
