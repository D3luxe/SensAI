"""
Rebases the TrueSkill leaderboard onto the calibrated anchor ladder.

Every existing rating was solved against a scale that pinned all anchors at mu 25.0 --
asserting that the BC baseline (46W-4536L) and Necto (2865W-611L) were equally strong --
and against a declared draw probability of 0.05 while games actually drew ~24% of the
time. Those numbers cannot be converted onto the new scale, because the error is in the
constraints the solver used, not in a units mismatch. They have to be re-earned.

This script backs up the leaderboard and league state, then clears the learned ratings
while re-pinning the anchors to ANCHOR_CALIBRATION. Checkpoints on disk survive; they
simply re-enter the league as un-rated and get graded against the corrected ladder.

STOP TRAINING BEFORE RUNNING THIS. The trainer's grading subprocess writes both files.

    python scripts/recalibrate_leaderboard.py            # dry run, prints what changes
    python scripts/recalibrate_leaderboard.py --apply    # back up and rewrite
"""

from __future__ import annotations
import argparse
import datetime
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.trueskill_evaluator import (  # noqa: E402
    ANCHOR_CALIBRATION, ANCHOR_SIGMA, DEFAULT_DRAW_PROBABILITY, get_anchor_calibration
)

LEADERBOARD = "logs/trueskill_leaderboard.json"
LEAGUE_STATE = "logs/league_state.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    parser.add_argument("--leaderboard", default=LEADERBOARD)
    parser.add_argument("--league-state", default=LEAGUE_STATE)
    args = parser.parse_args()

    if not os.path.exists(args.leaderboard):
        print(f"No leaderboard at {args.leaderboard}; nothing to do.")
        return 0

    with open(args.leaderboard, "r", encoding="utf-8") as f:
        data = json.load(f)
    ratings = data.get("ratings", {})

    anchors, learned = {}, 0
    for key, rec in ratings.items():
        calib = get_anchor_calibration(rec.get("path", key))
        if calib and rec.get("is_anchor"):
            anchors[key] = calib
        else:
            learned += 1

    print(f"Leaderboard: {args.leaderboard}")
    print(f"  learned ratings to clear : {learned}")
    print(f"  anchors to re-pin        : {len(anchors)}")
    print(f"  draw probability         : {DEFAULT_DRAW_PROBABILITY}")
    print("\nCalibrated ladder:")
    for name, mu in sorted(ANCHOR_CALIBRATION.items(), key=lambda kv: kv[1]):
        print(f"  {name:22s} mu={mu:5.1f}  sigma={ANCHOR_SIGMA}")

    for key, (mu, sigma) in anchors.items():
        old = ratings[key]
        print(f"\n  {key}\n    mu {old.get('mu'):.2f} -> {mu:.2f}   sigma {old.get('sigma'):.3f} -> {sigma}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to write.")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in (args.leaderboard, args.league_state):
        if os.path.exists(path):
            backup = f"{path}.pre_recalibration_{stamp}.bak"
            shutil.copyfile(path, backup)
            print(f"Backed up {path} -> {backup}")

    kept = {}
    for key, (mu, sigma) in anchors.items():
        rec = dict(ratings[key])
        rec.update({
            "mu": mu, "sigma": sigma, "conservative_rating": round(mu - 3.0 * sigma, 2),
            "matches_played": 0, "wins": 0, "losses": 0, "draws": 0,
            "goals_for": 0, "goals_against": 0, "goal_diff": 0, "win_rate": 0.0,
            "is_anchor": True, "rating_locked": False, "locked_at_matches": 0,
            "last_updated": datetime.datetime.now().isoformat(),
        })
        kept[key] = rec

    data["ratings"] = kept
    data["history"] = []
    with open(args.leaderboard, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Rewrote {args.leaderboard} with {len(kept)} calibrated anchors and no learned ratings.")

    if os.path.exists(args.league_state):
        os.remove(args.league_state)
        print(f"Removed {args.league_state}; it will be rebuilt on the next grading pass.")

    print("\nDone. Restart training to repopulate the league against the corrected ladder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
