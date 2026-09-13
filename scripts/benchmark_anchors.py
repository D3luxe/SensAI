"""
Absolute strength check: play a checkpoint against the fixed anchors and report win rates.

WHY THIS EXISTS

The TrueSkill ladder in logs/trueskill_leaderboard.json cannot answer "is this model good".
Anchors are pinned there -- Necto at 30.0, the heuristic at 15.0 -- but across the last 500
rated matches, ZERO involved an anchor. The checkpoint ladder and the anchor scale are two
disconnected subgraphs, so a checkpoint reading mu 138 has never had a result propagate to or
from anything with a fixed rating.

That makes the ladder a measure of "beats slightly older versions of itself", which inflates
without bound by construction. It rises forever on a policy going sideways in absolute terms,
as long as each checkpoint edges the one before, so a flattening mu is NOT a usable stopping
criterion. A win rate against a fixed external opponent is, because nothing about it can drift.

Run it every few thousand iterations. A win rate that stops climbing is real evidence the run
has given what it has.

USAGE

    python scripts/benchmark_anchors.py                      # latest_model.pt vs all anchors
    python scripts/benchmark_anchors.py --series 12          # tighter, slower
    python scripts/benchmark_anchors.py --opponents necto    # just one
    python scripts/benchmark_anchors.py checkpoints/checkpoint_iter_20000.pt
    python scripts/benchmark_anchors.py --history            # print past runs and exit

Results append to logs/anchor_benchmark.csv.

NOTE ON THE LEADERBOARD

This writes its ratings to a scratch file, never to logs/trueskill_leaderboard.json. Benchmark
games are not league games and must not move league standings, and evaluate_pairing saves the
leaderboard it was constructed with.
"""
import argparse
import io
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CSV = os.path.join(ROOT, "logs", "anchor_benchmark.csv")
SCRATCH_LEADERBOARD = os.path.join(ROOT, "logs", "_benchmark_scratch_leaderboard.json")

ANCHORS = {
    "necto": "checkpoints/necto-model.pt",
    "nexto": "checkpoints/nexto-model.pt",
    "baseline": "checkpoints/pretrained_baseline.pt",
    "heuristic": "heuristic",
}
COLUMNS = ["timestamp", "iteration", "opponent", "series", "series_won",
           "series_win_pct", "games_for", "games_against", "goal_diff_per_series"]


def load_iteration(path):
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return ck.get("iteration", -1)
    except Exception:
        return -1


def append_csv(rows):
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    header, existing = None, []
    if os.path.exists(CSV):
        text = io.open(CSV, encoding="utf-8").read().strip()
        if text:
            lines = text.split("\n")
            header = lines[0].split(",")
            existing = [l.split(",") for l in lines[1:]]

    def fmt(v):
        return ("%.4f" % v) if isinstance(v, float) else str(v)

    new_lines = [",".join(fmt(r[c]) for c in COLUMNS) for r in rows]
    if header is None:
        io.open(CSV, "w", encoding="utf-8").write(
            ",".join(COLUMNS) + "\n" + "\n".join(new_lines) + "\n")
        return
    if header == COLUMNS:
        with io.open(CSV, "a", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        return
    out = [",".join(COLUMNS)]
    for r in existing:
        vals = dict(zip(header, r))
        out.append(",".join(vals.get(c, "") for c in COLUMNS))
    out.extend(new_lines)
    io.open(CSV, "w", encoding="utf-8").write("\n".join(out) + "\n")


def print_history(limit=30):
    if not os.path.exists(CSV):
        print("no benchmark history yet at %s" % os.path.relpath(CSV, ROOT))
        return
    text = io.open(CSV, encoding="utf-8").read().strip()
    rows = [l.split(",") for l in text.split("\n")]
    head, body = rows[0], rows[1:][-limit:]
    show = ["iteration", "opponent", "series", "series_win_pct", "goal_diff_per_series"]
    idx = [head.index(c) if c in head else None for c in show]
    print("%-10s %-12s %8s %14s %20s"
          % ("iteration", "opponent", "series", "series win %", "goal diff / series"))
    print("-" * 70)
    for r in body:
        print("%-10s %-12s %8s %14s %20s"
              % tuple((r[i] if i is not None and i < len(r) else "-") for i in idx))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", default="checkpoints/latest_model.pt")
    ap.add_argument("--series", type=int, default=8,
                    help="best-of-9 series per opponent (default 8)")
    ap.add_argument("--opponents", default="all",
                    help="comma-separated: necto,nexto,baseline,heuristic or 'all'")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    if args.history:
        print_history()
        return

    from utils.trueskill_evaluator import TrueSkillEvaluator

    subject = args.checkpoint
    if not os.path.isabs(subject):
        subject = os.path.join(ROOT, subject)
    if not os.path.exists(subject):
        print("no checkpoint at %s" % subject)
        sys.exit(1)

    which = list(ANCHORS) if args.opponents == "all" else [
        o.strip() for o in args.opponents.split(",") if o.strip()]
    unknown = [o for o in which if o not in ANCHORS]
    if unknown:
        print("unknown opponent(s): %s; known: %s" % (unknown, list(ANCHORS)))
        sys.exit(1)

    iteration = load_iteration(subject)
    print("subject: %s (iteration %s)" % (os.path.relpath(subject, ROOT), iteration))
    print("%d best-of-9 series per opponent\n" % args.series)

    # Scratch leaderboard: benchmark games must not move league standings.
    if os.path.exists(SCRATCH_LEADERBOARD):
        os.remove(SCRATCH_LEADERBOARD)
    ev = TrueSkillEvaluator(leaderboard_path=SCRATCH_LEADERBOARD)

    print("%-12s %8s %12s %14s %12s %18s"
          % ("opponent", "series", "series won", "series win %", "games", "goal diff/series"))
    print("-" * 82)
    rows = []
    for name in which:
        path = ANCHORS[name]
        full = path if path == "heuristic" else os.path.join(ROOT, path)
        if path != "heuristic" and not os.path.exists(full):
            print("%-12s   missing (%s)" % (name, path))
            continue
        t0 = time.time()
        try:
            res = ev.evaluate_pairing(subject, full if path != "heuristic" else "heuristic",
                                      series_per_pair=args.series)
        except Exception as e:
            print("%-12s   failed: %r" % (name, e))
            continue
        if not res:
            print("%-12s   no results" % name)
            continue
        won = sum(1 for r in res if r.get("winner") == "a")
        gf = sum(int(r.get("a_score", 0)) for r in res)
        ga = sum(int(r.get("b_score", 0)) for r in res)
        n = len(res)
        rows.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "iteration": iteration,
            "opponent": name, "series": n, "series_won": won,
            "series_win_pct": 100.0 * won / n, "games_for": gf, "games_against": ga,
            "goal_diff_per_series": (gf - ga) / float(n),
        })
        print("%-12s %8d %12d %13.0f%% %6d-%-5d %18.2f   (%.0fs)"
              % (name, n, won, 100.0 * won / n, gf, ga, (gf - ga) / float(n),
                 time.time() - t0))

    if rows and not args.no_log:
        append_csv(rows)
        print()
        print("appended to %s" % os.path.relpath(CSV, ROOT))
        print()
        print_history()

    if os.path.exists(SCRATCH_LEADERBOARD):
        os.remove(SCRATCH_LEADERBOARD)


if __name__ == "__main__":
    main()
