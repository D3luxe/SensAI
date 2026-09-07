"""
Standalone CLI Benchmark Script for TrueSkill Evaluation of Rocket League Bots.
Runs symmetric round-robin tournaments across saved checkpoints and baseline bots.
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import pandas as pd
from tabulate import tabulate

from utils.trueskill_evaluator import (
    TrueSkillEvaluator, get_model_display_name, DEFAULT_LEADERBOARD_PATH
)


def parse_args():
    parser = argparse.ArgumentParser(description="SensAI TrueSkill Tournament & Model Benchmark")
    parser.add_argument(
        "--checkpoints", "-c", nargs="*", default=[],
        help="List of model checkpoint .pt paths to evaluate"
    )
    parser.add_argument(
        "--all-checkpoints", action="store_true",
        help="Automatically include all valid checkpoints found in checkpoints/"
    )
    parser.add_argument(
        "--include-baselines", action="store_true", default=True,
        help="Include reference anchors (BaselineChaser, Pretrained Baseline, Necto/Nexto if available)"
    )
    parser.add_argument(
        "--matches-per-pair", type=int, default=2,
        help="Number of symmetric matches per pair (alternating Blue/Orange; min 2, rounded up to even)"
    )
    parser.add_argument(
        "--steps", type=int, default=400,
        help="Maximum simulation steps per match (400 steps ≈ 26.6s in-game)"
    )
    parser.add_argument(
        "--no-overtime", action="store_true",
        help="Disable golden-goal sudden death overtime on ties"
    )
    parser.add_argument(
        "--save-plot", type=str, default="logs/trueskill_ratings.png",
        help="Path to export the leaderboard rating plot"
    )
    parser.add_argument(
        "--leaderboard-path", type=str, default=DEFAULT_LEADERBOARD_PATH,
        help="Path to persistent JSON leaderboard"
    )
    parser.add_argument(
        "--reset-leaderboard", action="store_true",
        help="Clear existing leaderboard before starting tournament"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device for evaluation ('cpu' recommended to avoid training VRAM conflicts)"
    )
    return parser.parse_args()


def main():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = parse_args()

    print("=" * 70)
    print("       SENSAI TRUESKILL TOURNAMENT & MODEL BENCHMARK          ")
    print("=" * 70)
    print(f"Matches Per Pairing: {args.matches_per_pair} (Symmetric Home/Away)")
    print(f"Max Steps / Match:   {args.steps} (Overtime: {'Off' if args.no_overtime else 'Sudden-Death'})")
    print(f"Device:              {args.device}")
    print(f"Leaderboard Store:   {args.leaderboard_path}")
    print("=" * 70)

    evaluator = TrueSkillEvaluator(leaderboard_path=args.leaderboard_path)

    if args.reset_leaderboard:
        print("[TrueSkill] Resetting persistent leaderboard...")
        evaluator.reset_leaderboard()

    models_to_eval = []

    # 1. Explicit checkpoints
    for p in args.checkpoints:
        if os.path.exists(p) or p.lower() in ("heuristic", "baseline"):
            models_to_eval.append(p)

    # 2. All checkpoints scan
    if args.all_checkpoints:
        found = glob.glob("checkpoints/*.pt")
        found.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        # Add latest model first, then select representative checkpoints
        for f in found:
            if not f.endswith(".bak") and not f.endswith(".tmp"):
                models_to_eval.append(f)

    # 3. Reference baselines & anchors
    if args.include_baselines:
        models_to_eval.append("heuristic")
        for anchor in [
            "checkpoints/pretrained_baseline.pt",
            "checkpoints/necto-model.pt",
            "checkpoints/nexto-model.pt",
            "checkpoints/latest_model.pt"
        ]:
            if os.path.exists(anchor) and anchor not in models_to_eval:
                models_to_eval.append(anchor)

    # Deduplicate while preserving order
    models_to_eval = list(dict.fromkeys([os.path.normpath(m).replace("\\", "/") for m in models_to_eval]))

    if len(models_to_eval) < 2:
        print("\n[TrueSkill] Error: At least 2 models are required to run a tournament.")
        print("Please provide models via --checkpoints or use --all-checkpoints.")
        sys.exit(1)

    print(f"\nTournament Contestants ({len(models_to_eval)} models):")
    for i, m in enumerate(models_to_eval, 1):
        print(f"  {i}. {get_model_display_name(m)} ({m})")
    print("-" * 70)

    # Execute tournament
    for update in evaluator.run_tournament(
        model_paths=models_to_eval,
        matches_per_pair=args.matches_per_pair,
        max_steps=args.steps,
        enable_overtime=not args.no_overtime,
        device=args.device
    ):
        p_idx = update["pairing_index"]
        total_p = update["total_pairings"]
        mA = update["model_a"]
        mB = update["model_b"]
        res_list = update["results"]

        summary_parts = []
        for r in res_list:
            w = r["winner_name"]
            ot = " (OT)" if r["overtime"] else ""
            summary_parts.append(f"{r['blue_name']} {r['blue_goals']}-{r['orange_goals']} {r['orange_name']}{ot}")

        print(f"[{p_idx:02d}/{total_p:02d}] {mA} vs {mB} -> {', '.join(summary_parts)}")

    # Print Final Leaderboard Table
    df = evaluator.get_leaderboard_dataframe()
    print("\n" + "=" * 85)
    print("                       FINAL TRUESKILL LEADERBOARD                      ")
    print("=" * 85)
    if not df.empty:
        try:
            print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
        except Exception:
            try:
                print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
            except Exception:
                print(df.to_string(index=False))
    else:
        print("No matches played.")
    print("=" * 85)

    # Render and save plot
    if args.save_plot:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_plot)), exist_ok=True)
        fig = evaluator.render_leaderboard_plot()
        fig.savefig(args.save_plot, dpi=130, bbox_inches="tight")
        print(f"\nSaved TrueSkill rating distribution plot to: {args.save_plot}")


if __name__ == "__main__":
    main()
