"""
Model Evaluation and Match Benchmark Script.
Evaluates a trained policy against baselines across multiple episodes.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import torch
from utils.visualizer import simulate_match


def main():
    import sys
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Evaluate Rocket League Policy")
    parser.add_argument("--model", type=str, default="checkpoints/latest_model.pt", help="Path to checkpoint model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--save-plot", type=str, default="logs/eval_match.png", help="Save visualization plot")
    parser.add_argument("--trueskill", action="store_true", help="Update TrueSkill rating and benchmark against baselines")
    args = parser.parse_args()

    print("=" * 60)
    print("      ROCKET LEAGUE BOT EVALUATION BENCHMARK          ")
    print("=" * 60)
    print(f"Model Path: {args.model}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    total_blue_goals = 0
    total_orange_goals = 0
    total_blue_touches = 0
    total_orange_touches = 0
    blue_wins = 0

    last_fig = None

    for ep in range(args.episodes):
        pitch_fig, reward_fig, stats = simulate_match(blue_model_path=args.model, orange_model_path="baseline", max_steps=args.steps)
        last_fig = pitch_fig

        bg = stats["blue_goals"]
        og = stats["orange_goals"]
        bt = stats["blue_touches"]
        ot = stats["orange_touches"]

        total_blue_goals += bg
        total_orange_goals += og
        total_blue_touches += bt
        total_orange_touches += ot

        if bg > og:
            blue_wins += 1
            result = "WIN"
        elif bg < og:
            result = "LOSS"
        else:
            result = "DRAW"

        print(f"Episode {ep+1:02d}/{args.episodes:02d}: {result} | Blue Goals: {bg} vs Orange: {og} | Touches: {bt} vs {ot}")

    win_rate = (blue_wins / args.episodes) * 100.0
    print("=" * 60)
    print(f"SUMMARY RESULTS ({args.episodes} Matches):")
    print(f"Win Rate: {win_rate:.1f}% ({blue_wins}/{args.episodes})")
    print(f"Total Goals: {total_blue_goals} (Blue) vs {total_orange_goals} (Orange)")
    print(f"Avg Touches/Game: {total_blue_touches / args.episodes:.1f} (Blue) vs {total_orange_touches / args.episodes:.1f} (Orange)")
    print("=" * 60)

    if last_fig and args.save_plot:
        os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
        last_fig.savefig(args.save_plot, dpi=120, bbox_inches="tight")
        print(f"Saved evaluation trajectory plot to: {args.save_plot}")

    if args.trueskill:
        print("\n" + "=" * 60)
        print("          TRUESKILL BAYESIAN RATING BENCHMARK          ")
        print("=" * 60)
        from utils.trueskill_evaluator import TrueSkillEvaluator
        evaluator = TrueSkillEvaluator()
        anchors = ["heuristic"]
        if os.path.exists("checkpoints/pretrained_baseline.pt") and args.model != "checkpoints/pretrained_baseline.pt":
            anchors.append("checkpoints/pretrained_baseline.pt")

        print(f"Evaluating {args.model} against reference anchors: {anchors}...")
        for anchor in anchors:
            evaluator.evaluate_pairing(
                model_a_path=args.model,
                model_b_path=anchor,
                series_per_pair=1,
                max_steps=args.steps,
            )

        rec = evaluator.get_or_create_rating(args.model)
        print(f"Updated TrueSkill for {rec.name}:")
        print(f"  Rating (mu):        {rec.mu:.2f}")
        print(f"  Uncertainty (sigma): +/-{rec.sigma:.2f}")
        print(f"  Conservative Score: {rec.conservative_rating:.2f}")
        print(f"  Record:             {rec.wins}W - {rec.losses}L - {rec.draws}D (Win Rate: {rec.win_rate:.1f}%)")
        print("=" * 60)


if __name__ == "__main__":
    main()
