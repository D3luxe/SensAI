"""
Standalone CLI Training Script for Rocket League Reinforcement Learning Bot.
"""

from __future__ import annotations
import argparse
import sys
from agent.ppo import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Rocket League ML Bot using Vectorized PPO")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to base config YAML")
    parser.add_argument("--live-config", type=str, default="config/live_config.json", help="Path to live dynamic config JSON")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file to resume training")
    parser.add_argument("--iterations", type=int, default=None, help="Maximum iterations to train")
    parser.add_argument("--device", type=str, default=None, help="Torch device ('cuda' or 'cpu')")

    args = parser.parse_args()

    print("=" * 60)
    print("      ROCKET LEAGUE MACHINE LEARNING BOT TRAINER      ")
    print("=" * 60)
    print(f"Base Config: {args.config}")
    print(f"Live Config: {args.live_config}")
    if args.checkpoint:
        print(f"Resuming Checkpoint: {args.checkpoint}")
    print("=" * 60)

    # 1. Automatic Pre-Flight Physics & Controls Verification
    from test_physics_and_controls import verify_physics_and_controls_pipeline
    try:
        verify_physics_and_controls_pipeline(verbose=True)
    except Exception as e:
        print(f"\n[FATAL] Pre-Flight Physics Verification Failed: {e}")
        sys.exit(1)

    trainer = PPOTrainer(
        config_path=args.config,
        live_config_path=args.live_config,
        device=args.device
    )

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    try:
        trainer.train(max_iterations=args.iterations)
    except KeyboardInterrupt:
        print("\n[Trainer] Training interrupted by user. Saving final checkpoint...")
        trainer.save_checkpoint("checkpoints/interrupted_checkpoint.pt")
        print("[Trainer] Checkpoint saved. Exiting.")


if __name__ == "__main__":
    main()
