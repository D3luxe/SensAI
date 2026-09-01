"""
Unit Tests for Baseline & Mixup Opponent Bots (Necto, Nexto, Sensei Checkpoints, Heuristics).
"""

from __future__ import annotations
import os
import unittest
import numpy as np
import torch

from env.physics_engine import RocketSimArena
from env.baseline_agent import (
    BaselineChaser, CheckpointOpponentBot, NectoNextoOpponentBot, create_opponent_bot
)
from env.rocket_env import RocketLeagueEnv, VectorizedRocketEnv
from utils.visualizer import simulate_match


class TestOpponentBots(unittest.TestCase):
    def test_factory_creation(self):
        b1 = create_opponent_bot("heuristic")
        self.assertIsInstance(b1, BaselineChaser)

        if os.path.exists("checkpoints/necto-model.pt"):
            b2 = create_opponent_bot("checkpoints/necto-model.pt")
            self.assertIsInstance(b2, NectoNextoOpponentBot)
            self.assertFalse(b2.is_nexto)

        if os.path.exists("checkpoints/nexto-model.pt"):
            b3 = create_opponent_bot("checkpoints/nexto-model.pt")
            self.assertIsInstance(b3, NectoNextoOpponentBot)
            self.assertTrue(b3.is_nexto)

        if os.path.exists("checkpoints/pretrained_baseline.pt"):
            b4 = create_opponent_bot("checkpoints/pretrained_baseline.pt")
            self.assertIsInstance(b4, CheckpointOpponentBot)

    def test_bot_actions(self):
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset()

        candidates = [
            "heuristic",
            "checkpoints/necto-model.pt",
            "checkpoints/nexto-model.pt",
            "checkpoints/pretrained_baseline.pt"
        ]
        for name in candidates:
            if not os.path.exists(name) and name != "heuristic":
                continue
            bot = create_opponent_bot(name)
            act = bot.get_action(arena.cars[1], arena)
            self.assertEqual(len(act), 8, f"Action length for {name} should be 8")
            self.assertTrue(np.all(np.isfinite(act)), f"Actions for {name} must be finite")

    def test_vectorized_env_mixup(self):
        vec_env = VectorizedRocketEnv(
            num_envs=4,
            game_mode="1v1",
            baseline_opponent_ratio=0.5,
            baseline_opponent_type="heuristic"
        )
        self.assertEqual(sum(env.is_baseline_env for env in vec_env.envs), 2)

        # Dynamic hot-swap to Nexto / Necto
        opp = "checkpoints/nexto-model.pt" if os.path.exists("checkpoints/nexto-model.pt") else "heuristic"
        vec_env.update_baseline_opponent(0.75, opp)
        self.assertEqual(sum(env.is_baseline_env for env in vec_env.envs), 3)

        obs = vec_env.reset()
        dummy_actions = np.zeros((4, 2, 8), dtype=np.float32)
        next_obs, rews, dones, infos = vec_env.step(dummy_actions)
        self.assertEqual(next_obs.shape, (4, 2, 74))

    def test_simulate_match(self):
        opp = "checkpoints/necto-model.pt" if os.path.exists("checkpoints/necto-model.pt") else "baseline"
        pitch_fig, rew_fig, stats = simulate_match(blue_model_path=None, orange_model_path=opp, max_steps=20)
        self.assertIsNotNone(pitch_fig)
        self.assertIsNotNone(rew_fig)
        self.assertEqual(stats["simulation_steps"], 20)


if __name__ == "__main__":
    unittest.main()
