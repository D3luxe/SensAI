"""
Comprehensive automated tests for Rocket League Simulation, PPO Trainer, and Process Manager.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import torch
import unittest

from env.physics_engine import RocketSimArena, BoostPad
from env.rewards import RewardManager
from env.observations import DefaultObservationBuilder
from env.actions import ContinuousActionParser, DiscreteActionParser
from env.rocket_env import RocketLeagueEnv, VectorizedRocketEnv
from agent.models import ActorCritic
from agent.ppo import PPOTrainer
from utils.visualizer import simulate_match


class TestRocketLeagueEnvironment(unittest.TestCase):
    def test_physics_arena(self):
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset(random_kickoff=True)
        self.assertEqual(len(arena.cars), 2)
        self.assertGreater(len(arena.boost_pads), 0)

        # Step arena with random actions
        actions = [np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32) for _ in range(2)]
        goal, scoring_team = arena.step(actions, dt=1.0 / 15.0)
        self.assertIsInstance(goal, bool)

    def test_rewards_and_observations(self):
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset(random_kickoff=True)

        obs_builder = DefaultObservationBuilder(symmetric=True)
        obs0 = obs_builder.build_obs(arena.cars[0], arena)
        self.assertEqual(len(obs0), obs_builder.obs_dim)
        self.assertFalse(np.isnan(obs0).any())

        rew_manager = RewardManager()
        rew, rew_dict = rew_manager.get_reward(arena.cars[0], arena, np.zeros(8), False, None)
        self.assertIsInstance(rew, float)
        self.assertIn("goal", rew_dict)
        self.assertIn("ball_strike", rew_dict)
        self.assertIn("locomotion", rew_dict)
        self.assertIn("aerial", rew_dict)
        self.assertIn("positioning", rew_dict)
        self.assertIn("boost_economy", rew_dict)

    def test_vectorized_env(self):
        vec_env = VectorizedRocketEnv(num_envs=4, game_mode="1v1", tick_skip=4)
        obs = vec_env.reset()
        self.assertEqual(obs.shape, (4, 2, vec_env.obs_dim))

        actions = np.zeros((4, 2, 8), dtype=np.float32)
        actions[:, :, 0] = 1.0  # Full throttle
        next_obs, rews, dones, infos = vec_env.step(actions)

        self.assertEqual(next_obs.shape, (4, 2, vec_env.obs_dim))
        self.assertEqual(rews.shape, (4, 2))
        self.assertEqual(len(infos), 4)

    def test_actor_critic_model(self):
        model = ActorCritic(obs_dim=64, act_dim=8, continuous_actions=True)
        obs_tensor = torch.randn(8, 64)
        action, log_prob, entropy, value = model.get_action_and_value(obs_tensor)
        self.assertEqual(action.shape, (8, 8))
        self.assertEqual(log_prob.shape, (8,))
        self.assertEqual(entropy.shape, (8,))
        self.assertEqual(value.shape, (8, 1))

    def test_mini_ppo_training_run(self):
        trainer = PPOTrainer(config_path="config/default_config.yaml")
        # Run 2 training iterations
        trainer.train(max_iterations=2)

        # Check that metrics were generated
        self.assertTrue(os.path.exists("logs/metrics.json"))
        self.assertTrue(os.path.exists("logs/history.jsonl"))

    def test_physics_and_controls_preflight(self):
        from test_physics_and_controls import verify_physics_and_controls_pipeline
        verified = verify_physics_and_controls_pipeline(verbose=False)
        self.assertTrue(verified)


if __name__ == "__main__":
    unittest.main()
