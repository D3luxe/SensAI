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
        self.assertIn("ball_to_goal", rew_dict)
        self.assertIn("player_to_ball", rew_dict)
        self.assertIn("touch", rew_dict)
        self.assertIn("boost", rew_dict)

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
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(config_path="config/default_config.yaml")
            trainer.save_dir = tmpdir
            # Run 2 training iterations
            trainer.train(max_iterations=2)

        # Check that metrics were generated
        self.assertTrue(os.path.exists("logs/metrics.json"))
        self.assertTrue(os.path.exists("logs/history.jsonl"))

    def test_baseline_chaser_and_vectorized_partitioning(self):
        from env.baseline_agent import BaselineChaser
        from env.physics_engine import CarState, BallState

        bot = BaselineChaser(continuous_actions=True)
        car = CarState(id=1, team=1, pos=np.array([0.0, 4608.0, 17.0], dtype=np.float32), rot=np.array([0.0, -np.pi/2, 0.0], dtype=np.float32))
        ball = BallState(pos=np.array([0.0, 0.0, 93.15], dtype=np.float32))
        act = bot.get_action(car, ball)
        self.assertEqual(len(act), 8)
        self.assertGreater(act[0], 0.5, "Baseline chaser must drive full throttle toward ball on kickoff")

        # Test Vectorized partition
        vec_env = VectorizedRocketEnv(num_envs=4, game_mode="1v1", baseline_opponent_ratio=0.5)
        mask = vec_env.get_learner_mask()
        self.assertEqual(len(mask), 8)
        # First 2 envs (4 actors) self-play -> True, True, True, True
        # Last 2 envs (4 actors) baseline -> True, False, True, False
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertTrue(mask[4])
        self.assertFalse(mask[5])

    def test_physics_and_controls_preflight(self):
        from test_physics_and_controls import verify_physics_and_controls_pipeline
        verified = verify_physics_and_controls_pipeline(verbose=False)
        self.assertTrue(verified)

    def test_anti_own_goal_rewards(self):
        """Verify that pushing/touching the ball towards defending net is strictly penalized."""
        from env.physics_engine import CarState, BallState, RocketSimArena
        from env.rewards import TouchBallReward, PlayerToBallVelocityReward, BallToGoalVelocityReward

        arena = RocketSimArena(num_players=2, game_mode="1v1")
        car0 = arena.cars[0]  # Team 0: Defending -5120, attacking +5120
        car0.pos = np.array([0.0, 1000.0, 17.0], dtype=np.float32)
        car0.vel = np.array([0.0, -1200.0, 0.0], dtype=np.float32)  # Moving towards defending net (-Y)
        car0.rot = np.array([0.0, -np.pi/2, 0.0], dtype=np.float32) # Facing -Y

        # Ball in front of car, also moving towards defending net
        arena.ball.pos = np.array([0.0, 800.0, 93.15], dtype=np.float32)
        arena.ball.vel = np.array([0.0, -1200.0, 0.0], dtype=np.float32)

        # 1. Test BallToGoalVelocityReward (asymmetric penalty)
        b2g = BallToGoalVelocityReward(weight=1.5)
        rew_b2g = b2g.get_reward(car0, arena, np.zeros(8), False, None)
        self.assertLess(rew_b2g, 0.0, "Ball moving towards defending goal must yield negative progression reward")

        # 2. Test PlayerToBallVelocityReward (wrong-side penalty)
        p2b = PlayerToBallVelocityReward(weight=0.15)
        p2b.reset(arena)
        rew_p2b = p2b.get_reward(car0, arena, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), False, None)
        self.assertLessEqual(rew_p2b, 0.0, "Chasing/pushing ball towards defending goal must not give positive matching reward")

        # 3. Test TouchBallReward on bad touch towards own net
        touch_rew = TouchBallReward(weight=1.2)
        touch_rew.reset(arena)
        car0.ball_touches += 1
        rew_touch = touch_rew.get_reward(car0, arena, np.zeros(8), False, None)
        self.assertLess(rew_touch, 0.0, "Touching ball directly towards defending goal must yield a negative penalty")


if __name__ == "__main__":
    unittest.main()

