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

    def _anneal_stub(self, targets, base, decay_steps, global_step):
        """
        A minimal stand-in for PPOTrainer carrying only the reward-annealing state, bound to the
        real methods. Constructing a trainer spawns the whole worker pool, which would dominate
        a test of two pieces of arithmetic.
        """
        from agent.ppo import PPOTrainer

        class Stub:
            pass

        stub = Stub()
        stub.reward_anneal_enabled = True
        stub.reward_anneal_targets = dict(targets)
        stub.reward_anneal_steps = int(decay_steps)
        stub.base_reward_weights = dict(base)
        stub._reward_anneal_start_steps = {}
        stub._last_pushed_reward_weights = None
        stub.global_step = int(global_step)
        stub.pushed = []

        class FakeEnv:
            def update_reward_weights(inner, weights):
                stub.pushed.append(dict(weights))

        stub.env = FakeEnv()
        stub._apply_reward_annealing = PPOTrainer._apply_reward_annealing.__get__(stub)
        stub._reward_anneal_progress = PPOTrainer._reward_anneal_progress.__get__(stub)
        stub._load_reward_anneal_clocks = PPOTrainer._load_reward_anneal_clocks.__get__(stub)
        return stub

    def test_a_target_added_mid_run_starts_its_own_anneal_clock(self):
        """
        Guarantees a newly configured target ramps from the current step rather than inheriting
        an elapsed schedule.

        Under a single shared clock, progress was pinned at 1.0 once the first target finished,
        so adding a second term dropped its weight to the final value on the very next iteration.
        That is a cliff in the reward function mid-run, not an anneal, and the policy has no
        chance to adapt to it gradually.
        """
        base = {"powerslide_weight": 0.2, "jump_bridge_weight": 0.55}
        targets = {"powerslide_weight": 0.0, "jump_bridge_weight": 0.15}
        stub = self._anneal_stub(targets, base, decay_steps=400_000_000, global_step=2_000_000_000)

        # powerslide has been running since step 0 and is long finished; jump_bridge is new.
        stub._reward_anneal_start_steps = {"powerslide_weight": 0}
        stub._apply_reward_annealing()

        pushed = stub.pushed[-1]
        self.assertAlmostEqual(pushed["powerslide_weight"], 0.0, places=6,
                               msg="a finished target must stay at its final value")
        self.assertAlmostEqual(pushed["jump_bridge_weight"], 0.55, places=6,
                               msg="a freshly added target must start at its base weight, not snap to the target")

        # Halfway through its own schedule it should be halfway down, with powerslide unmoved.
        stub.global_step = 2_200_000_000
        stub._apply_reward_annealing()
        pushed = stub.pushed[-1]
        self.assertAlmostEqual(pushed["jump_bridge_weight"], 0.35, places=6)
        self.assertAlmostEqual(pushed["powerslide_weight"], 0.0, places=6)

    def test_legacy_shared_anneal_clock_migrates_without_restarting_finished_targets(self):
        """
        Guarantees resuming a checkpoint written before the clocks were split keeps an already
        elapsed schedule elapsed, instead of handing the policy back a bootstrap reward it has
        grown out of.
        """
        base = {"powerslide_weight": 0.2}
        targets = {"powerslide_weight": 0.0}
        stub = self._anneal_stub(targets, base, decay_steps=400_000_000, global_step=2_161_983_488)

        stub._load_reward_anneal_clocks({
            "reward_anneal_start_step": 1_693_417_472,
            "config": {"reward_annealing": {"targets": {"powerslide_weight": 0.0}}},
        })
        self.assertEqual(stub._reward_anneal_start_steps, {"powerslide_weight": 1_693_417_472})

        stub._apply_reward_annealing()
        self.assertAlmostEqual(stub.pushed[-1]["powerslide_weight"], 0.0, places=6,
                               msg="a schedule that had already elapsed must not restart on resume")

    def test_legacy_clock_does_not_backdate_a_target_it_never_covered(self):
        """
        Guarantees the legacy shared clock is applied only to the targets the checkpoint was
        actually annealing, named by its own saved config.

        This is the bug that shipped: two targets added in the same release as the migration were
        seeded from a clock that had already run out, so both jumped to their final weights on the
        first iteration after the resume and per-step reward halved. A restarted ramp is cheap; a
        cliff is not, so an unprovable target gets a fresh clock.
        """
        base = {"powerslide_weight": 0.2, "jump_bridge_weight": 0.55, "player_to_ball_weight": 0.6}
        targets = {"powerslide_weight": 0.0, "jump_bridge_weight": 0.15, "player_to_ball_weight": 0.25}
        stub = self._anneal_stub(targets, base, decay_steps=400_000_000, global_step=2_190_344_192)

        stub._load_reward_anneal_clocks({
            "reward_anneal_start_step": 1_693_417_472,
            # The checkpoint was annealing powerslide alone; the other two are new.
            "config": {"reward_annealing": {"targets": {"powerslide_weight": 0.0}}},
        })
        self.assertEqual(stub._reward_anneal_start_steps, {"powerslide_weight": 1_693_417_472},
                         "only the target the checkpoint was annealing may inherit the legacy clock")

        stub._apply_reward_annealing()
        pushed = stub.pushed[-1]
        self.assertAlmostEqual(pushed["powerslide_weight"], 0.0, places=6)
        self.assertAlmostEqual(pushed["jump_bridge_weight"], 0.55, places=6,
                               msg="a target the legacy clock never covered must start at its base weight")
        self.assertAlmostEqual(pushed["player_to_ball_weight"], 0.6, places=6,
                               msg="a target the legacy clock never covered must start at its base weight")

    def test_legacy_clock_without_a_saved_config_starts_every_clock_fresh(self):
        """
        Guarantees the unprovable case fails toward a restarted ramp rather than a cliff.
        """
        base = {"powerslide_weight": 0.2}
        targets = {"powerslide_weight": 0.0}
        stub = self._anneal_stub(targets, base, decay_steps=400_000_000, global_step=2_000_000_000)

        stub._load_reward_anneal_clocks({"reward_anneal_start_step": 1_000_000_000})
        self.assertEqual(stub._reward_anneal_start_steps, {})

        stub._apply_reward_annealing()
        self.assertAlmostEqual(stub.pushed[-1]["powerslide_weight"], 0.2, places=6,
                               msg="with nothing to prove coverage, the ramp restarts instead of cliffing")

    def test_mini_ppo_training_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            import yaml
            with open("config/default_config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            cfg["logging"]["save_dir"] = tmpdir
            # Keep the test off the live logs/ directory: a real training run
            # writes metrics.json / history.jsonl there.
            cfg["logging"]["log_dir"] = tmpdir
            cfg["logging"]["tensorboard"] = False
            # Same reason: the league state file lives in logs/ by default.
            cfg.setdefault("league", {})["league_state_path"] = os.path.join(tmpdir, "league_state.json")
            cfg_path = os.path.join(tmpdir, "test_config.yaml")
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f)
            trainer = PPOTrainer(config_path=cfg_path)
            # Run 2 training iterations
            trainer.train(max_iterations=2)

            # Check that metrics were generated
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "metrics.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "history.jsonl")))

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
        # The term is authorship-gated: it prices what OUR touch did to the ball, so a
        # car driving the ball at its own net must have touched it to be charged.
        car0.ball_touches = 1
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

