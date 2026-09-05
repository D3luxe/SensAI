"""
Unit and Integration Tests for Phase 1: Environment & Episode Reset Integrity.
Verifies Fixes #5, #6, #7 and external bot mask control pass-through.
"""

import unittest
import numpy as np
import RocketSim as rsim

from env.rocket_env import RocketLeagueEnv
from env.physics_engine import RocketSimArena
from env.state_setters import KickoffSetter, AerialScenarioSetter, WallPlaySetter


class TestResetAndBoundaryIntegrity(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2)

    def test_auto_reset_observation_boundary(self):
        """Fix #5: Verify that when done=True, out_obs contains the post-reset state and terminal_observation is stored in info."""
        env = RocketLeagueEnv(max_episode_steps=5, tick_skip=8)
        initial_obs = env.reset()

        done = False
        step_count = 0
        last_out_obs = None
        last_info = None

        while not done and step_count < 10:
            actions = np.zeros((2, 8), dtype=np.float32)
            obs, rews, dones, info = env.step(actions)
            done = dones[0]
            step_count += 1
            last_out_obs = obs
            last_info = info

        self.assertTrue(done, "Episode should have completed within max_episode_steps")
        self.assertIn("terminal_observation", last_info, "info dictionary must contain 'terminal_observation' upon episode completion")
        
        # Terminal observation should reflect the state at step 5
        terminal_obs = last_info["terminal_observation"]
        self.assertEqual(terminal_obs.shape, (2, 74))

        # Returned observation should match the newly reset arena (step 0 of new episode)
        self.assertEqual(env.current_step, 0, "Environment current_step must be reset to 0")
        current_arena_obs = np.empty((2, 74), dtype=np.float32)
        for i, car in enumerate(env.arena.cars):
            env.obs_builder.build_obs(car, env.arena, out=current_arena_obs[i])

        np.testing.assert_allclose(last_out_obs, current_arena_obs, atol=1e-5,
                                   err_msg="Returned out_obs on done must match the post-reset arena state")

    def test_native_reset_state_cleanliness(self):
        """Fix #6: Verify that ball_touches, touch dictionary keys, and prediction caches are properly reset."""
        # Simulate state contamination
        for car in self.arena.cars:
            car.ball_touches = 7
        if hasattr(self.arena, "_touch_active_this_step"):
            for k in self.arena._touch_active_this_step:
                self.arena._touch_active_this_step[k] = True
        if hasattr(self.arena, "_car_was_touching"):
            for k in self.arena._car_was_touching:
                self.arena._car_was_touching[k] = True

        self.arena._cached_pred_step = 42
        self.arena._cached_rsim_preds = ["dummy_pred"]
        if hasattr(self.arena, "_cached_threat"):
            self.arena._cached_threat[0] = (True, 0.9, 500.0)

        # Execute reset
        self.arena.reset(random_kickoff=False)

        # Assert ball_touches is cleared
        for i, car in enumerate(self.arena.cars):
            self.assertEqual(car.ball_touches, 0, f"Car {i} ball_touches must be 0 after reset")

        # Assert touch dict keys are preserved and values are False
        if hasattr(self.arena, "_touch_active_this_step"):
            self.assertEqual(len(self.arena._touch_active_this_step), self.arena.num_players)
            for k in range(self.arena.num_players):
                self.assertIn(k, self.arena._touch_active_this_step)
                self.assertFalse(self.arena._touch_active_this_step[k])

        if hasattr(self.arena, "_car_was_touching"):
            self.assertEqual(len(self.arena._car_was_touching), self.arena.num_players)
            for k in range(self.arena.num_players):
                self.assertIn(k, self.arena._car_was_touching)
                self.assertFalse(self.arena._car_was_touching[k])

        # Assert caches are cleared
        self.assertEqual(self.arena._cached_pred_step, -1)
        self.assertIsNone(self.arena._cached_rsim_preds)
        if hasattr(self.arena, "_cached_threat"):
            self.assertEqual(len(self.arena._cached_threat), 0)

        # Assert boost pads are all active after reset
        for pad in self.arena.boost_pads:
            self.assertTrue(pad.is_active, "All boost pads must be active after reset")
            self.assertEqual(pad.cooldown_timer, 0.0)

    def test_scenario_setters_fresh_car_state(self):
        """Fix #7: Verify scenario setters use fresh CarState and do not retain demolition or double-jump flags."""
        rsim_arena = rsim.Arena(rsim.GameMode.SOCCAR)
        car = rsim_arena.add_car(rsim.Team.BLUE)

        # Contaminate car state
        dirty_cs = car.get_state()
        dirty_cs.is_demoed = True
        dirty_cs.demo_respawn_timer = 3.0
        dirty_cs.has_jumped = True
        dirty_cs.has_double_jumped = True
        dirty_cs.has_flipped = True
        dirty_cs.is_flipping = True
        dirty_cs.vel = rsim.Vec(1500.0, 1200.0, 800.0)
        car.set_state(dirty_cs)

        # Run KickoffSetter
        setter = KickoffSetter()
        setter.reset(rsim_arena, num_players=1)

        clean_cs = car.get_state()
        self.assertFalse(clean_cs.is_demoed, "Car must not be demoed on kickoff reset")
        self.assertEqual(clean_cs.demo_respawn_timer, 0.0)
        self.assertFalse(clean_cs.has_jumped, "Car must not have jumped on kickoff reset")
        self.assertFalse(clean_cs.has_double_jumped, "Car must not have double jumped on kickoff reset")
        self.assertFalse(clean_cs.has_flipped, "Car must not have flipped on kickoff reset")
        self.assertFalse(clean_cs.is_flipping, "Car must not be flipping on kickoff reset")
        self.assertEqual(clean_cs.vel.x, 0.0)
        self.assertEqual(clean_cs.vel.y, 0.0)
        self.assertEqual(clean_cs.vel.z, 0.0)

    def test_bot_mask_control_passthrough(self):
        """Fix #4 pass-through: Verify bot_mask allows external bots to bypass the jump sequencer."""
        arena = RocketSimArena(num_players=2)

        # Test action with immediate jump (tick 0)
        jump_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # jump=1.0
        actions = [np.zeros(8, dtype=np.float32), jump_action]

        # Step with bot_mask=[False, True]
        arena.step(actions, dt=1.0 / 120.0, bot_mask=[False, True])

        # Orange car (index 1) controls should have jump=True directly applied
        r_car_orange = arena._rsim_cars[1]
        c_state = r_car_orange.get_state()
        self.assertTrue(c_state.last_controls.jump, "External bot jump should be applied on tick 0 without gate")


if __name__ == "__main__":
    unittest.main()
