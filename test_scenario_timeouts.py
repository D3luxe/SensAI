"""
Unit Tests for Scenario Timeouts and Condition-Based Early Terminations.
"""

import unittest
import numpy as np
import RocketSim as rsim

from env.physics_engine import RocketSimArena
from env.rocket_env import RocketLeagueEnv, SCENARIO_TIMEOUTS


class TestScenarioTimeouts(unittest.TestCase):
    def setUp(self):
        self.env = RocketLeagueEnv(game_mode="1v1", tick_skip=8, max_episode_steps=1500)

    def test_scenario_names_and_timeout_mapping(self):
        """Test that SCENARIO_TIMEOUTS contains all expected scenarios and has valid timeout values."""
        expected_scenarios = [
            "kickoff", "aerial", "goalie_save", "wall_play",
            "wall_rebound", "turnaround", "dribble_flick", "replay", "custom"
        ]
        for name in expected_scenarios:
            self.assertIn(name, SCENARIO_TIMEOUTS)
            self.assertGreater(SCENARIO_TIMEOUTS[name], 0)
            self.assertLess(SCENARIO_TIMEOUTS[name], 1000)

        # Standard kickoff reset
        self.env.reset(random_kickoff=False)
        self.assertEqual(self.env.current_scenario, "kickoff")
        self.assertEqual(self.env.scenario_timeout, SCENARIO_TIMEOUTS["kickoff"])

    def test_hard_scenario_timeout(self):
        """Verify hard timeout triggers precisely at scenario_timeout steps."""
        self.env.reset(random_kickoff=False)
        self.env.current_scenario = "aerial"
        self.env.scenario_timeout = 120
        # Place ball high in air to avoid aerial early termination
        bs = self.env.arena._rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(0, 0, 1500)
        bs.vel = rsim.Vec(0, 0, 0)
        self.env.arena._rsim_arena.ball.set_state(bs)

        dummy_actions = np.zeros((self.env.num_players, self.env.act_dim), dtype=np.float32)

        for step in range(1, 120):
            # Keep ball high to prevent early floor landing
            bs = self.env.arena._rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(0, 0, 1500)
            bs.vel = rsim.Vec(0, 0, 0)
            self.env.arena._rsim_arena.ball.set_state(bs)

            _, _, dones, info = self.env.step(dummy_actions)
            if step < 120:
                self.assertFalse(dones[0], f"Step {step} should not be done yet")

        # Step 120: Should trigger timeout done
        bs = self.env.arena._rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(0, 0, 1500)
        bs.vel = rsim.Vec(0, 0, 0)
        self.env.arena._rsim_arena.ball.set_state(bs)
        _, _, dones, info = self.env.step(dummy_actions)
        self.assertTrue(info["done"])

    def test_aerial_whiff_early_termination(self):
        """Aerial scenario should terminate early if step > 20, touches == 0, and ball_z < 250."""
        self.env.reset(random_kickoff=False)
        self.env.current_scenario = "aerial"
        self.env.scenario_timeout = 120
        dummy_actions = np.zeros((self.env.num_players, self.env.act_dim), dtype=np.float32)

        # Step up to 20 with ball airborne
        for _ in range(20):
            bs = self.env.arena._rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(0, 0, 800)
            bs.vel = rsim.Vec(0, 0, 0)
            self.env.arena._rsim_arena.ball.set_state(bs)
            _, _, dones, _ = self.env.step(dummy_actions)
            self.assertFalse(dones[0])

        # Step 21: Force ball below 250 with zero touches
        bs = self.env.arena._rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(0, 0, 150)
        bs.vel = rsim.Vec(0, 0, 0)
        self.env.arena._rsim_arena.ball.set_state(bs)
        _, _, dones, info = self.env.step(dummy_actions)

        self.assertTrue(dones[0], "Whiffed aerial should terminate when ball drops below 250 Z after grace period")
        self.assertTrue(info["done"])

    def test_aerial_with_touches_does_not_terminate_early(self):
        """If player touched the ball during aerial, dropping below 250 Z should NOT terminate early."""
        self.env.reset(random_kickoff=False)
        self.env.current_scenario = "aerial"
        self.env.scenario_timeout = 120
        dummy_actions = np.zeros((self.env.num_players, self.env.act_dim), dtype=np.float32)

        # Step up to 20
        for _ in range(20):
            bs = self.env.arena._rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(0, 0, 800)
            bs.vel = rsim.Vec(0, 0, 0)
            self.env.arena._rsim_arena.ball.set_state(bs)
            self.env.step(dummy_actions)

        # Simulate that car 0 touched the ball
        self.env.arena.cars[0].ball_touches = 1
        self.env.episode_touches[0] = 1

        # Step 21: Ball is below 250, but touch occurred
        bs = self.env.arena._rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(0, 0, 150)
        bs.vel = rsim.Vec(0, 0, 0)
        self.env.arena._rsim_arena.ball.set_state(bs)
        _, _, dones, info = self.env.step(dummy_actions)

        self.assertFalse(dones[0], "Aerial with touches must not terminate early, allowing shot completion")

    def test_goalie_save_defend_sign_and_clear_resolution(self):
        """Verify goalie save defend sign correctly identifies goal and clears."""
        # Case A: Shot incoming toward Blue net (Y = -5120), bvy < 0
        self.env.reset(random_kickoff=False)
        self.env.current_scenario = "goalie_save"
        self.env.scenario_timeout = 90

        # Simulate initial ball velocity toward Blue net
        bs = self.env.arena._rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(0, -2000, 300)
        bs.vel = rsim.Vec(0, -1200, 100)
        self.env.arena._rsim_arena.ball.set_state(bs)

        # Re-derive defend sign as done in reset
        self.env._save_defend_sign = -1.0 if bs.vel.y < 0 else 1.0
        self.assertEqual(self.env._save_defend_sign, -1.0)

        dummy_actions = np.zeros((self.env.num_players, self.env.act_dim), dtype=np.float32)

        # Fast forward past grace period (> 30 steps) while shot is incoming
        for _ in range(31):
            bs = self.env.arena._rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(0, -3500, 300)
            bs.vel = rsim.Vec(0, -800, 0)  # Still incoming toward -5120
            self.env.arena._rsim_arena.ball.set_state(bs)
            _, _, dones, _ = self.env.step(dummy_actions)
            self.assertFalse(dones[0], "Incoming shot must NOT terminate early as saved")

        # Now simulate a save: ball cleared toward +Y (> 300 uu/s)
        bs = self.env.arena._rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(0, -3000, 100)
        bs.vel = rsim.Vec(0, 600, 200)  # Moving away from -5120
        self.env.arena._rsim_arena.ball.set_state(bs)
        self.env.episode_touches[0] = 1

        _, _, dones, info = self.env.step(dummy_actions)
        self.assertTrue(dones[0], "Cleared ball moving away from net should resolve save")
        self.assertTrue(info["done"])

        # Case B: Shot incoming toward Orange net (Y = +5120), bvy > 0
        self.env.reset(random_kickoff=False)
        self.env.current_scenario = "goalie_save"
        self.env.scenario_timeout = 90

        bs = self.env.arena._rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(0, 2000, 300)
        bs.vel = rsim.Vec(0, 1200, 100)
        self.env.arena._rsim_arena.ball.set_state(bs)
        self.env._save_defend_sign = -1.0 if bs.vel.y < 0 else 1.0
        self.assertEqual(self.env._save_defend_sign, 1.0)

        # Fast forward past 30 steps while shot is incoming toward Orange
        for _ in range(31):
            bs = self.env.arena._rsim_arena.ball.get_state()
            bs.pos = rsim.Vec(0, 3500, 300)
            bs.vel = rsim.Vec(0, 800, 0)
            self.env.arena._rsim_arena.ball.set_state(bs)
            _, _, dones, _ = self.env.step(dummy_actions)
            self.assertFalse(dones[0], "Incoming shot toward Orange net must NOT terminate early")

        # Now simulate save cleared toward -Y
        bs = self.env.arena._rsim_arena.ball.get_state()
        bs.pos = rsim.Vec(0, 3000, 100)
        bs.vel = rsim.Vec(0, -600, 200)
        self.env.arena._rsim_arena.ball.set_state(bs)
        self.env.episode_touches[1] = 1

        _, _, dones, info = self.env.step(dummy_actions)
        self.assertTrue(dones[0], "Cleared ball moving away from Orange net should resolve save")

    def test_kickoff_stagnation_terminates_and_sets_done(self):
        """Untouched kickoff ball after 75 steps should trigger stagnation and set info['done']."""
        self.env.reset(random_kickoff=False)
        self.assertEqual(self.env.current_scenario, "kickoff")
        dummy_actions = np.zeros((self.env.num_players, self.env.act_dim), dtype=np.float32)

        for step in range(1, 76):
            _, _, dones, info = self.env.step(dummy_actions)
            if step < 76:
                self.assertFalse(dones[0], f"Kickoff step {step} should not be done yet")

        # Step 76: Ball is untouched at center (0, 0)
        _, _, dones, info = self.env.step(dummy_actions)
        self.assertTrue(dones[0], "Untouched kickoff must terminate at step 76")
        self.assertTrue(info["done"])
        self.assertEqual(info["scenario"], "kickoff")


if __name__ == "__main__":
    unittest.main()
