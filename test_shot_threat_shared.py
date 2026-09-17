"""
Guarantees for the single shot-threat computation (env.physics_engine.compute_shot_threat) that
training (RocketSimArena) and live play (bot.py's MockArena) both call.

  1. A predicted trajectory entering the mouth decides the threat; intensity decays with time ahead.
  2. A grazing entry (inside the posts, not a ball radius clear) is worth 0.45 of a clean one.
  3. Without a usable trajectory the ballistic raycast still reports the threat.
  4. A ball moving away from the net is never a threat, whatever the trajectory says.
  5. RocketSimArena and bot.MockArena both delegate to it, so they cannot diverge again.
  6. Relative scenario and replay-pool paths resolve against the repo root, not the working directory.
"""

import inspect
import os
import unittest

import numpy as np

from env.physics_engine import (
    ARENA_EXTENT_Y, BALL_RADIUS, EFFECTIVE_GOAL_HALF_WIDTH, GOAL_HALF_WIDTH, SHOT_THREAT_HORIZON_S,
    compute_shot_threat,
)

BALL_POS = np.array([0.0, -3000.0, 93.0], dtype=np.float32)
BALL_TOWARD_BLUE_NET = np.array([0.0, -2000.0, 0.0], dtype=np.float32)


def straight_line(x, z, t_entry, steps=30):
    """(t, x, y, z) samples reaching the blue goal line at t_entry and continuing past it."""
    for i in range(steps + 1):
        t = t_entry * 1.5 * i / steps
        y = -3000.0 + (-ARENA_EXTENT_Y + 3000.0) * (t / t_entry)
        yield t, x, y, z


class TestComputeShotThreat(unittest.TestCase):
    def test_clean_entry_intensity_decays_with_time(self):
        early = compute_shot_threat(0, BALL_POS, BALL_TOWARD_BLUE_NET, straight_line(0.0, 300.0, 0.6))
        late = compute_shot_threat(0, BALL_POS, BALL_TOWARD_BLUE_NET, straight_line(0.0, 300.0, 2.4))
        self.assertTrue(early[0] and late[0])
        self.assertGreater(early[1], late[1])
        self.assertAlmostEqual(early[2], 300.0 / 642.775, places=3)

    def test_grazing_entry_is_discounted(self):
        clean = compute_shot_threat(0, BALL_POS, BALL_TOWARD_BLUE_NET, straight_line(0.0, 300.0, 1.0))
        graze_x = (EFFECTIVE_GOAL_HALF_WIDTH + GOAL_HALF_WIDTH) / 2.0
        graze = compute_shot_threat(0, BALL_POS, BALL_TOWARD_BLUE_NET, straight_line(graze_x, 300.0, 1.0))
        self.assertTrue(graze[0])
        self.assertAlmostEqual(graze[1], clean[1] * 0.45, delta=0.02)

    def test_wide_trajectory_falls_back_to_ballistic(self):
        # The prediction says wide, but the raycast from this state is on target, as before the merge
        # A slight lift keeps the parabola above the floor at the goal line (~1.06 s out)
        vel = np.array([0.0, -2000.0, 350.0], dtype=np.float32)
        wide = compute_shot_threat(0, BALL_POS, vel, straight_line(3000.0, 300.0, 1.0))
        none = compute_shot_threat(0, BALL_POS, vel, None)
        self.assertEqual(wide, none)
        self.assertTrue(none[0])

    def test_broken_trajectory_falls_back_to_ballistic(self):
        def broken():
            yield 0.0, 0.0, -3000.0, 93.0
            raise RuntimeError("prediction went away")
        self.assertEqual(compute_shot_threat(0, BALL_POS, BALL_TOWARD_BLUE_NET, broken()),
                         compute_shot_threat(0, BALL_POS, BALL_TOWARD_BLUE_NET, None))

    def test_ball_moving_away_is_never_a_threat(self):
        away = -BALL_TOWARD_BLUE_NET
        self.assertEqual(compute_shot_threat(0, BALL_POS, away, straight_line(0.0, 300.0, 1.0)), (False, 0.0, 0.0))

    def test_out_of_horizon_ballistic_is_not_a_threat(self):
        slow = np.array([0.0, -(ARENA_EXTENT_Y - 3000.0) / (SHOT_THREAT_HORIZON_S + 1.0), 0.0], dtype=np.float32)
        self.assertFalse(compute_shot_threat(0, np.array([0.0, -3000.0, BALL_RADIUS]), slow, None)[0])

    def test_training_and_live_both_delegate(self):
        import bot
        from env.physics_engine import RocketSimArena
        self.assertIn("compute_shot_threat(", inspect.getsource(RocketSimArena.get_shot_threat))
        self.assertIn("compute_shot_threat(", inspect.getsource(bot.MockArena.get_shot_threat))


class TestRootResolvedPaths(unittest.TestCase):
    def test_scenario_and_pool_paths_ignore_the_working_directory(self):
        from utils.replay_parser import DEFAULT_POOL_PATH, REPO_ROOT, ReplayParser
        from utils.scenario_manager import SCENARIOS_CONFIG_PATH
        self.assertTrue(os.path.isabs(DEFAULT_POOL_PATH) and DEFAULT_POOL_PATH.startswith(REPO_ROOT))
        self.assertTrue(os.path.isabs(SCENARIOS_CONFIG_PATH) and SCENARIOS_CONFIG_PATH.startswith(REPO_ROOT))
        cwd = os.getcwd()
        try:
            os.chdir(os.path.join(REPO_ROOT, "scripts"))
            self.assertEqual(ReplayParser(pool_path="data/replays/x.npz").pool_path,
                             os.path.join(REPO_ROOT, "data/replays/x.npz"))
            self.assertFalse(os.path.exists(os.path.join(REPO_ROOT, "scripts", "data")))
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
