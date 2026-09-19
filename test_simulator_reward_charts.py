"""
Unit tests for visualizer match simulation charts and reward breakdowns.
Verifies that all relevant reward components are accurately rendered,
legacy categories are eliminated, and dual-panel breakdowns/timelines work cleanly.
"""

import os
import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.visualizer import render_reward_breakdown_plot, simulate_match, REWARD_METADATA, CORE_REWARD_KEYS


class TestSimulatorRewardCharts(unittest.TestCase):
    def test_metadata_contains_all_active_rewards(self):
        """Verify that all core rewards and cost terms have defined metadata labels and categories."""
        expected_keys = [
            "goal", "ball_to_goal", "touch", "player_to_ball", "boost",
            "powerslide", "air_roll_recovery", "jump_bridge",
            "time_cost", "spin_cost", "jump_cost", "own_goal_threat",
            "lateral_slip", "slide_waste"
        ]
        for k in expected_keys:
            self.assertIn(k, REWARD_METADATA)
            self.assertIn("label", REWARD_METADATA[k])
            self.assertIn("category", REWARD_METADATA[k])
            self.assertIn("priority", REWARD_METADATA[k])

    def test_render_reward_breakdown_single_panel(self):
        """Verify single-panel breakdown rendering when no timeline is passed."""
        blue_rewards = {
            "goal": 30.0,
            "ball_to_goal": 2.5,
            "player_to_ball": 4.0,
            "boost": -1.2,
            "touch": 1.5,
            "time_cost": -0.5,
        }
        fig = render_reward_breakdown_plot(blue_rewards=blue_rewards, orange_rewards=None)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)

        ax = fig.axes[0]
        yticklabels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertIn("Goals, Concedes & Saves", yticklabels)
        self.assertIn("Ball to Goal Progression", yticklabels)
        self.assertNotIn("Module 1: Goals, Saves & Power Multiplier", yticklabels)
        self.assertNotIn("Module 2: Ball Strikes, xG Shots & Dodge Bounties", yticklabels)
        plt.close(fig)

    def test_render_reward_breakdown_dual_panel_with_timeline(self):
        """Verify dual-panel (breakdown + timeline) rendering when timelines and goal events are supplied."""
        blue_rewards = {
            "goal": 0.0,
            "ball_to_goal": 1.8,
            "touch": 0.8,
            "player_to_ball": 5.0,
            "boost": -1.1,
            "spin_cost": -0.15,
        }
        orange_rewards = {
            "goal": 30.0,
            "ball_to_goal": -0.2,
            "touch": 1.2,
            "player_to_ball": 4.5,
            "boost": -0.9,
            "spin_cost": -0.08,
        }
        blue_tl = list(np.cumsum(np.full(50, 0.1)))
        orange_tl = list(np.cumsum(np.full(50, 0.15)))
        goals = [{"step": 25, "team": 1}]

        fig = render_reward_breakdown_plot(
            blue_rewards=blue_rewards,
            orange_rewards=orange_rewards,
            match_type="Self-Play",
            blue_timeline=blue_tl,
            orange_timeline=orange_tl,
            goal_events=goals,
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 2)
        plt.close(fig)

    def test_simulate_match_breakdown_accuracy(self):
        """Verify that simulate_match produces accurate breakdowns and dual-panel reward figures."""
        pitch_fig, reward_fig, stats = simulate_match(blue_model_path=None, orange_model_path="baseline", max_steps=40)
        self.assertIsInstance(pitch_fig, plt.Figure)
        self.assertIsInstance(reward_fig, plt.Figure)
        self.assertEqual(len(reward_fig.axes), 2)

        self.assertIn("blue_breakdown", stats)
        self.assertIn("orange_breakdown", stats)
        self.assertIn("goal_events", stats)

        # The breakdown carries the active reward version's terms
        from env.reward_registry import active_version
        from env.rewards_v4 import TERM_NAMES as V4_TERMS
        from env.rewards_v3 import TERM_NAMES
        if active_version() == "v4":
            TERM_NAMES = V4_TERMS
        b_keys = stats["blue_breakdown"].keys()
        for core_k in (CORE_REWARD_KEYS if active_version() == "v2" else TERM_NAMES):
            self.assertIn(core_k, b_keys)

        plt.close(pitch_fig)
        plt.close(reward_fig)


if __name__ == "__main__":
    unittest.main()
