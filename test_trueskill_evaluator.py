"""
Unit Tests for TrueSkill Evaluator, Headless Match Simulation, and Rating Updates.
"""

from __future__ import annotations
import os
import shutil
import tempfile
import unittest
import numpy as np

from env.baseline_agent import BaselineChaser
from utils.trueskill_evaluator import (
    TrueSkillEvaluator, simulate_headless_match, get_model_display_name, ModelRating
)


class TestTrueSkillEvaluator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.leaderboard_path = os.path.join(self.temp_dir, "test_leaderboard.json")
        self.evaluator = TrueSkillEvaluator(leaderboard_path=self.leaderboard_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_display_names(self):
        self.assertEqual(get_model_display_name("heuristic"), "Baseline Chaser (Heuristic)")
        self.assertEqual(get_model_display_name("checkpoints/latest_model.pt"), "Latest Policy (latest_model.pt)")
        self.assertEqual(get_model_display_name("checkpoints/pretrained_baseline.pt"), "Pretrained Baseline (BC)")

    def test_simulate_headless_match(self):
        b1 = BaselineChaser()
        b2 = BaselineChaser()

        res = simulate_headless_match(blue_bot=b1, orange_bot=b2, max_steps=50, enable_overtime=False)
        self.assertIn("winner", res)
        self.assertIn("blue_goals", res)
        self.assertIn("orange_goals", res)
        self.assertEqual(res["total_steps"], 50)

    def test_evaluate_pairing_symmetry(self):
        # 2 matches between heuristic chasers
        results = self.evaluator.evaluate_pairing(
            model_a_path="heuristic",
            model_b_path="heuristic",
            matches_per_pair=2,
            max_steps=50
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(os.path.exists(self.leaderboard_path))

        # Check leaderboard dataframe
        df = self.evaluator.get_leaderboard_dataframe()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)  # Both resolve to "Baseline Chaser (Heuristic)"

    def test_atomic_persistence(self):
        rec = self.evaluator.get_or_create_rating("checkpoints/dummy_bot.pt")
        rec.mu = 32.5
        rec.sigma = 5.1
        rec.wins = 10
        rec.losses = 2
        rec.update_conservative()
        self.evaluator.save_leaderboard()

        # Reload into a fresh evaluator
        new_eval = TrueSkillEvaluator(leaderboard_path=self.leaderboard_path)
        self.assertIn("checkpoints/dummy_bot.pt", new_eval.ratings)
        loaded = new_eval.ratings["checkpoints/dummy_bot.pt"]
        self.assertEqual(loaded.mu, 32.5)
        self.assertEqual(loaded.wins, 10)
        self.assertEqual(loaded.losses, 2)

    def test_render_plot(self):
        self.evaluator.get_or_create_rating("checkpoints/model_a.pt")
        self.evaluator.get_or_create_rating("checkpoints/model_b.pt")
        fig = self.evaluator.render_leaderboard_plot()
        self.assertIsNotNone(fig)
        out_path = os.path.join(self.temp_dir, "test_plot.png")
        fig.savefig(out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 1000)


class TestMatchLengthSettings(unittest.TestCase):
    """
    Overtime settings are plumbed through and configurable.

    These were hardcoded defaults inside simulate_headless_match that evaluate_pairing
    never passed, so the league could not change them at all. The values now in place
    were chosen from measurement (n=102 per arm, near-peer pairings): reg 600 / OT 600
    with a random kickoff gave 16.7% draws against the old reg 400 / OT 200 fixed at
    53.9%, at equal decisive-results-per-second.
    """

    def test_defaults_match_the_measured_configuration(self):
        from utils.trueskill_evaluator import (
            DEFAULT_EVAL_MAX_STEPS, DEFAULT_OT_STEPS, DEFAULT_OT_RANDOM_KICKOFF
        )
        self.assertEqual(DEFAULT_EVAL_MAX_STEPS, 600)
        self.assertEqual(DEFAULT_OT_STEPS, 600)
        self.assertTrue(DEFAULT_OT_RANDOM_KICKOFF)

    def test_overtime_settings_are_plumbed_through(self):
        """evaluate_pairing and run_tournament must forward both overtime arguments."""
        import inspect
        from utils.trueskill_evaluator import (
            TrueSkillEvaluator, simulate_headless_match
        )
        for fn in (simulate_headless_match,
                   TrueSkillEvaluator.evaluate_pairing,
                   TrueSkillEvaluator.run_tournament):
            params = inspect.signature(fn).parameters
            self.assertIn("max_ot_steps", params, f"{fn.__name__} drops max_ot_steps")
            self.assertIn("ot_random_kickoff", params, f"{fn.__name__} drops ot_random_kickoff")

    def test_league_manager_forwards_its_configured_overtime(self):
        """A configured overtime reaches evaluate_pairing rather than being ignored."""
        import tempfile, os
        from unittest import mock
        from utils.league_manager import LeagueManager

        tmp = tempfile.mkdtemp(prefix="sensai_test_ot_")
        league = LeagueManager(
            config={
                "anchors": ["heuristic"],
                "eval_max_steps": 321,
                "eval_ot_steps": 654,
                "eval_ot_random_kickoff": False,
            },
            leaderboard_path=os.path.join(tmp, "lb.json"),
        )
        self.assertEqual(league.eval_ot_steps, 654)
        self.assertFalse(league.eval_ot_random_kickoff)

        with mock.patch.object(league.evaluator, "evaluate_pairing", return_value=[]) as ep:
            league.evaluator.get_or_create_rating("heuristic", is_anchor=True)
            league.king_of_the_hill = "heuristic"
            league.step_king_title_bout()

        for call in ep.call_args_list:
            self.assertEqual(call.kwargs.get("max_steps"), 321)
            self.assertEqual(call.kwargs.get("max_ot_steps"), 654)
            self.assertIs(call.kwargs.get("ot_random_kickoff"), False)


if __name__ == "__main__":
    unittest.main()
