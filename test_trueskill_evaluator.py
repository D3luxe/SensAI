"""
Unit Tests for TrueSkill Evaluator, Headless Match Simulation, and Rating Updates.
"""

from __future__ import annotations
import os
import shutil
import tempfile
import unittest
import numpy as np

from env.baseline_agent import BaselineChaser, create_opponent_bot
from utils.trueskill_evaluator import (
    TrueSkillEvaluator, simulate_headless_episode, simulate_headless_series,
    get_model_display_name, ModelRating
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

    def test_simulate_headless_episode_ends_at_first_goal(self):
        """An episode terminates on the first goal rather than running a fixed clock."""
        b1 = create_opponent_bot("heuristic")
        b2 = create_opponent_bot("heuristic")
        res = simulate_headless_episode(blue_bot=b1, orange_bot=b2, max_steps=400,
                                        no_touch_steps=500)
        self.assertIn(res["result"], (-1, 0, 1))
        self.assertLessEqual(res["steps"], 400)
        self.assertIn("no_touch_timeout", res)

    def test_no_touch_timeout_ends_a_dead_episode(self):
        """Two idle policies end the episode early instead of burning the step cap."""
        class Idle:
            def get_action(self, car, arena):
                import numpy as np
                return np.zeros(8, dtype=np.float32)

        res = simulate_headless_episode(blue_bot=Idle(), orange_bot=Idle(),
                                        max_steps=3000, no_touch_steps=60)
        self.assertEqual(res["result"], 0)
        self.assertLess(res["steps"], 3000, "no-touch timeout did not fire")

    def test_series_is_first_to_five_and_stops_early(self):
        """A series ends as soon as one side reaches wins_needed."""
        b1 = create_opponent_bot("heuristic")
        b2 = create_opponent_bot("heuristic")
        res = simulate_headless_series(b1, b2, series_length=9, wins_needed=5,
                                       max_steps=400, no_touch_steps=200)
        self.assertLessEqual(res["episodes_played"], 9)
        self.assertLessEqual(max(res["a_score"], res["b_score"]), 5)
        self.assertIn(res["winner"], ("a", "b", "draw"))
        self.assertEqual(res["score_diff"], res["a_score"] - res["b_score"])
        if max(res["a_score"], res["b_score"]) == 5:
            self.assertLess(res["episodes_played"], 10)

    def test_evaluate_pairing_rates_one_update_per_series(self):
        # Two series, each a best-of-9 -- and exactly two rating updates, not eighteen.
        results = self.evaluator.evaluate_pairing(
            model_a_path="heuristic",
            model_b_path="heuristic",
            series_per_pair=2,
            max_steps=50,
            no_touch_steps=25
        )
        self.assertEqual(len(results), 2)
        # Both sides of this pairing resolve to the same record, so it is credited once
        # per side per series. What matters is that a best-of-9 counts as one encounter
        # and not as nine: two series give four increments, never eighteen.
        rec = self.evaluator.get_or_create_rating("heuristic")
        self.assertEqual(rec.matches_played, 4)
        for r in results:
            self.assertLessEqual(r["episodes_played"], 9)
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


class TestSeriesSettings(unittest.TestCase):
    """
    A rated encounter is a best-of-9 series, first to 5, with episodes ending at the
    first goal. That makes a rated result decisive by construction, which is why the
    declared draw probability is low; fixed-length matches scored on goals drew 54% of
    the time between near-peers and no clock tuning got that under ~17%.
    """

    def test_defaults_describe_a_best_of_nine_series(self):
        from utils.trueskill_evaluator import (
            DEFAULT_SERIES_LENGTH, DEFAULT_SERIES_WINS_NEEDED,
            DEFAULT_NO_TOUCH_STEPS, DEFAULT_EVAL_MAX_STEPS, DEFAULT_DRAW_PROBABILITY
        )
        self.assertEqual(DEFAULT_SERIES_LENGTH, 9)
        self.assertEqual(DEFAULT_SERIES_WINS_NEEDED, 5)
        self.assertEqual(DEFAULT_NO_TOUCH_STEPS, 500)
        self.assertEqual(DEFAULT_EVAL_MAX_STEPS, 3000)
        self.assertLess(DEFAULT_DRAW_PROBABILITY, 0.1)

    def test_retired_overtime_kwargs_are_dropped_not_misapplied(self):
        """Old callers must not silently get 9x the games, and typos must still raise."""
        import tempfile, os
        from unittest import mock
        ev = TrueSkillEvaluator(leaderboard_path=os.path.join(tempfile.mkdtemp(), "lb.json"))
        with mock.patch("utils.trueskill_evaluator.simulate_headless_series",
                        return_value={"a_score": 5, "b_score": 1, "score_diff": 4,
                                      "winner": "a", "episodes_played": 6,
                                      "total_steps": 100, "episodes": []}) as sim, \
             mock.patch("utils.trueskill_evaluator.create_opponent_bot", return_value=object()):
            ev.evaluate_pairing("heuristic", "checkpoints/necto-model.pt",
                                matches_per_pair=4, enable_overtime=True,
                                max_ot_steps=600, ot_random_kickoff=True)
            # 4 legacy "matches" is read as 2 series, not 4 series of 9 episodes.
            self.assertEqual(sim.call_count, 2)

            with self.assertRaises(TypeError):
                ev.evaluate_pairing("heuristic", "checkpoints/necto-model.pt",
                                    definitely_not_a_real_kwarg=1)

    def test_league_manager_forwards_its_series_configuration(self):
        import tempfile, os
        from unittest import mock
        from utils.league_manager import LeagueManager

        tmp = tempfile.mkdtemp(prefix="sensai_test_series_")
        league = LeagueManager(
            config={"anchors": ["heuristic"], "eval_max_steps": 321,
                    "eval_no_touch_steps": 77, "series_length": 5, "series_wins_needed": 3},
            leaderboard_path=os.path.join(tmp, "lb.json"),
        )
        self.assertEqual(league.series_length, 5)
        self.assertEqual(league.series_wins_needed, 3)

        with mock.patch.object(league.evaluator, "evaluate_pairing", return_value=[]) as ep:
            league.evaluator.get_or_create_rating("heuristic", is_anchor=True)
            league.king_of_the_hill = "heuristic"
            league.step_king_title_bout()

        for call in ep.call_args_list:
            self.assertEqual(call.kwargs.get("max_steps"), 321)
            self.assertEqual(call.kwargs.get("no_touch_steps"), 77)
            self.assertEqual(call.kwargs.get("series_length"), 5)
            self.assertEqual(call.kwargs.get("wins_needed"), 3)


class TestAnchorCalibration(unittest.TestCase):
    """
    One rating is pinned, and only one. The old ladder (heuristic 15, BC 14.5, Necto 30, Nexto
    34.7) pinned four values from a round robin whose field no longer exists; two pins the
    data disagree with bend every rating between them. See ANCHOR_CALIBRATION.
    """

    def test_the_v3_king_is_the_only_pin(self):
        from utils.trueskill_evaluator import ANCHOR_CALIBRATION
        self.assertEqual(list(ANCHOR_CALIBRATION), ["v3_iter198000"])

    def test_calibration_is_applied_on_load(self):
        """A leaderboard carrying a stale value for the pinned rating is corrected on read."""
        import json, os, tempfile
        from utils.trueskill_evaluator import TrueSkillEvaluator, ANCHOR_CALIBRATION
        path = os.path.join(tempfile.mkdtemp(), "lb.json")
        key = "checkpoints/baselines/v3_iter198000.pt"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ratings": {key: {"name": "v3_iter198000", "path": key,
                                         "mu": 31.0, "sigma": 8.333}}}, f)
        ev = TrueSkillEvaluator(leaderboard_path=path)
        rec = ev.ratings[key]
        self.assertAlmostEqual(rec.mu, ANCHOR_CALIBRATION["v3_iter198000"], places=3)
        self.assertTrue(rec.is_anchor)

    def test_a_former_anchor_loads_as_fitted(self):
        """Necto keeps whatever the fit last gave it; nothing re-pins it at 30."""
        import json, os, tempfile
        from utils.trueskill_evaluator import TrueSkillEvaluator
        path = os.path.join(tempfile.mkdtemp(), "lb.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ratings": {"checkpoints/necto-model.pt": {
                "name": "Necto", "path": "checkpoints/necto-model.pt",
                "mu": 44.0, "sigma": 2.0, "is_anchor": True}}}, f)
        ev = TrueSkillEvaluator(leaderboard_path=path)
        self.assertEqual(ev.ratings["checkpoints/necto-model.pt"].mu, 44.0)
        self.assertFalse(ev.is_rating_frozen(ev.ratings["checkpoints/necto-model.pt"]))

if __name__ == "__main__":
    unittest.main()
