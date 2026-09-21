"""
The League tab (ui/league_board.py), rendered from a scratch leaderboard, results log and state.

The board is server-side HTML and SVG, so what it claims is testable without a browser:
  - it renders empty, and populated, with every panel and without the retired ticker
  - the King banner's numbers come from the fit (vs the v3 king, win probabilities)
  - form, head to head and recent series are read from the results log
  - series recovered by the migration are not counted as played recently
  - a run's checkpoints are tagged with the run; the league's duplicate events collapse
"""
import datetime
import os
import shutil
import tempfile
import unittest

from ui import league_board as lb
from utils import rating_fit
from utils.trueskill_evaluator import TrueSkillEvaluator

V3 = "checkpoints/baselines/v3_iter198000.pt"


class TestLeagueBoard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ev = TrueSkillEvaluator(leaderboard_path=os.path.join(self.tmp, "lb.json"))
        self.ev.get_or_create_rating(V3)
        self.necto = self.ev.get_or_create_rating("checkpoints/necto-model.pt", is_anchor=True)
        self.a = self.ev.get_or_create_rating("checkpoints/archive/v5_run/checkpoint_iter_222200.pt")
        self.b = self.ev.get_or_create_rating("checkpoints/archive/v5_run/checkpoint_iter_222000.pt")
        now = datetime.datetime.now().isoformat()
        rows = [(self.a.path, V3, 5, 2, "calibration"), (self.a.path, self.b.path, 5, 3, "league"),
                (self.b.path, self.a.path, 5, 4, "league"), (self.a.path, self.b.path, 5, 1, "league"),
                (self.a.path, "checkpoints/necto-model.pt", 0, 5, "benchmark")] * 6
        for a, b, sa, sb, kind in rows:
            self.ev.results.append(rating_fit.append_result(self.ev.results_path, a, b, sa, sb, kind, now))
        old = "2026-01-01T00:00:00"
        for _ in range(3):
            self.ev.results.append(rating_fit.append_result(self.ev.results_path, self.a.path, self.b.path,
                                                            5, 0, "migrated", old))
        self.ev.refit()
        for r in (self.a, self.b):
            r.matches_played, r.wins, r.losses = 40, 25, 15
        self.state = {
            "king_of_the_hill": self.a.path,
            "elite_pool": [self.a.path, self.b.path],
            "contenders": [],
            "event_history": [
                {"timestamp": now, "type": "promotion", "model": "checkpoint_iter_222000",
                 "matches": 36, "points_rate": 61.0},
                {"timestamp": now, "type": "coronation", "model": "checkpoint_iter_222200",
                 "old_king": "checkpoint_iter_219000"},
                {"timestamp": now, "type": "coronation", "model": "checkpoint_iter_222200",
                 "old_king": "checkpoint_iter_219000"},
            ],
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def board(self):
        return lb.board_html(self.ev, self.state, "v6", "<div>bench</div>")

    def test_empty_board_renders_every_panel(self):
        ev = TrueSkillEvaluator(leaderboard_path=os.path.join(self.tmp, "empty.json"))
        html = lb.board_html(ev, {}, "v6", "")
        for panel in ("league-board", "Rating ladder", "Elite pool", "Pipeline", "Benchmarks"):
            self.assertIn(panel, html)
        self.assertIn("standby", lb.king_banner_html(ev, {}, "v6").lower())

    def test_the_ticker_is_gone(self):
        html = self.board()
        self.assertNotIn("lb-ticker", html)
        self.assertNotIn("Gauntlet Wire", html)

    def test_king_banner_reads_the_fit(self):
        html = lb.king_banner_html(self.ev, self.state, "v6")
        self.assertIn("King of the Hill", html)
        self.assertIn("222", html)
        self.assertIn("vs v3 king", html)
        self.assertIn("Beats v3 king", html)
        self.assertIn(lb._signed(self.a.mu - 25.0), html)

    def test_win_probability_is_the_fits_own_and_complementary(self):
        p = lb.p_beats(self.a, self.b)
        self.assertAlmostEqual(p + lb.p_beats(self.b, self.a), 1.0, places=9)
        self.assertGreater(p, 0.5)

    def test_form_head_to_head_and_series_come_from_the_log(self):
        html = self.board()
        self.assertIn("lg-form-W", html)
        self.assertIn("lg-form-L", html)
        score, games = lb.BoardData(self.ev, self.state, "v6").head_to_head(self.a, self.b)
        self.assertEqual(games, 21)          # 18 league + 3 migrated
        self.assertEqual(score, 15)          # a wins 2 of every 3, plus the 3 migrated
        self.assertIn("15&ndash;6", html)
        self.assertIn("Recent series", html)

    def test_migrated_series_are_not_counted_as_recent(self):
        d = lb.BoardData(self.ev, self.state, "v6")
        self.assertEqual(len(lb._timed(d.results)), 30)
        banner = lb.king_banner_html(self.ev, self.state, "v6")
        self.assertIn("<b>30</b>", banner)   # last 24 h: the 30 timed series only

    def test_a_run_is_tagged_and_duplicate_events_collapse(self):
        html = self.board()
        self.assertIn('class="lg-run"', html)
        self.assertIn(">v5<", html)
        self.assertEqual(html.count("took the crown"), 1)
        self.assertIn("graduated to the elite pool", html)

    def test_an_empty_gauntlet_says_why(self):
        self.assertIn("usually empty between saves", self.board())


if __name__ == "__main__":
    unittest.main()
