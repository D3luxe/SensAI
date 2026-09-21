"""
The league's rating scale, rebuilt 2026-09-21 as a batch fit (utils/rating_fit.py).

History, because it explains every test here. Online TrueSkill against a pinned Necto drifted
twice: to mu 518-521 (2026-09-20: the solver threw past a ~229 mu gap and the anchor went
silent), and, after that was fixed, from 19 to 41 across the v5 run while every checkpoint
still lost to Necto 35-1. The second drift had nothing to do with the solver. Checkpoints never
change, but an online filter treats each result as news about a moving target, locks ratings
and inherits them, and against an anchor that wins every series nothing pulls back.

The replacement fits every rating at once from every series played, with one pinned
checkpoint (the v3 king, mu 25) and everything else -- Necto included -- fitted.

  1. the fit recovers known skills, and is order-independent
  2. an improving population does not drift away from the pinned rating
  3. an undefeated reference stays finite and cannot move the scale
  4. the evaluator logs every series, refits, and keeps the pinned rating fixed
  5. calibration series move ratings but are excluded from promotion and the loss streak
  6. the leaderboard reports against Necto's fitted rating
"""
import math
import os
import random
import shutil
import tempfile
import unittest

import numpy as np
from scipy.special import ndtr

from utils import rating_fit
from utils.rating_fit import SCALE, fit_ratings
from utils.trueskill_evaluator import (
    ANCHOR_CALIBRATION, ModelRating, TrueSkillEvaluator, get_anchor_calibration,
)

V3 = "checkpoints/baselines/v3_iter198000.pt"


def series(a, b, ta, tb, rng, kind="league", draw=0.07):
    """One simulated best-of-9 between true skills ta and tb, as a results-log row."""
    if rng.random() < draw:
        sa, sb = 4, 4
    else:
        sa, sb = (5, 3) if rng.random() < ndtr((ta - tb) / SCALE) else (3, 5)
    return {"a": a, "b": b, "a_score": sa, "b_score": sb, "kind": kind}


def necto_sweep(players, n):
    return [{"a": "necto", "b": p, "a_score": 5, "b_score": 0, "kind": "benchmark"}
            for p in players for _ in range(n)]


class TestTheFit(unittest.TestCase):
    def test_it_recovers_known_skills_within_its_own_error_bars(self):
        rng = random.Random(1)
        true = {"p0": 25.0, "p1": 29.0, "p2": 21.0, "p3": 33.0, "p4": 26.5}
        names = list(true)
        rows = []
        for _ in range(1200):
            a, b = rng.sample(names, 2)
            rows.append(series(a, b, true[a], true[b], rng))
        fit = fit_ratings(rows, {"p0": 25.0})
        z = [(fit[k][0] - true[k]) / fit[k][1] for k in names[1:]]
        self.assertLess(max(abs(x) for x in z), 3.0, f"z-scores {z}")
        self.assertEqual(fit["p0"], (25.0, 0.0), "the pinned rating must not move")

    def test_the_order_results_arrive_in_does_not_matter(self):
        """The property the online filter lacked: a checkpoint's rating is a fact about it."""
        rng = random.Random(2)
        true = {"a": 25.0, "b": 28.0, "c": 23.0}
        rows = []
        for _ in range(300):
            x, y = rng.sample(list(true), 2)
            rows.append(series(x, y, true[x], true[y], rng))
        shuffled = rows[:]
        random.Random(3).shuffle(shuffled)
        f1, f2 = fit_ratings(rows, {"a": 25.0}), fit_ratings(shuffled, {"a": 25.0})
        for k in true:
            self.assertAlmostEqual(f1[k][0], f2[k][0], places=6)

    def test_an_improving_population_does_not_drift(self):
        """
        The live failure, reduced: each checkpoint a little better than the last and noisy
        around the trend, graded against its recent predecessors with 10% of series against
        the pinned rating. Estimate minus truth must not trend. Online TrueSkill in the same
        setting drifted about -7 mu per 100 checkpoints (scratch gradesim.py).
        """
        rng = random.Random(4)
        true = {"v3": 25.0}
        rows, ck = [], []
        for g in range(80):
            name = f"c{g}"
            true[name] = 25.0 + 0.1 * g + rng.gauss(0, 1.5)
            opps = ck[-10:]
            for _ in range(30):
                o = "v3" if rng.random() < 0.10 or not opps else rng.choice(opps)
                rows.append(series(name, o, true[name], true[o], rng))
            ck.append(name)
        fit = fit_ratings(rows, {"v3": 25.0})
        err = np.array([fit[c][0] - true[c] for c in ck])
        slope = np.polyfit(np.arange(len(ck)), err, 1)[0] * 100
        self.assertLess(abs(slope), 3.0, f"drift {slope:+.2f} mu per 100 checkpoints")
        self.assertLess(abs(err.mean()), 1.5, f"level off by {err.mean():+.2f}")


class TestAnUndefeatedReference(unittest.TestCase):
    def _field(self, seed):
        rng = random.Random(seed)
        true = {"v3": 25.0, "x": 28.0, "y": 30.0}
        rows = []
        for _ in range(300):
            a, b = rng.sample(list(true), 2)
            rows.append(series(a, b, true[a], true[b], rng))
        return rows

    def test_it_stays_finite_and_above_everyone_it_beat(self):
        """Necto has never lost a series to this league; the prior is what keeps it finite."""
        fit = fit_ratings(self._field(5) + necto_sweep(("x", "y"), 100), {"v3": 25.0})
        self.assertTrue(math.isfinite(fit["necto"][0]))
        self.assertGreater(fit["necto"][0], fit["y"][0] + 5.0)

    def test_it_cannot_move_the_scale(self):
        """
        What pinning Necto got wrong. Losing every series to a reference says almost nothing
        about the loser once the reference is free to be as strong as the results say.
        """
        rows = self._field(6)
        before = fit_ratings(rows, {"v3": 25.0})
        after = fit_ratings(rows + necto_sweep(("x", "y"), 200), {"v3": 25.0})
        for p in ("x", "y"):
            self.assertLess(abs(after[p][0] - before[p][0]), 0.5, p)


class TestTheEvaluator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lb = os.path.join(self.tmp, "lb.json")
        self.ev = TrueSkillEvaluator(leaderboard_path=self.lb)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _series(self, a, b, a_score, b_score, kind="league"):
        self.ev.record_series(a, b, {"a_score": a_score, "b_score": b_score}, kind=kind)

    def test_only_the_v3_king_is_pinned(self):
        self.assertEqual(ANCHOR_CALIBRATION, {"v3_iter198000": 25.0})
        self.assertIsNotNone(get_anchor_calibration(V3))
        for ref in ("checkpoints/necto-model.pt", "checkpoints/nexto-model.pt", "heuristic"):
            self.assertIsNone(get_anchor_calibration(ref), ref)
        rec = self.ev.get_or_create_rating(V3)
        self.assertTrue(rec.is_anchor)
        self.assertEqual(rec.mu, 25.0)
        self.assertTrue(self.ev.is_rating_frozen(rec))

    def test_the_log_is_the_boards_own_and_not_the_live_one(self):
        self.assertEqual(self.ev.results_path, os.path.join(self.tmp, "lb.results.jsonl"))

    def test_a_series_is_logged_and_every_rating_refits(self):
        v3 = self.ev.get_or_create_rating(V3)
        a = self.ev.get_or_create_rating("checkpoints/checkpoint_iter_1.pt")
        b = self.ev.get_or_create_rating("checkpoints/checkpoint_iter_2.pt")
        for _ in range(6):
            self._series(a.path, V3, 5, 2)
            self._series(b.path, a.path, 5, 3)
        self.assertEqual(v3.mu, 25.0)
        self.assertGreater(a.mu, 25.0)
        self.assertGreater(b.mu, a.mu, "b never played the anchor and is placed through a")
        self.assertEqual(len(rating_fit.load_results(self.ev.results_path)), 12)

    def test_a_benchmark_moves_no_tally(self):
        a = self.ev.get_or_create_rating("checkpoints/checkpoint_iter_1.pt")
        self._series(a.path, "checkpoints/necto-model.pt", 0, 5, kind="benchmark")
        self.assertEqual((a.matches_played, a.wins, a.losses), (0, 0, 0))

    def test_a_reload_refits_to_the_same_ratings(self):
        a = self.ev.get_or_create_rating("checkpoints/checkpoint_iter_1.pt")
        self.ev.get_or_create_rating(V3)
        for s in ((5, 2), (4, 5), (5, 1)):
            self._series(a.path, V3, *s)
        self.ev.save_leaderboard()
        again = TrueSkillEvaluator(leaderboard_path=self.lb)
        again.refit()
        self.assertAlmostEqual(again.ratings[a.path].mu, a.mu, places=3)

    def test_nothing_locks_and_an_old_lock_is_lifted(self):
        rec = self.ev.get_or_create_rating("checkpoints/checkpoint_iter_1.pt")
        rec.matches_played, rec.sigma = 500, 0.5
        self.assertFalse(self.ev.maybe_lock_rating(rec))
        rec.rating_locked = True
        self.ev.save_leaderboard()
        again = TrueSkillEvaluator(leaderboard_path=self.lb)
        self.assertFalse(again.ratings[rec.path].rating_locked)
        self.assertFalse(again.is_rating_frozen(again.ratings[rec.path]))


class TestCalibrationIsSeparateFromPromotion(unittest.TestCase):
    """
    Calibration series tie the scale; they are not evidence about how a contender compares
    with its peers. Folding them into points_rate would let the choice of calibration
    opponent decide promotions -- against a saturated Necto it put the 40% promotion floor
    out of reach however good a contender was.
    """

    def _rec(self, **kw):
        return ModelRating(name="c", path="checkpoints/c.pt", **kw)

    def test_points_rate_ignores_calibration_series(self):
        r = self._rec(matches_played=10, wins=4, draws=2, losses=4)
        self.assertEqual(r.points_rate, 50.0)
        r.calibration_matches, r.calibration_wins, r.calibration_draws = 4, 0, 0
        self.assertEqual(r.points_rate, 83.3, "the peer record is 4W-2D of 6")

    def test_calibration_defeats_cannot_sink_the_promotion_gate(self):
        r = self._rec(matches_played=6, wins=6)
        clean = r.points_rate
        for _ in range(6):
            r.matches_played += 1
            r.losses += 1
            r.calibration_matches += 1
        self.assertEqual(r.points_rate, clean)
        self.assertLess(r.win_rate, clean, "the raw record still shows them, as it should")

    def test_calibration_only_record_claims_nothing(self):
        r = self._rec(matches_played=3, losses=3, calibration_matches=3)
        self.assertEqual(r.points_rate, 0.0)


class TestCalibrationShare(unittest.TestCase):
    def _manager(self, share):
        from utils.league_manager import LeagueManager
        return LeagueManager(config={"calibration_share": share, "enabled": False})

    def _realised(self, share, trials=400, per_trial=24):
        """Drive the real debt schedule and return the fraction of graded series calibrated."""
        lm = self._manager(share)
        played = calibrated = 0
        for _ in range(trials):
            rate = lm.calibration_share / (1.0 - lm.calibration_share)
            lm._calibration_debt += rate * per_trial
            due = int(lm._calibration_debt)
            lm._calibration_debt -= due
            played += per_trial
            calibrated += due
        return calibrated / (played + calibrated)

    def test_the_realised_share_is_the_configured_share(self):
        """A fraction of all graded series; applied to peer series alone it realises 9.1%."""
        for share in (0.05, 0.10, 0.15, 0.30):
            self.assertAlmostEqual(self._realised(share), share, delta=0.005, msg=f"{share=}")

    def test_zero_share_never_calibrates(self):
        lm = self._manager(0.0)
        self.assertEqual(lm._run_calibration_series("checkpoints/c.pt", None, 100), 0)

    def test_the_share_is_clamped(self):
        self.assertEqual(self._manager(-1.0).calibration_share, 0.0)
        self.assertEqual(self._manager(5.0).calibration_share, 1.0)

    def test_the_default_config_carries_a_share(self):
        """The 10% the drift simulation was run at; below it the chain to the pin lengthens."""
        from utils.config import effective_config
        share = float(effective_config().get("league", {}).get("calibration_share", 0.0))
        self.assertGreaterEqual(share, 0.10)

    def test_calibration_is_played_against_the_pinned_rating(self):
        """Only a pinned opponent ties the fit to a fixed scale; Necto is fitted now."""
        lm = self._manager(0.1)
        if not os.path.exists(V3):
            self.skipTest("v3 king baseline not on disk")
        rec = ModelRating(name="c", path="checkpoints/c.pt", mu=40.0)
        self.assertEqual(lm._calibration_anchor(rec, exclude="checkpoints/c.pt"), V3)


class TestTheBoardReportsAgainstNecto(unittest.TestCase):
    """Necto is fitted now, so the column is a measured gap rather than a declared one."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ev = TrueSkillEvaluator(leaderboard_path=os.path.join(self.tmp, "lb.json"))
        self.ev.get_or_create_rating("checkpoints/necto-model.pt", is_anchor=True).mu = 41.0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_necto_is_the_reference(self):
        self.assertEqual(self.ev.necto_reference_mu(), 41.0)

    def test_nexto_is_not_mistaken_for_necto(self):
        self.ev.get_or_create_rating("checkpoints/nexto-model.pt", is_anchor=True).mu = 50.0
        self.assertEqual(self.ev.necto_reference_mu(), 41.0)

    def test_the_column_is_in_the_table(self):
        r = self.ev.get_or_create_rating("checkpoints/x.pt")
        r.mu, r.sigma, r.matches_played = 32.0, 1.0, 40
        df = self.ev.get_leaderboard_dataframe(ascii_safe=True)
        self.assertIn("vs Necto", df.columns)
        self.assertEqual(df.iloc[0]["vs Necto"], "-9.0")

    def test_no_necto_on_the_board_is_reported_as_unknown(self):
        self.ev.ratings.clear()
        r = self.ev.get_or_create_rating("checkpoints/x.pt")
        r.mu, r.sigma, r.matches_played = 22.0, 1.0, 40
        self.assertIsNone(self.ev.vs_necto(r))
        self.assertEqual(self.ev.get_leaderboard_dataframe(ascii_safe=True).iloc[0]["vs Necto"], "n/a")


if __name__ == "__main__":
    unittest.main()
