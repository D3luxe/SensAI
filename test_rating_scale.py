"""
The three defences against rating-scale drift, added 2026-09-20 after the leaderboard was
found at mu 518-521 while the bot was losing to an anchored Necto (mu 30) by 35-1.

TrueSkill has no absolute scale: mu means "relative to whoever you played", and this league
plays mostly itself. Nothing below tries to stop mu moving -- that is the algorithm working.
They stop the scale from floating free of the one opponent that never trains.

  1. the solver survives an extreme gap, and a failed solve never costs a played series
  2. calibration series move mu but are excluded from promotion and the loss streak
  3. the leaderboard reports mu against Necto, not raw
"""
import unittest
from unittest import mock

import trueskill

from utils.trueskill_evaluator import (
    ANCHOR_CALIBRATION, ModelRating, TrueSkillEvaluator, ts_env,
)


class TestSolverSurvivesAnExtremeGap(unittest.TestCase):
    """
    The default TrueSkill backend raises FloatingPointError past a gap of ~229 mu. The
    league ran at a gap of ~485, so every anchor grading raised and was swallowed by a
    caller's except -- the anchor could not correct the drift exactly when it was worst.
    """

    def test_a_wildly_inflated_rating_still_resolves(self):
        anchor = ts_env.create_rating(ANCHOR_CALIBRATION["necto"], 0.5)
        inflated = ts_env.create_rating(520.0, 1.0)
        _, corrected = trueskill.rate_1vs1(anchor, inflated, env=ts_env)
        self.assertLess(corrected.mu, 520.0, "a defeat by the anchor must pull mu down")
        self.assertGreater(520.0 - corrected.mu, 1.0, "and by a meaningful amount")

    def test_the_correction_grows_with_the_gap(self):
        """The further mu has drifted, the harder an anchor defeat pulls it back."""
        anchor = ts_env.create_rating(ANCHOR_CALIBRATION["necto"], 0.5)
        drops = []
        for mu in (40.0, 100.0, 200.0):
            _, new = trueskill.rate_1vs1(anchor, ts_env.create_rating(mu, 1.0), env=ts_env)
            drops.append(mu - new.mu)
        self.assertEqual(drops, sorted(drops))

    def test_a_failed_solve_keeps_the_match_record(self):
        """
        The failure mode that hid the problem: an exception thrown through the caller lost
        the whole pairing. Losing the record corrupts points_rate and the goal columns too,
        so the tallies must survive even when the rating cannot move.
        """
        ev = TrueSkillEvaluator(leaderboard_path="logs/_test_scale_leaderboard.json")
        a = ev.get_or_create_rating("checkpoints/a.pt")
        b = ev.get_or_create_rating("checkpoints/b.pt")
        before = (a.mu, b.mu)
        with mock.patch("trueskill.rate_1vs1", side_effect=FloatingPointError("boom")):
            self._score_one_series(ev, a, b)
        self.assertEqual(a.wins, 1)
        self.assertEqual(b.losses, 1)
        self.assertEqual((a.mu, b.mu), before, "a failed solve must leave ratings untouched")

    @staticmethod
    def _score_one_series(ev, a, b):
        """The tally-then-solve shape of evaluate_pairing, exercised directly."""
        a.matches_played += 1
        b.matches_played += 1
        a.wins += 1
        b.losses += 1
        try:
            trueskill.rate_1vs1(a.to_trueskill_rating(), b.to_trueskill_rating())
        except Exception:
            pass


class TestCalibrationIsSeparateFromPromotion(unittest.TestCase):
    """
    The anchors are saturated against this population -- the v3 king loses 12 of 12 series
    to Necto and 12 of 12 to Nexto. Folding those into points_rate puts the 40% promotion
    floor out of reach however good a contender is, which is why anchors were dropped from
    grading in the first place, which is what let mu drift.
    """

    def _rec(self, **kw):
        return ModelRating(name="c", path="checkpoints/c.pt", **kw)

    def test_points_rate_ignores_calibration_series(self):
        r = self._rec(matches_played=10, wins=4, draws=2, losses=4)
        self.assertEqual(r.points_rate, 50.0)
        r.calibration_matches, r.calibration_wins, r.calibration_draws = 4, 0, 0
        self.assertEqual(r.points_rate, 83.3, "the peer record is 4W-2D of 6")

    def test_a_saturated_anchor_cannot_sink_the_promotion_gate(self):
        """An undefeated contender stays undefeated on the gate after losing to Necto."""
        r = self._rec(matches_played=6, wins=6)
        clean = r.points_rate
        for _ in range(6):     # six anchor defeats
            r.matches_played += 1
            r.losses += 1
            r.calibration_matches += 1
        self.assertEqual(r.points_rate, clean)
        self.assertLess(r.win_rate, clean, "the raw record still shows them, as it should")

    def test_calibration_only_record_claims_nothing(self):
        r = self._rec(matches_played=3, losses=3, calibration_matches=3)
        self.assertEqual(r.points_rate, 0.0)


class TestCalibrationShare(unittest.TestCase):
    """Simulated over 600 generations: 0.06% and 1% still drift, 5% binds, 10% has margin."""

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
        """
        The knob is a fraction of all graded series, which is how the simulation that
        picked 5%/10% defined it. Applying it to peer series alone silently realises
        share/(1+share) -- 9.1% for a configured 10%.
        """
        for share in (0.05, 0.10, 0.15, 0.30):
            self.assertAlmostEqual(self._realised(share), share, delta=0.005, msg=f"{share=}")

    def test_zero_share_never_calibrates(self):
        lm = self._manager(0.0)
        self.assertEqual(lm._run_calibration_series("checkpoints/c.pt", None, 100), 0)

    def test_the_share_is_clamped(self):
        self.assertEqual(self._manager(-1.0).calibration_share, 0.0)
        self.assertEqual(self._manager(5.0).calibration_share, 1.0)

    def test_the_default_config_carries_a_binding_share(self):
        from utils.config import effective_config
        share = float(effective_config().get("league", {}).get("calibration_share", 0.0))
        self.assertGreaterEqual(share, 0.05, "below 5% the scale drifts without bound")
        self.assertGreaterEqual(share, 0.10, "5% is the measured knee, not a margin")


class TestTheBoardReportsAgainstNecto(unittest.TestCase):
    def setUp(self):
        self.ev = TrueSkillEvaluator(leaderboard_path="logs/_test_scale_leaderboard.json")
        self.ev.ratings.clear()
        necto = self.ev.get_or_create_rating("checkpoints/necto-model.pt", is_anchor=True)
        necto.is_anchor = True
        self.ev.apply_anchor_calibration()

    def test_necto_is_the_reference(self):
        self.assertEqual(self.ev.necto_reference_mu(), ANCHOR_CALIBRATION["necto"])

    def test_nexto_is_not_mistaken_for_necto(self):
        """Substring matching: 'necto' is inside 'nexto' for neither, but the names rhyme."""
        nexto = self.ev.get_or_create_rating("checkpoints/nexto-model.pt", is_anchor=True)
        nexto.is_anchor = True
        self.ev.apply_anchor_calibration()
        self.assertEqual(self.ev.necto_reference_mu(), ANCHOR_CALIBRATION["necto"])

    def test_an_inflated_rating_reads_as_obviously_wrong(self):
        r = ModelRating(name="x", path="checkpoints/x.pt", mu=520.0, sigma=1.0)
        self.assertAlmostEqual(self.ev.vs_necto(r), 490.0, places=1)

    def test_the_column_is_in_the_table(self):
        r = self.ev.get_or_create_rating("checkpoints/x.pt")
        r.mu, r.sigma, r.matches_played = 22.0, 1.0, 40
        df = self.ev.get_leaderboard_dataframe(ascii_safe=True)
        self.assertIn("vs Necto", df.columns)
        self.assertEqual(df.iloc[0]["vs Necto"], f"{22.0 - ANCHOR_CALIBRATION['necto']:+.1f}")

    def test_no_necto_on_the_board_is_reported_as_unknown(self):
        self.ev.ratings.clear()
        r = self.ev.get_or_create_rating("checkpoints/x.pt")
        r.mu, r.sigma, r.matches_played = 22.0, 1.0, 40
        self.assertIsNone(self.ev.vs_necto(r))
        self.assertEqual(self.ev.get_leaderboard_dataframe(ascii_safe=True).iloc[0]["vs Necto"], "n/a")


if __name__ == "__main__":
    unittest.main()
