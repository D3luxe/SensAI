"""
Unit & Integration Tests for LeagueManager, Automated TrueSkill Grading, and Stratified Vectorized Self-Play.
"""

from __future__ import annotations
import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from agent.models import ActorCritic
from env.rocket_env import VectorizedRocketEnv
from utils.trueskill_evaluator import TrueSkillEvaluator
from utils.league_manager import LeagueManager


class TestLeagueManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sensai_test_league_")
        self.leaderboard_path = os.path.join(self.test_dir, "test_leaderboard.json")
        self.evaluator = TrueSkillEvaluator(leaderboard_path=self.leaderboard_path)

        # Create dummy checkpoint
        self.dummy_ckpt_path = os.path.join(self.test_dir, "checkpoint_iter_20.pt")
        model = ActorCritic(obs_dim=94, act_dim=8, continuous_actions=True)
        torch.save({
            "iteration": 20,
            "global_step": 20480,
            "model_state_dict": model.state_dict(),
            "continuous_actions": True,
            "use_layer_norm": True
        }, self.dummy_ckpt_path)

        self.league = LeagueManager(
            evaluator=self.evaluator,
            config={
                "enabled": True,
                "self_play_ratio": 0.50,
                "king_ratio": 0.25,
                "pool_ratio": 0.25,
                "eval_matches_per_grade": 2,
                "eval_max_steps": 50,  # Fast for unit tests
                "protect_top_k": 3
            },
            leaderboard_path=self.leaderboard_path
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cold_start_leaves_the_throne_empty(self):
        """With nothing rated, there is no King: anchors are barred from the crown."""
        self.assertIsNone(self.league.king_of_the_hill)
        self.assertGreaterEqual(len(self.league.elite_pool), 1)
        self.assertIn("heuristic", self.league.active_anchors)

    def test_anchor_never_takes_the_crown(self):
        """Even rated far above every checkpoint, an anchor cannot be King."""
        strong_anchor = self.evaluator.get_or_create_rating("heuristic", is_anchor=True)
        strong_anchor.mu, strong_anchor.sigma = 99.0, 0.5
        strong_anchor.update_conservative()

        ckpt = self.league._normalize_path(self.dummy_ckpt_path)
        rec = self.evaluator.get_or_create_rating(ckpt)
        rec.mu, rec.sigma, rec.matches_played = 20.0, 1.0, 40
        rec.update_conservative()

        self.league.refresh_pool()
        self.assertEqual(self.league.king_of_the_hill, ckpt)

    def test_cold_start_distribution_is_pure_self_play(self):
        """
        No rated checkpoint means the pool holds only anchors, which span a bot that
        loses 100-0 to everything and one that beats the policy four times in five.
        Neither is a useful sparring partner, so training falls back to self-play.
        """
        dist = self.league.get_stratified_distribution(64)
        self.assertEqual(len(dist), 64)
        self.assertEqual(dist.count(None), 64)

    def _seat_a_king(self):
        """Rate the dummy checkpoint so a King exists and the standard split applies."""
        ckpt = self.league._normalize_path(self.dummy_ckpt_path)
        rec = self.evaluator.get_or_create_rating(ckpt)
        rec.mu, rec.sigma, rec.matches_played = 26.0, 1.0, 40
        rec.update_conservative()
        self.league.refresh_pool()
        self.assertIsNotNone(self.league.king_of_the_hill)

    def test_stratified_distribution_allocation(self):
        """Test stratification logic across various environment counts."""
        self._seat_a_king()
        # 64 envs: 32 self-play, 16 king, 16 pool
        dist64 = self.league.get_stratified_distribution(64)
        self.assertEqual(len(dist64), 64)
        self.assertEqual(dist64.count(None), 32)
        # Remaining 32 are opponents
        self.assertEqual(sum(x is not None for x in dist64), 32)

        # 16 envs: 8 self-play, 4 king, 4 pool
        dist16 = self.league.get_stratified_distribution(16)
        self.assertEqual(len(dist16), 16)
        self.assertEqual(dist16.count(None), 8)
        self.assertEqual(sum(x is not None for x in dist16), 8)

        # 4 envs: 2 self-play, 1 king, 1 pool
        dist4 = self.league.get_stratified_distribution(4)
        self.assertEqual(len(dist4), 4)
        self.assertEqual(dist4.count(None), 2)
        self.assertEqual(sum(x is not None for x in dist4), 2)

        # 1 env: Temporal alternation across rollouts
        t1 = self.league.get_stratified_distribution(1)
        t2 = self.league.get_stratified_distribution(1)
        t3 = self.league.get_stratified_distribution(1)
        self.assertEqual(len(t1), 1)
        self.assertEqual(len(t2), 1)
        self.assertEqual(len(t3), 1)
        # Should have cycled through different modes
        cycle_types = [t1[0] is None, t2[0] is None, t3[0] is None]
        self.assertTrue(any(cycle_types), "Temporal alternation must include self-play")
        self.assertFalse(all(cycle_types), "Temporal alternation must also include opponent bots")

    def test_auto_grade_checkpoint(self):
        """Verify auto-grading runs matches, updates Bayesian ratings, and persists results."""
        rec = self.league.grade_checkpoint(self.dummy_ckpt_path)
        self.assertIsNotNone(rec)
        self.assertGreater(rec.matches_played, 0)
        self.assertGreater(rec.sigma, 0.0)

        # Check leaderboard file exists and contains the model
        self.assertTrue(os.path.exists(self.leaderboard_path))
        self.evaluator.load_leaderboard()
        norm_key = self.league._normalize_path(self.dummy_ckpt_path)
        self.assertIn(norm_key, self.evaluator.ratings)

    def test_anchors_rank_by_calibrated_mu_not_match_count(self):
        """A calibrated anchor outranks a weaker checkpoint despite having no record."""
        strong = os.path.join(self.test_dir, "checkpoint_iter_900.pt")
        shutil.copyfile(self.dummy_ckpt_path, strong)
        rec = self.evaluator.get_or_create_rating(self.league._normalize_path(strong))
        rec.mu, rec.sigma, rec.matches_played = 26.0, 1.0, 60
        rec.update_conservative()

        necto = self.evaluator.ratings.get("checkpoints/necto-model.pt")
        if necto is None:
            self.skipTest("necto anchor not present in this environment")

        # Anchors never accumulate a record -- their rating is declared, not earned -- so
        # gating them on match count would sort them below every ranked checkpoint.
        self.assertTrue(self.league._is_rank_eligible(necto))
        self.assertGreater(self.league._ranking_key(necto), self.league._ranking_key(rec))

    def test_protected_checkpoints(self):
        """Verify protected checkpoints include top-K models and King-of-the-Hill."""
        self.league.grade_checkpoint(self.dummy_ckpt_path)
        protected = self.league.get_protected_checkpoint_paths()
        abs_dummy = os.path.abspath(self.dummy_ckpt_path)
        self.assertIn(abs_dummy, protected)

    def test_telemetry(self):
        """Verify telemetry output structure."""
        telem = self.league.get_telemetry()
        self.assertIn("league_enabled", telem)
        self.assertIn("king_of_the_hill", telem)
        self.assertIn("king_score", telem)
        self.assertIn("elite_pool_size", telem)
        self.assertTrue(telem["league_enabled"])

    def test_latest_model_permanently_excluded(self):
        """Verify latest_model.pt is never assigned King of the Hill or entered into elite pool."""
        latest_path = os.path.join(self.test_dir, "latest_model.pt")
        shutil.copyfile(self.dummy_ckpt_path, latest_path)
        rec = self.evaluator.get_or_create_rating(latest_path)
        rec.mu = 40.0
        rec.sigma = 0.5
        rec.update_conservative()

        self.league.refresh_pool()
        norm_latest = self.league._normalize_path(latest_path)
        self.assertNotEqual(self.league.king_of_the_hill, norm_latest)
        self.assertNotIn(norm_latest, self.league.elite_pool)

    def test_contender_queue_admission_and_protection(self):
        """Verify qualification on debut and immunity in get_protected_checkpoint_paths."""
        ckpt_a = os.path.join(self.test_dir, "checkpoint_iter_40.pt")
        shutil.copyfile(self.dummy_ckpt_path, ckpt_a)
        rec_a = self.evaluator.get_or_create_rating(ckpt_a)
        rec_a.mu = 27.5
        rec_a.win_rate = 60.0
        rec_a.update_conservative()

        norm_a = self.league._normalize_path(ckpt_a)
        self.league._admit_contender(norm_a)
        self.assertIn(norm_a, self.league.contender_queue)

        protected = self.league.get_protected_checkpoint_paths()
        self.assertIn(os.path.abspath(ckpt_a), protected)

    def test_contender_queue_preemption(self):
        """Preemption evicts the most-converged contender, never the least-measured one."""
        self.league.max_active_contenders = 2
        self.league.eligibility_sigma = 1.5
        self.league.target_eval_matches = 16
        self.league.contender_queue.clear()

        # Two sitting contenders: one converged (rank-eligible), one still being measured.
        settled = os.path.join(self.test_dir, "checkpoint_iter_100.pt")
        unsettled = os.path.join(self.test_dir, "checkpoint_iter_120.pt")
        newcomer = os.path.join(self.test_dir, "checkpoint_iter_140.pt")
        specs = [(settled, 27.0, 0.9, 40), (unsettled, 26.5, 2.6, 8), (newcomer, 32.0, 3.5, 4)]
        paths = []
        for path, mu, sigma, matches in specs:
            shutil.copyfile(self.dummy_ckpt_path, path)
            rec = self.evaluator.get_or_create_rating(path)
            rec.mu, rec.sigma, rec.matches_played = mu, sigma, matches
            rec.wins, rec.draws = matches // 2, 0
            rec.update_conservative()
            paths.append(self.league._normalize_path(path))
        norm_settled, norm_unsettled, norm_new = paths

        self.league._admit_contender(norm_settled)
        self.league._admit_contender(norm_unsettled)
        self.assertEqual(len(self.league.contender_queue), 2)

        # The newcomer displaces the converged contender, not the half-measured one.
        # Under the old mu-based rule it would have evicted the *lowest* mu, which is the
        # contender with the most still to learn.
        self.league._admit_contender(norm_new)
        self.assertEqual(len(self.league.contender_queue), 2)
        self.assertNotIn(norm_settled, self.league.contender_queue)
        self.assertIn(norm_unsettled, self.league.contender_queue)
        self.assertIn(norm_new, self.league.contender_queue)

    def test_preemption_refuses_when_queue_is_all_unconverged(self):
        """A queue of un-measured contenders is left alone; the newcomer waits its turn."""
        self.league.max_active_contenders = 2
        self.league.eligibility_sigma = 1.5
        self.league.contender_queue.clear()

        paths = []
        for i, (mu, sigma, matches) in enumerate([(26.0, 2.8, 8), (26.5, 3.0, 4), (33.0, 4.0, 4)]):
            path = os.path.join(self.test_dir, f"checkpoint_iter_{700 + i * 20}.pt")
            shutil.copyfile(self.dummy_ckpt_path, path)
            rec = self.evaluator.get_or_create_rating(path)
            rec.mu, rec.sigma, rec.matches_played = mu, sigma, matches
            rec.update_conservative()
            paths.append(self.league._normalize_path(path))

        self.league._admit_contender(paths[0])
        self.league._admit_contender(paths[1])
        self.league._admit_contender(paths[2])

        self.assertEqual(len(self.league.contender_queue), 2)
        self.assertNotIn(paths[2], self.league.contender_queue)

    def test_gauntlet_graduation_and_demotion(self):
        """Verify graduation when matches reach target and demotion after grace period."""
        # 1. Graduation
        c_grad = os.path.join(self.test_dir, "checkpoint_iter_200.pt")
        shutil.copyfile(self.dummy_ckpt_path, c_grad)
        norm_grad = self.league._normalize_path(c_grad)
        r_grad = self.evaluator.get_or_create_rating(norm_grad)
        r_grad.mu = 29.0
        r_grad.matches_played = self.league.target_eval_matches
        r_grad.wins = 20
        r_grad.sigma = self.league.eligibility_sigma + 0.3   # still un-converged
        r_grad.update_conservative()
        self.league.max_consecutive_losses = 99   # isolate graduation from match outcomes
        # A production-sized 12-game trial would converge this rating past the gate
        # inside the first step, which is the opposite of what this fixture tests.
        self.league.contender_eval_matches_per_step = 2

        # Hitting the match target is not enough on its own: while sigma is above the
        # ranking gate, graduating would push the contender out of the queue and below
        # the pool, where it would never play again and never converge.
        self.league.contender_queue = [norm_grad]
        res_early = self.league.step_contender_gauntlet()
        self.assertIsNotNone(res_early)
        self.assertNotEqual(res_early["status"], "graduated")
        self.assertIn(norm_grad, self.league.contender_queue)

        # Once the rating converges past the gate, it graduates.
        r_grad.sigma = self.league.eligibility_sigma - 0.2
        r_grad.matches_played = max(r_grad.matches_played, self.league.target_eval_matches)
        r_grad.update_conservative()
        res = self.league.step_contender_gauntlet()
        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "graduated")
        self.assertNotIn(norm_grad, self.league.contender_queue)

        # 2. Demotion (after grace period)
        c_dem = os.path.join(self.test_dir, "checkpoint_iter_220.pt")
        shutil.copyfile(self.dummy_ckpt_path, c_dem)
        norm_dem = self.league._normalize_path(c_dem)
        r_dem = self.evaluator.get_or_create_rating(norm_dem)
        r_dem.mu = 23.0  # Below 25.5
        r_dem.matches_played = self.league.grace_period_matches + 1
        r_dem.update_conservative()

        self.league.contender_queue = [norm_dem]
        res_dem = self.league.step_contender_gauntlet()
        self.assertIsNotNone(res_dem)
        self.assertEqual(res_dem["status"], "demoted")
        self.assertNotIn(norm_dem, self.league.contender_queue)

    def test_league_state_persistence_and_events(self):
        """Verify events are properly recorded in event_history and saved/loaded to league_state.json."""
        # 1. Trigger admission
        c1 = os.path.join(self.test_dir, "checkpoint_iter_300.pt")
        shutil.copyfile(self.dummy_ckpt_path, c1)
        norm_c1 = self.league._normalize_path(c1)
        r1 = self.evaluator.get_or_create_rating(norm_c1)
        r1.mu = 27.5
        r1.win_rate = 60.0
        self.league._admit_contender(norm_c1)

        self.assertGreaterEqual(len(self.league.event_history), 1)
        latest_event = self.league.event_history[-1]
        self.assertEqual(latest_event["type"], "admission")

        # 2. Check persistence on disk
        state_file = self.league.league_state_path
        self.assertTrue(os.path.exists(state_file))

        # 3. Test get_contender_queue_details
        details = self.league.get_contender_queue_details()
        self.assertEqual(len(details), len(self.league.contender_queue))
        self.assertIn("progress_pct", details[0])
        self.assertIn("status", details[0])

        # 4. Test loading into a new manager instance
        new_league = LeagueManager(
            evaluator=self.evaluator,
            leaderboard_path=self.leaderboard_path,
            config={"league_state_path": state_file}
        )
        self.assertEqual(len(new_league.contender_queue), len(self.league.contender_queue))
        self.assertGreaterEqual(len(new_league.event_history), 1)

    def test_ui_league_wire_and_queue_rendering(self):
        """The league board renders in both the empty and the populated state."""
        from ui.app import build_league_wire_and_queue_html

        empty_html = build_league_wire_and_queue_html(self.evaluator, league_state={})
        self.assertIn("league-board", empty_html)
        self.assertIn("Gauntlet Wire", empty_html)
        self.assertIn("Gauntlet Trials", empty_html)
        self.assertIn("Elite Pool", empty_html)

        mock_state = {
            "king_of_the_hill": "checkpoints/checkpoint_iter_101940.pt",
            "event_history": [
                {
                    "timestamp": "2026-09-07T22:30:00",
                    "type": "promotion",
                    "model": "Iteration 100020",
                    "detail": "Graduated Gauntlet to Elite Pool"
                },
                {
                    "timestamp": "2026-09-07T22:35:00",
                    "type": "demotion",
                    "model": "Iteration 98200",
                    "detail": "Loss streak knockout"
                }
            ],
            "contenders": [
                {
                    "name": "Iteration 101940",
                    "path": "checkpoints/checkpoint_iter_101940.pt",
                    "mu": 31.94,
                    "sigma": 2.29,
                    "conservative_score": 25.06,
                    "matches_played": 8,
                    "target_matches": 16,
                    "progress_pct": 50.0,
                    "win_rate": 37.5,
                    "points_rate": 62.5,
                    "record": "3W-3L-2D",
                    "consecutive_losses": 0,
                    "max_consecutive_losses": 4,
                    "status": "In Trial"
                }
            ],
            "elite_pool_details": [
                {
                    "rank": 1, "name": "Iteration 101940",
                    "path": "checkpoints/checkpoint_iter_101940.pt",
                    "mu": 30.10, "sigma": 0.92, "conservative_score": 27.34,
                    "win_rate": 41.0, "points_rate": 55.0, "record": "41W-30L-29D",
                    "matches_played": 100, "is_anchor": False, "is_king": True
                },
                {
                    "rank": 2, "name": "Necto (EARL TorchScript)",
                    "path": "checkpoints/necto-model.pt",
                    "mu": 30.00, "sigma": 0.50, "conservative_score": 28.50,
                    "win_rate": 0.0, "points_rate": 0.0, "record": "0W-0L-0D",
                    "matches_played": 0, "is_anchor": True, "is_king": False
                },
            ],
        }
        populated = build_league_wire_and_queue_html(self.evaluator, league_state=mock_state)
        self.assertIn("Promoted", populated)
        self.assertIn("Demoted", populated)
        self.assertIn("Iteration 101940", populated)
        # Points rate, not raw win rate, is what the trial card reports.
        self.assertIn("62.5%", populated)
        self.assertNotIn("37.5%", populated)
        # A demotion animates downward, a promotion upward.
        self.assertIn("lb-tick-down", populated)
        self.assertIn("lb-tick-up", populated)
        # An anchor shows no fabricated points figure.
        self.assertIn("reference", populated)

    def test_title_bout_extended_trial(self):
        """Verify high-mu contenders are granted extended trial up to max_contender_matches."""
        c_title = os.path.join(self.test_dir, "checkpoint_iter_400.pt")
        shutil.copyfile(self.dummy_ckpt_path, c_title)
        norm_title = self.league._normalize_path(c_title)
        r = self.evaluator.get_or_create_rating(norm_title)
        r.mu = 32.0  # High skill
        r.matches_played = self.league.target_eval_matches  # Hit standard target
        r.sigma = 2.4  # Still above target_eval_sigma (1.8)
        r.wins = 25  # Ample win padding so headless eval test matches don't breach floor
        r.losses = 2
        r.update_conservative()

        # This test asserts the extended-trial *target*, not a match outcome. The trial
        # plays four real simulated games, so a chance losing streak would otherwise trip
        # the knockout rule and demote the contender, making the assertion a coin flip.
        self.league.max_consecutive_losses = 99
        # Keep the trial short so sigma stays above target_eval_sigma and the contender
        # remains in the extended-trial window this test is about.
        self.league.contender_eval_matches_per_step = 2

        self.league.contender_queue = [norm_title]
        details = self.league.get_contender_queue_details()
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["target_matches"], self.league.max_contender_matches)
        self.assertIn("Title Bout", details[0]["status"])

        # Stepping should progress instead of prematurely graduating
        res = self.league.step_contender_gauntlet()
        self.assertIsNotNone(res)
        self.assertEqual(res.get("status"), "progress")
        self.assertIn(norm_title, self.league.contender_queue)

        # Now simulate reaching max_contender_matches with sufficient wins
        r.matches_played = self.league.max_contender_matches - 2
        r.wins = 25
        r.losses = 5
        r.update_conservative()
        res_grad = self.league.step_contender_gauntlet()
        self.assertIsNotNone(res_grad)
        self.assertEqual(res_grad.get("status"), "graduated")
        self.assertNotIn(norm_title, self.league.contender_queue)

    def test_king_title_bout_coronation(self):
        """Verify King title bout runs and crowns a superior challenger."""
        c_king = os.path.join(self.test_dir, "checkpoint_iter_500.pt")
        shutil.copyfile(self.dummy_ckpt_path, c_king)
        norm_king = self.league._normalize_path(c_king)
        rk = self.evaluator.get_or_create_rating(norm_king)
        rk.mu = 26.0
        rk.sigma = 0.9
        rk.matches_played = 500
        rk.update_conservative()

        c_challenger = os.path.join(self.test_dir, "checkpoint_iter_520.pt")
        shutil.copyfile(self.dummy_ckpt_path, c_challenger)
        norm_chal = self.league._normalize_path(c_challenger)
        rc = self.evaluator.get_or_create_rating(norm_chal)
        rc.mu = 34.0
        rc.sigma = 2.0
        rc.matches_played = self.league.target_eval_matches
        rc.update_conservative()

        # Still above the eligibility sigma, so a high mu alone does not take the crown.
        # This is the debut protection: rank is gated on convergence, not on raw skill.
        self.league.refresh_pool()
        self.assertEqual(self.league.king_of_the_hill, norm_king)

        # Once its rating converges, the stronger challenger outranks the incumbent on
        # raw mu, with no sample-size penalty for the incumbent's 500 extra matches.
        rc.sigma = 1.1
        rc.update_conservative()
        self.league.refresh_pool()
        self.assertEqual(self.league.king_of_the_hill, norm_chal)

        # Direct title bout execution
        bout_res = self.league.step_king_title_bout()
        self.assertIsNotNone(bout_res)
        self.assertIn(bout_res.get("status"), ["coronation", "defended"])

    def test_round_robin_stratified_distribution(self):
        """Verify stratified distribution rotates evenly through pool candidates without skipping."""
        # 4 envs with 50% SP, 25% King, 25% Pool -> 1 pool slot per call (the last slot)
        self._seat_a_king()
        explicit = set(self.league.training_opponents)
        candidate_pool = [
            m for m in dict.fromkeys(self.league.elite_pool + self.league.active_anchors)
            if m not in explicit
        ]
        self.assertGreater(len(candidate_pool), 0)

        seen = []
        for _ in range(len(candidate_pool) * 2):
            dist = self.league.get_stratified_distribution(4)
            pool_slot = dist[-1]  # The pool assignment
            seen.append(pool_slot)

        expected = [candidate_pool[i % len(candidate_pool)] for i in range(len(candidate_pool) * 2)]
        self.assertEqual(seen, expected)



    def test_training_opponents_take_an_even_share(self):
        """
        The ratio squashes the standard split rather than carving out of one tier, and
        the listed opponents divide their share evenly. Listed models are excluded from
        the pool rotation so their exposure is exactly the ratio, not the ratio plus
        whatever round-robin slots they happen to draw.
        """
        self._seat_a_king()
        self.league.training_opponents = ["heuristic", "checkpoints/necto-model.pt"]
        self.league.training_opponent_ratio = 0.25

        counts = {}
        total = 0
        for _ in range(8):
            for slot in self.league.get_stratified_distribution(64):
                counts[slot] = counts.get(slot, 0) + 1
                total += 1

        listed = [op for op in self.league.training_opponents
                  if op == "heuristic" or os.path.exists(op)]
        shares = [counts.get(op, 0) / total for op in listed]
        self.assertEqual(len(shares), len(listed))
        # Roughly a quarter of environments, split evenly between the listed entries.
        self.assertAlmostEqual(sum(shares), 0.25, delta=0.03)
        for share in shares:
            self.assertAlmostEqual(share, 0.25 / len(listed), delta=0.02)
        # Self-play keeps its 50% of what remains.
        self.assertAlmostEqual(counts.get(None, 0) / total, 0.75 * 0.5, delta=0.03)

    def test_zero_ratio_reproduces_the_standard_split(self):
        """An empty list, or a ratio of zero, must leave the old behaviour untouched."""
        self._seat_a_king()
        self.league.training_opponents = ["heuristic"]
        self.league.training_opponent_ratio = 0.0
        dist = self.league.get_stratified_distribution(64)
        self.assertEqual(dist.count(None), 32)


class TestStratifiedVectorizedEnv(unittest.TestCase):
    def test_stratified_opponents_and_learner_mask(self):
        vec_env = VectorizedRocketEnv(
            num_envs=4,
            game_mode="1v1",
            tick_skip=8,
            max_episode_steps=100
        )

        # Env 0: None (Self-Play)
        # Env 1: heuristic
        # Env 2: None (Self-Play)
        # Env 3: heuristic
        assignments = [None, "heuristic", None, "heuristic"]
        vec_env.set_stratified_opponents(assignments)

        self.assertFalse(vec_env.envs[0].is_baseline_env)
        self.assertTrue(vec_env.envs[1].is_baseline_env)
        self.assertFalse(vec_env.envs[2].is_baseline_env)
        self.assertTrue(vec_env.envs[3].is_baseline_env)

        mask = vec_env.get_learner_mask()
        # Shape: 4 envs * 2 players = 8
        self.assertEqual(len(mask), 8)
        # Expected: [True, True, True, False, True, True, True, False]
        expected = np.array([True, True, True, False, True, True, True, False])
        np.testing.assert_array_equal(mask, expected)

        # Test stepping
        obs = vec_env.reset()
        self.assertEqual(obs.shape, (4, 2, vec_env.obs_dim))
        actions = np.zeros((4, 2, vec_env.act_dim), dtype=np.float32)
        next_obs, rews, dones, infos = vec_env.step(actions)
        self.assertEqual(next_obs.shape, (4, 2, vec_env.obs_dim))
        self.assertEqual(rews.shape, (4, 2))
        self.assertEqual(len(infos), 4)


class TestCheckpointRetentionTiers(unittest.TestCase):
    """Retention is the union of provisional / ranked / archive, with no rolling cap."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sensai_test_retention_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _touch(self, iteration: int) -> str:
        path = os.path.join(self.test_dir, f"checkpoint_iter_{iteration}.pt")
        with open(path, "w") as f:
            f.write("stub")
        return path

    def test_archive_tier_preserves_historical_spine(self):
        """One checkpoint per archive_stride survives, however old, alongside the newest."""
        from agent.ppo import PPOTrainer

        for i in range(200, 30001, 200):
            self._touch(i)
        latest = os.path.join(self.test_dir, "latest_model.pt")
        manual = os.path.join(self.test_dir, "my_manual_save.pt")
        for p in (latest, manual):
            with open(p, "w") as f:
                f.write("stub")

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.save_dir = self.test_dir
        trainer.archive_stride = 5000
        trainer.max_checkpoints_to_keep = 10
        trainer.league_manager = None
        trainer.cleanup_old_checkpoints()

        kept = sorted(
            int(f.replace("checkpoint_iter_", "").replace(".pt", ""))
            for f in os.listdir(self.test_dir) if f.startswith("checkpoint_iter_")
        )

        # Archive spine: earliest survivor of each 5000-wide bucket.
        for spine in (200, 5000, 10000, 15000, 20000, 25000):
            self.assertIn(spine, kept)
        # Recency backstop (no league manager attached).
        self.assertIn(30000, kept)
        # Everything outside both tiers is pruned.
        self.assertNotIn(12000, kept)
        # Non-numbered saves are never touched.
        self.assertTrue(os.path.exists(latest))
        self.assertTrue(os.path.exists(manual))

    def test_provisional_tier_protects_unconverged_newcomers(self):
        """Newest un-converged checkpoints survive until the evaluator reaches them."""
        cwd = os.getcwd()
        os.chdir(self.test_dir)
        try:
            os.makedirs("checkpoints", exist_ok=True)
            paths = []
            for i in range(1000, 60001, 1000):
                path = f"checkpoints/checkpoint_iter_{i}.pt"
                with open(path, "w") as f:
                    f.write("stub")
                paths.append((i, path))

            league = LeagueManager(
                config={
                    "max_provisional": 5,
                    "eligibility_sigma": 1.5,
                    "anchors": ["heuristic"],
                    "max_pool_size": 10,
                    "protect_top_k": 20,
                },
                leaderboard_path=os.path.join(self.test_dir, "lb.json")
            )
            for idx, (_, path) in enumerate(paths):
                rec = league.evaluator.get_or_create_rating(path)
                converged = idx < 20
                rec.mu = 30.0 + (idx * 0.01 if converged else 0.0)
                rec.sigma = 0.9 if converged else 3.0
                rec.matches_played = 100 if converged else 4
                rec.update_conservative()

            protected = league.get_protected_checkpoint_paths()
            kept = {i for i, path in paths if os.path.normcase(os.path.abspath(path)) in protected}
        finally:
            os.chdir(cwd)

        # The five newest un-converged checkpoints are held for measurement.
        for i in (56000, 57000, 58000, 59000, 60000):
            self.assertIn(i, kept)
        # Un-converged checkpoints that aged past the window lost their chance.
        for i in (25000, 30000, 40000, 50000):
            self.assertNotIn(i, kept)
        # Converged veterans are retained by the ranked tier.
        self.assertIn(1000, kept)


if __name__ == "__main__":
    unittest.main()
