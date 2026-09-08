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

    def test_cold_start_and_anchors(self):
        """Verify cold start gracefully falls back to anchor and never crashes."""
        self.assertIsNotNone(self.league.king_of_the_hill)
        self.assertGreaterEqual(len(self.league.elite_pool), 1)
        self.assertIn("heuristic", self.league.active_anchors)

    def test_stratified_distribution_allocation(self):
        """Test stratification logic across various environment counts."""
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
        """Verify preemption replaces lowest-mu contender when queue is full."""
        self.league.max_active_contenders = 2
        self.league.contender_queue.clear()

        paths = []
        for i, mu in enumerate([26.5, 27.0, 28.5]):
            p = os.path.join(self.test_dir, f"checkpoint_iter_{100 + i*20}.pt")
            shutil.copyfile(self.dummy_ckpt_path, p)
            r = self.evaluator.get_or_create_rating(p)
            r.mu = mu
            r.win_rate = 55.0
            r.update_conservative()
            paths.append(self.league._normalize_path(p))

        self.league._admit_contender(paths[0])
        self.league._admit_contender(paths[1])
        self.assertEqual(len(self.league.contender_queue), 2)
        self.assertIn(paths[0], self.league.contender_queue)
        self.assertIn(paths[1], self.league.contender_queue)

        # Higher mu (28.5) should preempt the lowest (26.5)
        self.league._admit_contender(paths[2])
        self.assertEqual(len(self.league.contender_queue), 2)
        self.assertNotIn(paths[0], self.league.contender_queue)
        self.assertIn(paths[1], self.league.contender_queue)
        self.assertIn(paths[2], self.league.contender_queue)

    def test_gauntlet_graduation_and_demotion(self):
        """Verify graduation when matches reach target and demotion after grace period."""
        # 1. Graduation
        c_grad = os.path.join(self.test_dir, "checkpoint_iter_200.pt")
        shutil.copyfile(self.dummy_ckpt_path, c_grad)
        norm_grad = self.league._normalize_path(c_grad)
        r_grad = self.evaluator.get_or_create_rating(norm_grad)
        r_grad.mu = 29.0
        r_grad.matches_played = self.league.target_eval_matches
        r_grad.wins = 12
        r_grad.sigma = 1.8
        r_grad.update_conservative()

        self.league.contender_queue = [norm_grad]
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
        """Verify UI functions render valid HTML with both populated and empty states."""
        from ui.app import build_league_wire_and_queue_html, load_league_state_safely

        # Render with empty evaluator
        empty_html = build_league_wire_and_queue_html(self.evaluator, league_state={})
        self.assertIn("sports-ticker-container", empty_html)
        self.assertIn("Live League Wire", empty_html)
        self.assertIn("promotion-queue-container", empty_html)

        # Render with active state
        mock_state = {
            "event_history": [
                {
                    "timestamp": "2026-09-07T22:30:00",
                    "type": "promotion",
                    "model": "Iteration 100020",
                    "detail": "Graduated Gauntlet with Score 25.99"
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
                    "win_rate": 62.5,
                    "record": "10W-4L-2D",
                    "consecutive_losses": 0,
                    "max_consecutive_losses": 4,
                    "status": "In Elimination Window"
                }
            ]
        }
        populated_html = build_league_wire_and_queue_html(self.evaluator, league_state=mock_state)
        self.assertIn("PROMOTED", populated_html)
        self.assertIn("DEMOTED", populated_html)
        self.assertIn("Iteration 101940", populated_html)
        self.assertIn("50.0%", populated_html)

    def test_title_bout_extended_trial(self):
        """Verify high-mu contenders are granted extended trial up to max_contender_matches."""
        c_title = os.path.join(self.test_dir, "checkpoint_iter_400.pt")
        shutil.copyfile(self.dummy_ckpt_path, c_title)
        norm_title = self.league._normalize_path(c_title)
        r = self.evaluator.get_or_create_rating(norm_title)
        r.mu = 32.0  # High skill
        r.matches_played = 16  # Hit standard target
        r.sigma = 2.4  # Still above target_eval_sigma (1.8)
        r.wins = 25  # Ample win padding so headless eval test matches don't breach floor
        r.losses = 2
        r.update_conservative()

        self.league.contender_queue = [norm_title]
        details = self.league.get_contender_queue_details()
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["target_matches"], 32)
        self.assertIn("Title Bout", details[0]["status"])

        # Stepping should progress instead of prematurely graduating
        res = self.league.step_contender_gauntlet()
        self.assertIsNotNone(res)
        self.assertEqual(res.get("status"), "progress")
        self.assertIn(norm_title, self.league.contender_queue)

        # Now simulate reaching max_contender_matches (32) with sufficient wins
        r.matches_played = 30
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
        rc.matches_played = 20
        rc.update_conservative()

        self.league.refresh_pool()
        # Challenger should be crowned King via competitive score
        self.assertEqual(self.league.king_of_the_hill, norm_chal)

        # Direct title bout execution
        bout_res = self.league.step_king_title_bout()
        self.assertIsNotNone(bout_res)
        self.assertIn(bout_res.get("status"), ["coronation", "defended"])

    def test_round_robin_stratified_distribution(self):
        """Verify stratified distribution rotates evenly through pool candidates without skipping."""
        # 4 envs with 50% SP, 25% King, 25% Pool -> 1 pool slot per call (the last slot)
        self.league.refresh_pool()
        candidate_pool = list(dict.fromkeys(self.league.elite_pool + self.league.active_anchors))
        self.assertGreater(len(candidate_pool), 0)

        seen = []
        for _ in range(len(candidate_pool) * 2):
            dist = self.league.get_stratified_distribution(4)
            pool_slot = dist[-1]  # The pool assignment
            seen.append(pool_slot)

        expected = [candidate_pool[i % len(candidate_pool)] for i in range(len(candidate_pool) * 2)]
        self.assertEqual(seen, expected)




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


if __name__ == "__main__":
    unittest.main()
