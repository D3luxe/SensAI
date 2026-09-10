"""
Unit & Integration Tests for LeagueManager, Automated TrueSkill Grading, and Stratified Vectorized Self-Play.
"""

from __future__ import annotations
import os
import shutil
import tempfile
import json
import unittest
import yaml
import numpy as np
import torch

from agent.models import ActorCritic
from env.rocket_env import VectorizedRocketEnv
from utils.trueskill_evaluator import TrueSkillEvaluator, trueskill
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
        """With nothing rated, there is no King and no pool: anchors fill neither."""
        self.assertIsNone(self.league.king_of_the_hill)
        self.assertEqual(self.league.elite_pool, [])
        self.assertIn("heuristic", self.league.active_anchors)

    def test_cold_start_falls_back_to_self_play(self):
        """
        An empty pool must still produce one assignment per environment.

        The pool tier has nowhere to draw from before any checkpoint is rated, and the
        anchors are not a substitute. Self-play is, and a short list would silently leave
        the tail environments facing whatever they faced last.
        """
        dist = self.league.get_stratified_distribution(8)
        self.assertEqual(len(dist), 8)
        self.assertTrue(all(x is None for x in dist))

    def test_a_reference_model_never_enters_the_pool(self):
        """
        Anchors are graded against, not trained against.

        Nexto carries the highest declared mu on the ladder, so ranking alone would seat
        it at the top of the pool permanently. Deliberate exposure to a reference is what
        the training_opponents list is for.
        """
        for name in ("heuristic", "checkpoints/necto-model.pt"):
            anchor = self.evaluator.get_or_create_rating(name, is_anchor=True)
            anchor.mu, anchor.sigma = 99.0, 0.4
            anchor.update_conservative()
        ckpt = self.league._normalize_path(self.dummy_ckpt_path)
        rec = self.evaluator.get_or_create_rating(ckpt)
        rec.mu, rec.sigma, rec.matches_played = 30.0, 1.0, 40
        rec.update_conservative()

        self.league.refresh_pool()
        self.assertIn(ckpt, self.league.elite_pool)
        for path in self.league.elite_pool:
            norm = self.league._normalize_path(path)
            self.assertNotEqual(norm, "heuristic")
            self.assertFalse(
                getattr(self.evaluator.ratings.get(norm), "is_anchor", False),
                f"anchor {norm} leaked into the elite pool",
            )

    def test_pool_groups_align_with_env_worker_slices(self):
        """
        Every subprocess worker must face exactly one opponent model.

        A worker owns a contiguous slice of environments and a rollout step waits on all
        of them, so a worker holding two models runs two unbatched forward passes and
        sets the pace for every environment. That holds only while pool_group_size equals
        environments per worker.
        """
        self._seat_a_king()
        for extra in range(6):
            rec = self.evaluator.get_or_create_rating(
                self.league._normalize_path(f"checkpoints/checkpoint_iter_{900 + extra}.pt")
            )
            rec.mu, rec.sigma, rec.matches_played = 26.0 - extra * 0.1, 1.0, 40
            rec.update_conservative()

        num_envs, workers = 128, 16
        per_worker = num_envs // workers
        self.league.pool_group_size = per_worker
        for _ in range(20):
            dist = self.league.get_stratified_distribution(num_envs)
            self.assertEqual(len(dist), num_envs)
            for w in range(workers):
                slice_ = set(dist[w * per_worker:(w + 1) * per_worker])
                self.assertEqual(len(slice_), 1, f"worker {w} straddles models: {slice_}")

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
        # An established King is what defines the floor; without one there is no skill
        # floor at all, which is the fresh-leaderboard safeguard.
        king_path = self.league._normalize_path(c_grad)
        king = self.evaluator.get_or_create_rating(king_path)
        king.mu, king.sigma, king.matches_played = 35.0, 1.0, 60
        king.update_conservative()
        self.league.king_of_the_hill = king_path

        r_dem.mu = 23.0            # confidently below king 35.0 - margin
        r_dem.sigma = 1.0
        r_dem.matches_played = self.league.grace_period_matches + 1
        r_dem.wins, r_dem.losses = 3, 4
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

    def _queued_contender(self, name, mu, sigma, wins, losses, draws=0):
        """Puts a rated contender in the queue so a gauntlet exit can be evaluated."""
        path = self.league._normalize_path(f"checkpoints/{name}.pt")
        rec = self.evaluator.get_or_create_rating(path)
        rec.mu, rec.sigma = mu, sigma
        rec.wins, rec.losses, rec.draws = wins, losses, draws
        rec.matches_played = wins + losses + draws
        rec.update_conservative()
        if path not in self.league.contender_queue:
            self.league.contender_queue.append(path)
        return path, rec

    def _gauntlet_exit(self, rec):
        """
        The exit decision for a contender, without playing any matches.

        Mirrors the demote-then-graduate order in step_contender_gauntlet, which is the
        property under test: the two branches must be exhaustive.
        """
        import math

        lm = self.league
        if rec.matches_played >= lm.grace_period_matches:
            floor = lm._contender_mu_floor()
            if floor is not None and (rec.mu + rec.sigma) < floor:
                return "demote"
            if rec.matches_played >= lm.min_matches_for_points_floor:
                rate = max(0.0, min(1.0, rec.points_rate / 100.0))
                spread = math.sqrt(
                    max(rate * (1.0 - rate), 0.01) / max(1, rec.matches_played)
                ) * 100.0
                if rec.points_rate + spread < lm.min_contender_points_rate:
                    return "demote"
        if lm._is_rank_eligible(rec) or rec.matches_played >= lm.max_contender_matches:
            return "graduate"
        return "keep playing"

    def test_a_new_checkpoint_inherits_its_predecessor_mu(self):
        """
        Seer's seeding rule: mu from the previous agent, sigma from the environment.

        Seeding every arrival at the global 25.0 held the population about three mu above
        where the anchor ladder measures it, because a handful of anchor games cannot pull
        back a prior that wide.
        """
        older = self.league._normalize_path("checkpoints/checkpoint_iter_100.pt")
        rec_old = self.evaluator.get_or_create_rating(older)
        rec_old.mu, rec_old.sigma, rec_old.matches_played = 21.4, 1.2, 40
        rec_old.update_conservative()

        newer = self.league._normalize_path("checkpoints/checkpoint_iter_300.pt")
        rec_new = self.evaluator.get_or_create_rating(newer)
        default_sigma = rec_new.sigma

        self.assertTrue(self.league.seed_from_predecessor(newer, rec_new))
        self.assertAlmostEqual(rec_new.mu, 21.4, places=3)
        self.assertAlmostEqual(rec_new.sigma, default_sigma, places=3,
                               msg="certainty must not be inherited, only the estimate")

    def test_seeding_ignores_a_barely_played_predecessor(self):
        """
        A one or two series rating is not a measurement worth inheriting.

        The abandoned debuts left on this leaderboard from before the debut skip range
        from mu 18.89 to 22.92 between neighbouring saves. Passing that straight to the
        next checkpoint would dress noise up as a starting point.
        """
        noisy = self.league._normalize_path("checkpoints/checkpoint_iter_200.pt")
        rn = self.evaluator.get_or_create_rating(noisy)
        rn.mu, rn.sigma, rn.matches_played = 19.0, 5.8, 2
        rn.update_conservative()

        newer = self.league._normalize_path("checkpoints/checkpoint_iter_400.pt")
        rec = self.evaluator.get_or_create_rating(newer)
        default_mu = rec.mu
        self.assertFalse(self.league.seed_from_predecessor(newer, rec))
        self.assertAlmostEqual(rec.mu, default_mu, places=3)

        # A measured one further back is inherited instead of the noisy neighbour.
        measured = self.league._normalize_path("checkpoints/checkpoint_iter_100.pt")
        rm = self.evaluator.get_or_create_rating(measured)
        rm.mu, rm.sigma, rm.matches_played = 21.4, 1.2, 40
        rm.update_conservative()
        rec2 = self.evaluator.get_or_create_rating(
            self.league._normalize_path("checkpoints/checkpoint_iter_500.pt")
        )
        self.assertTrue(self.league.seed_from_predecessor(
            self.league._normalize_path("checkpoints/checkpoint_iter_500.pt"), rec2))
        self.assertAlmostEqual(rec2.mu, 21.4, places=3)

    def test_seeding_only_applies_to_a_checkpoint_that_has_not_played(self):
        """A record with results of its own is measured, so it is not re-seeded."""
        older = self.league._normalize_path("checkpoints/checkpoint_iter_100.pt")
        r = self.evaluator.get_or_create_rating(older)
        r.mu, r.matches_played = 21.0, 30
        r.update_conservative()

        newer = self.league._normalize_path("checkpoints/checkpoint_iter_300.pt")
        rn = self.evaluator.get_or_create_rating(newer)
        rn.mu, rn.matches_played = 27.5, 4
        rn.update_conservative()
        self.assertFalse(self.league.seed_from_predecessor(newer, rn))
        self.assertAlmostEqual(rn.mu, 27.5, places=3)

        # Nor does a later checkpoint seed from one that comes after it.
        latest = self.league._normalize_path("checkpoints/checkpoint_iter_50.pt")
        rl = self.evaluator.get_or_create_rating(latest)
        self.assertFalse(self.league.seed_from_predecessor(latest, rl))

    def test_gauntlet_opponents_are_peers_not_anchors(self):
        """
        Every anchor is saturated against the current population.

        Measured over five checkpoints spanning 63,000 iterations, four series each: the
        heuristic and the BC baseline lost every series, Necto won every series, Nexto
        lost every series. A trial against any of them returns a foregone conclusion, so
        opponents must be chosen from rated peers.
        """
        self._seat_a_king()
        for i, mu in enumerate((26.4, 24.0, 30.0, 21.0)):
            # Real files: peer selection skips paths that are no longer on disk.
            disk = os.path.join(self.test_dir, f"checkpoint_iter_{500+i}.pt")
            with open(disk, "w") as f:
                f.write("stub")
            path = self.league._normalize_path(disk)
            r = self.evaluator.get_or_create_rating(path)
            r.mu, r.sigma, r.matches_played = mu, 1.2, 40
            r.update_conservative()
        for name in ("heuristic", "checkpoints/necto-model.pt"):
            a = self.evaluator.get_or_create_rating(name, is_anchor=True)
            a.is_anchor = True
            a.mu, a.sigma, a.matches_played = 26.5, 0.5, 200
            a.update_conservative()

        subject = self.evaluator.get_or_create_rating(
            self.league._normalize_path(self.dummy_ckpt_path)
        )
        subject.mu = 26.5
        peers = self.league.nearest_rated_peers(subject, exclude=subject.path, limit=3)
        self.assertTrue(peers, "expected peers to be available")
        for path in peers:
            rec = self.evaluator.ratings[self.league._normalize_path(path)]
            self.assertFalse(rec.is_anchor, f"{rec.name} is an anchor and must not spar")
            self.assertNotEqual(self.league._normalize_path(path), "heuristic")
        # Ordered by closeness, so the first is at least as near as the last.
        gaps = [abs(self.evaluator.ratings[p].mu - subject.mu) for p in peers]
        self.assertEqual(gaps, sorted(gaps))

    def test_nexto_is_a_benchmark_and_never_a_rated_anchor(self):
        """
        Nexto sits in a closed cycle: it beats Necto every series, loses to every
        checkpoint every series, and Necto beats those same checkpoints every series. No
        scalar rating can hold all three legs, and Nexto is pinned above the field, so
        rated wins against it would inflate the whole population.
        """
        for path in self.league.active_anchors:
            self.assertNotIn("nexto", str(path).lower())
        self.assertTrue(any("nexto" in str(x).lower() for x in self.league.benchmark_opponents))

    def test_benchmarks_do_not_move_any_rating(self):
        """A benchmark is played and reported; it must never update the leaderboard."""
        self.league.benchmark_opponents = ["heuristic"]
        self.league.benchmark_series = 1
        subject = self.league._normalize_path(self.dummy_ckpt_path)
        rec = self.evaluator.get_or_create_rating(subject)
        rec.mu, rec.sigma, rec.matches_played = 26.0, 1.4, 20
        rec.update_conservative()
        before = (rec.mu, rec.sigma, rec.matches_played, rec.wins, rec.losses)

        results = self.league.run_benchmarks(subject)

        after = (rec.mu, rec.sigma, rec.matches_played, rec.wins, rec.losses)
        self.assertEqual(before, after, "a benchmark changed a rating")
        if results:
            for res in results.values():
                self.assertIn("points_rate", res)
                self.assertGreaterEqual(res["series"], 1)

    def test_a_debut_is_skipped_when_nothing_can_come_of_it(self):
        """
        A checkpoint that cannot enter the gauntlet should not be graded.

        Preemption needs the most settled contender to be rank-eligible, and a fresh
        checkpoint always carries the highest sigma, so the answer is knowable before any
        match is played. Ten saves an hour were each spending two series to be told no,
        leaving records stranded at their debut sigma that never joined the pool either.
        """
        self._seat_a_king()
        self.league.max_active_contenders = 2

        # Room in the queue: a newcomer is worth grading.
        self.assertTrue(self.league.can_admit_a_newcomer())

        # Full, and nothing in it has converged: nobody can be displaced.
        for i, sigma in enumerate((2.4, 2.8)):
            path, rec = self._queued_contender(f"unconverged_{i}", 26.0, sigma, 6, 6)
            self.assertFalse(self.league._is_rank_eligible(rec))
        self.assertEqual(len(self.league.contender_queue), 2)
        self.assertFalse(self.league.can_admit_a_newcomer())

        # One converges, so it can be preempted and a newcomer is worth grading again.
        settled = self.evaluator.ratings[self.league.contender_queue[0]]
        settled.sigma, settled.matches_played = 1.1, 40
        settled.update_conservative()
        self.assertTrue(self.league._is_rank_eligible(settled))
        self.assertTrue(self.league.can_admit_a_newcomer())

    def test_a_skipped_debut_leaves_no_rating_behind(self):
        """grade_checkpoint returns without playing when admission is impossible."""
        self._seat_a_king()
        self.league.max_active_contenders = 1
        _, blocker = self._queued_contender("blocker", 26.0, 2.9, 6, 6)
        self.assertFalse(self.league.can_admit_a_newcomer())

        newcomer = self.league._normalize_path(self.dummy_ckpt_path)
        before = self.evaluator.ratings[newcomer].matches_played
        rec = self.league.grade_checkpoint(self.dummy_ckpt_path, device="cpu")
        self.assertEqual(rec.matches_played, before, "a deferred debut still played matches")

    def test_budget_estimate_matches_the_scheduling_arithmetic(self):
        """
        The panel's numbers must be the ones the scheduler actually produces.

        A grading event admits at most one checkpoint and delivers series_per_step series,
        while ranking one costs target_eval_matches. So the share of saves that get ranked
        is the ratio of those two, independent of the checkpoint interval and the queue
        depth, because both scale arrivals and grading events together.
        """
        from ui.app import gauntlet_budget_estimate

        league = {"target_eval_matches": 30, "max_active_contenders": 3,
                  "eval_series_per_grade": 1}
        logging_cfg = {"checkpoint_interval": 200}
        hp = {"batch_size": 16384}

        at_target = gauntlet_budget_estimate(30, league, logging_cfg, hp, 9500)
        self.assertAlmostEqual(at_target["ranked_pct"], 100.0, places=5)
        half = gauntlet_budget_estimate(15, league, logging_cfg, hp, 9500)
        self.assertAlmostEqual(half["ranked_pct"], 50.0, places=5)

        # Doubling the checkpoint interval leaves the ranked share untouched and only
        # stretches the wall clock. This is why raising the interval cannot fix a queue
        # that is falling behind.
        slower = gauntlet_budget_estimate(15, league, {"checkpoint_interval": 400}, hp, 9500)
        self.assertAlmostEqual(slower["ranked_pct"], half["ranked_pct"], places=5)
        self.assertAlmostEqual(slower["minutes_to_rank"], half["minutes_to_rank"] * 2, places=3)

        # More budget is monotonically faster and monotonically more expensive.
        prev = None
        for step in (4, 8, 12, 16, 20, 24, 28, 32):
            est = gauntlet_budget_estimate(step, league, logging_cfg, hp, 9500)
            if prev:
                self.assertLess(est["minutes_to_rank"], prev["minutes_to_rank"])
                self.assertGreater(est["duty_pct"], prev["duty_pct"])
            prev = est

    def test_a_contender_out_of_budget_always_leaves(self):
        """
        Demotion and graduation must be exhaustive at the budget.

        Graduation used to sit behind an absolute mu bar of 25.5 and a points-rate bar.
        A contender that cleared neither could not graduate, and if it was not confidently
        below the relative demotion floor it could not be demoted either. It drew gauntlet
        trials forever. With a King at mu 24.09 that trap was every rating between 20.09
        and 25.50, and one record reached 91 matches inside it.
        """
        self._seat_a_king()
        king = self.evaluator.ratings[self.league._normalize_path(self.dummy_ckpt_path)]
        king.mu, king.sigma, king.matches_played = 24.09, 1.25, 40
        king.update_conservative()
        self.league.refresh_pool()

        for mu in (20.5, 22.0, 24.0, 25.4, 26.0):
            for wins, losses in ((20, 28), (22, 26), (24, 24)):
                _, rec = self._queued_contender(
                    f"stuck_{mu}_{wins}", mu, 1.2, wins, losses
                )
                self.assertIn(
                    self._gauntlet_exit(rec), ("demote", "graduate"),
                    f"mu={mu} record={wins}W-{losses}L is stuck in the gauntlet",
                )

    def test_graduation_no_longer_needs_an_absolute_rating(self):
        """A contender below the retired 25.5 bar but measured still graduates."""
        self._seat_a_king()
        king = self.evaluator.ratings[self.league._normalize_path(self.dummy_ckpt_path)]
        king.mu, king.sigma, king.matches_played = 24.09, 1.25, 40
        king.update_conservative()
        self.league.refresh_pool()

        _, rec = self._queued_contender("measured_below_bar", 24.5, 1.2, 16, 18)
        self.assertLess(rec.mu, 25.5)
        self.assertEqual(self._gauntlet_exit(rec), "graduate")
        self.assertFalse(hasattr(self.league, "min_contender_mu"))

    def test_points_floor_needs_confidence_not_a_dip(self):
        """
        The floor bites on evidence, the way the skill floor already did.

        A points rate over 12 series carries a standard error near 14 points, so a bare
        comparison against a floor just under even odds evicted on variance.
        """
        self._seat_a_king()
        # 5W-7L is 41.7%, which used to evict against the old 45% floor.
        _, unlucky = self._queued_contender("unlucky", 26.0, 2.2, 5, 7)
        self.assertAlmostEqual(unlucky.points_rate, 41.7, places=1)
        self.assertNotEqual(self._gauntlet_exit(unlucky), "demote")

        # 3W-13L is 18.8%, which is a real signal rather than a dip.
        _, weak = self._queued_contender("weak", 26.0, 2.2, 3, 13)
        self.assertEqual(self._gauntlet_exit(weak), "demote")

    def test_loss_streak_cap_is_not_an_ordinary_event(self):
        """
        Four straight losses happen to most contenders over a full gauntlet.

        Simulated across 48 evenly matched series, a run of four occurs about 80% of the
        time, so as a knockout it carried almost no information. The cap must sit high
        enough that tripping it means something.
        """
        import random

        self.assertGreaterEqual(self.league.max_consecutive_losses, 8)
        random.seed(5)
        trials, tripped_at_four, tripped_at_cap = 4000, 0, 0
        for _ in range(trials):
            streak = best = 0
            for _ in range(48):
                if random.random() < 0.5:
                    streak = 0
                else:
                    streak += 1
                    best = max(best, streak)
            tripped_at_four += best >= 4
            tripped_at_cap += best >= self.league.max_consecutive_losses
        self.assertGreater(tripped_at_four / trials, 0.6)
        self.assertLess(tripped_at_cap / trials, 0.2)

    def test_tiers_snap_to_whole_worker_slices(self):
        """
        Every tier must end on an env-worker boundary.

        A worker owns a contiguous run of environments and a rollout step waits on all of
        them, so a tier ending mid-slice puts two opponent models in one worker and makes
        it set the pace for every environment.
        """
        from utils.league_manager import snap_tiers_to_worker_slices

        # 20% of 128 asks for 25.6 environments, which is what broke alignment.
        for quantum in (4, 8, 16):
            for ratio in (0.05, 0.1, 0.2, 0.33, 0.5):
                n_env = 128
                n_train = int(round(n_env * ratio))
                rest = n_env - n_train
                counts = snap_tiers_to_worker_slices(
                    n_env, n_train, rest // 2, rest // 4, rest - rest // 2 - rest // 4, quantum
                )
                self.assertEqual(sum(counts), n_env, (quantum, ratio, counts))
                for n in counts:
                    self.assertEqual(n % quantum, 0, (quantum, ratio, counts))

    def test_snapping_leaves_indivisible_env_counts_alone(self):
        """A count that does not divide by the quantum has no slices to snap to."""
        from utils.league_manager import snap_tiers_to_worker_slices

        original = (3, 10, 5, 2)
        self.assertEqual(snap_tiers_to_worker_slices(20, *original, 8), original)

    def test_snapping_never_deletes_a_tier(self):
        """
        Fewer slices than tiers means snapping would round some tier to nothing.

        At 4 environments with a quantum of 4 the King rounds up to all four and
        self-play disappears. One worker holding a mixed set is not a straddle any
        rounding can fix, so the requested mix wins.
        """
        from utils.league_manager import snap_tiers_to_worker_slices

        original = (0, 2, 1, 1)
        self.assertEqual(snap_tiers_to_worker_slices(4, *original, 4), original)
        self.assertEqual(snap_tiers_to_worker_slices(16, 0, 8, 4, 4, 8), (0, 8, 4, 4))

    def test_a_fixed_opponent_cannot_straddle_a_worker(self):
        """
        The failure this was written for: necto on the fixed list at 0.20.

        That share is 25.6 of 128 environments. Rounding it to 26 pushed five of sixteen
        workers off their boundary, and every environment paid the unbatched forward pass.
        """
        self._seat_a_king()
        self.league.training_opponents = [self.league._normalize_path("heuristic")]
        self.league.training_opponent_ratio = 0.20
        self.league.pool_group_size = 8

        for _ in range(15):
            dist = self.league.get_stratified_distribution(128)
            self.assertEqual(len(dist), 128)
            for w in range(16):
                slice_ = set(dist[w * 8:(w + 1) * 8])
                self.assertEqual(len(slice_), 1, f"worker {w} straddles models: {slice_}")

    def test_every_slider_position_is_one_the_scheduler_can_honour(self):
        """
        The opponent-share control steps in whole env-worker blocks.

        It used to be a percentage with a 0.01 step, which made it trivial to ask for a
        share that cannot land on a worker boundary: 0.20 of 128 environments is 25.6.
        In block units every reachable position round-trips through the stored ratio to
        exactly the count requested, with no worker straddling two models.
        """
        self._seat_a_king()
        num_envs, workers = 128, 16
        block = num_envs // workers
        self.league.pool_group_size = block
        opponent = self.league._normalize_path("heuristic")

        for count in range(0, num_envs + 1, block):
            self.league.training_opponents = [opponent] if count else []
            self.league.training_opponent_ratio = count / float(num_envs)
            dist = self.league.get_stratified_distribution(num_envs)
            self.assertEqual(len(dist), num_envs)
            self.assertEqual(
                sum(1 for x in dist if x == opponent and count), count,
                f"asked for {count} environments, scheduler assigned something else",
            )
            for w in range(workers):
                slice_ = set(dist[w * block:(w + 1) * block])
                self.assertEqual(len(slice_), 1, f"count {count}, worker {w}: {slice_}")

    def test_status_panel_reports_the_league_not_the_legacy_keys(self):
        """
        The panel must name the tiers actually running.

        baseline_opponent_type and baseline_opponent_ratio only take effect when the
        league is disabled. Quoting them while it runs advertised Nexto at 5% through a
        whole session in which no environment faced Nexto.
        """
        from ui.app import describe_opponent_mix, opponent_mix_shares

        league = {
            "enabled": True, "self_play_ratio": 0.5, "king_ratio": 0.25,
            "pool_ratio": 0.25, "training_opponents": [], "training_opponent_ratio": 0.0,
            "pool_group_size": 8,
        }
        line = describe_opponent_mix(
            league, {"king_of_the_hill": "checkpoint_iter_10", "elite_pool_size": 10},
            fallback="checkpoints/nexto-model.pt", num_envs=128,
        )
        self.assertIn("Self-play", line)
        self.assertIn("checkpoint_iter_10", line)
        self.assertNotIn("nexto", line)

        # The reported split is the one the scheduler runs, not the one requested.
        with_fixed = dict(league, training_opponents=["checkpoints/necto-model.pt"],
                          training_opponent_ratio=0.20)
        self.assertAlmostEqual(opponent_mix_shares(with_fixed)["fixed"], 20.0)
        self.assertAlmostEqual(opponent_mix_shares(with_fixed, 128)["fixed"], 18.75)

        off = dict(league, enabled=False)
        self.assertIn("nexto", describe_opponent_mix(
            off, {}, fallback="checkpoints/nexto-model.pt", num_envs=128))

    def test_how_it_works_reads_live_config(self):
        """
        The explainer quotes thresholds, so it must read them from config rather than
        hard-coding them. Copy that drifts from the gates it describes is worse than no
        copy at all.
        """
        from ui.app import build_how_it_works_html
        import yaml

        html = build_how_it_works_html()
        cfg = yaml.safe_load(open("config/default_config.yaml", encoding="utf-8"))
        league = cfg.get("league", {})

        self.assertIn(str(cfg["logging"]["checkpoint_interval"]), html)
        self.assertIn(str(league.get("series_length", 9)), html)
        self.assertIn(str(league.get("target_eval_matches", 30)), html)
        self.assertIn(str(league.get("eligibility_sigma", 1.5)), html)
        self.assertIn(str(league.get("rating_lock_matches", 64)), html)
        # The King's share is what the fixed training-opponent list leaves behind, so the
        # copy must not quote the raw king_ratio when that list is populated.
        if league.get("training_opponents"):
            self.assertNotIn("becomes 25% of training opponents", html)
        # Reachable without a pointer, and announced as interactive.
        self.assertIn('tabindex="0"', html)
        self.assertIn('role="tooltip"', html)

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
        candidate_pool = [m for m in dict.fromkeys(self.league.elite_pool) if m not in explicit]
        self.assertGreater(len(candidate_pool), 0)

        seen = []
        for _ in range(len(candidate_pool) * 2):
            dist = self.league.get_stratified_distribution(4)
            pool_slot = dist[-1]  # The pool assignment
            seen.append(pool_slot)

        expected = [candidate_pool[i % len(candidate_pool)] for i in range(len(candidate_pool) * 2)]
        self.assertEqual(seen, expected)



    def test_no_skill_floor_while_the_king_is_provisional(self):
        """
        On a fresh leaderboard nothing is measured well enough for "too weak" to mean
        anything. A fixed floor of 25.5 evicted contenders rated ABOVE a King at 23.26,
        because recalibrating the anchors moved the whole population under the constant.
        """
        ckpt = self.league._normalize_path(self.dummy_ckpt_path)
        king = self.evaluator.get_or_create_rating(ckpt)
        king.mu, king.sigma, king.matches_played = 23.26, 2.45, 17
        king.update_conservative()
        self.league.king_of_the_hill = ckpt

        self.assertFalse(self.league._is_rank_eligible(king))
        self.assertIsNone(self.league._contender_mu_floor())

    def test_skill_floor_is_relative_to_an_established_king(self):
        """Once the King's rating is established the floor tracks it, not a constant."""
        ckpt = self.league._normalize_path(self.dummy_ckpt_path)
        king = self.evaluator.get_or_create_rating(ckpt)
        king.mu, king.sigma, king.matches_played = 30.0, 1.0, 60
        king.update_conservative()
        self.league.king_of_the_hill = ckpt

        floor = self.league._contender_mu_floor()
        self.assertIsNotNone(floor)
        self.assertAlmostEqual(floor, 30.0 - self.league.contender_mu_margin, places=3)

        # Confidently below the floor -> demoted. Straddling it on a wide sigma -> kept,
        # because at that uncertainty the comparison is noise rather than evidence.
        self.assertLess(18.0 + 1.5, floor)
        self.assertGreater(20.0 + 7.0, floor)

    def test_points_floor_waits_for_a_real_sample(self):
        """A points rate over the 6-match grace period is 2-3 results wide."""
        self.assertGreater(
            self.league.min_matches_for_points_floor,
            self.league.grace_period_matches,
            "points floor must not bite at the grace period, where it decides on noise"
        )

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


class TestRatingLock(unittest.TestCase):
    """
    Past the series cap a rating stops moving.

    The motivation is not sample size. Against a fixed opponent more series is always
    better, but an established model faces a stream of fresh challengers, each entering
    at sigma 8.33. Simulated over that schedule with every player's true skill identical,
    the established rating random-walks (true spread 0.36 -> 0.70 mu between 30 and 300
    series) while its reported sigma flatlines near 0.94 and stops reflecting it.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.evaluator = TrueSkillEvaluator(
            leaderboard_path=os.path.join(self.tmp, "lb.json")
        )
        self.evaluator.rating_lock_matches = 64
        self.evaluator.eligibility_sigma = 1.5

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rec(self, matches, sigma, mu=30.0):
        rec = self.evaluator.get_or_create_rating(f"checkpoints/checkpoint_iter_{matches}.pt")
        rec.matches_played = matches
        rec.sigma = sigma
        rec.mu = mu
        rec.update_conservative()
        return rec

    def test_converged_rating_locks_at_the_cap(self):
        rec = self._rec(64, 0.95)
        self.assertTrue(self.evaluator.maybe_lock_rating(rec))
        self.assertTrue(rec.rating_locked)
        self.assertEqual(rec.locked_at_matches, 64)
        self.assertTrue(self.evaluator.is_rating_frozen(rec))

    def test_below_the_cap_stays_live(self):
        rec = self._rec(63, 0.95)
        self.assertFalse(self.evaluator.maybe_lock_rating(rec))
        self.assertFalse(self.evaluator.is_rating_frozen(rec))

    def test_unconverged_rating_is_never_locked(self):
        """
        Reaching the cap without converging means the model was not measured. Freezing it
        there would make the noise permanent instead of the estimate.
        """
        rec = self._rec(200, 2.4)
        self.assertFalse(self.evaluator.maybe_lock_rating(rec))
        self.assertFalse(rec.rating_locked)

    def test_locking_is_idempotent_and_survives_a_cap_change(self):
        """
        The lock is recorded on the rating, not recomputed from config, so raising the
        cap later cannot silently re-open a model that was already frozen.
        """
        rec = self._rec(64, 0.95)
        self.evaluator.maybe_lock_rating(rec)
        self.assertFalse(self.evaluator.maybe_lock_rating(rec), "locked twice")

        self.evaluator.rating_lock_matches = 500
        self.assertTrue(self.evaluator.is_rating_frozen(rec))
        self.assertEqual(rec.locked_at_matches, 64)

    def test_lock_survives_a_save_and_load_round_trip(self):
        rec = self._rec(64, 0.95)
        self.evaluator.maybe_lock_rating(rec)
        self.evaluator.save_leaderboard()

        reloaded = TrueSkillEvaluator(leaderboard_path=self.evaluator.leaderboard_path)
        again = reloaded.ratings[rec.path]
        self.assertTrue(again.rating_locked)
        self.assertEqual(again.locked_at_matches, 64)

    def test_a_leaderboard_written_before_the_lock_existed_still_loads(self):
        """Old JSON has neither field; ModelRating must fall back rather than raise."""
        path = os.path.join(self.tmp, "legacy.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ratings": {"checkpoints/old.pt": {
                "name": "old", "path": "checkpoints/old.pt",
                "mu": 28.0, "sigma": 1.1, "matches_played": 90,
            }}, "history": []}, f)

        loaded = TrueSkillEvaluator(leaderboard_path=path)
        rec = loaded.ratings["checkpoints/old.pt"]
        self.assertFalse(rec.rating_locked)
        self.assertFalse(loaded.is_rating_frozen(rec))

    def test_frozen_side_holds_its_rating_while_the_live_side_moves(self):
        """
        The whole point: a locked model keeps playing, and the series still measures the
        opponent. Only the locked side is exempt from the update.
        """
        locked = self._rec(64, 0.95, mu=32.0)
        self.evaluator.maybe_lock_rating(locked)
        live = self.evaluator.get_or_create_rating("checkpoints/checkpoint_iter_999.pt")
        before_locked, before_live = locked.mu, live.mu

        r_locked = locked.to_trueskill_rating()
        r_live = live.to_trueskill_rating()
        new_live, new_locked = trueskill.rate_1vs1(r_live, r_locked)
        if not self.evaluator.is_rating_frozen(locked):
            locked.from_trueskill_rating(new_locked)
        if not self.evaluator.is_rating_frozen(live):
            live.from_trueskill_rating(new_live)

        self.assertEqual(locked.mu, before_locked, "locked rating moved")
        self.assertGreater(live.mu, before_live, "live rating did not absorb the win")

    def test_the_gauntlet_always_finishes_before_the_lock_can_bite(self):
        """
        A contender frozen mid-trial could never graduate. The cap must sit above the
        Gauntlet's own ceiling for that to be structurally impossible.
        """
        cfg = yaml.safe_load(open("config/default_config.yaml", encoding="utf-8"))
        league = cfg.get("league", {})
        self.assertGreater(
            league.get("rating_lock_matches", 64),
            league.get("max_contender_matches", 48),
        )


if __name__ == "__main__":
    unittest.main()
