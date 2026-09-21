"""
Reward v5 (config/reward_versions/v5.json) against its specification, docs/reward_v5_gamma_spec.md.

v5 has no reward code of its own: its terms are v3's, and the only difference is gamma, 0.995 ->
0.9977 (a 20 s half-life at 15 actions/sec instead of 9.2 s). So the things worth testing are the
ones that could silently go wrong:

  - v5 pays v3's terms, term for term, at the same gamma, and differs only when gamma does
  - the shaping gamma the manager uses equals the gamma PPO optimises, however the manager is built
  - the horizon really is 20 s, i.e. the number in the snapshot is the one the spec's formula gives
  - closeness stays retired and annealing stays off
  - v3 -> v5 is a version change, so the trainer resets the return normaliser and warms the critic
"""
import math
import unittest
from types import SimpleNamespace

import numpy as np

from env.reward_registry import CODE_FILES, apply_reward_version, load_snapshot, make_reward_manager
from env.reward_version import code_sha, settings_sha
from env.rewards_v3 import RewardManagerV3

ACTIONS_PER_SEC = 15.0          # tick_skip 8 at 120 ticks
V5_GAMMA = 0.9977


def car(cid, team, pos, boost=33.0, touches=0):
    return SimpleNamespace(id=cid, team=team, pos=np.array(pos, dtype=np.float32), boost=boost,
                           ball_touches=touches)


def arena(ball_pos, ball_vel=(0.0, 0.0, 0.0), cars=()):
    return SimpleNamespace(ball=SimpleNamespace(pos=np.array(ball_pos, dtype=np.float32),
                                                vel=np.array(ball_vel, dtype=np.float32)),
                           cars=list(cars))


def half_life_seconds(gamma):
    """The spec's formula, inverted: gamma = exp(log(0.5) / (T * A))."""
    return math.log(0.5) / (math.log(gamma) * ACTIONS_PER_SEC)


class TestTheHorizon(unittest.TestCase):
    def test_v5_gamma_is_a_twenty_second_half_life(self):
        """0.9977 is Seer's published value for T = 20 s; carried to more digits it is 0.9976926,
        so the horizon it actually buys is 20.07 s. The rounded constant is the one in the spec."""
        self.assertAlmostEqual(half_life_seconds(load_snapshot("v5")["settings"]["gamma"]), 20.0, delta=0.1)

    def test_v3_gamma_was_a_nine_second_half_life(self):
        """The thing v5 exists to change: v3 sat at Seer's *initial* horizon, not its final one."""
        self.assertAlmostEqual(half_life_seconds(load_snapshot("v3")["settings"]["gamma"]), 9.2, delta=0.1)


class TestSettings(unittest.TestCase):
    def test_v5_is_v3s_terms_with_closeness_retired_and_a_longer_horizon(self):
        v3, v5 = load_snapshot("v3")["settings"], load_snapshot("v5")["settings"]
        self.assertEqual(v5["scenarios"], v3["scenarios"])
        self.assertEqual(v5["rewards"], {**v3["rewards"], "closeness_weight": 0.0})
        self.assertFalse(v5["reward_annealing"]["enabled"])
        self.assertEqual(v5["reward_annealing"]["targets"], {})
        self.assertEqual(v5["gamma"], V5_GAMMA)
        self.assertNotEqual(v5["gamma"], v3["gamma"])

    def test_v5_shares_v3s_code_but_not_its_settings(self):
        """No new reward file: the version is a settings change, and the identity must say so."""
        v3, v5 = load_snapshot("v3")["identity"], load_snapshot("v5")["identity"]
        self.assertEqual(CODE_FILES["v5"], CODE_FILES["v3"])
        self.assertEqual(v5["code_sha"], v3["code_sha"])
        self.assertNotEqual(v5["settings_sha"], v3["settings_sha"])

    def test_v5_starts_from_the_v3_king(self):
        snap = load_snapshot("v5")
        self.assertEqual(snap["start_checkpoint"], "checkpoints/baselines/v3_iter198000.pt")
        self.assertEqual(snap["baseline_eval"], "evals/baselines/v3_iter198000.json")

    def test_a_config_naming_v5_resolves_to_the_frozen_settings(self):
        cfg = apply_reward_version({"reward_version": "v5", "hyperparameters": {"gamma": V5_GAMMA}})
        self.assertEqual(settings_sha(cfg), load_snapshot("v5")["identity"]["settings_sha"])
        self.assertEqual(code_sha(CODE_FILES["v5"]), load_snapshot("v5")["identity"]["code_sha"])

    def test_v5_refuses_a_config_carrying_v3s_gamma(self):
        """The guard that forced v5 to be a version at all, rather than a config edit."""
        with self.assertRaises(ValueError) as e:
            apply_reward_version({"reward_version": "v5", "hyperparameters": {"gamma": 0.995}})
        self.assertIn("a new gamma is a new reward version", str(e.exception))


class TestShapingGammaMatchesPPO(unittest.TestCase):
    def test_a_manager_built_without_weights_uses_the_versions_frozen_gamma(self):
        """
        The silent-mismatch case. v5 has no reward module of its own, so the code default behind it
        is v3's 0.995; a caller that passes no weights must still get 0.9977.
        """
        self.assertAlmostEqual(make_reward_manager("v5").reward.w["gamma"], V5_GAMMA, places=9)
        self.assertAlmostEqual(make_reward_manager("v3").reward.w["gamma"], 0.995, places=9)

    def test_explicit_weights_still_win(self):
        self.assertAlmostEqual(make_reward_manager("v5", {"gamma": 0.99}).reward.w["gamma"], 0.99, places=9)

    def test_the_trainer_hands_the_manager_the_gamma_it_optimises(self):
        """ppo.py sets rew_cfg's gamma from hyperparameters.gamma; that is what keeps shaping valid."""
        cfg = apply_reward_version({"reward_version": "v5", "hyperparameters": {"gamma": V5_GAMMA}})
        rew_cfg = dict(cfg["rewards"])
        rew_cfg.setdefault("gamma", float(cfg["hyperparameters"]["gamma"]))
        manager = make_reward_manager("v5", rew_cfg)
        self.assertEqual(manager.reward.w["gamma"], cfg["hyperparameters"]["gamma"])


class TestPaysV3sTerms(unittest.TestCase):
    def _trajectory(self, m_a, m_b):
        """The same random trajectory through both managers; returns their (total, breakdown) pairs."""
        rng = np.random.default_rng(5)
        blue, orange = car(0, 0, (0, -2000, 17)), car(1, 1, (0, 2000, 17))
        a = arena((0, 0, 93), cars=[blue, orange])
        m_a.reset(a)
        m_b.reset(a)
        out = []
        for step in range(60):
            a.ball.pos = rng.uniform(-4000, 4000, 3).astype(np.float32)
            a.ball.vel = rng.uniform(-2000, 2000, 3).astype(np.float32)
            for c in (blue, orange):
                c.pos = rng.uniform(-4000, 4000, 3).astype(np.float32)
                c.boost = float(rng.uniform(0, 100))
                c.ball_touches += int(rng.random() < 0.2)
            goal = step % 25 == 24
            for c in (blue, orange):
                out.append((m_a.get_reward(c, a, None, goal, 0 if goal else None),
                            m_b.get_reward(c, a, None, goal, 0 if goal else None)))
        return out

    def test_at_the_same_gamma_v5_is_exactly_v3(self):
        v5 = make_reward_manager("v5", {"closeness_weight": 1.0, "gamma": 0.995})
        v3 = RewardManagerV3({"closeness_weight": 1.0})
        for (r5, b5), (r3, b3) in self._trajectory(v5, v3):
            self.assertAlmostEqual(r5, r3, places=9)
            for k, v in b3.items():
                self.assertAlmostEqual(v, b5[k], places=9, msg=k)

    def test_the_longer_horizon_actually_changes_the_payments(self):
        """Guards against a snapshot whose gamma never reaches the terms."""
        v5 = make_reward_manager("v5", {"closeness_weight": 1.0})
        v3 = RewardManagerV3({"closeness_weight": 1.0})
        self.assertTrue(any(abs(r5 - r3) > 1e-9 for (r5, _), (r3, _) in self._trajectory(v5, v3)))

    def test_closeness_pays_nothing_at_the_frozen_weights(self):
        m = make_reward_manager("v5", load_snapshot("v5")["settings"]["rewards"])
        blue, orange = car(0, 0, (0, -3000, 17)), car(1, 1, (0, 3000, 17))
        a = arena((0, 0, 93), cars=[blue, orange])
        m.reset(a)
        blue.pos = np.array([0, -100, 17], np.float32)
        self.assertEqual(m.get_reward(blue, a, None, False, None)[1]["closeness"], 0.0)


class TestVersionChange(unittest.TestCase):
    def test_v3_checkpoint_starts_a_new_v5_run(self):
        """
        Raising gamma roughly doubles the return the critic predicts, so the inherited value head is
        wrong by a large factor. The version change is what resets the normaliser and warms it up.
        """
        import torch
        from agent.ppo import PPOTrainer, RunningMeanStd
        s = SimpleNamespace(
            reward_version="v5", reward_version_pinned=True, global_step=9_000, reward_run_start_step=0,
            ret_rms=RunningMeanStd(), _needs_value_norm_migration=True, critic_warmup_iterations=50,
            _critic_warmup_remaining=0, agent=torch.nn.Linear(2, 2), lr=1.5e-4,
            _reward_anneal_start_steps={"closeness_weight": 10}, _last_pushed_reward_weights={"x": 1.0})
        s.ret_rms.mean, s.ret_rms.var = 3.0, 9.0
        s.optimizer = old_opt = torch.optim.AdamW(s.agent.parameters())
        PPOTrainer._check_reward_version.__get__(s)(
            {"reward_identity": {"version": "v3"}, "reward_run_start_step": 1_234, "global_step": 9_000})
        self.assertEqual(s.reward_run_start_step, 9_000)
        self.assertEqual(s._critic_warmup_remaining, 50)
        self.assertEqual(float(s.ret_rms.mean), 0.0)
        self.assertIsNot(s.optimizer, old_opt)
        self.assertEqual(s._reward_anneal_start_steps, {})


if __name__ == "__main__":
    unittest.main()
