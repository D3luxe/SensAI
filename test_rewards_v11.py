"""
v11 (docs/reward_v11_pretanh_spec.md): v10 plus a penalty in PPO's loss on the throttle and steer
pre-tanh magnitude (agent/pre_tanh_penalty.py).

Pins that v11 differs from v10 only by that section, that no earlier version picks it up, and that
the penalty is zero inside the threshold, reaches both passes of the mirrored forward, and moves
only the actor.
"""
import unittest

import torch

from agent.pre_tanh_penalty import PreTanhPenalty
from env.reward_registry import (CODE_FILES, apply_reward_version, known_versions, load_snapshot,
                                 make_reward_manager, reward_defaults)
from env.rewards_v3 import RewardManagerV3

START = "checkpoints/baselines/v5_iter222000.pt"


def resolved(v):
    return apply_reward_version({"reward_version": v, "hyperparameters": {"gamma": load_snapshot(v)["settings"]["gamma"]}})


class TestV11Snapshot(unittest.TestCase):
    def test_only_action_regularization_differs_from_v10(self):
        v10, v11 = load_snapshot("v10")["settings"], load_snapshot("v11")["settings"]
        self.assertEqual({k: v for k, v in v11.items() if k != "action_regularization"}, v10)
        self.assertEqual(v11["action_regularization"],
                         {"pre_tanh_weight": 0.001, "pre_tanh_threshold": 2.0, "pre_tanh_channels": [0, 1]})
        self.assertEqual(load_snapshot("v11")["replay_pool"], load_snapshot("v10")["replay_pool"])

    def test_code_is_v10s_plus_the_penalty(self):
        self.assertEqual(CODE_FILES["v11"], CODE_FILES["v10"] + ("agent/pre_tanh_penalty.py",))

    def test_no_other_version_gets_the_penalty(self):
        for v in known_versions():
            cfg = resolved(v)
            if v == "v11":
                self.assertIsNotNone(PreTanhPenalty.from_config(cfg.get("action_regularization")))
            else:
                self.assertNotIn("action_regularization", cfg, v)
                self.assertIsNone(PreTanhPenalty.from_config(cfg.get("action_regularization")), v)

    def test_reward_is_v5s(self):
        self.assertIsInstance(make_reward_manager("v11"), RewardManagerV3)
        self.assertEqual(reward_defaults("v11"), reward_defaults("v5"))
        self.assertEqual(load_snapshot("v11")["start_checkpoint"], START)


class TestPreTanhPenalty(unittest.TestCase):
    def setUp(self):
        self.p = PreTanhPenalty(weight=0.5, threshold=2.0, channels=[0, 1])

    def test_zero_inside_the_threshold(self):
        pre = torch.tensor([[1.9, -1.99, 50.0, -50.0, 9.0]])  # channels 2-4 are not charged
        self.assertEqual(float(self.p.loss([pre])), 0.0)

    def test_squared_excess_over_both_signs(self):
        pre = torch.tensor([[3.0, -4.0, 0.0, 0.0, 0.0], [0.0, 2.5, 0.0, 0.0, 0.0]])
        # excesses on channels 0,1: (1, 2), (0, 0.5) -> squares 1, 4, 0, 0.25 -> mean 1.3125
        self.assertAlmostEqual(float(self.p.loss([pre])), 0.5 * 1.3125, places=6)

    def test_every_captured_pass_counts(self):
        a = torch.tensor([[3.0, 0.0, 0, 0, 0]])
        b = torch.tensor([[0.0, 0.0, 0, 0, 0]])
        self.assertAlmostEqual(float(self.p.loss([a, b])), 0.5 * (1.0 / 4), places=6)

    def test_disabled_without_a_positive_weight(self):
        self.assertIsNone(PreTanhPenalty.from_config(None))
        self.assertIsNone(PreTanhPenalty.from_config({"pre_tanh_weight": 0.0}))

    def test_capture_sees_both_passes_and_moves_only_the_actor(self):
        from agent.checkpoint import load_policy
        model, _ = load_policy(START)
        obs_dim = model.obs_mirror_mask.shape[-1]
        obs = torch.randn(64, obs_dim)
        with self.p.capture(model.actor_mean) as outs:
            model.get_action_and_value(obs)
        self.assertEqual(len(outs), 2, "raw and mirrored passes")
        model.zero_grad()
        self.p.loss(outs).backward()
        self.assertIsNotNone(model.actor_mean.weight.grad)
        self.assertGreater(float(model.actor_mean.weight.grad.abs().sum()), 0.0)
        for name, param in model.named_parameters():
            if name.startswith("critic"):
                self.assertTrue(param.grad is None or float(param.grad.abs().sum()) == 0.0, name)

    def test_capture_releases_its_hook(self):
        from agent.checkpoint import load_policy
        model, _ = load_policy(START)
        with self.p.capture(model.actor_mean) as outs:
            pass
        model.get_action_and_value(torch.randn(4, model.obs_mirror_mask.shape[-1]))
        self.assertEqual(outs, [])


class TestV11TrainerPath(unittest.TestCase):
    def test_two_iterations_apply_and_log_the_penalty(self):
        """A mini run with critic warmup off goes through the penalty branch and logs its telemetry."""
        import json
        import os
        import tempfile
        from unittest import mock

        import yaml

        from agent.ppo import PPOTrainer
        with tempfile.TemporaryDirectory() as tmpdir:
            with open("config/default_config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg["reward_version"] = "v11"
            cfg["logging"].update(save_dir=tmpdir, log_dir=tmpdir, tensorboard=False)
            cfg.setdefault("league", {})["league_state_path"] = os.path.join(tmpdir, "league_state.json")
            cfg["hyperparameters"]["critic_warmup_iterations"] = 0
            path = os.path.join(tmpdir, "cfg.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f)
            with mock.patch("env.replay_sampling_v7.check_replay_pool", return_value=None):
                trainer = PPOTrainer(config_path=path)
            self.assertIsNotNone(trainer.pre_tanh_penalty)
            trainer.train(max_iterations=2)
            with open(os.path.join(tmpdir, "history.jsonl"), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            tel = rows[-1]["telemetry"]
            for key in ("pre_tanh_penalty", "pre_tanh_over_pct", "pre_tanh_pre_abs_median_0", "pre_tanh_pre_abs_median_1"):
                self.assertIn(key, tel)


if __name__ == "__main__":
    unittest.main()
