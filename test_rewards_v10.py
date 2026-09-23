"""
Reward v10 (docs/reward_v10_control_spec.md): v7 at the replay pool's natural tag frequencies.

v10 is a control, so what matters is what it does not change. These pin that its reward is v5's,
its code is v7's, and the only setting that differs from v7 is the replay tag weighting, which is
the pool's own frequencies (as in v8 and v9).
"""
import unittest

from env.reward_registry import CODE_FILES, apply_reward_version, load_snapshot, make_reward_manager, reward_defaults
from env.reward_version import code_sha
from env.rewards_v3 import RewardManagerV3


def diff_paths(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff_paths(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        return out
    return [] if a == b else [path]


class TestRewardV10(unittest.TestCase):
    def setUp(self):
        self.v5, self.v7, self.v9, self.v10 = (load_snapshot(v) for v in ("v5", "v7", "v9", "v10"))

    def test_only_the_tag_weights_differ_from_v7(self):
        self.assertEqual(diff_paths(self.v7["settings"], self.v10["settings"]),
                         [f"replay_sampling.tag_weights.{t}" for t in sorted(self.v7["settings"]["replay_sampling"]["tag_weights"])])

    def test_tag_weights_are_the_pools_natural_frequencies(self):
        pool = self.v10["replay_pool"]
        weights = self.v10["settings"]["replay_sampling"]["tag_weights"]
        self.assertEqual(set(weights), set(pool["tag_counts"]))
        for tag, n in pool["tag_counts"].items():
            self.assertAlmostEqual(weights[tag], n / pool["kept_frames"], places=5, msg=tag)
        self.assertEqual(weights, self.v9["settings"]["replay_sampling"]["tag_weights"])

    def test_reward_is_v5s(self):
        self.assertEqual(self.v10["settings"]["rewards"], self.v5["settings"]["rewards"])
        self.assertEqual(self.v10["settings"]["gamma"], self.v5["settings"]["gamma"])
        self.assertEqual(self.v10["settings"]["reward_annealing"], self.v5["settings"]["reward_annealing"])

    def test_code_is_v7s(self):
        self.assertEqual(CODE_FILES["v10"], CODE_FILES["v7"])
        self.assertEqual(self.v10["identity"]["code_sha"], self.v7["identity"]["code_sha"])
        self.assertEqual(code_sha(CODE_FILES["v10"]), self.v10["identity"]["code_sha"])
        self.assertNotEqual(self.v10["identity"]["settings_sha"], self.v7["identity"]["settings_sha"])

    def test_same_frozen_replay_pool_as_v7_to_v9(self):
        for v in (self.v7, self.v9):
            self.assertEqual(self.v10["replay_pool"], v["replay_pool"])

    def test_registry_builds_v5s_manager_at_v5s_gamma(self):
        m = make_reward_manager("v10")
        self.assertIsInstance(m, RewardManagerV3)
        self.assertEqual(reward_defaults("v10"), reward_defaults("v5"))
        cfg = apply_reward_version({"reward_version": "v10", "hyperparameters": {"gamma": 0.9977}})
        self.assertEqual(cfg["replay_sampling"], self.v10["settings"]["replay_sampling"])

    def test_starts_from_the_v5_king(self):
        self.assertEqual(self.v10["start_checkpoint"], "checkpoints/baselines/v5_iter222000.pt")


if __name__ == "__main__":
    unittest.main()
