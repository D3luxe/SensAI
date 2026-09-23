"""
Reward versions are frozen (env/reward_registry.py, config/reward_versions/<v>.json).

These fail on purpose if a snapshotted version's reward code or settings change. That is the rule,
not an inconvenience: a reward change is a new version and a new run, compared on the eval suite. If
one of these fails, the fix is to put the change in the next version, not to update the hash here.

  v2  env/rewards.py, git tag reward-v2
  v3  env/rewards_v3.py + env/scenarios_v3.py
  v4  env/rewards_v4.py (+ the v3 files it imports)
  v5  the v3 files, at gamma 0.9977 (only settings_sha differs from v3)
  v6  env/rewards_v6.py (+ the v3 files it imports)
  v7  v6's files + env/replay_sampling_v7.py; its replay pool is frozen separately (replay_pool)
  v8  env/rewards_v8.py (+ the v3 files it imports) + env/replay_sampling_v7.py
  v9  env/rewards_v9.py (+ the v3 files it imports) + env/replay_sampling_v7.py
  v10 v7's files, unchanged (same code_sha as v7); only the replay tag weights differ
"""
import unittest

from env.reward_registry import CODE_FILES, apply_reward_version, known_versions, load_snapshot
from env.reward_version import code_sha, reward_identity, reward_settings, settings_sha


def resolved(version):
    """A config naming `version`, resolved through the registry as the trainer resolves it."""
    return apply_reward_version({"reward_version": version,
                                 "hyperparameters": {"gamma": load_snapshot(version)["settings"]["gamma"]}})


class TestRewardVersionsFrozen(unittest.TestCase):
    def test_every_version_has_a_snapshot(self):
        for v in known_versions():
            self.assertEqual(load_snapshot(v)["version"], v)

    def test_reward_code_is_unchanged(self):
        for v in known_versions():
            self.assertEqual(code_sha(CODE_FILES[v]), load_snapshot(v)["identity"]["code_sha"],
                             f"reward {v}'s code changed: it is frozen, so this belongs in a new reward version")

    def test_settings_are_unchanged(self):
        for v in known_versions():
            snap = load_snapshot(v)
            self.assertEqual(reward_settings(resolved(v)), snap["settings"], v)
            self.assertEqual(settings_sha(resolved(v)), snap["identity"]["settings_sha"], v)

    def test_identity_matches_the_snapshot(self):
        for v in known_versions():
            self.assertEqual(reward_identity(resolved(v)), load_snapshot(v)["identity"], v)

    def test_v2_code_is_what_trained_to_iteration_173600(self):
        self.assertEqual(code_sha(), "7c5d7a2b15e057d7")
        self.assertEqual(load_snapshot("v2")["identity"]["settings_sha"], "20d320a8ec343cca")

    def test_v4_is_v3_plus_the_race_term_with_closeness_retired(self):
        v3, v4 = load_snapshot("v3")["settings"], load_snapshot("v4")["settings"]
        self.assertEqual(v4["scenarios"], v3["scenarios"])
        self.assertEqual(v4["gamma"], v3["gamma"])
        self.assertFalse(v4["reward_annealing"]["enabled"])
        expected = {**v3["rewards"], "closeness_weight": 0.0, "race_weight": 1.0}
        self.assertEqual(v4["rewards"], expected)
        self.assertEqual(load_snapshot("v4")["start_checkpoint"], "checkpoints/baselines/v3_iter198000.pt")

    def test_active_config_resolves_to_its_frozen_settings(self):
        from utils.config import effective_config
        cfg = effective_config()
        self.assertEqual(reward_identity(cfg), load_snapshot(cfg["reward_version"])["identity"])


if __name__ == "__main__":
    unittest.main()
