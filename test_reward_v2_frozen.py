"""
Reward v2 is frozen (env/reward_version.py, config/reward_versions/v2.json, git tag reward-v2).

These fail on purpose if the v2 reward code or its settings change. That is the rule, not an
inconvenience: a reward change is a new version and a new run, compared on the eval suite. If one
of these fails, the fix is to put the change in the next version, not to update the hash here.
"""
import json
import os
import unittest

from env.reward_version import code_sha, reward_settings, settings_sha

ROOT = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(ROOT, "config", "reward_versions", "v2.json")


class TestRewardV2Frozen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SNAPSHOT, encoding="utf-8") as fh:
            cls.snap = json.load(fh)

    def test_reward_code_is_unchanged(self):
        self.assertEqual(code_sha(), self.snap["identity"]["code_sha"],
                         "env/rewards.py changed: v2 is frozen, so this belongs in a new reward version")

    def test_snapshot_is_self_consistent(self):
        from utils.config import effective_config
        cfg = effective_config()
        cfg_from_snapshot = {**cfg, **{k: v for k, v in self.snap["settings"].items() if k != "gamma"}}
        cfg_from_snapshot["hyperparameters"] = {**(cfg.get("hyperparameters") or {}), "gamma": self.snap["settings"]["gamma"]}
        self.assertEqual(settings_sha(cfg_from_snapshot), self.snap["identity"]["settings_sha"])

    def test_live_settings_match_v2(self):
        live = reward_settings()
        for section in ("rewards", "reward_annealing", "scenarios"):
            self.assertEqual(live[section], self.snap["settings"][section],
                             f"config '{section}' differs from frozen v2: that is a new reward version")
        self.assertEqual(live["gamma"], self.snap["settings"]["gamma"])


if __name__ == "__main__":
    unittest.main()
