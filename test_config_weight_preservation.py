"""Reward weights without a UI slider must survive an "Apply".

The UI builds its reward payload from the sliders on the page, then wrote that dict over the
whole "rewards" block in both config files. Any weight with no slider was deleted. This
silently reverted jump_cost_weight and spin_cost_weight to their class defaults three separate
times during tuning, once discarding a weight change that had been applied deliberately, with
nothing logged to say it had happened.

Both writers now merge rather than replace. These tests pin that, and pin that every weight
CombinedReward reads is actually present in the config files.
"""

import io
import json
import os
import shutil
import tempfile
import unittest

import yaml

from utils.process_manager import TrainingProcessManager


class TestLiveConfigMerge(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "live_config.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _mgr(self):
        m = TrainingProcessManager.__new__(TrainingProcessManager)
        m.live_config_file = self.path
        return m

    def test_unknown_reward_keys_survive_an_update(self):
        with io.open(self.path, "w", encoding="utf-8") as f:
            json.dump({"rewards": {"goal_weight": 30.0, "spin_cost_weight": 0.015}}, f)

        # A UI apply that knows nothing about spin_cost.
        self._mgr().update_live_config({"rewards": {"goal_weight": 25.0}})

        out = json.load(io.open(self.path, encoding="utf-8"))["rewards"]
        self.assertAlmostEqual(out["goal_weight"], 25.0, msg="the edited weight must change")
        self.assertIn("spin_cost_weight", out,
                      "a weight with no slider must not be deleted by an unrelated edit")
        self.assertAlmostEqual(out["spin_cost_weight"], 0.015)

    def test_nested_blocks_merge_independently(self):
        with io.open(self.path, "w", encoding="utf-8") as f:
            json.dump({"rewards": {"a": 1.0}, "scenarios": {"kickoff_prob": 0.3}}, f)
        self._mgr().update_live_config({"scenarios": {"wall_prob": 0.1}})
        out = json.load(io.open(self.path, encoding="utf-8"))
        self.assertEqual(out["rewards"], {"a": 1.0}, "an untouched block must be untouched")
        self.assertEqual(out["scenarios"], {"kickoff_prob": 0.3, "wall_prob": 0.1})

    def test_scalars_still_replace(self):
        with io.open(self.path, "w", encoding="utf-8") as f:
            json.dump({"learning_rate": 0.0003}, f)
        self._mgr().update_live_config({"learning_rate": 0.0001})
        out = json.load(io.open(self.path, encoding="utf-8"))
        self.assertAlmostEqual(out["learning_rate"], 0.0001)

    def test_writes_to_a_missing_file(self):
        self._mgr().update_live_config({"rewards": {"goal_weight": 1.0}})
        out = json.load(io.open(self.path, encoding="utf-8"))
        self.assertAlmostEqual(out["rewards"]["goal_weight"], 1.0)


class TestEveryWeightIsConfigured(unittest.TestCase):
    """Catch a dropped key on the next test run instead of after a wasted training block."""

    def _weights(self):
        cfg = yaml.safe_load(io.open("config/default_config.yaml", encoding="utf-8").read())
        live = json.load(io.open("config/live_config.json", encoding="utf-8"))
        return cfg.get("rewards", {}), live.get("rewards", {})

    def test_cost_terms_present_in_both_files(self):
        cfg, live = self._weights()
        for key in ("jump_cost_weight", "spin_cost_weight", "spin_cost_deadband"):
            self.assertIn(key, cfg, "%s missing from default_config.yaml" % key)
            self.assertIn(key, live, "%s missing from live_config.json" % key)
            self.assertAlmostEqual(cfg[key], live[key],
                                   msg="%s disagrees between the two files; live wins at "
                                       "runtime, so training is not using the yaml value" % key)

    def test_yaml_and_live_do_not_disagree_anywhere(self):
        cfg, live = self._weights()
        for key in sorted(set(cfg) & set(live)):
            self.assertAlmostEqual(
                cfg[key], live[key], places=9,
                msg="%s is %s in the yaml but %s in live_config; live_config wins, so the "
                    "yaml value is a lie" % (key, cfg[key], live[key]))

    def test_ui_apply_handler_does_not_replace_the_rewards_block(self):
        """The specific line that caused this. Assignment deletes; update preserves."""
        src = io.open("ui/app.py", encoding="utf-8").read()
        self.assertNotIn('base_cfg["rewards"] = rewards', src,
                         "assigning the rewards block deletes every weight without a slider; "
                         "use base_cfg.setdefault('rewards', {}).update(rewards)")
        self.assertNotIn('base_cfg["scenarios"] = scenarios', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
