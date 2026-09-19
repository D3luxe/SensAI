"""
Single-source-of-truth guarantees for configuration, reward/scenario defaults and checkpoint loading.

  1. The reward and scenario default tables cover exactly the keys the configs carry, and
     CombinedReward / WeightedScenarioSetter are built from them.
  2. utils.config routes live overrides into the right yaml sections and applies annealing on the
     checkpoint's own clocks, the way PPOTrainer does.
  3. agent.checkpoint.load_policy reproduces the saved policy exactly, reads the architecture from
     the checkpoint, and refuses a shape mismatch unless migration is explicitly allowed.
  4. No other module re-implements the yaml + live merge.
"""

import io
import json
import os
import re
import tempfile
import unittest

import torch
import yaml

from agent.checkpoint import load_policy, migrate_state_dict
from agent.models import ActorCritic
from env.observations import OBS_DIM
from env.rewards import CombinedReward, REWARD_DEFAULTS, REWARD_KEYS_NOT_IN_CONFIG, REWARD_WEIGHT_SPECS
from env.state_setters import SCENARIO_DEFAULTS, WeightedScenarioSetter
from utils.config import (
    CONFIG_REWARD_KEYS, anneal_progress, anneal_start_steps_from_checkpoint, effective_reward_weights,
    overlay_live,
)


def _yaml():
    with io.open("config/default_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestDefaultTables(unittest.TestCase):
    def test_config_reward_keys_match_the_table(self):
        # Reward settings live in each version's frozen snapshot (env/reward_registry.py)
        from env.reward_registry import known_versions, load_snapshot, reward_defaults
        self.assertEqual(set(load_snapshot("v2")["settings"]["rewards"]), set(CONFIG_REWARD_KEYS))
        for version in known_versions():
            self.assertEqual(set(load_snapshot(version)["settings"]["rewards"]), set(reward_defaults(version)), version)
        self.assertTrue(REWARD_KEYS_NOT_IN_CONFIG <= set(REWARD_DEFAULTS))

    def test_config_scenario_keys_match_the_table(self):
        # v2 predates the v3 starts, whose zero defaults leave its sampling unchanged
        from env.reward_registry import known_versions, load_snapshot
        for version in known_versions():
            self.assertLessEqual(set(load_snapshot(version)["settings"]["scenarios"]), set(SCENARIO_DEFAULTS), version)
        self.assertEqual(set(load_snapshot("v3")["settings"]["scenarios"]), set(SCENARIO_DEFAULTS))
        self.assertEqual(set(SCENARIO_DEFAULTS) - set(load_snapshot("v2")["settings"]["scenarios"]),
                         {"bounce_drop_prob", "retreat_prob"})
        self.assertEqual(SCENARIO_DEFAULTS["bounce_drop_prob"], 0.0)
        self.assertEqual(SCENARIO_DEFAULTS["retreat_prob"], 0.0)

    def test_combined_reward_is_built_from_the_table(self):
        c = CombinedReward({})
        for key, term, attr, default in REWARD_WEIGHT_SPECS:
            self.assertAlmostEqual(getattr(c.rewards[term], attr), default, msg=key)

    def test_every_table_key_is_live_tunable(self):
        c = CombinedReward({})
        c.update_weights({key: 0.123 for key in REWARD_DEFAULTS})
        for key, term, attr, _ in REWARD_WEIGHT_SPECS:
            self.assertAlmostEqual(getattr(c.rewards[term], attr), 0.123, msg=key)

    def test_scenario_setter_is_built_from_the_table(self):
        setter = WeightedScenarioSetter()
        for key, value in SCENARIO_DEFAULTS.items():
            self.assertAlmostEqual(getattr(setter, key), value)
        setter.update_weights({"scenarios": {"aerial_prob": 0.5}})
        self.assertAlmostEqual(setter.aerial_prob, 0.5)
        with self.assertRaises(TypeError):
            WeightedScenarioSetter(not_a_scenario_prob=1.0)


class TestEffectiveConfig(unittest.TestCase):
    def test_live_overrides_route_to_their_sections(self):
        base = {"rewards": {"touch_weight": 0.4}, "hyperparameters": {"ent_coef": 0.01, "gamma": 0.995},
                "league": {"king_ratio": 0.25}}
        live = {"rewards": {"touch_weight": 0.9}, "scenarios": {"aerial_prob": 0.3}, "ent_coef": 0.002,
                "league_enabled": False, "king_ratio": 0.4, "use_action_masking": False,
                "torch_num_threads": 6}
        cfg = overlay_live(base, live)
        self.assertEqual(cfg["rewards"]["touch_weight"], 0.9)
        self.assertEqual(cfg["scenarios"]["aerial_prob"], 0.3)
        self.assertEqual(cfg["hyperparameters"]["ent_coef"], 0.002)
        self.assertEqual(cfg["hyperparameters"]["gamma"], 0.995)
        self.assertEqual(cfg["league"], {"king_ratio": 0.4, "enabled": False})
        self.assertFalse(cfg["model"]["use_action_masking"])
        self.assertEqual(cfg["environment"]["torch_num_threads"], 6)
        self.assertEqual(base["rewards"]["touch_weight"], 0.4, "inputs must not be modified")

    def test_annealing_uses_the_checkpoint_clocks(self):
        cfg = {"rewards": {"player_to_ball_weight": 0.35, "powerslide_weight": 0.2},
               "hyperparameters": {"gamma": 0.99},
               "reward_annealing": {"enabled": True, "decay_steps": 100,
                                    "targets": {"player_to_ball_weight": 0.25, "powerslide_weight": 0.0}}}
        w = effective_reward_weights(global_step=150, anneal_start_steps={"player_to_ball_weight": 100}, cfg=cfg)
        self.assertAlmostEqual(w["player_to_ball_weight"], 0.30)   # 50% along its own clock
        self.assertAlmostEqual(w["powerslide_weight"], 0.2)        # clock not started yet
        self.assertAlmostEqual(w["gamma"], 0.99)
        w = effective_reward_weights(global_step=150, anneal_start_steps=None, cfg=cfg)
        self.assertAlmostEqual(w["powerslide_weight"], 0.0)        # no clocks known: assume step 0
        self.assertAlmostEqual(effective_reward_weights(cfg=cfg)["player_to_ball_weight"], 0.35)

    def test_anneal_progress_matches_the_trainer_formula(self):
        self.assertEqual(anneal_progress(500, None, 1000), 0.0)
        self.assertEqual(anneal_progress(500, 700, 1000), 0.0)
        self.assertAlmostEqual(anneal_progress(1200, 700, 1000), 0.5)
        self.assertEqual(anneal_progress(10 ** 9, 0, 1000), 1.0)

    def test_legacy_single_clock_covers_only_its_own_targets(self):
        ckpt = {"reward_anneal_start_step": 40,
                "config": {"reward_annealing": {"targets": {"powerslide_weight": 0.0}}}}
        self.assertEqual(anneal_start_steps_from_checkpoint(ckpt), {"powerslide_weight": 40})
        self.assertIsNone(anneal_start_steps_from_checkpoint({}))

    def test_no_module_reimplements_the_merge(self):
        for path in ("ui/app.py", "utils/visualizer.py", "scripts/policy_health.py",
                     "scripts/retreat_comparison.py", "scripts/shot_quality.py", "utils/diagnostics.py"):
            with io.open(path, encoding="utf-8") as f:
                src = f.read()
            opens = re.findall(r'open\(\s*["\']config/live_config\.json', src)
            self.assertEqual(opens, [], f"{path} reads live_config.json directly; use utils.config")


class TestImportGraph(unittest.TestCase):
    def test_entry_modules_import_cleanly_in_a_fresh_interpreter(self):
        # Subprocess env workers start at env.state_setters via physics_engine; an eager import in a
        # package __init__ once closed a cycle there that only showed up when training restarted.
        import subprocess
        import sys
        for module in ("env.state_setters", "env.physics_engine", "env.rewards", "utils.config",
                       "agent.checkpoint", "agent.ppo", "bot", "utils.visualizer"):
            proc = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"import {module} failed:\n{proc.stderr[-2000:]}")


class TestCheckpointLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        gen = torch.Generator().manual_seed(3)
        self.obs = torch.randn(64, OBS_DIM, generator=gen)

    def _save(self, model, name, **extra):
        path = os.path.join(self.tmp.name, name)
        torch.save({"model_state_dict": model.state_dict(), "obs_dim": model.obs_dim, "act_dim": 8,
                    "continuous_actions": True, "use_layer_norm": model.use_layer_norm,
                    "activation": "leaky_relu", **extra}, path)
        return path

    def test_round_trip_is_exact(self):
        with torch.random.fork_rng():
            torch.manual_seed(5)
            model = ActorCritic(obs_dim=OBS_DIM, act_dim=8, continuous_actions=True)
        model.eval()
        loaded, ckpt = load_policy(self._save(model, "a.pt", iteration=7))
        self.assertEqual(ckpt["iteration"], 7)
        with torch.no_grad():
            self.assertTrue(torch.equal(model.get_action_and_value(self.obs, deterministic=True)[0],
                                        loaded.get_action_and_value(self.obs, deterministic=True)[0]))

    def test_architecture_comes_from_the_checkpoint(self):
        model = ActorCritic(obs_dim=OBS_DIM, act_dim=8, continuous_actions=True, use_layer_norm=False)
        loaded, _ = load_policy(self._save(model, "no_ln.pt"))
        self.assertFalse(loaded.use_layer_norm)

    def test_obs_dim_mismatch_is_refused_unless_migration_is_allowed(self):
        model = ActorCritic(obs_dim=OBS_DIM - 14, act_dim=8, continuous_actions=True)
        path = self._save(model, "old.pt")
        with self.assertRaises(ValueError):
            load_policy(path)
        loaded, _ = load_policy(path, allow_shape_migration=True)
        self.assertEqual(loaded.obs_dim, OBS_DIM)

    def test_migrate_state_dict_pads_with_zeros(self):
        saved = {"w": torch.ones(2, 3), "same": torch.full((2,), 5.0)}
        target = {"w": torch.full((3, 4), 9.0), "same": torch.zeros(2)}
        out, migrated = migrate_state_dict(saved, target)
        self.assertTrue(migrated)
        self.assertTrue(torch.equal(out["w"][:2, :3], torch.ones(2, 3)))
        self.assertEqual(float(out["w"][2:, :].abs().sum() + out["w"][:, 3:].abs().sum()), 0.0)
        self.assertTrue(torch.equal(out["same"], saved["same"]))


if __name__ == "__main__":
    unittest.main()
