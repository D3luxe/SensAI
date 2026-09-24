"""
Which reward version a run uses, and where its frozen settings come from.

config/default_config.yaml names the version with a top-level `reward_version` key. The version's
reward weights, anneal schedule and scenario mix then come from config/reward_versions/<v>.json
and nowhere else: the yaml carries no reward sections, and live_config.json overrides of `rewards`
or `scenarios` are ignored (spec rule R1 -- frozen per run; a change is a new version).

A config that names no version is a pre-versioning config: it ran v2 with its own `rewards`,
`reward_annealing` and `scenarios` sections plus live overrides, and is still read that way.

  VERSIONS                version -> (reward manager factory, defaults, code files hashed into its identity)
  version_of(cfg)         the version a config names
  active_version()        the version config/default_config.yaml names
  load_snapshot(v)        config/reward_versions/<v>.json
  apply_reward_version    a config with the version's frozen sections in place
  make_reward_manager     the reward manager for a version
  scenario_payload        what the environments are sent: the scenario mix, and v7's replay sampling
                          (v8-v10 reuse it)
"""
from __future__ import annotations

import copy
import functools
import io
import json
import os
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = "config/default_config.yaml"
SNAPSHOT_DIR = "config/reward_versions"
LEGACY_VERSION = "v2"
SETTINGS_SECTIONS = ("rewards", "reward_annealing", "scenarios")
# Sections only some versions carry. Hashed into settings_sha only when present, so adding one leaves
# every earlier version's identity untouched.
# action_regularization (v11) is a loss on the policy's output, not a reward, but it defines the run
# the same way, so it is frozen and hashed with the rest.
OPTIONAL_SETTINGS_SECTIONS = ("replay_sampling", "action_regularization")

CODE_FILES: Dict[str, Tuple[str, ...]] = {
    "v2": ("env/rewards.py",),
    "v3": ("env/rewards_v3.py", "env/scenarios_v3.py"),
    # v4 builds on v3's term functions and training starts, so their files are part of its identity
    "v4": ("env/rewards_v4.py", "env/rewards_v3.py", "env/scenarios_v3.py"),
    # v5 is v3's terms at a longer horizon: same code, so the same files and the same code_sha.
    # Only gamma differs, and gamma lives in the settings, so settings_sha is what separates them.
    "v5": ("env/rewards_v3.py", "env/scenarios_v3.py"),
    # v6 is v5 plus T6 align; it imports v3's term functions and trains on v3's starts
    "v6": ("env/rewards_v6.py", "env/rewards_v3.py", "env/scenarios_v3.py"),
    # v7 is v5's reward (v6 was not adopted) with its replay starts pruned, tagged and mirrored
    "v7": ("env/rewards_v3.py", "env/scenarios_v3.py", "env/replay_sampling_v7.py"),
    # v8 is v5's reward with T5 boost turned into the T7 ratchet, on v7's replay machinery at the
    # pool's own tag frequencies; it imports v3's term functions
    "v8": ("env/rewards_v8.py", "env/rewards_v3.py", "env/scenarios_v3.py", "env/replay_sampling_v7.py"),
    # v9 is v8 with the T7 ratchet retired for the T8 boost-edge state bonus; everything else --
    # terms, gamma, scenarios, replay machinery and tag frequencies -- is v8's, unchanged
    "v9": ("env/rewards_v9.py", "env/rewards_v3.py", "env/scenarios_v3.py", "env/replay_sampling_v7.py"),
    # v10 is v7 at the pool's natural tag frequencies: v5's reward and v7's code, byte for byte, so
    # v7's files and code_sha. Only the replay tag weights differ, and they live in the settings
    "v10": ("env/rewards_v3.py", "env/scenarios_v3.py", "env/replay_sampling_v7.py"),
    # v11 is v10 plus a penalty on the throttle/steer pre-tanh magnitude in PPO's loss; the reward
    # and starts are v10's, and the penalty's code joins the identity
    "v11": ("env/rewards_v3.py", "env/scenarios_v3.py", "env/replay_sampling_v7.py", "agent/pre_tanh_penalty.py"),
}


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def known_versions() -> Tuple[str, ...]:
    return tuple(CODE_FILES)


def version_of(cfg: Optional[Mapping[str, Any]]) -> str:
    v = str((cfg or {}).get("reward_version") or LEGACY_VERSION)
    if v not in CODE_FILES:
        raise ValueError(f"unknown reward_version {v!r}; known: {', '.join(CODE_FILES)}")
    return v


def active_version(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    full = _resolve(config_path)
    if not os.path.exists(full):
        return LEGACY_VERSION
    with io.open(full, "r", encoding="utf-8") as f:
        return version_of(yaml.safe_load(f) or {})


@functools.lru_cache(maxsize=None)
def _snapshot_text(version: str) -> str:
    with io.open(_resolve(os.path.join(SNAPSHOT_DIR, f"{version}.json")), "r", encoding="utf-8") as f:
        return f.read()


def load_snapshot(version: str) -> Dict[str, Any]:
    return json.loads(_snapshot_text(version))


def apply_reward_version(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    cfg with the named version's frozen settings sections in place of whatever it carried. A config
    naming no version is returned unchanged (as a copy). Raises if the config's gamma differs from
    the one the version was frozen with, since the potential terms telescope only under that gamma.
    """
    out = copy.deepcopy(dict(cfg))
    if not out.get("reward_version"):
        return out
    version = version_of(out)
    settings = load_snapshot(version)["settings"]
    for section in SETTINGS_SECTIONS:
        out[section] = copy.deepcopy(settings[section])
    for section in OPTIONAL_SETTINGS_SECTIONS:
        if section in settings:
            out[section] = copy.deepcopy(settings[section])
        else:
            out.pop(section, None)
    gamma = float((out.get("hyperparameters") or {}).get("gamma", settings["gamma"]))
    if abs(gamma - float(settings["gamma"])) > 1e-12:
        raise ValueError(f"hyperparameters.gamma {gamma} differs from reward {version}'s frozen gamma "
                         f"{settings['gamma']}: a new gamma is a new reward version")
    return out


def reward_defaults(version: str) -> Dict[str, float]:
    """Code-side reward defaults for a version (config keys only)."""
    if version == "v2":
        from env.rewards import REWARD_DEFAULTS, REWARD_KEYS_NOT_IN_CONFIG
        return {k: v for k, v in REWARD_DEFAULTS.items() if k not in REWARD_KEYS_NOT_IN_CONFIG}
    if version == "v3":
        from env.rewards_v3 import REWARD_V3_DEFAULTS
        return {k: v for k, v in REWARD_V3_DEFAULTS.items() if k != "gamma"}
    if version == "v4":
        from env.rewards_v4 import REWARD_V4_DEFAULTS
        return {k: v for k, v in REWARD_V4_DEFAULTS.items() if k != "gamma"}
    if version in ("v5", "v7", "v10", "v11"):
        from env.rewards_v3 import REWARD_V3_DEFAULTS
        return {k: v for k, v in REWARD_V3_DEFAULTS.items() if k != "gamma"}
    if version == "v6":
        from env.rewards_v6 import REWARD_V6_DEFAULTS
        return {k: v for k, v in REWARD_V6_DEFAULTS.items() if k != "gamma"}
    if version == "v8":
        from env.rewards_v8 import REWARD_V8_DEFAULTS
        return {k: v for k, v in REWARD_V8_DEFAULTS.items() if k != "gamma"}
    if version == "v9":
        from env.rewards_v9 import REWARD_V9_DEFAULTS
        return {k: v for k, v in REWARD_V9_DEFAULTS.items() if k != "gamma"}
    raise ValueError(f"unknown reward_version {version!r}")


def make_reward_manager(version: Optional[str] = None, reward_weights: Optional[Dict[str, float]] = None):
    """
    The reward manager for `version` (None: the version the default config names).

    Weights that do not carry a gamma get the version's frozen one. Without this a caller that
    builds an env without weights silently gets the reward module's code default, which is v3's
    0.995 for both v3 and v5 -- and v5 exists precisely because its gamma is 0.9977. Potentials
    telescope only under the gamma the policy is optimised with, so a mismatch here is the exact
    failure apply_reward_version refuses to let a config express.
    """
    version = version or active_version()
    weights = dict(reward_weights or {})
    weights.setdefault("gamma", float(load_snapshot(version)["settings"]["gamma"]))
    reward_weights = weights
    if version == "v2":
        from env.rewards import RewardManager
        return RewardManager(reward_weights=reward_weights)
    if version == "v3":
        from env.rewards_v3 import RewardManagerV3
        return RewardManagerV3(reward_weights=reward_weights)
    if version == "v4":
        from env.rewards_v4 import RewardManagerV4
        return RewardManagerV4(reward_weights=reward_weights)
    if version in ("v5", "v7", "v10", "v11"):
        # v5's terms are v3's, unchanged; the version differs only in gamma, which the
        # manager reads from its weights. Nothing reads RewardManagerV3.version. v7's reward is
        # v5's, unchanged; v7 differs in where episodes start, and v10 in how v7's starts are weighted.
        # v11's reward is v10's; it differs in PPO's loss (action_regularization).
        from env.rewards_v3 import RewardManagerV3
        return RewardManagerV3(reward_weights=reward_weights)
    if version == "v6":
        from env.rewards_v6 import RewardManagerV6
        return RewardManagerV6(reward_weights=reward_weights)
    if version == "v8":
        from env.rewards_v8 import RewardManagerV8
        return RewardManagerV8(reward_weights=reward_weights)
    if version == "v9":
        from env.rewards_v9 import RewardManagerV9
        return RewardManagerV9(reward_weights=reward_weights)
    raise ValueError(f"unknown reward_version {version!r}")


def scenario_payload(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    The scenario settings the environments are sent (WeightedScenarioSetter.update_weights): the
    mix, plus the version's replay sampling when it has one. None of v2-v6 has one, so theirs is
    the mix alone, exactly as before.
    """
    payload = dict(cfg.get("scenarios") or {})
    if cfg.get("replay_sampling"):
        payload["replay_sampling"] = copy.deepcopy(cfg["replay_sampling"])
    return payload
