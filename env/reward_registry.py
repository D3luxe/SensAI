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

CODE_FILES: Dict[str, Tuple[str, ...]] = {
    "v2": ("env/rewards.py",),
    "v3": ("env/rewards_v3.py", "env/scenarios_v3.py"),
    # v4 builds on v3's term functions and training starts, so their files are part of its identity
    "v4": ("env/rewards_v4.py", "env/rewards_v3.py", "env/scenarios_v3.py"),
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
    raise ValueError(f"unknown reward_version {version!r}")


def make_reward_manager(version: Optional[str] = None, reward_weights: Optional[Dict[str, float]] = None):
    """The reward manager for `version` (None: the version the default config names)."""
    version = version or active_version()
    if version == "v2":
        from env.rewards import RewardManager
        return RewardManager(reward_weights=reward_weights)
    if version == "v3":
        from env.rewards_v3 import RewardManagerV3
        return RewardManagerV3(reward_weights=reward_weights)
    if version == "v4":
        from env.rewards_v4 import RewardManagerV4
        return RewardManagerV4(reward_weights=reward_weights)
    raise ValueError(f"unknown reward_version {version!r}")
