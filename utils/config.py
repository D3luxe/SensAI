"""
Single source of truth for reading configuration.

Training runs on three layers, and every reader must apply all three or it reports something the
trainer is not doing:

  1. config/default_config.yaml    the base
  2. config/live_config.json       runtime overrides written by the UI, picked up by the trainer
  3. reward_annealing              decays the named reward targets over decay_steps, each on its own
                                   clock (PPOTrainer._reward_anneal_start_steps, saved per checkpoint)

effective_config() applies 1 and 2 and returns a full nested config in the yaml's shape.
effective_reward_weights() adds 3 plus the gamma the boost shaping must discount with.
Code-side defaults for missing keys come from REWARD_DEFAULTS and SCENARIO_DEFAULTS.

The trainer consumes the live file incrementally (PPOTrainer.check_live_config) because it applies
changes as they arrive; the key routing it uses is the LIVE_*_KEYS tables below, and its annealing
math is anneal_progress()/annealed_weights(), so both paths agree by construction.
"""
from __future__ import annotations

import copy
import io
import json
import os
from typing import Any, Dict, Mapping, Optional

import yaml

from env.rewards import REWARD_DEFAULTS, REWARD_KEYS_NOT_IN_CONFIG
from env.state_setters import SCENARIO_DEFAULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = "config/default_config.yaml"
LIVE_CONFIG_PATH = "config/live_config.json"

# Top-level live_config.json keys and the yaml section they override
LIVE_HYPERPARAMETER_KEYS = (
    "learning_rate", "ent_coef", "clip_range", "bc_regularization_weight", "bc_decay_steps",
    "rot_log_std_ceiling_final", "rot_log_std_anneal_iters",
)
LIVE_MODEL_KEYS = ("use_action_masking", "handbrake_height_buffer")
LIVE_LEAGUE_KEYS = {  # live key -> league key
    "league_enabled": "enabled",
    "self_play_ratio": "self_play_ratio",
    "king_ratio": "king_ratio",
    "pool_ratio": "pool_ratio",
    "training_opponents": "training_opponents",
    "training_opponent_ratio": "training_opponent_ratio",
    "contender_series_per_step": "contender_series_per_step",
}
LIVE_ENVIRONMENT_KEYS = ("baseline_opponent_ratio", "baseline_opponent_type", "torch_num_threads")

CONFIG_REWARD_KEYS = tuple(k for k in REWARD_DEFAULTS if k not in REWARD_KEYS_NOT_IN_CONFIG)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_yaml(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    full = _resolve(path)
    if not os.path.exists(full):
        return {}
    with io.open(full, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_live(path: str = LIVE_CONFIG_PATH) -> Dict[str, Any]:
    """The live overrides, or {} when the file is missing or mid-write."""
    full = _resolve(path)
    if not os.path.exists(full):
        return {}
    try:
        with io.open(full, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def overlay_live(base: Mapping[str, Any], live: Mapping[str, Any]) -> Dict[str, Any]:
    """The yaml-shaped config with live overrides routed into their sections. Inputs are not modified."""
    cfg = copy.deepcopy(dict(base))
    rewards = cfg.setdefault("rewards", {}) or {}
    scenarios = cfg.setdefault("scenarios", {}) or {}
    cfg["rewards"], cfg["scenarios"] = rewards, scenarios

    if isinstance(live.get("rewards"), dict):
        rewards.update(live["rewards"])
    if isinstance(live.get("scenarios"), dict):
        scenarios.update(live["scenarios"])

    hp = cfg.setdefault("hyperparameters", {}) or {}
    model = cfg.setdefault("model", {}) or {}
    league = cfg.setdefault("league", {}) or {}
    env = cfg.setdefault("environment", {}) or {}
    cfg.update(hyperparameters=hp, model=model, league=league, environment=env)
    for key in LIVE_HYPERPARAMETER_KEYS:
        if key in live:
            hp[key] = live[key]
    for key in LIVE_MODEL_KEYS:
        if key in live:
            model[key] = live[key]
    for live_key, league_key in LIVE_LEAGUE_KEYS.items():
        if live_key in live:
            league[league_key] = live[live_key]
    for key in LIVE_ENVIRONMENT_KEYS:
        if key in live:
            env[key] = live[key]
    if "baseline_opponent_type" not in live and "baseline_opponent_model" in live:
        env["baseline_opponent_type"] = live["baseline_opponent_model"]
    return cfg


def effective_config(
    base_path: str = DEFAULT_CONFIG_PATH,
    live_path: Optional[str] = LIVE_CONFIG_PATH,
    fill_defaults: bool = True,
) -> Dict[str, Any]:
    """
    The config the trainer is running with: yaml, then live overrides, then (optionally) code
    defaults for any reward or scenario key neither file carries. Pass live_path=None for the
    yaml alone.
    """
    cfg = overlay_live(load_yaml(base_path), load_live(live_path) if live_path else {})
    if fill_defaults:
        for key in CONFIG_REWARD_KEYS:
            cfg["rewards"].setdefault(key, REWARD_DEFAULTS[key])
        for key, value in SCENARIO_DEFAULTS.items():
            cfg["scenarios"].setdefault(key, value)
    return cfg


def anneal_progress(global_step: int, start_step: Optional[int], decay_steps: int) -> float:
    """Fraction of one anneal target's schedule elapsed, in [0, 1]. An unstarted clock is at 0."""
    if start_step is None:
        return 0.0
    return min(1.0, max(0, int(global_step) - int(start_step)) / max(1, int(decay_steps)))


def annealed_weights(
    base_weights: Mapping[str, float],
    anneal_cfg: Mapping[str, Any],
    progress: Mapping[str, float],
) -> Dict[str, float]:
    """base_weights with each enabled anneal target moved `progress[key]` of the way to its target."""
    weights = dict(base_weights)
    if not anneal_cfg or not anneal_cfg.get("enabled"):
        return weights
    for key, target in (anneal_cfg.get("targets") or {}).items():
        start_value = float(base_weights.get(key, REWARD_DEFAULTS.get(key, 0.0)))
        weights[key] = start_value + (float(target) - start_value) * float(progress.get(key, 0.0))
    return weights


def anneal_start_steps_from_checkpoint(checkpoint: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    """
    The per-target anneal clocks a checkpoint was saved with, or None when it carries none.
    A legacy single reward_anneal_start_step applies to the targets that checkpoint's own config
    named (see PPOTrainer._load_reward_anneal_clocks).
    """
    saved = checkpoint.get("reward_anneal_start_steps")
    if isinstance(saved, dict):
        return {str(k): int(v) for k, v in saved.items() if v is not None}
    legacy = checkpoint.get("reward_anneal_start_step")
    if legacy is None:
        return None
    covered = ((checkpoint.get("config") or {}).get("reward_annealing") or {}).get("targets") or {}
    return {str(k): int(legacy) for k in covered}


def effective_reward_weights(
    global_step: Optional[int] = None,
    anneal_start_steps: Optional[Mapping[str, int]] = None,
    cfg: Optional[Mapping[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Reward weights exactly as the trainer applies them at global_step.

    anneal_start_steps are the checkpoint's clocks (anneal_start_steps_from_checkpoint). Without
    them every clock is assumed to have started at step 0, which is right for a run whose targets
    were all configured from the start. With global_step=None no annealing is applied.
    """
    cfg = effective_config() if cfg is None else cfg
    base = {**{k: REWARD_DEFAULTS[k] for k in CONFIG_REWARD_KEYS}, **(cfg.get("rewards") or {})}
    base["gamma"] = float((cfg.get("hyperparameters") or {}).get("gamma", REWARD_DEFAULTS["gamma"]))

    anneal = cfg.get("reward_annealing") or {}
    if global_step is None or not anneal.get("enabled"):
        return base
    decay = int(anneal.get("decay_steps", 300_000_000))
    targets = anneal.get("targets") or {}
    progress = {
        key: anneal_progress(global_step, (anneal_start_steps or {}).get(key, 0 if anneal_start_steps is None else None), decay)
        for key in targets
    }
    weights = annealed_weights(base, anneal, progress)
    if verbose and targets:
        print("annealed weights: " + " | ".join(
            f"{k} {weights[k]:.4f} (from {base.get(k, 0.0):.3f}, {100.0 * progress[k]:.0f}% annealed)" for k in targets))
    return weights


def effective_reward_weights_for_checkpoint(
    checkpoint: Mapping[str, Any], cfg: Optional[Mapping[str, Any]] = None, verbose: bool = False
) -> Dict[str, float]:
    """effective_reward_weights at a checkpoint's global_step, on that checkpoint's anneal clocks."""
    return effective_reward_weights(
        checkpoint.get("global_step"), anneal_start_steps_from_checkpoint(checkpoint), cfg=cfg, verbose=verbose)
