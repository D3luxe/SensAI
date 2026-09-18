"""
Reward version identity.

A reward version is frozen: its code, weights, anneal schedule and scenario mix are fixed before a
run starts and never edited during one. A change is a new version and a new run, compared against
the old one on the evaluation suite (scripts/eval_suite.py). That is the whole point of this file:
every checkpoint records which reward produced it, so any result can be traced and reproduced.

  REWARD_VERSION   the version the trainer is running
  code_sha()       hash of the reward source (line endings normalised)
  settings_sha()   hash of the reward settings as the trainer merges them: weights, annealing, scenarios
  reward_identity  all three, as stamped into every checkpoint

v2 is the reward at git tag `reward-v2`, with settings snapshotted in config/reward_versions/v2.json.
test_reward_v2_frozen.py fails if either drifts: v2 is never edited in place, v3 lives beside it.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Mapping, Optional

REWARD_VERSION = "v2"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARD_CODE_FILES = ("env/rewards.py",)
SETTINGS_SECTIONS = ("rewards", "reward_annealing", "scenarios")


def code_sha(files=REWARD_CODE_FILES) -> str:
    h = hashlib.sha256()
    for rel in files:
        with open(os.path.join(ROOT, rel), "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def reward_settings(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The reward-defining sections of the merged config (yaml + live overrides), bookkeeping stripped."""
    if cfg is None:
        from utils.config import effective_config
        cfg = effective_config()
    out = {}
    for section in SETTINGS_SECTIONS:
        val = cfg.get(section) or {}
        if isinstance(val, Mapping):
            val = {k: v for k, v in val.items() if k != "updated_at"}
        out[section] = val
    out["gamma"] = float((cfg.get("hyperparameters") or {}).get("gamma", 0.995))
    return out


def settings_sha(cfg: Optional[Mapping[str, Any]] = None) -> str:
    blob = json.dumps(reward_settings(cfg), sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def reward_identity(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    return {"version": REWARD_VERSION, "code_sha": code_sha(), "settings_sha": settings_sha(cfg)}
