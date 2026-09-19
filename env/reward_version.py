"""
Reward version identity.

A reward version is frozen: its code, weights, anneal schedule and scenario mix are fixed before a
run starts and never edited during one. A change is a new version and a new run, compared against
the old one on the evaluation suite (scripts/eval_suite.py). That is the whole point of this file:
every checkpoint records which reward produced it, so any result can be traced and reproduced.

  code_sha(files)  hash of a version's source (line endings normalised); defaults to v2's files
  settings_sha()   hash of the reward settings as the trainer merges them: weights, annealing, scenarios
  reward_identity  version, code sha and settings sha, as stamped into every checkpoint

The version is the one the config names (env/reward_registry.py). Each version's identity is
snapshotted in config/reward_versions/<v>.json, and test_reward_versions_frozen.py fails if the
code or settings of any snapshotted version drift: versions are never edited in place.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Mapping, Optional

from env.reward_registry import CODE_FILES, SETTINGS_SECTIONS, version_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARD_CODE_FILES = CODE_FILES["v2"]


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
    if cfg is None:
        from utils.config import effective_config
        cfg = effective_config()
    version = version_of(cfg)
    return {"version": version, "code_sha": code_sha(CODE_FILES[version]), "settings_sha": settings_sha(cfg)}
