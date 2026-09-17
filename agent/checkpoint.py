"""
Single path for turning a checkpoint file into an evaluation-ready ActorCritic.

Used by the in-game bot, checkpoint opponents (league, TrueSkill, training pool), the match
visualizer and the diagnostic scripts, so every consumer plays exactly the policy that was saved:

  - the architecture comes from the checkpoint (obs/act dims, LayerNorm, activation, hidden sizes,
    action-masking config), not from each caller's guess
  - weights load strictly; a shape mismatch is an error unless allow_shape_migration is set, in
    which case overlapping slices are copied and the rest zero-filled, with a warning
  - the only post-load adjustment is sanitize_log_std(), which cannot change deterministic actions

The trainer's own resume (PPOTrainer.load_checkpoint) restores into its existing agent and
optimizer, and shares the shape migration through migrate_state_dict().
"""
from __future__ import annotations

import io
import os
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from agent.models import ActorCritic
from env.observations import OBS_DIM


def read_checkpoint(path: str, device: str = "cpu", retries: int = 5) -> Dict[str, Any]:
    """
    Load a checkpoint dict. The file is read into memory first, retrying briefly, because on
    Windows the trainer may hold it open mid-save and torch.load on the path would fail or lock it.
    """
    data = None
    for attempt in range(max(1, retries)):
        try:
            with open(path, "rb") as f:
                data = f.read()
            break
        except (PermissionError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(0.05)
    ckpt = torch.load(io.BytesIO(data), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt
    if isinstance(ckpt, dict):
        return {"model_state_dict": ckpt}  # bare state dict
    raise ValueError(f"{path} is not an ActorCritic checkpoint")


def migrate_state_dict(saved: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool]:
    """
    saved mapped onto target's shapes: matching tensors copied as-is, mismatched ones zero-filled
    with the overlapping slice copied in. Returns (state_dict, whether anything was reshaped).
    """
    out = dict(target)
    migrated = False
    for key, saved_param in saved.items():
        if key not in target:
            continue
        curr = target[key]
        if saved_param.shape == curr.shape:
            out[key] = saved_param
            continue
        migrated = True
        param = torch.zeros_like(curr)
        slices = tuple(slice(0, min(s, c)) for s, c in zip(saved_param.shape, curr.shape))
        param[slices] = saved_param[slices]
        out[key] = param
    return out, migrated


def _infer_architecture(ckpt: Mapping[str, Any]) -> Dict[str, Any]:
    state = ckpt["model_state_dict"]
    model_cfg = ((ckpt.get("config") or {}).get("model") or {})
    continuous = bool(ckpt.get("continuous_actions", "actor_mean.weight" in state))
    first = state.get("actor_backbone.0.weight")
    obs_dim = int(ckpt.get("obs_dim") or (first.shape[1] if first is not None else OBS_DIM))
    if continuous:
        act_dim = 8
    else:
        from env.actions import DiscreteActionParser
        act_dim = int(ckpt.get("act_dim") or DiscreteActionParser().action_dim)
    # With LayerNorm the backbone is Linear, LayerNorm, activation; module 1 then carries a weight
    use_layer_norm = bool(ckpt.get("use_layer_norm", "actor_backbone.1.weight" in state))
    return dict(
        obs_dim=obs_dim,
        act_dim=act_dim,
        continuous_actions=continuous,
        use_layer_norm=use_layer_norm,
        activation=str(ckpt.get("activation", model_cfg.get("activation", "leaky_relu"))),
        actor_hidden_dims=list(model_cfg.get("actor_hidden_dims", [256, 256, 128])),
        critic_hidden_dims=list(model_cfg.get("critic_hidden_dims", [256, 256, 128])),
        use_action_masking=bool(model_cfg.get("use_action_masking", True)),
        handbrake_height_buffer=float(model_cfg.get("handbrake_height_buffer", 120.0)),
    )


def build_policy(
    ckpt: Mapping[str, Any],
    device: str = "cpu",
    expected_obs_dim: Optional[int] = OBS_DIM,
    allow_shape_migration: bool = False,
    source: str = "checkpoint",
) -> ActorCritic:
    """An eval-mode ActorCritic holding ckpt's weights. See module docstring for the rules."""
    arch = _infer_architecture(ckpt)
    saved_obs_dim = arch["obs_dim"]
    if expected_obs_dim is not None and saved_obs_dim != expected_obs_dim:
        if not allow_shape_migration:
            raise ValueError(
                f"{source} was trained on {saved_obs_dim}-dim observations but the observation builder "
                f"produces {expected_obs_dim}; migrate it (scripts/migrate_checkpoint_to_94dim.py) or "
                f"pass allow_shape_migration=True")
        arch["obs_dim"] = expected_obs_dim

    model = ActorCritic(**arch).to(device)
    saved = ckpt["model_state_dict"]
    try:
        model.load_state_dict(saved)
    except RuntimeError as e:
        if not allow_shape_migration:
            raise ValueError(f"{source} does not match the ActorCritic architecture: {e}") from e
        state, _ = migrate_state_dict(saved, model.state_dict())
        model.load_state_dict(state)
        print(f"[Checkpoint] Warning: {source} shape-migrated ({saved_obs_dim} -> {arch['obs_dim']} obs); "
              f"padded weights are zero and the policy is not the one that was trained")
    model.sanitize_log_std()
    model.eval()
    return model


def load_policy(
    path: str,
    device: str = "cpu",
    expected_obs_dim: Optional[int] = OBS_DIM,
    allow_shape_migration: bool = False,
) -> Tuple[ActorCritic, Dict[str, Any]]:
    """(model, checkpoint dict) for the checkpoint at path."""
    ckpt = read_checkpoint(path, device=device)
    model = build_policy(ckpt, device=device, expected_obs_dim=expected_obs_dim,
                         allow_shape_migration=allow_shape_migration, source=os.path.basename(path))
    return model, ckpt
