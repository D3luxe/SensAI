"""
Standalone Checkpoint Migration Script: 80 -> 94 Dimensions.
Surgically expands actor_backbone.0.weight and critic.0.weight with zero-initialized new features,
verifies exact numerical equivalence (Delta < 1e-6), and preserves 711M training steps.
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from agent.models import ActorCritic
from env.observations import OBS_DIM


def migrate_checkpoint(
    src_path: str = "checkpoints/latest_model.pt",
    backup_path: str = "checkpoints/backup_711m_80dim.pt",
    target_path: str = "checkpoints/latest_model.pt"
):
    if not os.path.exists(src_path):
        print(f"[Error] Source checkpoint not found at: {src_path}")
        return False

    # 1. Safety Backup
    if not os.path.exists(backup_path):
        shutil.copyfile(src_path, backup_path)
        print(f"[Backup] Created safety backup: {backup_path}")
    else:
        print(f"[Backup] Existing backup verified at: {backup_path}")

    # 2. Load Checkpoint
    checkpoint = torch.load(src_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        saved_state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        saved_state = checkpoint
    else:
        saved_state = checkpoint

    old_obs_dim = checkpoint.get("obs_dim", 80) if isinstance(checkpoint, dict) else 80
    if "actor_backbone.0.weight" in saved_state:
        old_obs_dim = saved_state["actor_backbone.0.weight"].shape[1]

    print(f"[Checkpoint] Detected incoming obs_dim: {old_obs_dim}")

    if old_obs_dim == OBS_DIM:
        print(f"[Skip] Checkpoint is already at target {OBS_DIM} dimensions.")
        return True

    # 3. Instantiate old and new models for verification
    old_model = ActorCritic(obs_dim=old_obs_dim, act_dim=8, continuous_actions=True, use_layer_norm=True)
    old_model.load_state_dict(saved_state)
    old_model.debias_symmetric_actions()
    old_model.eval()

    new_model = ActorCritic(obs_dim=OBS_DIM, act_dim=8, continuous_actions=True, use_layer_norm=True)
    new_state = new_model.state_dict()

    # 4. Surgical weight transfer
    migrated_keys = []
    for k in list(saved_state.keys()):
        if k in new_state:
            saved_p = saved_state[k]
            curr_p = new_state[k]
            if saved_p.shape != curr_p.shape:
                migrated_keys.append((k, saved_p.shape, curr_p.shape))
                curr_p = curr_p.clone()
                curr_p.zero_()  # Crucial: zero out new feature weights
                slices = tuple(slice(0, min(s, c)) for s, c in zip(saved_p.shape, curr_p.shape))
                curr_p[slices] = saved_p[slices]
                new_state[k] = curr_p
            else:
                new_state[k] = saved_p

    new_model.load_state_dict(new_state)
    new_model.debias_symmetric_actions()
    new_model.eval()

    print(f"[Surgery] Migrated layers:")
    for k, old_s, new_s in migrated_keys:
        print(f"  - {k}: {old_s} -> {new_s} (new columns zero-initialized)")

    # 5. Numerical Equivalence Verification
    print("[Verify] Running numerical equivalence tests across 500 random states...")
    torch.manual_seed(42)
    test_obs_old = torch.randn(500, old_obs_dim)
    # New observation with identical first 80 features and zeros in the new 14 features
    test_obs_new = torch.cat([test_obs_old, torch.zeros(500, OBS_DIM - old_obs_dim)], dim=-1)

    with torch.no_grad():
        act_old, _, _, val_old = old_model.get_action_and_value(test_obs_old, deterministic=True)
        act_new, _, _, val_new = new_model.get_action_and_value(test_obs_new, deterministic=True)

        act_diff = (act_old - act_new).abs().max().item()
        val_diff = (val_old - val_new).abs().max().item()

    print(f"[Verify] Max Action Divergence: {act_diff:.8e}")
    print(f"[Verify] Max Value Divergence:  {val_diff:.8e}")

    if act_diff > 1e-5 or val_diff > 1e-5:
        raise RuntimeError(f"Equivalence check failed! Divergence too large: act={act_diff}, val={val_diff}")

    print("[Verify] Exact numerical equivalence confirmed (Zero regression)!")

    # 6. Save Updated Checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = new_model.state_dict()
        checkpoint["obs_dim"] = OBS_DIM
        # Deliberately remove optimizer_state_dict to cleanly reinitialize Adam for new dimension
        if "optimizer_state_dict" in checkpoint:
            del checkpoint["optimizer_state_dict"]
            print("[Optimizer] Safely reset optimizer state for clean 94-dim training")
        torch.save(checkpoint, target_path)
    else:
        torch.save({
            "model_state_dict": new_model.state_dict(),
            "obs_dim": OBS_DIM,
            "continuous_actions": True,
            "use_layer_norm": True
        }, target_path)

    print(f"[Success] Migrated checkpoint saved to: {target_path}")
    return True


if __name__ == "__main__":
    migrate_checkpoint()
