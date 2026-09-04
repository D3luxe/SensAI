"""
Supervised Alignment Script for SenseiBot Actor Network.
Eliminates degenerate forward backflip habit by aligning the actor head
to balanced forward/diagonal dodge demonstrations while preserving 275M steps
of learned critic value estimation and game positioning.
"""

import os
import sys
import shutil
import time
import torch
import numpy as np

sys.path.insert(0, ".")
from agent.models import ActorCritic
from agent.pretrainer import BehavioralCloningTrainer


def align_checkpoint(
    ckpt_path: str = "checkpoints/latest_model.pt",
    epochs: int = 25,
    batch_size: int = 512,
    lr: float = 3e-4,
    device: str = "cpu"
):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1. Create a safe backup of the current checkpoint
    backup_path = "checkpoints/latest_model_before_alignment_iter33620.pt"
    if not os.path.exists(backup_path):
        shutil.copy2(ckpt_path, backup_path)
        print(f"[Alignment] Backed up {ckpt_path} -> {backup_path}")

    # 2. Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    iteration = ckpt.get("iteration", 33620)
    global_step = ckpt.get("global_step", 275406848)
    print(f"[Alignment] Loaded checkpoint at Iteration {iteration:,}, Global Step {global_step:,}")

    model = ActorCritic(obs_dim=74, act_dim=8, continuous_actions=True, use_layer_norm=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # 3. Generate balanced demonstration dataset
    trainer = BehavioralCloningTrainer(device=device)
    print("[Alignment] Generating balanced mechanics dataset (front-flips, speed-flips, kickoffs, half-flips)...")
    obs_data, act_data = trainer.generate_pretrain_dataset(max_samples=40000)
    print(f"[Alignment] Dataset ready: {len(obs_data):,} frames.")

    obs_tensor = torch.tensor(obs_data, dtype=torch.float32, device=device)
    act_tensor = torch.tensor(act_data, dtype=torch.float32, device=device)
    dataset_size = len(obs_tensor)

    # 4. Train Actor parameters ONLY (preserve Critic 100% intact!)
    actor_params = (
        list(model.actor_backbone.parameters()) +
        list(model.actor_mean.parameters()) +
        list(model.actor_binary.parameters())
    )
    optimizer = torch.optim.Adam(actor_params, lr=lr, weight_decay=1e-5)

    print(f"[Alignment] Starting {epochs} alignment epochs (Batch size: {batch_size}, LR: {lr})...")
    start_time = time.time()
    num_batches = dataset_size // batch_size

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(dataset_size)
        epoch_loss = 0.0
        epoch_pitch_err = 0.0

        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            b_obs = obs_tensor[idx]
            b_act = act_tensor[idx]

            optimizer.zero_grad()
            feat = model.actor_backbone(b_obs)
            pred_cont = torch.tanh(model.actor_mean(feat))
            pred_bin = model.actor_binary(feat)

            target_cont = b_act[:, :5]
            target_bin = (b_act[:, 5:] > 0.0).float()

            loss_cont = torch.nn.functional.smooth_l1_loss(pred_cont, target_cont)
            loss_bin = torch.nn.functional.binary_cross_entropy_with_logits(pred_bin, target_bin)
            loss = loss_cont + 0.5 * loss_bin

            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pitch_err += torch.abs(pred_cont[:, 2] - target_cont[:, 2]).mean().item()

        if epoch % 5 == 0 or epoch == epochs:
            avg_loss = epoch_loss / num_batches
            avg_pitch_err = epoch_pitch_err / num_batches
            print(f"[Alignment] Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | Pitch L1 Error: {avg_pitch_err:.4f}")

    # 5. Debias symmetric heads & enforce healthy exploration bounds
    model.debias_symmetric_actions()

    # 6. Save aligned checkpoint preserving all metadata
    ckpt["model_state_dict"] = model.state_dict()
    # Reset optimizer state dict so PPO starts with clean momentum buffers matching the aligned actor
    if "optimizer_state_dict" in ckpt:
        del ckpt["optimizer_state_dict"]
    ckpt["aligned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    ckpt["alignment_epochs"] = epochs

    torch.save(ckpt, ckpt_path)
    # Also update manual checkpoint for Step 275202048 if present
    step_ckpt_path = "checkpoints/manual_checkpoint_step_275202048.pt"
    if os.path.exists(step_ckpt_path):
        torch.save(ckpt, step_ckpt_path)
        print(f"[Alignment] Also updated {step_ckpt_path}")

    elapsed = time.time() - start_time
    print(f"[Alignment] Completed successfully in {elapsed:.1f}s. Saved aligned model to {ckpt_path}.")


if __name__ == "__main__":
    align_checkpoint()
