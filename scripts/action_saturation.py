"""
Throttle and steer saturation: how often the policy's action mean sits at full lock on the ground,
and how much gradient reaches it there.

The action mean is tanh(actor_mean(features)) (averaged with the mirrored pass), and PPO's Normal
is centred on it. Once a pre-activation is well past full lock, (1 - tanh^2) is ~0 and the policy
gradient can no longer move that channel in that state. The env clips sampled actions to [-1, 1],
so a mean at the rail still acts as full lock however far out it drifts. Throttle's log_std floor
(sigma 0.082) leaves too little exploration around it to find braking.

This plays deterministic matches against Necto through the eval suite's own match loop
(scripts/shot_quality.py Game), records every observation where the model's grounded flag is set
(obs[19], the flag its action masking uses), then evaluates the action head on those states:

  sat%        share of grounded states with |mean| > --sat (default 0.95)
  grad med    median d(mean)/d(pre-activation), averaged over the raw and mirrored passes
  |pre| med   median pre-activation magnitude (tanh 0.95 is at 1.83)
  branch%     share of (state, pass) pairs where the raw or mirrored pass alone is past --sat; the
              mean averages the two, so opposed saturated passes read as an unsaturated mean
  opposed%    share of grounded states where both passes are past --sat in opposite directions,
              so the mean reads ~0 with no gradient (steer's cancellation)
  brake%      share of grounded states with a throttle mean below -0.1

Measured 2026-09-23 in a separate session: steer 76-85% and throttle 54-69% saturated for every
checkpoint from v3 198k to v10 229k, pretrained_baseline 11% / 4%. Re-measure with this before
claiming any fix.

    python scripts/action_saturation.py
    python scripts/action_saturation.py checkpoints/baselines/v5_iter222000.pt --steps 3000 --seeds 1 2
    python scripts/action_saturation.py --json logs/saturation_v11_100M.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("RS_COLLISION_MESHES", os.path.join(ROOT, "collision_meshes"))

NECTO = "checkpoints/necto-model.pt"
GROUNDED_IDX = 19
CHANNELS = ("throttle", "steer")
DEFAULT_CHECKPOINTS = [
    "checkpoints/pretrained_baseline.pt",
    "checkpoints/baselines/v3_iter198000.pt",
    "checkpoints/baselines/v5_iter222000.pt",
]


def grounded_observations(checkpoint: str, steps: int, seed: int) -> np.ndarray:
    """Every observation the policy acted on while grounded, over one deterministic match vs Necto."""
    from policy_health import load_agent, load_weights
    from shot_quality import Game
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    agent, ckpt = load_agent(checkpoint)
    seen: List[np.ndarray] = []
    original = agent.get_action_and_value

    def recording(obs, *a, **kw):
        # The env builds an observation for both cars; row 0 is SensAI (blue, arena.cars[0]). Row 1
        # is the orange car, whose action Necto overrides, so it is not a state SensAI acted on.
        seen.append(obs.detach().cpu().numpy().reshape(-1, obs.shape[-1])[:1].copy())
        return original(obs, *a, **kw)

    agent.get_action_and_value = recording
    Game(agent, load_weights(ckpt), NECTO).run(steps)
    agent.get_action_and_value = original
    obs = np.concatenate(seen, axis=0)
    return obs[obs[:, GROUNDED_IDX] > 0.5]


def head_readings(model, obs: np.ndarray) -> Dict[str, np.ndarray]:
    """Pre-activations, means and d(mean)/d(pre) for throttle and steer on the given states."""
    o = torch.from_numpy(obs).float()
    with torch.no_grad():
        pre = model.actor_mean(model.actor_backbone(o))[:, :2]
        mirrored = (o.shape[-1] == model.obs_mirror_mask.shape[-1]
                    and model.act_dim == model.act_mirror_mask.shape[-1])
        if mirrored:
            idx = getattr(model, "obs_mirror_indices", None)
            om = o * model.obs_mirror_mask
            if idx is not None and idx.shape[-1] == o.shape[-1]:
                om = om[..., idx]
            pre_m = model.actor_mean(model.actor_backbone(om))[:, :2]
            sign = model.act_mirror_mask[:2]
            mean = 0.5 * (torch.tanh(pre) + torch.tanh(pre_m) * sign)
            grad = 0.5 * (1 - torch.tanh(pre) ** 2) + 0.5 * (1 - torch.tanh(pre_m) ** 2)
            mag = 0.5 * (pre.abs() + pre_m.abs())
            branch = torch.stack([torch.tanh(pre).abs(), torch.tanh(pre_m).abs()], dim=0)
            # Each pass in the raw pass's frame, so opposite signs mean the two passes disagree
            signed = torch.stack([torch.tanh(pre), torch.tanh(pre_m) * sign], dim=0)
        else:
            mean = torch.tanh(pre)
            grad = 1 - mean ** 2
            mag = pre.abs()
            branch = mean.abs().unsqueeze(0)
            signed = mean.unsqueeze(0)
    return {"mean": mean.numpy(), "grad": grad.numpy(), "pre": mag.numpy(), "branch": branch.numpy(),
            "signed": signed.numpy()}


def measure(checkpoint: str, steps: int, seeds: List[int], sat: float) -> Dict[str, object]:
    from policy_health import load_agent
    obs = np.concatenate([grounded_observations(checkpoint, steps, s) for s in seeds], axis=0)
    model, ckpt = load_agent(checkpoint)
    r = head_readings(model, obs)
    sigma = torch.exp(model.clamped_log_std()).detach().numpy().reshape(-1)[:2] \
        if hasattr(model, "clamped_log_std") else [float("nan")] * 2
    out: Dict[str, object] = {"checkpoint": checkpoint, "iteration": ckpt.get("iteration"),
                              "version": (ckpt.get("reward_identity") or {}).get("version"),
                              "grounded_states": int(len(obs)), "steps": steps, "seeds": seeds}
    for i, ch in enumerate(CHANNELS):
        m = r["mean"][:, i]
        out[ch] = {
            "sat_pct": float(100.0 * np.mean(np.abs(m) > sat)),
            "sat_pos_pct": float(100.0 * np.mean(m > sat)),
            "sat_neg_pct": float(100.0 * np.mean(m < -sat)),
            # Each of the raw and mirrored passes on its own. The mean averages them, so two
            # saturated passes pointing opposite ways give an unsaturated mean with no gradient.
            "branch_sat_pct": float(100.0 * np.mean(r["branch"][:, :, i] > sat)),
            # Both passes past --sat in opposite directions: the mean reads ~0 with no gradient
            "opposed_pct": float(100.0 * np.mean((r["branch"][:, :, i] > sat).all(axis=0)
                                                 & (np.sign(r["signed"][0, :, i]) != np.sign(r["signed"][-1, :, i])))),
            "grad_median": float(np.median(r["grad"][:, i])),
            "pre_abs_median": float(np.median(r["pre"][:, i])),
            "sigma": float(sigma[i]),
        }
    out["throttle"]["brake_pct"] = float(100.0 * np.mean(r["mean"][:, 0] < -0.1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--steps", type=int, default=3000, help="steps per match (15 per second)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--sat", type=float, default=0.95)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for c in args.checkpoints:
        row = measure(c, args.steps, args.seeds, args.sat)
        rows.append(row)
        t, s = row["throttle"], row["steer"]
        print(f"{os.path.basename(c):32} n={row['grounded_states']:5}  "
              f"steer sat {s['sat_pct']:5.1f}% branch {s['branch_sat_pct']:5.1f}% opposed {s['opposed_pct']:4.1f}% grad {s['grad_median']:.3f} |pre| {s['pre_abs_median']:5.2f} sd {s['sigma']:.3f}   "
              f"throttle sat {t['sat_pct']:5.1f}% (+{t['sat_pos_pct']:.1f}/-{t['sat_neg_pct']:.1f}) branch {t['branch_sat_pct']:5.1f}% grad {t['grad_median']:.3f} "
              f"|pre| {t['pre_abs_median']:5.2f} sd {t['sigma']:.3f} brake {t['brake_pct']:4.1f}%", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
