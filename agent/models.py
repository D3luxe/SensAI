"""
Actor-Critic Neural Network Architecture for Rocket League PPO Agents.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, Bernoulli
from typing import List, Tuple, Optional, Union


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def get_activation_cls(activation: str):
    act = activation.lower()
    if act == "leaky_relu":
        return lambda: nn.LeakyReLU(negative_slope=0.1)
    elif act == "gelu":
        return nn.GELU
    elif act == "relu":
        return nn.ReLU
    elif act == "elu":
        return nn.ELU
    else:
        return nn.Tanh


from env.observations import OBS_DIM, OBS_MIRROR_MASK_NP, OBS_MIRROR_INDICES_NP, ACT_MIRROR_MASK_NP

try:
    from env.observations import OBS_LEGACY_MIRROR_MASK_NP
except ImportError:
    OBS_LEGACY_MIRROR_MASK_NP = OBS_MIRROR_MASK_NP


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = 8,
        actor_hidden_dims: List[int] = [256, 256, 128],
        critic_hidden_dims: List[int] = [256, 256, 128],
        activation: str = "leaky_relu",
        continuous_actions: bool = True,
        use_layer_norm: bool = True,
        legacy_mirror_mask: bool = False,
        use_action_masking: bool = True,
        handbrake_height_buffer: float = 120.0
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.continuous_actions = continuous_actions
        self.use_layer_norm = use_layer_norm
        self.legacy_mirror_mask = legacy_mirror_mask
        self.use_action_masking = use_action_masking
        self.handbrake_height_buffer = float(handbrake_height_buffer)
        self.height_buffer_norm = self.handbrake_height_buffer / 2044.0

        # Use legacy (pre-fix) mirror mask for evaluating old checkpoints, corrected mask for new training
        mirror_mask_np = OBS_LEGACY_MIRROR_MASK_NP if legacy_mirror_mask else OBS_MIRROR_MASK_NP
        if obs_dim < len(mirror_mask_np):
            mirror_mask_np = mirror_mask_np[:obs_dim]
        mirror_indices_np = OBS_MIRROR_INDICES_NP
        if obs_dim < len(mirror_indices_np):
            mirror_indices_np = np.arange(obs_dim, dtype=int)
        self.register_buffer("obs_mirror_mask", torch.tensor(mirror_mask_np, dtype=torch.float32), persistent=False)
        self.register_buffer("obs_mirror_indices", torch.tensor(mirror_indices_np, dtype=torch.long), persistent=False)
        self.register_buffer("act_mirror_mask", torch.tensor(ACT_MIRROR_MASK_NP, dtype=torch.float32), persistent=False)
        # Calibrated deterministic activation thresholds for binary Bernoulli buttons:
        # Index 0 (Jump): p > 0.15 (logit > -1.7346) - calibrated for deliberate takeoff/dodges without phantom low-speed turn hops
        # Index 1 (Boost): p > 0.25 (logit > -1.0986)
        # Index 2 (Handbrake): p > 0.52 (logit > 0.0800)
        self.register_buffer("bin_thresh_logits", torch.tensor([-1.7346, -1.0986, 0.0800], dtype=torch.float32), persistent=False)

        act_cls = get_activation_cls(activation)

        # Build Actor (Policy Network) with LayerNorm Regularization
        actor_layers = []
        prev_dim = obs_dim
        for hidden in actor_hidden_dims:
            actor_layers.append(layer_init(nn.Linear(prev_dim, hidden)))
            if use_layer_norm:
                actor_layers.append(nn.LayerNorm(hidden))
            actor_layers.append(act_cls())
            prev_dim = hidden
        self.actor_backbone = nn.Sequential(*actor_layers)

        if continuous_actions:
            # 5 Continuous Analog Axes: Throttle (0), Steer (1), Pitch (2), Yaw (3), Roll (4)
            self.actor_mean = layer_init(nn.Linear(prev_dim, 5), std=0.01)
            self.actor_log_std = nn.Parameter(torch.full((1, 5), -1.5))
            # 3 Binary Discrete Bernoulli Buttons: Jump (5), Boost (6), Handbrake (7)
            self.actor_binary = layer_init(nn.Linear(prev_dim, 3), std=0.01)
        else:
            self.actor_logits = layer_init(nn.Linear(prev_dim, act_dim), std=0.01)

        # Build Critic (Value Network) with LayerNorm Regularization
        critic_layers = []
        prev_dim = obs_dim
        for hidden in critic_hidden_dims:
            critic_layers.append(layer_init(nn.Linear(prev_dim, hidden)))
            if use_layer_norm:
                critic_layers.append(nn.LayerNorm(hidden))
            critic_layers.append(act_cls())
            prev_dim = hidden
        critic_layers.append(layer_init(nn.Linear(prev_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

    def debias_symmetric_actions(self):
        """
        Re-centers the actor output layer biases for antisymmetric action axes (steer, yaw, roll)
        to strictly zero, defaults binary button biases (jump, boost, handbrake) to neutral 0.0,
        and bounds actor_log_std within healthy exploration ranges.
        """
        with torch.no_grad():
            if self.continuous_actions and hasattr(self, "actor_mean"):
                # Steer (index 1), Pitch (index 2), Yaw (index 3), Roll (index 4)
                if self.actor_mean.bias is not None:
                    self.actor_mean.bias.data[1] = 0.0
                    self.actor_mean.bias.data[2] = 0.0
                    self.actor_mean.bias.data[3] = 0.0
                    self.actor_mean.bias.data[4] = 0.0

                if hasattr(self, "actor_binary") and self.actor_binary.bias is not None:
                    # Center Jump (0), Boost (1), Handbrake (2) logit biases to neutral (healthy exploration prior)
                    self.actor_binary.bias.data[0] = torch.clamp(self.actor_binary.bias.data[0], min=-0.2, max=0.5)
                    self.actor_binary.bias.data[1] = torch.clamp(self.actor_binary.bias.data[1], min=-0.5, max=0.5)
                    self.actor_binary.bias.data[2] = torch.clamp(self.actor_binary.bias.data[2], min=-0.5, max=0.5)

                if hasattr(self, "actor_log_std") and self.actor_log_std is not None:
                    # Recover from underflow or parameter drift
                    if torch.isnan(self.actor_log_std).any() or (self.actor_log_std.abs() < 1e-6).any() or (self.actor_log_std > -0.5).any():
                        self.actor_log_std.data.fill_(-1.1)
                    else:
                        self.actor_log_std.data.clamp_(min=-2.2, max=-0.7)
                    # Guarantee healthy exploration on Pitch (index 2) to discover forward/diagonal/flick dodges
                    self.actor_log_std.data[0, 2] = max(-1.0, float(self.actor_log_std.data[0, 2]))

                # Desaturate actor_mean weights if they exceeded linear analog range
                weight_norm = self.actor_mean.weight.data.norm(dim=1, keepdim=True)
                max_norm = 1.5
                scale = torch.clamp(max_norm / (weight_norm + 1e-6), max=1.0)
                self.actor_mean.weight.data *= scale

            # Clamp backbone linear weights if they exceeded healthy numerical thresholds
            for module in self.actor_backbone:
                if isinstance(module, nn.Linear):
                    w_norm = module.weight.data.norm(dim=1, keepdim=True)
                    w_max = 3.0
                    w_scale = torch.clamp(w_max / (w_norm + 1e-6), max=1.0)
                    module.weight.data *= w_scale

    def load_state_dict(self, state_dict, strict=True):
        # Auto-migrate legacy 8-channel continuous checkpoints into hybrid 5-continuous + 3-binary heads
        if self.continuous_actions and 'actor_mean.weight' in state_dict and state_dict['actor_mean.weight'].shape[0] == 8:
            new_sd = {}
            for k, v in state_dict.items():
                if k == 'actor_mean.weight':
                    new_sd['actor_mean.weight'] = v[:5]
                    new_sd['actor_binary.weight'] = v[5:]
                elif k == 'actor_mean.bias':
                    new_sd['actor_mean.bias'] = v[:5]
                    new_sd['actor_binary.bias'] = v[5:]
                elif k == 'actor_log_std':
                    new_sd['actor_log_std'] = v[:, :5]
                else:
                    new_sd[k] = v
            return super().load_state_dict(new_sd, strict=strict)
        return super().load_state_dict(state_dict, strict=strict)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        apply_masking: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.actor_backbone(obs)
        value = self.critic(obs)

        if self.continuous_actions:
            if obs.shape[-1] == self.obs_mirror_mask.shape[-1] and self.act_dim == self.act_mirror_mask.shape[-1]:
                # Equivariant Bilateral Symmetry Forward Pass with bilateral pad pair permutation
                if hasattr(self, "obs_mirror_indices") and self.obs_mirror_indices is not None and self.obs_mirror_indices.shape[-1] == obs.shape[-1]:
                    obs_mirr = (obs * self.obs_mirror_mask)[..., self.obs_mirror_indices]
                else:
                    obs_mirr = obs * self.obs_mirror_mask
                feat_mirr = self.actor_backbone(obs_mirr)

                raw_mean = torch.tanh(self.actor_mean(features))
                mirr_mean = torch.tanh(self.actor_mean(feat_mirr)) * self.act_mirror_mask[:5]
                action_mean = 0.5 * (raw_mean + mirr_mean)

                raw_bin = self.actor_binary(features)
                mirr_bin = self.actor_binary(feat_mirr)
                bin_logits = 0.5 * (raw_bin + mirr_bin)
            else:
                action_mean = torch.tanh(self.actor_mean(features))
                bin_logits = self.actor_binary(features)

            clamped_log_std = torch.clamp(self.actor_log_std, min=-2.5, max=-0.7)
            action_std = torch.exp(clamped_log_std).expand_as(action_mean)

            should_mask = self.use_action_masking if apply_masking is None else apply_masking
            if should_mask and obs.shape[-1] >= 22:
                # Step 3: Gated Action Masking with Physical Conditions
                # Indices in self car state (0..21):
                # 2: car_pos.z (normed by 2044.0)
                # 18: boost (normed by 0.01)
                # 19: on_ground (1.0 or 0.0)
                # 20: has_jump (1.0 or 0.0)
                # 21: has_flip (1.0 or 0.0)
                is_grounded = obs[..., 19] > 0.5
                is_airborne = ~is_grounded
                near_ground = obs[..., 2] < self.height_buffer_norm
                can_boost = obs[..., 18] > 0.001
                can_jump = (obs[..., 20] > 0.5) | (obs[..., 21] > 0.5)
                can_slide = is_grounded | near_ground

                # A. Mask Bernoulli Logits (-1e4 forces sigmoid to ~0.0)
                btn_mask = torch.stack([can_jump, can_boost, can_slide], dim=-1)
                masked_bin_logits = torch.where(
                    btn_mask, bin_logits, torch.full_like(bin_logits, -1e4)
                )

                # B. Mask Continuous Means & Collapse Standard Deviations (pure out-of-place construction)
                m0 = torch.where(is_airborne, torch.clamp(action_mean[..., 0], min=0.0), action_mean[..., 0])
                m1 = action_mean[..., 1]
                grd = is_grounded.unsqueeze(-1)
                m234 = torch.where(grd, torch.zeros_like(action_mean[..., 2:5]), action_mean[..., 2:5])
                masked_mean = torch.cat([m0.unsqueeze(-1), m1.unsqueeze(-1), m234], dim=-1)

                s01 = action_std[..., :2]
                s234 = torch.where(grd, torch.full_like(action_std[..., 2:5], 1e-4), action_std[..., 2:5])
                masked_std = torch.cat([s01, s234], dim=-1)

                # Step 4: Distribution Construction
                dist_cont = Normal(masked_mean, masked_std)
                dist_bin = Bernoulli(logits=masked_bin_logits)

                # Step 5: Action Sampling
                if action is None:
                    if deterministic:
                        act_cont = masked_mean
                        thresh = self.bin_thresh_logits.to(masked_bin_logits.device)
                        act_bin = (masked_bin_logits > thresh).float() * 2.0 - 1.0
                    else:
                        raw_act_cont = dist_cont.rsample()
                        # Strictly zero out ground rotational channels to eliminate float subnormal epsilon noise
                        act_c234 = torch.where(grd, torch.zeros_like(raw_act_cont[..., 2:5]), raw_act_cont[..., 2:5])
                        act_cont = torch.cat([raw_act_cont[..., :2], act_c234], dim=-1)
                        raw_act_bin = dist_bin.sample() * 2.0 - 1.0
                        act_bin = torch.where(btn_mask, raw_act_bin, torch.full_like(raw_act_bin, -1.0))
                    action = torch.cat([act_cont, act_bin], dim=-1)
                else:
                    act_cont = action[..., :5]
                    act_bin = action[..., 5:]

                # Exact Log Probabilities across active channels (zeroing masked channels eliminates dead gradient noise)
                bin_binary_01 = (act_bin > 0.0).float()
                raw_cont_logp = dist_cont.log_prob(act_cont)
                c_mask = torch.cat([torch.ones_like(raw_cont_logp[..., :2]), (~is_grounded).float().unsqueeze(-1).expand_as(raw_cont_logp[..., 2:5])], dim=-1)
                cont_logp = raw_cont_logp * c_mask

                raw_bin_logp = dist_bin.log_prob(bin_binary_01)
                bin_logp = raw_bin_logp * btn_mask.float()
                log_prob = cont_logp.sum(dim=-1) + bin_logp.sum(dim=-1)

                # Active Channel Entropy Normalization
                raw_ent_cont = dist_cont.entropy()
                ent_cont = raw_ent_cont * c_mask

                raw_ent_bin = dist_bin.entropy()
                ent_bin = raw_ent_bin * btn_mask.float()

                active_dims = (
                    2.0
                    + is_airborne.float() * 3.0
                    + can_jump.float()
                    + can_boost.float()
                    + can_slide.float()
                )
                # Rescale by (act_dim / active_dims) so standard ppo.py entropy division by act_dim yields exact active-mean entropy
                entropy = (ent_cont.sum(dim=-1) + ent_bin.sum(dim=-1)) * (float(self.act_dim) / active_dims)
                return action, log_prob, entropy, value
            else:
                # Unmasked baseline pass (identical to original behavior)
                dist_cont = Normal(action_mean, action_std)
                dist_bin = Bernoulli(logits=bin_logits)

                if action is None:
                    if deterministic:
                        act_cont = action_mean
                        thresh = self.bin_thresh_logits.to(bin_logits.device)
                        act_bin = (bin_logits > thresh).float() * 2.0 - 1.0
                    else:
                        act_cont = dist_cont.rsample()
                        act_bin = dist_bin.sample() * 2.0 - 1.0
                    action = torch.cat([act_cont, act_bin], dim=-1)
                else:
                    act_cont = action[..., :5]
                    act_bin = action[..., 5:]

                bin_binary_01 = (act_bin > 0.0).float()
                log_prob = dist_cont.log_prob(act_cont).sum(dim=-1) + dist_bin.log_prob(bin_binary_01).sum(dim=-1)
                entropy = dist_cont.entropy().sum(dim=-1) + dist_bin.entropy().sum(dim=-1)
                return action, log_prob, entropy, value
        else:
            logits = self.actor_logits(features)
            dist = Categorical(logits=logits)

            if action is None:
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, log_prob, entropy, value
