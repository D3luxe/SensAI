"""
Actor-Critic Neural Network Architecture for Rocket League PPO Agents.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
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


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = 64,
        act_dim: int = 8,
        actor_hidden_dims: List[int] = [256, 256, 128],
        critic_hidden_dims: List[int] = [256, 256, 128],
        activation: str = "leaky_relu",
        continuous_actions: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.continuous_actions = continuous_actions
        self.use_layer_norm = use_layer_norm

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
            self.actor_mean = layer_init(nn.Linear(prev_dim, act_dim), std=0.01)
            # Log std parameter for Gaussian policy: initialized to -1.5 (std ~ 0.22 for fine analog vehicle exploration)
            self.actor_log_std = nn.Parameter(torch.full((1, act_dim), -1.5))
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
        to strictly zero, defaults handbrake bias to negative (OFF), and prevents output tanh saturation.
        """
        with torch.no_grad():
            if self.continuous_actions and hasattr(self, "actor_mean"):
                # Steer (index 1), Pitch (index 2), Yaw (index 3), Roll (index 4)
                if self.actor_mean.bias is not None:
                    self.actor_mean.bias.data[1] = 0.0
                    self.actor_mean.bias.data[2] = 0.0
                    self.actor_mean.bias.data[3] = 0.0
                    self.actor_mean.bias.data[4] = 0.0
                    # Default Handbrake (index 7) bias to -2.0 (OFF unless deliberately triggered)
                    self.actor_mean.bias.data[7] = -2.0

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

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.actor_backbone(obs)
        value = self.critic(obs)

        if self.continuous_actions:
            action_mean = torch.tanh(self.actor_mean(features))
            # Clamp log_std to [-2.5, -1.2] so exploration std is bounded within [0.08, 0.30] (optimal vehicle control window)
            clamped_log_std = torch.clamp(self.actor_log_std, min=-2.5, max=-1.2)
            action_log_std = clamped_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            dist = Normal(action_mean, action_std)

            if action is None:
                if deterministic:
                    action = action_mean
                else:
                    action = dist.rsample()
            
            # Sum log probs across action dimensions
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
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
