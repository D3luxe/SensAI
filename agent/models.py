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


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = 64,
        act_dim: int = 8,
        actor_hidden_dims: List[int] = [256, 256, 128],
        critic_hidden_dims: List[int] = [256, 256, 128],
        activation: str = "tanh",
        continuous_actions: bool = True
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.continuous_actions = continuous_actions

        act_cls = nn.Tanh if activation == "tanh" else (nn.ReLU if activation == "relu" else nn.ELU)

        # Build Actor (Policy Network)
        actor_layers = []
        prev_dim = obs_dim
        for hidden in actor_hidden_dims:
            actor_layers.append(layer_init(nn.Linear(prev_dim, hidden)))
            actor_layers.append(act_cls())
            prev_dim = hidden
        self.actor_backbone = nn.Sequential(*actor_layers)

        if continuous_actions:
            self.actor_mean = layer_init(nn.Linear(prev_dim, act_dim), std=0.01)
            # Log std parameter for Gaussian policy
            self.actor_log_std = nn.Parameter(torch.zeros(1, act_dim))
        else:
            self.actor_logits = layer_init(nn.Linear(prev_dim, act_dim), std=0.01)

        # Build Critic (Value Network)
        critic_layers = []
        prev_dim = obs_dim
        for hidden in critic_hidden_dims:
            critic_layers.append(layer_init(nn.Linear(prev_dim, hidden)))
            critic_layers.append(act_cls())
            prev_dim = hidden
        critic_layers.append(layer_init(nn.Linear(prev_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

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
            action_log_std = self.actor_log_std.expand_as(action_mean)
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
