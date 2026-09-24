"""
Pre-tanh magnitude penalty on the continuous action head (v11, docs/reward_v11_pretanh_spec.md).

The action mean is 0.5 * (tanh(raw pass) + tanh(mirrored pass) * sign), each pass being
actor_mean(features). Every RL checkpoint since v3 drives the throttle and steer pre-activations
to a median |pre| of 3-5, where d tanh / d pre is 0.0005-0.01, so the policy gradient can no longer
move those channels in most ground states (scripts/action_saturation.py). In 24-28% of grounded
states steer's two passes sit on opposite rails and cancel to ~0, also with no gradient.

This charges each pass's pre-activation on the chosen channels for exceeding a threshold:

    penalty = weight * mean( relu(|pre| - threshold) ** 2 )

over every state in the minibatch, both passes, the chosen channels. Inside the threshold it is
exactly zero, so it never pulls a mean toward 0; it only stops pre-activations drifting into the
region where the gradient is gone. At threshold 2.0 the mean can still reach tanh(2) = 0.964 and
the gradient at the boundary is 0.071.

It is a loss on the policy's output, not a reward: nothing reads the action the env receives
(spec rule R2 is about reward terms), and the returns and advantages are untouched.

Capture works by a forward hook on actor_mean for the duration of one forward pass, so it sees
both passes of the mirrored forward without any change to agent/models.py.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

DEFAULTS = {"pre_tanh_weight": 0.0, "pre_tanh_threshold": 2.0, "pre_tanh_channels": [0, 1]}


class PreTanhPenalty:
    def __init__(self, weight: float, threshold: float, channels: Sequence[int]):
        self.weight = float(weight)
        self.threshold = float(threshold)
        self.channels = list(int(c) for c in channels)

    @classmethod
    def from_config(cls, section: Optional[Mapping[str, object]]) -> Optional["PreTanhPenalty"]:
        """None unless the config's action_regularization section sets a positive weight."""
        cfg = {**DEFAULTS, **dict(section or {})}
        if float(cfg["pre_tanh_weight"]) <= 0.0:
            return None
        return cls(float(cfg["pre_tanh_weight"]), float(cfg["pre_tanh_threshold"]), cfg["pre_tanh_channels"])

    @contextmanager
    def capture(self, head: nn.Module) -> Iterator[List[torch.Tensor]]:
        """Collect every output of `head` (actor_mean) produced inside the block."""
        outs: List[torch.Tensor] = []
        handle = head.register_forward_hook(lambda _m, _i, out: outs.append(out))
        try:
            yield outs
        finally:
            handle.remove()

    def loss(self, outs: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
        """weight * mean squared excess over the threshold, across passes, states and channels."""
        if not outs:
            return None
        pre = torch.cat([o[..., self.channels] for o in outs], dim=0)
        excess = torch.relu(pre.abs() - self.threshold)
        return self.weight * (excess ** 2).mean()

    @torch.no_grad()
    def stats(self, outs: Sequence[torch.Tensor]) -> Dict[str, float]:
        """Share of pre-activations over the threshold and their median magnitude, per channel."""
        out: Dict[str, float] = {}
        if not outs:
            return out
        pre = torch.cat([o[..., self.channels] for o in outs], dim=0).abs()
        out["over_pct"] = float((pre > self.threshold).float().mean() * 100.0)
        for i, ch in enumerate(self.channels):
            out[f"pre_abs_median_{ch}"] = float(pre[:, i].median())
        return out
