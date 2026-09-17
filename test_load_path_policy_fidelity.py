"""
Regression tests for policy fidelity across checkpoint loads.

bot.py and the trainer's resume path used to call ActorCritic.debias_symmetric_actions() on every
load, which capped actor row norms and zeroed/clamped output biases. At iter 116k that shifted
deterministic throttle by 0.18 on average and flipped ~2% of button decisions, so the in-game bot
played a different policy than training, league and TrueSkill evaluated, and every resumed run
restarted from a perturbed one.

Guarantees:
  1. sanitize_log_std() changes nothing except actor_log_std.
  2. A trained-scale policy produces identical deterministic actions after sanitize_log_std().
  3. debias_symmetric_actions() is still destructive (so the distinction is real).
  4. Neither bot.py nor the trainer's load path calls debias_symmetric_actions().
"""

import io
import re
import unittest

import torch

from agent.models import ActorCritic
from env.observations import OBS_DIM


def trained_scale_model(seed=0):
    """A policy with weight norms and biases in the range a long run reaches (rows ~2.5-4.6)."""
    # Seeded inside fork_rng so the global generator other test modules rely on is left alone
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = ActorCritic(obs_dim=OBS_DIM, act_dim=8, continuous_actions=True)
    with torch.no_grad():
        for module in model.actor_backbone:
            if isinstance(module, torch.nn.Linear):
                module.weight.mul_(4.0 / module.weight.norm(dim=1, keepdim=True))
        model.actor_mean.weight.mul_(2.6 / model.actor_mean.weight.norm(dim=1, keepdim=True))
        model.actor_mean.bias.copy_(torch.tensor([-0.09, -0.02, 0.09, -0.05, -0.03]))
        model.actor_binary.bias.copy_(torch.tensor([-0.24, -0.49, -0.70]))
    model.eval()
    return model


def deterministic_actions(model, obs):
    with torch.no_grad():
        return model.get_action_and_value(obs, deterministic=True)[0]


class TestLoadPathPolicyFidelity(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(1)
        self.obs = torch.randn(512, OBS_DIM, generator=gen) * 0.5

    def test_sanitize_touches_only_log_std(self):
        model = trained_scale_model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        model.sanitize_log_std()
        for k, v in model.state_dict().items():
            if k == "actor_log_std":
                continue
            self.assertTrue(torch.equal(before[k], v), f"sanitize_log_std changed {k}")

    def test_sanitize_preserves_deterministic_actions(self):
        model = trained_scale_model()
        before = deterministic_actions(model, self.obs)
        model.sanitize_log_std()
        self.assertTrue(torch.equal(before, deterministic_actions(model, self.obs)))

    def test_debias_is_destructive(self):
        model = trained_scale_model()
        before = deterministic_actions(model, self.obs)
        model.debias_symmetric_actions()
        after = deterministic_actions(model, self.obs)
        self.assertGreater(float((before[:, :5] - after[:, :5]).abs().mean()), 0.01)

    def test_load_paths_do_not_debias(self):
        for path in ("bot.py", "agent/ppo.py", "env/baseline_agent.py", "utils/visualizer.py"):
            with io.open(path, encoding="utf-8") as fh:
                calls = re.findall(r"\.debias_symmetric_actions\(\)", fh.read())
            self.assertEqual(calls, [], f"{path} calls debias_symmetric_actions() on a load path")


if __name__ == "__main__":
    unittest.main()
