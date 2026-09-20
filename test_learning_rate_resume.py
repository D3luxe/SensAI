"""
The configured learning rate survives a resume.

torch's optimizer.load_state_dict restores the learning rate stored in the checkpoint, so resuming
after a config change silently keeps training at the old rate while the history records the new
one. That happened once (run v5, docs/run_v5_lr_spec.md): 16M steps at 3e-4 while every record and
the dashboard said 1.5e-4, and the KL trace was the only evidence.
"""
import unittest

import torch

from agent.ppo import apply_learning_rate


def _optimizer(lr):
    return torch.optim.AdamW(torch.nn.Linear(4, 4).parameters(), lr=lr, eps=1e-5)


class TestLearningRateResume(unittest.TestCase):
    def test_checkpoint_state_carries_its_own_learning_rate(self):
        """The behaviour being guarded against: without a re-assert, the old rate wins."""
        old, new = _optimizer(3e-4), _optimizer(1.5e-4)
        new.load_state_dict(old.state_dict())
        self.assertAlmostEqual(new.param_groups[0]["lr"], 3e-4)

    def test_apply_learning_rate_restores_the_configured_rate(self):
        old, new = _optimizer(3e-4), _optimizer(1.5e-4)
        new.load_state_dict(old.state_dict())
        previous = apply_learning_rate(new, 1.5e-4)
        self.assertAlmostEqual(previous, 3e-4)
        for group in new.param_groups:
            self.assertAlmostEqual(group["lr"], 1.5e-4)

    def test_it_reports_no_change_when_the_rates_match(self):
        opt = _optimizer(1.5e-4)
        self.assertAlmostEqual(apply_learning_rate(opt, 1.5e-4), 1.5e-4)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1.5e-4)

    def test_every_param_group_is_set(self):
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW([{"params": [model.weight], "lr": 3e-4},
                                 {"params": [model.bias], "lr": 1e-3}], eps=1e-5)
        apply_learning_rate(opt, 1.5e-4)
        self.assertEqual([g["lr"] for g in opt.param_groups], [1.5e-4, 1.5e-4])

    def test_the_trainer_reasserts_it_after_loading_optimizer_state(self):
        """load_checkpoint must call it: the regression is one missing line."""
        import inspect
        from agent.ppo import PPOTrainer
        src = inspect.getsource(PPOTrainer.load_checkpoint)
        self.assertIn("apply_learning_rate", src)
        self.assertLess(src.index("optimizer.load_state_dict"), src.index("apply_learning_rate"))


if __name__ == "__main__":
    unittest.main()
