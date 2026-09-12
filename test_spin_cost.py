"""Regression tests for SpinCostReward.

Pitch, yaw and roll carried no price in either direction. Measured from iteration 7780 to
9700 the policy held mean |pitch| 0.72 (37% of airborne steps past 0.9) and the car spun at
3.66 rad/s against a 5.5 rad/s cap, with 75% of airborne steps above 2 rad/s. Replaying the
same policy with the rotational channels forced to zero produced 51% more touches and 154%
more ball progression, so the learned air control was worse than none.

What these tests protect:
  1. Only rotation ABOVE the deadband is charged, so ordinary air control stays free.
  2. The charge grows with the excess and saturates at the weight, so one bad step cannot
     dominate an episode.
  3. Grounded steps are never charged -- the rotational channels are masked there anyway.
  4. Dodges are exempt, so flips are not priced twice (JumpCostReward already charges once).
  5. It can only ever subtract.
"""

import unittest

import numpy as np

from env.physics_engine import CarState, BallState
from env.rewards import SpinCostReward, CombinedReward

W = 0.03
DEAD = 2.0
CAP = 5.5


class MockArena:
    def __init__(self, cars):
        self.ball = BallState(pos=np.array([0.0, 1000.0, 93.0], dtype=np.float32))
        self.cars = cars
        self.boost_pads = []


def _car(ang_speed, on_ground=False, dodging=False):
    c = CarState(id=0, team=0, pos=np.array([0.0, 0.0, 300.0], dtype=np.float32))
    c.on_ground = on_ground
    # Spread across all three axes: the term prices the vector magnitude, not any one channel.
    v = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    c.ang_vel = (v / np.linalg.norm(v) * ang_speed).astype(np.float32)
    c.is_dodging = dodging
    c.just_dodged = False
    return c


def _r(ang_speed, on_ground=False, dodging=False, weight=W, deadband=DEAD):
    c = _car(ang_speed, on_ground, dodging)
    rew = SpinCostReward(weight=weight, deadband=deadband)
    rew.reset(MockArena([c]))
    return rew.get_reward(c, MockArena([c]), np.zeros(8, dtype=np.float32), False, None)


class TestDeadband(unittest.TestCase):
    def test_ordinary_air_control_is_free(self):
        for s in (0.0, 0.5, 1.0, 1.5, 1.99):
            self.assertEqual(_r(s), 0.0,
                             "rotation below the deadband is deliberate control, not a tumble")

    def test_charge_begins_exactly_at_the_deadband(self):
        self.assertEqual(_r(DEAD), 0.0)
        self.assertLess(_r(DEAD + 0.01), 0.0)

    def test_no_cliff_at_the_boundary(self):
        """The charge must ramp from zero, not step. Cliffs are the defect class that produced
        most of the trouble in this reward set."""
        self.assertAlmostEqual(_r(DEAD + 0.001), 0.0, places=4)


class TestMagnitude(unittest.TestCase):
    def test_charge_grows_with_excess(self):
        vals = [_r(s) for s in (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)]
        for a, b in zip(vals, vals[1:]):
            self.assertLess(b, a, "faster spin must cost strictly more")

    def test_saturates_at_the_weight(self):
        self.assertAlmostEqual(_r(CAP), -W, places=6)
        # Beyond the physics cap the charge must not keep growing.
        self.assertAlmostEqual(_r(CAP * 3), -W, places=6)

    def test_halfway_costs_half(self):
        mid = DEAD + (CAP - DEAD) / 2.0
        self.assertAlmostEqual(_r(mid), -W / 2.0, places=6)

    def test_live_policy_mean_spin(self):
        """3.66 rad/s was the measured mean; record what it actually costs per step."""
        self.assertAlmostEqual(_r(3.66), -W * (3.66 - DEAD) / (CAP - DEAD), places=6)


class TestExemptions(unittest.TestCase):
    def test_grounded_is_never_charged(self):
        self.assertEqual(_r(5.0, on_ground=True), 0.0,
                         "pitch/yaw/roll are masked to zero on the ground")

    def test_dodge_is_exempt(self):
        self.assertEqual(_r(5.0, dodging=True), 0.0,
                         "a dodge is a scripted rotation; JumpCostReward already charged for it")

    def test_just_dodged_is_exempt(self):
        c = _car(5.0)
        c.just_dodged = True
        rew = SpinCostReward(weight=W, deadband=DEAD)
        rew.reset(MockArena([c]))
        self.assertEqual(rew.get_reward(c, MockArena([c]), np.zeros(8, dtype=np.float32),
                                        False, None), 0.0)

    def test_weight_zero_disables_it(self):
        self.assertEqual(_r(5.5, weight=0.0), 0.0)


class TestCannotBeFarmed(unittest.TestCase):
    def test_never_returns_a_positive_value(self):
        rng = np.random.default_rng(0)
        for _ in range(400):
            v = _r(float(rng.uniform(0.0, 8.0)),
                   on_ground=bool(rng.integers(2)), dodging=bool(rng.integers(2)))
            self.assertLessEqual(v, 0.0, "this term must only ever subtract")

    def test_all_three_axes_priced_together(self):
        """Pricing pitch alone would invite the same tumble rebuilt from yaw and roll."""
        out = []
        for axis in range(3):
            c = _car(0.0)
            v = np.zeros(3, dtype=np.float32)
            v[axis] = 4.0
            c.ang_vel = v
            rew = SpinCostReward(weight=W, deadband=DEAD)
            rew.reset(MockArena([c]))
            out.append(rew.get_reward(c, MockArena([c]), np.zeros(8, dtype=np.float32),
                                      False, None))
        self.assertTrue(all(abs(x - out[0]) < 1e-9 for x in out),
                        "every rotation axis must cost the same")
        self.assertLess(out[0], 0.0)


class TestWiring(unittest.TestCase):
    def test_registered(self):
        c = CombinedReward({})
        self.assertIn("spin_cost", c.rewards)

    def test_live_tunable(self):
        c = CombinedReward({"spin_cost_weight": 0.05, "spin_cost_deadband": 1.5})
        self.assertAlmostEqual(c.rewards["spin_cost"].weight, 0.05)
        self.assertAlmostEqual(c.rewards["spin_cost"].deadband, 1.5)
        c.update_weights({"spin_cost_weight": 0.0})
        self.assertAlmostEqual(c.rewards["spin_cost"].weight, 0.0)

    def test_both_configs_carry_it(self):
        import io
        import json
        import yaml
        cfg = yaml.safe_load(io.open("config/default_config.yaml", encoding="utf-8").read())
        live = json.load(io.open("config/live_config.json", encoding="utf-8"))
        for key in ("spin_cost_weight", "spin_cost_deadband"):
            self.assertIn(key, cfg["rewards"])
            self.assertIn(key, live["rewards"],
                          "live_config overrides the yaml at runtime; a weight missing here is "
                          "silently the class default")
            self.assertAlmostEqual(cfg["rewards"][key], live["rewards"][key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
