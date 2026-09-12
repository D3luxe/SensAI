"""Regression tests for TimeCostReward.

Every shaping term here is potential-based, which makes it unfarmable and also makes it blind
to time. Measured directly: the same approach took 27.9 steps at full throttle with boost and
paid 4.3688 of player_to_ball, and 48.0 steps at half throttle and paid 4.3912. Arriving 1.3
seconds later paid slightly more. So nothing whose only benefit is saving time -- half-flip,
powerslide turn, preserving 400 uu/s through a crooked landing -- could be worth a point.

This term prices time once, globally, instead of pricing each mechanic separately.

What these tests protect:
  1. It is flat and unconditional. Any state where stalling is free defeats the purpose.
  2. It can only ever subtract.
  3. It stays small enough that ending an episode by conceding is never worth the time saved.
"""

import unittest

import numpy as np

from env.physics_engine import CarState, BallState
from env.rewards import TimeCostReward, CombinedReward

W = 0.01


class MockArena:
    def __init__(self, cars, ball_pos=(0.0, 1000.0, 93.0)):
        self.ball = BallState(pos=np.array(ball_pos, dtype=np.float32))
        self.cars = cars
        self.boost_pads = []


def _car(**kw):
    c = CarState(id=0, team=0, pos=np.array(kw.get("pos", [0.0, 0.0, 17.0]),
                                            dtype=np.float32))
    c.on_ground = kw.get("on_ground", True)
    c.vel = np.array(kw.get("vel", [0.0, 0.0, 0.0]), dtype=np.float32)
    c.boost = kw.get("boost", 33.0)
    return c


def _r(weight=W, is_goal=False, **kw):
    c = _car(**kw)
    r = TimeCostReward(weight=weight)
    arena = MockArena([c], kw.get("ball_pos", (0.0, 1000.0, 93.0)))
    r.reset(arena)
    return r.get_reward(c, arena, np.zeros(8, dtype=np.float32), is_goal, None)


class TestFlatAndUnconditional(unittest.TestCase):
    def test_same_cost_in_every_state(self):
        """A gate would create a region where stalling is free, which is the behaviour
        this exists to price."""
        states = [
            {"on_ground": True, "vel": [0.0, 0.0, 0.0]},
            {"on_ground": False, "vel": [0.0, 2000.0, 500.0]},
            {"on_ground": True, "vel": [-1500.0, 0.0, 0.0]},
            {"pos": [0.0, -5000.0, 17.0]},
            {"ball_pos": (0.0, 60.0, 93.0)},
            {"ball_pos": (0.0, 5000.0, 1800.0)},
            {"boost": 0.0},
            {"boost": 100.0},
        ]
        for kw in states:
            self.assertAlmostEqual(_r(**kw), -W, places=9,
                                   msg="cost must not vary with state: %s" % kw)

    def test_charged_on_the_terminal_step_too(self):
        self.assertAlmostEqual(_r(is_goal=True), -W, places=9)

    def test_scales_linearly_with_weight(self):
        for w in (0.0025, 0.005, 0.01, 0.02, 0.05):
            self.assertAlmostEqual(_r(weight=w), -w, places=9)

    def test_weight_zero_disables_it(self):
        self.assertEqual(_r(weight=0.0), 0.0)


class TestCannotBeFarmed(unittest.TestCase):
    def test_never_returns_a_positive_value(self):
        rng = np.random.default_rng(0)
        r = TimeCostReward(weight=W)
        for _ in range(300):
            c = _car(on_ground=bool(rng.integers(2)),
                     vel=list(rng.uniform(-2300, 2300, 3)),
                     pos=list(rng.uniform(-4000, 4000, 3)))
            arena = MockArena([c])
            r.reset(arena)
            self.assertLessEqual(
                r.get_reward(c, arena, np.zeros(8, dtype=np.float32), False, None), 0.0)


class TestConcedeStaysWorse(unittest.TestCase):
    """A living cost gives the policy a reason to end the episode, and conceding ends it.
    What keeps that safe is arithmetic, not intent, so pin the arithmetic."""

    def _weights(self):
        import io
        import json
        import yaml
        cfg = yaml.safe_load(io.open("config/default_config.yaml", encoding="utf-8").read())
        live = json.load(io.open("config/live_config.json", encoding="utf-8"))
        rw = dict(cfg.get("rewards", {}))
        rw.update(live.get("rewards", {}))
        return rw

    def test_a_whole_episode_of_time_costs_far_less_than_one_concede(self):
        rw = self._weights()
        w = float(rw["time_cost_weight"])
        concede = abs(float(rw["concede_weight"]))
        worst_case = w * 600.0        # the longest an episode can run
        self.assertLess(worst_case, concede / 4.0,
                        "time_cost %.4f over a full episode is %.2f against a %.1f concede; "
                        "at this ratio throwing a goal to end the episode early becomes "
                        "attractive" % (w, worst_case, concede))

    def test_configured_weight_is_in_the_sane_band(self):
        w = float(self._weights()["time_cost_weight"])
        self.assertGreater(w, 0.0)
        self.assertLessEqual(w, 0.05, "above this the living cost rivals the objective terms")


class TestWiring(unittest.TestCase):
    def test_registered(self):
        self.assertIn("time_cost", CombinedReward({}).rewards)

    def test_live_tunable(self):
        c = CombinedReward({"time_cost_weight": 0.02})
        self.assertAlmostEqual(c.rewards["time_cost"].weight, 0.02)
        c.update_weights({"time_cost_weight": 0.0})
        self.assertAlmostEqual(c.rewards["time_cost"].weight, 0.0)

    def test_both_configs_carry_it(self):
        import io
        import json
        import yaml
        cfg = yaml.safe_load(io.open("config/default_config.yaml", encoding="utf-8").read())
        live = json.load(io.open("config/live_config.json", encoding="utf-8"))
        self.assertIn("time_cost_weight", cfg["rewards"])
        self.assertIn("time_cost_weight", live["rewards"])
        self.assertAlmostEqual(cfg["rewards"]["time_cost_weight"],
                               live["rewards"]["time_cost_weight"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
