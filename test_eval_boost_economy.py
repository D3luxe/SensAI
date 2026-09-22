"""
The eval suite's boost-economy metrics (scripts/eval_suite.py MotionStats), added with reward v7.

Fed a scripted sequence of car states, the tracker must count pads by size, split spending by
supersonic and airborne, and catch retreats that begin with too little boost to burn.
"""
import importlib.util
import os
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("eval_suite", os.path.join(ROOT, "scripts", "eval_suite.py"))
eval_suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_suite)


def arena(boost, speed=1000.0, car_y=-2000.0, on_ground=True, ball_y=3000.0, ball_vy=0.0):
    car = SimpleNamespace(pos=np.array([0.0, car_y, 17.0]), vel=np.array([0.0, speed, 0.0]), boost=boost,
                          on_ground=on_ground, get_up_vector=lambda: np.array([0.0, 0.0, 1.0]))
    ball = SimpleNamespace(pos=np.array([0.0, ball_y, 93.0]), vel=np.array([0.0, ball_vy, 0.0]))
    return SimpleNamespace(cars=[car], ball=ball)


class TestBoostEconomy(unittest.TestCase):
    def run_steps(self, states):
        m = eval_suite.MotionStats()
        for t, kw in enumerate(states):
            m(t, arena(**kw))
        return m, m.summary()

    def test_pads_are_counted_by_size(self):
        # 20 -> small (+12) -> big (to 100) -> burning while collecting a small pad nets +9.8: still small
        m, s = self.run_steps([dict(boost=20.0), dict(boost=32.0), dict(boost=100.0),
                               dict(boost=90.0), dict(boost=99.8)])
        self.assertEqual((m.small_pads, m.big_pads), (2, 1))
        self.assertAlmostEqual(m.collected, 12.0 + 68.0 + 9.8, places=4)
        self.assertAlmostEqual(s["boost_collected_per_min"], m.collected / (5 / eval_suite.STEPS_PER_MIN))

    def test_spending_is_split_by_supersonic_and_airborne(self):
        m, s = self.run_steps([dict(boost=50.0, speed=2250.0), dict(boost=46.0, speed=2280.0),   # 4 supersonic
                               dict(boost=42.0, speed=1500.0),                                  # 4 supersonic (was, last step)
                               dict(boost=40.0, speed=1600.0, on_ground=False)])                # 2 airborne
        self.assertAlmostEqual(m.spent, 10.0, places=4)
        self.assertAlmostEqual(s["boost_spent_supersonic_pct"], 80.0, places=3)
        self.assertAlmostEqual(s["boost_spent_air_pct"], 20.0, places=3)

    def test_retreats_begun_on_low_boost(self):
        # A retreat: the car upfield of a ball in our half. Two retreats, one begun on 5 boost.
        ret = dict(car_y=2000.0, ball_y=-1000.0)
        calm = dict(car_y=-3000.0, ball_y=2000.0)
        m, s = self.run_steps([dict(boost=5.0, **calm), dict(boost=5.0, **ret), dict(boost=5.0, **ret),
                               dict(boost=40.0, **calm), dict(boost=40.0, **ret), dict(boost=40.0, **ret)])
        self.assertEqual((m.retreat_starts, m.retreat_starts_low), (2, 1))
        self.assertAlmostEqual(s["retreat_starts_low_boost_pct"], 50.0)

    def test_a_goal_reset_is_not_a_pickup(self):
        m = eval_suite.MotionStats()
        m(0, arena(boost=5.0))
        m.episode_end()                     # goal: the kickoff refills to 33
        m(1, arena(boost=33.3))
        self.assertEqual((m.small_pads, m.big_pads, m.collected), (0, 0, 0.0))

    def test_new_metrics_have_a_direction_and_a_headline(self):
        from utils.eval_results import HEADLINE, better_direction
        keys = ("boost_collected_per_min", "small_pads_per_min", "big_pads_per_min",
                "boost_spent_supersonic_pct", "retreat_starts_low_boost_pct")
        self.assertEqual([better_direction(k) for k in keys], [1, 1, 1, -1, -1])
        self.assertTrue(set(keys) <= {k for _, k, _, _ in HEADLINE["Boost economy"]})


if __name__ == "__main__":
    unittest.main()
