"""
Reward v9 (env/rewards_v9.py) against its specification, docs/reward_v9_boost_spec.md.

  - T8 boost_edge: sqrt(self/100) - sqrt(opp/100), paid every step, no gamma, no goal-step zeroing
  - it is ZERO-SUM across the two cars on every step -- the property that stops a dense always-on
    bonus from becoming a survival bonus, and the reason it is a differential at all
  - no boost cycle pays: a trajectory returning to its starting tank has earned nothing, which is
    exactly the v8 failure it replaces
  - concavity prices the level, not the refill: holding 30 is worth more than half of holding 100
  - T7 is retired at weight 0, and T1-T5 are v5's terms unchanged
  - the reward never depends on the action (R2)
  - the registry builds v9 at v5's gamma with its frozen weights, on v8's scenario settings
  - the per-step weight is calibrated to tick_skip 8; the config still says 8
"""
import io
import math
import unittest
from types import SimpleNamespace

import numpy as np

from env.rewards_v3 import REWARD_V3_DEFAULTS, RewardManagerV3
from env.rewards_v8 import RewardManagerV8
from env.rewards_v9 import REWARD_V9_DEFAULTS, RewardManagerV9, TERM_NAMES, boost_edge

GAMMA = 0.9977
EDGE_W = 0.01


def car(cid, team, pos, boost=33.0, touches=0):
    return SimpleNamespace(id=cid, team=team, pos=np.array(pos, dtype=np.float32), boost=boost, ball_touches=touches)


def arena(ball_pos, ball_vel=(0.0, 0.0, 0.0), cars=()):
    return SimpleNamespace(ball=SimpleNamespace(pos=np.array(ball_pos, dtype=np.float32),
                                                vel=np.array(ball_vel, dtype=np.float32)), cars=list(cars))


def manager(**w):
    return RewardManagerV9({**REWARD_V9_DEFAULTS, **w})


def phi(b):
    return math.sqrt(max(0.0, min(100.0, b)) / 100.0)


def duel(boost_pairs, weight=EDGE_W):
    """
    Drive a blue car and an orange car through a sequence of (blue boost, orange boost) readings.

    Returns [(blue boost_edge, orange boost_edge)] per step, with every other term switched off.
    """
    m = manager(boost_edge_weight=weight, ball_position_weight=0.0, touch_weight=0.0)
    b0, o0 = boost_pairs[0]
    m.reset(arena((0.0, 0.0, 93.0), cars=[car(1, 0, (0.0, -1000.0, 17.0), boost=b0),
                                          car(2, 1, (0.0, 1000.0, 17.0), boost=o0)]))
    out = []
    for b, o in boost_pairs[1:]:
        blue, orange = car(1, 0, (0.0, -1000.0, 17.0), boost=b), car(2, 1, (0.0, 1000.0, 17.0), boost=o)
        ar = arena((0.0, 0.0, 93.0), cars=[blue, orange])
        _, tb = m.get_reward(blue, ar, np.zeros(8), False, None)
        _, to = m.get_reward(orange, ar, np.zeros(8), False, None)
        out.append((tb["boost_edge"], to["boost_edge"]))
    return out


class TestBoostEdge(unittest.TestCase):
    def test_pays_the_difference_of_the_v5_potentials(self):
        ((blue, _),) = duel([(0.0, 0.0), (100.0, 25.0)])
        self.assertAlmostEqual(blue, EDGE_W * (phi(100.0) - phi(25.0)), places=12)

    def test_is_zero_sum_across_the_two_cars_every_step(self):
        """The reason T8 is a differential: a dense per-step bonus that sums to zero cannot pay
        for extending an episode."""
        for blue, orange in duel([(0.0, 100.0), (100.0, 0.0), (40.0, 65.0), (12.0, 12.0), (0.0, 0.0)]):
            self.assertAlmostEqual(blue + orange, 0.0, places=12)

    def test_equal_tanks_pay_nothing(self):
        for blue, orange in duel([(50.0, 50.0), (0.0, 0.0), (100.0, 100.0), (33.0, 33.0)]):
            self.assertAlmostEqual(blue, 0.0, places=12)
            self.assertAlmostEqual(orange, 0.0, places=12)

    def test_no_boost_cycle_pays(self):
        """The v8 failure, directly: collect -> dump -> collect earns nothing once the term reads
        the level instead of the change. The opponent is held fixed so only the cycle is priced."""
        cycle = duel([(50.0, 50.0), (100.0, 50.0), (0.0, 50.0), (100.0, 50.0), (0.0, 50.0), (50.0, 50.0)])
        first, last = cycle[0][0], cycle[-1][0]
        self.assertGreater(first, 0.0)          # holding more than the opponent does pay
        self.assertAlmostEqual(last, 0.0, places=12)   # and returning to level pays nothing again

    def test_running_empty_is_punished_not_rewarded(self):
        """v8's shape made being empty the most valuable place to be; v9 inverts that."""
        (empty,), (full,) = duel([(50.0, 50.0), (0.0, 50.0)]), duel([(50.0, 50.0), (100.0, 50.0)])
        self.assertLess(empty[0], 0.0)
        self.assertGreater(full[0], 0.0)

    def test_concavity_prices_the_level(self):
        """Holding 30 against an empty opponent is worth more than half of holding 100."""
        ((low, _),) = duel([(0.0, 0.0), (30.0, 0.0)])
        ((high, _),) = duel([(0.0, 0.0), (100.0, 0.0)])
        self.assertGreater(low, 0.5 * high)
        self.assertAlmostEqual(low, EDGE_W * phi(30.0), places=12)

    def test_the_maximum_edge_is_the_weight(self):
        ((blue, orange),) = duel([(50.0, 50.0), (100.0, 0.0)])
        self.assertAlmostEqual(blue, EDGE_W, places=12)
        self.assertAlmostEqual(orange, -EDGE_W, places=12)

    def test_weight_zero_disables_it(self):
        for blue, orange in duel([(0.0, 100.0), (100.0, 0.0)], weight=0.0):
            self.assertEqual(blue, 0.0)
            self.assertEqual(orange, 0.0)

    def test_it_is_paid_on_the_goal_step(self):
        """T8 is not a potential, so R4's phi(terminal) = 0 does not apply to it."""
        m = manager(ball_position_weight=0.0, touch_weight=0.0)
        cars = [car(1, 0, (0.0, -1000.0, 17.0), boost=100.0), car(2, 1, (0.0, 1000.0, 17.0), boost=0.0)]
        m.reset(arena((0.0, 0.0, 93.0), cars=cars))
        _, terms = m.get_reward(cars[0], arena((0.0, 5200.0, 93.0), cars=cars), np.zeros(8), True, 0)
        self.assertAlmostEqual(terms["boost_edge"], EDGE_W, places=12)

    def test_with_no_opponent_there_is_no_edge(self):
        """Falling back to the car's own level would make it a survival bonus."""
        solo = car(1, 0, (0.0, -1000.0, 17.0), boost=100.0)
        self.assertEqual(boost_edge(solo, arena((0.0, 0.0, 93.0), cars=[solo])), 0.0)

    def test_it_is_a_pure_function_of_the_current_state(self):
        """No per-car history, so it cannot be farmed by a path and needs no reset seeding."""
        blue, orange = car(1, 0, (0.0, -1000.0, 17.0), boost=80.0), car(2, 1, (0.0, 1000.0, 17.0), boost=20.0)
        ar = arena((0.0, 0.0, 93.0), cars=[blue, orange])
        direct = boost_edge(blue, ar)
        m = manager(ball_position_weight=0.0, touch_weight=0.0)
        m.reset(arena((0.0, 0.0, 93.0), cars=[car(1, 0, (0.0, 0.0, 17.0), boost=0.0),
                                              car(2, 1, (0.0, 0.0, 17.0), boost=100.0)]))
        _, terms = m.get_reward(blue, ar, np.zeros(8), False, None)
        self.assertAlmostEqual(terms["boost_edge"], EDGE_W * direct, places=12)


class TestV8TermsRetired(unittest.TestCase):
    def test_t7_is_off_so_a_pickup_pays_nothing(self):
        m = manager(ball_position_weight=0.0, touch_weight=0.0, boost_edge_weight=0.0)
        cars0 = [car(1, 0, (0.0, -1000.0, 17.0), boost=0.0), car(2, 1, (0.0, 1000.0, 17.0), boost=0.0)]
        m.reset(arena((0.0, 0.0, 93.0), cars=cars0))
        cars1 = [car(1, 0, (0.0, -1000.0, 17.0), boost=100.0), car(2, 1, (0.0, 1000.0, 17.0), boost=0.0)]
        _, terms = m.get_reward(cars1[0], arena((0.0, 0.0, 93.0), cars=cars1), np.zeros(8), False, None)
        self.assertEqual(terms["boost_gain"], 0.0)

    def test_with_t8_off_and_t7_restored_v9_pays_exactly_v8(self):
        w = {**REWARD_V3_DEFAULTS, "closeness_weight": 0.0, "boost_weight": 0.0, "gamma": GAMMA}
        v8 = RewardManagerV8({**w, "boost_gain_weight": 2.0})
        v9 = RewardManagerV9({**w, "boost_gain_weight": 2.0, "boost_edge_weight": 0.0})
        cars = [car(1, 0, (0.0, -1000.0, 17.0), boost=40.0), car(2, 1, (200.0, 800.0, 17.0), boost=70.0)]
        start = arena((0.0, 0.0, 93.0), cars=cars)
        v8.reset(start)
        v9.reset(start)
        for step, (bp, bv, boost, touches, goal) in enumerate([
                ((100.0, 500.0, 120.0), (300.0, 900.0, 40.0), 55.0, 1, False),
                ((400.0, 2000.0, 300.0), (800.0, 1500.0, -60.0), 20.0, 1, False),
                ((0.0, 5200.0, 200.0), (100.0, 2000.0, 0.0), 95.0, 2, True)]):
            c = car(1, 0, (0.0, -900.0 + 400 * step, 17.0), boost=boost, touches=touches)
            ar = arena(bp, bv, cars=[c, cars[1]])
            t8, _ = v8.get_reward(c, ar, np.zeros(8), goal, 0 if goal else None)
            t9, _ = v9.get_reward(c, ar, np.zeros(8), goal, 0 if goal else None)
            self.assertAlmostEqual(t8, t9, places=9, msg=f"step {step}")

    def test_with_both_boost_terms_off_and_t5_restored_v9_pays_exactly_v5(self):
        w5 = {**REWARD_V3_DEFAULTS, "closeness_weight": 0.0, "gamma": GAMMA}
        v5 = RewardManagerV3(dict(w5))
        v9 = RewardManagerV9({**w5, "boost_weight": 1.0, "boost_gain_weight": 0.0, "boost_edge_weight": 0.0})
        cars = [car(1, 0, (0.0, -1000.0, 17.0), boost=40.0), car(2, 1, (200.0, 800.0, 17.0), boost=70.0)]
        start = arena((0.0, 0.0, 93.0), cars=cars)
        v5.reset(start)
        v9.reset(start)
        for step, (bp, boost, touches, goal) in enumerate([
                ((100.0, 500.0, 120.0), 55.0, 1, False),
                ((400.0, 2000.0, 300.0), 20.0, 1, False),
                ((0.0, 5200.0, 200.0), 95.0, 2, True)]):
            c = car(1, 0, (0.0, -900.0 + 400 * step, 17.0), boost=boost, touches=touches)
            ar = arena(bp, (300.0, 900.0, 40.0), cars=[c, cars[1]])
            t5, _ = v5.get_reward(c, ar, np.zeros(8), goal, 0 if goal else None)
            t9, _ = v9.get_reward(c, ar, np.zeros(8), goal, 0 if goal else None)
            self.assertAlmostEqual(t5, t9, places=9, msg=f"step {step}")

    def test_defaults_are_v8s_with_t7_retired_and_t8_added(self):
        expected = {**REWARD_V3_DEFAULTS, "closeness_weight": 0.0, "boost_weight": 0.0,
                    "boost_gain_weight": 0.0, "boost_edge_weight": EDGE_W, "gamma": GAMMA}
        self.assertEqual(REWARD_V9_DEFAULTS, expected)

    def test_breakdown_names_the_seven_terms_and_sums_to_total(self):
        m = manager()
        cars0 = [car(1, 0, (0.0, -1000.0, 17.0), boost=10.0), car(2, 1, (0.0, 1000.0, 17.0), boost=60.0)]
        m.reset(arena((0.0, 0.0, 93.0), cars=cars0))
        cars1 = [car(1, 0, (100.0, -800.0, 17.0), boost=80.0, touches=1), car(2, 1, (0.0, 900.0, 17.0), boost=30.0)]
        total, terms = m.get_reward(cars1[0], arena((50.0, 300.0, 100.0), (500.0, 700.0, 0.0), cars=cars1),
                                    np.zeros(8), False, None)
        self.assertEqual(tuple(terms), TERM_NAMES)
        self.assertEqual(TERM_NAMES,
                         ("goal", "ball_position", "touch", "closeness", "boost", "boost_gain", "boost_edge"))
        self.assertAlmostEqual(total, sum(terms.values()), places=12)


class TestActionIndependence(unittest.TestCase):
    def test_reward_ignores_the_action(self):
        totals = []
        for action in (np.zeros(8), np.ones(8), -np.ones(8)):
            m = manager()
            cars0 = [car(1, 0, (0.0, -1000.0, 17.0), boost=25.0), car(2, 1, (0.0, 1000.0, 17.0), boost=75.0)]
            m.reset(arena((0.0, 0.0, 93.0), cars=cars0))
            cars1 = [car(1, 0, (0.0, -600.0, 17.0), boost=60.0, touches=1), car(2, 1, (0.0, 900.0, 17.0), boost=10.0)]
            total, _ = m.get_reward(cars1[0], arena((0.0, 400.0, 93.0), (0.0, 800.0, 0.0), cars=cars1),
                                    action, False, None)
            totals.append(total)
        self.assertEqual(len(set(totals)), 1)

    def test_source_never_reads_the_action(self):
        with io.open("env/rewards_v9.py", encoding="utf-8") as f:
            body = f.read().split("def get_reward", 1)[1]
        self.assertIn("del action", body)
        self.assertNotIn("action[", body)


class TestRegistry(unittest.TestCase):
    def test_registry_builds_v9_with_its_frozen_weights_and_gamma(self):
        from env.reward_registry import load_snapshot, make_reward_manager
        snap = load_snapshot("v9")
        m = make_reward_manager("v9", dict(snap["settings"]["rewards"]))
        self.assertIsInstance(m, RewardManagerV9)
        self.assertEqual(m.reward.w["gamma"], snap["settings"]["gamma"])
        self.assertEqual(m.reward.w["boost_edge_weight"], EDGE_W)
        self.assertEqual(m.reward.w["boost_gain_weight"], 0.0)
        self.assertEqual(m.reward.w["boost_weight"], 0.0)

    def test_v9_starts_from_v5_since_v6_v7_and_v8_were_not_adopted(self):
        from env.reward_registry import load_snapshot
        self.assertEqual(load_snapshot("v9")["start_checkpoint"], "checkpoints/baselines/v5_iter222000.pt")

    def test_v9_changes_only_the_boost_term_against_v8(self):
        """Single variable: scenarios, gamma, replay sampling and pool are v8's, byte for byte."""
        from env.reward_registry import load_snapshot
        v8, v9 = load_snapshot("v8"), load_snapshot("v9")
        self.assertEqual(v9["settings"]["scenarios"], v8["settings"]["scenarios"])
        self.assertEqual(v9["settings"]["gamma"], v8["settings"]["gamma"])
        self.assertEqual(v9["settings"]["replay_sampling"], v8["settings"]["replay_sampling"])
        self.assertEqual(v9["settings"]["reward_annealing"], v8["settings"]["reward_annealing"])
        self.assertEqual(v9["replay_pool"], v8["replay_pool"])
        changed = {k for k in set(v8["settings"]["rewards"]) | set(v9["settings"]["rewards"])
                   if v8["settings"]["rewards"].get(k) != v9["settings"]["rewards"].get(k)}
        self.assertEqual(changed, {"boost_gain_weight", "boost_edge_weight"})

    def test_the_per_step_weight_is_calibrated_to_tick_skip_8(self):
        """T8 is paid every step, so its weight is only meaningful at the step rate it was sized at."""
        import yaml
        with io.open("config/default_config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["environment"]["tick_skip"], 8)


if __name__ == "__main__":
    unittest.main()
