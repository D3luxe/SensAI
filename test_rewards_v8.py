"""
Reward v8 (env/rewards_v8.py) against its specification, docs/reward_v8_boost_spec.md.

  - T7 boost_gain: pays the rise in sqrt(boost/100), pays nothing on the fall
  - path independence: one big pad and eight small ones covering the same range pay the same
  - concavity: a pickup from empty is worth several times the same pickup from nearly full
  - T7 is NOT a potential, so a collect/spend cycle pays every time -- the property that makes it
    able to move the policy, and the one that has to be weight-limited rather than gated
  - T1-T5 are v5's terms unchanged: with T7 off and boost_weight restored, v8 pays exactly v5
  - the reward never depends on the action (R2)
  - the registry builds v8 at v5's gamma with its frozen weights
"""
import math
import unittest
from types import SimpleNamespace

import numpy as np

from env.rewards_v3 import REWARD_V3_DEFAULTS, RewardManagerV3
from env.rewards_v8 import REWARD_V8_DEFAULTS, RewardManagerV8, TERM_NAMES

GAMMA = 0.9977


def car(cid, team, pos, boost=33.0, touches=0):
    return SimpleNamespace(id=cid, team=team, pos=np.array(pos, dtype=np.float32), boost=boost, ball_touches=touches)


def arena(ball_pos, ball_vel=(0.0, 0.0, 0.0), cars=()):
    return SimpleNamespace(ball=SimpleNamespace(pos=np.array(ball_pos, dtype=np.float32),
                                                vel=np.array(ball_vel, dtype=np.float32)), cars=list(cars))


def manager(**w):
    return RewardManagerV8({**REWARD_V8_DEFAULTS, **w})


def phi(b):
    return math.sqrt(max(0.0, min(100.0, b)) / 100.0)


def run_boosts(boosts, weight=2.0, ball=(0.0, 0.0, 93.0)):
    """Drive one car through a sequence of boost readings; return the boost_gain paid at each step."""
    m = manager(boost_gain_weight=weight, ball_position_weight=0.0, touch_weight=0.0)
    c0 = car(1, 0, (0.0, -1000.0, 17.0), boost=boosts[0])
    m.reset(arena(ball, cars=[c0]))
    out = []
    for b in boosts[1:]:
        c = car(1, 0, (0.0, -1000.0, 17.0), boost=b)
        _, terms = m.get_reward(c, arena(ball, cars=[c]), np.zeros(8), False, None)
        out.append(terms["boost_gain"])
    return out


class TestBoostGain(unittest.TestCase):
    def test_pays_the_rise_in_the_v5_potential(self):
        (paid,) = run_boosts([0.0, 100.0])
        self.assertAlmostEqual(paid, 2.0 * (phi(100.0) - phi(0.0)), places=9)

    def test_pays_nothing_on_the_fall(self):
        self.assertEqual(run_boosts([100.0, 0.0]), [0.0])
        self.assertEqual(run_boosts([100.0, 60.0, 20.0]), [0.0, 0.0])

    def test_pays_nothing_when_boost_is_unchanged(self):
        self.assertEqual(run_boosts([50.0, 50.0, 50.0]), [0.0, 0.0])

    def test_is_path_independent_across_a_monotonic_rise(self):
        one_jump = sum(run_boosts([0.0, 100.0]))
        in_eights = sum(run_boosts([0.0] + [12.5 * k for k in range(1, 9)]))
        self.assertAlmostEqual(one_jump, in_eights, places=9)

    def test_refuelling_from_empty_beats_topping_up(self):
        from_empty = sum(run_boosts([0.0, 30.0]))
        near_full = sum(run_boosts([70.0, 100.0]))
        self.assertAlmostEqual(from_empty, 2.0 * (phi(30.0) - phi(0.0)), places=9)
        self.assertAlmostEqual(near_full, 2.0 * (phi(100.0) - phi(70.0)), places=9)
        self.assertGreater(from_empty, 3.0 * near_full)

    def test_a_full_refill_is_worth_exactly_the_weight(self):
        for w in (0.5, 1.0, 2.0, 4.0):
            self.assertAlmostEqual(sum(run_boosts([0.0, 100.0], weight=w)), w, places=9)

    def test_a_collect_spend_cycle_pays_every_time(self):
        """The whole point of the ratchet: unlike a potential, a loop does not sum to zero."""
        cycle = run_boosts([0.0, 100.0, 0.0, 100.0, 0.0, 100.0])
        self.assertAlmostEqual(sum(cycle), 3 * 2.0, places=9)

    def test_weight_zero_disables_it(self):
        self.assertEqual(run_boosts([0.0, 100.0], weight=0.0), [0.0])

    def test_the_first_step_measures_from_the_reset_state(self):
        """A car that starts full must not bank a pickup it never made."""
        m = manager()
        c = car(1, 0, (0.0, -1000.0, 17.0), boost=100.0)
        m.reset(arena((0.0, 0.0, 93.0), cars=[c]))
        _, terms = m.get_reward(c, arena((0.0, 0.0, 93.0), cars=[c]), np.zeros(8), False, None)
        self.assertEqual(terms["boost_gain"], 0.0)

    def test_a_pickup_on_the_goal_step_is_still_paid(self):
        """T7 is not a potential, so R4's phi(terminal) = 0 does not apply to it."""
        m = manager(ball_position_weight=0.0, touch_weight=0.0)
        c0 = car(1, 0, (0.0, -1000.0, 17.0), boost=0.0)
        m.reset(arena((0.0, 0.0, 93.0), cars=[c0]))
        c1 = car(1, 0, (0.0, -1000.0, 17.0), boost=100.0)
        _, terms = m.get_reward(c1, arena((0.0, 5200.0, 93.0), cars=[c1]), np.zeros(8), True, 0)
        self.assertAlmostEqual(terms["boost_gain"], 2.0, places=9)

    def test_per_car_state_is_independent(self):
        m = manager(ball_position_weight=0.0, touch_weight=0.0)
        a0, b0 = car(1, 0, (0.0, -1000.0, 17.0), boost=0.0), car(2, 1, (0.0, 1000.0, 17.0), boost=100.0)
        m.reset(arena((0.0, 0.0, 93.0), cars=[a0, b0]))
        a1, b1 = car(1, 0, (0.0, -1000.0, 17.0), boost=100.0), car(2, 1, (0.0, 1000.0, 17.0), boost=0.0)
        ar = arena((0.0, 0.0, 93.0), cars=[a1, b1])
        _, ta = m.get_reward(a1, ar, np.zeros(8), False, None)
        _, tb = m.get_reward(b1, ar, np.zeros(8), False, None)
        self.assertAlmostEqual(ta["boost_gain"], 2.0, places=9)
        self.assertEqual(tb["boost_gain"], 0.0)


class TestV5TermsUnchanged(unittest.TestCase):
    def test_with_t7_off_and_t5_restored_v8_pays_exactly_v5(self):
        w5 = {**REWARD_V3_DEFAULTS, "closeness_weight": 0.0, "gamma": GAMMA}
        v5, v8 = RewardManagerV3(dict(w5)), RewardManagerV8({**w5, "boost_weight": 1.0, "boost_gain_weight": 0.0})
        cars = [car(1, 0, (0.0, -1000.0, 17.0), boost=40.0), car(2, 1, (200.0, 800.0, 17.0), boost=70.0)]
        start = arena((0.0, 0.0, 93.0), cars=cars)
        v5.reset(start)
        v8.reset(start)
        for step, (bp, bv, boost, touches, goal) in enumerate([
                ((100.0, 500.0, 120.0), (300.0, 900.0, 40.0), 55.0, 1, False),
                ((400.0, 2000.0, 300.0), (800.0, 1500.0, -60.0), 20.0, 1, False),
                ((0.0, 5200.0, 200.0), (100.0, 2000.0, 0.0), 95.0, 2, True)]):
            c = car(1, 0, (0.0, -900.0 + 400 * step, 17.0), boost=boost, touches=touches)
            ar = arena(bp, bv, cars=[c, cars[1]])
            t5, _ = v5.get_reward(c, ar, np.zeros(8), goal, 0 if goal else None)
            t8, _ = v8.get_reward(c, ar, np.zeros(8), goal, 0 if goal else None)
            self.assertAlmostEqual(t5, t8, places=9, msg=f"step {step}")

    def test_defaults_are_v5s_with_t5_retired_and_t7_added(self):
        expected = {**REWARD_V3_DEFAULTS, "closeness_weight": 0.0, "boost_weight": 0.0,
                    "boost_gain_weight": 2.0, "gamma": GAMMA}
        self.assertEqual(REWARD_V8_DEFAULTS, expected)

    def test_breakdown_names_the_six_terms_and_sums_to_total(self):
        m = manager()
        c0 = car(1, 0, (0.0, -1000.0, 17.0), boost=10.0)
        m.reset(arena((0.0, 0.0, 93.0), cars=[c0]))
        c1 = car(1, 0, (100.0, -800.0, 17.0), boost=80.0, touches=1)
        total, terms = m.get_reward(c1, arena((50.0, 300.0, 100.0), (500.0, 700.0, 0.0), cars=[c1]),
                                    np.zeros(8), False, None)
        self.assertEqual(tuple(terms), TERM_NAMES)
        self.assertEqual(TERM_NAMES, ("goal", "ball_position", "touch", "closeness", "boost", "boost_gain"))
        self.assertAlmostEqual(total, sum(terms.values()), places=9)


class TestActionIndependence(unittest.TestCase):
    def test_reward_ignores_the_action(self):
        totals = []
        for action in (np.zeros(8), np.ones(8), -np.ones(8)):
            m = manager()
            c0 = car(1, 0, (0.0, -1000.0, 17.0), boost=25.0)
            m.reset(arena((0.0, 0.0, 93.0), cars=[c0]))
            c1 = car(1, 0, (0.0, -600.0, 17.0), boost=60.0, touches=1)
            total, _ = m.get_reward(c1, arena((0.0, 400.0, 93.0), (0.0, 800.0, 0.0), cars=[c1]),
                                    action, False, None)
            totals.append(total)
        self.assertEqual(len(set(totals)), 1)

    def test_source_never_reads_the_action(self):
        import io
        with io.open("env/rewards_v8.py", encoding="utf-8") as f:
            body = f.read().split("def get_reward", 1)[1]
        self.assertIn("del action", body)
        self.assertNotIn("action[", body)


class TestRegistry(unittest.TestCase):
    def test_registry_builds_v8_with_its_frozen_weights_and_gamma(self):
        from env.reward_registry import load_snapshot, make_reward_manager
        snap = load_snapshot("v8")
        m = make_reward_manager("v8", dict(snap["settings"]["rewards"]))
        self.assertIsInstance(m, RewardManagerV8)
        self.assertEqual(m.reward.w["gamma"], snap["settings"]["gamma"])
        self.assertEqual(m.reward.w["boost_gain_weight"], 2.0)
        self.assertEqual(m.reward.w["boost_weight"], 0.0)

    def test_v8_starts_from_v5_since_v6_and_v7_were_not_adopted(self):
        from env.reward_registry import load_snapshot
        self.assertEqual(load_snapshot("v8")["start_checkpoint"], "checkpoints/baselines/v5_iter222000.pt")

    def test_v8_replay_tags_are_the_pools_own_frequencies(self):
        """v7's tag weighting is dropped: sampling a kept frame is uniform again."""
        from env.reward_registry import load_snapshot
        counts = load_snapshot("v7")["replay_pool"]["tag_counts"]
        total = sum(counts.values())
        weights = load_snapshot("v8")["settings"]["replay_sampling"]["tag_weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        for tag, n in counts.items():
            self.assertAlmostEqual(weights[tag], n / total, places=5, msg=tag)


if __name__ == "__main__":
    unittest.main()
