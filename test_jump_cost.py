"""
Regression tests for JumpCostReward.

Why the term exists. The three binary action channels are Bernoulli, and an entropy bonus
pulls a Bernoulli with no reward gradient toward probability 0.5. Boost and handbrake are both
priced and both sit far from 0.5 in the learned policy; jump was priced only by
JumpBridgeReward, and once that went to zero nothing charged for pressing it. Measured at
iteration 5360, raw P(jump) was 0.377 airborne and 0.613 one step after landing, against 0.042
for boost. The policy was flipping a coin on every frame a jump was available, which at 15 Hz
looks exactly like re-jumping the instant the wheels touch down.

What these tests protect:
  1. The fee is charged on the rising edge of takeoff and nowhere else, so holding jump for
     full height costs nothing extra and an airborne dodge is not taxed a second time.
  2. Leaving the ground without spending a jump -- driving off a ramp or the lip of a wall --
     is free.
  3. It can only ever subtract. The reward it replaces was farmable precisely because it paid.
  4. It is unconditional. No distance or ball-height gate, because a gate creates a cliff at
     its own boundary, which is the defect class that produced most of the trouble here.
"""

import unittest

import numpy as np

from env.physics_engine import CarState, BallState
from env.rewards import JumpCostReward, CombinedReward

FEE = 0.025


class MockArena:
    def __init__(self, cars, ball_pos=(0.0, 1500.0, 93.0)):
        self.ball = BallState(pos=np.array(ball_pos, dtype=np.float32))
        self.cars = cars
        self.boost_pads = []


def _run(states, weight=FEE, ball_pos=(0.0, 1500.0, 93.0)):
    """Drive a car through (on_ground, has_jump) states; return the reward at each step."""
    car = CarState(id=0, team=0, pos=np.array([0.0, 0.0, 17.0], dtype=np.float32))
    car.on_ground, car.has_jump = states[0]
    arena = MockArena([car], ball_pos)
    rew = JumpCostReward(weight=weight)
    rew.reset(arena)
    act = np.zeros(8, dtype=np.float32)
    out = []
    for on_ground, has_jump in states[1:]:
        car.on_ground, car.has_jump = on_ground, has_jump
        out.append(round(rew.get_reward(car, arena, act, False, None), 6))
    return out


class TestChargedOnceOnTakeoff(unittest.TestCase):
    def test_single_charge_on_the_rising_edge(self):
        got = _run([(True, True), (False, False), (False, False), (False, False)])
        self.assertEqual(got, [-FEE, 0.0, 0.0],
                         "the fee must land once, on the frame the wheels leave the ground")

    def test_holding_jump_for_height_costs_nothing_extra(self):
        """Full-height jumps must not be discouraged: only the decision is priced."""
        airborne = [(False, False)] * 12
        got = _run([(True, True)] + airborne)
        self.assertEqual(got[0], -FEE)
        self.assertEqual(sum(got[1:]), 0.0,
                         "holding jump across twelve steps must cost nothing beyond takeoff")

    def test_each_separate_takeoff_is_charged(self):
        got = _run([(True, True), (False, False), (True, True),
                    (False, False), (True, True), (False, False)])
        self.assertEqual(got.count(-FEE), 3, "three takeoffs, three fees")
        self.assertAlmostEqual(sum(got), -3 * FEE, places=9)

    def test_airborne_dodge_is_not_taxed_again(self):
        """A dodge happens mid-flight, with no ground transition, so it is never charged."""
        got = _run([(False, False), (False, False), (False, False), (False, False)])
        self.assertEqual(sum(got), 0.0)


class TestTakeoffIsDistinguishedFromRollingOff(unittest.TestCase):
    def test_driving_off_a_ledge_is_free(self):
        """Leaving the ground is not enough on its own. has_jump is consumed by an actual
        jump and stays available otherwise, which separates a takeoff from a roll-off."""
        got = _run([(True, True), (False, True), (False, True), (True, True)])
        self.assertEqual(sum(got), 0.0,
                         "a car that rolled off a ramp never spent a jump and owes nothing")

    def test_landing_is_free(self):
        got = _run([(False, False), (True, True), (True, True)])
        self.assertEqual(sum(got), 0.0, "coming back down is not a jump")


class TestTermCannotBeFarmed(unittest.TestCase):
    def test_never_returns_a_positive_value(self):
        """JumpBridgeReward was farmable because it paid. A cost cannot be."""
        rng = np.random.default_rng(0)
        states = [(bool(rng.integers(2)), bool(rng.integers(2))) for _ in range(400)]
        for v in _run(states):
            self.assertLessEqual(v, 0.0, "this term must only ever subtract")

    def test_weight_zero_disables_it(self):
        got = _run([(True, True), (False, False), (True, True), (False, False)], weight=0.0)
        self.assertEqual(sum(got), 0.0)


class TestUnconditional(unittest.TestCase):
    def test_same_fee_regardless_of_ball_position(self):
        """No distance or height gate: a gate creates a reward cliff at its own boundary."""
        near = _run([(True, True), (False, False)], ball_pos=(0.0, 60.0, 93.0))
        far = _run([(True, True), (False, False)], ball_pos=(0.0, 5000.0, 93.0))
        high = _run([(True, True), (False, False)], ball_pos=(0.0, 1500.0, 1600.0))
        self.assertEqual(near, far)
        self.assertEqual(near, high)
        self.assertEqual(near, [-FEE])


class TestWiring(unittest.TestCase):
    def test_registered_in_the_combined_reward(self):
        c = CombinedReward({})
        self.assertIn("jump_cost", c.rewards)

    def test_weight_comes_from_config_and_is_live_tunable(self):
        c = CombinedReward({"jump_cost_weight": 0.05})
        self.assertAlmostEqual(c.rewards["jump_cost"].weight, 0.05)
        c.update_weights({"jump_cost_weight": 0.0})
        self.assertAlmostEqual(c.rewards["jump_cost"].weight, 0.0)

    def test_both_configs_carry_the_weight(self):
        import io
        import json
        import yaml
        cfg = yaml.safe_load(io.open("config/default_config.yaml", encoding="utf-8").read())
        live = json.load(io.open("config/live_config.json", encoding="utf-8"))
        self.assertIn("jump_cost_weight", cfg["rewards"])
        self.assertIn("jump_cost_weight", live["rewards"],
                      "live_config overrides the yaml at runtime; a weight missing here is "
                      "silently the class default")
        self.assertAlmostEqual(cfg["rewards"]["jump_cost_weight"],
                               live["rewards"]["jump_cost_weight"])


class TestAgainstTheSimulator(unittest.TestCase):
    def test_a_real_jump_is_charged_exactly_once(self):
        """Synthetic states can disagree with the simulator; this drives a real jump."""
        from env.rocket_env import RocketLeagueEnv

        # Two things make this awkward to script. Assigning to car.pos or car.vel does nothing
        # -- RocketSim owns the state and overwrites both on the next step -- so the car has to
        # be driven into position with actions. And reset() randomises the kickoff, so the seed
        # is pinned: without it the car sometimes starts in a state where the jump does not
        # leave the ground within the window and the test fails at random.
        import random

        np.random.seed(17)
        random.seed(17)
        env = RocketLeagueEnv(game_mode="1v1", max_episode_steps=600)
        env.reset()

        total = 0.0
        charges = 0
        airborne_seen = False
        for i in range(30):
            a = np.zeros((2, env.act_dim), dtype=np.float32)
            a[0][0] = 1.0              # throttle, so the car is rolling rather than stationary
            if 4 <= i <= 10:           # press and then hold
                a[0][5] = 1.0
            _, _, _, info = env.step(a, include_breakdown=True)
            v = float((info.get("reward_breakdown") or {}).get("jump_cost", 0.0))
            total += v
            if v != 0.0:
                charges += 1
            if not env.arena.cars[0].on_ground:
                airborne_seen = True

        self.assertTrue(airborne_seen, "the scripted jump never left the ground; test is invalid")
        self.assertEqual(charges, 1, "one takeoff must produce exactly one charge")
        self.assertAlmostEqual(total, -FEE, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
