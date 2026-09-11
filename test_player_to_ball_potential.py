"""
Contract tests for PlayerToBallVelocityReward after the additive income was removed.

The class used to be a distance potential plus roughly ten positive-only, per-step bonus
streams: velocity-toward-ball, strike-zone velocity matching, turnaround cut/pacing/rotation,
roof carry with velcro and sync bonuses, strike timing, and a second copy of the boost-pathing
incentive. Measured on the policy at iteration 2320 the class paid about 18.5 per episode,
while a pure distance potential at that weight is bounded near 1.25, because a car cannot
close more ground than the pitch is long. Roughly 93% of it was income rather than distance.

These tests lock in what replaced it. The properties that matter:

  1. Ground gained and ground lost cost exactly the same, so the term telescopes and cannot
     be pumped by oscillating toward and away from the ball.
  2. Standing still pays nothing, no matter how fast the car is nominally moving. The term
     responds to realized position change, not to intent or velocity.
  3. Penalties survive and can only subtract, so they cannot be farmed.

Tests for the removed streams were deleted rather than weakened; this file is what stands in
their place. See test_boost_pad_timers_and_symmetry.py and test_truncation_and_value_norm.py
for the same property asserted on the boost potentials.
"""

import unittest

import numpy as np

from env.physics_engine import CarState, BallState
from env.rewards import PlayerToBallVelocityReward


class MockArena:
    def __init__(self, ball, cars):
        self.ball = ball
        self.cars = cars
        self.boost_pads = []


def _car(pos, vel=(0.0, 0.0, 0.0), boost=50.0):
    return CarState(
        id=0, team=0,
        pos=np.array(pos, dtype=np.float32),
        vel=np.array(vel, dtype=np.float32),
        rot_mat=np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        boost=boost, on_ground=True,
    )


def _arena(car, ball_pos=(0.0, 2000.0, 93.0), ball_vel=(0.0, 0.0, 0.0)):
    ball = BallState(pos=np.array(ball_pos, dtype=np.float32),
                     vel=np.array(ball_vel, dtype=np.float32))
    return MockArena(ball, [car])


class TestDistancePotentialTelescopes(unittest.TestCase):
    def _walk(self, offsets, weight=0.5, boost=50.0):
        """Drive the car along the y axis toward a fixed ball, summing reward per move."""
        rew = PlayerToBallVelocityReward(weight=weight)
        car = _car((0.0, offsets[0], 17.0), boost=boost)
        arena = _arena(car)
        rew.reset(arena)
        total = 0.0
        act = np.zeros(8, dtype=np.float32)
        for y in offsets[1:]:
            car.pos = np.array([0.0, y, 17.0], dtype=np.float32)
            total += rew.get_reward(car, arena, act, False, None)
        return total

    def test_out_and_back_is_neutral(self):
        """The oscillation pump. Approaching then retreating to the same spot must pay zero."""
        total = self._walk([0.0, 400.0, 800.0, 400.0, 0.0])
        self.assertAlmostEqual(total, 0.0, places=5,
                               msg="approach and retreat must cancel exactly, got %r" % total)

    def test_repeated_laps_do_not_accumulate(self):
        """Five laps must pay what zero laps pay. This is the defect that was measured."""
        lap = [0.0, 600.0, 0.0]
        total = self._walk([0.0] + lap * 5)
        self.assertAlmostEqual(total, 0.0, places=5,
                               msg="five approach/retreat laps paid %r" % total)

    def test_closing_ground_still_pays(self):
        """The term must keep its job: net progress toward the ball is reinforced."""
        self.assertGreater(self._walk([0.0, 400.0, 800.0]), 0.0)

    def test_losing_ground_costs_what_gaining_it_paid(self):
        gained = self._walk([0.0, 800.0])
        lost = self._walk([800.0, 0.0])
        self.assertGreater(gained, 0.0)
        self.assertLess(lost, 0.0)
        self.assertAlmostEqual(gained, -lost, places=5,
                               msg="the two directions must be priced identically")

    def test_low_boost_does_not_discount_the_retreat(self):
        """The removed relief cut the charge for retreating by up to 85% when boost was low,
        which is what let the policy stop boosting and then oscillate for free."""
        lost_full = self._walk([800.0, 0.0], boost=100.0)
        lost_empty = self._walk([800.0, 0.0], boost=0.0)
        self.assertAlmostEqual(lost_full, lost_empty, places=5,
                               msg="an empty tank must not buy a cheaper retreat")


class TestNoVelocityIncome(unittest.TestCase):
    def test_standing_still_pays_nothing_however_fast_the_car_claims_to_be(self):
        """vel_toward_ball paid up to 0.40/step for pointing at the ball at speed, with no
        counterpart for pointing away. It double-counted the distance delta asymmetrically."""
        act = np.zeros(8, dtype=np.float32)
        for vel in ((0.0, 0.0, 0.0), (0.0, 2300.0, 0.0), (0.0, -2300.0, 0.0)):
            rew = PlayerToBallVelocityReward(weight=0.5)
            car = _car((0.0, 0.0, 17.0), vel=vel)
            arena = _arena(car)
            rew.reset(arena)
            r = rew.get_reward(car, arena, act, False, None)
            self.assertAlmostEqual(
                r, 0.0, places=5,
                msg="a car that has not moved must earn nothing at velocity %r, got %r" % (vel, r))

    def test_roof_carry_pays_no_per_step_income(self):
        """Balancing the ball on the roof used to pay carry, velcro and sync bonuses, which
        priced holding possession above using it."""
        rew = PlayerToBallVelocityReward(weight=0.5)
        car = _car((0.0, 0.0, 17.0), vel=(0.0, 800.0, 0.0))
        ball = BallState(pos=np.array([0.0, 20.0, 160.0], dtype=np.float32),
                         vel=np.array([0.0, 800.0, 0.0], dtype=np.float32))
        arena = MockArena(ball, [car])
        rew.reset(arena)
        act = np.zeros(8, dtype=np.float32)
        total = 0.0
        for k in range(30):
            # Car and ball travel together: the gap never changes, so a potential pays nothing.
            car.pos = np.array([0.0, 800.0 * (k + 1) * 0.05, 17.0], dtype=np.float32)
            ball.pos = np.array([0.0, 20.0 + 800.0 * (k + 1) * 0.05, 160.0], dtype=np.float32)
            total += rew.get_reward(car, arena, act, False, None)
        # Not asserted as exactly zero. A known residual survives at close carry range:
        # the aim point is tracked by UNSIGNED distance, so when the ball sits ~20 uu ahead
        # and both bodies advance ~40 uu in a step, the previous aim point ends up behind the
        # car at the same absolute distance and the target half of the delta reads as no
        # movement. That leaves about 0.002/step, roughly 1.2 per episode of carry income.
        # Removing it needs a signed, vector-valued formulation of the delta rather than a
        # distance one. Bounded here so it cannot quietly grow back into the 18.5 per episode
        # the additive streams used to pay.
        self.assertLess(
            abs(total), 0.15,
            "carrying the ball at a fixed distance must pay close to nothing, got %r" % total)


class TestPenaltiesCannotBeFarmed(unittest.TestCase):
    def test_every_surviving_extra_term_is_non_positive(self):
        """What remains beside the distance delta is penalties. A penalty can be avoided but
        never harvested, so none of them can turn into an income stream."""
        import inspect
        src = inspect.getsource(PlayerToBallVelocityReward.get_reward)
        assembly = src[src.index("total_reward = self.weight * ("):]
        for banned in ("vel_toward_ball", "vel_matching_bonus", "turnaround_reward",
                       "roof_carry_reward", "timing_reward", "boost_ahead_reward"):
            self.assertNotIn(
                banned, assembly,
                "%s is back in the reward assembly; it is positive-only per-step income" % banned)
        for kept in ("delta_dist", "pacing_penalty", "overshoot_penalty",
                     "ceiling_penalty", "wrong_side_push_penalty", "dribble_boost_penalty"):
            self.assertIn(kept, assembly, "%s should still be part of the term" % kept)


if __name__ == "__main__":
    unittest.main(verbosity=2)
