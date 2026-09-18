"""
Dropping balls: the pursuit target holds off the contact point, and OvershootCost prices running under.

Covers:
  1. solve_contact_point finds where and when the ball comes down to CONTACT_Z_UU.
  2. hold_point: the standoff shrinks to the contact point HOLD_LEAD_S before contact, sits goal-side
     when the car is underneath, and charges only HOLD_INSIDE_WEIGHT of a shortfall.
  3. PlayerToBallVelocityReward: on a dropping ball, arriving early scores worse than holding back;
     the hold gives way when the opponent reaches the ball first, and leaves a rolling ball alone.
  4. OvershootCost: a grounded car running under a dropping ball is charged; an aerial pass-by above
     600 uu (an air dribble lost) is not; a ground-ball miss is charged as before.
"""
import unittest

import numpy as np

import env.rewards as rewards
from env.physics_engine import BallState, CarState
from env.rewards import (
    CONTACT_Z_UU, HOLD_APPROACH_UU_S, HOLD_INSIDE_WEIGHT, HOLD_LEAD_S, HOLD_MAX_UU, OvershootCost,
    PlayerToBallVelocityReward, hold_point, solve_contact_point,
)

GRAVITY_HALF = 325.0


class BallisticArena:
    """A ball on a gravity arc with a matching prediction (no bounce)."""

    def __init__(self, ball_pos, ball_vel):
        self.ball = BallState(pos=np.array(ball_pos, dtype=np.float32), vel=np.array(ball_vel, dtype=np.float32))
        self.step_count = 0
        self.last_step_dt = 8.0 / 120.0
        self.cars = []

    def get_predicted_ball_pos(self, ticks):
        t = ticks / 120.0
        p, v = self.ball.pos, self.ball.vel
        return np.array([p[0] + v[0] * t, p[1] + v[1] * t,
                         max(rewards.BALL_RADIUS, p[2] + v[2] * t - GRAVITY_HALF * t * t)], dtype=np.float32)


def car_at(x, y, z=17.0, vel=(0.0, 0.0, 0.0), cid=0, team=0, boost=33.0):
    c = CarState(id=cid, team=team, pos=np.array([x, y, z], dtype=np.float32),
                 vel=np.array(vel, dtype=np.float32), rot=np.array([0.0, 1.5708, 0.0]))
    c.boost = boost
    c.on_ground = z < 30.0
    return c


def drop_time(z0, z1=CONTACT_Z_UU):
    return float(np.sqrt((z0 - z1) / GRAVITY_HALF))


class TestContactPoint(unittest.TestCase):
    def test_drop_is_met_at_contact_height(self):
        arena = BallisticArena([500.0, -300.0, 1000.0], [0.0, 0.0, -1.0])
        pos, t = solve_contact_point(arena)
        self.assertLessEqual(float(pos[2]), CONTACT_Z_UU)
        self.assertGreater(float(pos[2]), CONTACT_Z_UU - 20.0)
        self.assertAlmostEqual(t, drop_time(1000.0), delta=2.0 / 120.0)

    def test_low_ball_is_already_in_contact(self):
        arena = BallisticArena([0.0, 0.0, 93.0], [800.0, 0.0, 0.0])
        pos, t = solve_contact_point(arena)
        self.assertEqual(t, 0.0)


class TestHoldPoint(unittest.TestCase):
    def test_standoff_slides_in_to_the_contact_point(self):
        contact = np.array([0.0, 0.0, CONTACT_Z_UU], dtype=np.float32)
        car = np.array([0.0, -2000.0, 17.0], dtype=np.float32)
        prev = None
        for t in np.arange(1.5, HOLD_LEAD_S - 0.01, -1.0 / 15.0):
            h = hold_point(car, contact, float(t), 0)
            standoff = float(np.linalg.norm(h[:2]))
            self.assertAlmostEqual(standoff, min(HOLD_MAX_UU, HOLD_APPROACH_UU_S * max(0.0, t - HOLD_LEAD_S)), delta=1.0)
            self.assertAlmostEqual(float(h[2]), CONTACT_Z_UU)
            if prev is not None:
                self.assertLess(standoff, prev + 1e-3)
            prev = standoff
        np.testing.assert_allclose(hold_point(car, contact, HOLD_LEAD_S, 0)[:2], contact[:2], atol=1e-3)

    def test_car_underneath_is_sent_goal_side(self):
        contact = np.array([0.0, 500.0, CONTACT_Z_UU], dtype=np.float32)
        h = hold_point(np.array([0.0, 500.0, 17.0]), contact, 1.0, 0)
        self.assertLess(float(h[1]), 500.0, "blue defends -Y: the hold point is on that side")

    def test_inside_the_standoff_only_part_of_the_shortfall_counts(self):
        contact = np.array([0.0, 0.0, CONTACT_Z_UU], dtype=np.float32)
        t = 1.25                                          # full standoff 1000 uu
        car = np.array([0.0, -300.0, 17.0], dtype=np.float32)
        h = hold_point(car, contact, t, 0)
        full = HOLD_APPROACH_UU_S * (t - HOLD_LEAD_S)
        gap = float(np.linalg.norm(h[:2] - car[:2]))
        self.assertLess(gap, full - 300.0, "a close car is not sent all the way back")
        self.assertGreater(gap, 0.5 * HOLD_INSIDE_WEIGHT * (full - 300.0))


class TestPursuitHoldsOffADroppingBall(unittest.TestCase):
    def _target(self, car, ball_pos=(0.0, 0.0, 900.0), opp=None):
        arena = BallisticArena(list(ball_pos), [0.0, 0.0, -1.0])
        arena.cars = [car] + ([opp] if opp is not None else [car_at(0.0, 5000.0, cid=1, team=1, boost=0.0)])
        p2b = PlayerToBallVelocityReward(weight=1.0)
        return p2b, p2b._get_target_pos(car.pos, arena, False, car.vel, car.id, car.team, car=car,
                                        boost_amount=float(car.boost))

    def test_arriving_early_scores_worse_than_holding_back(self):
        early = car_at(0.0, -60.0, vel=(0.0, 900.0, 0.0))
        p2b, target_early = self._target(early)
        held = car_at(0.0, -900.0, vel=(0.0, 900.0, 0.0))
        _, target_held = self._target(held)
        self.assertGreater(p2b._calc_dist(early.pos, target_early), p2b._calc_dist(held.pos, target_held) + 100.0)
        self.assertLess(float(target_early[2]), CONTACT_Z_UU + 30.0, "the target is at contact height, not up at the ball")

    def test_hold_gives_way_when_the_opponent_gets_there_first(self):
        car = car_at(0.0, -900.0, vel=(0.0, 900.0, 0.0))
        opp = car_at(0.0, 150.0, vel=(0.0, -500.0, 0.0), cid=1, team=1, boost=100.0)
        _, contested = self._target(car, opp=opp)
        _, free = self._target(car)
        self.assertGreater(float(contested[1]), float(free[1]) + 300.0,
                           "contested: the target stays up at the ball rather than holding back")

    def test_rolling_ball_is_untouched(self):
        car = car_at(0.0, -900.0, vel=(0.0, 900.0, 0.0))
        arena = BallisticArena([0.0, 0.0, 93.0], [0.0, 400.0, 0.0])
        arena.cars = [car, car_at(0.0, 5000.0, cid=1, team=1, boost=0.0)]
        p2b = PlayerToBallVelocityReward(weight=1.0)
        with_hold = p2b._get_target_pos(car.pos, arena, False, car.vel, 0, 0, car=car, boost_amount=33.0)
        orig = rewards.cached_contact_point
        rewards.cached_contact_point = lambda a: None
        try:
            arena.step_count += 1
            without = p2b._get_target_pos(car.pos, arena, False, car.vel, 0, 0, car=car, boost_amount=33.0)
        finally:
            rewards.cached_contact_point = orig
        np.testing.assert_allclose(with_hold, without, atol=1e-3)


class TestOvershootUnderADroppingBall(unittest.TestCase):
    def _pass(self, ball_pos, car_z=17.0, lateral=0.0, speed=1200.0, steps=30):
        arena = BallisticArena(list(ball_pos), [0.0, 0.0, 0.0])
        arena.get_predicted_ball_pos = lambda ticks: arena.ball.pos.copy()
        car = car_at(lateral, -700.0, z=car_z, vel=(0.0, speed, 0.0))
        arena.cars = [car, car_at(0.0, 4000.0, cid=1, team=1)]
        cost = OvershootCost(weight=1.0)
        cost.reset(arena)
        total = 0.0
        for _ in range(steps):
            car.pos = car.pos + car.vel * (8.0 / 120.0)
            arena.step_count += 1
            total += cost.get_reward(car, arena, np.zeros(8, dtype=np.float32), False, None)
        return total

    def test_running_under_a_ball_in_jump_reach_is_charged(self):
        self.assertLess(self._pass((0.0, 0.0, 400.0)), -0.5)

    def test_a_ball_far_overhead_is_not_a_miss(self):
        self.assertEqual(self._pass((0.0, 0.0, 900.0)), 0.0)

    def test_an_aerial_pass_by_is_still_free(self):
        # Flying past a ball 800 uu up (a lost air dribble): priced as before, which is nothing
        self.assertEqual(self._pass((0.0, 0.0, 800.0), car_z=780.0), 0.0)

    def test_ground_ball_miss_is_still_charged(self):
        self.assertLess(self._pass((0.0, 0.0, 93.0), lateral=150.0, speed=1400.0), -0.3)


if __name__ == "__main__":
    unittest.main()
