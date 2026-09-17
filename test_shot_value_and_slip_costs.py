"""
Guarantees for attacking shot value and the smooth, state-based slide costs.

  1. Deep in the attacking third a no-angle smash into the backboard is valued near 0, a center
     into the slot and a shot on target near 1, from both the predicted and the ballistic path.
  2. From the same tight-angle spot, TouchBallReward pays the center and the shot on target more
     than the backboard smash, and ball_to_goal keeps only B2G_OFF_TARGET_FLOOR of an off-target rate.
  3. Outside the attacking third and on kickoff touches nothing changes.
  4. LateralSlipCost and SlideWasteCost have no cliffs at their old thresholds, and SlideWasteCost
     reads only car state: the same state with different inputs costs the same.
"""

import unittest
from unittest import mock

import numpy as np

import env.rewards as rewards
from env.physics_engine import ARENA_EXTENT_Y, BallState, CarState
from env.rewards import (
    B2G_OFF_TARGET_FLOOR, BallToGoalVelocityReward, LateralSlipCost, SlideWasteCost, TouchBallReward,
    attacking_shot_value, attacking_third_weight, centering_factor, goal_mouth_open_angle, on_target_factor,
)

GOAL_Y = ARENA_EXTENT_Y  # blue attacks +Y
CORNER = [2900.0, 4500.0, 120.0]
SMASH_VEL = [-200.0, 2200.0, 250.0]    # into the backboard beside the post
CENTER_VEL = [-2000.0, -250.0, 350.0]  # across to the slot
SHOT_VEL = [-2300.0, 500.0, 150.0]     # at the mouth from the same spot


class BallisticArena:
    """A ball on a gravity arc with a matching prediction and a blue car on it."""

    def __init__(self, ball_pos, ball_vel):
        self.ball = BallState(pos=np.array(ball_pos, dtype=np.float32), vel=np.array(ball_vel, dtype=np.float32))
        self.step_count = 0
        self.last_step_dt = 8.0 / 120.0
        self.cars = []

    def get_predicted_ball_pos(self, ticks):
        t = ticks / 120.0
        p, v = self.ball.pos, self.ball.vel
        return np.array([p[0] + v[0] * t, p[1] + v[1] * t,
                         max(rewards.BALL_RADIUS, p[2] + v[2] * t - 325.0 * t * t)], dtype=np.float32)


class NoPredictionArena(BallisticArena):
    """The same ball with no trajectory, so the ballistic fallbacks run."""

    def __getattribute__(self, name):
        if name == "get_predicted_ball_pos":
            raise AttributeError(name)
        return super().__getattribute__(name)


def value(pos, vel, arena_cls=BallisticArena):
    arena = arena_cls(pos, vel)
    return attacking_shot_value(arena, GOAL_Y, on_target_factor(arena, GOAL_Y))


class TestShotValue(unittest.TestCase):
    def test_open_angle(self):
        self.assertGreater(goal_mouth_open_angle(np.array([0.0, 2000.0, 93.0]), GOAL_Y), 25.0)
        self.assertLess(goal_mouth_open_angle(np.array(CORNER), GOAL_Y), 12.0)
        self.assertEqual(goal_mouth_open_angle(np.array([0.0, GOAL_Y + 10.0, 93.0]), GOAL_Y), 0.0)

    def test_values_from_a_tight_angle(self):
        for cls in (BallisticArena, NoPredictionArena):
            with self.subTest(arena=cls.__name__):
                self.assertLess(value(CORNER, SMASH_VEL, cls), 0.1)
                self.assertGreater(value(CORNER, CENTER_VEL, cls), 0.7)
                self.assertGreater(value(CORNER, SHOT_VEL, cls), 0.9)

    def test_backboard_rebound_into_the_slot_is_not_a_center(self):
        class BackWallArena(BallisticArena):
            def get_predicted_ball_pos(self, ticks):
                p = super().get_predicted_ball_pos(ticks)
                limit = GOAL_Y - rewards.BALL_RADIUS
                if p[1] > limit:
                    p[1] = 2.0 * limit - p[1]
                return p
        # Hard at the back wall off the far post, drifting in: the rebound comes back out through
        # the slot and would score 1.0 as a center if the scan did not stop at the wall
        arena = BackWallArena([1600.0, 4200.0, 120.0], [-900.0, 2600.0, 300.0])
        self.assertLess(centering_factor(arena, GOAL_Y), 0.05)
        self.assertGreater(centering_factor(BackWallArena(CORNER, CENTER_VEL), GOAL_Y), 0.7)

    def test_ball_already_in_the_slot_is_not_a_center(self):
        self.assertLess(centering_factor(BallisticArena([0.0, 4000.0, 93.0], [0.0, 0.0, 0.0]), GOAL_Y), 0.05)

    def test_center_counts_for_less_when_the_mouth_is_open(self):
        # From a wide spot with a clear sight of goal, the same kind of pass is worth less
        open_spot = value([1500.0, 2600.0, 93.0], [-1400.0, 900.0, 250.0])
        self.assertGreater(value(CORNER, CENTER_VEL), open_spot)

    def test_attacking_third_weight(self):
        self.assertEqual(attacking_third_weight(np.array([0.0, 3000.0, 93.0]), GOAL_Y), 1.0)
        self.assertEqual(attacking_third_weight(np.array([0.0, -500.0, 93.0]), GOAL_Y), 0.0)
        mid = attacking_third_weight(np.array([0.0, 1000.0, 93.0]), GOAL_Y)
        self.assertTrue(0.0 < mid < 1.0)


class TestTouchAndProgressionPay(unittest.TestCase):
    def _touch(self, pos, vel):
        arena = BallisticArena(pos, [0.0, 0.0, 0.0])
        car = CarState(id=0, team=0, pos=np.array([pos[0] + 120.0, pos[1] - 120.0, 17.0], dtype=np.float32),
                       vel=np.array([-900.0, 900.0, 0.0], dtype=np.float32), rot=np.array([0.0, 2.36, 0.0]))
        opp = CarState(id=1, team=1, pos=np.array([0.0, 5000.0, 17.0], dtype=np.float32))
        arena.cars = [car, opp]
        r = TouchBallReward(weight=1.0)
        r.reset(arena)
        car.ball_touches += 1
        arena.ball.vel = np.array(vel, dtype=np.float32)
        return r.get_reward(car, arena, np.zeros(8, dtype=np.float32), False, None)

    def test_tight_angle_smash_pays_less_than_center_or_shot(self):
        smash, center, shot = self._touch(CORNER, SMASH_VEL), self._touch(CORNER, CENTER_VEL), self._touch(CORNER, SHOT_VEL)
        self.assertLess(smash, center)
        self.assertLess(smash, shot)
        self.assertGreater(smash, 0.0, "touching the ball still pays")

    def test_midfield_strike_is_unchanged(self):
        with mock.patch.object(rewards, "attacking_shot_value", side_effect=AssertionError("not consulted")):
            self._touch([2900.0, -1500.0, 120.0], SMASH_VEL)

    def test_ball_to_goal_off_target_floor(self):
        car = CarState(id=0, team=0, pos=np.array([2800.0, 4300.0, 17.0], dtype=np.float32))
        arena = BallisticArena(CORNER, SMASH_VEL)
        arena.cars = [car]
        b2g = BallToGoalVelocityReward(weight=1.0)
        scaled = b2g.get_reward(car, arena, np.zeros(8), False, None)
        with mock.patch.object(rewards, "attacking_third_weight", return_value=0.0):
            unscaled = b2g.get_reward(car, BallisticArena(CORNER, SMASH_VEL), np.zeros(8), False, None)
        self.assertGreater(unscaled, 0.0)
        self.assertAlmostEqual(scaled / unscaled, B2G_OFF_TARGET_FLOOR, delta=0.02)

    def test_ball_to_goal_losses_are_not_discounted(self):
        # An off-target smash earns the floor going in; its rebound costs full coming back out
        car = CarState(id=0, team=0, pos=np.array([2800.0, 4300.0, 17.0], dtype=np.float32))
        fwd_arena, back_arena = BallisticArena(CORNER, SMASH_VEL), BallisticArena(CORNER, [200.0, -2200.0, -250.0])
        fwd_arena.cars, back_arena.cars = [car], [car]
        b2g = BallToGoalVelocityReward(weight=1.0)
        fwd = b2g.get_reward(car, fwd_arena, np.zeros(8), False, None)
        back = b2g.get_reward(car, back_arena, np.zeros(8), False, None)
        self.assertGreater(fwd, 0.0)
        self.assertLess(fwd + back, 0.0)
        self.assertEqual(attacking_third_weight(np.array([0.0, 0.0, 93.0]), GOAL_Y), 0.0, "midfield still cancels")


class TestSlideCosts(unittest.TestCase):
    def _car(self, vel, ang_vel=(0.0, 0.0, 0.0)):
        # Facing +X, upright
        return CarState(id=0, team=0, pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
                        vel=np.array(vel, dtype=np.float32), ang_vel=np.array(ang_vel, dtype=np.float32),
                        rot=np.array([0.0, 0.0, 0.0]), on_ground=True)

    def test_lateral_slip_has_no_distance_cliff(self):
        cost = LateralSlipCost()
        car = self._car([300.0, -450.0, 0.0])
        near = cost.get_reward(car, BallisticArena([219.0, 0.0, 93.0], [0, 0, 0]), np.zeros(8), False, None)
        far = cost.get_reward(car, BallisticArena([221.0, 0.0, 93.0], [0, 0, 0]), np.zeros(8), False, None)
        self.assertLess(near, 0.0)
        self.assertAlmostEqual(near, far, delta=0.01)
        self.assertEqual(cost.get_reward(car, BallisticArena([400.0, 0.0, 93.0], [0, 0, 0]), np.zeros(8), False, None), 0.0)

    def test_slide_waste_reads_state_not_inputs(self):
        cost = SlideWasteCost()
        arena = BallisticArena([3000.0, 3000.0, 93.0], [0, 0, 0])
        car = self._car([1200.0, 500.0, 0.0])
        idle = np.zeros(8, dtype=np.float32)
        held = idle.copy()
        held[1], held[7] = 0.3, 1.0
        self.assertLess(cost.get_reward(car, arena, idle, False, None), 0.0)
        self.assertEqual(cost.get_reward(car, arena, idle, False, None), cost.get_reward(car, arena, held, False, None))

    def test_slide_waste_is_smooth_and_spares_real_turns(self):
        cost = SlideWasteCost()
        arena = BallisticArena([3000.0, 3000.0, 93.0], [0, 0, 0])
        straight = cost.get_reward(self._car([1500.0, 0.0, 0.0]), arena, np.zeros(8), False, None)
        turning = cost.get_reward(self._car([1200.0, 500.0, 0.0], (0.0, 0.0, 2.5)), arena, np.zeros(8), False, None)
        self.assertEqual(straight, 0.0)
        self.assertEqual(turning, 0.0)
        a = cost.get_reward(self._car([1200.0, 249.0, 0.0]), arena, np.zeros(8), False, None)
        b = cost.get_reward(self._car([1200.0, 251.0, 0.0]), arena, np.zeros(8), False, None)
        self.assertAlmostEqual(a, b, delta=0.005)


if __name__ == "__main__":
    unittest.main()
