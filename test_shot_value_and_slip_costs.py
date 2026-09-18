"""
Guarantees for attacking shot value and the smooth, state-based slide costs.

  1. Deep in the attacking third a no-angle smash into the backboard is valued near 0, a center
     into the slot and a shot on target near 1, from both the predicted and the ballistic path.
  2. From the same tight-angle spot, TouchBallReward pays the center and the shot on target more
     than the backboard smash, and ball_to_goal keeps only B2G_OFF_TARGET_FLOOR of an off-target rate.
  3. Outside the attacking third and on kickoff touches nothing changes.
  4. An on-target ball is only paid as a shot when the defence cannot reach the entry point
     first, which is what makes a long shot a pass; a goal that goes in is never discounted.
  5. A committed miss -- drove at the ball, got inside touching range, carried itself past
     without a touch -- is charged once, and only when it was really a miss.
  6. LateralSlipCost and SlideWasteCost have no cliffs at their old thresholds, and SlideWasteCost
     reads only car state: the same state with different inputs costs the same.
"""

import unittest
from unittest import mock

import numpy as np

import env.rewards as rewards
from env.physics_engine import ARENA_EXTENT_Y, BallState, CarState
from env.rewards import (
    B2G_OFF_TARGET_FLOOR, SHOT_CLEARANCE_FLOOR, SHOT_COVERED_FLOOR, BallToGoalVelocityReward, GoalReward,
    LateralSlipCost, OvershootCost, SlideWasteCost, TouchBallReward, attacking_shot_value, attacking_third_weight,
    centering_factor, covered_shot_scale, goal_mouth_open_angle, on_target_factor,
    project_ball_to_endline, shot_clearance,
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


class TestShotClearance(unittest.TestCase):
    """Blue shoots at +Y from `dist` uu out; the defender sits at (opp_x, opp_y)."""

    def _arena(self, dist, opp_x=0.0, opp_y=4900.0, vx=0.0):
        y = GOAL_Y - dist
        arena = BallisticArena([0.0, y, 93.0], [vx, 2300.0, 60.0])
        car = CarState(id=0, team=0, pos=np.array([0.0, y - 120.0, 17.0], dtype=np.float32),
                       vel=np.array([0.0, 900.0, 0.0], dtype=np.float32), rot=np.array([0.0, 1.5708, 0.0]))
        arena.cars = [car, CarState(id=1, team=1, pos=np.array([opp_x, opp_y, 17.0], dtype=np.float32))]
        return arena, car

    def test_endline_projection_reports_flight_time(self):
        arena, _ = self._arena(2300.0)
        x_i, z_i, t_i = project_ball_to_endline(arena, GOAL_Y)
        self.assertAlmostEqual(abs(x_i), 0.0, delta=20.0)
        self.assertAlmostEqual(t_i, 1.0, delta=0.15)  # 2300 uu at 2300 uu/s
        self.assertGreater(z_i, 0.0)

    def test_defender_on_the_entry_point_floors_it(self):
        arena, car = self._arena(5500.0)
        self.assertAlmostEqual(shot_clearance(car, arena, GOAL_Y), SHOT_CLEARANCE_FLOOR, delta=0.02)

    def test_the_same_shot_is_clear_with_the_defender_upfield(self):
        arena, car = self._arena(5500.0, opp_y=-2000.0)
        self.assertGreater(shot_clearance(car, arena, GOAL_Y), 0.95)

    def test_placement_away_from_the_defender_earns_clearance(self):
        centred, car_c = self._arena(1200.0)
        far_post, car_f = self._arena(1200.0, vx=1100.0)
        self.assertGreater(shot_clearance(car_f, far_post, GOAL_Y), shot_clearance(car_c, centred, GOAL_Y) + 0.2)

    def test_clearance_is_smooth_in_defender_position(self):
        vals = []
        for opp_y in np.linspace(4900.0, 1000.0, 200):
            arena, car = self._arena(3000.0, opp_y=opp_y)
            vals.append(shot_clearance(car, arena, GOAL_Y))
        self.assertLess(max(abs(b - a) for a, b in zip(vals, vals[1:])), 0.05)

    def test_a_ball_not_on_frame_keeps_its_full_strike_credit(self):
        arena, car = self._arena(5500.0, vx=1800.0)
        self.assertEqual(on_target_factor(arena, GOAL_Y), 0.0)
        self.assertEqual(covered_shot_scale(car, arena, GOAL_Y, 0.0), 1.0)

    def test_covered_on_target_strike_keeps_the_floor(self):
        arena, car = self._arena(5500.0)
        self.assertAlmostEqual(covered_shot_scale(car, arena, GOAL_Y, 1.0), SHOT_COVERED_FLOOR, delta=0.02)

    def test_touch_pays_a_covered_long_shot_less_than_an_open_one(self):
        def touch(**kw):
            arena, car = self._arena(5500.0, **kw)
            r = TouchBallReward(weight=1.0)
            r.reset(arena)
            car.ball_touches += 1
            return r.get_reward(car, arena, np.zeros(8, dtype=np.float32), False, None)
        covered, open_net = touch(), touch(opp_y=-2000.0)
        # The strike credit takes the floor; the possession payout for touching it does not
        self.assertLess(covered, 0.85 * open_net)
        self.assertGreater(covered, 0.5 * open_net)

    def test_ball_to_goal_keeps_its_progression_but_loses_the_bonus(self):
        covered_arena, covered_car = self._arena(5500.0)
        open_arena, open_car = self._arena(5500.0, opp_y=-2000.0)
        b2g = BallToGoalVelocityReward(weight=1.0)
        covered = b2g.get_reward(covered_car, covered_arena, np.zeros(8), False, None)
        clear = b2g.get_reward(open_car, open_arena, np.zeros(8), False, None)
        self.assertLess(covered, clear)
        self.assertGreater(covered, 0.7 * clear, "progression itself must survive")

    def test_a_goal_is_never_discounted_for_being_covered(self):
        # GoalReward is the only term that pays on the scoring step, and it prices where the ball
        # entered relative to the defenders itself -- shot clearance must not compound with it
        arena, car = self._arena(5500.0)
        goal = GoalReward(goal_weight=5.0)
        goal.reset(arena)
        scored = goal.get_reward(car, arena, np.zeros(8), True, 0)
        self.assertGreaterEqual(scored, 5.0)
        b2g = BallToGoalVelocityReward(weight=1.0)
        self.assertEqual(b2g.get_reward(car, arena, np.zeros(8), True, 0), 0.0)


class TestOvershootCost(unittest.TestCase):
    """Drives a car in a straight line past a ball and sums what the pass costs it."""

    def _drive(self, ball_pos=(0.0, 0.0, 93.0), ball_vel=(0.0, 0.0, 0.0), lateral=150.0,
               speed=1400.0, touch_at=None, opp_touch_at=None, steps=24):
        arena = BallisticArena(list(ball_pos), list(ball_vel))
        car = CarState(id=0, team=0, pos=np.array([lateral, -700.0, 17.0], dtype=np.float32),
                       vel=np.array([0.0, speed, 0.0], dtype=np.float32), rot=np.array([0.0, 1.5708, 0.0]))
        opp = CarState(id=1, team=1, pos=np.array([0.0, 4000.0, 17.0], dtype=np.float32))
        arena.cars = [car, opp]
        cost = OvershootCost(weight=1.0)
        cost.reset(arena)
        dt = 8.0 / 120.0
        charges = []
        for i in range(steps):
            car.pos = car.pos + car.vel * dt
            arena.ball.pos = arena.ball.pos + arena.ball.vel * dt
            arena.step_count += 1
            if touch_at == i:
                car.ball_touches += 1
            if opp_touch_at == i:
                opp.ball_touches += 1
            charges.append(cost.get_reward(car, arena, np.zeros(8, dtype=np.float32), False, None))
        return charges

    def test_committed_miss_is_charged_once(self):
        charges = self._drive()
        charged = [c for c in charges if c != 0.0]
        self.assertEqual(len(charged), 1, charges)
        self.assertLess(charged[0], -0.3)

    def test_a_touch_during_the_approach_is_never_charged(self):
        self.assertEqual([c for c in self._drive(touch_at=8) if c != 0.0], [])

    def test_an_opponent_touch_during_the_approach_is_never_charged(self):
        self.assertEqual([c for c in self._drive(opp_touch_at=8) if c != 0.0], [])

    def test_a_ball_out_of_reach_overhead_is_not_a_miss(self):
        self.assertEqual([c for c in self._drive(ball_pos=(0.0, 0.0, 700.0)) if c != 0.0], [])

    def test_a_ball_that_rolls_away_from_a_slow_car_is_not_a_miss(self):
        # Car barely moving, ball leaving on its own: the separation is not the car's doing
        charges = self._drive(ball_vel=(0.0, 1500.0, 0.0), speed=200.0, lateral=100.0, steps=30)
        self.assertGreater(min(charges, default=0.0), -0.05, charges)

    def test_a_peel_off_across_the_ball_costs_almost_nothing(self):
        # Passing 330 uu wide, never really driving at it
        charges = self._drive(lateral=330.0)
        self.assertGreater(min(charges, default=0.0), -0.1, charges)

    def test_charge_is_smooth_in_how_close_it_came(self):
        # Slowly, so which 15 Hz sample lands closest to the ball does not dominate the sweep
        totals = [sum(self._drive(lateral=x, speed=400.0, steps=70)) for x in np.linspace(100.0, 380.0, 57)]
        self.assertLess(max(abs(b - a) for a, b in zip(totals, totals[1:])), 0.08, totals)


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
