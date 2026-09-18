"""
Reach-aware intercept timing.

Covers:
  1. climb_time / wall_surface_weight: height is priced as a jump, double jump or boost-limited
     aerial in open air, and as driving up the surface against a wall -- smoothly throughout.
  2. compute_trajectory_arrival_time: a ball overhead costs the climb on top of the drive.
  3. solve_intercept_point: a dropping ball is met at a height the car can reach, not at a slice
     600 uu up; the meeting time is continuous as the car moves (no rung jumps); an even race reads
     even from both sides; a boostless opponent is not timed to a ball hanging out of its reach.
  4. PlayerToBallVelocityReward._calc_dist: horizontal error under a high ball is priced in full,
     and the metric never moves faster than the ball or car.
"""
import unittest

import numpy as np
import RocketSim as rsim

from env.physics_engine import ARENA_EXTENT_X, CarState, PREDICTION_TICK_RATE, RocketSimArena
from env.rewards import (
    CLIMB_DOUBLE_UU, CLIMB_FREE_UU, PlayerToBallVelocityReward, climb_time, compute_opponent_threats,
    compute_trajectory_arrival_time, solve_intercept_point, wall_surface_weight,
)


def make_car(pos, vel=(0.0, 0.0, 0.0), cid=0, team=0, boost=33.0):
    car = CarState(id=cid, team=team)
    car.pos = np.array(pos, dtype=np.float32)
    car.vel = np.array(vel, dtype=np.float32)
    car.boost = boost
    car.on_ground = pos[2] < 30.0
    car.rot_mat = np.eye(3, dtype=np.float32)
    return car


def arena_with(ball_pos, ball_vel, cars):
    """A RocketSim arena whose engine ball matches the Python one, so predictions are trusted."""
    arena = RocketSimArena(num_players=2, game_mode="1v1")
    arena.ball.pos = np.array(ball_pos, dtype=np.float32)
    arena.ball.vel = np.array(ball_vel, dtype=np.float32)
    b = arena._rsim_arena.ball.get_state()
    b.pos = rsim.Vec(*[float(v) for v in ball_pos])
    b.vel = rsim.Vec(*[float(v) for v in ball_vel])
    b.ang_vel = rsim.Vec(0.0, 0.0, 0.0)
    arena._rsim_arena.ball.set_state(b)
    arena.cars = cars
    return arena


class TestReachModel(unittest.TestCase):
    def test_climb_is_free_near_the_floor_and_smooth_above(self):
        self.assertEqual(climb_time(0.0), 0.0)
        self.assertEqual(climb_time(CLIMB_FREE_UU), 0.0)
        prev = 0.0
        for h in np.arange(0.0, 2000.0, 1.0):
            t = climb_time(h, 33.0)
            self.assertGreaterEqual(t, prev - 1e-12, f"climb time fell at h={h}")
            self.assertLess(t - prev, 0.004, f"climb time jumped at h={h}")
            prev = t

    def test_boost_is_what_brings_a_high_ball_into_reach(self):
        self.assertGreater(climb_time(900.0, 0.0), 3.0, "no boost: a ball 900 uu up is out of reach")
        self.assertLess(climb_time(900.0, 100.0), 1.2, "full tank: an aerial to 900 uu is on")

    def test_ball_overhead_costs_the_climb_on_top_of_the_drive(self):
        car = np.array([0.0, 0.0, 17.0], dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        ground, _, _ = compute_trajectory_arrival_time(car, vel, np.array([600.0, 0.0, 93.0]), boost_amount=33.0)
        high, _, _ = compute_trajectory_arrival_time(car, vel, np.array([600.0, 0.0, 700.0]), boost_amount=33.0)
        self.assertAlmostEqual(high - ground, climb_time(683.0, 33.0), delta=0.03)

    def test_a_ball_up_a_side_wall_is_reached_by_driving_up_it(self):
        car = np.array([ARENA_EXTENT_X - 1000.0, 0.0, 17.0], dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        on_wall = np.array([ARENA_EXTENT_X - 93.0, 0.0, 900.0], dtype=np.float32)
        t_wall, _, _ = compute_trajectory_arrival_time(car, vel, on_wall, boost_amount=0.0)
        # Same height in open air with no boost is out of reach; up the wall it is a drive
        self.assertLess(t_wall, 2.0)
        in_air = np.array([ARENA_EXTENT_X - 1500.0, 0.0, 900.0], dtype=np.float32)
        t_air, _, _ = compute_trajectory_arrival_time(car, vel, in_air, boost_amount=0.0)
        self.assertGreater(t_air, 3.0)

    def test_wall_weight_is_smooth_and_open_in_the_goal_mouth(self):
        prev = wall_surface_weight(np.array([3000.0, 0.0, 300.0]))
        for x in np.arange(3000.0, ARENA_EXTENT_X, 2.0):
            w = wall_surface_weight(np.array([x, 0.0, 300.0]))
            self.assertLess(abs(w - prev), 0.02)
            prev = w
        self.assertAlmostEqual(wall_surface_weight(np.array([ARENA_EXTENT_X - 93.0, 0.0, 300.0])), 1.0)
        self.assertAlmostEqual(wall_surface_weight(np.array([0.0, 5050.0, 300.0])), 0.0,
                               msg="the goal mouth has no wall behind it")
        self.assertAlmostEqual(wall_surface_weight(np.array([2000.0, 5050.0, 300.0])), 1.0)


class TestInterceptSolverReach(unittest.TestCase):
    def test_a_dropping_ball_is_met_within_reach(self):
        # The drop scenario: ball released 1000 uu up, car 470 uu short of it at rest, 33 boost.
        # The old straight-line model aimed at slices 400-630 uu up from the first step.
        car = make_car([-2750.0, -690.0, 17.0])
        arena = arena_with([-2750.0, -220.0, 1000.0], [0.0, 0.0, -1.0], [car])
        pos, t = solve_intercept_point(car.pos, car.vel, arena, boost_amount=car.boost)
        self.assertLess(float(pos[2]) - 17.0, CLIMB_DOUBLE_UU + 60.0, f"met at z={pos[2]:.0f}")
        self.assertGreater(t, 1.0)
        arrival, _, _ = compute_trajectory_arrival_time(car.pos, car.vel, pos, boost_amount=car.boost)
        self.assertLessEqual(arrival, t + 1e-6, "the reported meeting time must be one the car can make")
        self.assertGreater(arrival, t - 2.0 / PREDICTION_TICK_RATE, "and not a rung rounded up past it")

    def test_meeting_time_is_continuous_as_the_car_moves(self):
        # Rung crossings used to move the intercept by up to a whole rung of ball travel
        times = []
        for x in np.arange(-3200.0, -800.0, 20.0):
            car = make_car([x, 0.0, 17.0], [800.0, 0.0, 0.0])
            arena = arena_with([0.0, 0.0, 93.0], [900.0, 300.0, 0.0], [car])
            times.append(solve_intercept_point(car.pos, car.vel, arena, boost_amount=car.boost)[1])
        steps = np.abs(np.diff(times))
        self.assertLess(float(steps.max()), 0.05, f"meeting time jumped by {steps.max():.3f}s")

    def test_an_even_race_reads_even_from_both_sides(self):
        car = make_car([-1500.0, 0.0, 17.0], [600.0, 0.0, 0.0], cid=0, team=0)
        opp = make_car([1500.0, 0.0, 17.0], [-600.0, 0.0, 0.0], cid=1, team=1)
        arena = arena_with([0.0, 0.0, 93.0], [0.0, 700.0, 0.0], [car, opp])
        _, t_self = solve_intercept_point(car.pos, car.vel, arena, boost_amount=car.boost)
        t_opp = compute_opponent_threats(car, arena, use_intercept=True)[0].arrival_time
        self.assertAlmostEqual(t_self, t_opp, delta=1.5 / PREDICTION_TICK_RATE)

    def test_a_boostless_opponent_is_not_timed_to_a_ball_out_of_its_reach(self):
        # Ball rising off a bounce 1000 uu from a boostless opponent. The straight-line model timed it
        # to the ball near its apex; it cannot play it until it comes back down into jump reach.
        car = make_car([0.0, -4000.0, 17.0], cid=0, team=0)
        opp = make_car([1000.0, 2000.0, 17.0], cid=1, team=1, boost=0.0)
        arena = arena_with([0.0, 2000.0, 700.0], [0.0, 0.0, 500.0], [car, opp])
        t_opp = compute_opponent_threats(car, arena, use_intercept=True)[0].arrival_time
        in_reach = None
        for tick in range(1, 361):
            if float(arena.get_predicted_ball_pos(tick)[2]) - 17.0 <= CLIMB_DOUBLE_UU + 100.0:
                in_reach = tick / PREDICTION_TICK_RATE
                break
        self.assertIsNotNone(in_reach)
        self.assertGreater(t_opp, in_reach - 0.1, f"opponent timed at {t_opp:.2f}s, ball in reach at {in_reach:.2f}s")


class TestPursuitDistanceUnderAHighBall(unittest.TestCase):
    def setUp(self):
        self.p2b = PlayerToBallVelocityReward(weight=1.0)

    def test_a_miss_under_a_high_ball_costs_its_full_width(self):
        ball = np.array([0.0, 0.0, 600.0], dtype=np.float32)
        under = self.p2b._calc_dist(np.array([0.0, 0.0, 17.0]), ball)
        wide = self.p2b._calc_dist(np.array([200.0, 0.0, 17.0]), ball)
        self.assertAlmostEqual(wide - under, 200.0, delta=1.0)

    def test_climbing_toward_a_high_ball_still_pays(self):
        ball = np.array([0.0, 0.0, 800.0], dtype=np.float32)
        low = self.p2b._calc_dist(np.array([100.0, 0.0, 17.0]), ball)
        high = self.p2b._calc_dist(np.array([100.0, 0.0, 400.0]), ball)
        self.assertLess(high, low - 300.0)

    def test_the_metric_never_moves_faster_than_the_ball(self):
        car = np.array([300.0, 0.0, 17.0], dtype=np.float32)
        prev = self.p2b._calc_dist(car, np.array([0.0, 0.0, 93.0]))
        for z in np.arange(94.0, 1500.0, 1.0):
            d = self.p2b._calc_dist(car, np.array([0.0, 0.0, z]))
            self.assertLessEqual(abs(d - prev), 1.0 + 1e-6, f"z={z}")
            prev = d


if __name__ == "__main__":
    unittest.main()
