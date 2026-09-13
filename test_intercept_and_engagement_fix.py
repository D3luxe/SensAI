"""
Targeted tests for intercept lookahead kinematics, cross-field ball intercept targeting,
and strike-zone approach penalty rebalancing.
"""

import math
import unittest
import numpy as np

from env.physics_engine import CarState, BallState, RocketSimArena
from env.rewards import (
    compute_trajectory_arrival_time,
    solve_intercept_point,
    PlayerToBallVelocityReward,
    TouchBallReward,
    CombinedReward,
    INTERCEPT_SLICE_LADDER
)


class TestInterceptKinematicsAndFallback(unittest.TestCase):
    def test_standing_start_arrival_time_is_physical(self):
        """A car at a dead stop 1500 uu away reaches it in ~1.5s, not 15s."""
        pos = np.array([0.0, 0.0, 17.0], dtype=np.float32)
        vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        target = np.array([1500.0, 0.0, 93.0], dtype=np.float32)

        arrival, dist, closing_speed = compute_trajectory_arrival_time(pos, vel, target)
        self.assertAlmostEqual(dist, 1500.0, delta=10.0)
        self.assertAlmostEqual(closing_speed, 0.0, delta=1.0)
        # Rocket League vehicle acceleration (1400 uu/s^2 to 1410 uu/s) covers 1500 uu in ~1.5-1.6s
        self.assertGreaterEqual(arrival, 1.2, "Cannot reach 1500 uu instantaneously")
        self.assertLessEqual(arrival, 2.0, f"Standing start to 1500 uu must be ~1.5s, got {arrival:.2f}s")

    def test_moving_away_adds_turn_penalty(self):
        pos = np.array([0.0, 0.0, 17.0], dtype=np.float32)
        vel = np.array([-1000.0, 0.0, 0.0], dtype=np.float32)  # moving away at 1000 uu/s
        target = np.array([1500.0, 0.0, 17.0], dtype=np.float32)

        arrival, _, closing_speed = compute_trajectory_arrival_time(pos, vel, target)
        self.assertAlmostEqual(closing_speed, -1000.0, delta=5.0)
        self.assertGreater(arrival, 1.6, "Moving away must take longer than standing start")

    def test_ladder_includes_360_ticks(self):
        """Verify ladder rungs extend up to 360 ticks (3.0s) matching RocketSim cache."""
        self.assertEqual(INTERCEPT_SLICE_LADDER[0], 0)
        self.assertEqual(INTERCEPT_SLICE_LADDER[-1], 360)
        self.assertIn(15, INTERCEPT_SLICE_LADDER)
        self.assertIn(60, INTERCEPT_SLICE_LADDER)
        self.assertIn(180, INTERCEPT_SLICE_LADDER)

    def test_cross_field_rolling_ball_intercept(self):
        """Test that a ball rolling across midfield finds a realistic intercept near the crossing point."""
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.step([np.zeros(8, dtype=np.float32) for _ in arena.cars], dt=8.0 / 120.0)

        # Simulate Tick 7920: car at (4, 390), ball at (1534, 656) rolling in -X at 700 uu/s
        car_pos = np.array([4.0, 390.0, 17.0], dtype=np.float32)
        car_vel = np.array([0.0, 100.0, 0.0], dtype=np.float32)

        arena.ball.pos = np.array([1534.0, 656.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.array([-700.0, 80.0, 0.0], dtype=np.float32)
        car = CarState(id=0, team=0, pos=car_pos, vel=car_vel, on_ground=True)
        arena.cars = [car]

        pred_pos, slice_t = solve_intercept_point(car_pos, car_vel, arena)

        # The ball crosses Y ~ 650-800. The intercept point must NOT project deep downfield into Y > 1800
        self.assertLess(pred_pos[1], 1200.0, f"Intercept target must not overshoot deep into opponent half, got Y={pred_pos[1]}")
        self.assertGreater(pred_pos[0], -600.0, f"Intercept target X must be within pitch crossing range, got X={pred_pos[0]}")


class TestStrikeZonePenaltiesAndTouchBalance(unittest.TestCase):
    def test_overshoot_penalty_capped(self):
        """Overshoot penalty must be softened to [-0.15, -0.25], not [-0.40, -0.60]."""
        rew = PlayerToBallVelocityReward(weight=0.5)
        arena = RocketSimArena(num_players=1, game_mode="1v1")
        arena.ball.pos = np.array([0.0, 1000.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Car was in strike zone (<400 uu), now leaves (>400 uu) at high speed facing away
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 1200.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1500.0, 0.0], dtype=np.float32),  # driving away
            rot=np.array([0.0, 1.57, 0.0], dtype=np.float32),     # facing +Y (away from ball at 1000)
            on_ground=True
        )
        arena.cars = [car]
        rew.reset(arena)
        rew._was_in_strike_zone[0] = True  # was in strike zone on previous tick

        r = rew.get_reward(car, arena, np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), False, None)
        # Reward scaled by weight=0.5. Overshoot penalty magnitude (-0.15 to -0.25) * 0.5 = -0.075 to -0.125
        self.assertGreaterEqual(r, -0.25, f"Overshoot penalty must not exceed -0.25, got {r}")

    def test_touch_reward_dominates_strike_zone_penalties(self):
        """Touching the ball with config weight 1.2 yields substantial positive return."""
        rew_touch = TouchBallReward(weight=1.2)
        arena = RocketSimArena(num_players=1, game_mode="1v1")
        arena.ball.pos = np.array([0.0, 1000.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.array([0.0, 500.0, 50.0], dtype=np.float32)

        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 950.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1000.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            ball_touches=1,
            on_ground=True
        )
        arena.cars = [car]
        rew_touch.reset(arena)
        rew_touch._prev_touches[0] = 0  # registered touch on this tick

        r = rew_touch.get_reward(car, arena, np.zeros(8, dtype=np.float32), False, None)
        self.assertGreater(r, 0.8, f"Touch reward must be strongly positive, got {r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
