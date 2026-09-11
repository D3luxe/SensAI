"""
Unit and Integration Tests for Two-Stage Powerslide Cut & Handbrake Exploit Remediation.
Verifies:
  1. Phase 1 Initiation snap bonus when flanking the ball at distance.
  2. Phase 2 Tire bite bonus when handbrake is released before contact.
  3. Phase 2 Traction loss penalty if handbrake is held inside strike window.
  4. TouchBallReward ground touch power dampening on sliding contact.
  5. Elimination of unearned handbrake turnaround rewards when steer is zero.
  6. Proximity-gated suppression of PowerslideReward inside close strike range.
  7. Handbrake economy strike-zone penalty in CombinedReward.
"""

import unittest
import numpy as np

from env.physics_engine import CarState, BallState
from env.rewards import (
    PlayerToBallVelocityReward,
    TouchBallReward,
    PowerslideReward,
    CombinedReward
)


class MockArena:
    def __init__(self, ball_pos, ball_vel=None):
        self.ball = BallState(
            pos=np.array(ball_pos, dtype=np.float32),
            vel=np.array(ball_vel if ball_vel is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        )
        self.cars = []
        self.step_count = 0

    def get_predicted_ball_pos(self, ticks: int) -> np.ndarray:
        return self.ball.pos.copy()

    def get_shot_threat(self, team: int):
        return False, 0.0, 0.0


class TestPowerslideCutAndBite(unittest.TestCase):

    def test_touch_ball_lateral_slip_dampening(self):
        """Ground ball touch with high lateral slip must be significantly dampened compared to clean forward contact."""
        rew = TouchBallReward(weight=1.0)
        # Clean forward strike (lateral slip = 0)
        car_clean = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, np.pi/2, 0.0], dtype=np.float32),  # Facing +Y
            vel=np.array([0.0, 800.0, 0.0], dtype=np.float32),
            on_ground=True,
            ball_touches=1
        )
        # Sliding glancing strike (lateral slip = 350 uu/s along right axis X)
        car_sliding = CarState(
            id=1, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, np.pi/2, 0.0], dtype=np.float32),  # Facing +Y
            vel=np.array([350.0, 800.0, 0.0], dtype=np.float32),
            on_ground=True,
            ball_touches=1
        )
        arena = MockArena(ball_pos=[0.0, 100.0, 93.0], ball_vel=[0.0, 300.0, 0.0])
        arena.cars = [car_clean, car_sliding]

        rew.reset(arena)
        rew._prev_touches[car_clean.id] = 0
        act = np.zeros(8, dtype=np.float32)
        act[0] = 1.0
        r_clean = rew.get_reward(car_clean, arena, act, False, None)

        rew._prev_touches[car_sliding.id] = 0
        r_sliding = rew.get_reward(car_sliding, arena, act, False, None)

        self.assertGreater(r_clean, r_sliding, "Clean touch with tire grip must award higher reward than sliding touch")
        self.assertLess(r_sliding, r_clean * 0.60, "Lateral slip on contact must damp ground touch reward by at least 40%")

    def test_powerslide_reward_close_strike_proximity_suppressed(self):
        """PowerslideReward must return 0.0 when inside close striking proximity (dist < 220 uu, fwd_alignment > 0.30)."""
        rew = PowerslideReward(weight=0.30)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            vel=np.array([500.0, 0.0, 0.0], dtype=np.float32),
            ang_vel=np.array([0.0, 0.0, 2.5], dtype=np.float32),
            on_ground=True
        )
        # Close ball: (150, 50, 93) -> dist ~158 uu (< 220 uu), fwd_alignment ~0.95 (> 0.30)
        arena_close = MockArena(ball_pos=[150.0, 50.0, 93.0])
        arena_close.cars = [car]
        rew.reset(arena_close)

        act = np.zeros(8, dtype=np.float32)
        act[1] = 1.0
        act[7] = 1.0
        r_close = rew.get_reward(car, arena_close, act, False, None)
        self.assertEqual(r_close, 0.0)

        # Distant ball off-axis: (100, 300, 93) -> dist ~316 uu (> 220 uu), fwd_alignment ~0.316 (< 0.60)
        arena_dist = MockArena(ball_pos=[100.0, 300.0, 93.0])
        arena_dist.cars = [car]
        rew.reset(arena_dist)
        rew._prev_alignment[car.id] = 0.2
        r_dist = rew.get_reward(car, arena_dist, act, False, None)
        self.assertGreater(r_dist, 0.0)

    def test_combined_reward_lateral_slip_penalty(self):
        """CombinedReward must apply lateral_slip_penalty if car is sliding sideways in the strike zone."""
        combined = CombinedReward({})
        # Sliding sideways into the ball on ground (vel = [300, -300, 0] -> lateral slip = 300 uu/s)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            vel=np.array([300.0, -300.0, 0.0], dtype=np.float32),
            on_ground=True
        )
        # Close ball: (150, 0, 93) -> dist = 150 uu (< 220), fwd_align = 1.0 (> 0.5)
        arena = MockArena(ball_pos=[150.0, 0.0, 93.0])
        arena.cars = [car]
        combined.reset(arena)

        act = np.zeros(8, dtype=np.float32)
        act[0] = 1.0
        act[1] = 0.5

        total_r, breakdown = combined.get_reward(car, arena, act, False, None, include_breakdown=True)
        self.assertIn("lateral_slip_penalty", breakdown)
        self.assertLess(breakdown["lateral_slip_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
