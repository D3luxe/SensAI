"""
Unit tests for BallToGoalVelocityReward Proximity & Custody Falloff.

Guarantees that:
1. Full progression reward is earned when the car is within close custody (dist <= near_dist).
2. Reward is strictly 0.0 when the car is far away (dist >= far_dist), preventing point farming
   on loose balls, wall rebounds, or opponent counter-attacks without engagement.
3. Reward decays smoothly and monotonically across the transition zone.
4. Losing ground within close custody is penalized symmetrically.
5. Custom distance thresholds and dynamic RewardManager weight updates are respected.
"""

import unittest
import numpy as np

from env.physics_engine import CarState, BallState
from env.rewards import BallToGoalVelocityReward, CombinedReward, BALL_MAX_SPEED


class MockArena:
    def __init__(self, cars, ball_pos, ball_vel):
        self.ball = BallState(pos=np.array(ball_pos, dtype=np.float32))
        self.ball.vel = np.array(ball_vel, dtype=np.float32)
        self.cars = cars
        self.boost_pads = []


class TestBallToGoalProximity(unittest.TestCase):
    def setUp(self):
        self.weight = 0.25
        self.near_dist = 500.0
        self.far_dist = 1500.0
        self.reward_fn = BallToGoalVelocityReward(
            weight=self.weight,
            near_dist=self.near_dist,
            far_dist=self.far_dist
        )

    def _eval(self, car_pos, ball_pos, ball_vel):
        car = CarState(id=0, team=0, pos=np.array(car_pos, dtype=np.float32))
        opp = CarState(id=1, team=1, pos=np.array([0.0, 4000.0, 17.0], dtype=np.float32))
        arena = MockArena([car, opp], ball_pos, ball_vel)
        self.reward_fn.reset(arena)
        action = np.zeros(8, dtype=np.float32)
        return self.reward_fn.get_reward(car, arena, action, False, None)

    def test_full_reward_in_close_custody(self):
        """Within near_dist (<= 500 uu), proximity factor is 1.0 (full reward earned)."""
        ball_pos = [0.0, 0.0, 93.0]
        ball_vel = [0.0, 1800.0, 0.0]
        # Car right behind ball: dist = 200 uu (< near_dist = 500 uu)
        car_pos = [0.0, -200.0, 17.0]
        r_custody = self._eval(car_pos, ball_pos, ball_vel)

        self.assertGreater(r_custody, 0.0)
        # Verify it equals unattenuated reward
        expected = self.weight * (1800.0 / BALL_MAX_SPEED)
        self.assertAlmostEqual(r_custody, expected, delta=0.05)

    def test_zero_reward_at_far_distance(self):
        """When car is far from ball (>= far_dist), reward is strictly 0.0 to prevent farming on loose balls."""
        ball_pos = [0.0, 2000.0, 93.0]
        ball_vel = [0.0, 2400.0, 0.0]  # Fast ball rolling into attacking half

        # Car 2500 uu away in defensive half (e.g. watching a stray wall bounce)
        car_pos = [0.0, -500.0, 17.0]
        r_distant = self._eval(car_pos, ball_pos, ball_vel)

        self.assertEqual(
            r_distant, 0.0,
            f"Ball rolling downfield without car custody must yield 0.0, got {r_distant}"
        )

    def test_smooth_monotonic_decay(self):
        """Reward must decay monotonically as distance increases from near_dist to far_dist."""
        ball_pos = [0.0, 0.0, 93.0]
        ball_vel = [0.0, 1500.0, 0.0]

        distances = [200.0, 400.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 1800.0]
        rewards = []
        for d in distances:
            car_pos = [0.0, -d, 93.0]
            rewards.append(self._eval(car_pos, ball_pos, ball_vel))

        # Close custody rewards (d <= 500) should be identical (factor = 1.0)
        self.assertAlmostEqual(rewards[0], rewards[1], places=6)
        self.assertAlmostEqual(rewards[1], rewards[2], places=6)

        # Monotonic decrease through transition zone
        for i in range(2, 6):
            self.assertGreater(
                rewards[i], rewards[i + 1],
                f"Reward at d={distances[i]} ({rewards[i]}) must exceed reward at d={distances[i+1]} ({rewards[i+1]})"
            )

        # At and beyond far_dist (1500 uu), reward is zero
        self.assertEqual(rewards[6], 0.0)
        self.assertEqual(rewards[7], 0.0)

    def test_custody_loss_ground_penalty(self):
        """Losing ground in close custody is penalized; across the pitch it is 0.0."""
        ball_pos = [0.0, 0.0, 93.0]
        ball_vel = [0.0, -1500.0, 0.0]  # Ball moving toward our own net

        # Close custody (lost challenge): should be negative
        r_close_loss = self._eval([0.0, 200.0, 17.0], ball_pos, ball_vel)
        self.assertLess(r_close_loss, 0.0)

        # Distant ball (opponent playing ball in their corner 3000 uu away): should be 0.0
        r_distant_loss = self._eval([0.0, -3000.0, 17.0], ball_pos, ball_vel)
        self.assertEqual(r_distant_loss, 0.0)

    def test_reward_manager_dynamic_update(self):
        """RewardManager initializes with and dynamically updates near_dist and far_dist."""
        weights = {
            "ball_to_goal_weight": 0.25,
            "ball_to_goal_near_dist": 450.0,
            "ball_to_goal_far_dist": 1200.0,
        }
        mgr = CombinedReward(weights)
        b2g = mgr.rewards["ball_to_goal"]
        self.assertIsInstance(b2g, BallToGoalVelocityReward)
        self.assertEqual(b2g.weight, 0.25)
        self.assertEqual(b2g.near_dist, 450.0)
        self.assertEqual(b2g.far_dist, 1200.0)

        # Live update
        mgr.update_weights({
            "ball_to_goal_weight": 0.30,
            "ball_to_goal_near_dist": 600.0,
            "ball_to_goal_far_dist": 1600.0,
        })
        self.assertEqual(b2g.weight, 0.30)
        self.assertEqual(b2g.near_dist, 600.0)
        self.assertEqual(b2g.far_dist, 1600.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
