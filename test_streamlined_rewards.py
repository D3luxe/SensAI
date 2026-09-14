"""
Unit tests for Streamlined Kinematics, Continuous Boost Credit,
Shadow-Line Blending, and Asymmetric Accountability.
"""

import unittest
import math
import numpy as np

from env.physics_engine import CarState, BallState
from env.rewards import (
    compute_trajectory_arrival_time,
    compute_car_arrival_time,
    compute_opponent_threats,
    cached_intercept_point,
    BallToGoalVelocityReward,
    PlayerToBallVelocityReward,
    BALL_MAX_SPEED,
    BALL_RADIUS,
    ARENA_EXTENT_Y,
)


class MockArena:
    def __init__(self, cars, ball_pos, ball_vel=None):
        self.ball = BallState(
            pos=np.array(ball_pos, dtype=np.float32),
            vel=np.array(ball_vel if ball_vel is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        )
        self.cars = cars
        self.step_count = 0
        self.last_step_dt = 8.0 / 120.0

    def get_predicted_ball_pos(self, ticks: int) -> np.ndarray:
        dt = ticks / 120.0
        return self.ball.pos + self.ball.vel * dt


class TestStreamlinedRewards(unittest.TestCase):

    def test_universal_kinematics_and_continuous_boost_credit(self):
        """Universal top speed V_MAX = 2300 uu/s is accessible without boost, and boost provides linear credit."""
        pos = np.array([0.0, 0.0, 17.0], dtype=np.float32)
        vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        target = np.array([0.0, 2000.0, 17.0], dtype=np.float32)

        # 1. Unboosted car accelerating from 0 toward 2300 uu/s at 1400 uu/s^2
        t_unboosted, dist, _ = compute_trajectory_arrival_time(pos, vel, target, boost_amount=0.0)
        self.assertAlmostEqual(dist, 2000.0, delta=1.0)
        self.assertAlmostEqual(t_unboosted, 1.69, delta=0.08)

        # 2. Car with 100 boost: gets full 0.40s credit (dist >= 800 uu)
        t_100_boost, _, _ = compute_trajectory_arrival_time(pos, vel, target, boost_amount=100.0)
        self.assertAlmostEqual(t_unboosted - t_100_boost, 0.40, delta=0.01)

        # 3. Car with 50 boost: gets exactly 0.20s credit (perfect linearity, zero hard gates)
        t_50_boost, _, _ = compute_trajectory_arrival_time(pos, vel, target, boost_amount=50.0)
        self.assertAlmostEqual(t_unboosted - t_50_boost, 0.20, delta=0.01)

        # 4. Small pad collection (+12 boost): smooth ~0.05s credit
        t_12_boost, _, _ = compute_trajectory_arrival_time(pos, vel, target, boost_amount=12.0)
        self.assertAlmostEqual(t_unboosted - t_12_boost, 0.40 * 0.12, delta=0.01)

        # 5. Momentum preservation: car at 2000 uu/s accelerating to 2300 uu/s over 2000 uu
        vel_fast = np.array([0.0, 2000.0, 0.0], dtype=np.float32)
        t_cruising, _, _ = compute_trajectory_arrival_time(pos, vel_fast, target, boost_amount=0.0)
        self.assertAlmostEqual(t_cruising, 0.88, delta=0.05)

    def test_logistic_challenge_weight_properties(self):
        """Verifies logistic challenge sigmoid: 50/50 contest ~0.69, smooth transition, no sharp kinks."""
        def calc_w(delta_t):
            return 1.0 / (1.0 + math.exp(min(20.0, max(-20.0, 8.0 * (delta_t - 0.10)))))

        # Bot clearly first (Delta t = -0.20s): high attack weight
        self.assertGreater(calc_w(-0.20), 0.90)

        # Dead-even 50/50 contest (Delta t = 0.00s): solid contest incentive
        self.assertAlmostEqual(calc_w(0.00), 0.69, delta=0.02)

        # Slightly late (Delta t = +0.10s): midpoint weight
        self.assertAlmostEqual(calc_w(0.10), 0.50, delta=0.01)

        # Decisively beaten (Delta t = +0.35s): smooth decay to shadow defense
        self.assertLess(calc_w(0.35), 0.15)
        self.assertGreater(calc_w(0.35), 0.0)

    def test_no_dead_zone_and_positive_shadow_gradient(self):
        """When beaten, rotating back toward shadow anchor yields positive reward; idling yields zero."""
        rew = PlayerToBallVelocityReward(weight=1.0)

        # Ball on sidewall at midfield
        ball_pos = [-3500.0, 0.0, 93.0]
        ball_vel = [0.0, 0.0, 0.0]

        # Bot at midfield, team 0 (defending -5120)
        bot = CarState(id=0, team=0, pos=np.array([0.0, -500.0, 17.0], dtype=np.float32), vel=np.zeros(3, dtype=np.float32))

        # Opponent is closer to the sidewall ball: bot is beaten
        opp = CarState(id=1, team=1, pos=np.array([-3000.0, 0.0, 17.0], dtype=np.float32), vel=np.zeros(3, dtype=np.float32))

        arena = MockArena([bot, opp], ball_pos, ball_vel)
        rew.reset(arena)

        # Step 1: Bot idles / stalls (vel = 0)
        action_idle = np.zeros(8, dtype=np.float32)
        r_idle = rew.get_reward(bot, arena, action_idle, False, None)
        self.assertAlmostEqual(r_idle, 0.0, delta=0.01, msg="Idling at midfield must yield ~0.0")

        # Step 2: Bot rotates back toward shadow line (moving toward [x_ball*0.5, y_shadow])
        bot_retreating = CarState(
            id=0, team=0,
            pos=np.array([-150.0, -700.0, 17.0], dtype=np.float32),
            vel=np.array([-500.0, -800.0, 0.0], dtype=np.float32)
        )
        arena.cars = [bot_retreating, opp]
        action_move = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r_retreat = rew.get_reward(bot_retreating, arena, action_move, False, None)

        self.assertGreater(r_retreat, 0.0, "Moving toward defensive shadow line must yield active positive reward!")

    def test_target_shift_reward_spike_immunity(self):
        """Target shifting between intercept and shadow anchor must NOT produce reward spikes when car is stationary."""
        rew = PlayerToBallVelocityReward(weight=1.0)

        # Ball is downfield (> 600 uu)
        ball_pos = [0.0, 1500.0, 93.0]
        bot = CarState(id=0, team=0, pos=np.array([0.0, 0.0, 17.0], dtype=np.float32), vel=np.zeros(3, dtype=np.float32))

        # Initial arena: no opponents (target is ball intercept)
        arena1 = MockArena([bot], ball_pos)
        rew.reset(arena1)
        target1 = rew._get_target_pos(bot.pos, arena1, False, bot.vel, bot.id, bot.team)

        # Next tick: opponent suddenly appears closer (w_challenge drops from 1.0 to ~0.1, shifting target)
        opp = CarState(id=1, team=1, pos=np.array([0.0, 1200.0, 17.0], dtype=np.float32), vel=np.zeros(3, dtype=np.float32))
        arena2 = MockArena([bot, opp], ball_pos)

        target2 = rew._get_target_pos(bot.pos, arena2, False, bot.vel, bot.id, bot.team)
        self.assertGreater(np.linalg.norm(target1 - target2), 200.0)

        # Stationary bot evaluates reward across this target shift
        action = np.zeros(8, dtype=np.float32)
        r_shift = rew.get_reward(bot, arena2, action, False, None)

        # Bot did not move, so reward must be strictly 0.0 (no target-shift spike)
        self.assertAlmostEqual(r_shift, 0.0, delta=0.01, msg=f"Target shift produced reward spike: {r_shift}")

    def test_pure_custody_and_asymmetric_accountability_in_ball_to_goal(self):
        """BallToGoalVelocityReward rewards shots at full value when in custody, and penalizes concessions fully without concede immunity."""
        rew = BallToGoalVelocityReward(weight=1.5, near_dist=500.0, far_dist=1500.0)

        # Scenario 1: Striker is shooting on a goalkeeper sitting in the opponent net
        ball_pos = [0.0, 2000.0, 93.0]
        ball_vel = [0.0, 2000.0, 0.0]
        striker = CarState(id=0, team=0, pos=np.array([0.0, 1900.0, 17.0], dtype=np.float32), vel=np.array([0.0, 1800.0, 0.0], dtype=np.float32))
        goalie = CarState(id=1, team=1, pos=np.array([0.0, 5000.0, 17.0], dtype=np.float32), vel=np.zeros(3, dtype=np.float32))

        arena_shot = MockArena([striker, goalie], ball_pos, ball_vel)
        rew.reset(arena_shot)
        action = np.zeros(8, dtype=np.float32)
        r_shot = rew.get_reward(striker, arena_shot, action, False, None)

        expected_shot = 1.5 * (2000.0 / BALL_MAX_SPEED)
        self.assertGreater(r_shot, expected_shot * 0.95, "Shooting on a waiting goalkeeper must earn full progression credit!")

        # Scenario 2: 50/50 defensive loss (ball shoots back toward our own net)
        ball_loss_vel = [0.0, -2000.0, 0.0]
        defender_close = CarState(id=0, team=0, pos=np.array([0.0, 2100.0, 17.0], dtype=np.float32), vel=np.zeros(3, dtype=np.float32))
        arena_loss = MockArena([defender_close, goalie], ball_pos, ball_loss_vel)
        rew.reset(arena_loss)
        r_loss = rew.get_reward(defender_close, arena_loss, action, False, None)

        expected_loss = 1.5 * (-2000.0 / BALL_MAX_SPEED)
        self.assertAlmostEqual(r_loss, expected_loss, delta=0.1, msg="Conceding ball velocity toward our net in custody must not be discounted!")

        # Scenario 3: High-speed challenge from 1800 uu closing fast on a ball rolling at 500 uu/s
        ball_rolling = [0.0, 500.0, 0.0]
        fast_challenger = CarState(id=0, team=0, pos=np.array([0.0, 200.0, 17.0], dtype=np.float32), vel=np.array([0.0, 2200.0, 0.0], dtype=np.float32))
        arena_fast = MockArena([fast_challenger, goalie], ball_pos, ball_rolling)
        rew.reset(arena_fast)
        r_fast_challenge = rew.get_reward(fast_challenger, arena_fast, action, False, None)

        self.assertGreater(r_fast_challenge, 0.0, "High-speed challenge outside far_dist must register TTI custody!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
