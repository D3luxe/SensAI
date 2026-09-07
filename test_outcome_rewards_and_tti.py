import unittest
import math
import numpy as np
import torch

from env.physics_engine import CarState, BallState
from env.rewards import (
    compute_trajectory_arrival_time,
    compute_car_arrival_time,
    compute_opponent_threats,
    PowerslideReward,
    PlayerToBallVelocityReward,
    TouchBallReward,
    JumpBridgeReward,
    BoostReward,
)
from agent.models import ActorCritic


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


class TestOutcomeRewardsAndTTI(unittest.TestCase):

    def test_kinematics_arrival_time(self):
        car_pos = np.array([0.0, 0.0, 17.0], dtype=np.float32)
        car_vel = np.array([0.0, 1000.0, 0.0], dtype=np.float32)
        ball_pos = np.array([0.0, 1000.0, 93.0], dtype=np.float32)
        ball_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        t_arr, dist, closing = compute_trajectory_arrival_time(car_pos, car_vel, ball_pos, ball_vel)
        self.assertAlmostEqual(t_arr, 1.0, delta=0.05)
        self.assertAlmostEqual(dist, 1000.0, delta=5.0)
        self.assertAlmostEqual(closing, 1000.0, delta=5.0)

        car = CarState(id=0, team=0, pos=car_pos, vel=car_vel)
        t_car = compute_car_arrival_time(car, ball_pos, ball_vel)
        self.assertIsInstance(t_car, float)
        self.assertAlmostEqual(t_car, 1.0, delta=0.05)

        car_vel_perp = np.array([1000.0, 0.0, 0.0], dtype=np.float32)
        t_fallback, _, closing_fallback = compute_trajectory_arrival_time(car_pos, car_vel_perp, ball_pos, ball_vel)
        self.assertTrue(math.isfinite(t_fallback))
        self.assertGreater(t_fallback, 0.0)
        self.assertLess(t_fallback, 5.0)
        self.assertLessEqual(closing_fallback, 50.0)

    def test_powerslide_outcome_driven_no_button_required(self):
        rew = PowerslideReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            vel=np.array([500.0, 0.0, 0.0], dtype=np.float32),
            ang_vel=np.array([0.0, 0.0, 2.5], dtype=np.float32),
            on_ground=True
        )
        arena = MockArena(ball_pos=[0.0, 800.0, 93.0])
        arena.cars = [car]
        rew.reset(arena)

        action = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r = rew.get_reward(car, arena, action, False, None)
        self.assertGreater(r, 0.0, 'Outcome-driven PowerslideReward should award reward without action[7] > 0')

    def test_powerslide_strike_zone_and_tti_suppression(self):
        rew = PowerslideReward(weight=1.0)
        car_close = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            vel=np.array([500.0, 0.0, 0.0], dtype=np.float32),
            ang_vel=np.array([0.0, 0.0, 2.5], dtype=np.float32),
            on_ground=True
        )
        arena_close = MockArena(ball_pos=[0.0, 250.0, 93.0])
        arena_close.cars = [car_close]
        rew.reset(arena_close)
        action = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        r_close = rew.get_reward(car_close, arena_close, action, False, None)
        self.assertEqual(r_close, 0.0, 'PowerslideReward must be 0 within 300 uu')

        car_fast = CarState(
            id=1, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            vel=np.array([0.0, 1800.0, 0.0], dtype=np.float32),
            ang_vel=np.array([0.0, 0.0, 2.0], dtype=np.float32),
            on_ground=True
        )
        arena_fast = MockArena(ball_pos=[0.0, 500.0, 93.0])
        arena_fast.cars = [car_fast]
        rew.reset(arena_fast)
        r_fast = rew.get_reward(car_fast, arena_fast, action, False, None)
        self.assertEqual(r_fast, 0.0, 'PowerslideReward must be 0 when self_tti < 0.40s')

    def test_strike_zone_arrival_pacing_and_anti_stacking(self):
        rew = PlayerToBallVelocityReward(weight=1.0)

        # Matched speed car approaching ball in strike zone (dist 200 uu, closing 250 -> tti = 0.80s < 0.85s)
        car_matched = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            vel=np.array([0.0, 450.0, 0.0], dtype=np.float32),
            on_ground=True
        )
        arena = MockArena(ball_pos=[0.0, 200.0, 17.0], ball_vel=[0.0, 200.0, 0.0])
        arena.cars = [car_matched]
        rew.reset(arena)

        act = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r_matched = rew.get_reward(car_matched, arena, act, False, None)
        # Avoids pacing penalty and earns positive progress
        self.assertGreater(r_matched, 0.0)

        # Overspeed car (1200 uu/s) closing fast on slower ball (200 uu/s) inside 0.40s
        car_overspeed = CarState(
            id=1, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            vel=np.array([0.0, 1200.0, 0.0], dtype=np.float32),
            on_ground=True
        )
        arena_fast = MockArena(ball_pos=[0.0, 200.0, 17.0], ball_vel=[0.0, 200.0, 0.0])
        arena_fast.cars = [car_overspeed]
        rew.reset(arena_fast)
        # Throttle incurs pacing penalty
        act_thr = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r_thr = rew.get_reward(car_overspeed, arena_fast, act_thr, False, None)
        # Braking action: pure outcome-driven, no artificial action bonus
        act_brake = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r_brake = rew.get_reward(car_overspeed, arena_fast, act_brake, False, None)
        self.assertEqual(r_brake, r_thr, 'No artificial input bounty for holding reverse')
        self.assertGreater(r_matched, r_thr, 'Paced approach must exceed overspeeding approach due to penalty avoidance')

    def test_touch_ball_soft_catch_bonus(self):
        rew = TouchBallReward(weight=1.0)

        # Initial state before touch (touches = 0)
        car_init = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 420.0, 0.0], dtype=np.float32),
            ball_touches=0,
            on_ground=True
        )
        arena = MockArena(ball_pos=[0.0, 50.0, 93.0], ball_vel=[0.0, 200.0, 0.0])
        arena.cars = [car_init]
        rew.reset(arena)

        # Soft touch event (touches increments 0 -> 1, relative speed = 220, inside [150, 350] soft catch window)
        car_soft = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 420.0, 0.0], dtype=np.float32),
            ball_touches=1,
            on_ground=True
        )
        act = np.zeros(8, dtype=np.float32)
        r_soft = rew.get_reward(car_soft, arena, act, False, None)

        # Booming touch: relative speed = 1300 >> 350
        car_boom_init = CarState(
            id=1, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1500.0, 0.0], dtype=np.float32),
            ball_touches=0,
            on_ground=True
        )
        arena_boom = MockArena(ball_pos=[0.0, 50.0, 93.0], ball_vel=[0.0, 200.0, 0.0])
        arena_boom.cars = [car_boom_init]
        rew.reset(arena_boom)

        car_boom = CarState(
            id=1, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1500.0, 0.0], dtype=np.float32),
            ball_touches=1,
            on_ground=True
        )
        r_boom = rew.get_reward(car_boom, arena_boom, act, False, None)

        # Soft touch should earn substantial reward (> 0.8) including soft catch bonus
        self.assertGreater(r_soft, 0.8)

    def test_jump_bridge_5050_synchronization(self):
        rew = JumpBridgeReward(weight=1.0)
        bot = CarState(
            id=0, team=0,
            pos=np.array([0.0, -200.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            vel=np.array([0.0, 1000.0, 100.0], dtype=np.float32),
            on_ground=False
        )
        opp_sync = CarState(
            id=1, team=1,
            pos=np.array([0.0, 200.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, -math.pi / 2, 0.0], dtype=np.float32),
            vel=np.array([0.0, -1000.0, 0.0], dtype=np.float32),
            on_ground=True
        )
        bot_prev = CarState(
            id=0, team=0,
            pos=np.array([0.0, -200.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            vel=np.array([0.0, 1000.0, 0.0], dtype=np.float32),
            on_ground=True
        )
        arena_init = MockArena(ball_pos=[0.0, 0.0, 93.0])
        arena_init.cars = [bot_prev, opp_sync]
        rew.reset(arena_init)

        arena = MockArena(ball_pos=[0.0, 0.0, 93.0], ball_vel=[0.0, 0.0, 0.0])
        arena.cars = [bot, opp_sync]
        act = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        r = rew.get_reward(bot, arena, act, False, None)
        self.assertGreater(r, 1.0)

    def test_boost_strike_zone_overspeed_suppression(self):
        rew = BoostReward(gain_weight=0.6, lose_weight=0.3)
        bot_prev = CarState(
            id=0, team=0,
            pos=np.array([0.0, -200.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1200.0, 0.0], dtype=np.float32),
            boost=50.0,
            on_ground=True
        )
        arena = MockArena(ball_pos=[0.0, 0.0, 93.0], ball_vel=[0.0, 100.0, 0.0])
        arena.cars = [bot_prev]
        rew.reset(arena)

        bot_curr = CarState(
            id=0, team=0,
            pos=np.array([0.0, -150.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1300.0, 0.0], dtype=np.float32),
            boost=45.0,
            on_ground=True
        )
        act = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        r = rew.get_reward(bot_curr, arena, act, False, None)
        self.assertLess(r, -0.25)

    def test_handbrake_activation_threshold_logit(self):
        ac = ActorCritic(obs_dim=115, continuous_actions=True)
        handbrake_logit = float(ac.bin_thresh_logits[2])
        self.assertAlmostEqual(handbrake_logit, 0.0800, places=3)
        prob = float(torch.sigmoid(ac.bin_thresh_logits[2]))
        self.assertGreater(prob, 0.519)
        self.assertLess(prob, 0.521)


if __name__ == '__main__':
    unittest.main()
