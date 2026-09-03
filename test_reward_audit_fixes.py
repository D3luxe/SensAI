"""
Targeted tests for Reward System Audit Fixes.
Verifies vector normalization under ball, attacking vs defensive clear coordinates,
PBRS conservation, roll dodge detection, wavedash impulse gating, and touchdown descent gating.
"""

import math
import unittest
import numpy as np

from env.physics_engine import CarState, BallState, RocketSimArena, ARENA_EXTENT_Y, GOAL_HALF_WIDTH
from env.rewards import (
    GoalReward, BallToGoalVelocityReward, PlayerToBallVelocityReward,
    TouchBallReward, JumpBridgeReward, BoostReward, PowerslideReward,
    AirRollRecoveryReward, CombinedReward, RewardManager
)


class TestRewardAuditFixes(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")
        self.arena.reset(random_kickoff=False)

    def test_player_to_ball_3d_normalization_under_ball(self):
        """Test that being positioned directly under a grounded ball does NOT cause infinite/exploded vector normalization."""
        rew = PlayerToBallVelocityReward(weight=0.6)
        car = CarState(
            id=0, team=0,
            pos=np.array([1000.0, 1000.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32)
        )
        self.arena.ball.pos = np.array([1000.0, 1000.0, 200.0], dtype=np.float32)
        rew.reset(self.arena)

        # Step reward with car facing forward
        action = np.zeros(8, dtype=np.float32)
        action[0] = 1.0  # throttle
        r = rew.get_reward(car, self.arena, action, False, None)

        self.assertFalse(np.isnan(r), "Reward must not be NaN")
        self.assertFalse(np.isinf(r), "Reward must not be infinite")
        self.assertLess(abs(r), 10.0, f"Reward magnitude should be well bounded, got {r}")

    def test_ball_to_goal_defensive_clear_not_dampened(self):
        """Test that a booming defensive clearance from deep defensive half is not dampened as a wide backwall shot."""
        rew = BallToGoalVelocityReward(weight=1.5)
        car = CarState(
            id=0, team=0,
            pos=np.array([-2000.0, -3500.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32)
        )
        
        # Ball in deep defensive corner moving fast downfield towards opponent half
        self.arena.ball.pos = np.array([-2500.0, -3500.0, 100.0], dtype=np.float32)
        self.arena.ball.vel = np.array([500.0, 2500.0, 300.0], dtype=np.float32)
        
        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)

        # Ball is progressing forward from defensive half at 2500 uu/s
        self.assertGreater(r, 0.4, f"Defensive clear forward should yield strong positive progression reward, got {r}")

    def test_jump_bridge_air_roll_dodge(self):
        """Test that a pure air-roll dodge (action[4] > 0.5) is recognized as a directional dodge."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 100.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),  # Facing +Y towards ball
            on_ground=False,
            has_flip=False
        )
        # Ball positioned to the right of car, so a right roll dodge moves towards objective
        self.arena.ball.pos = np.array([500.0, 0.0, 100.0], dtype=np.float32)
        
        rew._prev_on_ground[car.id] = False
        rew._prev_has_flip[car.id] = True  # Just used flip on this step
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = car.vel.copy()

        # Pure roll dodge (Pitch=0, Yaw=0, Roll=1.0)
        action = np.zeros(8, dtype=np.float32)
        action[4] = 1.0
        action[5] = 1.0  # jump

        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertGreater(r, 0.2, f"Roll dodge should be rewarded as an airborne directional dodge, got {r}")

    def test_jump_bridge_ground_boost_not_falsely_wavedash(self):
        """Test that normal ground boost acceleration is NOT falsely rewarded as a wavedash."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 800.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.ball.pos = np.array([0.0, 2000.0, 93.0], dtype=np.float32)
        
        rew._prev_on_ground[car.id] = True
        rew._prev_has_flip[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = np.array([0.0, 650.0, 0.0], dtype=np.float32)  # delta_v = 150 from boost

        action = np.zeros(8, dtype=np.float32)
        action[6] = 1.0  # boost
        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertEqual(r, 0.0, f"Grounded boost acceleration alone without landing/dodge should yield 0 in JumpBridge, got {r}")

    def test_air_roll_touchdown_bunny_hop_prevention(self):
        """Test that a clean forward jump ascending off the ground does not generate passive touchdown rewards."""
        rew = AirRollRecoveryReward(weight=0.10)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 120.0], dtype=np.float32),
            vel=np.array([0.0, 1200.0, 300.0], dtype=np.float32),  # Ascending (vel_z > 0)
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),  # Facing +Y in travel direction
            on_ground=False
        )
        self.arena.ball.pos = np.array([0.0, 2000.0, 93.0], dtype=np.float32)
        rew._prev_up_z[car.id] = 1.0
        rew._prev_heading[car.id] = 1.0
        rew._airborne_ticks[car.id] = 5
        rew._was_disoriented[car.id] = False

        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertEqual(r, 0.0, f"Clean ascending jump without disorientation should not yield touchdown reward, got {r}")

    def test_touch_ball_defensive_clear_reward(self):
        """Test that a defensive touch clearing the ball downfield is granted positive reward including clear bonus."""
        rew = TouchBallReward(weight=1.2)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, -4000.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            ball_touches=1
        )
        self.arena.ball.pos = np.array([0.0, -3800.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 1500.0, 200.0], dtype=np.float32)
        rew._prev_touches[car.id] = 0

        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertGreater(r, 1.0, f"Defensive clear touch should yield strong positive reward, got {r}")

    def test_5050_challenge_opponent_gate(self):
        """Test that a stationary jump IS rewarded when an opponent is actively challenging within 650 uu."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, -300.0, 17.0], dtype=np.float32),  # 300 uu from ball (<= 450)
            vel=np.zeros(3, dtype=np.float32),  # Stationary!
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Any orientation (e.g. side block)
            on_ground=False  # Just lifted off
        )
        car.vel[2] = 292.0  # Jump liftoff velocity
        self.arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)
        
        # Opponent is close, contesting within 500 uu
        opp = CarState(id=1, team=1, pos=np.array([0.0, 500.0, 17.0], dtype=np.float32))
        self.arena.cars = [car, opp]

        rew._prev_on_ground[car.id] = True
        rew._prev_has_flip[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = np.zeros(3, dtype=np.float32)

        action = np.zeros(8, dtype=np.float32)
        action[5] = 1.0  # jump
        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertGreater(r, 0.3, f"50/50 jump should be rewarded when opponent is challenging within 650 uu, got {r}")

    def test_5050_challenge_uncontested_no_liftoff(self):
        """Test that a stationary jump is NOT rewarded when no opponent is nearby (e.g. overshot in open space)."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 300.0, 17.0], dtype=np.float32),  # Overshot 300 uu past ball
            vel=np.zeros(3, dtype=np.float32),  # Stopped
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),  # Facing away from ball
            on_ground=False  # Lifted off
        )
        car.vel[2] = 292.0
        self.arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)

        # Opponent is far away in defensive half (3000 uu away)
        opp = CarState(id=1, team=1, pos=np.array([0.0, 3000.0, 17.0], dtype=np.float32))
        self.arena.cars = [car, opp]

        rew._prev_on_ground[car.id] = True
        rew._prev_has_flip[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = np.zeros(3, dtype=np.float32)

        action = np.zeros(8, dtype=np.float32)
        action[5] = 1.0  # jump
        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertEqual(r, 0.0, f"Stationary jump with no opponent nearby should receive 0.0, got {r}")

    def test_no_continuous_nose_push_farming(self):
        """Test that pushing a grounded ball forward with nose contact does NOT award velocity-matching bonus."""
        rew = PlayerToBallVelocityReward(weight=0.6)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 1000.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1400.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        # Ball directly on the nose (< 180 uu), grounded (Z=93), matching speed exactly
        self.arena.ball.pos = np.array([0.0, 1140.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 1400.0, 0.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)

        action = np.zeros(8, dtype=np.float32)
        action[0] = 1.0  # full throttle
        r = rew.get_reward(car, self.arena, action, False, None)

        # Distance is constant, velocity matching is blocked for ground nose-pushing
        self.assertAlmostEqual(r, 0.0, places=3, msg=f"Grounded nose-pushing should not farm velocity matching reward, got {r}")

    def test_attacking_backwall_push_zero_reward(self):
        """Test that pushing the ball into the backwall/corner wide of the net in attacking half yields 0 progression."""
        rew = BallToGoalVelocityReward(weight=1.5)
        car = CarState(
            id=0, team=0,
            pos=np.array([2000.0, 3000.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32)
        )
        # Ball in attacking half (Y=3200) pushed straight forward (+Y) into corner/backwall (X=2000 > GOAL_HALF_WIDTH)
        self.arena.ball.pos = np.array([2000.0, 3200.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 1400.0, 0.0], dtype=np.float32)  # Heading straight into X=2000 backwall

        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertEqual(r, 0.0, f"Pushing wide into attacking backwall/corner should yield 0.0 progression reward, got {r}")

    def test_on_target_shot_progression_bonus(self):
        """Test that a shot on target into the net opening receives full on-target progression multiplier."""
        rew = BallToGoalVelocityReward(weight=1.5)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 3000.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32)
        )
        # Ball moving toward center net opening (X=0)
        self.arena.ball.pos = np.array([0.0, 3200.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 1400.0, 0.0], dtype=np.float32)

        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertGreater(r, 0.5, f"On-target shot should receive strong progression bonus, got {r}")

    def test_touch_ball_power_strike_vs_gentle_push(self):
        """Test that a high-speed power strike on goal significantly out-rewards a gentle grounded nose push."""
        rew = TouchBallReward(weight=1.2)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 3000.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1200.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            ball_touches=1,
            on_ground=True
        )
        # 1. Gentle ground push (rel_speed = 0)
        self.arena.ball.pos = np.array([0.0, 3120.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 1200.0, 0.0], dtype=np.float32)
        self.arena.cars = [car]
        rew._prev_touches[car.id] = 0

        action = np.zeros(8, dtype=np.float32)
        r_push = rew.get_reward(car, self.arena, action, False, None)

        # 2. Booming power strike on goal (ball moving 1800 uu/s towards net)
        car.ball_touches = 2
        rew._prev_touches[car.id] = 1
        self.arena.ball.vel = np.array([0.0, 1800.0, 200.0], dtype=np.float32)
        r_strike = rew.get_reward(car, self.arena, action, False, None)

        self.assertGreater(r_strike, r_push * 2.5, f"Power strike ({r_strike}) should heavily out-reward gentle push ({r_push})")

    def test_strike_zone_repositioning_no_penalty_cliff(self):
        """Test that maneuvering/circling around the ball inside the strike zone does NOT incur negative distance penalty."""
        rew = PlayerToBallVelocityReward(weight=0.6)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 1000.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        # Ball 250 uu away (inside strike zone)
        self.arena.ball.pos = np.array([0.0, 1250.0, 93.0], dtype=np.float32)
        rew.reset(self.arena)

        # Car backs off slightly from 250 to 300 uu to angle a cut (prev_dist = 250, curr_dist = 300)
        rew._prev_dist[car.id] = 250.0
        car.pos = np.array([0.0, 950.0, 17.0], dtype=np.float32)  # curr_dist = 300.0

        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertGreaterEqual(r, 0.0, f"Circling/spacing within strike zone should not incur negative distance cliff, got {r}")


if __name__ == "__main__":
    unittest.main()
