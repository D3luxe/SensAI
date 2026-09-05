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
from env.state_setters import DribbleFlickScenarioSetter, WeightedScenarioSetter


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

    def test_jump_bridge_directional_dodge(self):
        """Test that a directional joystick dodge (action[3] > 0.5) is recognized as a directional dodge."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 100.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),  # Facing +Y towards ball
            on_ground=False,
            has_flip=False
        )
        # Ball positioned to the right of car, so a right joystick dodge moves towards objective
        self.arena.ball.pos = np.array([500.0, 0.0, 100.0], dtype=np.float32)
        
        rew._prev_on_ground[car.id] = False
        rew._prev_has_flip[car.id] = True  # Just used flip on this step
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = car.vel.copy()

        # Pure directional dodge (Pitch=0, Yaw=1.0, Roll=0.0)
        action = np.zeros(8, dtype=np.float32)
        action[3] = 1.0
        action[5] = 1.0  # jump

        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertGreater(r, 0.2, f"Directional dodge should be rewarded as an airborne directional dodge, got {r}")

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
        # With strike zone pacing, distance delta is damped by 50% (r = -0.0075 instead of full -0.015 penalty cliff),
        # preventing both severe cliffs and infinite positive reward pumps.
        self.assertGreater(r, -0.01, f"Spacing within strike zone should be softly paced without steep cliff, got {r}")

    def test_elevated_car_on_wall_evaluates_3d_distance(self):
        """Test that climbing vertically on the wall away from a low ball increases 3D distance and incurs a penalty."""
        rew = PlayerToBallVelocityReward(weight=0.6)
        car = CarState(
            id=0, team=0,
            pos=np.array([3600.0, 1000.0, 400.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            vel=np.zeros(3, dtype=np.float32),
            on_ground=True
        )
        self.arena.ball.pos = np.array([3000.0, 2000.0, 93.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)

        # Car drives vertically up the wall from Z=400 to Z=1200 while moving slightly downfield
        # In 3D: (1200 - 93)^2 is much larger than (400 - 93)^2, so true 3D distance grew substantially!
        car.pos = np.array([3600.0, 1200.0, 1200.0], dtype=np.float32)
        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)

        self.assertLess(r, 0.0, f"Climbing vertically away from a grounded ball on the wall must not yield positive distance closure, got {r}")

    def test_ceiling_climb_and_boost_waste_penalty_low_ball(self):
        """Test that riding the ceiling and boosting while the ball is low (Z <= 350) incurs ceiling penalty."""
        rew = PlayerToBallVelocityReward(weight=0.6)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 1000.0, 1950.0], dtype=np.float32),  # On ceiling
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            vel=np.zeros(3, dtype=np.float32),
            on_ground=True
        )
        self.arena.ball.pos = np.array([0.0, 1500.0, 120.0], dtype=np.float32)  # Low ball (not elevated aerial)
        self.arena.cars = [car]
        rew.reset(self.arena)

        action = np.zeros(8, dtype=np.float32)
        action[6] = 1.0  # boosting along ceiling
        r = rew.get_reward(car, self.arena, action, False, None)

        self.assertLess(r, 0.0, f"Boosting across the ceiling while ball is below must be penalized even when ball is low, got {r}")

    def test_boost_waste_penalty_when_climbing_above_ball(self):
        """Test that burning boost when climbing vertically above a lower ball incurs waste penalty."""
        rew = BoostReward(gain_weight=0.6, lose_weight=0.3)
        car = CarState(
            id=0, team=0,
            pos=np.array([3600.0, 1000.0, 600.0], dtype=np.float32),  # On wall at Z=600
            vel=np.array([0.0, 500.0, 400.0], dtype=np.float32),     # Climbing up (vel[2] > 100)
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            boost=50.0,
            on_ground=True
        )
        self.arena.ball.pos = np.array([2500.0, 1000.0, 100.0], dtype=np.float32)  # Ball below at Z=100
        self.arena.cars = [car]
        rew.reset(self.arena)

        car.boost = 45.0  # Spent 5% boost
        action = np.zeros(8, dtype=np.float32)
        action[6] = 1.0   # Actively boosting
        r = rew.get_reward(car, self.arena, action, False, None)

        self.assertLess(r, -0.20, f"Boosting up the wall away from a lower ball must incur vertical climb boost penalty, got {r}")

    def test_wall_crawling_dampened_when_ball_infield(self):
        """Test that distance delta is heavily dampened when car stays on the side wall but ball is in the infield."""
        rew = PlayerToBallVelocityReward(weight=0.6)
        car = CarState(
            id=0, team=0,
            pos=np.array([3600.0, 1000.0, 300.0], dtype=np.float32),  # On side wall
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            vel=np.array([0.0, 100.0, 0.0], dtype=np.float32),
            on_ground=True
        )
        # Ball has bounced into midfield (X = 1500)
        self.arena.ball.pos = np.array([1500.0, 1500.0, 150.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)

        # Car moves 100 uu forward down the wall
        rew._prev_dist[car.id] = 2500.0
        car.pos = np.array([3600.0, 1100.0, 300.0], dtype=np.float32)

        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)

        # Undampened delta would be > 0.10. With 0.15 dampening, it is reduced by 85%.
        self.assertLessEqual(r, 0.025, f"Wall-crawling when ball is infield must be heavily dampened, got {r}")


    def test_powerslide_reward_low_speed_yaw_rate(self):
        """Test that PowerslideReward activates at low speed (120 uu/s) when rapid yaw pivoting occurs with handbrake."""
        rew = PowerslideReward(weight=0.30)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([120.0, 0.0, 0.0], dtype=np.float32),
            ang_vel=np.array([0.0, 0.0, 2.5], dtype=np.float32),  # Rapid yaw pivot
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),      # Facing +X
            on_ground=True
        )
        # Ball is 90 degrees to the left at (0, 500) -> fwd_alignment is ~0.0
        self.arena.ball.pos = np.array([0.0, 500.0, 93.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)

        action = np.zeros(8, dtype=np.float32)
        action[0] = 1.0   # throttle
        action[1] = -1.0  # steer left
        action[7] = 1.0   # handbrake active

        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertGreater(r, 0.05, f"PowerslideReward should reward low-speed yaw cuts with handbrake, got {r}")

    def test_wrong_way_throttle_exempt_when_steering(self):
        """Test that forward throttle while facing away from the ball is NOT penalized if the bot is actively steering to turn."""
        rew = PlayerToBallVelocityReward(weight=0.6)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Facing +X
            vel=np.array([50.0, 0.0, 0.0], dtype=np.float32),
            on_ground=True
        )
        # Ball is behind the car at (-500, 0) -> fwd_alignment = -1.0
        self.arena.ball.pos = np.array([-500.0, 0.0, 93.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)
        rew._prev_dist[car.id] = 500.0

        # Scenario A: driving straight away (steer = 0.0) -> penalized
        act_straight = np.zeros(8, dtype=np.float32)
        act_straight[0] = 1.0
        act_straight[1] = 0.0
        r_straight = rew.get_reward(car, self.arena, act_straight, False, None)

        # Scenario B: turning hard to rotate back to ball (steer = 1.0) -> exempt from wrong-way penalty
        rew.reset(self.arena)
        rew._prev_dist[car.id] = 500.0
        act_turn = np.zeros(8, dtype=np.float32)
        act_turn[0] = 1.0
        act_turn[1] = 1.0
        r_turn = rew.get_reward(car, self.arena, act_turn, False, None)

        self.assertGreater(r_turn, r_straight, "Actively steering to complete a turn must not incur the wrong-way throttle penalty")

    def test_forward_backflip_traversal_restricted(self):
        """Test that backflips are not rewarded for open-field traversal when car is already driving forward."""
        rew = JumpBridgeReward(weight=0.5)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, -1000.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),  # Facing +Y
            vel=np.array([0.0, 800.0, 0.0], dtype=np.float32),        # Moving forward downfield
            on_ground=False,
            has_flip=False,
            just_dodged=True
        )
        # Ball downfield at (0, 2000) -> open field (dist > 650)
        self.arena.ball.pos = np.array([0.0, 2000.0, 93.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)
        rew._prev_has_flip[car.id] = True  # Just flipped

        # Pitch back (backflip input) while driving forward
        act_backflip = np.zeros(8, dtype=np.float32)
        act_backflip[2] = -1.0  # Backflip pitch

        r = rew.get_reward(car, self.arena, act_backflip, False, None)
        # The flip traversal bonus should be penalized (<= 0.0) because car_fwd_speed > 250 and pitch < -0.3
        self.assertLessEqual(r, 0.0, f"Backflipping while traveling forward at speed in open field should be penalized (<= 0.0), got {r}")

    def test_forward_and_diagonal_dodge_traversal_rewarded(self):
        """Test that forward flips and diagonal speed-flips are actively rewarded during open field traversal."""
        from env.rewards import JumpBridgeReward
        from env.physics_engine import CarState

        rew = JumpBridgeReward(weight=0.5)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 50.0], dtype=np.float32),
            vel=np.array([0.0, 1000.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            rot_mat=np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
            on_ground=False,
            has_flip=False
        )
        self.arena.ball.pos = np.array([0.0, 2500.0, 93.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)
        rew._prev_has_flip[car.id] = True  # Just dodged

        # 1. Forward flip (+1.0 pitch)
        act_fwd = np.zeros(8, dtype=np.float32)
        act_fwd[2] = 1.0
        r_fwd = rew.get_reward(car, self.arena, act_fwd, False, None)
        self.assertGreater(r_fwd, 0.3, f"Forward dodge downfield should receive strong traversal reward, got {r_fwd}")

        # 2. Diagonal speed-flip (+0.8 pitch, +0.8 yaw)
        rew._prev_has_flip[car.id] = True  # Re-prime dodge trigger
        act_diag = np.zeros(8, dtype=np.float32)
        act_diag[2] = 0.8
        act_diag[3] = 0.8
        r_diag = rew.get_reward(car, self.arena, act_diag, False, None)
        self.assertGreater(r_diag, 0.4, f"Diagonal speed-flip should receive speed-flip coordination bonus, got {r_diag}")

    def test_uncontested_dribble_overshoot_backflip_penalized(self):
        """Test that backflipping when overshooting an uncontested dribble is penalized and receives no dodge reward."""
        rew = JumpBridgeReward(weight=0.5)
        # Car has overshot ball: car at Y=100, ball at Y=0, car facing +Y (ball behind it)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 100.0, 30.0], dtype=np.float32),
            vel=np.array([0.0, 50.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),  # Facing +Y, ball is at -Y
            on_ground=False,
            has_flip=False
        )
        self.arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)
        # Opponent is far away (uncontested)
        opp = CarState(id=1, team=1, pos=np.array([0.0, 3000.0, 17.0], dtype=np.float32))
        self.arena.cars = [car, opp]
        rew.reset(self.arena)
        rew._prev_has_flip[car.id] = True  # Executing dodge

        # Backflip action: pitch = -1.0
        act_backflip = np.zeros(8, dtype=np.float32)
        act_backflip[2] = -1.0

        r = rew.get_reward(car, self.arena, act_backflip, False, None)
        self.assertLess(r, 0.0, f"Backflipping when overshooting uncontested dribble must be penalized (< 0.0), got {r}")
        self.assertAlmostEqual(r, -0.80 * 0.5, places=3, msg=f"Expected strict -0.40 penalty, got {r}")
        self.assertFalse(rew._halfflip_in_progress.get(car.id, False), "Uncontested dribble overshoot backflip must NOT be tracked as a valid half-flip")

    def test_contested_5050_backflip_permitted(self):
        """Test that backflipping or absorbing with rear bumper during a contested 50/50 is NOT penalized."""
        rew = JumpBridgeReward(weight=0.5)
        # Car at Y=100, ball at Y=0, car facing +Y (ball behind it or at rear), opponent at Y=-200 (within 50/50 range)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 100.0, 30.0], dtype=np.float32),
            vel=np.array([0.0, 50.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=False,
            has_flip=False
        )
        self.arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)
        # Opponent challenging within 300 uu of the ball!
        opp = CarState(id=1, team=1, pos=np.array([0.0, -250.0, 17.0], dtype=np.float32), vel=np.array([0.0, 1000.0, 0.0], dtype=np.float32))
        self.arena.cars = [car, opp]
        rew.reset(self.arena)
        rew._prev_has_flip[car.id] = True

        # Backflip action: pitch = -1.0
        act_backflip = np.zeros(8, dtype=np.float32)
        act_backflip[2] = -1.0

        r = rew.get_reward(car, self.arena, act_backflip, False, None)
        self.assertGreaterEqual(r, 0.0, f"50/50 backflip challenge must NOT be penalized, got {r}")
        self.assertTrue(rew._challenge_jump_active.get(car.id, False), "50/50 backflip challenge should activate _challenge_jump_active")

    def test_dribble_overshoot_braking_and_coasting_rewarded(self):
        """Test that tap-braking and coasting/throttle release are both rewarded when overshooting a dribble."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        # Car at Y=150, ball at Y=0 (local_x < 0, ball behind bumper, dist=150 < 300)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 150.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 200.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),  # Facing +Y
            on_ground=True
        )
        self.arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 150.0, 0.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)

        # 1. Tap-braking: throttle = -1.0
        act_brake = np.zeros(8, dtype=np.float32)
        act_brake[0] = -1.0
        r_brake = rew.get_reward(car, self.arena, act_brake, False, None)

        # 2. Coasting / Throttle release: throttle = 0.0, boost = 0.0
        act_coast = np.zeros(8, dtype=np.float32)
        act_coast[0] = 0.0
        r_coast = rew.get_reward(car, self.arena, act_coast, False, None)

        # 3. Driving away: throttle = 1.0
        act_drive_away = np.zeros(8, dtype=np.float32)
        act_drive_away[0] = 1.0
        r_drive_away = rew.get_reward(car, self.arena, act_drive_away, False, None)

        self.assertGreater(r_brake, 0.0, f"Tap-braking on dribble overshoot should be positive, got {r_brake}")
        self.assertGreater(r_coast, 0.0, f"Coasting/throttle release on dribble overshoot should be positive, got {r_coast}")
        self.assertGreater(r_brake, r_coast, f"Active tap-braking should be rewarded more than passive coasting, got brake={r_brake} vs coast={r_coast}")
        self.assertLess(r_drive_away, 0.0, f"Driving away from ball on overshoot should be penalized, got {r_drive_away}")

    def test_lateral_pocket_pacing_and_cut_in(self):
        """Test that driving alongside the ball downfield awards pocket pacing, hook cuts, and letting ball roll ahead."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        # Car at (0, 0, 17), facing +Y downfield, moving at 800 uu/s
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 800.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),  # Facing +Y
            on_ground=True
        )
        # Ball is on the right hip: local_x = 0, local_y = +100 (in pocket), rolling at 800 uu/s downfield
        # In world space: right vector is +X when facing +Y, so pos = (100, 0, 93)
        self.arena.ball.pos = np.array([100.0, 0.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 800.0, 0.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)

        # 1. Pacing alongside downfield: throttle matching speed (e.g. throttle = 0.5)
        act_pace = np.zeros(8, dtype=np.float32)
        act_pace[0] = 0.5
        r_pace = rew.get_reward(car, self.arena, act_pace, False, None)
        self.assertGreater(r_pace, 0.20, f"Pacing alongside ball downfield in pocket should receive positive pacing reward, got {r_pace}")

        # 2. Hook cut: steering right into the ball (steer = +1.0) with powerslide
        act_cut = np.zeros(8, dtype=np.float32)
        act_cut[0] = 0.5
        act_cut[1] = 1.0  # Steer right into ball
        act_cut[7] = 1.0  # Powerslide
        r_cut = rew.get_reward(car, self.arena, act_cut, False, None)
        self.assertGreater(r_cut, r_pace, f"Executing a hook cut into the ball should yield higher reward than pacing, got {r_cut} vs {r_pace}")

        # 3. Letting ball roll ahead: car slightly ahead (pos_y = 40) outrunning ball (car vel = 950, ball vel = 800)
        car_ahead = CarState(
            id=0, team=0,
            pos=np.array([0.0, 40.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 950.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        act_let_roll = np.zeros(8, dtype=np.float32)
        act_let_roll[0] = 0.0  # Coast to let ball slip forward into 50/50 block
        r_let_roll = rew.get_reward(car_ahead, self.arena, act_let_roll, False, None)
        self.assertGreater(r_let_roll, 0.15, f"Coasting to let ball roll ahead from pocket should be rewarded, got {r_let_roll}")

    def test_speed_differential_overshoot_resolution(self):
        """Test that overshoot rewards dynamically adapt when ball is already overtaking the car."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        # Car at (0, 150, 17), facing +Y. Ball at (0, 0, 93) trailing behind.
        # Case: Ball is rolling FASTER than car (ball_vel = 800, car_vel = 300 -> rel_fwd_speed = -500)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 150.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 300.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 800.0, 0.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)

        # Coasting / feather throttle to receive the overtaking ball smoothly
        act_coast = np.zeros(8, dtype=np.float32)
        act_coast[0] = 0.1
        r_coast = rew.get_reward(car, self.arena, act_coast, False, None)

        # Hard reverse braking (-1.0) when ball is already overtaking rapidly
        act_hard_reverse = np.zeros(8, dtype=np.float32)
        act_hard_reverse[0] = -1.0
        r_hard_reverse = rew.get_reward(car, self.arena, act_hard_reverse, False, None)

        self.assertGreater(r_coast, r_hard_reverse, f"When ball is already overtaking from behind, coasting must be preferred over slamming reverse into it! (got coast={r_coast} vs rev={r_hard_reverse})")

    def test_roof_carry_goal_directed_reward(self):
        """Test that carrying the ball on the roof towards the opponent goal is rewarded, while carrying towards own goal is not."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        # 1. Carrying forward downfield towards opponent goal (+Y)
        car_fwd = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 900.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.ball.pos = np.array([0.0, 15.0, 145.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 900.0, 0.0], dtype=np.float32)
        self.arena.cars = [car_fwd]
        rew.reset(self.arena)

        act = np.zeros(8, dtype=np.float32)
        act[0] = 0.5
        r_fwd = rew.get_reward(car_fwd, self.arena, act, False, None)
        self.assertGreater(r_fwd, 0.35, f"Roof carry towards opponent goal should receive high positive reward, got {r_fwd}")

        # 2. Carrying backward towards own goal (-Y)
        car_bwd = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, -900.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, -math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.ball.pos = np.array([0.0, -15.0, 145.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, -900.0, 0.0], dtype=np.float32)
        self.arena.cars = [car_bwd]
        rew.reset(self.arena)

        r_bwd = rew.get_reward(car_bwd, self.arena, act, False, None)
        self.assertGreater(r_fwd, r_bwd, f"Advancing toward opponent goal must be rewarded much higher than own-goal retreat! (fwd={r_fwd} vs bwd={r_bwd})")

    def test_roof_carry_settling_bonus(self):
        """Test that a settled ball on the roof yields a higher settling bonus than a violently bouncing ball."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 900.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.cars = [car]
        rew.reset(self.arena)
        act = np.zeros(8, dtype=np.float32)

        # Case A: Settled ball (vz = 0)
        self.arena.ball.pos = np.array([0.0, 10.0, 145.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 900.0, 0.0], dtype=np.float32)
        r_settled = rew.get_reward(car, self.arena, act, False, None)

        # Case B: Bouncing ball (vz = 300)
        rew.reset(self.arena)
        self.arena.ball.pos = np.array([0.0, 10.0, 145.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 900.0, 300.0], dtype=np.float32)
        r_bouncing = rew.get_reward(car, self.arena, act, False, None)

        self.assertGreater(r_settled, r_bouncing, f"Settled ball on roof must receive higher reward than bouncing ball (settled={r_settled} vs bouncing={r_bouncing})")

    def test_roof_carry_boost_exemption(self):
        """Test that feathering boost while carrying on roof is exempt from dribble boost penalty."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 900.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            boost=50.0,
            on_ground=True
        )
        self.arena.cars = [car]
        rew.reset(self.arena)

        # Ball on roof
        self.arena.ball.pos = np.array([0.0, 10.0, 145.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 900.0, 0.0], dtype=np.float32)

        act_boost = np.zeros(8, dtype=np.float32)
        act_boost[6] = 1.0  # Boost input
        r_roof = rew.get_reward(car, self.arena, act_boost, False, None)

        # Ball rolling on ground in front of car (not roof carry, prone to dribble boost penalty)
        rew.reset(self.arena)
        self.arena.ball.pos = np.array([0.0, 150.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 900.0, 0.0], dtype=np.float32)
        r_ground = rew.get_reward(car, self.arena, act_boost, False, None)

        self.assertGreater(r_roof, r_ground, f"Roof carry should exempt boost penalty, yielding higher reward than ground dribble boost (roof={r_roof} vs ground={r_ground})")

    def test_flick_window_backflip_unpenalized(self):
        """Test that backward flips during active flick window (e.g. Musty/backflip flick) are not penalized."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 80.0], dtype=np.float32),
            vel=np.array([0.0, 800.0, 100.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=False,
            has_flip=False
        )
        # Ball in flick pocket above car
        self.arena.ball.pos = np.array([0.0, 10.0, 170.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 800.0, 100.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)

        # Setup flick window active
        rew._prev_on_ground[car.id] = False
        rew._prev_has_flip[car.id] = True  # Consumed flip on this tick
        rew._flick_window_active[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = car.vel.copy()

        # Execute backflip dodge (pitch = -1.0)
        act_backflip = np.zeros(8, dtype=np.float32)
        act_backflip[2] = -1.0  # Pitch down
        act_backflip[5] = 1.0   # Jump

        r_flick_backflip = rew.get_reward(car, self.arena, act_backflip, False, None)

        # Compare with uncontested dribble backflip without flick window
        rew.reset(self.arena)
        rew._prev_on_ground[car.id] = False
        rew._prev_has_flip[car.id] = True
        rew._flick_window_active[car.id] = False
        self.arena.ball.pos = np.array([0.0, 500.0, 93.0], dtype=np.float32)  # Ball far away
        r_uncontested_backflip = rew.get_reward(car, self.arena, act_backflip, False, None)

        self.assertGreater(r_flick_backflip, r_uncontested_backflip, f"Flick backflip should NOT be penalized like an erroneous open field backflip! (got flick={r_flick_backflip} vs err={r_uncontested_backflip})")

    def test_flick_launch_impulse_rewarded(self):
        """Test that launching the ball with high impulse towards the target goal during a flick is heavily rewarded."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 80.0], dtype=np.float32),
            vel=np.array([0.0, 900.0, 50.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=False,
            has_flip=False,
            ball_touches=1
        )
        self.arena.cars = [car]
        rew.reset(self.arena)
        rew._prev_on_ground[car.id] = False
        rew._prev_has_flip[car.id] = True
        rew._flick_window_active[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = car.vel.copy()
        rew._prev_ball_vel[car.id] = np.array([0.0, 800.0, 50.0], dtype=np.float32)

        # 1. Ball launched at high speed downfield towards opponent goal
        self.arena.ball.pos = np.array([0.0, 50.0, 160.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 1600.0, 350.0], dtype=np.float32)  # Explosive launch +Y

        act = np.zeros(8, dtype=np.float32)
        act[2] = 1.0  # Front flip
        act[5] = 1.0
        r_launch = rew.get_reward(car, self.arena, act, False, None)
        self.assertGreater(r_launch, 0.8, f"Explosive flick launch toward opponent net must yield strong launch bonus, got {r_launch}")

        # 2. Ball launched backwards toward own goal
        rew.reset(self.arena)
        rew._prev_on_ground[car.id] = False
        rew._prev_has_flip[car.id] = True
        rew._flick_window_active[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = car.vel.copy()
        rew._prev_ball_vel[car.id] = np.array([0.0, -800.0, 50.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, -1600.0, 350.0], dtype=np.float32)  # Launched -Y (own net)

        r_own_net = rew.get_reward(car, self.arena, act, False, None)
        self.assertGreater(r_launch, r_own_net, f"Flick launch toward opponent goal must be rewarded significantly higher than flick toward own net! (launch={r_launch} vs own_net={r_own_net})")

    def test_dribble_flick_scenario_setter(self):
        """Test that DribbleFlickScenarioSetter spawns settled ball on roof with matching speeds."""
        setter = DribbleFlickScenarioSetter()
        setter.reset(self.arena, num_players=2)

        ball = self.arena.ball
        # Identify the dribbler (the car directly under the ball)
        dribbler = min(self.arena.cars, key=lambda c: float(np.linalg.norm(ball.pos[:2] - c.pos[:2])))

        # Check car speed
        car_speed = float(np.linalg.norm(dribbler.vel))
        self.assertGreaterEqual(car_speed, 650.0, f"Car speed should be at least 650, got {car_speed}")
        self.assertLessEqual(car_speed, 1250.0, f"Car speed should be at most 1250, got {car_speed}")

        # Check ball is on roof
        rel_pos = ball.pos - dribbler.pos
        self.assertLess(float(np.linalg.norm(rel_pos[:2])), 80.0, f"Ball should be centered horizontally on car, got dist {np.linalg.norm(rel_pos[:2])}")
        self.assertGreaterEqual(ball.pos[2], 120.0, f"Ball Z should be roof height, got {ball.pos[2]}")
        self.assertLessEqual(ball.pos[2], 200.0, f"Ball Z should be roof height, got {ball.pos[2]}")

        # Check velocity synchronization
        rel_vel = ball.vel - dribbler.vel
        self.assertLess(float(np.linalg.norm(rel_vel[:2])), 40.0, f"Ball and car horizontal velocities should match, got rel_vel {rel_vel}")

        # Check WeightedScenarioSetter integration
        weighted = WeightedScenarioSetter()
        weighted.dribble_flick_prob = 1.0
        weighted.kickoff_prob = 0.0
        weighted.replay_prob = 0.0
        weighted.aerial_prob = 0.0
        weighted.wall_prob = 0.0
        weighted.save_prob = 0.0
        weighted.turnaround_prob = 0.0
        weighted.wall_rebound_prob = 0.0
        weighted.custom_prob = 0.0

        chosen = weighted.reset(self.arena, num_players=2)
        self.assertEqual(chosen, "dribble_flick", f"WeightedScenarioSetter should select dribble_flick, got {chosen}")


    def test_air_roll_touchdown_on_ground_transition(self):
        """Test that touchdown alignment reward fires on ground contact after mid-air disorientation."""
        rew = AirRollRecoveryReward(weight=0.10)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 500.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.cars = [car]
        rew.reset(self.arena)

        # Simulate airborne disoriented frame
        rew._prev_on_ground[car.id] = False
        rew._disoriented_this_flight[car.id] = True
        rew._halfflip_cancel_executed[car.id] = True

        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)

        # Touchdown should award upright landing bonus + speed/heading alignment + half-flip completion
        self.assertGreater(r, 0.15, f"Touchdown after half-flip disorientation recovery must award completion bonus, got {r}")
        self.assertFalse(rew._disoriented_this_flight.get(car.id, False), "Disorientation flag should reset after touchdown")

    def test_roof_carry_negative_local_x_no_penalty(self):
        """Test that carrying the ball on the rear portion of the roof (local_x < 0) does not incur overshoot penalty."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 800.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.cars = [car]
        # Ball settled slightly behind center of mass on the roof (local_x = -20)
        self.arena.ball.pos = np.array([0.0, -20.0, 145.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 800.0, 0.0], dtype=np.float32)
        rew.reset(self.arena)

        act = np.zeros(8, dtype=np.float32)
        act[0] = 1.0  # Full throttle downfield
        r_behind = rew.get_reward(car, self.arena, act, False, None)

        # Ball slightly ahead of center (local_x = +20)
        rew.reset(self.arena)
        self.arena.ball.pos = np.array([0.0, 20.0, 145.0], dtype=np.float32)
        r_ahead = rew.get_reward(car, self.arena, act, False, None)

        self.assertAlmostEqual(r_behind, r_ahead, places=2,
                               msg=f"Roof carry with local_x=-20 ({r_behind}) should not be penalized compared to local_x=+20 ({r_ahead})")

    def test_ball_to_goal_high_crossbar_miss_no_on_target_bonus(self):
        """Test that a high boomer shot flying over the crossbar does not receive on-target multiplier."""
        rew = BallToGoalVelocityReward(weight=1.5)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 2000.0, 17.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32)
        )
        # Ball at Y=3000 moving forward at 1500 uu/s, but climbing high so it hits backwall at Z=1200 (> GOAL_HEIGHT 642)
        self.arena.ball.pos = np.array([0.0, 3000.0, 100.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 1500.0, 1000.0], dtype=np.float32)

        action = np.zeros(8, dtype=np.float32)
        r_high = rew.get_reward(car, self.arena, action, False, None)

        # Low shot straight into net opening (vz = 0, z_impact ~ 100)
        self.arena.ball.vel = np.array([0.0, 1500.0, 0.0], dtype=np.float32)
        r_target = rew.get_reward(car, self.arena, action, False, None)

        self.assertGreater(r_target, r_high * 1.5, f"Direct shot into net opening ({r_target}) should receive on-target multiplier over high crossbar miss ({r_high})")

    def test_flick_launch_angled_goal_shot(self):
        """Test that an angled flick from the flank into the opponent goal opening receives flick power bonus."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([1500.0, 2000.0, 80.0], dtype=np.float32),
            vel=np.array([-500.0, 800.0, 50.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=False,
            has_flip=False,
            ball_touches=1
        )
        self.arena.cars = [car]
        rew.reset(self.arena)
        rew._prev_on_ground[car.id] = False
        rew._prev_has_flip[car.id] = True
        rew._flick_window_active[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = car.vel.copy()
        rew._prev_ball_vel[car.id] = np.array([-400.0, 600.0, 50.0], dtype=np.float32)

        # Flicks diagonally toward net opening at (0, 5120):
        self.arena.ball.pos = np.array([1500.0, 2050.0, 160.0], dtype=np.float32)
        self.arena.ball.vel = np.array([-800.0, 1500.0, 300.0], dtype=np.float32)

        act = np.zeros(8, dtype=np.float32)
        act[2] = 1.0
        act[5] = 1.0
        r = rew.get_reward(car, self.arena, act, False, None)
        self.assertGreater(r, 0.5, f"Angled flick toward goal opening should receive flick power bonus, got {r}")

    def test_strike_zone_opponent_clear_no_whiff_penalty(self):
        """Test that an opponent booming the ball away from the strike zone does NOT penalize the bot with a whiff penalty."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 100.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            ball_touches=0
        )
        opp = CarState(
            id=1, team=1,
            pos=np.array([0.0, 300.0, 17.0], dtype=np.float32),
            ball_touches=0
        )
        self.arena.cars = [car, opp]
        self.arena.ball.pos = np.array([0.0, 200.0, 93.0], dtype=np.float32)  # inside strike zone (< 400)
        rew.reset(self.arena)

        # Step 1: Car is in strike zone
        act = np.zeros(8, dtype=np.float32)
        rew.get_reward(car, self.arena, act, False, None)
        self.assertTrue(rew._was_in_strike_zone[car.id])

        # Step 2: Opponent touches ball and booms it behind the car
        opp.ball_touches = 1
        self.arena.ball.pos = np.array([0.0, -800.0, 300.0], dtype=np.float32)  # Behind car at -Y
        self.arena.ball.vel = np.array([0.0, -2000.0, 500.0], dtype=np.float32)

        r = rew.get_reward(car, self.arena, act, False, None)
        # Should NOT receive the -0.40 to -0.60 whiff penalty because opponent cleared it
        self.assertGreater(r, -0.30, f"Opponent clear should not inflict whiff overshoot penalty, got {r}")

    def test_powerslide_cut_no_handbrake_penalty_overlap(self):
        """Test that a sharp powerslide cut does not incur handbrake economy penalty in CombinedReward."""
        weights = {"powerslide_weight": 0.50}
        combined = CombinedReward(weights)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 400.0, 0.0], dtype=np.float32),
            ang_vel=np.array([0.0, 0.0, 2.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Facing +X
            on_ground=True
        )
        # Ball is off-axis at (0, 500) -> fwd_align = 0.0 (< 0.60)
        self.arena.ball.pos = np.array([0.0, 500.0, 93.0], dtype=np.float32)
        self.arena.cars = [car]
        combined.reset(self.arena)

        act = np.zeros(8, dtype=np.float32)
        act[0] = 0.5
        act[1] = 0.35  # Active cut steering
        act[7] = 1.0   # Handbrake

        total_r, breakdown = combined.get_reward(car, self.arena, act, False, None, include_breakdown=True)
        self.assertNotIn("handbrake_penalty", breakdown, "Active powerslide cut must not receive handbrake economy penalty")
        self.assertGreater(breakdown.get("powerslide", 0.0), 0.0, "PowerslideReward should be active")

    def test_reward_weight_zero_suppresses_bonuses(self):
        """Test that setting touch_weight and boost weights to 0.0 completely suppresses bonuses."""
        touch_rew = TouchBallReward(weight=0.0)
        boost_rew = BoostReward(gain_weight=0.0, lose_weight=0.0)
        car = CarState(id=0, team=0, pos=np.array([0.0, 0.0, 17.0], dtype=np.float32), on_ground=False, ball_touches=1)
        self.arena.ball.pos = np.array([0.0, 50.0, 500.0], dtype=np.float32)
        self.arena.cars = [car]

        touch_rew.reset(self.arena)
        touch_rew._prev_touches[car.id] = 0
        r_touch = touch_rew.get_reward(car, self.arena, np.zeros(8), False, None)
        self.assertEqual(r_touch, 0.0, f"Weight 0.0 must yield 0.0 touch reward, got {r_touch}")

        boost_rew.reset(self.arena)
        car.boost = 100.0
        boost_rew._prev_boost[car.id] = 0.1  # Big pad pickup
        r_boost = boost_rew.get_reward(car, self.arena, np.zeros(8), False, None)
        self.assertEqual(r_boost, 0.0, f"Gain weight 0.0 must yield 0.0 boost reward, got {r_boost}")

    def test_wavedash_low_airborne_impulse(self):
        """Test that executing a dodge while low to the turf onto ground triggers wavedash reward."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1200.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True,
            has_flip=False
        )
        self.arena.ball.pos = np.array([0.0, 2000.0, 93.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)

        # Was airborne low (prev_pos_z = 35 < 55) and dodged
        rew._prev_on_ground[car.id] = False
        rew._prev_pos_z[car.id] = 35.0
        rew._prev_has_flip[car.id] = True  # Consumed flip on landing
        rew._prev_vel[car.id] = np.array([0.0, 950.0, 0.0], dtype=np.float32)  # delta_v = 250 > 120

        act = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, act, False, None)
        self.assertGreater(r, 0.20, f"Wavedash ground impulse should be rewarded, got {r}")


if __name__ == "__main__":
    unittest.main()


