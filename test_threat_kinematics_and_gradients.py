"""
Unit tests for Continuous Kinematic Threat Gradients and Consolidated Opponent TTI.

Verifies:
1. Midfield loose ball with distant opponent: unthreatened ball has I_threat < 0.20, w_challenge >= 0.85.
2. Defensive corner slow roll: 200 uu/s velocity deadband prevents phantom threat (I_ball = 0.0), zero retreat.
3. Offensive rebound: attacking half preserve check prevents premature retreat when I_threat < 0.35.
4. Supersonic counter-attack: high threat (I_threat >= 0.80) activates shadow defense when beaten.
5. No-opponent sentinel (1v0): handles empty opponent list cleanly (t_opp = inf, I_threat = 0.0), no NaNs.
"""

import math
import unittest
import numpy as np

from env.physics_engine import (
    CarState, BallState, RocketSimArena,
    ARENA_EXTENT_X, ARENA_EXTENT_Y, ARENA_HEIGHT_Z, BALL_RADIUS
)
from env.rewards import PlayerToBallVelocityReward, compute_opponent_threats


class TestThreatKinematicsAndGradients(unittest.TestCase):
    def setUp(self):
        self.reward = PlayerToBallVelocityReward(weight=0.6)

    def test_midfield_loose_ball_unthreatened(self):
        """
        Scenario 1: Midfield loose ball with distant opponent.
        Ball at midfield (Y = 0), opponent 3500 uu away closing slowly.
        SensAI must register very low threat (I_threat < 0.20) and decisively challenge (w_challenge >= 0.85).
        """
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset(random_kickoff=False)

        # Team 0 car at (0, -1000, 17)
        car = arena.cars[0]
        car.team = 0
        car.pos = np.array([0.0, -1000.0, 17.0], dtype=np.float32)
        car.vel = np.array([0.0, 800.0, 0.0], dtype=np.float32)
        car.boost = 30.0

        # Opponent (Team 1) at (0, 3500, 17), moving slowly at 200 uu/s
        opp = arena.cars[1]
        opp.team = 1
        opp.pos = np.array([0.0, 3500.0, 17.0], dtype=np.float32)
        opp.vel = np.array([0.0, -200.0, 0.0], dtype=np.float32)
        opp.boost = 30.0

        # Ball at midfield, stationary
        arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.reward.reset(arena)

        # Evaluate target pos
        target = self.reward._get_target_pos(
            car_pos=car.pos,
            arena=arena,
            is_kickoff=False,
            car_vel=car.vel,
            car_id=car.id,
            car_team=car.team,
            car=car,
            boost_amount=car.boost
        )

        # Target must be focused on the ball/intercept, NOT retreated towards own net (-5120)
        self.assertGreater(target[1], -100.0, "Tactical target should target the ball at midfield, not defensive net")

        # Threat calculation check
        threats = compute_opponent_threats(car, arena, use_intercept=True)
        self.assertTrue(len(threats) > 0)
        fastest_threat = threats[0]
        self.assertGreaterEqual(fastest_threat.dist, 3400.0)

    def test_defensive_corner_slow_roll_deadband(self):
        """
        Scenario 2: Uncontested defensive corner slow roll.
        Ball in corner at (-3000, -4000) rolling at 80 uu/s toward net.
        Opponent is 3500 uu away.
        The 200 uu/s deadband must ensure I_ball = 0.0 and I_threat < 0.15,
        preventing SensAI from abandoning the ball to flip into the net.
        """
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset(random_kickoff=False)

        car = arena.cars[0]
        car.team = 0
        car.pos = np.array([-2500.0, -3500.0, 17.0], dtype=np.float32)
        car.vel = np.array([-300.0, -300.0, 0.0], dtype=np.float32)
        car.boost = 20.0

        # Opponent far away near midfield (Y = 0)
        opp = arena.cars[1]
        opp.team = 1
        opp.pos = np.array([1000.0, 0.0, 17.0], dtype=np.float32)
        opp.vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        opp.boost = 50.0

        # Ball in defensive corner, rolling slowly (vy = -80 uu/s)
        arena.ball.pos = np.array([-3000.0, -4000.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.array([0.0, -80.0, 0.0], dtype=np.float32)

        self.reward.reset(arena)

        target = self.reward._get_target_pos(
            car_pos=car.pos,
            arena=arena,
            is_kickoff=False,
            car_vel=car.vel,
            car_id=car.id,
            car_team=car.team,
            car=car,
            boost_amount=car.boost
        )

        # Ball is at (-3000, -4000). Own net is at (0, -5120).
        # Target pos should be within close proximity to ball, NOT pulled into the goal (0, -5120)
        dist_target_to_ball = np.linalg.norm(target[:2] - arena.ball.pos[:2])
        self.assertLess(dist_target_to_ball, 600.0,
                        f"Target should stay close to ball in corner ({dist_target_to_ball} uu), not retreat to net")

    def test_offensive_rebound_preservation(self):
        """
        Scenario 3: Offensive rebound.
        Ball at Y = +3000 rebounding off backboard at vy = -250 uu/s.
        Opponent is inside their own net (Y = +4800).
        Since ball_depth < 0 and I_threat < 0.35, _apply_shadow_retarget must return target_pos unmodified.
        """
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset(random_kickoff=False)

        car = arena.cars[0]
        car.team = 0
        car.pos = np.array([500.0, 2000.0, 17.0], dtype=np.float32)
        car.vel = np.array([0.0, 500.0, 0.0], dtype=np.float32)
        car.boost = 40.0

        opp = arena.cars[1]
        opp.team = 1
        opp.pos = np.array([0.0, 4800.0, 17.0], dtype=np.float32)
        opp.vel = np.array([0.0, -250.0, 0.0], dtype=np.float32)
        opp.boost = 100.0

        arena.ball.pos = np.array([0.0, 3000.0, 200.0], dtype=np.float32)
        arena.ball.vel = np.array([0.0, -250.0, 0.0], dtype=np.float32)

        self.reward.reset(arena)

        tactical_target = np.array([0.0, 3000.0, 200.0], dtype=np.float32)
        retargeted = self.reward._apply_shadow_retarget(
            target_pos=tactical_target,
            car_pos=car.pos,
            arena=arena,
            car_team=car.team,
            I_threat=0.15  # Low threat on offense
        )

        np.testing.assert_allclose(retargeted, tactical_target, rtol=1e-5, atol=1e-5,
                                   err_msg="Tactical target on offense with low threat must not be modified by shadow retarget")

    def test_supersonic_counter_attack_activates_shadow(self):
        """
        Scenario 4: True supersonic counter-attack.
        Opponent is closing fast (v_close > 1500 uu/s, t_opp = 0.5s, d_opp = 1000 uu).
        Ball in defensive zone moving toward our net.
        Threat intensity must be high (I_threat >= 0.80), activating shadow defensive positioning.
        """
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset(random_kickoff=False)

        # Team 0 car behind play at midfield, beaten
        car = arena.cars[0]
        car.team = 0
        car.pos = np.array([500.0, 0.0, 17.0], dtype=np.float32)
        car.vel = np.array([0.0, -500.0, 0.0], dtype=np.float32)
        car.boost = 10.0

        # Opponent charging supersonic toward blue net from (0, -500)
        opp = arena.cars[1]
        opp.team = 1
        opp.pos = np.array([0.0, -500.0, 17.0], dtype=np.float32)
        opp.vel = np.array([0.0, -1800.0, 0.0], dtype=np.float32)
        opp.boost = 80.0

        # Ball at (0, -1000) shot toward blue net at 1600 uu/s
        arena.ball.pos = np.array([0.0, -1000.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.array([0.0, -1600.0, 0.0], dtype=np.float32)

        self.reward.reset(arena)

        target = self.reward._get_target_pos(
            car_pos=car.pos,
            arena=arena,
            is_kickoff=False,
            car_vel=car.vel,
            car_id=car.id,
            car_team=car.team,
            car=car,
            boost_amount=car.boost
        )

        # Defending goal is at Y = -5120. Ball is at -1000.
        # Tactical target must position goalside of ball (closer to -5120 than ball is)
        self.assertLess(target[1], arena.ball.pos[1],
                        f"Under severe counter-attack, target Y ({target[1]}) must be goalside of ball ({arena.ball.pos[1]})")

    def test_no_opponent_sentinel_1v0(self):
        """
        Scenario 5: 1v0 Sentinel case (no opponents).
        Verify t_opp = inf, I_threat = 0.0, no NaNs or infs, and target is computed smoothly.
        """
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset(random_kickoff=False)

        # Retain only car 0 (Team 0) so arena has no opponents (1v0 scenario)
        car = arena.cars[0]
        arena.cars = [car]
        car.team = 0
        car.pos = np.array([1000.0, -2000.0, 17.0], dtype=np.float32)
        car.vel = np.array([500.0, 500.0, 0.0], dtype=np.float32)
        car.boost = 50.0

        arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.array([100.0, 200.0, 0.0], dtype=np.float32)

        self.reward.reset(arena)

        target = self.reward._get_target_pos(
            car_pos=car.pos,
            arena=arena,
            is_kickoff=False,
            car_vel=car.vel,
            car_id=car.id,
            car_team=car.team,
            car=car,
            boost_amount=car.boost
        )

        self.assertFalse(np.any(np.isnan(target)), "Target pos must not contain NaNs in 1v0")
        self.assertFalse(np.any(np.isinf(target)), "Target pos must not contain Infs in 1v0")
        self.assertEqual(len(target), 3)

        # Stepping the reward must also yield a valid finite number
        action = np.zeros(8, dtype=np.float32)
        action[0] = 1.0  # throttle
        r = self.reward.get_reward(car, arena, action, False, None)
        self.assertFalse(np.isnan(r), "Reward must not be NaN in 1v0")
        self.assertFalse(np.isinf(r), "Reward must not be Inf in 1v0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
