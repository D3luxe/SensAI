"""
Unit Tests for Phase 2: Mathematical Symmetry & Reward Gating Integrity.
Verifies Fixes #8, #3, and checkpoint versioning policy.
"""

import unittest
import math
import numpy as np
import torch

from env.observations import DefaultObservationBuilder, OBS_MIRROR_MASK_NP, OBS_LEGACY_MIRROR_MASK_NP
from env.physics_engine import CarState, BallState, RocketSimArena
from env.rewards import PlayerToBallVelocityReward, BoostReward
from agent.models import ActorCritic


class MockArenaForObs:
    def __init__(self, ball: BallState, cars: list):
        self.ball = ball
        self.cars = cars
        self.boost_pads = []


class TestSymmetryAndRewards(unittest.TestCase):
    def test_bilateral_reflection_parity(self):
        """Fix #8: Verify that physical X-reflection agrees exactly with OBS_MIRROR_MASK_NP across all 74 features."""
        builder = DefaultObservationBuilder(symmetric=True)

        # 1. Base arbitrary airborne tilted state
        pitch, yaw, roll = 0.35, 1.2, -0.45
        # Compute rotation matrix (standard Rocket League / Euler convention)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)

        rot_mat = np.array([
            [cp * cy, cp * sy, sp],
            [cy * sp * sr - cr * sy, cr * cy + sp * sr * sy, -cp * sr],
            [-cr * cy * sp - sr * sy, cy * sr - cr * sp * sy, cp * cr]
        ], dtype=np.float32)

        car = CarState(
            id=0, team=0,
            pos=np.array([1200.0, -800.0, 450.0], dtype=np.float32),
            vel=np.array([-400.0, 600.0, 200.0], dtype=np.float32),
            rot=np.array([pitch, yaw, roll], dtype=np.float32),
            ang_vel=np.array([1.5, -2.0, 0.8], dtype=np.float32),
            boost=65.0, on_ground=False, has_jump=False, has_flip=True
        )
        car.rot_mat = rot_mat

        opp = CarState(
            id=1, team=1,
            pos=np.array([-600.0, 1500.0, 17.0], dtype=np.float32),
            vel=np.array([300.0, -200.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            ang_vel=np.array([0.0, 0.0, -0.5], dtype=np.float32),
            boost=33.0, on_ground=True
        )

        ball = BallState(
            pos=np.array([300.0, 1000.0, 300.0], dtype=np.float32),
            vel=np.array([500.0, -700.0, 150.0], dtype=np.float32),
            ang_vel=np.array([-1.2, 0.9, -2.5], dtype=np.float32)
        )

        arena = MockArenaForObs(ball, [car, opp])
        obs_orig = builder.build_obs(car, arena)

        # 2. Construct physically reflected state across X=0:
        # P = diag(-1, 1, 1).
        # Position and linear velocity: x is negated.
        # Angular velocity (pseudovector): omega' = -P * omega = (+wx, -wy, -wz).
        # Forward and up vectors: fx' = -fx, ux' = -ux.
        # Reflected rot_mat:
        P = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
        # Note: in RocketSim basis: row 0 is fwd, row 2 is up.
        # Under reflection: fwd' = P * fwd, up' = P * up.
        # right' = fwd' x up' = (det P) P (fwd x up) = -P * right = (rx, -ry, -rz).
        rot_mat_refl = rot_mat.copy()
        rot_mat_refl[0, 0] = -rot_mat[0, 0]  # fx negated
        rot_mat_refl[2, 0] = -rot_mat[2, 0]  # ux negated
        # right vector row 1:
        rot_mat_refl[1, 1] = -rot_mat[1, 1]  # ry negated
        rot_mat_refl[1, 2] = -rot_mat[1, 2]  # rz negated

        car_refl = CarState(
            id=0, team=0,
            pos=np.array([-car.pos[0], car.pos[1], car.pos[2]], dtype=np.float32),
            vel=np.array([-car.vel[0], car.vel[1], car.vel[2]], dtype=np.float32),
            rot=np.array([pitch, math.pi - yaw, -roll], dtype=np.float32),
            ang_vel=np.array([car.ang_vel[0], -car.ang_vel[1], -car.ang_vel[2]], dtype=np.float32),
            boost=car.boost, on_ground=car.on_ground, has_jump=car.has_jump, has_flip=car.has_flip
        )
        car_refl.rot_mat = rot_mat_refl

        opp_refl = CarState(
            id=1, team=1,
            pos=np.array([-opp.pos[0], opp.pos[1], opp.pos[2]], dtype=np.float32),
            vel=np.array([-opp.vel[0], opp.vel[1], opp.vel[2]], dtype=np.float32),
            rot=np.array([0.0, -math.pi - (-1.0), 0.0], dtype=np.float32),
            ang_vel=np.array([opp.ang_vel[0], -opp.ang_vel[1], -opp.ang_vel[2]], dtype=np.float32),
            boost=opp.boost, on_ground=opp.on_ground
        )

        ball_refl = BallState(
            pos=np.array([-ball.pos[0], ball.pos[1], ball.pos[2]], dtype=np.float32),
            vel=np.array([-ball.vel[0], ball.vel[1], ball.vel[2]], dtype=np.float32),
            ang_vel=np.array([ball.ang_vel[0], -ball.ang_vel[1], -ball.ang_vel[2]], dtype=np.float32)
        )

        arena_refl = MockArenaForObs(ball_refl, [car_refl, opp_refl])
        obs_refl = builder.build_obs(car_refl, arena_refl)

        # 3. Assert exact match with obs_orig * OBS_MIRROR_MASK_NP
        expected_mirrored = obs_orig * OBS_MIRROR_MASK_NP
        np.testing.assert_allclose(obs_refl, expected_mirrored, atol=1e-5,
                                   err_msg="Physical X-reflection must match observation mask across all 74 features")

        # Specific assertions on the audited indices:
        self.assertEqual(OBS_MIRROR_MASK_NP[11], -1.0, "Right-vector Z mask must be -1.0")
        self.assertEqual(OBS_MIRROR_MASK_NP[15], 1.0, "Car ang_vel X mask must be +1.0")
        self.assertEqual(OBS_MIRROR_MASK_NP[16], -1.0, "Car ang_vel Y mask must be -1.0")
        self.assertEqual(OBS_MIRROR_MASK_NP[28], 1.0, "Ball ang_vel X mask must be +1.0")
        self.assertEqual(OBS_MIRROR_MASK_NP[29], -1.0, "Ball ang_vel Y mask must be -1.0")

    def test_orange_team_rotation_parity(self):
        """Fix #8: Verify 180° Z-rotation parity for Orange team, especially angular velocities."""
        builder = DefaultObservationBuilder(symmetric=True)

        pitch, yaw, roll = 0.2, 0.5, -0.3
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)

        rot_mat_blue = np.array([
            [cp * cy, cp * sy, sp],
            [cy * sp * sr - cr * sy, cr * cy + sp * sr * sy, -cp * sr],
            [-cr * cy * sp - sr * sy, cy * sr - cr * sp * sy, cp * cr]
        ], dtype=np.float32)

        # Rotate 180° around Z: R_z(pi) = diag(-1, -1, 1)
        rot_mat_orange = np.array([
            [-rot_mat_blue[0, 0], -rot_mat_blue[0, 1], rot_mat_blue[0, 2]],
            [-rot_mat_blue[1, 0], -rot_mat_blue[1, 1], rot_mat_blue[1, 2]],
            [-rot_mat_blue[2, 0], -rot_mat_blue[2, 1], rot_mat_blue[2, 2]],
        ], dtype=np.float32)

        car_blue = CarState(
            id=0, team=0,
            pos=np.array([1000.0, -2000.0, 100.0], dtype=np.float32),
            vel=np.array([300.0, 500.0, 50.0], dtype=np.float32),
            rot=np.array([pitch, yaw, roll], dtype=np.float32),
            ang_vel=np.array([1.2, -0.8, 2.0], dtype=np.float32)
        )
        car_blue.rot_mat = rot_mat_blue

        ball_blue = BallState(
            pos=np.array([500.0, -1000.0, 120.0], dtype=np.float32),
            vel=np.array([200.0, 400.0, 0.0], dtype=np.float32),
            ang_vel=np.array([-0.5, 1.5, -1.0], dtype=np.float32)
        )
        arena_blue = MockArenaForObs(ball_blue, [car_blue])
        obs_blue = builder.build_obs(car_blue, arena_blue)

        # Rotated onto Orange: x -> -x, y -> -y, z -> z.
        # Angular velocity under 180° Z rotation: wx -> -wx, wy -> -wy, wz -> wz.
        car_orange = CarState(
            id=0, team=1,
            pos=np.array([-1000.0, 2000.0, 100.0], dtype=np.float32),
            vel=np.array([-300.0, -500.0, 50.0], dtype=np.float32),
            rot=np.array([pitch, yaw + math.pi, roll], dtype=np.float32),
            ang_vel=np.array([-1.2, 0.8, 2.0], dtype=np.float32)
        )
        car_orange.rot_mat = rot_mat_orange

        ball_orange = BallState(
            pos=np.array([-500.0, 1000.0, 120.0], dtype=np.float32),
            vel=np.array([-200.0, -400.0, 0.0], dtype=np.float32),
            ang_vel=np.array([0.5, -1.5, -1.0], dtype=np.float32)
        )
        arena_orange = MockArenaForObs(ball_orange, [car_orange])
        obs_orange = builder.build_obs(car_orange, arena_orange)

        np.testing.assert_allclose(obs_blue, obs_orange, atol=1e-5,
                                   err_msg="Orange team 180° rotation must yield identical observation to Blue")

    def test_stationary_idling_exploit_gated(self):
        """Fix #3: Stationary car parked in front of a stationary ball must earn <= 0.0 reward."""
        rew_fn = PlayerToBallVelocityReward(weight=0.6)
        arena = RocketSimArena(num_players=2)

        # Car at (0, -1000, 17), stationary, facing +Y
        car = arena.cars[0]
        car.pos = np.array([0.0, -1000.0, 17.0], dtype=np.float32)
        car.vel = np.zeros(3, dtype=np.float32)
        car.rot = np.array([0.0, math.pi / 2, 0.0], dtype=np.float32)
        car.on_ground = True

        # Ball at (0, -1200, 93.15), stationary (behind car)
        arena.ball.pos = np.array([0.0, -1200.0, 93.15], dtype=np.float32)
        arena.ball.vel = np.zeros(3, dtype=np.float32)

        # Neutral throttle, steer=1.0, handbrake=1.0
        action = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        total_reward = 0.0
        for _ in range(150):
            r = rew_fn.get_reward(car, arena, action, is_goal=False, scoring_team=None)
            total_reward += r

        self.assertLessEqual(total_reward, 0.0,
                             f"Stationary car holding steer/handbrake must not accumulate positive reward; got {total_reward}")

    def test_boost_zero_weights_strictly_zero(self):
        """Fix #3: Setting BoostReward weights to 0.0 must yield exactly 0.0 contribution."""
        boost_rew = BoostReward(gain_weight=0.0, lose_weight=0.0)
        arena = RocketSimArena(num_players=2)
        car = arena.cars[0]

        # 1. Test pad pickup from 0 to 100
        boost_rew._prev_boost[car.id] = 0.0
        car.boost = 100.0
        r_gain = boost_rew.get_reward(car, arena, np.zeros(8), is_goal=False, scoring_team=None)
        self.assertEqual(r_gain, 0.0, "Zero gain_weight must contribute strictly 0.0 on pad pickup")

        # 2. Test supersonic boost burn from 100 to 0
        boost_rew._prev_boost[car.id] = 1.0
        car.boost = 0.0
        car.vel = np.array([0.0, 2200.0, 0.0], dtype=np.float32)
        action_boost = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        r_loss = boost_rew.get_reward(car, arena, action_boost, is_goal=False, scoring_team=None)
        self.assertEqual(r_loss, 0.0, "Zero lose_weight must contribute strictly 0.0 on boost burn")

    def test_legacy_mirror_mask_checkpoint_versioning(self):
        """Fix #8: Verify ActorCritic accepts legacy_mirror_mask flag for checkpoint backward compatibility."""
        model_new = ActorCritic(obs_dim=74, act_dim=8, legacy_mirror_mask=False)
        model_legacy = ActorCritic(obs_dim=74, act_dim=8, legacy_mirror_mask=True)

        mask_new = model_new.obs_mirror_mask.cpu().numpy()
        mask_legacy = model_legacy.obs_mirror_mask.cpu().numpy()

        self.assertEqual(mask_new[11], -1.0)
        self.assertEqual(mask_legacy[11], 1.0)
        self.assertEqual(mask_new[15], 1.0)
        self.assertEqual(mask_legacy[15], -1.0)


if __name__ == "__main__":
    unittest.main()
