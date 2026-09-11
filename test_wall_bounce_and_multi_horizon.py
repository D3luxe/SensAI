"""
Unit and Integration Tests for Multi-Horizon Ball Prediction and Bounce-Reading Pursuit.
Verifies 80-dimensional observation space, exact reflection symmetry, multi-substep fallback physics,
rebound target positioning, and checkpoint migration.
"""

import math
import unittest
import numpy as np
import torch
import RocketSim as rsim

from env.physics_engine import (
    RocketSimArena, CarState, BallState,
    ARENA_EXTENT_X, ARENA_EXTENT_Y, ARENA_HEIGHT_Z,
    WALL_BOUNCE_VX_THRESHOLD, WALL_BOUNCE_VY_THRESHOLD,
    BALL_PRED_SHORT_TICKS, BALL_PRED_MEDIUM_TICKS,
    PREDICTION_HORIZON_TICKS, PREDICTION_SLICE_COUNT,
    SHOT_THREAT_HORIZON_TICKS, SHOT_THREAT_HORIZON_S
)
from env.observations import (
    OBS_DIM,
    DefaultObservationBuilder,
    OBS_MIRROR_MASK_NP,
    OBS_LEGACY_MIRROR_MASK_NP,
    mirror_obs,
    mirror_act
)
from env.rewards import PlayerToBallVelocityReward
from agent.models import ActorCritic


class MockArenaForObs:
    def __init__(self, ball: BallState, cars: list):
        self.ball = ball
        self.cars = cars
        self.boost_pads = []


class TestWallBounceAndMultiHorizon(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")
        self.obs_builder = DefaultObservationBuilder(symmetric=True)

    def test_observation_dimensions_and_structure(self):
        """Verify DefaultObservationBuilder outputs OBS_DIM features and matches mask dimensions."""
        self.assertEqual(self.obs_builder.obs_dim, OBS_DIM)
        self.assertEqual(len(OBS_MIRROR_MASK_NP), OBS_DIM)
        self.assertEqual(len(OBS_LEGACY_MIRROR_MASK_NP), OBS_DIM)

        car = self.arena.cars[0]
        obs = self.obs_builder.build_obs(car, self.arena)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertFalse(np.isnan(obs).any(), "Observation vector must not contain NaNs")

    def test_reflection_symmetry_across_80_dimensions(self):
        """
        Verify exact bilateral mirror symmetry:
        State A (left) and State B (reflected across X=0) must produce obs_B == mirror_obs(obs_A).
        """
        pitch, yaw, roll = 0.35, 1.2, -0.45
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
        obs_orig = self.obs_builder.build_obs(car, arena)

        # Reflected state
        rot_mat_refl = rot_mat.copy()
        rot_mat_refl[0, 0] = -rot_mat[0, 0]  # fx negated
        rot_mat_refl[2, 0] = -rot_mat[2, 0]  # ux negated
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
        obs_refl = self.obs_builder.build_obs(car_refl, arena_refl)

        expected_mirrored = mirror_obs(obs_orig)
        diff = np.abs(obs_refl - expected_mirrored)
        max_err_idx = int(np.argmax(diff))
        self.assertLess(
            diff[max_err_idx], 1e-4,
            f"Symmetry broken at feature index {max_err_idx}: obs_refl={obs_refl[max_err_idx]}, expected_mirrored={expected_mirrored[max_err_idx]}"
        )

    def test_legacy_mirror_mask_model_construction(self):
        """Verify ActorCritic constructs cleanly with legacy_mirror_mask=True and obs_dim=80."""
        model = ActorCritic(obs_dim=80, act_dim=8, continuous_actions=True, legacy_mirror_mask=True)
        self.assertEqual(model.obs_mirror_mask.shape[0], 80)
        dummy_obs = torch.randn(4, 80)
        action, value, log_prob, _ = model.get_action_and_value(dummy_obs)
        self.assertEqual(action.shape, (4, 8))
        self.assertEqual(value.shape[0], 4)

    def test_pure_python_multi_substep_fallback_physics(self):
        """Verify the multi-substep fallback in get_predicted_ball_pos accurately simulates a wall bounce."""
        # Create pure-Python arena by unsetting _rsim_arena
        py_arena = RocketSimArena(num_players=2, game_mode="1v1")
        py_arena._use_rsim = False
        py_arena._rsim_arena = None

        # Launch ball toward right sidewall (X = +4096)
        py_arena.ball.pos = np.array([3000.0, 0.0, 300.0], dtype=np.float32)
        py_arena.ball.vel = np.array([1600.0, 0.0, 100.0], dtype=np.float32)

        # Slice 60 (0.5s): Ball is approaching wall, X should be near 3800
        p_0_5s = py_arena.get_predicted_ball_pos(60)
        self.assertGreater(p_0_5s[0], 3500.0, "At 0.5s, ball should be near the right sidewall")

        # Slice 180 (1.5s): Ball has bounced off sidewall (X=4096) and is rebounding back toward center
        p_1_5s = py_arena.get_predicted_ball_pos(180)
        self.assertLess(p_1_5s[0], 3600.0, "At 1.5s, ball should have rebounded back inward toward the infield")
        self.assertGreater(p_1_5s[2], 91.0, "Ball must remain above the floor")
        self.assertLess(p_1_5s[2], ARENA_HEIGHT_Z, "Ball must remain below ceiling")

    def test_get_target_pos_wall_bounce_rebound_targeting(self):
        """Verify _get_target_pos detects wall bounces and targets the 1.5s rebound position."""
        rew = PlayerToBallVelocityReward(weight=0.6)
        rew.reset(self.arena)

        # Ball moving at high speed toward the right sidewall
        self.arena.ball.pos = np.array([3200.0, 1000.0, 300.0], dtype=np.float32)
        self.arena.ball.vel = np.array([1500.0, 200.0, 100.0], dtype=np.float32)

        car_pos = np.array([1500.0, 1000.0, 17.0], dtype=np.float32)
        target_pos = rew._get_target_pos(car_pos, self.arena, is_kickoff=False)

        # Target must be bounded within the arena
        self.assertLess(abs(target_pos[0]), ARENA_EXTENT_X)
        self.assertLess(abs(target_pos[1]), ARENA_EXTENT_Y)
        self.assertGreater(target_pos[2], 90.0)

        # Because it's an impending wall bounce, the target blends with the 1.5s rebound,
        # which points back into the infield (X < 4000) rather than pinning against the wall
        self.assertLess(target_pos[0], 3950.0)

    def test_checkpoint_dimension_migration(self):
        """Verify seamless weight migration when loading a 74-dim model checkpoint into an 80-dim model."""
        # Old 74-dim model
        old_model = ActorCritic(obs_dim=74, act_dim=8, continuous_actions=True, use_layer_norm=True)
        old_state_dict = old_model.state_dict()

        # New 80-dim model
        new_model = ActorCritic(obs_dim=80, act_dim=8, continuous_actions=True, use_layer_norm=True)
        new_state_dict = new_model.state_dict()

        # Migrate weights via slice assignment
        for k in list(old_state_dict.keys()):
            if k in new_state_dict:
                saved_param = old_state_dict[k]
                curr_param = new_state_dict[k]
                if saved_param.shape != curr_param.shape:
                    slices = tuple(slice(0, min(s, c)) for s, c in zip(saved_param.shape, curr_param.shape))
                    curr_param[slices] = saved_param[slices]
                    new_state_dict[k] = curr_param
                else:
                    new_state_dict[k] = saved_param

        new_model.load_state_dict(new_state_dict)

        # Ensure new model runs inference on 80-dim observations cleanly
        test_obs = torch.randn(8, 80)
        action, value, log_prob, _ = new_model.get_action_and_value(test_obs)
        self.assertEqual(action.shape, (8, 8))
        self.assertEqual(value.shape[0], 8)
        self.assertFalse(torch.isnan(action).any())


class TestTrainingAndLiveHorizonParity(unittest.TestCase):
    """
    The live bot and the training arena must agree on what a prediction index means.

    Before the v5 migration they did not. The bot halved every index to suit RLBot v4's 60 Hz
    prediction while decaying shot-threat intensity over a hardcoded 120, so the same shot read
    one intensity in training and a different one in a real match, and the 0.5s and 1.5s
    observations were really 0.25s and 0.75s.
    """

    SCENARIOS = [
        ((200.0, -1500.0, 300.0), (-60.0, -1800.0, 120.0), 0, "direct shot at blue net"),
        ((0.0, 2000.0, 500.0), (100.0, 1600.0, -50.0), 1, "lofted shot at orange net"),
        ((3500.0, -1000.0, 600.0), (900.0, 600.0, 200.0), 0, "corner wall rebound"),
        ((-1500.0, -800.0, 93.0), (700.0, -900.0, 300.0), 0, "slow angled roll"),
    ]

    def _arena_pair(self, pos, vel):
        """A training arena and a live MockArena fed the same trajectory."""
        import rlbot_fakes
        from bot import MockArena, PredictionIndexer

        arena = RocketSimArena()
        arena.reset()
        bs = rsim.BallState()
        bs.pos = rsim.Vec(*pos)
        bs.vel = rsim.Vec(*vel)
        arena._rsim_arena.ball.set_state(bs)
        arena._sync_from_rsim()

        # The live side sees the same trajectory as an RLBot v5 prediction: 720 slices at 120 Hz.
        slices = arena._rsim_arena.get_ball_prediction(720)
        fake = rlbot_fakes.ball_prediction([(s.pos.x, s.pos.y, s.pos.z) for s in slices])
        live = MockArena(
            BallState(pos=np.array(pos, dtype=np.float32), vel=np.array(vel, dtype=np.float32)),
            [], predictor=PredictionIndexer(fake.slices)
        )
        return arena, live

    def test_shot_threat_matches_between_training_and_live(self):
        for pos, vel, team, label in self.SCENARIOS:
            with self.subTest(label):
                arena, live = self._arena_pair(pos, vel)
                t_flag, t_int, t_z = arena.get_shot_threat(team)
                l_flag, l_int, l_z = live.get_shot_threat(team)
                self.assertEqual(t_flag, l_flag, f"threat detection disagrees for {label}")
                self.assertAlmostEqual(t_int, l_int, places=4, msg=f"intensity disagrees for {label}")
                self.assertAlmostEqual(t_z, l_z, places=4, msg=f"entry height disagrees for {label}")

    def test_predicted_positions_match_at_every_horizon(self):
        horizons = (BALL_PRED_SHORT_TICKS, BALL_PRED_MEDIUM_TICKS, PREDICTION_HORIZON_TICKS)
        for pos, vel, _team, label in self.SCENARIOS:
            with self.subTest(label):
                arena, live = self._arena_pair(pos, vel)
                for ticks in horizons:
                    train_pos = arena.get_predicted_ball_pos(ticks)
                    live_pos = live.get_predicted_ball_pos(ticks)
                    self.assertLess(
                        float(np.abs(train_pos - live_pos).max()), 1.0,
                        f"{label}: tick {ticks} disagrees between training and live"
                    )

    def test_deepest_horizon_comes_from_the_engine(self):
        """
        The requested array must reach the deepest index any consumer asks for.

        One slice short and that request falls out of range into the approximate pure-Python
        simulator, whose state does not line up with the engine slices beside it.
        """
        self.assertGreater(PREDICTION_SLICE_COUNT, PREDICTION_HORIZON_TICKS)
        self.assertGreaterEqual(PREDICTION_HORIZON_TICKS, BALL_PRED_MEDIUM_TICKS)
        self.assertGreaterEqual(PREDICTION_HORIZON_TICKS, SHOT_THREAT_HORIZON_TICKS)

        arena = RocketSimArena()
        arena.reset()
        bs = rsim.BallState()
        bs.pos = rsim.Vec(1000.0, -2000.0, 800.0)
        bs.vel = rsim.Vec(300.0, -400.0, 250.0)
        arena._rsim_arena.ball.set_state(bs)
        arena._sync_from_rsim()

        arena.get_predicted_ball_pos(PREDICTION_HORIZON_TICKS)
        self.assertIsNotNone(arena._cached_rsim_preds, "engine prediction was not consulted")
        self.assertGreater(
            len(arena._cached_rsim_preds), PREDICTION_HORIZON_TICKS,
            "requested prediction is too short to answer the deepest horizon"
        )

    def test_shot_threat_horizon_is_three_seconds(self):
        """The scan window and the intensity ramp must describe the same span of time."""
        self.assertAlmostEqual(SHOT_THREAT_HORIZON_S, SHOT_THREAT_HORIZON_TICKS / 120.0, places=6)
        self.assertAlmostEqual(SHOT_THREAT_HORIZON_S, 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
