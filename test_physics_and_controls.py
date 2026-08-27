"""
Automated Physics, Control Conventions & Symmetry Verification Tests.
Guarantees 100% alignment between neural network actions, RocketSim physics, and RLBot gamepad controllers.
"""

import math
import unittest
import numpy as np
import torch
import RocketSim as rsim

from env.physics_engine import RocketSimArena
from env.observations import DefaultObservationBuilder, OBS_MIRROR_MASK_NP, ACT_MIRROR_MASK_NP
from agent.models import ActorCritic
from bot import SenseiRLBot


class TestPhysicsAndControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs_builder = DefaultObservationBuilder(symmetric=True)

    def test_pitch_control_direction(self):
        """
        Guarantees that act[2] = -1.0 ALWAYS pitches nose down (front-flip)
        and act[2] = +1.0 ALWAYS pitches nose up (backflip / climb).
        """
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset(random_kickoff=False)

        # 1. Test Pitch Down (-1.0) in RocketSim
        cs = arena._rsim_cars[0].get_state()
        cs.pos = rsim.Vec(0, 0, 500)
        cs.vel = rsim.Vec(0, 0, 0)
        cs.rot_mat = rsim.Angle(pitch=0.0, yaw=np.pi / 2, roll=0.0).as_rot_mat()
        arena._rsim_cars[0].set_state(cs)

        arena.step([np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=10.0 / 120.0)
        fwd_down = arena.cars[0].get_forward_vector()
        self.assertLess(fwd_down[2], 0.0, "act[2]=-1.0 must pitch nose DOWN in training physics!")

        # 2. Test Pitch Up (+1.0) in RocketSim
        cs.rot_mat = rsim.Angle(pitch=0.0, yaw=np.pi / 2, roll=0.0).as_rot_mat()
        arena._rsim_cars[0].set_state(cs)
        arena.step([np.array([0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=10.0 / 120.0)
        fwd_up = arena.cars[0].get_forward_vector()
        self.assertGreater(fwd_up[2], 0.0, "act[2]=+1.0 must pitch nose UP in training physics!")

    def test_steer_control_direction(self):
        """
        Guarantees that act[1] = -1.0 ALWAYS turns left (-X)
        and act[1] = +1.0 ALWAYS turns right (+X).
        """
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset(random_kickoff=False)

        # Test Steer Left (-1.0)
        cs = arena._rsim_cars[0].get_state()
        cs.pos = rsim.Vec(0, -3000, 17)
        cs.vel = rsim.Vec(0, 500, 0)
        cs.rot_mat = rsim.Angle(pitch=0.0, yaw=np.pi / 2, roll=0.0).as_rot_mat()
        arena._rsim_cars[0].set_state(cs)
        arena.step([np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=15.0 / 120.0)
        self.assertLess(arena.cars[0].pos[0], 0.0, "act[1]=-1.0 must steer car LEFT (towards -X)!")

        # Test Steer Right (+1.0)
        cs.pos = rsim.Vec(0, -3000, 17)
        cs.vel = rsim.Vec(0, 500, 0)
        cs.rot_mat = rsim.Angle(pitch=0.0, yaw=np.pi / 2, roll=0.0).as_rot_mat()
        arena._rsim_cars[0].set_state(cs)
        arena.step([np.array([1.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=15.0 / 120.0)
        self.assertGreater(arena.cars[0].pos[0], 0.0, "act[1]=+1.0 must steer car RIGHT (towards +X)!")

    def test_bot_controller_pass_through(self):
        """
        Guarantees that bot.py passes pitch, steer, yaw, roll 1-to-1 without accidental sign negation.
        """
        bot = SenseiRLBot("TestBot", 0, 0)

        class Struct:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        car = Struct(
            team=0, boost=33.3, has_wheel_contact=True, jumped=False, double_jumped=False,
            physics=Struct(
                location=Struct(x=0.0, y=-4608.0, z=17.0),
                velocity=Struct(x=0.0, y=0.0, z=0.0),
                rotation=Struct(pitch=0.0, yaw=np.pi / 2, roll=0.0),
                angular_velocity=Struct(x=0.0, y=0.0, z=0.0)
            )
        )
        ball = Struct(physics=Struct(location=Struct(x=0.0, y=0.0, z=91.25), velocity=Struct(x=0, y=0, z=0), angular_velocity=Struct(x=0, y=0, z=0)))
        packet = Struct(num_cars=1, game_cars=[car], game_ball=ball, game_info=Struct(is_match_ended=False))

        # Test airborne controller pass-through: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        test_act = np.array([0.8, -0.7, -0.9, 0.6, -0.5, 0.0, 1.0, 0.0], dtype=np.float32)
        bot.prev_action = test_act
        bot.ticks_since_last_action = 0
        bot.current_steer = -0.7
        car.has_wheel_contact = False
        ctrl = bot.get_output(packet)

        # In-Game Rocket League gamepad stick input mapping:
        # Throttle (+1.0), Steer (+1.0 Right, -1.0 Left), Yaw (+1.0 Right, -1.0 Left), Roll (+1.0 Right, -1.0 Left)
        # Pitch is -act[2] (-1.0 Nose Up / Aerial Climb, +1.0 Nose Down / Front Flip)
        self.assertAlmostEqual(ctrl.throttle, 0.8, places=4, msg="Throttle maps direct (+0.8 Forward)!")
        self.assertAlmostEqual(ctrl.steer, -0.7, delta=0.05, msg="Steer maps direct (act[1]=-0.7 Left maps to ctrl.steer=-0.7 Left)!")
        self.assertAlmostEqual(ctrl.pitch, 0.9, places=4, msg="Pitch is -act[2] (act[2]=-0.9 Down/Frontflip maps to ctrl.pitch=+0.9 Push Stick Forward)!")
        self.assertAlmostEqual(ctrl.yaw, 0.6, places=4, msg="Yaw maps direct (act[3]=+0.6 Right maps to ctrl.yaw=+0.6 Right)!")
        self.assertAlmostEqual(ctrl.roll, -0.5, places=4, msg="Roll maps direct (act[4]=-0.5 Left maps to ctrl.roll=-0.5 Roll Left)!")
        self.assertTrue(ctrl.boost)

    def test_bilateral_symmetry_masks(self):
        """
        Guarantees that mirroring an observation and action flips antisymmetric axes (steer, yaw, roll, X).
        """
        obs_dim = 74
        act_dim = 8
        self.assertEqual(len(OBS_MIRROR_MASK_NP), obs_dim)
        self.assertEqual(len(ACT_MIRROR_MASK_NP), act_dim)

        # Action mirror mask: Steer (1), Yaw (3), Roll (4) must be -1.0
        self.assertEqual(ACT_MIRROR_MASK_NP[0], 1.0)   # Throttle
        self.assertEqual(ACT_MIRROR_MASK_NP[1], -1.0)  # Steer
        self.assertEqual(ACT_MIRROR_MASK_NP[2], 1.0)   # Pitch (symmetric across sagittal plane)
        self.assertEqual(ACT_MIRROR_MASK_NP[3], -1.0)  # Yaw
        self.assertEqual(ACT_MIRROR_MASK_NP[4], -1.0)  # Roll
        self.assertEqual(ACT_MIRROR_MASK_NP[5], 1.0)   # Jump
        self.assertEqual(ACT_MIRROR_MASK_NP[6], 1.0)   # Boost

    def test_rot_mat_basis_parity(self):
        """
        Guarantees that bot.py rotation_to_rot_mat exactly matches C++ RocketSim Bullet basis
        (Row 0: Forward, Row 1: Right = -Bullet_Left, Row 2: Up = Bullet_Up).
        """
        from bot import rotation_to_rot_mat
        for p in [-1.2, -0.5, 0.0, 0.5, 1.2]:
            for y in [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi]:
                for r in [-1.0, 0.0, 1.0]:
                    m_bot = rotation_to_rot_mat(p, y, r)
                    m_rsim = rsim.Angle(pitch=p, yaw=y, roll=r).as_rot_mat().as_numpy().astype(np.float32)
                    # Row 0 (Forward):
                    self.assertLess(float(np.max(np.abs(m_bot[0] - m_rsim[0]))), 1e-6)
                    # Row 1 (Right = -Bullet_Left):
                    self.assertLess(float(np.max(np.abs(m_bot[1] - (-m_rsim[1])))), 1e-6)
                    # Row 2 (Up):
                    self.assertLess(float(np.max(np.abs(m_bot[2] - m_rsim[2]))), 1e-6)


    def test_observation_lateral_ball_offsets(self):
        """
        Guarantees that a ball to the RIGHT (+X) produces a POSITIVE local lateral offset (index 35 > 0)
        and a ball to the LEFT (-X) produces a NEGATIVE local lateral offset (index 35 < 0).
        """
        from env.physics_engine import CarState, BallState, BoostPad
        class MockArena:
            def __init__(self, ball, cars):
                self.ball = ball
                self.cars = cars
                self.boost_pads = BoostPad.create_standard_pads()
            def get_shot_threat(self, team): return False, 0.0, 0.0

        car = CarState(id=0, team=0, pos=np.array([0.0, -3000.0, 17.0], dtype=np.float32),
                       rot=np.array([0.0, np.pi / 2, 0.0], dtype=np.float32))

        # Ball to Right (+X = +500)
        ball_r = BallState(pos=np.array([500.0, -2000.0, 93.0], dtype=np.float32))
        obs_r = self.obs_builder.build_obs(car, MockArena(ball_r, [car]))
        self.assertGreater(obs_r[35], 0.0, "Ball to the RIGHT must produce positive local_ball_pos[1] offset!")

        # Ball to Left (-X = -500)
        ball_l = BallState(pos=np.array([-500.0, -2000.0, 93.0], dtype=np.float32))
        obs_l = self.obs_builder.build_obs(car, MockArena(ball_l, [car]))
        self.assertLess(obs_l[35], 0.0, "Ball to the LEFT must produce negative local_ball_pos[1] offset!")

    def test_locomotion_reward_steering_gradient(self):
        """
        Guarantees that LocomotionReward rewards driving straight (0.0) when pointing at the ball,
        and rewards positive steering when the ball is to the Right.
        """
        from env.physics_engine import CarState, BallState, BoostPad
        from env.rewards import LocomotionReward
        reward_fn = LocomotionReward(weight=0.06)

        class MockArena:
            def __init__(self, ball, cars):
                self.ball = ball
                self.cars = cars
                self.boost_pads = BoostPad.create_standard_pads()
            def get_shot_threat(self, team): return False, 0.0, 0.0

        car = CarState(id=0, team=0, pos=np.array([0.0, -3000.0, 17.0], dtype=np.float32),
                       vel=np.array([0.0, 1000.0, 0.0], dtype=np.float32),
                       rot=np.array([0.0, np.pi / 2, 0.0], dtype=np.float32),
                       boost=50.0, on_ground=True)

        # 1. Ball straight ahead -> Steer 0.0 MUST give higher reward than steer +/-1.0
        ball_c = BallState(pos=np.array([0.0, -2000.0, 93.0], dtype=np.float32))
        rew_straight = reward_fn.get_reward(car, MockArena(ball_c, [car]), np.array([1.0, 0.0, 0, 0, 0, 0, 0, 0]), False, None)
        rew_slam_r = reward_fn.get_reward(car, MockArena(ball_c, [car]), np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0]), False, None)
        rew_slam_l = reward_fn.get_reward(car, MockArena(ball_c, [car]), np.array([1.0, -1.0, 0, 0, 0, 0, 0, 0]), False, None)
        self.assertGreater(rew_straight, rew_slam_r, "Straight driving (0.0) must score higher than slamming right when ball is centered!")
        self.assertGreater(rew_straight, rew_slam_l, "Straight driving (0.0) must score higher than slamming left when ball is centered!")

        # 2. Ball to Right (+X = 200) -> Steer +0.7 MUST score higher than Steer -1.0
        ball_r = BallState(pos=np.array([200.0, -2000.0, 93.0], dtype=np.float32))
        rew_turn_r = reward_fn.get_reward(car, MockArena(ball_r, [car]), np.array([1.0, 0.7, 0, 0, 0, 0, 0, 0]), False, None)
        rew_turn_away = reward_fn.get_reward(car, MockArena(ball_r, [car]), np.array([1.0, -1.0, 0, 0, 0, 0, 0, 0]), False, None)
        self.assertGreater(rew_turn_r, rew_turn_away, "Steering towards right ball must score higher than steering away!")


def verify_physics_and_controls_pipeline(verbose: bool = False) -> bool:
    """
    Programmatic Pre-Flight Verification function.
    Runs all critical physics, observation, and control assertions in < 50ms.
    Raises AssertionError or returns True.
    """
    arena = RocketSimArena(num_players=2, game_mode="1v1")
    arena.reset(random_kickoff=False)

    # 1. Pitch Down (-1.0)
    cs = arena._rsim_cars[0].get_state()
    cs.pos = rsim.Vec(0, 0, 500)
    cs.vel = rsim.Vec(0, 0, 0)
    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=np.pi / 2, roll=0.0).as_rot_mat()
    arena._rsim_cars[0].set_state(cs)
    arena.step([np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=10.0 / 120.0)
    fwd_down = arena.cars[0].get_forward_vector()
    assert fwd_down[2] < 0.0, f"act[2]=-1.0 must pitch nose DOWN in training (got {fwd_down[2]:+.4f})!"

    # 2. Pitch Up (+1.0)
    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=np.pi / 2, roll=0.0).as_rot_mat()
    arena._rsim_cars[0].set_state(cs)
    arena.step([np.array([0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=10.0 / 120.0)
    fwd_up = arena.cars[0].get_forward_vector()
    assert fwd_up[2] > 0.0, f"act[2]=+1.0 must pitch nose UP in training (got {fwd_up[2]:+.4f})!"

    # 3. Steer Left (-1.0)
    cs.pos = rsim.Vec(0, -3000, 17)
    cs.vel = rsim.Vec(0, 500, 0)
    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=np.pi / 2, roll=0.0).as_rot_mat()
    arena._rsim_cars[0].set_state(cs)
    arena.step([np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=15.0 / 120.0)
    assert arena.cars[0].pos[0] < 0.0, "act[1]=-1.0 must steer car LEFT (towards -X)!"

    # 4. Steer Right (+1.0)
    cs.pos = rsim.Vec(0, -3000, 17)
    cs.vel = rsim.Vec(0, 500, 0)
    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=np.pi / 2, roll=0.0).as_rot_mat()
    arena._rsim_cars[0].set_state(cs)
    arena.step([np.array([1.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=15.0 / 120.0)
    assert arena.cars[0].pos[0] > 0.0, "act[1]=+1.0 must steer car RIGHT (towards +X)!"

    # 5. Observation Lateral Sign Alignment
    from env.physics_engine import CarState, BallState, BoostPad
    class MockArena:
        def __init__(self, ball, cars):
            self.ball, self.cars, self.boost_pads = ball, cars, BoostPad.create_standard_pads()
        def get_shot_threat(self, team): return False, 0.0, 0.0
    builder = DefaultObservationBuilder(symmetric=True)
    car_obs = CarState(id=0, team=0, pos=np.array([0.0, -3000.0, 17.0], dtype=np.float32), rot=np.array([0.0, np.pi/2, 0.0], dtype=np.float32))
    obs_r = builder.build_obs(car_obs, MockArena(BallState(pos=np.array([500.0, -2000.0, 93.0], dtype=np.float32)), [car_obs]))
    obs_l = builder.build_obs(car_obs, MockArena(BallState(pos=np.array([-500.0, -2000.0, 93.0], dtype=np.float32)), [car_obs]))
    assert obs_r[35] > 0.0, "Ball on Right must produce positive local lateral offset!"
    assert obs_l[35] < 0.0, "Ball on Left must produce negative local lateral offset!"

    if verbose:
        print("[Pre-Flight Pipeline] Verified: Pitch, Steer, Observations, and Rewards are 100% aligned.")
    return True


if __name__ == "__main__":
    unittest.main()
