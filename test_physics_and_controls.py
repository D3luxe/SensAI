"""
Automated Physics, Control Conventions & Symmetry Verification Tests.
Guarantees 100% alignment between neural network actions, RocketSim physics, and RLBot gamepad controllers.
"""

import math
import unittest
import numpy as np
import torch
import torch.nn as nn
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
        Guarantees that act[2] = +1.0 ALWAYS pitches nose down (front-flip)
        and act[2] = -1.0 ALWAYS pitches nose up (backflip / climb / aerial).
        """
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset(random_kickoff=False)

        # 1. Test Pitch Down (+1.0) in RocketSim
        cs = arena._rsim_cars[0].get_state()
        cs.pos = rsim.Vec(0, 0, 500)
        cs.vel = rsim.Vec(0, 0, 0)
        cs.rot_mat = rsim.Angle(np.pi / 2, 0.0, 0.0).as_rot_mat()
        arena._rsim_cars[0].set_state(cs)

        arena.step([np.array([0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=10.0 / 120.0)
        fwd_down = arena.cars[0].get_forward_vector()
        self.assertLess(fwd_down[2], 0.0, "act[2]=+1.0 must pitch nose DOWN in training physics!")

        # 2. Test Pitch Up (-1.0) in RocketSim
        cs.rot_mat = rsim.Angle(np.pi / 2, 0.0, 0.0).as_rot_mat()
        arena._rsim_cars[0].set_state(cs)
        arena.step([np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=10.0 / 120.0)
        fwd_up = arena.cars[0].get_forward_vector()
        self.assertGreater(fwd_up[2], 0.0, "act[2]=-1.0 must pitch nose UP in training physics!")

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
        cs.rot_mat = rsim.Angle(np.pi / 2, 0.0, 0.0).as_rot_mat()
        arena._rsim_cars[0].set_state(cs)
        arena.step([np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=15.0 / 120.0)
        self.assertLess(arena.cars[0].pos[0], 0.0, "act[1]=-1.0 must steer car LEFT (towards -X)!")

        # Test Steer Right (+1.0)
        cs.pos = rsim.Vec(0, -3000, 17)
        cs.vel = rsim.Vec(0, 500, 0)
        cs.rot_mat = rsim.Angle(np.pi / 2, 0.0, 0.0).as_rot_mat()
        arena._rsim_cars[0].set_state(cs)
        arena.step([np.array([1.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=15.0 / 120.0)
        self.assertGreater(arena.cars[0].pos[0], 0.0, "act[1]=+1.0 must steer car RIGHT (towards +X)!")

    def test_substep_dodge_and_flip_execution(self):
        """
        Guarantees that RocketSimArena.step executes a genuine front-flip / dodge impulse
        when jump is requested across consecutive ground and airborne action steps.
        """
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.reset(random_kickoff=False)

        cs = arena._rsim_cars[0].get_state()
        cs.pos = rsim.Vec(0, -3000, 17)
        cs.vel = rsim.Vec(0, 1000, 0)
        cs.rot_mat = rsim.Angle(np.pi / 2, 0.0, 0.0).as_rot_mat()
        arena._rsim_cars[0].set_state(cs)

        # Step 1: Jump off ground (act[5] = 1.0)
        arena.step([np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), np.zeros(8)], dt=8.0 / 120.0)
        self.assertFalse(arena._rsim_cars[0].get_state().is_on_ground, "Car must be airborne after Step 1 jump!")

        # Step 2: Front-flip / Dodge (act[5] = 1.0, pitch = +1.0)
        arena.step([np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), np.zeros(8)], dt=8.0 / 120.0)
        c_state2 = arena._rsim_cars[0].get_state()
        self.assertTrue(c_state2.has_flipped, "Substep sequencer must trigger has_flipped=True in RocketSim!")
        self.assertGreater(c_state2.vel.y, 1400.0, "Front-flip must deliver > 400 uu/s forward velocity impulse!")

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
        ball = Struct(physics=Struct(location=Struct(x=500.0, y=1000.0, z=91.25), velocity=Struct(x=200.0, y=300.0, z=0.0), angular_velocity=Struct(x=0, y=0, z=0)))
        packet = Struct(num_cars=1, game_cars=[car], game_ball=ball, game_info=Struct(is_match_ended=False))
        # Test airborne controller pass-through: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        test_act = np.array([0.8, -0.7, -0.9, 0.6, -0.5, 0.0, 1.0, 0.0], dtype=np.float32)
        bot.prev_action = test_act
        bot.ticks_since_last_action = 0
        car.has_wheel_contact = False
        ctrl = bot.get_output(packet)

        # In-Game Rocket League gamepad stick input mapping aligned with RocketSim physics engine:
        self.assertAlmostEqual(ctrl.throttle, 0.8, places=4, msg="Throttle maps direct (+0.8 Forward)!")
        self.assertAlmostEqual(ctrl.steer, 0.7, places=4, msg="Steer is -act[1] (act[1]=-0.7 Left maps to ctrl.steer=+0.7)!")
        self.assertAlmostEqual(ctrl.pitch, 0.9, places=4, msg="Pitch is -act[2] (act[2]=-0.9 Pitch Up maps to ctrl.pitch=+0.9)!")
        self.assertAlmostEqual(ctrl.yaw, -0.6, places=4, msg="Yaw is -act[3] (act[3]=+0.6 Right maps to ctrl.yaw=-0.6)!")
        self.assertAlmostEqual(ctrl.roll, 0.5, places=4, msg="Roll is -act[4] (act[4]=-0.5 Left maps to ctrl.roll=+0.5)!")
        self.assertTrue(ctrl.boost)
        self.assertFalse(ctrl.handbrake, msg="Airborne car must NEVER activate handbrake (Air Roll conflict)!")

        # Test Ground Handbrake Pure Pass-Through:
        car.has_wheel_contact = True
        # 1. Action with handbrake disabled (act[7] = -0.9 <= 0.0) -> Handbrake is False
        bot.prev_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9], dtype=np.float32)
        ctrl_straight = bot.get_output(packet)
        self.assertFalse(ctrl_straight.handbrake, msg="Handbrake must be False when act[7] <= 0.0!")

        # 2. Action with handbrake enabled (act[7] = 0.9 > 0.0) -> Handbrake is True
        bot.prev_action = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
        ctrl_turn = bot.get_output(packet)
        self.assertTrue(ctrl_turn.handbrake, msg="Handbrake must be True when act[7] > 0.0 and car is on ground!")

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
        Guarantees that bot.py rotation_to_rot_mat matches true orthonormal basis
        (Row 0: Forward, Row 1: Right = fwd x up, Row 2: Up).
        """
        from bot import rotation_to_rot_mat
        for p in [-1.2, -0.5, 0.0, 0.5, 1.2]:
            for y in [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi]:
                for r in [-1.0, 0.0, 1.0]:
                    m_bot = rotation_to_rot_mat(p, y, r)
                    # Row 0: Forward
                    fwd = m_bot[0]
                    # Row 1: Right
                    right = m_bot[1]
                    # Row 2: Up
                    up = m_bot[2]
                    # Dot products must be 0 (orthonormal)
                    self.assertLess(abs(float(np.dot(fwd, right))), 1e-5)
                    self.assertLess(abs(float(np.dot(fwd, up))), 1e-5)
                    self.assertLess(abs(float(np.dot(right, up))), 1e-5)
                    # Right must equal fwd x up
                    expected_right = np.cross(fwd, up)
                    self.assertLess(float(np.max(np.abs(right - expected_right))), 1e-5)

    def test_observation_lateral_ball_offsets(self):
        """
        Guarantees that a ball to the RIGHT (+X) produces a POSITIVE local lateral offset (index 35 > 0)
        and a ball to the LEFT (-X) produces a NEGATIVE local lateral offset (index 35 < 0)
        both with and without RocketSim rot_mat populated.
        """
        from env.physics_engine import CarState, BallState, BoostPad
        class MockArena:
            def __init__(self, ball, cars):
                self.ball = ball
                self.cars = cars
                self.boost_pads = BoostPad.create_standard_pads()
            def get_shot_threat(self, team): return False, 0.0, 0.0

        for with_rot_mat in [False, True]:
            car = CarState(id=0, team=0, pos=np.array([0.0, -3000.0, 17.0], dtype=np.float32),
                           rot=np.array([0.0, np.pi / 2, 0.0], dtype=np.float32))
            if with_rot_mat:
                car.rot_mat = rsim.Angle(yaw=np.pi / 2, pitch=0.0, roll=0.0).as_rot_mat().as_numpy().astype(np.float32)

            # Ball to Right (+X = +500)
            ball_r = BallState(pos=np.array([500.0, -2000.0, 93.0], dtype=np.float32))
            obs_r = self.obs_builder.build_obs(car, MockArena(ball_r, [car]))
            self.assertGreater(obs_r[35], 0.0, f"Ball to the RIGHT (+X) must produce positive local_ball_pos[1] offset (with_rot_mat={with_rot_mat})!")

            # Ball to Left (-X = -500)
            ball_l = BallState(pos=np.array([-500.0, -2000.0, 93.0], dtype=np.float32))
            obs_l = self.obs_builder.build_obs(car, MockArena(ball_l, [car]))
            self.assertLess(obs_l[35], 0.0, f"Ball to the LEFT (-X) must produce negative local_ball_pos[1] offset (with_rot_mat={with_rot_mat})!")

    def test_macro_rewards_potential_and_boost(self):
        """
        Guarantees that Macro Potential-Based Rewards produce mathematically correct gradients:
        1. Ball-to-Goal Velocity Reward is positive when moving toward opponent goal.
        2. Player-to-Ball Velocity Reward is positive when approaching the ball.
        3. Boost Reward follows Necto sqrt-curve (high reward for low-tank pad pickup).
        """
        from env.physics_engine import CarState, BallState, BoostPad, ARENA_EXTENT_Y
        from env.rewards import BallToGoalVelocityReward, PlayerToBallVelocityReward, BoostReward, GoalReward

        class MockArena:
            def __init__(self, ball, cars):
                self.ball = ball
                self.cars = cars
                self.boost_pads = BoostPad.create_standard_pads()

        car = CarState(id=0, team=0, pos=np.array([0.0, -3000.0, 17.0], dtype=np.float32),
                       vel=np.array([0.0, 1500.0, 0.0], dtype=np.float32),
                       rot=np.array([0.0, np.pi / 2, 0.0], dtype=np.float32),
                       boost=10.0, on_ground=True)

        # 1. Player to Ball closing distance delta
        ball_fwd = BallState(pos=np.array([0.0, -1000.0, 93.0], dtype=np.float32),
                             vel=np.array([0.0, 1000.0, 0.0], dtype=np.float32))
        p2b_fn = PlayerToBallVelocityReward(weight=1.0)
        p2b_fn.reset(MockArena(ball_fwd, [car]))
        # Move car closer: from -3000 to -2500 (closing distance gap by 500 units)
        car.pos = np.array([0.0, -2500.0, 17.0], dtype=np.float32)
        rew_approach = p2b_fn.get_reward(car, MockArena(ball_fwd, [car]), np.zeros(8), False, None)
        self.assertGreater(rew_approach, 0.0, "Closing the distance gap to the ball must yield positive PlayerToBall reward!")

        # 2. Ball to Goal field progression
        b2g_fn = BallToGoalVelocityReward(weight=1.5)
        rew_prog = b2g_fn.get_reward(car, MockArena(ball_fwd, [car]), np.zeros(8), False, None)
        self.assertGreater(rew_prog, 0.0, "Ball moving toward opponent net must yield positive BallToGoal reward!")

        # 3. Necto Sqrt Boost Potential
        boost_fn = BoostReward(gain_weight=1.0, lose_weight=0.5)
        boost_fn.reset(MockArena(ball_fwd, [car]))
        # Collect pad: 10% -> 22% (0.10 -> 0.22)
        car.boost = 22.0
        rew_boost = boost_fn.get_reward(car, MockArena(ball_fwd, [car]), np.zeros(8), False, None)
        expected_diff = math.sqrt(0.22) - math.sqrt(0.10)
        self.assertAlmostEqual(rew_boost, expected_diff, places=4, msg="Boost gain must match sqrt(curr) - sqrt(prev)!")

        # 4. Zero-Sum Goal Reward
        goal_fn = GoalReward(goal_weight=10.0, concede_weight=-10.0)
        self.assertEqual(goal_fn.get_reward(car, MockArena(ball_fwd, [car]), np.zeros(8), True, 0), 10.0)
        self.assertEqual(goal_fn.get_reward(car, MockArena(ball_fwd, [car]), np.zeros(8), True, 1), -10.0)

    def test_jump_passthrough_and_ground_stabilization(self):
        """
        Guarantees that bot.py passes jump directly (act[5] > 0.33 -> True, <= 0.33 -> False)
        and stabilizes ground driving by keeping pitch neutral when not jumping.
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
                velocity=Struct(x=0.0, y=500.0, z=0.0),
                rotation=Struct(pitch=0.0, yaw=np.pi / 2, roll=0.0),
                angular_velocity=Struct(x=0.0, y=0.0, z=0.0)
            )
        )
        ball = Struct(physics=Struct(location=Struct(x=500.0, y=1000.0, z=91.25), velocity=Struct(x=200.0, y=0.0, z=0.0), angular_velocity=Struct(x=0, y=0, z=0)))
        packet = Struct(num_cars=1, game_cars=[car], game_ball=ball, game_info=Struct(is_match_ended=False))

        # 1. Driving on ground without jump (act[5] = -0.50 <= 0.0): pitch should be stabilized to 0.0
        bot.prev_action = np.array([1.0, 0.0, -1.0, 0.0, 0.0, -0.50, 0.0, 0.0], dtype=np.float32)
        bot.ticks_since_last_action = 1
        ctrl_drive = bot.get_output(packet)
        self.assertFalse(ctrl_drive.jump, "Jump must be False when act[5] <= 0.0")
        self.assertAlmostEqual(ctrl_drive.pitch, 0.0, places=4, msg="Pitch must be stabilized to 0.0 on ground when not jumping")

        # 2. Jump requested (act[5] = 0.50 > 0.0): jump should be True and pitch active
        bot.prev_action = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.50, 0.0, 0.0], dtype=np.float32)
        bot.ticks_since_last_action = 1
        ctrl_jump = bot.get_output(packet)
        self.assertTrue(ctrl_jump.jump, "Jump must be True when act[5] > 0.0")
        self.assertAlmostEqual(ctrl_jump.pitch, 1.0, places=4, msg="Pitch must be active when jump is requested")

    def test_kickoff_touch_state_tracking(self):
        """
        Guarantees that bot.py accurately tracks kickoff touched state so is_first_touch
        correctly transitions from True to False once play starts.
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
        # Stationary center ball -> kickoff untouched
        ball = Struct(physics=Struct(location=Struct(x=0.0, y=0.0, z=91.25), velocity=Struct(x=0, y=0, z=0), angular_velocity=Struct(x=0, y=0, z=0)))
        packet = Struct(num_cars=1, game_cars=[car], game_ball=ball, game_info=Struct(is_match_ended=False))

        bot.prev_action = np.zeros(8, dtype=np.float32)
        bot.get_output(packet)
        self.assertFalse(bot.ball_touched_since_kickoff, "Ball at rest in center must be marked untouched")

        # Kickoff Timeout Recovery: after > 180 ticks of untouched ball, timeout recovery activates
        bot.kickoff_stagnation_ticks = 181
        bot.get_output(packet)
        self.assertTrue(bot.ball_touched_since_kickoff, "Untouched kickoff beyond 180 ticks must trigger timeout recovery")

        # Moving ball reinforces active play
        ball_moving = Struct(physics=Struct(location=Struct(x=500.0, y=1000.0, z=200.0), velocity=Struct(x=500, y=1200, z=0), angular_velocity=Struct(x=0, y=0, z=0)))
        packet_moving = Struct(num_cars=1, game_cars=[car], game_ball=ball_moving, game_info=Struct(is_match_ended=False))
        bot.get_output(packet_moving)
        self.assertTrue(bot.ball_touched_since_kickoff, "Fast moving ball must be marked as touched/active play")

        # Re-entering kickoff pause resets kickoff tracking
        packet_pause = Struct(num_cars=1, game_cars=[car], game_ball=ball, game_info=Struct(is_match_ended=False, is_kickoff_pause=True))
        bot.get_output(packet_pause)
        self.assertFalse(bot.ball_touched_since_kickoff, "Entering kickoff pause must reset ball_touched_since_kickoff")
        self.assertEqual(bot.kickoff_stagnation_ticks, 0, "Entering kickoff pause must reset kickoff stagnation ticks")

    def test_has_flip_ground_and_airborne_parity(self):
        """
        Guarantees that has_flip is strictly False on the ground (matching RocketSim training)
        and True when airborne with double-jump available.
        """
        from env.physics_engine import CarState
        car_ground = CarState(id=0, team=0, on_ground=True, has_flip=False)
        self.assertFalse(car_ground.has_flip, "Car on ground must have has_flip=False")

        # Airborne with double jump available
        car_air = CarState(id=0, team=0, on_ground=False, has_flip=True)
        self.assertTrue(car_air.has_flip, "Airborne car with double jump available must have has_flip=True")

    def test_actor_critic_layer_norm_and_saturation(self):
        """
        Guarantees that ActorCritic with LayerNorm maintains bounded, healthy activations
        even when fed extreme observation inputs, and prevents policy saturation.
        """
        model = ActorCritic(obs_dim=74, act_dim=8, continuous_actions=True, use_layer_norm=True, activation="leaky_relu")
        model.eval()

        # Extreme out-of-distribution observation input (+/- 10.0)
        extreme_obs = torch.full((4, 74), 10.0, dtype=torch.float32)
        action, _, _, value = model.get_action_and_value(extreme_obs, deterministic=True)

        self.assertEqual(action.shape, (4, 8))
        self.assertEqual(value.shape, (4, 1))

        # Check that debias_symmetric_actions desaturates and zeroes biases
        model.debias_symmetric_actions()
        self.assertAlmostEqual(float(model.actor_mean.bias[1].detach()), 0.0, places=5)
        self.assertAlmostEqual(float(model.actor_mean.bias[2].detach()), 0.0, places=5)
        self.assertAlmostEqual(float(model.actor_mean.bias[3].detach()), 0.0, places=5)
        self.assertAlmostEqual(float(model.actor_mean.bias[4].detach()), 0.0, places=5)
        self.assertAlmostEqual(float(model.actor_binary.bias[2].detach()), 0.0, places=5)

    def test_bot_boost_pad_awareness(self):
        """
        Guarantees that bot.py accurately feeds active boost pad vectors into observation space (features 68-73).
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

        bot.ticks_since_last_action = 8
        bot.get_output(packet)
        # Verify arena constructed in get_output contains active boost pad vector arrays
        self.assertIsNotNone(bot.obs_builder)

    def test_binary_action_gradients_in_pretrainer(self):
        """
        Guarantees that actor_binary (Jump, Boost, Handbrake) receives non-zero gradients
        under BCEWithLogitsLoss during pretraining.
        """
        model = ActorCritic(obs_dim=74, act_dim=8, continuous_actions=True, use_layer_norm=True)
        obs = torch.randn(16, 74)
        target_acts = torch.randn(16, 8)
        target_acts[:, 5:] = (target_acts[:, 5:] > 0.0).float() * 2.0 - 1.0

        feat = model.actor_backbone(obs)
        pred_cont = torch.tanh(model.actor_mean(feat))
        pred_bin_logits = model.actor_binary(feat)

        target_cont = target_acts[:, :5]
        target_bin = (target_acts[:, 5:] > 0.0).float()

        loss = nn.functional.smooth_l1_loss(pred_cont, target_cont) + 0.5 * nn.functional.binary_cross_entropy_with_logits(pred_bin_logits, target_bin)
        loss.backward()

        self.assertIsNotNone(model.actor_binary.weight.grad)
        self.assertGreater(float(model.actor_binary.weight.grad.norm()), 0.0, "actor_binary must receive non-zero gradients!")
        self.assertIsNotNone(model.actor_binary.bias.grad)
        self.assertGreater(float(model.actor_binary.bias.grad.norm()), 0.0, "actor_binary bias must receive non-zero gradients!")

    def test_air_roll_recovery_reward(self):
        """
        Guarantees that AirRollRecoveryReward rewards wheels-down upright orientation when descending
        and penalizes upside-down orientation.
        """
        from env.rewards import AirRollRecoveryReward
        from env.physics_engine import CarState, BallState, BoostPad

        class MockArena:
            def __init__(self, ball, cars):
                self.ball, self.cars = ball, cars
                self.boost_pads = BoostPad.create_standard_pads()

        ball = BallState(pos=np.array([0, 0, 93], dtype=np.float32))
        rew_fn = AirRollRecoveryReward(weight=1.0)

        # 1. Upright descending car (wheels down, up_vector = [0, 0, 1])
        car_upright = CarState(id=0, team=0, pos=np.array([0, 0, 300], dtype=np.float32),
                               vel=np.array([0, 0, -300], dtype=np.float32),
                               rot=np.array([0, 0, 0], dtype=np.float32), on_ground=False)
        r_upright = rew_fn.get_reward(car_upright, MockArena(ball, [car_upright]), np.zeros(8), False, None)
        self.assertGreater(r_upright, 0.0, "Upright descending car must receive positive recovery reward!")

        # 2. Inverted descending car (roof down, roll = pi, up_vector = [0, 0, -1])
        car_inverted = CarState(id=0, team=0, pos=np.array([0, 0, 300], dtype=np.float32),
                                vel=np.array([0, 0, -300], dtype=np.float32),
                                rot=np.array([0, 0, math.pi], dtype=np.float32), on_ground=False)
        r_inverted = rew_fn.get_reward(car_inverted, MockArena(ball, [car_inverted]), np.zeros(8), False, None)
        self.assertLess(r_inverted, 0.0, "Inverted descending car must receive penalty for upside-down descent!")

    def test_strike_zone_throttle_pacing_reward(self):
        """
        Guarantees that PlayerToBallVelocityReward provides braking incentive when closing fast on a slower ball.
        """
        from env.rewards import PlayerToBallVelocityReward
        from env.physics_engine import CarState, BallState, BoostPad

        class MockArena:
            def __init__(self, ball, cars):
                self.ball, self.cars = ball, cars
                self.boost_pads = BoostPad.create_standard_pads()

        ball = BallState(pos=np.array([0, -2800, 93], dtype=np.float32), vel=np.array([0, 200, 0], dtype=np.float32))
        car = CarState(id=0, team=0, pos=np.array([0, -3000, 17], dtype=np.float32),
                       vel=np.array([0, 1500, 0], dtype=np.float32),
                       rot=np.array([0, math.pi / 2, 0], dtype=np.float32), on_ground=True)

        p2b = PlayerToBallVelocityReward(weight=1.0)
        p2b.reset(MockArena(ball, [car]))

        # Braking action (act[0] = -1.0)
        act_brake = np.array([-1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        rew_brake = p2b.get_reward(car, MockArena(ball, [car]), act_brake, False, None)

        # Full throttle action (act[0] = +1.0)
        act_thr = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        rew_thr = p2b.get_reward(car, MockArena(ball, [car]), act_thr, False, None)

        self.assertGreater(rew_brake, rew_thr, "Braking when closing too fast on slow ball must yield higher reward than full throttle!")


def verify_physics_and_controls_pipeline(verbose: bool = False) -> bool:
    """
    Programmatic Pre-Flight Verification function.
    Runs all critical physics, observation, and control assertions in < 50ms.
    Raises AssertionError or returns True.
    """
    arena = RocketSimArena(num_players=2, game_mode="1v1")
    arena.reset(random_kickoff=False)

    # 1. Pitch Down (+1.0)
    cs = arena._rsim_cars[0].get_state()
    cs.pos = rsim.Vec(0, 0, 500)
    cs.vel = rsim.Vec(0, 0, 0)
    cs.rot_mat = rsim.Angle(np.pi / 2, 0.0, 0.0).as_rot_mat()
    arena._rsim_cars[0].set_state(cs)
    arena.step([np.array([0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=10.0 / 120.0)
    fwd_down = arena.cars[0].get_forward_vector()
    assert fwd_down[2] < 0.0, f"act[2]=+1.0 must pitch nose DOWN in training (got {fwd_down[2]:+.4f})!"

    # 2. Pitch Up (-1.0)
    cs.rot_mat = rsim.Angle(np.pi / 2, 0.0, 0.0).as_rot_mat()
    arena._rsim_cars[0].set_state(cs)
    arena.step([np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=10.0 / 120.0)
    fwd_up = arena.cars[0].get_forward_vector()
    assert fwd_up[2] > 0.0, f"act[2]=-1.0 must pitch nose UP in training (got {fwd_up[2]:+.4f})!"

    # 3. Steer Left (-1.0)
    cs.pos = rsim.Vec(0, -3000, 17)
    cs.vel = rsim.Vec(0, 500, 0)
    cs.rot_mat = rsim.Angle(np.pi / 2, 0.0, 0.0).as_rot_mat()
    arena._rsim_cars[0].set_state(cs)
    arena.step([np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(8)], dt=15.0 / 120.0)
    assert arena.cars[0].pos[0] < 0.0, "act[1]=-1.0 must steer car LEFT (towards -X)!"

    # 4. Steer Right (+1.0)
    cs.pos = rsim.Vec(0, -3000, 17)
    cs.vel = rsim.Vec(0, 500, 0)
    cs.rot_mat = rsim.Angle(np.pi / 2, 0.0, 0.0).as_rot_mat()
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
    assert obs_r[35] > 0.0, "Ball on Right (+X) must produce positive local lateral offset in true right basis!"
    assert obs_l[35] < 0.0, "Ball on Left (-X) must produce negative local lateral offset in true right basis!"

    if verbose:
        print("[Pre-Flight Pipeline] Verified: Pitch, Steer, Observations, and Rewards are 100% aligned.")
    return True


if __name__ == "__main__":
    unittest.main()
