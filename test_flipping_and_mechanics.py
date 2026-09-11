"""
Unit tests for flipping mechanics, half-flips, wave dashes, and control calibrations.
Guarantees:
  1. Front flip downfield maintains positive PlayerToBallVelocityReward without artificial damping.
  2. Uncancelled backflip landing backwards receives 0.0 touchdown alignment reward (no backwards exploit).
  3. Half-flip (backflip -> cancel + air roll -> touchdown forward) receives cancel reward and +1.50 turnaround bonus.
  4. Open-field traversal flips require speed > 350 uu/s, eliminating 0-speed flips in place.
  5. Low-altitude flip slam into turf is recognized as a wavedash and awarded the dedicated impulse bonus.
  6. Deterministic jump threshold is calibrated to p > 0.30 (logit -0.8473).
"""

import math
import unittest
import numpy as np
import torch
import RocketSim as rsim

from env.physics_engine import RocketSimArena, CarState, BallState
from env.rewards import JumpBridgeReward, AirRollRecoveryReward, PlayerToBallVelocityReward
from agent.models import ActorCritic
from bot import AirState, MatchPhase


def set_rsim_car_state(arena: RocketSimArena, car_idx: int, pos, vel, fwd, right, up, boost=50.0, on_ground=True):
    r_car = arena._rsim_cars[car_idx]
    cs = rsim.CarState()
    cs.pos = rsim.Vec(pos[0], pos[1], pos[2])
    cs.vel = rsim.Vec(vel[0], vel[1], vel[2])
    cs.rot_mat = rsim.RotMat(
        rsim.Vec(fwd[0], fwd[1], fwd[2]),
        rsim.Vec(right[0], right[1], right[2]),
        rsim.Vec(up[0], up[1], up[2]),
    )
    cs.boost = boost
    cs.is_on_ground = on_ground
    r_car.set_state(cs)
    arena._sync_from_rsim()


def set_rsim_ball_state(arena: RocketSimArena, pos, vel):
    bs = rsim.BallState()
    bs.pos = rsim.Vec(pos[0], pos[1], pos[2])
    bs.vel = rsim.Vec(vel[0], vel[1], vel[2])
    arena._rsim_arena.ball.set_state(bs)
    arena._sync_from_rsim()


class TestFlippingAndMechanics(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena()
        self.arena.reset()

    def test_front_flip_maintains_positive_pursuit_reward(self):
        """Guarantees that front flips toward the ball are not penalized by pitch inversion."""
        set_rsim_car_state(self.arena, 0, [0.0, 0.0, 17.0], [0.0, 800.0, 0.0],
                           fwd=[0, 1, 0], right=[-1, 0, 0], up=[0, 0, 1], boost=30.0, on_ground=True)
        set_rsim_ball_state(self.arena, [0.0, 3000.0, 93.0], [0.0, 0.0, 0.0])

        p2b = PlayerToBallVelocityReward(weight=1.0)
        p2b.reset(self.arena)

        # Step 1: Liftoff
        act_hop = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self.arena.step([act_hop, np.zeros(8)])
        r_hop = p2b.get_reward(self.arena.cars[0], self.arena, act_hop, False, None)
        self.assertGreater(r_hop, 0.0, "Liftoff step toward ball must maintain positive reward")

        # Step 2: Front flip dodge (act[2] = +1.0)
        act_flip = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self.arena.step([act_flip, np.zeros(8)])
        r_flip = p2b.get_reward(self.arena.cars[0], self.arena, act_flip, False, None)
        self.assertGreater(r_flip, 0.0, "Front flip step must maintain positive reward")

        # Step 3..6: Car pitches inverted midair but rockets toward ball at >1200 uu/s
        for _ in range(4):
            act_glide = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.arena.step([act_glide, np.zeros(8)])
            r_glide = p2b.get_reward(self.arena.cars[0], self.arena, act_glide, False, None)
            self.assertGreater(r_glide, 0.0, "Inverted flight of front flip must maintain positive reward via travel velocity")

    def test_uncancelled_backflip_no_backwards_landing_reward(self):
        """Guarantees that landing backwards from an uncancelled backflip receives 0 touchdown reward."""
        recovery = AirRollRecoveryReward(weight=1.0)
        recovery.reset(self.arena)

        # Simulate car at touchdown: car_z = 50, vel_z = -100, up_z = 0.9 (upright), nose facing +Y, travel vel is -Y
        car = self.arena.cars[0]
        car.on_ground = False
        car.pos = np.array([0.0, 0.0, 50.0], dtype=np.float32)
        car.vel = np.array([0.0, -800.0, -100.0], dtype=np.float32)
        # Facing +Y (fwd = [0, 1, 0]) while traveling -Y -> heading is -1.0
        car.rot_mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
        recovery._airborne_ticks[car.id] = 15
        recovery._was_disoriented[car.id] = True
        recovery._prev_surface_align[car.id] = 1.0
        recovery._prev_heading[car.id] = -1.0

        # Landing backwards with reverse throttle
        act_reverse = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rew = recovery.get_reward(car, self.arena, act_reverse, False, None)

        # Only upright touchdown bonus (1.0 * 0.5 = 0.50) allowed; backwards landing bonus must be 0.0
        self.assertAlmostEqual(rew, 0.50, places=4, msg="Landing backwards must NOT receive any backwards heading bonus!")

    def test_half_flip_sequence_awards_cancel_and_turnaround_bonus(self):
        """Guarantees that active flip cancel and roll earns cancel reward and +1.50 turnaround bonus."""
        recovery = AirRollRecoveryReward(weight=1.0)
        self.arena.ball.pos = np.array([0.0, 3000.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        recovery.reset(self.arena)

        car = self.arena.cars[0]
        car.on_ground = False
        car.pos = np.array([0.0, 0.0, 150.0], dtype=np.float32)
        car.vel = np.array([0.0, -750.0, -50.0], dtype=np.float32)
        # Car is inverted (wheels up, up_z = -0.8)
        car.rot_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float32)
        recovery._airborne_ticks[car.id] = 8
        recovery._was_disoriented[car.id] = True
        recovery._prev_surface_align[car.id] = -0.8
        recovery._prev_heading[car.id] = -0.5
        # A cancel happens WHILE the dodge is running: that is what distinguishes it from an
        # uncancelled backflip, whose pitch stays at full rate for the whole animation. The
        # flip-cancel detector now requires it, so the fixture has to reflect it. Before this
        # flag existed the detector credited a cancel on low pitch rate alone, which any
        # backward-moving flight satisfied -- including a car that never flipped at all.
        car.is_dodging = True
        recovery._takeoff_heading[car.id] = -1.0

        # Active Flip-Cancel (act[2] = +1.0) + Air-Roll (act[4] = +1.0)
        act_cancel = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rew_cancel = recovery.get_reward(car, self.arena, act_cancel, False, None)
        self.assertGreater(rew_cancel, 0.10, "Active flip-cancel must earn dedicated cancel reward")
        self.assertTrue(recovery._halfflip_cancel_executed[car.id], "Half-flip cancel execution flag must be set")

        # Now car completes roll, turns around, and lands facing forward (heading > 0.6)
        car.pos[2] = 50.0
        car.vel[2] = -80.0
        # rot_mat: now facing -Y (fwd = [0, -1, 0], vel is -Y -> heading is +1.0, wheels down)
        car.rot_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        act_land = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rew_touchdown = recovery.get_reward(car, self.arena, act_land, False, None)

        # Must include +1.50 turnaround bonus!
        self.assertGreaterEqual(rew_touchdown, 1.80, "Half-flip forward touchdown must receive the +1.50 completion bonus!")

    def test_self_inflicted_flip_earns_no_recovery_income(self):
        """A backflip is not a disorientation to be paid for fixing.

        The detector reads pure geometry -- on its back, or travelling backwards -- and a
        backflip satisfies both, so the car could invert itself for free and then collect for
        righting what it had just broken. With jump_bridge zeroed that loop was the last
        standing reason to flip: at iteration 1129 the policy was airborne 71% of the time,
        never reversed, and drew 2.55 per episode from this term with 91% earned in the air.
        Measured on live trajectories the same term now pays 0.35.
        """
        def fly(self_dodge):
            rew = AirRollRecoveryReward(weight=1.0)
            self.arena.ball.pos = np.array([0.0, 3000.0, 93.0], dtype=np.float32)
            rew.reset(self.arena)
            car = self.arena.cars[0]
            total = 0.0
            act = np.zeros(8, dtype=np.float32)
            for i, up_z in enumerate([-0.9, -0.4, 0.2, 0.7, 0.95]):
                car.on_ground = False
                car.is_dodging = bool(self_dodge and i < 3)
                car.just_dodged = bool(self_dodge and i == 0)
                car.pos = np.array([0.0, -60.0 * i, 140.0], dtype=np.float32)
                car.vel = np.array([0.0, -700.0, -40.0], dtype=np.float32)
                lateral = float(np.sqrt(max(0.0, 1.0 - up_z ** 2)))
                car.rot_mat = np.array([[0, 1, 0], [1, 0, 0], [0, lateral, up_z]],
                                       dtype=np.float32)
                rew._airborne_ticks[car.id] = i + 4
                total += rew.get_reward(car, self.arena, act, False, None)
            return total

        bumped = fly(self_dodge=False)
        flipped = fly(self_dodge=True)
        self.assertGreater(bumped, 0.0,
                           "recovering from a genuine knock must still be reinforced")
        self.assertLess(flipped, bumped,
                        "righting a flip the car chose must not pay what a real recovery pays")

    def test_uncancelled_backflip_is_not_credited_as_a_half_flip(self):
        """The cancel used to be credited on low pitch rate alone, so a car with no rotation
        at all read as a perfectly executed half-flip. A cancel is the conjunction of a dodge
        in progress AND its pitch already arrested; an uncancelled backflip fails the second
        for the length of the animation and the first once it ends."""
        rew = AirRollRecoveryReward(weight=1.0)
        self.arena.ball.pos = np.array([0.0, 3000.0, 93.0], dtype=np.float32)
        rew.reset(self.arena)
        car = self.arena.cars[0]
        car.on_ground = False
        car.pos = np.array([0.0, 0.0, 150.0], dtype=np.float32)
        car.vel = np.array([0.0, -750.0, -50.0], dtype=np.float32)
        car.rot_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float32)
        rew._airborne_ticks[car.id] = 8
        rew._takeoff_heading[car.id] = -1.0
        rew._prev_surface_align[car.id] = -0.8

        # Still tumbling at full dodge pitch rate: not a cancel.
        car.is_dodging = True
        car.ang_vel = np.array([5.5, 0.0, 0.0], dtype=np.float32)
        rew.get_reward(car, self.arena, np.zeros(8, dtype=np.float32), False, None)
        self.assertFalse(rew._halfflip_cancel_executed[car.id],
                         "a flip still rotating at full rate has not been cancelled")

        # Animation over, rotation decayed on its own: also not a cancel.
        car.is_dodging = False
        car.ang_vel = np.array([0.2, 0.0, 0.0], dtype=np.float32)
        rew.get_reward(car, self.arena, np.zeros(8, dtype=np.float32), False, None)
        self.assertFalse(rew._halfflip_cancel_executed[car.id],
                         "a flip that simply ran its course was never cancelled")

    def test_open_field_standstill_flip_unrewarded(self):
        """Guarantees that open-field traversal flips require speed > 350 uu/s to prevent flipping in place."""
        car = self.arena.cars[0]
        car.on_ground = False
        car.has_flip = False
        # Ball is far away downfield (open field: dist > 650)
        self.arena.ball.pos = np.array([0.0, 2500.0, 93.0], dtype=np.float32)
        car.pos = np.array([0.0, 0.0, 40.0], dtype=np.float32)
        # Car is nearly stationary: speed = 50 uu/s (< 350 uu/s)
        car.vel = np.array([0.0, 50.0, 0.0], dtype=np.float32)
        car.rot_mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

        bridge = JumpBridgeReward(weight=1.0)
        bridge.reset(self.arena)
        bridge._prev_has_flip[car.id] = True # Dodge event

        # Attempt front flip from dead stop
        act_flip = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        rew = bridge.get_reward(car, self.arena, act_flip, False, None)
        self.assertEqual(rew, 0.0, "Open-field dodge flip from near-zero speed must award 0.0 to stop flipping in place!")

    def test_explicit_wavedash_reward(self):
        """Guarantees that low-altitude flip slams into turf are recognized as wavedashes."""
        bridge = JumpBridgeReward(weight=1.0)
        bridge.reset(self.arena)

        car = self.arena.cars[0]
        car.pos = np.array([0.0, 0.0, 40.0], dtype=np.float32)
        car.on_ground = True # Just contacted turf
        car.just_dodged = True
        bridge._prev_pos_z[car.id] = 40.0 # Low height when dodging (< 55 uu)
        bridge._prev_vel[car.id] = np.array([0.0, 500.0, 0.0], dtype=np.float32)
        # Immediate impulse acceleration along tactical vector (ball at +Y): 500 -> 800 uu/s (+300 delta)
        car.vel = np.array([0.0, 800.0, 0.0], dtype=np.float32)
        self.arena.ball.pos = np.array([0.0, 2500.0, 93.0], dtype=np.float32)

        act = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        rew = bridge.get_reward(car, self.arena, act, False, None)
        self.assertGreaterEqual(rew, 1.0, "Low altitude flip slam into turf must award explicit wavedash bonus!")

    def test_model_jump_threshold_calibrated(self):
        """Guarantees that ActorCritic deterministic jump threshold is set to p > 0.15 (-1.7346)."""
        model = ActorCritic(obs_dim=74, act_dim=8, continuous_actions=True)
        self.assertAlmostEqual(model.bin_thresh_logits[0].item(), -1.7346, places=3,
                               msg="Jump threshold logit must be calibrated to -1.7346 (p > 0.15)!")

    def test_mid_flip_inverted_flight_velocity_reward_undampened(self):
        """Guarantees that inverted flight during a front flip/speedflip receives 100% velocity reward without dampening."""
        from env.rewards import PlayerToBallVelocityReward
        p2b = PlayerToBallVelocityReward(weight=1.0)
        p2b.reset(self.arena)

        car = self.arena.cars[0]
        car.team = 0
        # Car is mid-flip: airborne, flip already spent (has_flip=False), rocketing forward toward ball at +Y
        car.on_ground = False
        car.has_flip = False
        car.just_dodged = False  # Critical: dodge tick has passed!
        car.pos = np.array([0.0, 0.0, 80.0], dtype=np.float32)
        car.vel = np.array([0.0, 1400.0, 50.0], dtype=np.float32)
        # Inverted orientation mid-flip: nose points backwards -Y (fwd = [0, -1, 0]), wheels up
        car.rot_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float32)
        self.arena.ball.pos = np.array([0.0, 2000.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.zeros(3, dtype=np.float32)

        p2b._prev_dist[car.id] = 2100.0 # Car is closing distance (2100 -> 2000 = +100 uu delta)
        act = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rew = p2b.get_reward(car, self.arena, act, False, None)

        # Delta dist = 100 / 2000 = 0.05. If dampened by 80%, rew would be ~0.01.
        self.assertGreaterEqual(rew, 0.04, "Inverted mid-flip flight must NOT dampen forward distance closure rewards!")

    def test_bot_low_speed_steer_jump_suppression(self):
        """Guarantees that bot.py suppresses jump when turning hard at low speeds."""
        from bot import SenseiRLBot

        bot = SenseiRLBot("TestBot", 0, 0)
        bot.prev_action = np.array([1.0, 0.9, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        class Struct:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        car = Struct(
            team=0, boost=33.3, air_state=AirState.OnGround, has_jumped=False, has_double_jumped=False, has_dodged=False,
            physics=Struct(
                location=Struct(x=0.0, y=0.0, z=17.0),
                velocity=Struct(x=0.0, y=100.0, z=0.0),
                angular_velocity=Struct(x=0.0, y=0.0, z=0.0),
                rotation=Struct(pitch=0.0, yaw=0.0, roll=0.0)
            )
        )
        packet = Struct(
            players=[car],
            balls=[Struct(
                physics=Struct(
                    location=Struct(x=0.0, y=1000.0, z=93.0),
                    velocity=Struct(x=0.0, y=0.0, z=0.0),
                    angular_velocity=Struct(x=0.0, y=0.0, z=0.0)
                )
            )],
            boost_pads=[],
            match_info=Struct(match_phase=MatchPhase.Active)
        )
        ctrl = bot.get_output(packet)
        self.assertFalse(ctrl.jump, "Jump must be suppressed when turning sharply at low speed to prevent tumbling!")

    def test_neutral_double_jump_0_boost_rewarded(self):
        """Guarantees that a neutral double jump with 0 boost under an elevated ball receives vertical impulse reward."""
        bridge = JumpBridgeReward(weight=1.0)
        bridge.reset(self.arena)

        car = self.arena.cars[0]
        car.team = 0
        car.boost = 0.0
        car.on_ground = False
        car.pos = np.array([0.0, 0.0, 80.0], dtype=np.float32)
        car.vel = np.array([0.0, 600.0, 350.0], dtype=np.float32)
        car.rot_mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
        car.has_double_jumped = True
        self.arena.ball.pos = np.array([0.0, 600.0, 450.0], dtype=np.float32)

        bridge._prev_on_ground[car.id] = False
        bridge._prev_has_double_jumped[car.id] = False

        # Neutral stick inputs
        act = np.zeros(8, dtype=np.float32)
        rew = bridge.get_reward(car, self.arena, act, False, None)
        self.assertGreater(rew, 0.40, f"Neutral double jump under elevated ball with 0 boost must award vertical climb bonus, got {rew}")

    def test_fast_aerial_pitch_up_not_penalized_as_backflip(self):
        """Guarantees that pulling pitch back to aim nose up under a high aerial ball is NOT penalized as a forward backflip."""
        bridge = JumpBridgeReward(weight=1.0)
        bridge.reset(self.arena)

        car = self.arena.cars[0]
        car.team = 0
        car.on_ground = False
        car.has_flip = False
        car.pos = np.array([0.0, -500.0, 100.0], dtype=np.float32)
        car.vel = np.array([0.0, 800.0, 200.0], dtype=np.float32)
        # Nose tilted up towards high ball
        car.rot_mat = np.array([[0, 0.8, 0.6], [-1, 0, 0], [0, -0.6, 0.8]], dtype=np.float32)
        self.arena.ball.pos = np.array([0.0, 800.0, 650.0], dtype=np.float32)

        bridge._prev_on_ground[car.id] = False
        bridge._prev_has_flip[car.id] = True  # Second jump executed

        # Pitch back input to tilt up
        act = np.zeros(8, dtype=np.float32)
        act[2] = -0.50
        rew = bridge.get_reward(car, self.arena, act, False, None)
        self.assertGreaterEqual(rew, 0.0, f"Pitching nose up under elevated ball must NOT receive -0.80 backflip penalty, got {rew}")

    def test_dodge_strike_outcome_bounty(self):
        """Guarantees that striking the ball during or after a flip awards the Dodge Strike outcome bounty."""
        bridge = JumpBridgeReward(weight=1.0)
        bridge.reset(self.arena)

        car = self.arena.cars[0]
        car.team = 0
        car.on_ground = False
        car.has_flip = False
        car.just_dodged = True
        car.ball_touches = 1
        car.pos = np.array([0.0, 1000.0, 60.0], dtype=np.float32)
        car.vel = np.array([0.0, 1200.0, 50.0], dtype=np.float32)
        car.rot_mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

        self.arena.ball.pos = np.array([0.0, 1100.0, 80.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 1400.0, 200.0], dtype=np.float32)

        bridge._prev_touches[car.id] = 0
        bridge._prev_ball_vel[car.id] = np.array([0.0, 200.0, 0.0], dtype=np.float32)
        bridge._prev_has_flip[car.id] = True

        act = np.zeros(8, dtype=np.float32)
        rew = bridge.get_reward(car, self.arena, act, False, None)
        self.assertGreater(rew, 0.80, f"Impacting ball with a dodge towards opponent net must award dodge strike bounty, got {rew}")

    def test_0_boost_reachable_ball_takeoff_rewarded(self):
        """Guarantees that taking off towards a reachable elevated ball (Z <= 500) with 0 boost is rewarded without penalty."""
        bridge = JumpBridgeReward(weight=1.0)
        bridge.reset(self.arena)

        car = self.arena.cars[0]
        car.team = 0
        car.boost = 0.0
        car.on_ground = False
        car.pos = np.array([0.0, 0.0, 25.0], dtype=np.float32)
        car.vel = np.array([0.0, 600.0, 250.0], dtype=np.float32)
        car.rot_mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
        self.arena.ball.pos = np.array([0.0, 400.0, 320.0], dtype=np.float32)

        bridge._prev_on_ground[car.id] = True  # Just lifted off

        act = np.zeros(8, dtype=np.float32)
        rew = bridge.get_reward(car, self.arena, act, False, None)
        self.assertGreater(rew, 0.50, f"Liftoff towards reachable ball with 0 boost must award positive takeoff bonus, got {rew}")


if __name__ == "__main__":
    unittest.main()
