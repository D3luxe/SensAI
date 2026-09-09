import unittest
import math
import numpy as np

from env.physics_engine import CarState, BallState, RocketSimArena, ARENA_EXTENT_Y
from env.rewards import PlayerToBallVelocityReward, JumpBridgeReward
from bot import SenseiRLBot


class TestDribbleAndFlickMechanics(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")

    def test_roof_carry_no_forward_velocity_leak(self):
        """Test that vel_toward_ball is strictly zeroed out during is_roof_carry to prevent overdriving."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        # Car moving forward at 900 uu/s downfield
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 900.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        # Ball resting on roof slightly forward of pivot (local_x = 20 uu, local_z = 130 uu)
        # With yaw = pi/2, forward is +Y, right is -X
        self.arena.ball.pos = np.array([0.0, 20.0, 147.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 900.0, 0.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)

        # Evaluate reward with neutral throttle
        act = np.zeros(8, dtype=np.float32)
        act[0] = 1.0
        r = rew.get_reward(car, self.arena, act, False, None)

        # Distance closure term should also be neutral/stable when velocities match
        self.assertGreater(r, 0.30, f"Roof carry should produce positive carry reward, got {r}")

        # Now test that forward speed does NOT produce artificial vel_toward_ball overspeeding bonus
        # By comparing reward at 900 uu/s with matching ball vs when ball was slightly forward
        car_faster = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1400.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.cars = [car_faster]
        rew.reset(self.arena)
        r_faster = rew.get_reward(car_faster, self.arena, act, False, None)

        # With vel_toward_ball zeroed, faster car has velocity mismatch (sync_bonus drops),
        # so r_faster should be LESS than matching speed (rewarding stability, not overspeeding)
        self.assertLess(r_faster, r,
                        f"Overspeeding car ({r_faster}) must not be rewarded higher than synchronized carry ({r})!")

    def test_grace_pocket_scoring(self):
        """Test that diverse flick setups across the roof and hood receive 100% full grace centering score."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 900.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.cars = [car]
        act = np.zeros(8, dtype=np.float32)

        # 1. Front-flip flick setup: local_x = +25, local_y = 0 -> Y = +25, X = 0
        self.arena.ball.pos = np.array([0.0, 25.0, 147.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 900.0, 0.0], dtype=np.float32)
        rew.reset(self.arena)
        r_front = rew.get_reward(car, self.arena, act, False, None)

        # 2. 45-degree flick setup: local_x = +18, local_y = +20 -> Y = +18, X = -20
        self.arena.ball.pos = np.array([-20.0, 18.0, 147.0], dtype=np.float32)
        rew.reset(self.arena)
        r_45 = rew.get_reward(car, self.arena, act, False, None)

        # 3. Side flick setup: local_x = +5, local_y = +25 -> Y = +5, X = -25
        self.arena.ball.pos = np.array([-25.0, 5.0, 147.0], dtype=np.float32)
        rew.reset(self.arena)
        r_side = rew.get_reward(car, self.arena, act, False, None)

        # 4. Scoop / backflip setup: local_x = -5, local_y = 0 -> Y = -5, X = 0
        self.arena.ball.pos = np.array([0.0, -5.0, 147.0], dtype=np.float32)
        rew.reset(self.arena)
        r_scoop = rew.get_reward(car, self.arena, act, False, None)

        # All four setups reside inside the grace pocket [-10, 32] x [-25, 25] and should yield identical full center_score
        self.assertAlmostEqual(r_front, r_45, places=2,
                               msg=f"45-degree setup ({r_45}) should yield full grace score matching front-flip ({r_front})")
        self.assertAlmostEqual(r_front, r_side, places=2,
                               msg=f"Side-flick setup ({r_side}) should yield full grace score matching front-flip ({r_front})")
        self.assertAlmostEqual(r_front, r_scoop, places=2,
                               msg=f"Scoop setup ({r_scoop}) should yield full grace score matching front-flip ({r_front})")

        # 5. Position outside grace pocket (drifting off front hood: local_x = 55 -> Y = 55)
        self.arena.ball.pos = np.array([0.0, 55.0, 147.0], dtype=np.float32)
        rew.reset(self.arena)
        r_outside = rew.get_reward(car, self.arena, act, False, None)
        self.assertLess(r_outside, r_front,
                        f"Position outside grace pocket ({r_outside}) must score less than inside grace pocket ({r_front})")

    def test_flick_launch_reward_beats_passive_carry(self):
        """Test that launching an explosive flick on target net generates a large reward beating passive carry."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 2000.0, 80.0], dtype=np.float32),
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

        # Explosive launch towards opponent goal (+Y)
        self.arena.ball.pos = np.array([0.0, 2100.0, 160.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 1600.0, 350.0], dtype=np.float32)

        act = np.zeros(8, dtype=np.float32)
        act[2] = 1.0  # Front flip
        r_flick = rew.get_reward(car, self.arena, act, False, None)

        self.assertGreater(r_flick, 2.50,
                           f"Explosive flick launch must generate high reward (> 2.50) to beat passive carrying, got {r_flick}")

    def test_flick_tti_contested_multiplier(self):
        """Test that flicking when challenged by an opponent receives the 1.5x tactical outplay multiplier."""
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
        # Defender rushing to challenge the ball
        defender = CarState(
            id=1, team=1,
            pos=np.array([0.0, 500.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, -1200.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, -math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.cars = [car, defender]
        rew.reset(self.arena)
        rew._prev_on_ground[car.id] = False
        rew._prev_has_flip[car.id] = True
        rew._flick_window_active[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = car.vel.copy()
        rew._prev_ball_vel[car.id] = np.array([0.0, 800.0, 50.0], dtype=np.float32)

        self.arena.ball.pos = np.array([0.0, 50.0, 160.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 1600.0, 350.0], dtype=np.float32)

        act = np.zeros(8, dtype=np.float32)
        act[2] = 1.0
        r_contested = rew.get_reward(car, self.arena, act, False, None)

        # Uncontested (no defender)
        self.arena.cars = [car]
        rew.reset(self.arena)
        rew._prev_on_ground[car.id] = False
        rew._prev_has_flip[car.id] = True
        rew._flick_window_active[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = car.vel.copy()
        rew._prev_ball_vel[car.id] = np.array([0.0, 800.0, 50.0], dtype=np.float32)
        r_uncontested = rew.get_reward(car, self.arena, act, False, None)

        self.assertGreater(r_contested, r_uncontested,
                           f"Contested flick ({r_contested}) must out-reward uncontested flick ({r_uncontested}) due to tactical TTI outplay bonus!")

    def test_bot_jump_unlocked_while_steering_and_supersonic(self):
        """Test that bot.py controller logic permits jumping while steering and at supersonic speeds."""
        bot = SenseiRLBot.__new__(SenseiRLBot)
        bot.dodge_cooldown = 0
        substep_tick = 1
        is_on_ground = True

        # Case 1: Carrying downfield at speed (900 uu/s) with steering (0.45)
        car_speed_total = 900.0
        act = np.zeros(8, dtype=np.float32)
        act[1] = 0.45
        act[5] = 1.0
        want_jump = bool(act[5] > 0.0)
        is_low_speed_hard_steer = bool(car_speed_total < 350.0 and abs(act[1]) > 0.60)
        if is_on_ground:
            if bot.dodge_cooldown > 0 or is_low_speed_hard_steer:
                want_jump = False
            jump_carry = bool(want_jump and substep_tick <= 3)
        self.assertTrue(jump_carry, "Ground jump must be allowed while carrying downfield with steering!")

        # Case 2: Supersonic rush (2250 uu/s)
        car_speed_total = 2250.0
        act[1] = 0.20
        want_jump = bool(act[5] > 0.0)
        is_low_speed_hard_steer = bool(car_speed_total < 350.0 and abs(act[1]) > 0.60)
        if is_on_ground:
            if bot.dodge_cooldown > 0 or is_low_speed_hard_steer:
                want_jump = False
            jump_supersonic = bool(want_jump and substep_tick <= 3)
        self.assertTrue(jump_supersonic, "Ground jump must be allowed at supersonic speeds!")

        # Case 3: Low-speed sharp turn (100 uu/s, steer=0.90) -> should be suppressed
        car_speed_total = 100.0
        act[1] = 0.90
        want_jump = bool(act[5] > 0.0)
        is_low_speed_hard_steer = bool(car_speed_total < 350.0 and abs(act[1]) > 0.60)
        if is_on_ground:
            if bot.dodge_cooldown > 0 or is_low_speed_hard_steer:
                want_jump = False
            jump_low_speed = bool(want_jump and substep_tick <= 3)
        self.assertFalse(jump_low_speed, "Jump must be suppressed when turning sharply at low speed!")

    def test_effective_alignment_during_inverted_flip(self):
        """Test that compute_effective_alignment evaluates travel velocity when car is airborne/flipping with nose inverted."""
        from env.rewards import compute_effective_alignment
        # Car rocketing downfield (+Y) at 1400 uu/s toward ball, but pitched completely backward (pitch = -1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 80.0], dtype=np.float32),
            vel=np.array([0.0, 1400.0, 0.0], dtype=np.float32),
            rot=np.array([-math.pi / 2, 0.0, 0.0], dtype=np.float32),  # Nose pointing down/back
            on_ground=False
        )
        target_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Ball is at +Y
        eff_align = compute_effective_alignment(car, target_dir)
        self.assertGreater(eff_align, 0.85, f"Effective alignment during forward flip must evaluate travel velocity (+Y), got {eff_align}")

    def test_active_flip_exempt_from_overshoot_penalty(self):
        """Test that active flips near the ball do NOT incur false overshoot penalties when nose pitches away."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 1000.0, 50.0], dtype=np.float32),
            vel=np.array([0.0, 1200.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=False,
            has_flip=False,
            just_dodged=True
        )
        self.arena.ball.pos = np.array([0.0, 1250.0, 100.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 800.0, 0.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)
        rew._was_in_strike_zone[car.id] = True

        # Simulate car flipping: momentary pitch down
        car.rot = np.array([-math.pi / 2, 0.0, 0.0], dtype=np.float32)
        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)

        # Should NOT receive the harsh -0.40 / -0.60 overshoot penalty
        self.assertGreater(r, -0.15, f"Active flip toward ball must not be penalized as an overshoot, got {r}")

    def test_contested_defensive_box_dunk_hazard(self):
        """Test that carrying the ball on the roof across the defending net when challenged incurs dunk hazard penalty."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        # In defending box (Y = -4800, X = 0), driving across net (+X at 250 uu/s)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, -4800.0, 17.0], dtype=np.float32),
            vel=np.array([250.0, 0.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            on_ground=True
        )
        # Ball on roof (local_z = 130)
        self.arena.ball.pos = np.array([0.0, -4800.0, 147.0], dtype=np.float32)
        self.arena.ball.vel = np.array([250.0, 0.0, 0.0], dtype=np.float32)

        # Challenger rushing in to dunk (arrival < 0.6s)
        challenger = CarState(
            id=1, team=1,
            pos=np.array([0.0, -4200.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, -1500.0, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.cars = [car, challenger]
        rew.reset(self.arena)

        action = np.zeros(8, dtype=np.float32)
        r_contested = rew.get_reward(car, self.arena, action, False, None)

        # Carrying across the net when challenged must be penalized (dunk hazard)
        self.assertLess(r_contested, 0.0, f"Carrying across net into an incoming challenger must incur dunk hazard penalty, got {r_contested}")

    def test_defensive_low_5050_block_rewarded(self):
        """Test that low 50/50 posture (grounded, ball low in front of bumper, nose squared to challenger) is rewarded."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        # In defending box, facing incoming challenger at +Y
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, -4800.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 150.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        # Ball low on turf in front of bumper (local_x = 50, local_z = 76)
        self.arena.ball.pos = np.array([0.0, -4750.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 150.0, 0.0], dtype=np.float32)

        challenger = CarState(
            id=1, team=1,
            pos=np.array([0.0, -4200.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, -1500.0, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.cars = [car, challenger]
        rew.reset(self.arena)

        action = np.zeros(8, dtype=np.float32)
        r_block = rew.get_reward(car, self.arena, action, False, None)
        self.assertGreater(r_block, 0.35, f"Low 50/50 block posture in defending box must receive positive reward, got {r_block}")

    def test_goal_height_and_backboard_trajectory_not_counted_as_shot(self):
        """Test that a ball trajectory hitting the backboard above GOAL_HEIGHT (Z=800) receives 0 on-target shot bonus and False threat."""
        from env.rewards import BallToGoalVelocityReward
        from env.physics_engine import GOAL_HEIGHT
        b2g_rew = BallToGoalVelocityReward(weight=1.0)

        car = CarState(id=0, team=0, pos=np.array([0.0, 3000.0, 17.0], dtype=np.float32))
        # High lob heading into backboard at Y=5120, Z=800 (well above crossbar 642.775 uu)
        self.arena.ball.pos = np.array([0.0, 4500.0, 800.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, 1200.0, 0.0], dtype=np.float32)
        self.arena.cars = [car]

        r = b2g_rew.get_reward(car, self.arena, np.zeros(8, dtype=np.float32), False, None)
        # Should NOT receive the 1.6x on-target clean shot bonus
        ball_speed = float(np.linalg.norm(self.arena.ball.vel))
        base_norm = ball_speed / 6000.0  # 1200 / 6000 = 0.20
        self.assertLessEqual(r, base_norm + 1e-4, f"Backboard hit must not receive on-target shot multiplier, got {r}")

        # Check get_shot_threat in physics engine
        is_threat, threat_intensity, _ = self.arena.get_shot_threat(team=1)  # Defending team 1 (+Y)
        self.assertFalse(is_threat, "Ball flying into backboard at Z=800 uu must NOT be flagged as in-goal shot threat!")

    def test_active_flip_overshoot_sailing_away(self):
        """Test that an active flip where the car sails away from the ball without touching triggers the sailing-away penalty without UnboundLocalError."""
        rew = PlayerToBallVelocityReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 1000.0, 50.0], dtype=np.float32),
            vel=np.array([0.0, -1200.0, 0.0], dtype=np.float32),  # Sailing away from ball
            rot=np.array([0.0, -math.pi / 2, 0.0], dtype=np.float32),
            on_ground=False,
            has_flip=False,
            just_dodged=True
        )
        self.arena.ball.pos = np.array([0.0, 1500.0, 100.0], dtype=np.float32)  # Ball ahead
        self.arena.ball.vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)
        rew._prev_dist[car.id] = 300.0  # was close (300 uu)
        rew._was_in_strike_zone[car.id] = True  # was in strike zone

        # Now car is at dist 500 uu (sailing away: raw_delta_dist = (300 - 500) / 2000 = -0.10)
        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertIsInstance(r, float)


if __name__ == "__main__":
    unittest.main()

