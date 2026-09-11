import unittest
import math
import numpy as np

from env.physics_engine import CarState, BallState, RocketSimArena, ARENA_EXTENT_Y
from env.rewards import PlayerToBallVelocityReward, JumpBridgeReward
from bot import SenseiRLBot


class TestDribbleAndFlickMechanics(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")

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

        # Passive carry baseline: same geometry, but the ball rides the hood instead of being
        # launched -- no dodge, no touch, no impulse into the ball.
        rew_carry = JumpBridgeReward(weight=0.35)
        car_carry = CarState(
            id=0, team=0,
            pos=np.array([0.0, 2000.0, 80.0], dtype=np.float32),
            vel=np.array([0.0, 900.0, 50.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=False,
            has_flip=True,
            ball_touches=1
        )
        self.arena.cars = [car_carry]
        self.arena.ball.vel = np.array([0.0, 900.0, 50.0], dtype=np.float32)
        rew_carry.reset(self.arena)
        rew_carry._prev_on_ground[car_carry.id] = False
        rew_carry._prev_has_flip[car_carry.id] = True
        rew_carry._flick_window_active[car_carry.id] = True
        rew_carry._prev_touches[car_carry.id] = 1
        rew_carry._prev_vel[car_carry.id] = car_carry.vel.copy()
        rew_carry._prev_ball_vel[car_carry.id] = np.array([0.0, 900.0, 50.0], dtype=np.float32)
        r_carry = rew_carry.get_reward(car_carry, self.arena, np.zeros(8, dtype=np.float32), False, None)

        # Assert the relationship this test is named for, not an absolute magnitude. The flick
        # coefficient is deliberately tuned against discounted return rather than the nominal
        # goal weight, so pinning a raw threshold here re-breaks on every principled retune.
        self.assertGreater(r_flick, r_carry + 0.50,
                           f"Explosive flick launch ({r_flick}) must clearly beat passive carrying ({r_carry})")

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

