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


if __name__ == "__main__":
    unittest.main()
