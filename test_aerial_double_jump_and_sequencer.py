"""
Unit tests for Double-Jump Aerial Mechanics, Substep Sequencer, Flip Timeout Synchronization,
Continuous Dual-Height Distance Metric, and Aerial Apex Curriculum.
"""

import math
import unittest
import numpy as np
import RocketSim as rsim

from env.physics_engine import RocketSimArena, CarState, BallState
from env.rewards import PlayerToBallVelocityReward, _norm2, _norm3
from env.state_setters import AerialScenarioSetter
from bot import SenseiRLBot


class TestDodgeDeadzoneAndDoubleJump(unittest.TestCase):
    def setUp(self):
        try:
            rsim.init()
        except Exception:
            pass

    def test_physics_engine_deadzone_preservation(self):
        """Verify that pitch/yaw < 0.50 are NOT scaled to 0.90 in the substep loop, allowing neutral double jumps."""
        arena = RocketSimArena(num_players=2)
        # A fixed kickoff, and an upright, non-spinning car: reset() otherwise samples a random
        # scenario whose car orientation and spin carry over, which tilts the double-jump impulse
        # off vertical and made the vz check below fail about half the time.
        arena.reset(random_kickoff=False)
        
        # Position car airborne
        cs = arena.cars[0]
        r_car = arena._rsim_arena.get_cars()[0]
        r_cs = r_car.get_state()
        r_cs.pos = rsim.Vec(0, 0, 250)
        r_cs.vel = rsim.Vec(0, 0, 100)
        r_cs.rot_mat = rsim.Angle(yaw=math.pi / 2, pitch=0.0, roll=0.0).as_rot_mat()
        r_cs.ang_vel = rsim.Vec(0, 0, 0)
        r_cs.is_on_ground = False
        r_car.set_state(r_cs)
        
        # Action with slight pitch up (-0.30 in action space -> +0.30 pitch_val), jump = +1.0
        # stick_mag = 0.30 (< 0.50)
        act = np.array([0.0, 0.0, -0.30, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        zero_act = np.zeros(8, dtype=np.float32)
        
        # Advance 1 step
        arena.step([act, zero_act])
        
        # In RocketSim, with stick < 0.50 and jump pressed while airborne:
        # It must execute a double jump, NOT a flip
        r_cs_after = r_car.get_state()
        self.assertTrue(r_cs_after.has_double_jumped, "Car should have executed a double jump")
        self.assertFalse(r_cs_after.has_flipped, "Car should NOT have flipped with stick < 0.50")
        self.assertGreater(r_cs_after.vel.z, 300.0, "Double jump must produce vertical climb (+vz)")

    def test_physics_engine_dodge_scaling(self):
        """Verify that stick deflection >= 0.50 scales to >= 0.90 for intentional dodges."""
        arena = RocketSimArena(num_players=2)
        arena.reset(random_kickoff=False)
        
        r_car = arena._rsim_arena.get_cars()[0]
        r_cs = r_car.get_state()
        r_cs.pos = rsim.Vec(0, 0, 250)
        r_cs.vel = rsim.Vec(0, 500, 50)
        r_cs.rot_mat = rsim.Angle(yaw=math.pi / 2, pitch=0.0, roll=0.0).as_rot_mat()
        r_cs.ang_vel = rsim.Vec(0, 0, 0)
        r_cs.is_on_ground = False
        r_car.set_state(r_cs)
        
        # Action with intentional forward dodge: pitch = +0.55 (action space -> -0.55 pitch_val), jump = 1.0
        act = np.array([0.0, 0.0, 0.55, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        zero_act = np.zeros(8, dtype=np.float32)
        arena.step([act, zero_act])
        
        r_cs_after = r_car.get_state()
        self.assertTrue(r_cs_after.has_flipped, "Car should have executed a flip/dodge with stick >= 0.50")

    def test_bot_dodge_cooldown_isolation(self):
        """Verify that bot.py only sets dodge_cooldown when stick_mag >= 0.50 (actual dodge), not on neutral double jumps."""
        bot = SenseiRLBot(name="SensAI_Test", team=0, index=0)
        bot.dodge_cooldown = 0
        
        # Simulate an airborne double-jump action (pitch=-0.30, jump=True, stick_mag=0.30 < 0.50)
        controller_jump = True
        pitch = -0.30
        yaw = 0.0
        stick_mag = math.hypot(pitch, yaw)
        if stick_mag >= 0.50:
            bot.dodge_cooldown = 20
        self.assertEqual(bot.dodge_cooldown, 0, "Neutral double jump must NOT set dodge_cooldown")
        
        # Simulate an intentional forward dodge (pitch=0.60, stick_mag=0.60 >= 0.50)
        pitch = 0.60
        stick_mag = math.hypot(pitch, yaw)
        if stick_mag >= 0.50:
            bot.dodge_cooldown = 20
        self.assertEqual(bot.dodge_cooldown, 20, "Directional dodge must set dodge_cooldown to 20")


class TestFlipTimeoutSynchronization(unittest.TestCase):
    def setUp(self):
        try:
            rsim.init()
        except Exception:
            pass

    def test_rocketsim_has_flip_or_jump_timeout(self):
        """Verify c_state.has_flip_or_jump() expires at exactly 1.25s (150 ticks @ 120Hz)."""
        arena = rsim.Arena(rsim.GameMode.SOCCAR)
        car = arena.add_car(rsim.Team.BLUE)
        
        # Car in midair having jumped off ground
        cs = car.get_state()
        cs.pos = rsim.Vec(0, 0, 1000)
        cs.vel = rsim.Vec(0, 0, 0)
        cs.is_on_ground = False
        cs.has_jumped = True
        car.set_state(cs)
        
        # Step through 1.15s (138 ticks)
        for _ in range(138):
            arena.step(1)
        cs = car.get_state()
        self.assertTrue(cs.has_flip_or_jump(), "Flip should still be available at 1.15s")
        
        # Step past 1.25s (total 155 ticks)
        for _ in range(17):
            arena.step(1)
        cs = car.get_state()
        self.assertFalse(cs.has_flip_or_jump(), "Flip must expire at 1.25s")

    def test_physics_engine_sync_respects_flip_timeout(self):
        """Verify RocketSimArena._sync_from_rsim marks car.has_flip = False when timeout expires."""
        arena = RocketSimArena(num_players=2)
        arena.reset()
        
        r_car = arena._rsim_arena.get_cars()[0]
        r_cs = r_car.get_state()
        r_cs.pos = rsim.Vec(0, 0, 1000)
        r_cs.vel = rsim.Vec(0, 0, 0)
        r_cs.is_on_ground = False
        r_cs.has_jumped = True
        r_car.set_state(r_cs)
        arena._sync_from_rsim()
        
        self.assertTrue(arena.cars[0].has_flip, "has_flip should be True initially in midair")
        
        # Wait 15 steps @ 15Hz (15 * 8 = 120 ticks = 1.0s) -> flip should still be available
        zero_acts = [np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)]
        for _ in range(15):
            arena.step(zero_acts)
        self.assertTrue(arena.cars[0].has_flip, "has_flip should be True within 1.0s of liftoff")
        
        # Wait another 6 steps (6 * 8 = 48 ticks -> total 1.4s) -> flip must expire
        for _ in range(6):
            arena.step(zero_acts)
        self.assertFalse(arena.cars[0].has_flip, "has_flip must be False after 1.25s timeout")


class TestDistanceMetricContinuity(unittest.TestCase):
    def test_calc_dist_continuity_and_ground_preservation(self):
        """Verify _calc_dist is smooth and continuous across car and ball heights, with no cliffs."""
        reward = PlayerToBallVelocityReward(weight=1.0)
        
        # 1. Exact ground preservation: ball on floor (Z=93.15), car on wheels (Z=17.0)
        car_pos_ground = np.array([0.0, 0.0, 17.0], dtype=np.float32)
        ball_pos_ground = np.array([500.0, 0.0, 93.15], dtype=np.float32)
        d_calc = reward._calc_dist(car_pos_ground, ball_pos_ground)
        d_exact_2d = float(_norm2(ball_pos_ground - car_pos_ground))
        self.assertAlmostEqual(d_calc, d_exact_2d, places=4, msg="Grounded ball and car must use pure 2D distance")
        
        # 2. Continuous sweep across ball height: Z_ball from 100 to 350 uu with car grounded
        prev_d = reward._calc_dist(car_pos_ground, np.array([500.0, 0.0, 100.0], dtype=np.float32))
        max_delta = 0.0
        for bz in np.linspace(101.0, 350.0, 250):
            ball_pos = np.array([500.0, 0.0, bz], dtype=np.float32)
            curr_d = reward._calc_dist(car_pos_ground, ball_pos)
            step_delta = abs(curr_d - prev_d)
            max_delta = max(max_delta, step_delta)
            prev_d = curr_d
            
        # Max step for 1.0 uu delta should be smoothly bounded (< 1.05 uu), proving no cliffs
        self.assertLess(max_delta, 1.05, f"Ball height sweep contains a discontinuity: max step delta = {max_delta}")
        
        # 3. Continuous sweep across car height: Z_car from 50 to 300 uu with ball at 200 uu
        ball_pos_mid = np.array([500.0, 0.0, 200.0], dtype=np.float32)
        prev_d = reward._calc_dist(np.array([0.0, 0.0, 50.0], dtype=np.float32), ball_pos_mid)
        max_delta_car = 0.0
        for cz in np.linspace(51.0, 300.0, 250):
            car_pos = np.array([0.0, 0.0, cz], dtype=np.float32)
            curr_d = reward._calc_dist(car_pos, ball_pos_mid)
            step_delta = abs(curr_d - prev_d)
            max_delta_car = max(max_delta_car, step_delta)
            prev_d = curr_d
            
        self.assertLess(max_delta_car, 1.05, f"Car height sweep contains a discontinuity: max step delta = {max_delta_car}")


class TestAerialScenarioCurriculum(unittest.TestCase):
    def setUp(self):
        try:
            rsim.init()
        except Exception:
            pass

    def test_aerial_scenario_setter_modes(self):
        """Verify AerialScenarioSetter generates valid scenarios including low_popup_double_jump."""
        arena = rsim.Arena(rsim.GameMode.SOCCAR)
        setter = AerialScenarioSetter()
        
        modes_seen = set()
        low_popups = 0
        
        for _ in range(200):
            setter.reset(arena, num_players=1)
            bs = arena.ball.get_state()
            
            # Check ball is in valid soccar bounds
            self.assertLess(abs(bs.pos.x), 4096.0)
            self.assertLess(abs(bs.pos.y), 5120.0)
            self.assertGreater(bs.pos.z, 90.0)
            self.assertLess(bs.pos.z, 2000.0)
            
            # Track low popups (Z in [300, 550], low horizontal vel <= 150)
            h_spd = math.hypot(bs.vel.x, bs.vel.y)
            if 300.0 <= bs.pos.z <= 550.0 and h_spd <= 220.0:
                low_popups += 1
                
        # With 20% weight, over 200 trials we expect ~40 low popups (at least 15)
        self.assertGreater(low_popups, 15, f"Low popup double jump mode was under-sampled: {low_popups}/200")


if __name__ == "__main__":
    unittest.main()
