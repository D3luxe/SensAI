"""
Regression tests for the bunny-hop fix while preserving wavedash mechanics.
Guarantees:
  1. Open-field non-kickoff ground liftoff earns no JumpBridgeReward (Branch 1d removed).
  2. Kickoff liftoff still earns weight * 1.30 * forward_alignment.
  3. A plain hop landing (no dodge, no speed gain) earns nothing from the landing impulse path.
  4. A low flip slam into turf is still rewarded as a wavedash.
  5. is_car_grounded only overrides OnGround shortly after the bot's own jump press.
  6. handbrake_allowed matches the model's can_slide buffer (grounded or z < 120).
  7. Deterministic thresholds come from BIN_THRESH_LOGITS, with handbrake at -0.4055.
"""

import unittest
import numpy as np

from env.physics_engine import RocketSimArena
from env.rewards import JumpBridgeReward, compute_effective_alignment
from agent.models import ActorCritic, BIN_THRESH_LOGITS
from bot import is_car_grounded, handbrake_allowed

ACT = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0], dtype=np.float32)


class TestBunnyHopAndWavedash(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena()
        self.arena.reset()
        self.car = self.arena.cars[0]
        self.car.team = 0
        self.car.rot_mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

    def _liftoff_reward(self, x_offset: float) -> float:
        """Car lifts off 2000 uu behind the ball, driving straight at it at 1200 uu/s."""
        bridge = JumpBridgeReward(weight=1.0)
        bridge.reset(self.arena)
        car = self.car
        car.pos = np.array([x_offset, -2000.0, 30.0], dtype=np.float32)
        car.vel = np.array([0.0, 1200.0, 300.0], dtype=np.float32)
        car.on_ground = False
        car.has_flip = True
        car.just_dodged = False
        self.arena.ball.pos = np.array([x_offset, 0.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.zeros(3, dtype=np.float32)
        bridge._prev_on_ground[car.id] = True
        bridge._prev_has_flip[car.id] = True
        bridge._prev_vel[car.id] = car.vel.copy()
        bridge._prev_pos_z[car.id] = 17.0
        return bridge.get_reward(car, self.arena, ACT, False, None)

    def test_open_field_liftoff_unrewarded(self):
        self.assertAlmostEqual(self._liftoff_reward(x_offset=1500.0), 0.0, places=4)

    def test_kickoff_liftoff_rewarded(self):
        rew = self._liftoff_reward(x_offset=0.0)
        to_ball = self.arena.ball.pos - self.car.pos
        align = compute_effective_alignment(self.car, to_ball / np.linalg.norm(to_ball))
        self.assertAlmostEqual(rew, 1.30 * align, places=3)

    def test_plain_hop_landing_unrewarded(self):
        bridge = JumpBridgeReward(weight=1.0)
        bridge.reset(self.arena)
        car = self.car
        car.pos = np.array([1500.0, -2000.0, 17.0], dtype=np.float32)
        car.vel = np.array([0.0, 1200.0, 0.0], dtype=np.float32)
        car.on_ground = True
        car.has_flip = True
        car.just_dodged = False
        self.arena.ball.pos = np.array([1500.0, 0.0, 93.0], dtype=np.float32)
        bridge._prev_on_ground[car.id] = False
        bridge._prev_has_flip[car.id] = True
        bridge._prev_vel[car.id] = car.vel.copy()
        bridge._prev_pos_z[car.id] = 30.0
        self.assertAlmostEqual(bridge.get_reward(car, self.arena, ACT, False, None), 0.0, places=4)

    def test_wavedash_still_rewarded(self):
        bridge = JumpBridgeReward(weight=1.0)
        bridge.reset(self.arena)
        car = self.car
        car.pos = np.array([0.0, 0.0, 40.0], dtype=np.float32)
        car.on_ground = True
        car.just_dodged = True
        bridge._prev_pos_z[car.id] = 40.0
        bridge._prev_vel[car.id] = np.array([0.0, 500.0, 0.0], dtype=np.float32)
        car.vel = np.array([0.0, 800.0, 0.0], dtype=np.float32)
        self.arena.ball.pos = np.array([0.0, 2500.0, 93.0], dtype=np.float32)
        self.assertGreaterEqual(bridge.get_reward(car, self.arena, ACT, False, None), 1.0)

    def test_grounding_override_scoped_to_own_jump(self):
        self.assertFalse(is_car_grounded(True, recently_jumped=True, z=25.0, vz=150.0))
        # Driving onto the ball / suspension rebound without a recent press stays grounded.
        self.assertTrue(is_car_grounded(True, recently_jumped=False, z=25.0, vz=150.0))
        # Resting on the floor right after a press (not yet rising) stays grounded.
        self.assertTrue(is_car_grounded(True, recently_jumped=True, z=17.0, vz=0.0))
        self.assertFalse(is_car_grounded(False, recently_jumped=False, z=17.0, vz=0.0))

    def test_handbrake_landing_buffer(self):
        self.assertTrue(handbrake_allowed(1.0, on_ground=False, z=80.0))
        self.assertFalse(handbrake_allowed(1.0, on_ground=False, z=200.0))
        self.assertTrue(handbrake_allowed(1.0, on_ground=True, z=500.0))
        self.assertFalse(handbrake_allowed(-1.0, on_ground=True, z=17.0))

    def test_thresholds_shared(self):
        model = ActorCritic(obs_dim=74, act_dim=8, continuous_actions=True)
        self.assertEqual(len(BIN_THRESH_LOGITS), 3)
        self.assertAlmostEqual(BIN_THRESH_LOGITS[2], -0.4055, places=4)
        for i, expected in enumerate(BIN_THRESH_LOGITS):
            self.assertAlmostEqual(model.bin_thresh_logits[i].item(), expected, places=4)


if __name__ == "__main__":
    unittest.main()
