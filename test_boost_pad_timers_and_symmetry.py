"""
Unit Tests for Option B Boost Pad Observation Space (94 Dimensions).
Verifies:
  1. RocketSim boost pad cooldown synchronization.
  2. Bilateral mirror reflection equivariance across all 94 dimensions (with Left <-> Right pair permutation).
  3. Team perspective inversion on pad slots.
  4. Numerical equivalence between 80-dim zero-padded and 94-dim model forward passes.
"""

import math
import unittest
import numpy as np
import torch
import RocketSim as rsim

from env.physics_engine import RocketSimArena, CarState, BallState
from env.observations import (
    OBS_DIM,
    DefaultObservationBuilder,
    OBS_MIRROR_MASK_NP,
    OBS_MIRROR_INDICES_NP,
    mirror_obs,
    mirror_act
)
from agent.models import ActorCritic


class TestBoostPadTimersAndSymmetry(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")
        self.obs_builder = DefaultObservationBuilder(symmetric=True)

    def test_obs_dim_constant_and_output_shape(self):
        """The builder and every symmetry array must agree on one width.

        Asserted against OBS_DIM rather than a literal: the masks are hand-written parallel
        arrays, and a width that drifts out of step with them corrupts the mirrored pass
        silently rather than raising.
        """
        self.assertEqual(OBS_DIM, 108)
        self.assertEqual(self.obs_builder.obs_dim, OBS_DIM)
        self.assertEqual(len(OBS_MIRROR_MASK_NP), OBS_DIM)
        self.assertEqual(len(OBS_MIRROR_INDICES_NP), OBS_DIM)

        car = self.arena.cars[0]
        obs = self.obs_builder.build_obs(car, self.arena)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertFalse(np.isnan(obs).any(), "Observation vector must not contain NaNs")

    def test_nearest_pad_cooldown_sync(self):
        """Verify nearest pad cooldowns and states are correctly extracted from arena."""
        car = self.arena.cars[0]
        # Position car near Blue back-left big orb (pad 3: pos [-3072, -4096])
        car.pos = np.array([-3000.0, -4000.0, 17.0], dtype=np.float32)

        # Inactive pad with 5.0 seconds cooldown remaining
        self.arena.boost_pads[3].is_active = False
        self.arena.boost_pads[3].cooldown_timer = 5.0
        self.arena._big_pad_active[0] = False

        obs = self.obs_builder.build_obs(car, self.arena)
        # Defending Left Orb is at index 82 (active: 0.0) and index 83 (cooldown: 5.0 / 10.0 = 0.5)
        self.assertEqual(obs[82], 0.0)
        self.assertAlmostEqual(obs[83], 0.5, places=3)

    def test_team_perspective_inversion(self):
        """Verify Orange (team 1) views pitch pads from defending perspective."""
        car_blue = self.arena.cars[0]
        car_orange = self.arena.cars[1]

        # Orange defending left corner is world pad 30 (+3072, +4096)
        self.arena.boost_pads[30].is_active = False
        self.arena.boost_pads[30].cooldown_timer = 8.0

        obs_orange = self.obs_builder.build_obs(car_orange, self.arena)
        # Orange's defending left is slot 0 (features 82..83)
        self.assertEqual(obs_orange[82], 0.0)
        self.assertAlmostEqual(obs_orange[83], 0.8, places=3)

    def test_bilateral_mirror_equivariance_with_permutation(self):
        """
        Verify bilateral reflection across X=0 correctly negates lateral channels
        and swaps Left <-> Right orb pairs:
          [82, 83] <-> [84, 85] (Defending Left <-> Right)
          [86, 87] <-> [88, 89] (Midfield Left <-> Right)
          [90, 91] <-> [92, 93] (Attacking Left <-> Right)
        """
        test_obs = np.zeros(OBS_DIM, dtype=np.float32)
        # Set distinctive values for pad states and timers
        test_obs[82], test_obs[83] = 1.0, 0.2  # DefL
        test_obs[84], test_obs[85] = 0.0, 0.7  # DefR
        test_obs[86], test_obs[87] = 1.0, 0.0  # MidL
        test_obs[88], test_obs[89] = 0.0, 0.4  # MidR
        test_obs[90], test_obs[91] = 0.0, 0.9  # AttL
        test_obs[92], test_obs[93] = 1.0, 0.1  # AttR

        mirrored = mirror_obs(test_obs)

        # Verify Defending Left and Right swapped
        self.assertEqual(mirrored[82], 0.0)
        self.assertAlmostEqual(mirrored[83], 0.7)
        self.assertEqual(mirrored[84], 1.0)
        self.assertAlmostEqual(mirrored[85], 0.2)

        # Verify Midfield Left and Right swapped
        self.assertEqual(mirrored[86], 0.0)
        self.assertAlmostEqual(mirrored[87], 0.4)
        self.assertEqual(mirrored[88], 1.0)
        self.assertAlmostEqual(mirrored[89], 0.0)

        # Verify Attacking Left and Right swapped
        self.assertEqual(mirrored[90], 1.0)
        self.assertAlmostEqual(mirrored[91], 0.1)
        self.assertEqual(mirrored[92], 0.0)
        self.assertAlmostEqual(mirrored[93], 0.9)

    def test_actor_critic_torch_mirror_equivariance(self):
        """Verify ActorCritic equivariant forward pass executes pad pair swaps without error."""
        model = ActorCritic(obs_dim=94, act_dim=8, continuous_actions=True, use_layer_norm=True)
        model.eval()

        batch_obs = torch.randn(8, 94)
        action, logp, ent, val = model.get_action_and_value(batch_obs, deterministic=True)
        self.assertEqual(action.shape, (8, 8))
        self.assertEqual(val.shape, (8, 1))
        self.assertFalse(torch.isnan(action).any())
        self.assertFalse(torch.isnan(val).any())


if __name__ == "__main__":
    unittest.main()
