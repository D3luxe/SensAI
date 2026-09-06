"""
Unit and Integration Tests for Physical Condition Action Masking in SenseiBot.
Validates all 6 gating rules, std collapse, active entropy normalization, and bilateral equivariance.
"""

import unittest
import torch
import numpy as np

from agent.models import ActorCritic
from env.observations import OBS_MIRROR_MASK_NP, ACT_MIRROR_MASK_NP


class TestActionMasking(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.obs_dim = 80
        self.act_dim = 8
        self.model = ActorCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            continuous_actions=True,
            use_action_masking=True,
            handbrake_height_buffer=120.0
        )
        self.model.eval()

    def _create_test_obs(
        self,
        z: float = 17.0,
        boost: float = 33.3,
        on_ground: bool = True,
        has_jump: bool = True,
        has_flip: bool = True
    ) -> torch.Tensor:
        """Helper to create a single 80-dim observation tensor with specified self-car indicators."""
        obs = torch.zeros(self.obs_dim, dtype=torch.float32)
        # 2: car_pos.z normalized by 2044.0
        obs[2] = z / 2044.0
        # 18: boost normalized by 0.01
        obs[18] = boost * 0.01
        # 19: on_ground
        obs[19] = 1.0 if on_ground else 0.0
        # 20: has_jump
        obs[20] = 1.0 if has_jump else 0.0
        # 21: has_flip
        obs[21] = 1.0 if has_flip else 0.0
        return obs

    def test_boost_gating(self):
        """Boost press must be masked when boost == 0, and available when boost > 0."""
        # 1. Zero boost -> Boost button (bin index 1 / act index 6) must be masked
        obs_empty = self._create_test_obs(boost=0.0).unsqueeze(0)
        with torch.no_grad():
            act, logp, ent, val = self.model.get_action_and_value(obs_empty, deterministic=False)
        self.assertEqual(act[0, 6].item(), -1.0, "Boost button should be strictly off (-1.0) when boost is 0")

        # Deterministic check
        with torch.no_grad():
            act_det, _, _, _ = self.model.get_action_and_value(obs_empty, deterministic=True)
        self.assertEqual(act_det[0, 6].item(), -1.0, "Deterministic boost button must be -1.0 when boost is 0")

        # 2. Positive boost -> Boost button should be eligible (not hard-masked to -1e4)
        obs_has_boost = self._create_test_obs(boost=50.0).unsqueeze(0)
        # Artificially set actor_binary bias high for boost to test activation
        with torch.no_grad():
            self.model.actor_binary.bias.data[1] = 5.0
            act_boost, _, _, _ = self.model.get_action_and_value(obs_has_boost, deterministic=True)
        self.assertEqual(act_boost[0, 6].item(), 1.0, "Boost button should be pressed when boost > 0 and logit is high")

    def test_jump_gating(self):
        """Jump should be permitted on ground or when flip is available, but masked when airborne with no flip."""
        # Ground with jump -> allowed
        obs_ground = self._create_test_obs(on_ground=True, has_jump=True, has_flip=True).unsqueeze(0)
        with torch.no_grad():
            self.model.actor_binary.bias.data[0] = 5.0
            act_ground, _, _, _ = self.model.get_action_and_value(obs_ground, deterministic=True)
        self.assertEqual(act_ground[0, 5].item(), 1.0, "Jump should be allowed on ground")

        # Airborne with flip -> allowed (double jump / dodge / flip reset)
        obs_air_flip = self._create_test_obs(on_ground=False, z=500.0, has_jump=False, has_flip=True).unsqueeze(0)
        with torch.no_grad():
            self.model.actor_binary.bias.data[0] = 5.0
            act_air_flip, _, _, _ = self.model.get_action_and_value(obs_air_flip, deterministic=True)
        self.assertEqual(act_air_flip[0, 5].item(), 1.0, "Jump should be allowed in air when has_flip=True")

        # Airborne with no flip -> strictly masked
        obs_air_noflip = self._create_test_obs(on_ground=False, z=500.0, has_jump=False, has_flip=False).unsqueeze(0)
        with torch.no_grad():
            act_air_noflip, _, _, _ = self.model.get_action_and_value(obs_air_noflip, deterministic=True)
            act_air_noflip_stoch, _, _, _ = self.model.get_action_and_value(obs_air_noflip, deterministic=False)
        self.assertEqual(act_air_noflip[0, 5].item(), -1.0, "Jump must be -1.0 when airborne with no flip")
        self.assertEqual(act_air_noflip_stoch[0, 5].item(), -1.0, "Stochastic jump must be -1.0 when airborne with no flip")

    def test_handbrake_landing_buffer(self):
        """Handbrake allowed on ground and below 120 uu, masked at high altitude in air."""
        # 1. Ground -> allowed
        obs_ground = self._create_test_obs(on_ground=True, z=17.0).unsqueeze(0)
        with torch.no_grad():
            self.model.actor_binary.bias.data[2] = 5.0
            act_ground, _, _, _ = self.model.get_action_and_value(obs_ground, deterministic=True)
        self.assertEqual(act_ground[0, 7].item(), 1.0, "Handbrake allowed on ground")

        # 2. Airborne but below 120 uu (landing recovery / wavedash prep) -> allowed
        obs_near_ground = self._create_test_obs(on_ground=False, z=80.0).unsqueeze(0)
        with torch.no_grad():
            self.model.actor_binary.bias.data[2] = 5.0
            act_near, _, _, _ = self.model.get_action_and_value(obs_near_ground, deterministic=True)
        self.assertEqual(act_near[0, 7].item(), 1.0, "Handbrake allowed within landing buffer (< 120 uu)")

        # 3. Airborne at high altitude (> 120 uu) -> strictly masked
        obs_high_air = self._create_test_obs(on_ground=False, z=500.0).unsqueeze(0)
        with torch.no_grad():
            act_high_det, _, _, _ = self.model.get_action_and_value(obs_high_air, deterministic=True)
            act_high_stoch, _, _, _ = self.model.get_action_and_value(obs_high_air, deterministic=False)
        self.assertEqual(act_high_det[0, 7].item(), -1.0, "Handbrake must be -1.0 at high altitude")
        self.assertEqual(act_high_stoch[0, 7].item(), -1.0, "Stochastic handbrake must be -1.0 at high altitude")

    def test_ground_rotation_freeze(self):
        """Pitch (2), Yaw (3), and Roll (4) must be strictly 0 on ground with collapsed std (1e-4)."""
        obs_ground = self._create_test_obs(on_ground=True).unsqueeze(0)
        with torch.no_grad():
            act_det, _, _, _ = self.model.get_action_and_value(obs_ground, deterministic=True)
            # Sample 20 times to confirm stochastic sampling never deviates from 0.0
            samples = [self.model.get_action_and_value(obs_ground, deterministic=False)[0] for _ in range(20)]

        # Check deterministic mean
        self.assertEqual(act_det[0, 2].item(), 0.0, "Deterministic Pitch on ground must be 0.0")
        self.assertEqual(act_det[0, 3].item(), 0.0, "Deterministic Yaw on ground must be 0.0")
        self.assertEqual(act_det[0, 4].item(), 0.0, "Deterministic Roll on ground must be 0.0")

        # Check stochastic samples
        for s in samples:
            self.assertAlmostEqual(s[0, 2].item(), 0.0, places=4, msg="Stochastic Pitch on ground must be 0.0")
            self.assertAlmostEqual(s[0, 3].item(), 0.0, places=4, msg="Stochastic Yaw on ground must be 0.0")
            self.assertAlmostEqual(s[0, 4].item(), 0.0, places=4, msg="Stochastic Roll on ground must be 0.0")

        # Airborne -> rotational axes must be active and free to explore
        obs_air = self._create_test_obs(on_ground=False, z=500.0).unsqueeze(0)
        with torch.no_grad():
            # Inject bias to test that airborne rotation is responsive
            self.model.actor_mean.bias.data[2] = 0.8
            act_air, _, _, _ = self.model.get_action_and_value(obs_air, deterministic=True)
        self.assertGreater(act_air[0, 2].item(), 0.5, "Airborne pitch must be responsive to policy")

    def test_airborne_throttle_clamp(self):
        """Reverse throttle (< 0.0) must be clamped to >= 0.0 while airborne."""
        # Set negative throttle bias
        with torch.no_grad():
            self.model.actor_mean.bias.data[0] = -0.9

        # Ground -> negative throttle allowed (reversing)
        obs_ground = self._create_test_obs(on_ground=True).unsqueeze(0)
        with torch.no_grad():
            act_ground, _, _, _ = self.model.get_action_and_value(obs_ground, deterministic=True)
        self.assertLess(act_ground[0, 0].item(), 0.0, "Ground reverse throttle should be negative")

        # Airborne -> negative throttle clamped to >= 0.0
        obs_air = self._create_test_obs(on_ground=False, z=500.0).unsqueeze(0)
        with torch.no_grad():
            act_air, _, _, _ = self.model.get_action_and_value(obs_air, deterministic=True)
        self.assertGreaterEqual(act_air[0, 0].item(), 0.0, "Airborne throttle must be clamped to >= 0.0")

    def test_active_entropy_normalization(self):
        """Entropy must be valid, finite, and non-zero across varied states."""
        obs_batch = torch.stack([
            self._create_test_obs(on_ground=True, boost=0.0, has_jump=True, has_flip=True),
            self._create_test_obs(on_ground=False, z=500.0, boost=100.0, has_jump=False, has_flip=False),
            self._create_test_obs(on_ground=False, z=50.0, boost=50.0, has_jump=False, has_flip=True)
        ])
        with torch.no_grad():
            act, logp, ent, val = self.model.get_action_and_value(obs_batch)

        self.assertEqual(ent.shape, (3,), "Entropy shape should match batch size")
        self.assertTrue(torch.isfinite(ent).all(), "Entropy values must be finite")
        self.assertTrue((ent > 0.0).all(), "Entropy values must be strictly positive")
        self.assertTrue(torch.isfinite(logp).all(), "Log-probs must be finite")

    def test_bilateral_equivariance_with_masking(self):
        """Masking must preserve strict bilateral symmetry under left-right reflection."""
        # Create an arbitrary asymmetric observation
        obs = torch.randn(self.obs_dim)
        obs[2] = 200.0 / 2044.0   # in air
        obs[18] = 0.5             # has boost
        obs[19] = 0.0             # airborne
        obs[20] = 0.0
        obs[21] = 1.0             # has flip

        obs_batch = obs.unsqueeze(0)
        obs_mirror = (obs * self.model.obs_mirror_mask).unsqueeze(0)

        with torch.no_grad():
            act1, _, _, _ = self.model.get_action_and_value(obs_batch, deterministic=True)
            act2, _, _, _ = self.model.get_action_and_value(obs_mirror, deterministic=True)

        # act2 should match act1 * act_mirror_mask
        expected_act2 = act1 * self.model.act_mirror_mask
        diff = (act2 - expected_act2).abs().max().item()
        self.assertLess(diff, 1e-5, f"Bilateral equivariance violated with difference: {diff}")

    def test_unmasked_toggle(self):
        """When use_action_masking=False or apply_masking=False, model operates unmasked."""
        obs_ground = self._create_test_obs(on_ground=True).unsqueeze(0)
        with torch.no_grad():
            self.model.actor_mean.bias.data[2] = 0.8
            # Pass apply_masking=False
            act_unmasked, _, _, _ = self.model.get_action_and_value(obs_ground, deterministic=True, apply_masking=False)
        self.assertGreater(act_unmasked[0, 2].item(), 0.5, "Unmasked pass should allow pitch on ground")


if __name__ == "__main__":
    unittest.main()
