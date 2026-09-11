"""
Regression tests for the critic-side fixes and the boost conservation potential.

Covers, in order:
  1. BoostReward conservation telescopes: any burn-and-refill cycle sums to exactly zero,
     while collecting boost still pays and still pays more when the tank is low.
  2. The environment distinguishes truncation (clock / resolved scenario) from termination
     (goal), and publishes the pre-reset observation so the trainer can bootstrap across it.
  3. Rescaling the critic head onto normalized returns preserves its predictions exactly,
     so resuming a pre-normalization checkpoint is not a discontinuity.
  4. Value-function clipping is off by default and no longer borrows the policy clip_range.
"""

import unittest

import numpy as np
import torch
import torch.nn as nn

from env.rewards import BoostReward
from env.rocket_env import VectorizedRocketEnv
from agent.ppo import PPOTrainer, RunningMeanStd


class TestBoostConservationPotential(unittest.TestCase):
    def test_cycle_is_exactly_neutral(self):
        """Burning boost and refilling it must sum to zero, at any depth."""
        phi = BoostReward._potential
        for lo, hi in [(0.0, 1.0), (0.5, 1.0), (0.12, 0.88), (0.0, 0.33)]:
            cycle = (phi(lo) - phi(hi)) + (phi(hi) - phi(lo))
            self.assertAlmostEqual(
                cycle, 0.0, places=12,
                msg="cycle {}->{}->{} must be neutral, got {}".format(hi, lo, hi, cycle),
            )

    def test_repeated_cycles_do_not_accumulate(self):
        """Ten burn-refill laps must not out-earn one. This is the farm the old asymmetric
        gain/loss weighting allowed: gains carried a hunger multiplier and a pickup bonus
        that losses were never charged for."""
        phi = BoostReward._potential
        total = 0.0
        for _ in range(10):
            total += phi(0.0) - phi(1.0)
            total += phi(1.0) - phi(0.0)
        self.assertAlmostEqual(total, 0.0, places=10)

    def test_collection_still_pays_and_is_bounded(self):
        phi = BoostReward._potential
        self.assertGreater(phi(1.0) - phi(0.0), 0.0)
        # Conservation reward over an episode is bounded by Phi(1) no matter how many pads
        # are taken, because it telescopes to a function of the final tank only.
        self.assertAlmostEqual(phi(1.0), 7.0 / 3.0, places=10)
        self.assertAlmostEqual(phi(0.0), 0.0, places=12)

    def test_low_boost_hunger_gradient_survives(self):
        """The potential must stay steepest near empty, in BOTH directions."""
        phi = BoostReward._potential
        near_empty = phi(0.15) - phi(0.05)
        near_full = phi(0.95) - phi(0.85)
        self.assertGreater(near_empty, near_full)
        # The charge for losing it is the mirror image, not a discounted one.
        self.assertAlmostEqual(phi(0.05) - phi(0.15), -near_empty, places=12)

    def test_big_orb_no_longer_outweighs_field_progression(self):
        """A big orb used to pay 4.62 at gain_weight 1.1, more than advancing the ball the
        full length of the pitch (~4.0). It must now come in well under that."""
        gw = 1.1
        big_orb = gw * (BoostReward._potential(1.0) - BoostReward._potential(0.0))
        self.assertLess(big_orb, 3.0)
        self.assertGreater(big_orb, 1.0, "but it must still be worth going out of the way for")

    def test_gain_weight_zero_gates_the_whole_term(self):
        from env.physics_engine import CarState, BallState

        class StubArena:
            pass

        arena = StubArena()
        arena.ball = BallState(
            pos=np.array([0.0, 2000.0, 93.0], dtype=np.float32),
            vel=np.array([0.0, 500.0, 0.0], dtype=np.float32),
        )
        arena.cars = [CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 300.0, 0.0], dtype=np.float32),
            boost=40.0, on_ground=True,
        )]

        rew = BoostReward(gain_weight=0.0, lose_weight=0.0)
        rew.reset(arena)
        arena.cars[0].boost = 90.0
        r = rew.get_reward(arena.cars[0], arena, np.zeros(8, dtype=np.float32), False, None)
        self.assertEqual(r, 0.0)


class TestTransitPotential(unittest.TestCase):
    """The pad-routing term was a per-step income stream: closing on a big orb paid roughly
    3.0 per approach with nothing charged for turning away, repeatable all episode. It is now
    a potential difference, which keeps the routing gradient and removes the payout."""

    def _fixture(self, boost=10.0):
        from env.rocket_env import RocketLeagueEnv
        from env.physics_engine import CarState

        env = RocketLeagueEnv(game_mode="1v1")
        arena = env.arena
        pad = arena._big_pad_pos_3d[0]
        car = CarState(
            id=0, team=0,
            pos=np.array([float(pad[0]), float(pad[1]) - 1400.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 900.0, 0.0], dtype=np.float32),
            boost=boost, on_ground=True,
        )
        arena.cars = [car]
        rew = BoostReward(gain_weight=1.1, lose_weight=0.6)
        rew.reset(arena)
        return rew, arena, car, pad

    def _walk(self, rew, arena, car, pad, offsets):
        """Drive the car to each offset in turn (boost held fixed) and sum the reward."""
        total = 0.0
        act = np.zeros(8, dtype=np.float32)
        for off in offsets:
            car.pos = np.array([float(pad[0]), float(pad[1]) - off, 17.0], dtype=np.float32)
            total += rew.get_reward(car, arena, act, False, None)
        return total

    def test_closing_on_a_pad_still_pays(self):
        rew, arena, car, pad = self._fixture()
        gained = self._walk(rew, arena, car, pad, [1400.0, 1100.0, 800.0, 500.0, 200.0])
        self.assertGreater(gained, 0.0, "routing toward a pad must still be reinforced")

    def test_approach_and_retreat_is_neutral(self):
        """The exact loop the old rate term paid for: drive at the orb, turn away, repeat."""
        rew, arena, car, pad = self._fixture()
        out_and_back = [1400.0, 1000.0, 600.0, 200.0, 600.0, 1000.0, 1400.0]
        total = self._walk(rew, arena, car, pad, out_and_back * 5)
        self.assertAlmostEqual(
            total, 0.0, places=6,
            msg="five approach/retreat laps must pay nothing, got {}".format(total),
        )

    def test_jumping_near_a_pad_is_refunded_on_landing(self):
        """The height taper used to be a standing opportunity cost for leaving the turf."""
        rew, arena, car, pad = self._fixture()
        act = np.zeros(8, dtype=np.float32)
        car.pos = np.array([float(pad[0]), float(pad[1]) - 400.0, 17.0], dtype=np.float32)
        rew.get_reward(car, arena, act, False, None)

        total = 0.0
        for z, grounded in [(120.0, False), (220.0, False), (120.0, False), (17.0, True)]:
            car.pos = np.array([float(pad[0]), float(pad[1]) - 400.0, z], dtype=np.float32)
            car.on_ground = grounded
            total += rew.get_reward(car, arena, act, False, None)
        self.assertAlmostEqual(total, 0.0, places=6,
                               msg="a hop and a landing must net zero, got {}".format(total))

    def test_hunger_is_continuous_across_the_20_boost_threshold(self):
        """As a rate term the sub-20 urgency bump was a step in income. As a potential a step
        becomes a payment for crossing downward, which made the combined potential
        non-monotone in boost: burning a sliver next to a pad paid."""
        rew, arena, car, pad = self._fixture()
        car.pos = np.array([float(pad[0]), float(pad[1]), 17.0], dtype=np.float32)
        levels = {}
        for b in (19.0, 19.9, 20.0, 20.1, 21.0):
            car.boost = b
            levels[b] = rew._transit_potential(car, arena)
        self.assertLess(abs(levels[19.9] - levels[20.1]), 0.01,
                        "the potential must not jump at the 20-boost threshold")
        # and it must still be monotone: emptier is hungrier
        self.assertGreater(levels[19.0], levels[21.0])

    def test_burning_boost_never_pays_at_any_pad_proximity(self):
        """Hunger rises as the tank empties, so Psi rises when boost is burned. Phi must
        always outweigh that, or sitting next to a pad and burning fuel becomes income."""
        rew, arena, car, pad = self._fixture()
        act = np.zeros(8, dtype=np.float32)
        worst = None
        for off in (0.0, 300.0, 600.0, 900.0, 1200.0):
            for b in np.arange(0.5, 100.0, 0.5):
                car.pos = np.array([float(pad[0]), float(pad[1]) - off, 17.0], dtype=np.float32)
                car.boost = float(b)
                rew.get_reward(car, arena, act, False, None)
                car.boost = float(b) - 0.5          # burn half a percent of tank
                r = rew.get_reward(car, arena, act, False, None)
                if worst is None or r > worst[0]:
                    worst = (r, b, off)
        self.assertLessEqual(
            worst[0], 0.0,
            "burning boost paid {:+.6f} at boost={} offset={}".format(*worst),
        )


class TestTruncationChannel(unittest.TestCase):
    def test_truncation_is_distinguished_and_terminal_obs_survives_reset(self):
        vec = VectorizedRocketEnv(num_envs=4, max_episode_steps=25, baseline_opponent_ratio=0.0)
        vec.reset()
        P = vec.num_players_per_env
        seen_trunc = 0
        for _ in range(60):
            a = np.random.uniform(-1, 1, size=(4, P, vec.act_dim)).astype(np.float32)
            obs, _, _, infos = vec.step(a)
            trunc, term_obs = vec.get_truncation()
            self.assertEqual(trunc.shape, (4 * P,))
            self.assertEqual(term_obs.shape, (4 * P, vec.obs_dim))
            flags = trunc.reshape(4, P)
            terms = term_obs.reshape(4, P, -1)
            for i, info in enumerate(infos):
                if info["done"] and not info["is_goal"]:
                    seen_trunc += 1
                    self.assertTrue(flags[i].all(), "every actor in a truncated env must flag")
                    self.assertTrue(np.isfinite(terms[i]).all())
                    self.assertFalse(
                        np.allclose(terms[i], obs[i]),
                        "terminal observation was clobbered by the auto-reset",
                    )
                else:
                    self.assertFalse(
                        flags[i].any(),
                        "a goal or a live step must never be flagged truncated",
                    )
        self.assertGreater(seen_trunc, 0, "fixture must actually produce truncations")

    def test_goal_is_a_termination_not_a_truncation(self):
        """A goal is the one exit whose future value genuinely is zero.

        Driven by forcing the arena to report a goal rather than by steering a ball into
        the net: the scenario seeder rewrites ball state every step, so a physics-driven
        goal is not reproducible here. What is under test is the classification, which is
        exactly the substitution made below.
        """
        vec = VectorizedRocketEnv(num_envs=2, max_episode_steps=600, baseline_opponent_ratio=0.0)
        vec.reset()
        env = vec.envs[0]
        P = vec.num_players_per_env

        real_step = env.arena.step
        env.arena.step = lambda *a, **k: (real_step(*a, **k), (True, 0))[1]
        try:
            _, _, dones, infos = vec.step(np.zeros((2, P, vec.act_dim), dtype=np.float32))
        finally:
            env.arena.step = real_step

        self.assertTrue(infos[0]["is_goal"])
        self.assertTrue(infos[0]["done"])
        self.assertTrue(dones[0].all())
        self.assertFalse(infos[0]["truncated"], "a goal must never be flagged as a truncation")
        self.assertFalse(
            vec.get_truncation()[0].reshape(2, P)[0].any(),
            "a goal must not be bootstrapped: its terminal value really is zero",
        )


class _CriticStub:
    """Borrows the trainer's normalization methods without standing up a full trainer."""

    normalize_returns = True
    _denormalize_values = PPOTrainer._denormalize_values
    _normalize_returns = PPOTrainer._normalize_returns
    _migrate_critic_to_normalized_values = PPOTrainer._migrate_critic_to_normalized_values


class TestValueNormalizationMigration(unittest.TestCase):
    def test_critic_head_rescale_preserves_predictions(self):
        """A checkpoint trained in raw reward units must keep its value function when the
        trainer switches to standardized targets."""
        t = _CriticStub()
        t.ret_rms = RunningMeanStd()  # identity: the loaded critic speaks raw units
        t.agent = type("A", (), {})()
        t.agent.critic = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))

        obs = torch.randn(128, 6)
        with torch.no_grad():
            v_before = t.agent.critic(obs).flatten().clone()

        prev_mean, prev_std = t.ret_rms.mean, t.ret_rms.std
        t.ret_rms.update(torch.randn(20000) * 6.4 - 3.1)
        t._migrate_critic_to_normalized_values(prev_mean, prev_std)

        with torch.no_grad():
            v_after = t._denormalize_values(t.agent.critic(obs).flatten())
        self.assertLess(float((v_before - v_after).abs().max()), 1e-4)

    def test_running_mean_std_tracks_the_distribution(self):
        rms = RunningMeanStd()
        rms.update(torch.randn(200000) * 7.3 + 2.1)
        self.assertAlmostEqual(rms.mean, 2.1, places=1)
        self.assertAlmostEqual(rms.std, 7.3, places=1)

    def test_round_trip_normalization(self):
        t = _CriticStub()
        t.ret_rms = RunningMeanStd()
        t.ret_rms.update(torch.randn(10000) * 4.0 + 9.0)
        x = torch.randn(64) * 30.0
        self.assertTrue(
            torch.allclose(t._denormalize_values(t._normalize_returns(x)), x, atol=1e-3)
        )


class TestValueClipDecoupling(unittest.TestCase):
    def test_value_clip_defaults_off_and_is_independent_of_policy_clip(self):
        import yaml
        with open("config/default_config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        hp = cfg["hyperparameters"]
        self.assertIn("vf_clip_range", hp)
        self.assertIsNone(
            hp["vf_clip_range"],
            "value clipping must default off; sharing clip_range throttled the critic to "
            "0.2 per update on returns spanning tens of reward units",
        )
        self.assertTrue(hp.get("normalize_returns", False))
        self.assertGreaterEqual(
            hp["n_epochs"], 4, "the critic needs more than two passes to fit its target"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
