"""
Reward v6 (env/rewards_v6.py) against its specification, docs/reward_v6_align_spec.md.

  - T6 align: the values at the spec's canonical positions, the corrected attacking half, mirrored
    for orange, and the degenerate-point guard
  - T1-T5 are v5's terms unchanged: with the align term off, v6 pays exactly what v5 pays
  - potentials telescope with the align term included, and phi(terminal) = 0 on a goal step
  - the reward never depends on the action (R2)
  - the registry builds v6 at v5's gamma, and the trainer treats v5 -> v6 as a version change
"""
import unittest
from types import SimpleNamespace

import numpy as np

from env.rewards_v3 import RewardManagerV3, ball_position_potential, boost_potential, goal_centres
from env.rewards_v6 import REWARD_V6_DEFAULTS, RewardManagerV6, TERM_NAMES, align_potential

GAMMA = 0.9977


def car(cid, team, pos, boost=33.0, touches=0):
    return SimpleNamespace(id=cid, team=team, pos=np.array(pos, dtype=np.float32), boost=boost, ball_touches=touches)


def arena(ball_pos, ball_vel=(0.0, 0.0, 0.0), cars=()):
    return SimpleNamespace(ball=SimpleNamespace(pos=np.array(ball_pos, dtype=np.float32),
                                                vel=np.array(ball_vel, dtype=np.float32)), cars=list(cars))


def manager(**w):
    return RewardManagerV6({**REWARD_V6_DEFAULTS, **w})


def halves(car_pos, ball_pos, team=0):
    """(defending, attacking) halves, computed independently of the module under test."""
    own, opp = goal_centres(team)
    c, b = np.array(car_pos, float), np.array(ball_pos, float)
    cos = lambda u, v: float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
    return cos(b - c, c - own), cos(b - c, opp - c)


class TestAlign(unittest.TestCase):
    # (car, ball, defending, attacking, phi) -- the table in the spec, section 2
    CANONICAL = {
        "behind the ball on the axis": ((0, -3000, 17), (0, 0, 93), 0.99, 1.00, 0.99),
        "caught upfield": ((0, 2000, 17), (0, -1000, 93), -1.00, -0.99, -1.00),
        "centre kickoff": ((0, -4608, 17), (0, 0, 93), 0.85, 1.00, 0.93),
        "off-centre kickoff": ((-256, -3840, 17), (0, 0, 93), 0.94, 1.00, 0.97),
        "diagonal kickoff": ((-2048, -2560, 17), (0, 0, 93), 0.22, 0.92, 0.57),
        "level with the ball, out wide": ((0, 0, 17), (3000, 0, 93), 0.00, 0.00, 0.00),
        "net mouth, ball in the corner": ((0, -4500, 17), (3500, -4000, 93), 0.12, 0.14, 0.13),
    }

    def test_the_canonical_positions(self):
        for name, (c, b, d, at, phi) in self.CANONICAL.items():
            blue = car(0, 0, c)
            got = align_potential(blue, arena(b, cars=[blue]))
            self.assertAlmostEqual(got, phi, delta=0.006, msg=name)
            hd, ha = halves(c, b)
            self.assertAlmostEqual(hd, d, delta=0.006, msg=f"{name}: defending")
            self.assertAlmostEqual(ha, at, delta=0.006, msg=f"{name}: attacking")
            self.assertAlmostEqual(got, 0.5 * hd + 0.5 * ha, places=5, msg=name)

    def test_the_attacking_half_is_rlgyms_not_the_papers_misprint(self):
        """
        Seer prints the attacking half as cos(c - b, opp - c). Behind the ball on the line to the
        opponent's net, that reads -1: it would punish the position the paper says it rewards.
        """
        c, b = np.array([0, -3000, 93.0]), np.array([0, 0, 93.0])
        _, opp = goal_centres(0)
        printed = float((c - b) @ (opp - c) / np.linalg.norm(c - b) / np.linalg.norm(opp - c))
        self.assertLess(printed, -0.99)
        blue = car(0, 0, c)
        self.assertGreater(align_potential(blue, arena(b, cars=[blue])), 0.98)

    def test_orange_is_the_mirror_of_blue(self):
        rng = np.random.default_rng(5)
        for _ in range(50):
            c, b = rng.uniform(-4000, 4000, 3), rng.uniform(-4000, 4000, 3)
            mirror = np.array([1.0, -1.0, 1.0])
            blue, orange = car(0, 0, c), car(1, 1, c * mirror)
            self.assertAlmostEqual(align_potential(blue, arena(b, cars=[blue])),
                                   align_potential(orange, arena(b * mirror, cars=[orange])), places=5)

    def test_bounded(self):
        rng = np.random.default_rng(6)
        for _ in range(500):
            blue = car(0, 0, rng.uniform(-4000, 4000, 3))
            v = align_potential(blue, arena(rng.uniform(-4000, 4000, 3), cars=[blue]))
            self.assertLessEqual(abs(v), 1.0 + 1e-9)

    def test_a_degenerate_half_is_zero(self):
        """Car on the ball: both directions from the car to the ball are undefined."""
        blue = car(0, 0, (100, 200, 93))
        self.assertEqual(align_potential(blue, arena((100, 200, 93), cars=[blue])), 0.0)
        # car on its own goal centre: only the defending half is undefined
        own, _ = goal_centres(0)
        blue = car(0, 0, own)
        _, attacking = halves(own + np.array([0, 1e-3, 0]), (0, 0, 93))
        self.assertAlmostEqual(align_potential(blue, arena((0, 0, 93), cars=[blue])), 0.5 * attacking, places=4)

    def test_pays_weighted_gamma_difference(self):
        blue, orange = car(0, 0, (0, 2000, 17)), car(1, 1, (0, 3000, 17))
        a = arena((0, 0, 93), cars=[blue, orange])
        m = manager(goal_reward=0.0, touch_weight=0.0, ball_position_weight=0.0, boost_weight=0.0, align_weight=2.0)
        m.reset(a)
        old = align_potential(blue, a)
        blue.pos = np.array([0, -2000, 17], np.float32)      # recovers from upfield to goal-side
        r, b = m.get_reward(blue, a, None, False, None)
        self.assertAlmostEqual(r, 2.0 * (GAMMA * align_potential(blue, a) - old), places=6)
        self.assertGreater(b["align"], 3.5)


class TestV5TermsUnchanged(unittest.TestCase):
    def test_align_off_pays_exactly_v5(self):
        """With T6 at zero, v6 is v5 (v3's terms at 0.9977), term for term, over a random trajectory."""
        rng = np.random.default_rng(11)
        blue, orange = car(0, 0, (0, -2000, 17)), car(1, 1, (0, 2000, 17))
        a = arena((0, 0, 93), cars=[blue, orange])
        v5 = RewardManagerV3({"closeness_weight": 0.0, "gamma": GAMMA})
        v6 = manager(align_weight=0.0)
        v5.reset(a)
        v6.reset(a)
        for step in range(60):
            a.ball.pos = rng.uniform(-4000, 4000, 3).astype(np.float32)
            a.ball.vel = rng.uniform(-2000, 2000, 3).astype(np.float32)
            for c in (blue, orange):
                c.pos = rng.uniform(-4000, 4000, 3).astype(np.float32)
                c.boost = float(rng.uniform(0, 100))
                c.ball_touches += int(rng.random() < 0.2)
            goal = step % 25 == 24
            for c in (blue, orange):
                r5, b5 = v5.get_reward(c, a, None, goal, 0 if goal else None)
                r6, b6 = v6.get_reward(c, a, None, goal, 0 if goal else None)
                self.assertAlmostEqual(r5, r6, places=9)
                for k, v in b5.items():
                    self.assertAlmostEqual(v, b6[k], places=9, msg=k)
                self.assertEqual(b6["align"], 0.0)

    def test_defaults_are_v5s_plus_align(self):
        from env.reward_registry import load_snapshot
        v5 = load_snapshot("v5")["settings"]
        self.assertEqual(REWARD_V6_DEFAULTS["gamma"], v5["gamma"])
        self.assertEqual({k: v for k, v in REWARD_V6_DEFAULTS.items() if k not in ("gamma", "align_weight")},
                         v5["rewards"])
        self.assertEqual(REWARD_V6_DEFAULTS["align_weight"], 1.0)

    def test_breakdown_names_the_six_terms_and_sums_to_total(self):
        blue, orange = car(0, 0, (0, -2000, 17)), car(1, 1, (0, 2000, 17))
        a = arena((0, 0, 93), cars=[blue, orange])
        m = manager()
        m.reset(a)
        blue.pos = np.array([300, -500, 17], np.float32)
        r, b = m.get_reward(blue, a, None, False, None)
        self.assertEqual(TERM_NAMES, ("goal", "ball_position", "touch", "closeness", "boost", "align"))
        self.assertEqual(list(b), list(TERM_NAMES))
        self.assertAlmostEqual(sum(b.values()), r, places=6)


class TestPotentials(unittest.TestCase):
    def _loop_sum(self, gamma):
        rng = np.random.default_rng(3)
        states = [(rng.uniform(-3000, 3000, 3), rng.uniform(-3000, 3000, 3), rng.uniform(0, 100))
                  for _ in range(12)]
        states.append(states[0])   # closed loop
        c, o = car(0, 0, states[0][1], boost=states[0][2]), car(1, 1, (0, 3000, 17))
        a = arena(states[0][0], cars=[c, o])
        m = manager(goal_reward=0.0, touch_weight=0.0, gamma=gamma)
        m.reset(a)
        total, phis = 0.0, []
        for ball, pos, boost in states[1:]:
            a.ball.pos, c.pos, c.boost = np.array(ball, np.float32), np.array(pos, np.float32), boost
            phis.append(5.0 * ball_position_potential(c, a) + boost_potential(c, a) + align_potential(c, a))
            total += m.get_reward(c, a, None, False, None)[0]
        return total, phis

    def test_closed_loop_sums_to_zero_undiscounted(self):
        total, _ = self._loop_sum(1.0)
        self.assertAlmostEqual(total, 0.0, places=5)

    def test_closed_loop_leaves_only_the_discount_remainder(self):
        total, phis = self._loop_sum(GAMMA)
        self.assertAlmostEqual(total, (GAMMA - 1.0) * sum(phis), places=5)

    def test_potential_is_zero_at_a_goal(self):
        c, o = car(0, 0, (500, -2000, 17), boost=64.0), car(1, 1, (0, 4000, 17))
        a = arena((0, 5000, 300), cars=[c, o])
        m = manager(goal_reward=0.0, touch_weight=0.0)
        m.reset(a)
        phi = 5.0 * ball_position_potential(c, a) + boost_potential(c, a) + align_potential(c, a)
        r, _ = m.get_reward(c, a, None, True, 0)
        self.assertAlmostEqual(r, -phi, places=5)


class TestActionIndependence(unittest.TestCase):
    def test_reward_ignores_the_action(self):
        """R2: the same transition pays the same, whatever the controller did."""
        from env.rocket_env import RocketLeagueEnv
        env = RocketLeagueEnv(game_mode="1v1", max_episode_steps=10 ** 6, reward_version="v6")
        env.reset()
        self.assertIsInstance(env.reward_manager, RewardManagerV6)
        rng = np.random.default_rng(0)
        m1, m2 = RewardManagerV6(), RewardManagerV6()
        m1.reset(env.arena)
        m2.reset(env.arena)
        for _ in range(200):
            env.step(rng.uniform(-1, 1, (2, env.act_dim)).astype(np.float32))
            for c in env.arena.cars:
                r1 = m1.get_reward(c, env.arena, rng.uniform(-1, 1, 8), False, None)
                r2 = m2.get_reward(c, env.arena, np.zeros(8), False, None)
                self.assertEqual(r1, r2)

    def test_source_never_reads_the_action(self):
        import inspect
        import env.rewards_v6 as v6
        src = inspect.getsource(v6.RewardV6.get_reward)
        self.assertEqual(src.count("action"), 2, "get_reward may only accept and discard the action")


class TestRegistryAndVersionChange(unittest.TestCase):
    def test_registry_builds_v6_with_its_frozen_weights_and_gamma(self):
        from env.reward_registry import load_snapshot, make_reward_manager, reward_defaults
        m = make_reward_manager("v6")
        self.assertIsInstance(m, RewardManagerV6)
        self.assertAlmostEqual(m.reward.w["gamma"], GAMMA, places=9)
        self.assertEqual(reward_defaults("v6"), load_snapshot("v6")["settings"]["rewards"])

    def test_snapshot_starts_from_the_v5_king(self):
        from env.reward_registry import load_snapshot
        snap = load_snapshot("v6")
        self.assertEqual(snap["start_checkpoint"], "checkpoints/baselines/v5_iter222000.pt")
        self.assertEqual(snap["baseline_eval"], "evals/baselines/v5_iter222000.json")
        v5 = load_snapshot("v5")["settings"]
        self.assertEqual(snap["settings"]["scenarios"], v5["scenarios"])
        self.assertEqual(snap["settings"]["gamma"], v5["gamma"])
        self.assertEqual(snap["settings"]["rewards"], {**v5["rewards"], "align_weight": 1.0})
        self.assertFalse(snap["settings"]["reward_annealing"]["enabled"])

    def test_v5_checkpoint_starts_a_new_v6_run(self):
        import torch
        from agent.ppo import PPOTrainer, RunningMeanStd
        s = SimpleNamespace(
            reward_version="v6", reward_version_pinned=True, global_step=9_000, reward_run_start_step=0,
            ret_rms=RunningMeanStd(), _needs_value_norm_migration=True, critic_warmup_iterations=50,
            _critic_warmup_remaining=0, agent=torch.nn.Linear(2, 2), lr=1.5e-4,
            _reward_anneal_start_steps={}, _last_pushed_reward_weights={"x": 1.0})
        s.ret_rms.mean, s.ret_rms.var = 3.0, 9.0
        s.optimizer = old_opt = torch.optim.AdamW(s.agent.parameters())
        PPOTrainer._check_reward_version.__get__(s)(
            {"reward_identity": {"version": "v5"}, "reward_run_start_step": 1_234, "global_step": 9_000})
        self.assertEqual(s.reward_run_start_step, 9_000)
        self.assertEqual(s._critic_warmup_remaining, 50)
        self.assertEqual(float(s.ret_rms.mean), 0.0)
        self.assertIsNot(s.optimizer, old_opt)


if __name__ == "__main__":
    unittest.main()
