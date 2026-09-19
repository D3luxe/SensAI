"""
Reward v4 (env/rewards_v4.py) against its specification, docs/reward_v4_spec.md.

  - T6 ball race: its formula, zero-sum in 1v1, the nearest opponent, no opponent means no race
  - T1-T5 are v3's terms unchanged: with the race off, v4 pays exactly what v3 pays
  - potentials telescope with the race included, and phi(terminal) = 0 on a goal step
  - the reward never depends on the action (R2)
  - the registry builds v4, and the trainer treats v3 -> v4 as a version change
"""
import unittest
from types import SimpleNamespace

import numpy as np

from env.rewards_v3 import RewardManagerV3, ball_position_potential, boost_potential
from env.rewards_v4 import REWARD_V4_DEFAULTS, RewardManagerV4, TERM_NAMES, race_potential


def car(cid, team, pos, boost=33.0, touches=0):
    return SimpleNamespace(id=cid, team=team, pos=np.array(pos, dtype=np.float32), boost=boost, ball_touches=touches)


def arena(ball_pos, ball_vel=(0.0, 0.0, 0.0), cars=()):
    return SimpleNamespace(ball=SimpleNamespace(pos=np.array(ball_pos, dtype=np.float32),
                                                vel=np.array(ball_vel, dtype=np.float32)), cars=list(cars))


def manager(**w):
    return RewardManagerV4({**REWARD_V4_DEFAULTS, **w})


class TestRace(unittest.TestCase):
    def test_formula(self):
        blue, orange = car(0, 0, (0, -1000, 17)), car(1, 1, (3000, 2000, 17))
        a = arena((0, 0, 93), cars=[blue, orange])
        d_blue = np.linalg.norm(np.array([0, 0, 93]) - np.array([0, -1000, 17]))
        d_orange = np.linalg.norm(np.array([0, 0, 93]) - np.array([3000, 2000, 17]))
        self.assertAlmostEqual(race_potential(blue, a), (d_orange - d_blue) / 12000, places=6)
        self.assertGreater(race_potential(blue, a), 0.0)

    def test_zero_sum_in_1v1(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            blue, orange = car(0, 0, rng.uniform(-4000, 4000, 3)), car(1, 1, rng.uniform(-4000, 4000, 3))
            a = arena(rng.uniform(-4000, 4000, 3), cars=[blue, orange])
            self.assertAlmostEqual(race_potential(blue, a), -race_potential(orange, a), places=9)

    def test_level_race_is_zero(self):
        blue, orange = car(0, 0, (-2048, -2560, 17)), car(1, 1, (2048, 2560, 17))
        self.assertAlmostEqual(race_potential(blue, arena((0, 0, 93), cars=[blue, orange])), 0.0, places=6)

    def test_nearest_opponent_counts_and_teammates_do_not(self):
        me, mate = car(0, 0, (0, -2000, 17)), car(2, 0, (0, -10, 17))
        far, near = car(1, 1, (0, 4000, 17)), car(3, 1, (0, 1000, 17))
        a = arena((0, 0, 17), cars=[me, far, mate, near])
        self.assertAlmostEqual(race_potential(me, a), (1000 - 2000) / 12000, places=6)

    def test_no_opponent_no_race(self):
        c = car(0, 0, (0, -2000, 17))
        self.assertEqual(race_potential(c, arena((0, 0, 93), cars=[c])), 0.0)

    def test_pays_weighted_gamma_difference(self):
        blue, orange = car(0, 0, (0, -3000, 17)), car(1, 1, (0, 3000, 17))
        a = arena((0, 0, 93), cars=[blue, orange])
        m = manager(goal_reward=0.0, touch_weight=0.0, ball_position_weight=0.0, boost_weight=0.0, race_weight=2.0)
        m.reset(a)
        old = race_potential(blue, a)
        blue.pos = np.array([0, -1000, 17], np.float32)
        r, b = m.get_reward(blue, a, None, False, None)
        self.assertAlmostEqual(r, 2.0 * (0.995 * race_potential(blue, a) - old), places=6)
        self.assertGreater(b["race"], 0.0)


class TestV3TermsUnchanged(unittest.TestCase):
    def test_race_off_pays_exactly_v3(self):
        """With T6 at zero and T4 at v3's weight, v4 is v3, term for term, over a random trajectory."""
        rng = np.random.default_rng(11)
        blue, orange = car(0, 0, (0, -2000, 17)), car(1, 1, (0, 2000, 17))
        a = arena((0, 0, 93), cars=[blue, orange])
        v3 = RewardManagerV3({"closeness_weight": 1.0})
        v4 = manager(closeness_weight=1.0, race_weight=0.0)
        v3.reset(a)
        v4.reset(a)
        for step in range(60):
            a.ball.pos = rng.uniform(-4000, 4000, 3).astype(np.float32)
            a.ball.vel = rng.uniform(-2000, 2000, 3).astype(np.float32)
            for c in (blue, orange):
                c.pos = rng.uniform(-4000, 4000, 3).astype(np.float32)
                c.boost = float(rng.uniform(0, 100))
                c.ball_touches += int(rng.random() < 0.2)
            goal = step % 25 == 24
            for c in (blue, orange):
                r3, b3 = v3.get_reward(c, a, None, goal, 0 if goal else None)
                r4, b4 = v4.get_reward(c, a, None, goal, 0 if goal else None)
                self.assertAlmostEqual(r3, r4, places=9)
                for k, v in b3.items():
                    self.assertAlmostEqual(v, b4[k], places=9, msg=k)
                self.assertEqual(b4["race"], 0.0)

    def test_breakdown_names_the_six_terms_and_sums_to_total(self):
        blue, orange = car(0, 0, (0, -2000, 17)), car(1, 1, (0, 2000, 17))
        a = arena((0, 0, 93), cars=[blue, orange])
        m = manager()
        m.reset(a)
        blue.pos = np.array([0, -500, 17], np.float32)
        r, b = m.get_reward(blue, a, None, False, None)
        self.assertEqual(list(b), list(TERM_NAMES))
        self.assertEqual(TERM_NAMES, ("goal", "ball_position", "touch", "closeness", "boost", "race"))
        self.assertAlmostEqual(sum(b.values()), r, places=6)

    def test_closeness_is_retired(self):
        self.assertEqual(REWARD_V4_DEFAULTS["closeness_weight"], 0.0)
        blue, orange = car(0, 0, (0, -3000, 17)), car(1, 1, (0, 3000, 17))
        a = arena((0, 0, 93), cars=[blue, orange])
        m = manager()
        m.reset(a)
        blue.pos = np.array([0, -100, 17], np.float32)
        self.assertEqual(m.get_reward(blue, a, None, False, None)[1]["closeness"], 0.0)


class TestPotentials(unittest.TestCase):
    def _loop_sum(self, gamma):
        rng = np.random.default_rng(3)
        states = [(rng.uniform(-3000, 3000, 3), rng.uniform(-3000, 3000, 3), rng.uniform(-3000, 3000, 3),
                   rng.uniform(0, 100)) for _ in range(12)]
        states.append(states[0])   # closed loop
        c, o = car(0, 0, states[0][1], boost=states[0][3]), car(1, 1, states[0][2])
        a = arena(states[0][0], cars=[c, o])
        m = manager(goal_reward=0.0, touch_weight=0.0, gamma=gamma)
        m.reset(a)
        total, phis = 0.0, []
        for ball, pos, opp, boost in states[1:]:
            a.ball.pos, c.pos, o.pos, c.boost = (np.array(ball, np.float32), np.array(pos, np.float32),
                                                 np.array(opp, np.float32), boost)
            phis.append(5.0 * ball_position_potential(c, a) + boost_potential(c, a) + race_potential(c, a))
            total += m.get_reward(c, a, None, False, None)[0]
        return total, phis

    def test_closed_loop_sums_to_zero_undiscounted(self):
        total, _ = self._loop_sum(1.0)
        self.assertAlmostEqual(total, 0.0, places=5)

    def test_closed_loop_leaves_only_the_discount_remainder(self):
        total, phis = self._loop_sum(0.995)
        self.assertAlmostEqual(total, (0.995 - 1.0) * sum(phis), places=5)

    def test_potential_is_zero_at_a_goal(self):
        c, o = car(0, 0, (500, -2000, 17), boost=64.0), car(1, 1, (0, 4000, 17))
        a = arena((0, 5000, 300), cars=[c, o])
        m = manager(goal_reward=0.0, touch_weight=0.0)
        m.reset(a)
        phi = 5.0 * ball_position_potential(c, a) + boost_potential(c, a) + race_potential(c, a)
        r, _ = m.get_reward(c, a, None, True, 0)
        self.assertAlmostEqual(r, -phi, places=5)


class TestActionIndependence(unittest.TestCase):
    def test_reward_ignores_the_action(self):
        """R2: the same transition pays the same, whatever the controller did."""
        from env.rocket_env import RocketLeagueEnv
        env = RocketLeagueEnv(game_mode="1v1", max_episode_steps=10 ** 6, reward_version="v4")
        env.reset()
        self.assertIsInstance(env.reward_manager, RewardManagerV4)
        rng = np.random.default_rng(0)
        m1, m2 = RewardManagerV4(), RewardManagerV4()
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
        import env.rewards_v4 as v4
        src = inspect.getsource(v4.RewardV4.get_reward)
        self.assertEqual(src.count("action"), 2, "get_reward may only accept and discard the action")


class TestRegistryAndVersionChange(unittest.TestCase):
    def test_registry_builds_v4_with_its_frozen_weights(self):
        from env.reward_registry import load_snapshot, make_reward_manager, reward_defaults
        self.assertIsInstance(make_reward_manager("v4"), RewardManagerV4)
        self.assertEqual(reward_defaults("v4"), load_snapshot("v4")["settings"]["rewards"])

    def test_v3_checkpoint_starts_a_new_v4_run(self):
        import torch
        from agent.ppo import PPOTrainer, RunningMeanStd
        s = SimpleNamespace(
            reward_version="v4", reward_version_pinned=True, global_step=9_000, reward_run_start_step=0,
            ret_rms=RunningMeanStd(), _needs_value_norm_migration=True, critic_warmup_iterations=50,
            _critic_warmup_remaining=0, agent=torch.nn.Linear(2, 2), lr=3e-4,
            _reward_anneal_start_steps={"closeness_weight": 10}, _last_pushed_reward_weights={"x": 1.0})
        s.ret_rms.mean, s.ret_rms.var = 3.0, 9.0
        s.optimizer = old_opt = torch.optim.AdamW(s.agent.parameters())
        PPOTrainer._check_reward_version.__get__(s)(
            {"reward_identity": {"version": "v3"}, "reward_run_start_step": 1_234, "global_step": 9_000})
        self.assertEqual(s.reward_run_start_step, 9_000)
        self.assertEqual(s._critic_warmup_remaining, 50)
        self.assertEqual(float(s.ret_rms.mean), 0.0)
        self.assertIsNot(s.optimizer, old_opt)
        self.assertEqual(s._reward_anneal_start_steps, {})


if __name__ == "__main__":
    unittest.main()
