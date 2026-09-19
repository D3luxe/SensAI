"""
Reward v3 (env/rewards_v3.py) against its specification, docs/reward_v3_spec.md.

  - each term's formula, as written in the spec
  - potentials telescope: any closed loop of states sums to zero (gamma = 1), and with the run's
    gamma a loop sums to exactly (gamma - 1) * sum(phi), the policy-invariant remainder
  - phi(terminal) = 0 on a goal step
  - the reward never depends on the action (R2)
  - the two v3 training starts, and that v2's scenario sampling is unchanged by them
  - the trainer's handling of a reward version change
"""
import math
import random
import unittest
from types import SimpleNamespace

import numpy as np

from env.rewards_v3 import (
    REWARD_V3_DEFAULTS, RewardManagerV3, ball_position_potential, boost_potential, closeness_potential,
)


def car(cid, team, pos, boost=33.0, touches=0):
    return SimpleNamespace(id=cid, team=team, pos=np.array(pos, dtype=np.float32), boost=boost, ball_touches=touches)


def arena(ball_pos, ball_vel=(0.0, 0.0, 0.0), cars=()):
    return SimpleNamespace(ball=SimpleNamespace(pos=np.array(ball_pos, dtype=np.float32),
                                                vel=np.array(ball_vel, dtype=np.float32)), cars=list(cars))


def manager(**w):
    return RewardManagerV3({**REWARD_V3_DEFAULTS, **w})


class TestTerms(unittest.TestCase):
    def test_goal_uses_aggression_bias(self):
        blue, orange = car(0, 0, (0, -1000, 17)), car(1, 1, (0, 1000, 17))
        a = arena((0, 5200, 300), cars=[blue, orange])
        m = manager(ball_position_weight=0.0, closeness_weight=0.0, boost_weight=0.0, touch_weight=0.0)
        m.reset(a)
        self.assertAlmostEqual(m.get_reward(blue, a, None, True, 0)[0], 10.0)
        self.assertAlmostEqual(m.get_reward(orange, a, None, True, 0)[0], -7.5)
        m = manager(ball_position_weight=0.0, closeness_weight=0.0, boost_weight=0.0, touch_weight=0.0,
                    goal_reward=4.0, aggression_bias=0.5)
        m.reset(a)
        self.assertAlmostEqual(m.get_reward(orange, a, None, True, 0)[0], -2.0)

    def test_ball_position_potential_formula_and_zero_sum(self):
        for bp in [(0, 0, 93), (1500, -3000, 400), (-3000, 4500, 1200)]:
            a = arena(bp)
            own, opp = np.array([0, -5120, 642.775 / 2]), np.array([0, 5120, 642.775 / 2])
            expected = (np.linalg.norm(np.array(bp) - own) - np.linalg.norm(np.array(bp) - opp)) / 10240
            self.assertAlmostEqual(ball_position_potential(car(0, 0, (0, 0, 17)), a), expected, places=6)
            self.assertAlmostEqual(ball_position_potential(car(0, 0, (0, 0, 17)), a),
                                   -ball_position_potential(car(1, 1, (0, 0, 17)), a), places=9)
        self.assertAlmostEqual(ball_position_potential(car(0, 0, (0, 0, 17)), arena((0, 5120, 321.3875))), 1.0, places=6)
        self.assertAlmostEqual(ball_position_potential(car(0, 0, (0, 0, 17)), arena((0, -5120, 321.3875))), -1.0, places=6)

    def test_closeness_and_boost_potentials(self):
        a = arena((300, 400, 1200))
        self.assertAlmostEqual(closeness_potential(car(0, 0, (0, 0, 0)), a), -1300.0 / 12000.0, places=6)
        self.assertAlmostEqual(boost_potential(car(0, 0, (0, 0, 17), boost=25.0), a), 0.5)
        self.assertAlmostEqual(boost_potential(car(0, 0, (0, 0, 17), boost=100.0), a), 1.0)
        self.assertAlmostEqual(boost_potential(car(0, 0, (0, 0, 17), boost=0.0), a), 0.0)

    def test_potential_step_pays_weighted_gamma_difference(self):
        c = car(0, 0, (0, 0, 17), boost=25.0)
        a = arena((0, 1000, 93), cars=[c])
        m = manager(goal_reward=0.0, touch_weight=0.0, ball_position_weight=0.0, closeness_weight=0.0,
                    boost_weight=2.0, gamma=0.9)
        m.reset(a)
        c.boost = 100.0
        r, b = m.get_reward(c, a, None, False, None)
        self.assertAlmostEqual(r, 2.0 * (0.9 * 1.0 - 0.5), places=6)
        self.assertAlmostEqual(b["boost"], r, places=6)

    def test_touch_pays_ball_speed_change_capped(self):
        c = car(0, 0, (0, 0, 17))
        a = arena((0, 200, 93), (0, 0, 0), cars=[c])
        m = manager(goal_reward=0.0, ball_position_weight=0.0, closeness_weight=0.0, boost_weight=0.0)
        m.reset(a)
        a.ball.vel = np.array([0, 1150, 0], dtype=np.float32)
        self.assertAlmostEqual(m.get_reward(c, a, None, False, None)[0], 0.0)   # no touch: nothing
        c.ball_touches = 1
        a.ball.vel = np.array([0, 2300, 0], dtype=np.float32)                  # dv 1150 on the touch step
        self.assertAlmostEqual(m.get_reward(c, a, None, False, None)[0], 0.5 * 1150 / 2300, places=5)
        c.ball_touches = 2
        a.ball.vel = np.array([0, -3000, 0], dtype=np.float32)                 # dv 5300: capped at 1
        self.assertAlmostEqual(m.get_reward(c, a, None, False, None)[0], 0.5, places=6)

    def test_breakdown_names_the_five_terms_and_sums_to_total(self):
        c = car(0, 0, (0, 0, 17))
        a = arena((0, 900, 93), cars=[c])
        m = manager()
        m.reset(a)
        a.ball.pos = np.array([100, 1400, 200], dtype=np.float32)
        r, b = m.get_reward(c, a, None, False, None)
        self.assertEqual(list(b), ["goal", "ball_position", "touch", "closeness", "boost"])
        self.assertAlmostEqual(sum(b.values()), r, places=6)


class TestPotentials(unittest.TestCase):
    def _loop_sum(self, gamma):
        rng = np.random.default_rng(3)
        states = [(rng.uniform(-3000, 3000, 3), rng.uniform(-3000, 3000, 3), rng.uniform(0, 100)) for _ in range(12)]
        states.append(states[0])   # closed loop
        c = car(0, 0, states[0][1], boost=states[0][2])
        a = arena(states[0][0], cars=[c])
        m = manager(goal_reward=0.0, touch_weight=0.0, gamma=gamma)
        m.reset(a)
        total, phis = 0.0, []
        for ball, pos, boost in states[1:]:
            a.ball.pos, c.pos, c.boost = np.array(ball, np.float32), np.array(pos, np.float32), boost
            phis.append(5.0 * ball_position_potential(c, a) + closeness_potential(c, a) + boost_potential(c, a))
            total += m.get_reward(c, a, None, False, None)[0]
        return total, phis

    def test_closed_loop_sums_to_zero_undiscounted(self):
        total, _ = self._loop_sum(1.0)
        self.assertAlmostEqual(total, 0.0, places=5)

    def test_closed_loop_leaves_only_the_discount_remainder(self):
        total, phis = self._loop_sum(0.995)
        self.assertAlmostEqual(total, (0.995 - 1.0) * sum(phis), places=5)

    def test_potential_is_zero_at_a_goal(self):
        c = car(0, 0, (500, -2000, 17), boost=64.0)
        a = arena((0, 5000, 300), cars=[c])
        m = manager(goal_reward=0.0, touch_weight=0.0)
        m.reset(a)
        phi = 5.0 * ball_position_potential(c, a) + closeness_potential(c, a) + boost_potential(c, a)
        r, _ = m.get_reward(c, a, None, True, 0)
        self.assertAlmostEqual(r, -phi, places=5)

    def test_reset_restarts_the_potentials(self):
        c = car(0, 0, (0, 0, 17), boost=0.0)
        a = arena((0, 0, 93), cars=[c])
        m = manager(goal_reward=0.0, touch_weight=0.0, ball_position_weight=0.0, closeness_weight=0.0)
        m.reset(a)
        c.boost = 100.0
        m.reset(a)   # a new episode begins holding 100: no reward for the jump across the reset
        self.assertAlmostEqual(m.get_reward(c, a, None, False, None)[0], (0.995 - 1.0) * 1.0, places=6)


class TestActionIndependence(unittest.TestCase):
    def test_reward_ignores_the_action(self):
        """R2: the same transition pays the same, whatever the controller did."""
        from env.rocket_env import RocketLeagueEnv
        env = RocketLeagueEnv(game_mode="1v1", max_episode_steps=10 ** 6, reward_version="v3")
        env.reset()
        rng = np.random.default_rng(0)
        m1, m2 = RewardManagerV3(), RewardManagerV3()
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
        import env.rewards_v3 as v3
        src = inspect.getsource(v3.RewardV3.get_reward)
        self.assertEqual(src.count("action"), 2, "get_reward may only accept and discard the action")


class TestScenarios(unittest.TestCase):
    KEYS = ("kickoff_prob", "replay_prob", "aerial_prob", "wall_prob", "save_prob", "turnaround_prob",
            "wall_rebound_prob", "dribble_flick_prob", "custom_prob", "bounce_drop_prob", "retreat_prob")

    def _arena(self, only):
        from env.physics_engine import RocketSimArena
        a = RocketSimArena(num_players=2, game_mode="1v1")
        a.set_scenario_weights({**{k: 0.0 for k in self.KEYS}, only: 1.0})
        return a

    def test_retreat_puts_a_car_upfield_of_a_ball_heading_for_its_net(self):
        a = self._arena("retreat_prob")
        random.seed(5)
        for _ in range(100):
            a.reset(random_kickoff=True)
            self.assertEqual(a.current_scenario, "retreat")
            b, v = a.ball.pos, a.ball.vel
            self.assertTrue(any((c.pos[1] - b[1]) * (1 if c.team == 0 else -1) > 900
                                and v[1] * (1 if c.team == 0 else -1) < 0 for c in a.cars))
            self.assertTrue(all(abs(c.pos[0]) < 4096 and abs(c.pos[1]) < 5120 for c in a.cars))

    def test_bounce_drop_puts_the_ball_in_the_field_near_a_car(self):
        a = self._arena("bounce_drop_prob")
        random.seed(6)
        teams = set()
        for _ in range(100):
            a.reset(random_kickoff=True)
            self.assertEqual(a.current_scenario, "bounce_drop")
            b = a.ball.pos
            self.assertLess(abs(b[0]), 4096)
            self.assertLess(abs(b[1]), 5120 - 400)
            near = min(np.linalg.norm(b[:2] - c.pos[:2]) for c in a.cars)
            self.assertLess(near, 3800)
            teams.add(min(a.cars, key=lambda c: np.linalg.norm(b[:2] - c.pos[:2])).team)
        self.assertEqual(teams, {0, 1})

    def test_v2_sampling_never_reaches_the_new_starts(self):
        from env.reward_registry import load_snapshot
        a = self._arena("kickoff_prob")
        a.set_scenario_weights(load_snapshot("v2")["settings"]["scenarios"])
        random.seed(7)
        seen = set()
        for _ in range(300):
            a.reset(random_kickoff=True)
            seen.add(a.current_scenario)
        self.assertFalse(seen & {"bounce_drop", "retreat"})


class TestVersionChange(unittest.TestCase):
    def _stub(self, version, pinned=True):
        import torch
        from agent.ppo import PPOTrainer, RunningMeanStd
        stub = SimpleNamespace(
            reward_version=version, reward_version_pinned=pinned, global_step=5_000, reward_run_start_step=0,
            ret_rms=RunningMeanStd(), _needs_value_norm_migration=True, critic_warmup_iterations=50,
            _critic_warmup_remaining=0, agent=torch.nn.Linear(2, 2), lr=3e-4,
            _reward_anneal_start_steps={"player_to_ball_weight": 10}, _last_pushed_reward_weights={"x": 1.0})
        stub.ret_rms.mean, stub.ret_rms.var = 3.0, 9.0
        stub.optimizer = torch.optim.AdamW(stub.agent.parameters())
        stub._check_reward_version = PPOTrainer._check_reward_version.__get__(stub)
        return stub

    def test_new_version_resets_what_was_fitted_to_the_old_reward(self):
        s = self._stub("v3")
        old_opt = s.optimizer
        s._check_reward_version({"reward_identity": {"version": "v2"}, "global_step": 5_000})
        self.assertEqual(s.reward_run_start_step, 5_000)
        self.assertEqual(s._critic_warmup_remaining, 50)
        self.assertEqual(float(s.ret_rms.mean), 0.0)
        self.assertIsNot(s.optimizer, old_opt)
        self.assertEqual(s._reward_anneal_start_steps, {})
        self.assertFalse(s._needs_value_norm_migration)

    def test_unstamped_checkpoint_counts_as_v2(self):
        s = self._stub("v3")
        s._check_reward_version({})
        self.assertEqual(s._critic_warmup_remaining, 50)

    def test_same_version_resumes_the_run(self):
        s = self._stub("v3")
        old_opt = s.optimizer
        s._check_reward_version({"reward_identity": {"version": "v3"}, "reward_run_start_step": 1_234})
        self.assertEqual(s.reward_run_start_step, 1_234)
        self.assertIs(s.optimizer, old_opt)
        self.assertEqual(s._critic_warmup_remaining, 0)
        self.assertEqual(s._reward_anneal_start_steps, {"player_to_ball_weight": 10})


if __name__ == "__main__":
    unittest.main()
