"""
Regression tests for the slow-sideways-dribble farm and touch/goal reward smoothness.

Guarantees:
  1. A 10-bounce roof carry earns less than one clean catch plus an on-target shot.
  2. A sideways carry earns ~nothing after its first touch.
  3. Fresh catches and strikes still pay; a strike inside a carry keeps its impulse payout.
  4. TouchBallReward changes smoothly with relative speed, ball height, goal alignment and
     time since the previous touch (no step between near-identical touches).
  5. GoalReward placement fades defenders in/out smoothly around their goal line (no 2000 uu cliff),
     and an open net still scores full placement.
"""

import math
import unittest
import numpy as np

from env.physics_engine import RocketSimArena, ARENA_EXTENT_Y
from env.rewards import TouchBallReward, GoalReward

DT = 8.0 / 120.0
ACT = np.zeros(8, dtype=np.float32)

_ARENA = None


def shared_arena():
    """One RocketSim arena for every sample; building one per sample makes the sweeps crawl."""
    global _ARENA
    if _ARENA is None:
        _ARENA = RocketSimArena(num_players=2)
        _ARENA.reset(random_kickoff=False)
    return _ARENA


def max_step(values):
    return max(abs(b - a) for a, b in zip(values, values[1:]))


class TouchHarness:
    """Drives TouchBallReward step by step on a real arena with an opponent stuck in our corner."""

    def __init__(self, weight=1.0):
        self.arena = shared_arena()
        self.arena.last_step_dt = DT
        self.car = self.arena.cars[0]
        self.car.team = 0
        self.opp = self.arena.cars[1]
        self.opp.team = 1
        self.opp.pos = np.array([-3500.0, -4600.0, 17.0], dtype=np.float32)
        self.opp.vel = np.zeros(3, dtype=np.float32)
        self.car.on_ground = True
        self.car.ball_touches = 0
        self.opp.ball_touches = 0
        self.rew = TouchBallReward(weight=weight)
        self.rew.reset(self.arena)

    def set_state(self, car_pos, car_vel, ball_pos, ball_vel):
        self.car.pos = np.array(car_pos, dtype=np.float32)
        self.car.vel = np.array(car_vel, dtype=np.float32)
        self.car.rot_mat = np.eye(3, dtype=np.float32)  # facing +x, no lateral slip for +x travel
        self.arena.ball.pos = np.array(ball_pos, dtype=np.float32)
        self.arena.ball.vel = np.array(ball_vel, dtype=np.float32)

    def idle(self, steps):
        for _ in range(steps):
            self.rew.get_reward(self.car, self.arena, ACT, False, None)

    def touch(self, prev_ball_vel=None):
        if prev_ball_vel is not None:
            self.rew._prev_ball_vel[self.car.id] = np.array(prev_ball_vel, dtype=np.float32)
        self.car.ball_touches += 1
        return self.rew.get_reward(self.car, self.arena, ACT, False, None)


def single_touch(car_vel, ball_pos, ball_vel, prev_ball_vel, t_since, car_pos=(0.0, -800.0, 17.0)):
    h = TouchHarness()
    h.set_state(car_pos, car_vel, ball_pos, ball_vel)
    h.rew._time_since_touch[h.car.id] = t_since
    h.rew._prev_ball_vel[h.car.id] = np.array(prev_ball_vel, dtype=np.float32)
    h.car.ball_touches += 1
    return h.rew.get_reward(h.car, h.arena, ACT, False, None)


class TestDribbleFarm(unittest.TestCase):
    def _roof_carry(self, ball_dir):
        h = TouchHarness()
        v = 800.0 * np.array(ball_dir, dtype=np.float32)
        h.set_state([0.0, -800.0, 17.0], v, [0.0, -800.0, 175.0], v)
        rewards = []
        for _ in range(10):
            rewards.append(h.touch(prev_ball_vel=v))
            h.idle(6)  # a bounce every 7 steps (~0.47 s)
        return rewards

    def test_sideways_carry_pays_nothing_after_first_touch(self):
        rewards = self._roof_carry([1.0, 0.0, 0.0])
        self.assertGreater(rewards[0], 0.3, f"The initial catch still pays: {rewards}")
        self.assertLess(sum(rewards[1:]), 0.05 * rewards[0] * 9, f"Carry re-contacts must be ~free: {rewards}")

    def test_carry_earns_less_than_catch_plus_shot(self):
        carry_total = sum(self._roof_carry([1.0, 0.0, 0.0]))

        h = TouchHarness()
        v = np.array([800.0, 0.0, 0.0], dtype=np.float32)
        h.set_state([0.0, -800.0, 17.0], v, [0.0, -800.0, 175.0], v)
        catch = h.touch(prev_ball_vel=v)
        h.idle(6)
        shot_vel = np.array([0.0, 2600.0, 150.0], dtype=np.float32)
        h.set_state([0.0, -800.0, 17.0], [0.0, 1600.0, 0.0], [0.0, -700.0, 175.0], shot_vel)
        shot = h.touch(prev_ball_vel=v)
        self.assertGreater(shot, 1.0, "A strike inside a carry keeps its impulse payout")
        self.assertLess(carry_total, catch + shot, f"carry={carry_total:.3f} catch={catch:.3f} shot={shot:.3f}")

    def test_goalward_carry_still_earns_a_little(self):
        rewards = self._roof_carry([0.0, 1.0, 0.0])
        self.assertGreater(sum(rewards[1:]), 0.0, "Carrying toward goal keeps a small progress signal")
        self.assertLess(sum(rewards[1:]), 9 * rewards[0], "...but well below re-paying the catch each bounce")

    def test_fresh_catch_after_gap_pays_again(self):
        h = TouchHarness()
        v = np.array([800.0, 0.0, 0.0], dtype=np.float32)
        h.set_state([0.0, -800.0, 17.0], v, [0.0, -800.0, 175.0], v)
        first = h.touch(prev_ball_vel=v)
        h.idle(int(3.2 / DT))
        again = h.touch(prev_ball_vel=v)
        self.assertAlmostEqual(again, first, places=3)


class TestTouchSmoothness(unittest.TestCase):
    def test_relative_speed_and_height_sweep(self):
        # Covers the old gentle-push edges (rel 150 uu/s, ball z 130) and soft-catch edges (rel 350, z 200)
        by_rel = [single_touch([800.0, 0.0, 0.0], [0.0, -800.0, 100.0], [800.0 + dv, 300.0, 0.0],
                               [800.0, 0.0, 0.0], 5.0) for dv in np.linspace(0.0, 500.0, 101)]
        self.assertLess(max_step(by_rel), 0.08, by_rel)
        by_z = [single_touch([800.0, 0.0, 0.0], [0.0, -800.0, z], [900.0, 300.0, 0.0],
                             [800.0, 0.0, 0.0], 5.0) for z in np.linspace(95.0, 260.0, 111)]
        self.assertLess(max_step(by_z), 0.08, by_z)

    def test_goal_alignment_sweep_across_forward_backward_fork(self):
        vals = []
        for ang in np.linspace(-0.6, 0.6, 121):
            vel = [1200.0 * math.cos(ang), 1200.0 * math.sin(ang), 0.0]
            vals.append(single_touch([600.0, 0.0, 0.0], [0.0, -800.0, 93.0], vel, [300.0, 0.0, 0.0], 5.0))
        self.assertLess(max_step(vals), 0.15, vals)

    def test_time_since_touch_sweep(self):
        vals = [single_touch([800.0, 0.0, 0.0], [0.0, -800.0, 175.0], [800.0, 0.0, 0.0],
                             [800.0, 0.0, 0.0], t) for t in np.linspace(0.0, 4.0, 81)]
        self.assertLess(max_step(vals), 0.08, vals)
        self.assertAlmostEqual(vals[0], 0.0, places=3)


class TestGoalPlacementFade(unittest.TestCase):
    def _goal_value(self, defender_depth, defender_x=0.0):
        arena = shared_arena()
        arena.cars[0].team, arena.cars[1].team = 0, 1
        arena.cars[1].pos = np.array([defender_x, ARENA_EXTENT_Y - defender_depth, 17.0], dtype=np.float32)
        arena.last_goal_ball_pos = np.array([600.0, ARENA_EXTENT_Y, 100.0], dtype=np.float32)
        arena.last_goal_ball_vel = np.array([0.0, 1500.0, 0.0], dtype=np.float32)
        rew = GoalReward(goal_weight=5.0, concede_weight=-5.0, save_weight=2.0)
        rew.reset(arena)
        return rew.get_reward(arena.cars[0], arena, ACT, is_goal=True, scoring_team=0)

    def test_no_cliff_around_defender_range(self):
        vals = [self._goal_value(d) for d in np.linspace(1000.0, 3000.0, 201)]
        self.assertLess(max_step(vals), 0.05, vals)
        self.assertGreater(vals[-1], vals[0], "Defender far upfield leaves a more open net")

    def test_open_net_full_placement(self):
        # Speed 1500 -> s_speed = 0.40; open net -> placement 1.0
        self.assertAlmostEqual(self._goal_value(6000.0), 5.0 * (1.0 + 0.5 * 0.4 + 0.5), places=3)


if __name__ == "__main__":
    unittest.main()
