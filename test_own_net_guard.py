"""
Regression tests for the own-net guard in PlayerToBallVelocityReward (the "ball rolls across
the defensive half into the corner, SensAI cuts along its goal line into its own net and pins
against the side netting" clip).

Guarantees:
  1. Closure toward a corner target pays nothing once the car is inside its own net, while the
     same motion out on the field still pays.
  2. Driving hard at the goal line near it toward a far-away target is tapered.
  3. A save (target in the goal mouth) keeps its closure pay, even from inside the net.
  4. Sitting pinned inside the net costs a small amount every step.
  5. The exit shaping telescopes: entering and leaving by the same path nets zero, leaving pays.
  6. Everything is smooth across the goal line and the posts.
"""

import unittest
import numpy as np

from env.physics_engine import ARENA_EXTENT_Y, GOAL_HALF_WIDTH
from env.rewards import PlayerToBallVelocityReward, own_net_depth_factor
from test_touch_possession_and_goal_placement import shared_arena, max_step, DT

ACT = np.zeros(8, dtype=np.float32)
GOAL_LINE = -ARENA_EXTENT_Y  # blue defends -Y


class NetHarness:
    def __init__(self, target):
        self.arena = shared_arena()
        self.arena.last_step_dt = DT
        self.car, opp = self.arena.cars[0], self.arena.cars[1]
        self.car.team, opp.team = 0, 1
        opp.pos = np.array([3000.0, 4000.0, 17.0], dtype=np.float32)
        opp.vel = np.zeros(3, dtype=np.float32)
        self.car.ball_touches = opp.ball_touches = 0
        self.arena.ball.pos = np.array(target, dtype=np.float32)
        self.arena.ball.vel = np.zeros(3, dtype=np.float32)
        self.car.on_ground = True
        self.car.rot_mat = np.eye(3, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.rew = PlayerToBallVelocityReward(weight=1.0)
        self.rew._get_target_pos = lambda *a, **k: self.target

    def run(self, path, vel):
        """Places the car at each point in turn (after a reset at the first) and returns the rewards."""
        self.car.vel = np.array(vel, dtype=np.float32)
        self.car.pos = np.array(path[0], dtype=np.float32)
        self.rew.reset(self.arena)
        out = []
        for p in path[1:]:
            self.car.pos = np.array(p, dtype=np.float32)
            out.append(self.rew.get_reward(self.car, self.arena, ACT, False, None))
        return out


def diagonal_path(start, end, n):
    return [np.array(start) + (np.array(end) - np.array(start)) * t for t in np.linspace(0.0, 1.0, n)]


class TestOwnNetGuard(unittest.TestCase):
    CORNER_TARGET = [-3900.0, -3800.0, 93.0]

    def test_depth_factor_shape(self):
        self.assertEqual(own_net_depth_factor(np.array([0.0, GOAL_LINE + 10.0, 17.0]), 0, 150.0), 0.0)
        self.assertAlmostEqual(own_net_depth_factor(np.array([0.0, GOAL_LINE - 400.0, 17.0]), 0, 150.0), 1.0)
        self.assertEqual(own_net_depth_factor(np.array([2000.0, GOAL_LINE - 400.0, 17.0]), 0, 150.0), 0.0)
        # Orange mirror
        self.assertAlmostEqual(own_net_depth_factor(np.array([0.0, -GOAL_LINE + 400.0, 17.0]), 1, 150.0), 1.0)

    def test_closure_inside_net_pays_nothing(self):
        # Moving -x toward the corner target from deep inside the net vs the same move out on the field
        h = NetHarness(self.CORNER_TARGET)
        in_net = h.run(diagonal_path([200.0, GOAL_LINE - 400.0, 17.0], [-300.0, GOAL_LINE - 400.0, 17.0], 6),
                       [-1500.0, 0.0, 0.0])
        field = h.run(diagonal_path([200.0, -2500.0, 17.0], [-300.0, -2500.0, 17.0], 6), [-1500.0, 0.0, 0.0])
        self.assertGreater(sum(field), 0.02)
        self.assertLess(sum(in_net), 0.0, f"in_net={in_net}")

    def test_diving_at_goal_line_is_tapered(self):
        h = NetHarness(self.CORNER_TARGET)
        near = h.run(diagonal_path([600.0, GOAL_LINE + 500.0, 17.0], [300.0, GOAL_LINE + 200.0, 17.0], 6),
                     [-1000.0, -1000.0, 0.0])
        far = h.run(diagonal_path([600.0, -2500.0, 17.0], [300.0, -2800.0, 17.0], 6), [-1000.0, -1000.0, 0.0])
        self.assertGreater(sum(far), 0.0)
        self.assertLess(sum(near), 0.5 * sum(far), f"near={sum(near):.4f} far={sum(far):.4f}")

    def test_save_keeps_closure_pay(self):
        # Ball on the goal line in the mouth: driving out of the net toward it keeps full closure
        # pay. The same 100 uu of closure toward a corner target from the same spot is faded.
        path = diagonal_path([0.0, GOAL_LINE - 200.0, 17.0], [0.0, GOAL_LINE - 100.0, 17.0], 4)
        save = NetHarness([0.0, GOAL_LINE + 50.0, 93.0]).run(path, [0.0, 1500.0, 0.0])
        closure = 100.0 / 2000.0
        self.assertGreater(sum(save), 0.8 * closure, save)

    def test_pinned_in_net_costs_each_step(self):
        h = NetHarness(self.CORNER_TARGET)
        stuck = h.run([[-850.0, GOAL_LINE - 500.0, 17.0]] * 11, [0.0, 0.0, 0.0])
        self.assertTrue(all(v < 0.0 for v in stuck), stuck)
        self.assertAlmostEqual(stuck[-1], -0.03, places=4)

    def test_exit_telescopes_and_leaving_pays(self):
        # Straight in and straight back out along the same path: the exit potential returns to its
        # start, so in + out shaping sums to zero and only the depth penalty remains.
        h = NetHarness([0.0, -1500.0, 93.0])  # off centre: (0, 0) is treated as a kickoff
        depths = np.linspace(100.0, -600.0, 15)
        path_in = [[0.0, GOAL_LINE + d, 17.0] for d in depths]
        path = path_in + path_in[-2::-1]
        h.car.vel = np.zeros(3, dtype=np.float32)
        h.car.pos = np.array(path[0], dtype=np.float32)
        h.rew.reset(h.arena)
        potentials = []
        rewards = []
        for p in path[1:]:
            h.car.pos = np.array(p, dtype=np.float32)
            rewards.append(h.rew.get_reward(h.car, h.arena, ACT, False, None))
            potentials.append(h.rew._prev_net_potential[h.car.id])
        self.assertAlmostEqual(potentials[-1], 0.0, places=6)
        self.assertLess(min(potentials), -0.39)
        # Leaving pays more than entering, net of closure and penalties
        half = len(path_in) - 1
        self.assertGreater(sum(rewards[half:]), sum(rewards[:half]))

    def test_smooth_across_line_and_post(self):
        h = NetHarness(self.CORNER_TARGET)
        ys = np.linspace(GOAL_LINE + 800.0, GOAL_LINE - 700.0, 301)
        vals = h.run([[0.0, y, 17.0] for y in ys], [-800.0, -800.0, 0.0])
        self.assertLess(max_step(vals), 0.01, max_step(vals))
        xs = np.linspace(0.0, GOAL_HALF_WIDTH + 400.0, 1301)  # 1 uu steps; the post ramp is 150 uu wide
        vals = []
        for x in xs:
            vals.append(h.run([[x, GOAL_LINE - 300.0, 17.0], [x - 50.0, GOAL_LINE - 300.0, 17.0]],
                              [-750.0, 0.0, 0.0])[0])
        self.assertLess(max_step(vals), 0.01, max_step(vals))


if __name__ == "__main__":
    unittest.main()
