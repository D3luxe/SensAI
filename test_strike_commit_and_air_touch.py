"""
Regression tests for committed strikes and the airborne touch bonus (the "jumped at bouncing
balls and dribbled softly into the back wall with an open net" clip).

Guarantees:
  1. shot_line_factor is 1 lined up behind the ball toward goal, 0 when well off-line, and smooth.
  2. The strike-zone overspeed boost penalty is waived when lined up for a shot, still applies when
     not, and changes smoothly with closing speed.
  3. PlayerToBallVelocityReward pays a lined-up approach inside the strike zone more than an
     equally fast off-line approach (pacing relaxed for committed strikes).
  4. Airborne touch bonus: none for a ball near car height, full for a genuinely high ball, not
     re-paid for a low hop-and-tap chain, still paid on every contact of a high air dribble,
     and smooth across ball height.
"""

import math
import unittest
import numpy as np

from env.physics_engine import CarState, RocketSimArena
from env.rewards import BoostReward, PlayerToBallVelocityReward, shot_line_factor
from test_boost_trajectory_and_urgency import MockArena
from test_touch_possession_and_goal_placement import TouchHarness, max_step, DT, shared_arena

BOOST_ACT = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0], dtype=np.float32)


class TestShotLine(unittest.TestCase):
    def test_lined_up_and_off_line(self):
        ball = np.array([0.0, 3000.0, 93.0])
        self.assertAlmostEqual(shot_line_factor(np.array([0.0, 2000.0, 17.0]), ball, 0), 1.0)
        self.assertAlmostEqual(shot_line_factor(np.array([1000.0, 3000.0, 17.0]), ball, 0), 0.0)
        self.assertAlmostEqual(shot_line_factor(np.array([0.0, 4000.0, 17.0]), ball, 0), 0.0)
        # Orange attacks -Y
        self.assertAlmostEqual(shot_line_factor(np.array([0.0, -2000.0, 17.0]), -ball, 1), 1.0)

    def test_smooth_around_ball(self):
        ball = np.array([0.0, 2000.0, 93.0])
        vals = [shot_line_factor(np.array([800.0 * math.sin(a), 2000.0 - 800.0 * math.cos(a), 17.0]), ball, 0)
                for a in np.linspace(0.0, math.pi, 181)]
        self.assertLess(max_step(vals), 0.06)


class TestOverspeedPenalty(unittest.TestCase):
    def _boost_step(self, car_pos, speed):
        ball_pos = np.array([0.0, 3000.0, 93.0], dtype=np.float32)
        d = ball_pos[:2] - np.array(car_pos[:2], dtype=np.float32)
        u = d / np.linalg.norm(d)
        yaw = math.atan2(float(u[1]), float(u[0]))
        car = CarState(id=0, team=0, pos=np.array(car_pos, dtype=np.float32),
                       vel=np.array([u[0] * speed, u[1] * speed, 0.0], dtype=np.float32),
                       rot=np.array([0.0, yaw, 0.0], dtype=np.float32), boost=80.0, on_ground=True)
        arena = MockArena([car], ball_pos=list(ball_pos))
        rew = BoostReward(gain_weight=0.0, lose_weight=0.3, gamma=1.0)
        rew.reset(arena)
        car.boost = 79.0
        return rew.get_reward(car, arena, BOOST_ACT, False, None)

    def test_waived_when_lined_up(self):
        lined = self._boost_step([0.0, 2700.0, 17.0], 1600.0)
        off_line = self._boost_step([300.0, 3000.0, 17.0], 1600.0)
        self.assertAlmostEqual(lined, 0.0, places=4)
        self.assertLess(off_line, -0.2)

    def test_smooth_in_speed(self):
        vals = [self._boost_step([300.0, 3000.0, 17.0], v) for v in np.linspace(0.0, 1800.0, 181)]
        self.assertLess(max_step(vals), 0.02)


class TestCommittedStrikePacing(unittest.TestCase):
    def _approach_reward(self, start, direction):
        arena = shared_arena()
        arena.last_step_dt = DT
        car, opp = arena.cars[0], arena.cars[1]
        car.team, opp.team = 0, 1
        opp.pos = np.array([-3500.0, -4600.0, 17.0], dtype=np.float32)
        opp.vel = np.zeros(3, dtype=np.float32)
        car.ball_touches = opp.ball_touches = 0
        arena.ball.pos = np.array([0.0, 2000.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.zeros(3, dtype=np.float32)
        u = np.array(direction, dtype=np.float32)
        car.vel = u * 1400.0
        car.rot_mat = np.eye(3, dtype=np.float32)
        car.on_ground = True
        rew = PlayerToBallVelocityReward(weight=1.0)
        car.pos = np.array(start, dtype=np.float32)
        rew.reset(arena)
        total = 0.0
        for _ in range(3):
            car.pos = car.pos + u * 1400.0 * DT
            total += rew.get_reward(car, arena, BOOST_ACT, False, None)
        return total

    def test_lined_up_approach_pays_more_inside_strike_zone(self):
        lined = self._approach_reward([0.0, 1650.0, 17.0], [0.0, 1.0, 0.0])
        side = self._approach_reward([-350.0, 2000.0, 17.0], [1.0, 0.0, 0.0])
        self.assertGreater(lined, side * 1.5, f"lined={lined:.4f} side={side:.4f}")


class TestAirborneTouchBonus(unittest.TestCase):
    def _air_touch(self, ball_z, t_since=5.0, airborne=True):
        h = TouchHarness()
        v = np.array([0.0, 900.0, 0.0], dtype=np.float32)
        h.set_state([0.0, -1000.0, max(17.0, ball_z - 120.0)], [0.0, 900.0, 0.0], [0.0, -900.0, ball_z], v)
        h.car.on_ground = not airborne
        h.rew._time_since_touch[h.car.id] = t_since
        return h.touch(prev_ball_vel=v)

    def test_no_bonus_near_car_height(self):
        # Jumping into a ball at car height must not out-earn meeting it on the ground
        self.assertLessEqual(self._air_touch(200.0, airborne=True), self._air_touch(200.0, airborne=False))
        # ...and carries no airborne premium over a touch just below the bonus fade-in
        # (the small remaining gap is the ordinary ball-height multiplier)
        self.assertAlmostEqual(self._air_touch(200.0), self._air_touch(245.0), delta=0.10)

    def test_high_ball_bonus_and_air_dribble(self):
        ground_like = self._air_touch(900.0, airborne=False)
        fresh_air = self._air_touch(900.0, airborne=True)
        dribble_air = self._air_touch(900.0, t_since=0.3, airborne=True)
        self.assertGreater(fresh_air - ground_like, 1.0)
        self.assertGreater(dribble_air, 1.0, "High air-dribble contacts keep the airborne bonus")

    def test_low_hop_chain_not_repaid(self):
        repeat = self._air_touch(330.0, t_since=0.3, airborne=True)
        fresh = self._air_touch(330.0, t_since=5.0, airborne=True)
        self.assertLess(repeat, 0.25 * fresh)

    def test_smooth_in_height(self):
        vals = [self._air_touch(z, t_since=1.0) for z in np.linspace(150.0, 900.0, 151)]
        self.assertLess(max_step(vals), 0.08)


if __name__ == "__main__":
    unittest.main()
