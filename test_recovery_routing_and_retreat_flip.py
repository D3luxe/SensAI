"""
Regression tests for defensive recovery routing (the "swung wide to the corner boost" goal).

Guarantees:
  1. defensive_recovery_point sits goalside of the ball on the ball-to-goal line and is continuous near the net.
  2. go_around_offset is zero when the ball is far off the route (the clip case), and routes a car
     around a ball that blocks its straight line home, on the side the car is already on.
  3. go_around_offset has no cliffs: sweeping the ball across x=0 and across the route changes it smoothly.
  4. Pursuit shadow retarget bends the target around a blocking ball, and leaves goalside cars alone.
  5. Retreat pad scoring gates out an off-corridor big pad but keeps an on-route small pad, with a
     smooth near-pad deadzone.
  6. Boost pad-seeking shuts off for a beaten car while the ball is still far from our net.
  7. RetreatFlipReward pays a dodge toward the recovery point when beaten, nothing when goalside or
     dodging away, and ramps smoothly with how far upfield the car is.
  8. CombinedReward registers retreat_flip and update_weights reaches it.
"""

import math
import unittest
import numpy as np

from env.physics_engine import CarState
from env.rewards import (
    BoostReward, PlayerToBallVelocityReward, RetreatFlipReward, CombinedReward,
    defensive_recovery_point, go_around_offset, _best_active_pad_score, ARENA_EXTENT_X, ARENA_EXTENT_Y,
    RECOVERY_STANDOFF,
)
from test_boost_trajectory_and_urgency import MockArena


def make_car(pos, vel, yaw, boost=20.0, team=0, cid=0):
    return CarState(id=cid, team=team, pos=np.array(pos, dtype=np.float32),
                    vel=np.array(vel, dtype=np.float32),
                    rot=np.array([0.0, yaw, 0.0], dtype=np.float32),
                    boost=boost, on_ground=True)


def max_step(values):
    return max(abs(b - a) for a, b in zip(values, values[1:]))


class TestRecoveryPoint(unittest.TestCase):
    def test_point_is_goalside_on_ball_goal_line(self):
        ball = np.array([-1000.0, 1000.0, 93.0], dtype=np.float32)
        p = defensive_recovery_point(ball, car_team=0)
        self.assertLess(p[1], ball[1])
        # Collinear with ball and blue goal centre (0, -5120)
        gx, gy = 0.0 - ball[0], -ARENA_EXTENT_Y - ball[1]
        cross = (p[0] - ball[0]) * gy - (p[1] - ball[1]) * gx
        self.assertAlmostEqual(cross / math.hypot(gx, gy), 0.0, places=2)
        self.assertAlmostEqual(math.hypot(p[0] - ball[0], p[1] - ball[1]), 700.0, places=1)

    def test_point_is_held_clear_of_the_goal_line_and_walls(self):
        # A ball deep in our own corner used to put the point ~200 uu off the back wall, so the
        # distance gradient aimed a retreating car at the backboard
        for ball in ([300.0, -4900.0, 93.0], [3900.0, -4600.0, 93.0], [-3900.0, -5000.0, 93.0]):
            p = defensive_recovery_point(np.array(ball, dtype=np.float32), car_team=0)
            self.assertGreaterEqual(float(p[1]), -ARENA_EXTENT_Y + RECOVERY_STANDOFF - 1.0, ball)
            self.assertLessEqual(abs(float(p[0])), ARENA_EXTENT_X - RECOVERY_STANDOFF + 1.0, ball)

    def test_continuous_near_goal(self):
        ys = np.linspace(-4000.0, -5000.0, 101)
        pts = [float(defensive_recovery_point(np.array([0.0, y, 93.0]), 0)[1]) for y in ys]
        self.assertLess(max_step(pts), 25.0)


class TestGoAroundOffset(unittest.TestCase):
    def test_clip_case_ball_far_and_to_the_side_is_zero(self):
        # Tick 16680 of the logged goal: car deep upfield right, ball far away on the left.
        car = np.array([1326.0, 3992.0, 17.0])
        ball = np.array([-1010.0, 1683.0, 93.0])
        target = defensive_recovery_point(ball, 0)
        np.testing.assert_allclose(go_around_offset(car, ball, target, np.array([700.0, -400.0, 0.0])), [0.0, 0.0])

    def test_blocking_ball_routes_car_around_on_its_own_side(self):
        # Car 800 uu upfield of the ball, slightly to the right of the ball-goal line.
        ball = np.array([0.0, 0.0, 93.0])
        car = np.array([60.0, 800.0, 17.0])
        target = defensive_recovery_point(ball, 0)
        off = go_around_offset(car, ball, target, np.array([0.0, -1000.0, 0.0]))
        self.assertGreater(np.linalg.norm(off), 300.0, "A ball dead on the route must bend the target")
        self.assertGreater(off[0], 0.0, "Car right of the ball passes on the right")

        car_left = np.array([-60.0, 800.0, 17.0])
        off_left = go_around_offset(car_left, ball, defensive_recovery_point(ball, 0), np.array([0.0, -1000.0, 0.0]))
        self.assertLess(off_left[0], 0.0, "Car left of the ball passes on the left")

    def test_no_flip_when_ball_crosses_centre_line(self):
        car = np.array([300.0, 1000.0, 17.0])
        vel = np.array([0.0, -1200.0, 0.0])
        offs = []
        for bx in np.linspace(-150.0, 150.0, 61):
            ball = np.array([bx, 200.0, 93.0])
            offs.append(float(go_around_offset(car, ball, defensive_recovery_point(ball, 0), vel)[0]))
        # Car stays right of the ball throughout, so the shift never swaps sign and changes smoothly.
        self.assertTrue(all(o >= 0.0 for o in offs), offs)
        self.assertLess(max_step(offs), 40.0)

    def test_smooth_across_route_and_distance(self):
        ball = np.array([0.0, 0.0, 93.0])
        vel = np.array([0.0, -1000.0, 0.0])
        lateral = []
        for cx in np.linspace(-800.0, 800.0, 321):
            car = np.array([cx, 900.0, 17.0])
            lateral.append(float(go_around_offset(car, ball, defensive_recovery_point(ball, 0), vel)[0]))
        self.assertLess(max_step(lateral), 60.0)

        distance = []
        for cy in np.linspace(300.0, 2500.0, 221):
            car = np.array([50.0, cy, 17.0])
            distance.append(float(go_around_offset(car, ball, defensive_recovery_point(ball, 0), vel)[0]))
        self.assertLess(max_step(distance), 40.0)
        self.assertAlmostEqual(distance[-1], 0.0, places=3, msg="Fades out far from the ball")


class TestShadowRetargetGoAround(unittest.TestCase):
    def setUp(self):
        self.reward = PlayerToBallVelocityReward(weight=1.0)

    def test_target_bends_around_blocking_ball(self):
        car = make_car([80.0, -1200.0, 17.0], [0.0, -1200.0, 0.0], -math.pi / 2)
        arena = MockArena([car], ball_pos=[0.0, -2000.0, 93.0], ball_vel=[0.0, -800.0, 0.0])
        ball_target = arena.ball.pos.copy()
        out = self.reward._apply_shadow_retarget(ball_target, car.pos, arena, 0, I_threat=1.0, car_vel=car.vel)
        self.assertGreater(out[0], 150.0, f"Target should be pushed to the car's (right) side, got {out}")

    def test_goalside_car_untouched(self):
        car = make_car([80.0, -3000.0, 17.0], [0.0, 1000.0, 0.0], math.pi / 2)
        arena = MockArena([car], ball_pos=[0.0, -2000.0, 93.0])
        target = arena.ball.pos.copy()
        out = self.reward._apply_shadow_retarget(target, car.pos, arena, 0, I_threat=1.0, car_vel=car.vel)
        np.testing.assert_allclose(out, target)


class TestRetreatPadScoring(unittest.TestCase):
    def _score(self, pads, car_xy, heading, target_vec, is_big, w_recover, boost=5.0):
        pads = np.array(pads, dtype=np.float32)
        return _best_active_pad_score(
            np.ones(len(pads), dtype=bool), pads, car_xy[0], car_xy[1], 1200.0 if is_big else 550.0,
            np.array(heading, dtype=np.float32), np.array(heading, dtype=np.float32), True,
            np.array(target_vec, dtype=np.float32), is_big=is_big, car_boost=boost, w_recover=w_recover)

    def test_off_corridor_big_pad_gated_on_retreat(self):
        # Car heading +x toward a big pad while the route home points -y.
        pads = [[600.0, 0.0, 73.0]]
        attack = self._score(pads, (0.0, 0.0), [1.0, 0.0], [0.0, -1.0], True, w_recover=0.0)
        retreat = self._score(pads, (0.0, 0.0), [1.0, 0.0], [0.0, -1.0], True, w_recover=1.0)
        self.assertGreater(attack, 0.3)
        self.assertAlmostEqual(retreat, 0.0, places=4)

    def test_on_route_small_pad_kept_on_retreat(self):
        pads = [[100.0, -400.0, 73.0]]
        retreat = self._score(pads, (0.0, 0.0), [0.0, -1.0], [0.0, -1.0], False, w_recover=1.0)
        self.assertGreater(retreat, 0.15)

    def test_pad_scores_continuous_in_retreat_weight_and_distance(self):
        pads = [[700.0, -700.0, 73.0]]
        by_w = [self._score(pads, (0.0, 0.0), [1.0, 0.0], [0.0, -1.0], True, w) for w in np.linspace(0, 1, 101)]
        self.assertLess(max_step(by_w), 0.05)
        by_d = [self._score([[d, 0.0, 73.0]], (0.0, 0.0), [0.0, 1.0], [0.0, 1.0], True, 0.5)
                for d in np.linspace(20.0, 400.0, 191)]
        self.assertLess(max_step(by_d), 0.05, "Near-pad deadzone must not step at its boundary")


class TestBeatenCarPadSeekingShutsOff(unittest.TestCase):
    def test_clip_scenario_budget_collapses_early(self):
        # Beaten upfield, low boost, ball ~6800 uu from our net rolling home at 1500 uu/s.
        car = make_car([1326.0, 3992.0, 17.0], [700.0, -400.0, 0.0], -0.5, boost=5.0)
        arena = MockArena([car], ball_pos=[-1010.0, 1683.0, 93.0], ball_vel=[0.0, -1500.0, 0.0])
        rew = BoostReward(gain_weight=1.0, lose_weight=0.3, gamma=1.0)
        rew.reset(arena)
        budgets = []
        for _ in range(8):
            rew._transit_potential(car, arena)
            budgets.append(rew._safety_budget_filter[car.id])
        self.assertLess(budgets[-1], 0.05, budgets)
        self.assertLessEqual(max_step(budgets), 0.2001, "Slew limiter still bounds the ramp")

    def test_unbeaten_attacker_keeps_pad_seeking(self):
        car = make_car([0.0, 0.0, 17.0], [0.0, 800.0, 0.0], math.pi / 2, boost=5.0)
        arena = MockArena([car], ball_pos=[0.0, 2500.0, 93.0], ball_vel=[0.0, 0.0, 0.0])
        rew = BoostReward(gain_weight=1.0, lose_weight=0.3, gamma=1.0)
        rew.reset(arena)
        for _ in range(8):
            rew._transit_potential(car, arena)
        self.assertGreater(rew._safety_budget_filter[car.id], 0.95)


class TestRetreatFlipReward(unittest.TestCase):
    def _dodge(self, car_pos, prev_vel, vel, ball_pos=(0.0, 0.0, 93.0), ball_vel=(0.0, -1000.0, 0.0), team=0):
        car = make_car(car_pos, prev_vel, -math.pi / 2, team=team)
        arena = MockArena([car], ball_pos=list(ball_pos), ball_vel=list(ball_vel))
        r = RetreatFlipReward(weight=1.0)
        r.reset(arena)
        car.vel = np.array(vel, dtype=np.float32)
        car.just_dodged = True
        return r.get_reward(car, arena, np.zeros(8, dtype=np.float32), False, None)

    def test_beaten_flip_toward_goal_pays(self):
        rew = self._dodge([1500.0, 2500.0, 17.0], [-300.0, -1000.0, 0.0], [-450.0, -1500.0, 0.0])
        self.assertGreater(rew, 0.3)
        self.assertLessEqual(rew, 1.0)

    def test_goalside_or_away_flip_pays_nothing(self):
        self.assertEqual(self._dodge([0.0, -2000.0, 17.0], [0.0, -1000.0, 0.0], [0.0, -1500.0, 0.0]), 0.0)
        self.assertEqual(self._dodge([1500.0, 2500.0, 17.0], [0.0, 1000.0, 0.0], [0.0, 1500.0, 0.0]), 0.0)

    def test_flip_into_the_back_wall_pays_far_less_than_one_that_lands_home(self):
        home = self._dodge([1500.0, 2500.0, 17.0], [-300.0, -1000.0, 0.0], [-450.0, -1500.0, 0.0])
        into_wall = self._dodge([1500.0, -3600.0, 17.0], [-300.0, -1000.0, 0.0], [-450.0, -1800.0, 0.0],
                                ball_pos=(1200.0, -2600.0, 93.0))
        self.assertGreater(home, 0.3)
        self.assertLess(into_wall, 0.5 * home, f"home={home:.3f} into_wall={into_wall:.3f}")

    def test_still_ball_in_our_half_pays_little(self):
        # The old 0.4 threat baseline paid a full-speed flip home at a ball sitting still
        still = self._dodge([1500.0, 2500.0, 17.0], [-300.0, -1000.0, 0.0], [-450.0, -1500.0, 0.0],
                            ball_vel=(0.0, 0.0, 0.0))
        incoming = self._dodge([1500.0, 2500.0, 17.0], [-300.0, -1000.0, 0.0], [-450.0, -1500.0, 0.0])
        self.assertGreater(incoming, 0.3)
        self.assertLess(still, 0.15 * incoming, f"still={still:.3f} incoming={incoming:.3f}")

    def test_no_payout_without_dodge(self):
        car = make_car([1500.0, 2500.0, 17.0], [0.0, -1500.0, 0.0], -math.pi / 2)
        arena = MockArena([car], ball_pos=[0.0, 0.0, 93.0], ball_vel=[0.0, -1000.0, 0.0])
        r = RetreatFlipReward(weight=1.0)
        r.reset(arena)
        self.assertEqual(r.get_reward(car, arena, np.zeros(8, dtype=np.float32), False, None), 0.0)

    def test_smooth_in_upfield_margin(self):
        vals = [self._dodge([1500.0, y, 17.0], [-300.0, -1000.0, 0.0], [-450.0, -1500.0, 0.0])
                for y in np.linspace(-200.0, 1200.0, 141)]
        self.assertEqual(vals[0], 0.0)
        self.assertLess(max_step(vals), 0.05)

    def test_orange_team_mirrors(self):
        rew = self._dodge([-1500.0, -2500.0, 17.0], [300.0, 1000.0, 0.0], [450.0, 1500.0, 0.0],
                          ball_vel=(0.0, 1000.0, 0.0), team=1)
        self.assertGreater(rew, 0.3)


class TestRegistration(unittest.TestCase):
    def test_combined_reward_registers_and_updates(self):
        c = CombinedReward({"retreat_flip_weight": 0.4})
        self.assertIn("retreat_flip", c.rewards)
        self.assertAlmostEqual(c.rewards["retreat_flip"].weight, 0.4)
        c.update_weights({"retreat_flip_weight": 0.1})
        self.assertAlmostEqual(c.rewards["retreat_flip"].weight, 0.1)


if __name__ == "__main__":
    unittest.main()
