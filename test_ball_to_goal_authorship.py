"""BallToGoalVelocityReward must price ball progression regardless of who touched it last.

An authorship gate (pay/charge only while our team was the last toucher) broke telescoping
across possession changes: our touch forward was paid, the opponent's return was free. At
iteration 8200 the term paid +11.1 against -2.1 per episode and out-earned goals, and the policy
stopped contesting balls the opponent had touched. These tests pin the ungated behaviour.
"""

import unittest

import numpy as np

from env.physics_engine import CarState, BallState
from env.rewards import BallToGoalVelocityReward, CombinedReward


class MockArena:
    def __init__(self, cars, ball_pos, ball_vel):
        self.ball = BallState(pos=np.array(ball_pos, dtype=np.float32))
        self.ball.vel = np.array(ball_vel, dtype=np.float32)
        self.cars = cars
        self.boost_pads = []


def _pair(me_pos=None):
    p = np.array([0.0, -1000.0, 17.0], dtype=np.float32) if me_pos is None else np.array(me_pos, dtype=np.float32)
    me = CarState(id=0, team=0, pos=p)
    opp = CarState(id=1, team=1, pos=np.array([0.0, 1000.0, 17.0], dtype=np.float32))
    me.ball_touches = 0
    opp.ball_touches = 0
    return me, opp


def _reward(touch_by, ball_vel=(0.0, 1800.0, 0.0), ball_pos=(0.0, 0.0, 93.0), car_pos=None):
    """touch_by: None, 'me' or 'opp' -- who registers a touch before the reward is read."""
    me, opp = _pair(me_pos=car_pos)
    arena = MockArena([me, opp], ball_pos, ball_vel)
    r = BallToGoalVelocityReward(weight=1.5)
    r.reset(arena)
    if touch_by == "me":
        me.ball_touches += 1
    elif touch_by == "opp":
        opp.ball_touches += 1
    return r.get_reward(me, arena, np.zeros(8, dtype=np.float32), False, None)


class TestNoAuthorshipGate(unittest.TestCase):
    def test_toucher_does_not_change_the_reward(self):
        for vel in ((0.0, 1800.0, 0.0), (0.0, -2500.0, 0.0)):
            vals = {who: _reward(who, ball_vel=vel) for who in (None, "me", "opp")}
            self.assertAlmostEqual(vals["me"], vals["opp"], places=6, msg=str(vel))
            self.assertAlmostEqual(vals["me"], vals[None], places=6, msg=str(vel))

    def test_goalward_pays_and_return_charges(self):
        self.assertGreater(_reward("me"), 0.0)
        self.assertLess(_reward("opp", ball_vel=(0.0, -2500.0, 0.0)), 0.0,
                        "the opponent driving the ball back at our net must cost ground")

    def test_opponent_return_cancels_our_push(self):
        """Forward then straight back at the same speed must net to zero."""
        car_pos = (3000.0, -300.0, 17.0)
        fwd = _reward("me", ball_vel=(0.0, 1500.0, 0.0), ball_pos=(3000.0, 0.0, 93.0), car_pos=car_pos)
        back = _reward("opp", ball_vel=(0.0, -1500.0, 0.0), ball_pos=(3000.0, 0.0, 93.0), car_pos=car_pos)
        self.assertGreater(fwd, 0.0)
        self.assertLess(back, 0.0)
        # Off-target (x=3000) so the placement bonus is zero on the forward leg.
        self.assertAlmostEqual(fwd + back, 0.0, places=5)

    def test_goal_still_returns_zero(self):
        me, opp = _pair()
        arena = MockArena([me, opp], (0.0, 0.0, 93.0), (0.0, 1800.0, 0.0))
        r = BallToGoalVelocityReward(weight=1.5)
        r.reset(arena)
        self.assertEqual(r.get_reward(me, arena, np.zeros(8, dtype=np.float32), True, 0), 0.0)


class TestStillWired(unittest.TestCase):
    def test_combined_reward_still_has_the_term(self):
        c = CombinedReward({})
        self.assertIn("ball_to_goal", c.rewards)
        self.assertIsInstance(c.rewards["ball_to_goal"], BallToGoalVelocityReward)

    def test_own_goal_threat_on_by_default(self):
        c = CombinedReward({})
        self.assertEqual(c.rewards["own_goal_threat"].weight, 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
