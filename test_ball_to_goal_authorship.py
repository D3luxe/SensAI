"""BallToGoalVelocityReward must only pay the team that last touched the ball.

The term prices ball progression toward the opponent net. Without an authorship gate it also
prices progression the car had nothing to do with: an opponent's clear, a whiff, a rebound off
our own backboard. Measured under the live policy at iteration 7600, opponent-owned steps
contributed -3.17 per episode and pre-first-touch steps +1.33, against +1.03 for steps the bot
actually owned -- most of the term's magnitude was uncontrollable.

The gate is symmetric by design. Suppressing only the positive side would leave a penalty the
policy cannot avoid; conceding remains priced by the terminal concede reward.
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


def _pair():
    me = CarState(id=0, team=0, pos=np.array([0.0, -1000.0, 17.0], dtype=np.float32))
    opp = CarState(id=1, team=1, pos=np.array([0.0, 1000.0, 17.0], dtype=np.float32))
    me.ball_touches = 0
    opp.ball_touches = 0
    return me, opp


def _reward(touch_by, ball_vel=(0.0, 1800.0, 0.0), ball_pos=(0.0, 0.0, 93.0)):
    """touch_by: None, 'me' or 'opp' -- who registers a touch before the reward is read."""
    me, opp = _pair()
    arena = MockArena([me, opp], ball_pos, ball_vel)
    r = BallToGoalVelocityReward(weight=1.5)
    r.reset(arena)
    if touch_by == "me":
        me.ball_touches += 1
    elif touch_by == "opp":
        opp.ball_touches += 1
    return r.get_reward(me, arena, np.zeros(8, dtype=np.float32), False, None)


class TestAuthorshipGate(unittest.TestCase):
    def test_pays_the_team_that_touched_last(self):
        self.assertGreater(_reward("me"), 0.0,
                           "our own touch sending the ball goalward must still pay")

    def test_opponent_touch_pays_us_nothing(self):
        self.assertEqual(_reward("opp"), 0.0,
                         "an opponent clear that happens to travel toward their net is not ours")

    def test_before_any_touch_pays_nothing(self):
        self.assertEqual(_reward(None), 0.0,
                         "nobody has authored the ball's motion yet")

    def test_gate_is_symmetric(self):
        """The opponent driving the ball at OUR net must not charge this term either."""
        toward_our_net = (0.0, -2500.0, 0.0)
        self.assertEqual(_reward("opp", ball_vel=toward_our_net), 0.0,
                         "a one-way penalty the policy cannot switch off is worse than none")
        self.assertLess(_reward("me", ball_vel=toward_our_net), 0.0,
                        "our own touch driving the ball at our net must still be charged")

    def test_ownership_transfers_on_the_next_touch(self):
        me, opp = _pair()
        arena = MockArena([me, opp], (0.0, 0.0, 93.0), (0.0, 1800.0, 0.0))
        r = BallToGoalVelocityReward(weight=1.5)
        r.reset(arena)
        act = np.zeros(8, dtype=np.float32)

        me.ball_touches += 1
        self.assertGreater(r.get_reward(me, arena, act, False, None), 0.0)
        opp.ball_touches += 1
        self.assertEqual(r.get_reward(me, arena, act, False, None), 0.0)
        me.ball_touches += 1
        self.assertGreater(r.get_reward(me, arena, act, False, None), 0.0)

    def test_ownership_persists_between_touches(self):
        """A touch is one frame; the possession it creates lasts until someone else touches."""
        me, opp = _pair()
        arena = MockArena([me, opp], (0.0, 0.0, 93.0), (0.0, 1800.0, 0.0))
        r = BallToGoalVelocityReward(weight=1.5)
        r.reset(arena)
        act = np.zeros(8, dtype=np.float32)
        me.ball_touches += 1
        vals = [r.get_reward(me, arena, act, False, None) for _ in range(10)]
        self.assertTrue(all(v > 0.0 for v in vals),
                        "cumulative ball_touches must not be re-read as a fresh touch")

    def test_reset_clears_ownership(self):
        me, opp = _pair()
        arena = MockArena([me, opp], (0.0, 0.0, 93.0), (0.0, 1800.0, 0.0))
        r = BallToGoalVelocityReward(weight=1.5)
        r.reset(arena)
        me.ball_touches += 1
        act = np.zeros(8, dtype=np.float32)
        self.assertGreater(r.get_reward(me, arena, act, False, None), 0.0)
        r.reset(arena)
        self.assertEqual(r.get_reward(me, arena, act, False, None), 0.0,
                         "a new point starts with no owner")

    def test_goal_still_returns_zero(self):
        me, opp = _pair()
        arena = MockArena([me, opp], (0.0, 0.0, 93.0), (0.0, 1800.0, 0.0))
        r = BallToGoalVelocityReward(weight=1.5)
        r.reset(arena)
        me.ball_touches += 1
        self.assertEqual(r.get_reward(me, arena, np.zeros(8, dtype=np.float32), True, 0), 0.0)


class TestStillWired(unittest.TestCase):
    def test_combined_reward_still_has_the_term(self):
        c = CombinedReward({})
        self.assertIn("ball_to_goal", c.rewards)
        self.assertIsInstance(c.rewards["ball_to_goal"], BallToGoalVelocityReward)


if __name__ == "__main__":
    unittest.main(verbosity=2)
