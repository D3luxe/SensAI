"""
Regression tests for the kickoff speed-loss charge in PlayerToBallVelocityReward (the "stops
mid-kickoff on the straight-offset spawns" live logs: boost runs out, throttle sits slightly
negative, and the car full-brakes to a stop).

Guarantees:
  1. Shedding speed toward the ball during a kickoff costs reward on the step it happens.
  2. Coasting at constant speed and accelerating cost nothing extra.
  3. A full-brake stall earns clearly less over the kickoff than coasting the same distance.
  4. The charge is smooth in the amount of speed lost.
  5. It only applies during the kickoff (ball still at centre), not in open play.
  6. Speed lost to a bump from another car is not charged, fading back in smoothly with car distance.
"""

import unittest
import numpy as np

from env.rewards import PlayerToBallVelocityReward
from test_touch_possession_and_goal_placement import shared_arena, max_step, DT

ACT = np.zeros(8, dtype=np.float32)


class KickoffHarness:
    def __init__(self, ball_pos=(0.0, 0.0, 93.0)):
        self.arena = shared_arena()
        self.arena.last_step_dt = DT
        self.car, opp = self.arena.cars[0], self.arena.cars[1]
        self.car.team, opp.team = 0, 1
        opp.pos = np.array([256.0, 3840.0, 17.0], dtype=np.float32)
        opp.vel = np.zeros(3, dtype=np.float32)
        self.car.ball_touches = opp.ball_touches = 0
        self.arena.ball.pos = np.array(ball_pos, dtype=np.float32)
        self.arena.ball.vel = np.zeros(3, dtype=np.float32)
        self.car.on_ground = True
        self.car.rot_mat = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.rew = PlayerToBallVelocityReward(weight=1.0)

    def drive(self, speeds, start_y=-2300.0):
        """Straight up the middle toward the ball at the given speed each step."""
        y = start_y
        self.car.pos = np.array([0.0, y, 17.0], dtype=np.float32)
        self.car.vel = np.array([0.0, speeds[0], 0.0], dtype=np.float32)
        self.rew.reset(self.arena)
        out = []
        for v in speeds[1:]:
            y += v * DT
            self.car.pos = np.array([0.0, y, 17.0], dtype=np.float32)
            self.car.vel = np.array([0.0, v, 0.0], dtype=np.float32)
            out.append(self.rew.get_reward(self.car, self.arena, ACT, False, None))
        return out


class TestKickoffSpeedLoss(unittest.TestCase):
    def test_braking_step_is_charged(self):
        coast = KickoffHarness().drive([1960.0, 1960.0])[0]
        brake = KickoffHarness().drive([1960.0, 1730.0])[0]
        self.assertLess(brake, coast - 0.25, f"brake={brake:.4f} coast={coast:.4f}")

    def test_coasting_and_accelerating_not_charged(self):
        h = KickoffHarness()
        coast = h.drive([1500.0] * 6)
        accel = KickoffHarness().drive([1500.0, 1560.0, 1620.0, 1680.0, 1740.0, 1800.0])
        self.assertTrue(all(v > 0.0 for v in coast), coast)
        self.assertTrue(all(v > 0.0 for v in accel), accel)
        self.assertGreater(sum(accel), sum(coast))

    def test_stall_earns_less_than_coasting(self):
        speeds_coast = [1960.0] * 10
        speeds_stall = [1960.0 - 230.0 * i for i in range(9)] + [0.0]
        coast = sum(KickoffHarness().drive(speeds_coast))
        stall = sum(KickoffHarness().drive(speeds_stall))
        self.assertLess(stall, coast - 1.5, f"stall={stall:.3f} coast={coast:.3f}")
        self.assertLess(stall, 0.0)

    def test_smooth_in_speed_loss(self):
        vals = [KickoffHarness().drive([1960.0, 1960.0 - loss])[0] for loss in np.linspace(0.0, 400.0, 401)]
        self.assertLess(max_step(vals), 0.01, max_step(vals))

    def _brake_with_car_ahead(self, gap):
        h = KickoffHarness()
        # Opponent parked `gap` uu ahead of where the braking step ends
        end_y = -2300.0 + 1730.0 * DT
        h.arena.cars[1].pos = np.array([0.0, end_y + gap, 17.0], dtype=np.float32)
        return h.drive([1960.0, 1730.0])[0]

    def test_bump_is_not_charged(self):
        coast = KickoffHarness().drive([1960.0, 1960.0])[0]
        bumped = self._brake_with_car_ahead(150.0)
        clear = self._brake_with_car_ahead(1000.0)
        self.assertGreater(bumped, clear + 0.25, f"bumped={bumped:.4f} clear={clear:.4f}")
        self.assertGreater(bumped, coast - 0.05, f"bumped={bumped:.4f} coast={coast:.4f}")

    def test_bump_exemption_is_smooth_in_car_distance(self):
        vals = [self._brake_with_car_ahead(g) for g in np.linspace(150.0, 400.0, 251)]
        self.assertLess(max_step(vals), 0.01, max_step(vals))

    def test_open_play_unaffected(self):
        # Ball off centre: not a kickoff, so the same brake gets no kickoff charge
        rew_open = KickoffHarness(ball_pos=(0.0, 800.0, 93.0))
        brake = rew_open.drive([1960.0, 1730.0])[0]
        coast = KickoffHarness(ball_pos=(0.0, 800.0, 93.0)).drive([1960.0, 1960.0])[0]
        self.assertGreater(brake, coast - 0.05, f"open-play brake={brake:.4f} coast={coast:.4f}")
        self.assertNotIn(rew_open.car.id, rew_open.rew._prev_kickoff_speed)


if __name__ == "__main__":
    unittest.main()
