"""
Unit tests for Trajectory-Aligned & Urgency-Gated Boost Transit shaping in BoostReward.

Guarantees:
1. Pad-trajectory alignment: pads ahead on travel trajectory are scored; pads behind are zeroed.
2. Dynamic back-post targeting on defensive retreat: pads along defensive rotation routes are rewarded.
3. Singularity deadzone: d < 150 uu locks trajectory_mult = 1.0 with no zero-division.
4. Boost deficit multiplier: starving cars strongly prioritize big 100-orbs over small pennies.
5. Dynamic urgency gating: opponent threats reduce safety budget, buffered by a slew limiter.
6. Telescoping & No Reward Pump: approach-and-reverse loops net zero (or <= 0 under gamma=0.99).
7. Reset cleanly reinitializes stateful memory.
"""

import unittest
import numpy as np

from env.physics_engine import CarState, BallState
from env.rewards import BoostReward, ARENA_EXTENT_Y


class MockArena:
    def __init__(self, cars, ball_pos, ball_vel=None):
        self.ball = BallState(pos=np.array(ball_pos, dtype=np.float32))
        self.ball.vel = np.array([0.0, 0.0, 0.0] if ball_vel is None else ball_vel, dtype=np.float32)
        self.cars = cars
        self.boost_pads = []
        # Setup big and small pads
        # Big pad at (3072, 0, 73) - side right pad
        # Big pad at (-3072, 0, 73) - side left pad
        # Big pad at (-3072, -4096, 73) - defensive corner pad
        self._big_pad_pos_3d = np.array([
            [3072.0, 0.0, 73.0],
            [-3072.0, 0.0, 73.0],
            [-3072.0, -4096.0, 73.0],
            [3072.0, -4096.0, 73.0],
        ], dtype=np.float32)
        self._big_pad_active = np.ones(4, dtype=bool)

        self._small_pad_pos_3d = np.array([
            [0.0, -1000.0, 73.0],
            [0.0, -2000.0, 73.0],
            [-1000.0, -3000.0, 73.0],
        ], dtype=np.float32)
        self._small_pad_active = np.ones(3, dtype=bool)


class TestBoostTrajectoryAndUrgency(unittest.TestCase):
    def test_trajectory_alignment_forward_vs_reverse(self):
        """Pad directly in front of the car gets positive score; pad directly behind gets 0.0."""
        rew = BoostReward(gain_weight=1.0, lose_weight=0.3, gamma=1.0)
        car = CarState(id=0, team=0, pos=np.array([2500.0, 0.0, 17.0], dtype=np.float32),
                       vel=np.array([500.0, 0.0, 0.0], dtype=np.float32),
                       rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Facing +X toward (3072, 0)
                       boost=20.0, on_ground=True)
        arena = MockArena([car], ball_pos=[0.0, 2000.0, 93.0])
        rew.reset(arena)

        psi_forward = rew._transit_potential(car, arena)
        self.assertGreater(psi_forward, 0.0, "Car driving toward big pad ahead must earn positive transit potential")

        # Now face car 180 degrees away (-X)
        car.rot = np.array([0.0, np.pi, 0.0], dtype=np.float32)
        car.vel = np.array([-500.0, 0.0, 0.0], dtype=np.float32)
        psi_reverse = rew._transit_potential(car, arena)
        self.assertAlmostEqual(psi_reverse, 0.0, places=4, msg="Pad directly behind car must yield zero transit score")

    def test_defensive_backpost_rotation(self):
        """When retreating on defense, corridor targets the back-post rather than the ball behind."""
        rew = BoostReward(gain_weight=1.0, lose_weight=0.3, gamma=1.0)
        # Blue car caught ahead of ball at Y = 1000, ball at Y = 0 (goalside_margin = 1000 > 300)
        # Ball is on positive X (X = 1500), so backpost should be negative X (-800, -5120)
        car = CarState(id=0, team=0, pos=np.array([-2000.0, 500.0, 17.0], dtype=np.float32),
                       vel=np.array([0.0, -800.0, 0.0], dtype=np.float32),  # Sprinting back to defense (-Y)
                       rot=np.array([0.0, -np.pi / 2, 0.0], dtype=np.float32),  # Facing -Y
                       boost=20.0, on_ground=True)
        arena = MockArena([car], ball_pos=[1500.0, 0.0, 93.0])
        rew.reset(arena)

        # Pad at (-3072, -4096) is in the defensive corner along the negative X retreat route
        psi_retreat = rew._transit_potential(car, arena)
        self.assertTrue(rew._retreat_mode[car.id], "Car caught ahead of ball and sprinting back must enter retreat mode")
        self.assertGreater(psi_retreat, 0.0, "Retreating car should be reinforced for routing through defensive pads")

    def test_vector_singularity_deadzone(self):
        """Inside 150 uu of pad center, trajectory_mult locks to 1.0 with no zero division or angular flutter."""
        rew = BoostReward(gain_weight=1.0, lose_weight=0.3, gamma=1.0)
        pad = [3072.0, 0.0, 73.0]
        # Car stationary (v=0) right next to pad (d = 10 uu)
        car = CarState(id=0, team=0, pos=np.array([pad[0] - 10.0, pad[1], 17.0], dtype=np.float32),
                       vel=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                       rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                       boost=10.0, on_ground=True)
        arena = MockArena([car], ball_pos=[0.0, 2000.0, 93.0])
        rew.reset(arena)

        psi = rew._transit_potential(car, arena)
        self.assertFalse(np.isnan(psi), "Potential must not be NaN at small distance")
        self.assertGreater(psi, 0.0, "Close proximity inside deadzone must yield strong potential")

    def test_pad_type_deficit_multiplier(self):
        """Starving car (0 boost) gives big pad 2.0x priority multiplier compared to full tank."""
        rew = BoostReward(gain_weight=1.0, lose_weight=0.3, gamma=1.0)
        car_starving = CarState(id=0, team=0, pos=np.array([2500.0, 0.0, 17.0], dtype=np.float32),
                                vel=np.array([500.0, 0.0, 0.0], dtype=np.float32),
                                rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                                boost=0.0, on_ground=True)
        car_full = CarState(id=0, team=0, pos=np.array([2500.0, 0.0, 17.0], dtype=np.float32),
                            vel=np.array([500.0, 0.0, 0.0], dtype=np.float32),
                            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                            boost=49.0, on_ground=True)
        arena = MockArena([car_starving], ball_pos=[0.0, 2000.0, 93.0])
        rew.reset(arena)

        psi_starving = rew._transit_potential(car_starving, arena)
        psi_full = rew._transit_potential(car_full, arena)
        self.assertGreater(psi_starving, psi_full * 2.0, "Starving car should have vastly higher pad attraction")

    def test_dynamic_urgency_and_slew_rate(self):
        """Threat intensity from charging opponent reduces safety budget, buffered by slew limiter."""
        rew = BoostReward(gain_weight=1.0, lose_weight=0.3, gamma=1.0)
        car = CarState(id=0, team=0, pos=np.array([2500.0, 0.0, 17.0], dtype=np.float32),
                       vel=np.array([500.0, 0.0, 0.0], dtype=np.float32),
                       rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                       boost=20.0, on_ground=True)
        # Opponent charging ball at high speed
        opp = CarState(id=1, team=1, pos=np.array([0.0, 1000.0, 17.0], dtype=np.float32),
                       vel=np.array([0.0, 1800.0, 0.0], dtype=np.float32),
                       rot=np.array([0.0, np.pi / 2, 0.0], dtype=np.float32))
        arena = MockArena([car, opp], ball_pos=[0.0, 1200.0, 93.0])
        rew.reset(arena)

        # Step 1: safety budget slew limiter prevents instantaneous collapse from 1.0 to 0.0
        rew._safety_budget_filter[car.id] = 1.0
        psi_step1 = rew._transit_potential(car, arena)
        budget1 = rew._safety_budget_filter[car.id]
        self.assertGreaterEqual(budget1, 0.79, f"Safety budget must not drop more than 0.20 per step, got {budget1}")

        # Over multiple steps, budget smoothly ramps down
        for _ in range(5):
            rew._transit_potential(car, arena)
        final_budget = rew._safety_budget_filter[car.id]
        self.assertLess(final_budget, 0.20, f"Safety budget should smoothly ramp down under high threat, got {final_budget}")

    def test_telescoping_and_no_reward_pump(self):
        """Approach-and-reverse loop proves that no reward pump exploit exists."""
        rew_undiscounted = BoostReward(gain_weight=1.0, lose_weight=0.3, gamma=1.0)
        rew_discounted = BoostReward(gain_weight=1.0, lose_weight=0.3, gamma=0.99)

        car = CarState(id=0, team=0, pos=np.array([2000.0, 0.0, 17.0], dtype=np.float32),
                       vel=np.array([500.0, 0.0, 0.0], dtype=np.float32),
                       rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                       boost=20.0, on_ground=True)
        arena = MockArena([car], ball_pos=[0.0, 2000.0, 93.0])

        act = np.zeros(8, dtype=np.float32)

        # 1. Test undiscounted (gamma = 1.0): 5 approach and reverse cycles must net exactly 0.0
        rew_undiscounted.reset(arena)
        total_undiscounted = 0.0
        for _ in range(5):
            # Approach pad at (3072, 0)
            for x in (2200.0, 2400.0, 2600.0, 2800.0):
                car.pos[0] = x
                total_undiscounted += rew_undiscounted.get_reward(car, arena, act, False, None)
            # Reverse back to start
            for x in (2600.0, 2400.0, 2200.0, 2000.0):
                car.pos[0] = x
                total_undiscounted += rew_undiscounted.get_reward(car, arena, act, False, None)
        self.assertAlmostEqual(total_undiscounted, 0.0, places=5,
                               msg=f"Undiscounted approach-and-reverse loop must net exactly 0, got {total_undiscounted}")

        # 2. Test discounted (gamma = 0.99): must net <= 0.0, proving no reward pump exploit exists
        rew_discounted.reset(arena)
        total_discounted = 0.0
        for _ in range(5):
            for x in (2200.0, 2400.0, 2600.0, 2800.0):
                car.pos[0] = x
                total_discounted += rew_discounted.get_reward(car, arena, act, False, None)
            for x in (2600.0, 2400.0, 2200.0, 2000.0):
                car.pos[0] = x
                total_discounted += rew_discounted.get_reward(car, arena, act, False, None)
        self.assertLessEqual(total_discounted, 1e-6,
                             f"Discounted approach-and-reverse loop must be non-positive, got {total_discounted}")

    def test_reset_clears_stateful_memory(self):
        """Reset clears retreat mode, initializes safety budget to 1.0, and sets transit potential."""
        rew = BoostReward(gain_weight=1.0, lose_weight=0.3, gamma=1.0)
        car = CarState(id=0, team=0, pos=np.array([2500.0, 0.0, 17.0], dtype=np.float32),
                       vel=np.array([500.0, 0.0, 0.0], dtype=np.float32),
                       boost=20.0, on_ground=True)
        arena = MockArena([car], ball_pos=[0.0, 2000.0, 93.0])

        # Pollute state
        rew._retreat_mode[car.id] = True
        rew._safety_budget_filter[car.id] = 0.1

        # Reset
        rew.reset(arena)
        self.assertFalse(rew._retreat_mode[car.id])
        self.assertEqual(rew._safety_budget_filter[car.id], 1.0)
        self.assertIn(car.id, rew._prev_transit)

    def test_goalside_defending_car_never_forced_into_retreat(self):
        """A car that is already goalside of the ball must never be forced into retreat even when driving fast defensively."""
        rew = BoostReward(gain_weight=1.0, lose_weight=0.3, gamma=1.0)
        # Blue car at Y = -1500 (goalside), ball at Y = 0 (goalside_margin = -1500 <= 0)
        # Sprinting towards own net at -1800 uu/s
        car = CarState(id=0, team=0, pos=np.array([0.0, -1500.0, 17.0], dtype=np.float32),
                       vel=np.array([0.0, -1800.0, 0.0], dtype=np.float32),
                       rot=np.array([0.0, -np.pi / 2, 0.0], dtype=np.float32),
                       boost=20.0, on_ground=True)
        arena = MockArena([car], ball_pos=[500.0, 0.0, 93.0])
        rew.reset(arena)

        rew._transit_potential(car, arena)
        self.assertFalse(rew._retreat_mode[car.id], "Goalside car must NEVER enter retreat mode")


if __name__ == "__main__":
    unittest.main(verbosity=2)

