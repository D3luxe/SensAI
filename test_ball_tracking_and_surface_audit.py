"""
Regression suite for the ball-tracking / surface-orientation reward audit:
  1. Landing-surface prediction and wall-aware air-roll recovery.
  2. Wall and curved-ramp jump takeoff credit.
  3. Grounded balls near a wall are not misclassified as wall balls.
  4. Stationary pursuit potential (no free reward from target discontinuities).
  5. Ball-prediction trust guard when the engine ball is out of sync.
  6. Arrival-time-matched intercept solving.
  7. Interception timing and opponent-touch re-read shaping.
  8. Pre-play boost routing ahead of the ball.
"""

import unittest
import numpy as np
import RocketSim as rsim

from env.physics_engine import CarState, RocketSimArena, ARENA_EXTENT_X
from env.rewards import (
    AirRollRecoveryReward, JumpBridgeReward, PlayerToBallVelocityReward,
    compute_landing_surface_normal, solve_intercept_point,
    compute_trajectory_arrival_time, FLOOR_NORMAL,
)

ACT = np.zeros(8, dtype=np.float32)


def make_car(pos, vel, up=(0.0, 0.0, 1.0), fwd=(1.0, 0.0, 0.0),
             cid=0, team=0, boost=50.0, on_ground=False):
    """Builds a car with an orthonormal basis from the requested up/forward pair."""
    car = CarState(id=cid, team=team)
    car.pos = np.array(pos, dtype=np.float32)
    car.vel = np.array(vel, dtype=np.float32)
    car.on_ground = on_ground
    car.boost = boost
    car.ang_vel = np.zeros(3, dtype=np.float32)
    u = np.array(up, dtype=np.float64)
    u /= np.linalg.norm(u)
    f = np.array(fwd, dtype=np.float64)
    f = f - u * float(np.dot(f, u))
    f /= np.linalg.norm(f)
    car.rot_mat = np.array([f, np.cross(u, f), u], dtype=np.float32)
    return car


def push_to_engine(arena):
    """Mirrors the Python-side ball and cars into RocketSim.

    Assigning arena.ball.* alone leaves the engine predicting a different ball, which the
    trust guard in solve_intercept_point deliberately refuses to use. Tests that want the
    prediction path exercised must sync first.
    """
    b = arena._rsim_arena.ball.get_state()
    b.pos = rsim.Vec(*[float(v) for v in arena.ball.pos])
    b.vel = rsim.Vec(*[float(v) for v in arena.ball.vel])
    b.ang_vel = rsim.Vec(0.0, 0.0, 0.0)
    arena._rsim_arena.ball.set_state(b)
    for i, c in enumerate(arena.cars):
        rc = arena._rsim_cars[i]
        st = rc.get_state()
        st.pos = rsim.Vec(*[float(v) for v in c.pos])
        st.vel = rsim.Vec(*[float(v) for v in c.vel])
        st.boost = float(c.boost)
        rc.set_state(st)
    arena._cached_pred_step = -1
    arena._cached_rsim_preds = None
    arena._cached_pred_slices = {}


class TestLandingSurface(unittest.TestCase):
    def test_wall_bound_car_targets_wall_normal(self):
        car = make_car([3500.0, 0.0, 600.0], [900.0, 0.0, -50.0])
        np.testing.assert_allclose(compute_landing_surface_normal(car), [-1.0, 0.0, 0.0], atol=1e-6)

    def test_open_field_car_targets_floor(self):
        car = make_car([0.0, 0.0, 600.0], [200.0, 0.0, -300.0])
        np.testing.assert_allclose(compute_landing_surface_normal(car), FLOOR_NORMAL, atol=1e-6)

    def test_car_pressed_against_wall_stays_on_wall(self):
        car = make_car([ARENA_EXTENT_X - 20.0, 0.0, 800.0], [0.0, 300.0, 0.0], up=[-1.0, 0.0, 0.0], fwd=[0.0, 1.0, 0.0])
        np.testing.assert_allclose(compute_landing_surface_normal(car), [-1.0, 0.0, 0.0], atol=1e-6)

    def test_slow_drift_away_from_wall_falls_to_floor(self):
        car = make_car([1000.0, 0.0, 500.0], [-10.0, 0.0, -400.0])
        np.testing.assert_allclose(compute_landing_surface_normal(car), FLOOR_NORMAL, atol=1e-6)


class TestWallAwareAirRoll(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")
        self.arena.ball.pos = np.array([4000.0, 0.0, 900.0], dtype=np.float32)
        self.arena.ball.vel = np.zeros(3, dtype=np.float32)

    def _fly(self, up_sequence):
        rew = AirRollRecoveryReward(weight=1.0)
        seed = make_car([3500.0, 0.0, 600.0], [600.0, 300.0, 100.0], up=[0.0, 0.0, -1.0], fwd=[0.0, 1.0, 0.0])
        self.arena.cars = [seed]
        rew.reset(self.arena)
        total = 0.0
        for i, up in enumerate(up_sequence):
            car = make_car([3500.0 + i * 25.0, 0.0, 600.0], [600.0, 300.0, 100.0], up=up, fwd=[0.0, 1.0, 0.0])
            self.arena.cars = [car]
            total += rew.get_reward(car, self.arena, ACT, False, None)
        return total

    def test_wall_attitude_beats_floor_attitude_when_heading_at_wall(self):
        """A car flying at the side wall must be paid to put its wheels on the WALL, not the floor.

        Scoring recovery against world +Z taught it to land on its door and scrub off all speed.
        """
        to_floor = self._fly([[0, 0, -1], [0, 0, -1], [0, 0, -1],
                              [0.5, 0, -0.5], [0.2, 0, 0.3], [0.05, 0, 0.9], [0, 0, 1.0]])
        to_wall = self._fly([[0, 0, -1], [0, 0, -1], [0, 0, -1],
                             [-0.5, 0, -0.5], [-0.9, 0, -0.2], [-1.0, 0, 0.05], [-1.0, 0, 0.0]])
        self.assertGreater(
            to_wall, to_floor,
            f"Wall-normal recovery ({to_wall}) must out-reward floor recovery ({to_floor}) when flying at a wall")

    def test_recovery_budget_is_not_exceeded(self):
        """Roll credit and the settling bonus must share one ledger, not overwrite each other."""
        rew = AirRollRecoveryReward(weight=1.0)
        seed = make_car([0.0, 0.0, 900.0], [500.0, 0.0, -60.0], up=[0.0, 0.0, -1.0])
        self.arena.cars = [seed]
        rew.reset(self.arena)
        # Rolls steadily from inverted to upright. The small Y term keeps every entry a valid
        # direction: a pure [0, 0, 0] rung would be unnormalizable.
        ups = [[0, 0, -1], [0, 0, -1], [0, 0, -1]] + [[0.0, 0.2, -1.0 + 0.25 * k] for k in range(1, 9)]
        for i, up in enumerate(ups):
            car = make_car([float(i * 10), 0.0, 900.0 - i * 5.0], [500.0, 0.0, -60.0], up=up)
            self.arena.cars = [car]
            rew.get_reward(car, self.arena, ACT, False, None)
        spent = rew._airborne_recovery_total.get(0, 0.0)
        self.assertLessEqual(spent, 0.80 + 1e-6, f"In-flight recovery budget overrun: {spent}")


class TestWallJumpCredit(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")

    def test_jump_off_flat_wall_is_rewarded(self):
        """A jump on the vertical wall pushes along -X, so a world-Z gate never saw it."""
        self.arena.ball.pos = np.array([3900.0, 0.0, 1100.0], dtype=np.float32)
        self.arena.ball.vel = np.array([-50.0, 0.0, -100.0], dtype=np.float32)
        rew = JumpBridgeReward(weight=1.0)
        grounded = make_car([4030.0, 0.0, 900.0], [0.0, 0.0, 700.0],
                            up=[-1.0, 0.0, 0.0], fwd=[0.0, 0.0, 1.0], on_ground=True, boost=60.0)
        self.arena.cars = [grounded]
        rew.reset(self.arena)
        rew._prev_on_ground[grounded.id] = True
        airborne = make_car([4020.0, 0.0, 940.0], [-292.0, 0.0, 700.0],
                            up=[-1.0, 0.0, 0.0], fwd=[0.0, 0.0, 1.0], on_ground=False, boost=60.0)
        self.arena.cars = [airborne]
        self.assertGreater(rew.get_reward(airborne, self.arena, ACT, False, None), 0.0,
                           "Wall takeoff toward a wall ball must earn takeoff credit")

    def test_wall_flip_is_not_charged_as_a_wasteful_backflip(self):
        """Horizontal nose projections are degenerate on a wall; the backflip penalty must not fire."""
        self.arena.ball.pos = np.array([3900.0, 0.0, 1400.0], dtype=np.float32)
        self.arena.ball.vel = np.array([-200.0, 0.0, -50.0], dtype=np.float32)
        rew = JumpBridgeReward(weight=1.0)
        climbing = make_car([4020.0, 0.0, 1100.0], [-100.0, 0.0, 900.0],
                            up=[-1.0, 0.0, 0.0], fwd=[0.0, 0.0, 1.0], on_ground=False, boost=40.0)
        self.arena.cars = [climbing]
        rew.reset(self.arena)
        rew._prev_on_ground[climbing.id] = False
        rew._prev_has_flip[climbing.id] = True
        climbing.has_flip = False
        act = np.zeros(8, dtype=np.float32)
        act[2] = -1.0  # pitch back: away from the wall, toward the ball
        self.assertGreaterEqual(rew.get_reward(climbing, self.arena, act, False, None), 0.0,
                                "Pitching off a wall toward the ball must not be charged as a backflip")


class TestGroundedBallNearWallClassification(unittest.TestCase):
    """A resting ball sits at z = 91.25, so a z > 80 gate called every corner ball a wall ball."""

    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")
        self.rew = PlayerToBallVelocityReward(weight=1.0)

    def _wall_climb_applied(self, ball_pos):
        self.arena.ball.pos = np.array(ball_pos, dtype=np.float32)
        self.arena.ball.vel = np.array([200.0, 0.0, 0.0], dtype=np.float32)
        ball_z = float(self.arena.ball.pos[2])
        in_goal_mouth = bool(abs(self.arena.ball.pos[0]) < 900.0 and ball_z < 650.0)
        return bool((abs(self.arena.ball.pos[0]) > 3350.0 or abs(self.arena.ball.pos[1]) > 4350.0)
                    and not in_goal_mouth and ball_z > 160.0)

    def test_resting_ball_near_sidewall_is_not_a_wall_ball(self):
        self.assertFalse(self._wall_climb_applied([3500.0, 0.0, 91.25]))

    def test_ball_climbing_the_ramp_is_a_wall_ball(self):
        self.assertTrue(self._wall_climb_applied([3500.0, 0.0, 400.0]))


class TestStationaryPursuitPotential(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")

    def test_target_discontinuity_does_not_pay_out(self):
        """A stationary car must earn ~nothing however violently the ball's trajectory jumps.

        The pursuit target blends toward a predicted intercept, so it moves discontinuously on
        any touch or bounce. Differencing raw distances across that jump paid the car for an
        opponent knocking the ball toward it.
        """
        self.arena.ball.pos = np.array([1500.0, 0.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([900.0, 0.0, 0.0], dtype=np.float32)
        car = make_car([0.0, 0.0, 17.0], [0.0, 0.0, 0.0], on_ground=True, boost=100.0)
        opp = make_car([1500.0, 900.0, 17.0], [0.0, 0.0, 0.0], cid=1, team=1, on_ground=True)
        self.arena.cars = [car, opp]
        push_to_engine(self.arena)

        rew = PlayerToBallVelocityReward(weight=1.0)
        rew.reset(self.arena)
        rew.get_reward(car, self.arena, ACT, False, None)

        # Opponent strikes the ball hard back toward the stationary car.
        self.arena.ball.vel = np.array([-2200.0, 0.0, 0.0], dtype=np.float32)
        opp.ball_touches += 1
        push_to_engine(self.arena)
        reward = rew.get_reward(car, self.arena, ACT, False, None)

        self.assertLess(abs(reward), 0.25,
                        f"Stationary car banked {reward} purely from an opponent's touch")


class TestPredictionTrustGuard(unittest.TestCase):
    def test_desynced_prediction_falls_back_to_live_ball(self):
        """If the engine is predicting a different ball, chasing it is worse than not predicting."""
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        # Step once so the engine's own trajectory prediction is primed. Before that
        # get_predicted_ball_pos integrates the Python ball itself and cannot desync.
        arena.step([np.zeros(8, dtype=np.float32) for _ in arena.cars], dt=8.0 / 120.0)
        # Assigned without push_to_engine: the engine still holds its own kickoff ball.
        arena.ball.pos = np.array([1500.0, -4500.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.array([500.0, 1600.0, 200.0], dtype=np.float32)
        car = make_car([1500.0, -4500.0, 17.0], [0.0, 1000.0, 0.0], on_ground=True)
        arena.cars = [car]
        pos, t = solve_intercept_point(car.pos, car.vel, arena)
        np.testing.assert_allclose(pos, arena.ball.pos, atol=1e-4)
        self.assertEqual(t, 0.0)


class TestInterceptSolver(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")

    def test_intercept_leads_a_fast_ball(self):
        self.arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([1500.0, 0.0, 0.0], dtype=np.float32)
        car = make_car([-2000.0, 0.0, 17.0], [1400.0, 0.0, 0.0], on_ground=True)
        self.arena.cars = [car]
        push_to_engine(self.arena)
        pos, t = solve_intercept_point(car.pos, car.vel, self.arena)
        self.assertGreater(t, 0.0, "A ball racing away must be met downrange, not where it is now")
        self.assertGreater(float(pos[0]), float(self.arena.ball.pos[0]))

    def test_reachable_stationary_ball_is_met_where_it_is(self):
        self.arena.ball.pos = np.array([500.0, 0.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([400.0, 0.0, 0.0], dtype=np.float32)
        car = make_car([0.0, 0.0, 17.0], [1600.0, 0.0, 0.0], on_ground=True)
        self.arena.cars = [car]
        push_to_engine(self.arena)
        _, t = solve_intercept_point(car.pos, car.vel, self.arena)
        self.assertLessEqual(t, 0.5, "A ball the car is already on top of needs no long lead")


class TestBoostRoutingAheadOfBall(unittest.TestCase):
    """The reported failure: ball rolling to the side wall, bot follows it up with 0 boost."""

    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")
        self.arena.ball.pos = np.array([2400.0, 200.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([850.0, 250.0, 0.0], dtype=np.float32)

    def _sample(self, boost, opp_pos):
        car = make_car([900.0, -500.0, 17.0], [1000.0, 700.0, 0.0], on_ground=True, boost=boost)
        opp = make_car(opp_pos, [0.0, 0.0, 0.0], cid=1, team=1, on_ground=True, boost=100.0)
        self.arena.cars = [car, opp]
        push_to_engine(self.arena)
        rew = PlayerToBallVelocityReward(weight=1.0, boost_pathing_threshold=50.0)
        rew.reset(self.arena)
        return rew.get_reward(car, self.arena, ACT, False, None)

    def test_low_boost_car_is_paid_to_route_through_a_pad_ahead(self):
        empty = self._sample(0.0, [-1500.0, 3800.0, 17.0])
        full = self._sample(100.0, [-1500.0, 3800.0, 17.0])
        self.assertGreater(empty, full,
                           f"Low-boost routing ({empty}) must beat the same line on full boost ({full})")

    def test_no_refuel_credit_when_the_opponent_is_contesting(self):
        """Detouring is only affordable while the race is comfortably won."""
        safe = self._sample(0.0, [-1500.0, 3800.0, 17.0])
        contested = self._sample(0.0, [2600.0, 400.0, 17.0])
        self.assertGreater(safe, contested,
                           f"Refuel credit ({safe}) must not survive an opponent on the ball ({contested})")


class TestInterceptionTiming(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")

    def test_opponent_touch_does_not_pay_the_re_read_baseline_step(self):
        """The step an opponent touches re-seeds the timing baseline rather than banking the jump."""
        self.arena.ball.pos = np.array([1200.0, 0.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([800.0, 0.0, 0.0], dtype=np.float32)
        car = make_car([0.0, 0.0, 17.0], [900.0, 0.0, 0.0], on_ground=True, boost=100.0)
        opp = make_car([2500.0, 0.0, 17.0], [-800.0, 0.0, 0.0], cid=1, team=1, on_ground=True)
        self.arena.cars = [car, opp]
        push_to_engine(self.arena)
        rew = PlayerToBallVelocityReward(weight=1.0)
        rew.reset(self.arena)
        rew.get_reward(car, self.arena, ACT, False, None)

        self.arena.ball.vel = np.array([-1800.0, 600.0, 0.0], dtype=np.float32)
        opp.ball_touches += 1
        push_to_engine(self.arena)
        rew.get_reward(car, self.arena, ACT, False, None)
        self.assertGreater(rew._reread_ticks.get(car.id, 0), 0,
                           "An opponent touch must open the re-read window")


if __name__ == "__main__":
    unittest.main(verbosity=2)
