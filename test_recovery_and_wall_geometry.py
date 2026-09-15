"""
Unit tests for the recovery and wall geometry fixes:
1. is_car_on_wall arena geometry accuracy (bounds, ramps, diagonal corners, net floor, aerials).
2. Continuous monotonic angular reversal remap in PlayerToBallVelocityReward.
3. Goal-proximity challenge urgency and slow-opponent dribble tackle floor in _get_target_pos.
4. Off-axis boost waste penalty removal in BoostConsumptionPenaltyReward.
"""

import unittest
import numpy as np

from env.physics_engine import CarState, BallState, RocketSimArena, ARENA_EXTENT_X, ARENA_EXTENT_Y, CORNER_LIMIT
from env.rewards import (
    is_car_on_wall, PlayerToBallVelocityReward, BoostReward,
    compute_trajectory_arrival_time, cached_intercept_point
)


def make_car(pos, vel=None, up=(0.0, 0.0, 1.0), fwd=(1.0, 0.0, 0.0),
             cid=0, team=0, boost=50.0, on_ground=True):
    car = CarState(id=cid, team=team)
    car.pos = np.array(pos, dtype=np.float32)
    car.vel = np.array([0.0, 0.0, 0.0] if vel is None else vel, dtype=np.float32)
    car.on_ground = on_ground
    car.boost = float(boost)
    car.ang_vel = np.zeros(3, dtype=np.float32)
    u = np.array(up, dtype=np.float64)
    u /= np.linalg.norm(u)
    f = np.array(fwd, dtype=np.float64)
    if abs(np.dot(u, f)) > 0.99:
        f = np.array([0.0, 0.0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0.0, 0.0], dtype=np.float64)
    f = f - u * float(np.dot(f, u))
    f /= np.linalg.norm(f)
    car.rot_mat = np.array([f, np.cross(u, f), u], dtype=np.float32)
    return car


class TestIsCarOnWall(unittest.TestCase):
    def test_flat_ground_and_net_floor_are_not_wall(self):
        # Center of pitch
        car_center = make_car([0.0, 0.0, 17.0], on_ground=True)
        self.assertFalse(is_car_on_wall(car_center), "Center pitch floor must not be on wall")

        # Flat floor inside goal net (Y = 5500, Z = 17)
        car_net = make_car([0.0, 5500.0, 17.0], on_ground=True)
        self.assertFalse(is_car_on_wall(car_net), "Flat goal net floor must not be on wall")

        # In corner but flat on floor (Z = 17)
        car_corner_floor = make_car([3500.0, 4400.0, 17.0], on_ground=True)
        self.assertFalse(is_car_on_wall(car_corner_floor), "Corner floor must not be on wall")

    def test_airborne_cars_are_never_on_wall(self):
        # Jump inside goal net (Z = 250, on_ground=False)
        car_jump_net = make_car([0.0, 5200.0, 250.0], on_ground=False)
        self.assertFalse(is_car_on_wall(car_jump_net), "Airborne car jumping in net must not be on wall")

        # Aerial save attempt near backboard
        car_aerial = make_car([1000.0, 4800.0, 600.0], up=[0.0, 0.7, 0.7], on_ground=False)
        self.assertFalse(is_car_on_wall(car_aerial), "Airborne car attempting aerial save must not be on wall")

    def test_ceiling_is_not_wall(self):
        car_ceiling = make_car([0.0, 0.0, 1950.0], up=[0.0, 0.0, -1.0], on_ground=True)
        self.assertFalse(is_car_on_wall(car_ceiling), "Ceiling driving must not be classified as wall")

    def test_vertical_walls_detected(self):
        # Sidewall at X = 4000 (> 3980), elevated
        car_sidewall = make_car([4000.0, 0.0, 300.0], up=[-1.0, 0.0, 0.0], on_ground=True)
        self.assertTrue(is_car_on_wall(car_sidewall), "Sidewall driving must be classified as wall")

        # Backboard at Y = 5000 (> 4980), elevated
        car_backboard = make_car([0.0, 5000.0, 300.0], up=[0.0, -1.0, 0.0], on_ground=True)
        self.assertTrue(is_car_on_wall(car_backboard), "Backboard driving must be classified as wall")

    def test_transition_ramp_detected(self):
        # Transition ramp at X = 3800 (> 3700) with tilt up_tilt = 0.75 (< 0.88)
        car_ramp = make_car([3800.0, 0.0, 150.0], up=[-0.66, 0.0, 0.75], on_ground=True)
        self.assertTrue(is_car_on_wall(car_ramp), "Tilted transition ramp must be classified as wall")

    def test_diagonal_corner_detected(self):
        # Real Soccar clipped 45-degree corner wall:
        # X = 3500, Y = 4600 -> sum = 8100 > CORNER_LIMIT - 100 (7964)
        car_corner = make_car([3500.0, 4600.0, 200.0], up=[-0.7, -0.7, 0.1], on_ground=True)
        self.assertTrue(is_car_on_wall(car_corner), "Diagonal corner wall driving must be detected")


class TestReversalLinearRemap(unittest.TestCase):
    def test_reversal_is_strictly_positive_and_monotonic(self):
        rew = PlayerToBallVelocityReward(weight=1.0)
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.zeros(3, dtype=np.float32)

        # Car facing 180 deg away (fwd = [1, 0, 0], ball at [0, 0, 93], car at [1000, 0, 17])
        car_180 = make_car([1000.0, 0.0, 17.0], vel=[-300.0, 0.0, 0.0], fwd=[1.0, 0.0, 0.0], boost=50.0)
        arena.cars = [car_180]
        rew.reset(arena)

        # Move car closer to ball in reverse
        car_180_next = make_car([950.0, 0.0, 17.0], vel=[-300.0, 0.0, 0.0], fwd=[1.0, 0.0, 0.0], boost=50.0)
        arena.cars = [car_180_next]
        act = np.zeros(8, dtype=np.float32)
        act[0] = -1.0  # throttle reverse
        r_180 = rew.get_reward(car_180_next, arena, act, False, None)

        self.assertGreater(r_180, 0.0, "Reversing towards ball at 180 degrees must earn strictly positive reward (0.15x, not 0.0)")

        # Compare with car at 90 degrees (facing +Y while ball is at -X)
        car_90 = make_car([1000.0, 0.0, 17.0], vel=[-300.0, 0.0, 0.0], fwd=[0.0, 1.0, 0.0], boost=50.0)
        arena.cars = [car_90]
        rew.reset(arena)
        car_90_next = make_car([950.0, 0.0, 17.0], vel=[-300.0, 0.0, 0.0], fwd=[0.0, 1.0, 0.0], boost=50.0)
        arena.cars = [car_90_next]
        r_90 = rew.get_reward(car_90_next, arena, act, False, None)

        self.assertGreater(r_90, r_180, "90-degree alignment (0.275x) must yield higher reward than 180-degree reversal (0.15x)")


class TestGoalUrgencyAndSlowOpponent(unittest.TestCase):
    def setUp(self):
        self.rew = PlayerToBallVelocityReward(weight=1.0)
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")

    def test_goal_proximity_forces_full_challenge_urgency(self):
        # Ball deep in our defensive zone (Y = -4500, defend goal at -5120 -> d = 620 < 1500)
        self.arena.ball.pos = np.array([0.0, -4500.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, -200.0, 0.0], dtype=np.float32)

        bot = make_car([1500.0, -3500.0, 17.0], team=0, cid=0, boost=80.0)
        # Opponent touching the ball
        opp = make_car([0.0, -4400.0, 17.0], vel=[0.0, -200.0, 0.0], team=1, cid=1, boost=0.0)
        self.arena.cars = [bot, opp]

        # In _get_target_pos, goal proximity should force w_challenge = 1.0
        target = self.rew._get_target_pos(bot.pos, self.arena, False, bot.vel, bot.id, bot.team, car=bot, boost_amount=bot.boost)
        # When w_challenge is 1.0, target is the predicted intercept, NOT defensive shadow line
        pred_pos = cached_intercept_point(self.arena, bot.id, bot.pos, bot.vel, boost_amount=bot.boost)[0]
        dist_to_intercept = np.linalg.norm(target[:2] - pred_pos[:2])
        self.assertLess(dist_to_intercept, 100.0, "Near our net, target must track ball intercept directly (w_challenge = 1.0) rather than falling back to shadow line")

    def test_slow_opponent_activates_challenge_floor(self):
        # Midfield slow dribble: opponent rolling ball at 300 uu/s in midfield
        self.arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, -300.0, 0.0], dtype=np.float32)

        bot = make_car([1800.0, 200.0, 17.0], team=0, cid=0, boost=90.0)  # On flank
        opp = make_car([0.0, 100.0, 17.0], vel=[0.0, -300.0, 0.0], team=1, cid=1, boost=0.0)  # Pushing ball slowly
        self.arena.cars = [bot, opp]

        target = self.rew._get_target_pos(bot.pos, self.arena, False, bot.vel, bot.id, bot.team, car=bot, boost_amount=bot.boost)
        # Even though opp is at ball (arrival race would say bot is beat), w_challenge >= 0.45
        # prevents retreating all the way to net line
        defend_goal_y = -ARENA_EXTENT_Y
        self.assertGreater(target[1], defend_goal_y + 2000.0, "Target must step up into midfield to challenge slow dribble")


class TestBoostConsumptionPenaltyCleanup(unittest.TestCase):
    def test_off_axis_boost_penalty_is_not_charged(self):
        rew = BoostReward(gain_weight=1.0, lose_weight=0.3)
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        arena.ball.pos = np.array([0.0, 2000.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.zeros(3, dtype=np.float32)

        # Bot facing away from ball (facing -Y, ball at +Y)
        bot = make_car([0.0, 0.0, 17.0], vel=[0.0, -800.0, 0.0], fwd=[0.0, -1.0, 0.0], boost=90.0, on_ground=True)
        arena.cars = [bot]
        rew.reset(arena)

        # Bot uses boost (boost drops from 90 to 88, action[6] = 1.0)
        bot_used_boost = make_car([0.0, -50.0, 17.0], vel=[0.0, -900.0, 0.0], fwd=[0.0, -1.0, 0.0], boost=88.0, on_ground=True)
        arena.cars = [bot_used_boost]
        act = np.zeros(8, dtype=np.float32)
        act[6] = 1.0  # boost pressed

        r = rew.get_reward(bot_used_boost, arena, act, False, None)
        # Symmetrical potential delta for 2 boost loss is around -0.02
        # If the old off-axis penalty were active, it added an extra -0.15 * (1.0 - (-1.0)) = -0.30 penalty
        self.assertGreater(r, -0.10, f"Reward {r} must not include the old -0.30 off-axis boost penalty")


if __name__ == "__main__":
    unittest.main()
