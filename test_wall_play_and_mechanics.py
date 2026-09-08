"""
Unit test suite validating wall play mechanics remediation:
  1. Precision wall geometry & goal-mouth trap exclusion.
  2. Unblocked vel_toward_ball & delta_dist for cars approaching and climbing walls.
  3. Preservation of wall-riding dampening when ball rebounds into open infield.
  4. Multi-surface landing evaluator in AirRollRecoveryReward (4 wheels on wall rewarded, not crash-penalized).
  5. Wall strike bonus in TouchBallReward.
  6. On-wall spawn generation in WallPlaySetter.
"""

import unittest
import math
import numpy as np

from env.physics_engine import CarState, BallState, ARENA_EXTENT_X, ARENA_EXTENT_Y
from env.rewards import PlayerToBallVelocityReward, AirRollRecoveryReward, TouchBallReward
import RocketSim as rsim
from env.state_setters import WallPlaySetter


class MockArena:
    def __init__(self, ball: BallState, cars: list[CarState]):
        self.ball = ball
        self.cars = cars

    def get_predicted_ball_pos(self, slice_ticks: int = 60) -> np.ndarray:
        return self.ball.pos.copy()


class TestWallPlayMechanics(unittest.TestCase):

    def test_precision_wall_geometry_and_goal_mouth_exclusion(self):
        reward_fn = PlayerToBallVelocityReward(weight=1.0)

        # 1. Sidewall ball (X=3900, Z=500): On wall, NOT elevated open-air aerial
        ball_sidewall = BallState(pos=np.array([3900.0, 0.0, 500.0], dtype=np.float32))
        car_ground = CarState(id=0, team=0, pos=np.array([3000.0, 0.0, 17.0], dtype=np.float32), vel=np.array([1000.0, 0.0, 0.0], dtype=np.float32))
        arena1 = MockArena(ball_sidewall, [car_ground])
        reward_fn.reset(arena1)

        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        car_ground.pos[0] += 50.0
        r1 = reward_fn.get_reward(car_ground, arena1, action, False, None)
        self.assertGreater(r1, 0.05, f"Car approaching sidewall ball should receive positive reward, got {r1}")

        # 2. Backboard ball above crossbar (Y=4900, Z=800, X=0): On backboard wall
        ball_backboard = BallState(pos=np.array([0.0, 4900.0, 800.0], dtype=np.float32))
        car2 = CarState(
            id=0, team=0,
            pos=np.array([0.0, 3500.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1000.0, 0.0], dtype=np.float32),
            rot_mat=np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        )
        arena2 = MockArena(ball_backboard, [car2])
        reward_fn.reset(arena2)
        car2.pos[1] += 50.0
        r2 = reward_fn.get_reward(car2, arena2, action, False, None)
        self.assertGreater(r2, 0.05, f"Car approaching backboard ball should receive positive reward, got {r2}")

    def test_car_climbing_wall_rewarded(self):
        reward_fn = PlayerToBallVelocityReward(weight=1.0)

        car_wall = CarState(
            id=0, team=0,
            pos=np.array([4000.0, 500.0, 400.0], dtype=np.float32),
            vel=np.array([0.0, 0.0, 1000.0], dtype=np.float32),
            rot_mat=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
            on_ground=True
        )
        ball_wall = BallState(pos=np.array([4000.0, 500.0, 700.0], dtype=np.float32))
        arena = MockArena(ball_wall, [car_wall])
        reward_fn.reset(arena)

        car_wall.pos[2] += 50.0
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r = reward_fn.get_reward(car_wall, arena, action, False, None)
        self.assertGreater(r, 0.045, f"Car climbing wall toward wall ball must receive positive reward, got {r}")

        # Downfield wall climb (dist > 500 uu):
        ball_far = BallState(pos=np.array([4000.0, 500.0, 1100.0], dtype=np.float32))
        arena_far = MockArena(ball_far, [car_wall])
        reward_fn.reset(arena_far)
        car_wall.pos[2] += 50.0
        r_far = reward_fn.get_reward(car_wall, arena_far, action, False, None)
        self.assertGreater(r_far, 0.10, f"Downfield wall climb must receive strong velocity & distance reward, got {r_far}")

    def test_wall_riding_dampening_when_ball_infield(self):
        reward_fn = PlayerToBallVelocityReward(weight=1.0)

        car_wall = CarState(
            id=0, team=0,
            pos=np.array([4000.0, 0.0, 500.0], dtype=np.float32),
            vel=np.array([0.0, 1000.0, 0.0], dtype=np.float32),
            rot_mat=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
            on_ground=True
        )
        ball_infield = BallState(pos=np.array([1000.0, 500.0, 200.0], dtype=np.float32))
        arena = MockArena(ball_infield, [car_wall])
        reward_fn.reset(arena)

        car_wall.pos[1] += 50.0
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r_infield = reward_fn.get_reward(car_wall, arena, action, False, None)

        ball_on_wall = BallState(pos=np.array([4000.0, 500.0, 500.0], dtype=np.float32))
        car_wall2 = CarState(
            id=0, team=0,
            pos=np.array([4000.0, 0.0, 500.0], dtype=np.float32),
            vel=np.array([0.0, 1000.0, 0.0], dtype=np.float32),
            rot_mat=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
            on_ground=True
        )
        arena2 = MockArena(ball_on_wall, [car_wall2])
        reward_fn.reset(arena2)
        car_wall2.pos[1] += 50.0
        r_wall = reward_fn.get_reward(car_wall2, arena2, action, False, None)

        self.assertGreater(r_wall, r_infield * 2.0, f"Chasing wall ball ({r_wall}) must greatly exceed wall-riding ({r_infield})")

    def test_air_roll_recovery_wall_landing(self):
        reward_fn = AirRollRecoveryReward(weight=1.0)

        car = CarState(
            id=0, team=0,
            pos=np.array([3950.0, 0.0, 400.0], dtype=np.float32),
            vel=np.array([100.0, 0.0, 0.0], dtype=np.float32),
            rot_mat=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
            on_ground=False
        )
        ball = BallState(pos=np.array([3950.0, 500.0, 500.0], dtype=np.float32))
        arena = MockArena(ball, [car])
        reward_fn.reset(arena)

        action = np.zeros(8, dtype=np.float32)

        for _ in range(4):
            car.pos[0] += 5.0
            reward_fn.get_reward(car, arena, action, False, None)

        car.on_ground = True
        car.pos[0] = 4000.0
        landing_reward = reward_fn.get_reward(car, arena, action, False, None)

        self.assertGreaterEqual(landing_reward, 0.0, f"Landing on 4 wheels on wall must not receive crash penalty, got {landing_reward}")

    def test_touch_ball_wall_strike_bonus(self):
        touch_fn = TouchBallReward(weight=1.0)

        car = CarState(
            id=0, team=0,
            pos=np.array([4000.0, 0.0, 500.0], dtype=np.float32),
            vel=np.array([0.0, 800.0, 0.0], dtype=np.float32),
            rot_mat=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
            on_ground=True,
            ball_touches=0
        )
        ball = BallState(
            pos=np.array([4000.0, 50.0, 500.0], dtype=np.float32),
            vel=np.array([0.0, 800.0, 0.0], dtype=np.float32)
        )
        arena = MockArena(ball, [car])
        touch_fn.reset(arena)

        car.ball_touches = 1
        ball.vel = np.array([-200.0, 1200.0, 200.0], dtype=np.float32)
        action = np.zeros(8, dtype=np.float32)
        r = touch_fn.get_reward(car, arena, action, False, None)
        self.assertGreater(r, 1.2, f"Wall strike should award enhanced touch reward, got {r}")

    def test_wall_play_setter_generates_on_wall_spawns(self):
        rsim_arena = rsim.Arena(rsim.GameMode.SOCCAR)
        rsim_arena.add_car(rsim.Team.BLUE)
        rsim_arena.add_car(rsim.Team.ORANGE)

        setter = WallPlaySetter()
        on_wall_count = 0
        total_trials = 40

        for _ in range(total_trials):
            setter.reset(rsim_arena, 2)
            cars = rsim_arena.get_cars()
            c0 = cars[0].get_state()
            c1 = cars[1].get_state()
            active_car = c0 if abs(c0.pos.x) > 2000.0 else c1

            if abs(active_car.pos.x) > 3900.0 and active_car.pos.z > 200.0:
                on_wall_count += 1
                rot_mat = active_car.rot_mat
                up_vec = rot_mat.up
                wall_side = 1.0 if active_car.pos.x > 0 else -1.0
                self.assertLess(up_vec.x * wall_side, -0.7, f"Car on wall must have wheels facing wall: up.x={up_vec.x}")
                speed = math.hypot(active_car.vel.y, active_car.vel.z)
                self.assertGreater(speed, 600.0, f"Car spawned on wall should have momentum, got {speed}")

        self.assertGreater(on_wall_count, 5, f"Should have generated multiple on-wall spawns out of {total_trials}, got {on_wall_count}")


if __name__ == '__main__':
    unittest.main()