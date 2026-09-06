import unittest
import math
import numpy as np

from env.physics_engine import CarState, BallState
from env.rewards import AirRollRecoveryReward


class MockArena:
    def __init__(self, ball_pos=None, ball_vel=None):
        self.ball = BallState(
            pos=np.array(ball_pos if ball_pos is not None else [0.0, 2000.0, 93.0], dtype=np.float32),
            vel=np.array(ball_vel if ball_vel is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        )
        self.cars = []


class TestAirRollRecoveryDamping(unittest.TestCase):

    def setUp(self):
        self.arena = MockArena()

    def test_roll_rate_damping_settling_bonus(self):
        """Test that an uprighting car with damped roll rate earns a stabilization bonus."""
        rew = AirRollRecoveryReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 400.0], dtype=np.float32),
            vel=np.array([0.0, 500.0, -100.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # up_z = 1.0, forward = [1, 0, 0]
            on_ground=False
        )
        car.ang_vel = np.array([0.2, 0.0, 0.0], dtype=np.float32)  # Damped roll rate = 0.2 rad/s
        self.arena.cars = [car]
        rew.reset(self.arena)

        rew._was_disoriented[car.id] = True
        rew._prev_up_z[car.id] = 0.88  # prev_up_z < 0.90 entering flat attitude
        rew._prev_on_ground[car.id] = False

        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)

        self.assertGreater(r, 0.05, f"Damped roll rate near flat attitude must award settling bonus, got {r}")

    def test_excess_roll_momentum_penalty(self):
        """Test that continuing to spin violently near upright attitude incurs an excess momentum penalty."""
        rew = AirRollRecoveryReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 400.0], dtype=np.float32),
            vel=np.array([0.0, 500.0, -100.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # up_z = 1.0, forward = [1, 0, 0]
            on_ground=False
        )
        car.ang_vel = np.array([5.0, 0.0, 0.0], dtype=np.float32)  # Violent roll rate = 5.0 rad/s
        self.arena.cars = [car]
        rew.reset(self.arena)

        rew._was_disoriented[car.id] = True
        rew._prev_up_z[car.id] = 0.92  # prev_up_z >= 0.90 so delta_up reward is shut off
        rew._prev_on_ground[car.id] = False

        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)

        self.assertLess(r, 0.0, f"Violent roll spin at upright attitude must incur excess momentum penalty, got {r}")

    def test_upright_roll_input_suppression(self):
        """Test that holding roll input when already upright in the air incurs a release penalty."""
        rew = AirRollRecoveryReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 500.0], dtype=np.float32),
            vel=np.array([0.0, 800.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # up_z = 1.0
            on_ground=False
        )
        self.arena.cars = [car]
        rew.reset(self.arena)
        rew._airborne_ticks[car.id] = 25  # Beyond half-flip window

        # Action with roll = 0.8
        act_roll = np.zeros(8, dtype=np.float32)
        act_roll[4] = 0.8
        r_roll = rew.get_reward(car, self.arena, act_roll, False, None)

        # Action with roll = 0.0
        act_neutral = np.zeros(8, dtype=np.float32)
        r_neutral = rew.get_reward(car, self.arena, act_neutral, False, None)

        self.assertLess(r_roll, r_neutral, "Holding roll input when upright airborne must be penalized compared to neutral")
        self.assertLess(r_roll, -0.04, f"Holding roll=0.8 when upright should yield penalty < -0.04, got {r_roll}")
        self.assertEqual(r_neutral, 0.0, f"Neutral roll when upright should yield 0.0, got {r_neutral}")

    def test_stabilized_recovery_exit_condition(self):
        """Test that recovery state does NOT clear if the car is upright but still spinning rapidly."""
        rew = AirRollRecoveryReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 400.0], dtype=np.float32),
            vel=np.array([0.0, 500.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),  # Heading aligned with +Y vel
            on_ground=False
        )
        self.arena.cars = [car]
        rew.reset(self.arena)

        # Case A: Upright but spinning rapidly (roll_rate = 3.5 rad/s)
        car.ang_vel = np.array([0.0, 3.5, 0.0], dtype=np.float32)  # forward is +Y, so ang_vel[1] is roll rate
        rew._was_disoriented[car.id] = True
        rew._prev_up_z[car.id] = 0.95
        rew._prev_heading[car.id] = 0.95
        action = np.zeros(8, dtype=np.float32)

        rew.get_reward(car, self.arena, action, False, None)
        self.assertTrue(
            rew._was_disoriented[car.id],
            "Car upright but spinning with roll_rate=3.5 must NOT exit recovery disorientation"
        )

        # Case B: Upright and settled (roll_rate = 0.5 rad/s)
        car.ang_vel = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        rew.get_reward(car, self.arena, action, False, None)
        self.assertFalse(
            rew._was_disoriented[car.id],
            "Car upright and damped with roll_rate=0.5 must cleanly conclude recovery disorientation"
        )

    def test_touchdown_side_landing_penalty(self):
        """Test that landing on side/door (0.0 <= up_z < 0.65) is penalized rather than ignored."""
        rew = AirRollRecoveryReward(weight=1.0)
        # Door landing: roll = 75 degrees (cos(75 deg) approx 0.259)
        car_door = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 400.0, 0.0], dtype=np.float32),
            rot=np.array([75.0 * math.pi / 180.0, 0.0, 0.0], dtype=np.float32),
            on_ground=True
        )
        self.arena.cars = [car_door]
        rew.reset(self.arena)
        rew._prev_on_ground[car_door.id] = False
        rew._disoriented_this_flight[car_door.id] = True

        action = np.zeros(8, dtype=np.float32)
        r_door = rew.get_reward(car_door, self.arena, action, False, None)

        self.assertLess(
            r_door, -0.10,
            f"Landing on door (up_z={car_door.get_up_vector()[2]:.2f}) must receive side crash penalty, got {r_door}"
        )

    def test_touchdown_spinning_penalty(self):
        """Test that landing on wheels while spinning rapidly incurs a touchdown spin penalty."""
        rew = AirRollRecoveryReward(weight=1.0)

        # Wheels down, stationary spin
        car_settled = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 400.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        car_settled.ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.arena.cars = [car_settled]
        rew.reset(self.arena)
        rew._prev_on_ground[car_settled.id] = False
        rew._disoriented_this_flight[car_settled.id] = True

        action = np.zeros(8, dtype=np.float32)
        r_settled = rew.get_reward(car_settled, self.arena, action, False, None)

        # Wheels down, rapid spin (ang_vel = 4.0 rad/s)
        car_spinning = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 400.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=True
        )
        car_spinning.ang_vel = np.array([4.0, 0.0, 0.0], dtype=np.float32)

        self.arena.cars = [car_spinning]
        rew.reset(self.arena)
        rew._prev_on_ground[car_spinning.id] = False
        rew._disoriented_this_flight[car_spinning.id] = True

        r_spinning = rew.get_reward(car_spinning, self.arena, action, False, None)

        self.assertGreater(r_settled, 0.5, f"Settled touchdown must award clean landing bonus, got {r_settled}")
        self.assertLess(r_spinning, r_settled, "Spinning touchdown must receive lower reward due to spin penalty")
        self.assertAlmostEqual(r_settled - r_spinning, 0.30, delta=0.05, msg="Spin penalty should reduce touchdown reward by ~0.30")

    def test_pitch_tumble_denied_settling_bonus(self):
        """Test that a car upright in roll but violently tumbling in pitch (ang_vel > 1.8) is denied settling bonus."""
        rew = AirRollRecoveryReward(weight=1.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 400.0], dtype=np.float32),
            vel=np.array([0.0, 500.0, -100.0], dtype=np.float32),
            rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # up_z = 1.0, forward = [1, 0, 0]
            on_ground=False
        )
        # Roll rate is 0.0 (fwd is [1,0,0], ang_vel is [0, 3.5, 0] along lateral pitch axis)
        car.ang_vel = np.array([0.0, 3.5, 0.0], dtype=np.float32)
        self.arena.cars = [car]
        rew.reset(self.arena)

        rew._was_disoriented[car.id] = True
        rew._prev_up_z[car.id] = 0.95  # prev_up_z >= 0.90 so Section 1a (delta_up) does not fire
        rew._prev_on_ground[car.id] = False

        action = np.zeros(8, dtype=np.float32)
        r = rew.get_reward(car, self.arena, action, False, None)

        self.assertEqual(r, 0.0, f"Pitch tumbling car (ang_speed=3.5) must NOT claim settling bonus, got {r}")

    def test_monotonic_touchdown_crash_penalty(self):
        """Test that crash penalty is strictly monotonic: flat roof (up_z=-1) < inverted roof (up_z=-0.1) < door (up_z=0.0) < tilted door (up_z=0.3)."""
        rew = AirRollRecoveryReward(weight=1.0)

        def eval_landing(up_val: float) -> float:
            # Set up_vector directly via rot_mat:
            # up_val is in [-1.0, 1.0]. Let up_vector = [0, sqrt(1 - up_val^2), up_val]
            uy = math.sqrt(max(0.0, 1.0 - up_val * up_val))
            rot_mat = np.array([
                [1.0, 0.0, 0.0],
                [0.0, up_val, -uy],
                [0.0, uy, up_val]
            ], dtype=np.float32)
            car = CarState(
                id=0, team=0,
                pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
                vel=np.array([0.0, 400.0, 0.0], dtype=np.float32),
                rot_mat=rot_mat,
                on_ground=True
            )
            self.arena.cars = [car]
            rew.reset(self.arena)
            rew._prev_on_ground[car.id] = False
            rew._disoriented_this_flight[car.id] = True
            return rew.get_reward(car, self.arena, np.zeros(8, dtype=np.float32), False, None)

        r_inverted_full = eval_landing(-1.0)
        r_inverted_slight = eval_landing(-0.1)
        r_door = eval_landing(0.0)
        r_tilted = eval_landing(0.3)

        self.assertLess(r_inverted_full, r_inverted_slight, "Flat roof crash must be worse than slight roof tilt")
        self.assertLess(r_inverted_slight, r_door, "Inverted roof crash must be strictly worse than door landing (no cliff at 0)")
        self.assertLess(r_door, r_tilted, "Door landing must be worse than tilted landing")
        self.assertLess(r_tilted, 0.0, "Tilted landing (up_z=0.3) must still be penalized")


if __name__ == "__main__":
    unittest.main()
