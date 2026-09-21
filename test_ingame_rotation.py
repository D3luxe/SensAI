"""
The in-game bot's orientation conversion against RocketSim's, the frame the policy trained in.

Every other in-game test builds its packet with angular velocity and rotation at zero, so none
of them could see a disagreement in the rotation basis. This one checks it directly: bot.py
rebuilds the car's basis from the packet's pitch/yaw/roll, and that basis has to equal what
RocketSim.Angle.as_rot_mat() gives for the same angles.

Found 2026-09-21: row 1 was forward x up, the reverse of RocketSim's in every orientation.
Nothing read it, so the policy was unaffected, but it was a trap for the next reader.
"""
import math
import unittest

import numpy as np

try:
    import RocketSim as rs
except Exception:          # the simulator is optional for the rest of the suite
    rs = None


def rocketsim_basis(pitch, yaw, roll):
    a = rs.Angle()
    a.yaw, a.pitch, a.roll = yaw, pitch, roll
    return np.asarray(a.as_rot_mat().as_numpy(), dtype=np.float64)


def random_angles(n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield (rng.uniform(-1.55, 1.55), rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi))


@unittest.skipIf(rs is None, "RocketSim is not installed")
class TestBotBasisMatchesRocketSim(unittest.TestCase):
    def setUp(self):
        from bot import rotation_to_rot_mat
        self.convert = rotation_to_rot_mat

    def test_every_row_matches_in_every_orientation(self):
        for p, y, r in random_angles(2000):
            ours, truth = self.convert(p, y, r), rocketsim_basis(p, y, r)
            for row, name in enumerate(("forward", "right", "up")):
                np.testing.assert_allclose(ours[row], truth[row], atol=1e-5,
                                           err_msg=f"{name} row at pitch={p:.3f} yaw={y:.3f} roll={r:.3f}")

    def test_it_is_a_rotation_not_a_reflection(self):
        """The old row 1 made the determinant -1: a mirror image of the car, not an orientation."""
        for p, y, r in random_angles(200, seed=1):
            self.assertAlmostEqual(float(np.linalg.det(self.convert(p, y, r))), 1.0, places=4)

    def test_right_is_up_cross_forward_because_the_frame_is_left_handed(self):
        m = self.convert(0.0, 0.0, 0.0)                  # facing +x
        np.testing.assert_allclose(m[1], np.cross(m[2], m[0]), atol=1e-6)
        np.testing.assert_allclose(m[1], [0.0, 1.0, 0.0], atol=1e-6)


@unittest.skipIf(rs is None, "RocketSim is not installed")
class TestTheRightSideIsWhereARightTurnGoes(unittest.TestCase):
    """The physical fact the fix rests on, checked in the simulator rather than asserted."""

    def test_steer_plus_one_curves_toward_row_one(self):
        from env.physics_engine import RocketSimArena
        from bot import rotation_to_rot_mat
        arena = RocketSimArena()
        arena.reset()
        if not getattr(arena, "_use_rsim", False):
            self.skipTest("arena is not running RocketSim")
        car = arena._rsim_cars[0]
        st = car.get_state()
        st.pos, st.vel, st.ang_vel = rs.Vec(0, 0, 17), rs.Vec(0, 0, 0), rs.Vec(0, 0, 0)
        a = rs.Angle()
        a.yaw = a.pitch = a.roll = 0.0
        st.rot_mat = a.as_rot_mat()
        car.set_state(st)
        car.set_controls(rs.CarControls(throttle=1.0, steer=1.0))   # a right turn, in-game
        for _ in range(60):
            arena._rsim_arena.step(1)
        p = car.get_state().pos.as_numpy()
        right = rotation_to_rot_mat(0.0, 0.0, 0.0)[1]
        self.assertGreater(float(np.dot(p[:2], right[:2])), 0.0)


class TestTheMirroredRightStaysAsTrained(unittest.TestCase):
    """
    The policy's own convention is the mirror image: the observation builder and
    CarState.get_right_vector use forward x up, and SensAI's steer is negated on the way to
    the game in both training and deployment. That is consistent end to end, and changing it
    would silently invert every lateral feature the trained policy relies on.
    """

    def test_get_right_vector_is_still_forward_cross_up(self):
        from env.physics_engine import CarState
        from bot import rotation_to_rot_mat
        car = CarState(id=0, team=0) if "id" in CarState.__dataclass_fields__ else CarState()
        car.rot_mat = rotation_to_rot_mat(0.2, 0.7, 0.3)
        np.testing.assert_allclose(car.get_right_vector(),
                                   np.cross(car.rot_mat[0], car.rot_mat[2]), atol=1e-6)


@unittest.skipIf(rs is None, "RocketSim is not installed")
class TestStateSetterBasis(unittest.TestCase):
    """
    env/state_setters.py carries its own copy of the conversion with the same reversed row.
    Also unread, and left alone because it is training code and a run is live. Marked as a
    known failure so it reports an unexpected success the day it is fixed.
    """

    @unittest.expectedFailure
    def test_state_setter_right_row_matches_rocketsim(self):
        from env.state_setters import rotation_to_rot_mat
        for p, y, r in random_angles(50, seed=2):
            np.testing.assert_allclose(rotation_to_rot_mat(p, y, r)[1], rocketsim_basis(p, y, r)[1], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
