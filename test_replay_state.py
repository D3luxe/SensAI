"""Reconstruction of unrecorded replay state (utils/replay_state.py) on hand-built pools."""
import math
import unittest

import numpy as np

from env.physics_engine import BoostPad
from utils.replay_state import ReplayStateReconstructor

DT = 1.0 / 30.0


def make_pool(n, segments=None):
    """A 2-car pool: blue parked on the floor at (-1000, -2000), orange at (1000, 2000), ball resting off-centre."""
    t = np.arange(n) * DT
    car_pos = np.zeros((n, 2, 3), np.float32)
    car_pos[:, 0] = [-1000.0, -2000.0, 17.0]
    car_pos[:, 1] = [1000.0, 2000.0, 17.0]
    car_rot = np.zeros((n, 2, 3), np.float32)
    car_rot[:, 0, 1] = math.pi / 2
    car_rot[:, 1, 1] = -math.pi / 2
    return {
        "ball_pos": np.tile(np.array([500.0, 500.0, 93.0], np.float32), (n, 1)),
        "ball_vel": np.zeros((n, 3), np.float32),
        "car_pos": car_pos,
        "car_vel": np.zeros((n, 2, 3), np.float32),
        "car_rot": car_rot,
        "car_boost": np.full((n, 2), 33.0, np.float32),
        "car_ang_vel": np.zeros((n, 2, 3), np.float32),
        "frame_time": t,
        "car_time": np.stack([t, t], axis=1),
        "segment_id": np.zeros(n, np.int32) if segments is None else segments,
    }


class TestTouchSinceKickoff(unittest.TestCase):
    def test_untouched_from_kickoff_pause_until_the_ball_moves(self):
        d = make_pool(60)
        d["ball_pos"][:30] = [0.0, 0.0, 93.0]          # resting at centre
        d["ball_pos"][30:] = [0.0, 300.0, 93.0]        # hit
        d["ball_vel"][30:] = [0.0, 900.0, 0.0]
        r = ReplayStateReconstructor(d)
        self.assertTrue(r.untouched[:30].all())
        self.assertFalse(r.untouched[30:].any())

    def test_segment_without_a_kickoff_counts_as_touched(self):
        r = ReplayStateReconstructor(make_pool(20))
        self.assertFalse(r.untouched.any())


class TestPads(unittest.TestCase):
    def test_small_pickup_disables_nearest_small_pad_for_four_seconds(self):
        pads = BoostPad.create_standard_pads()
        k = next(i for i, p in enumerate(pads) if not p.is_big)
        d = make_pool(200)
        d["car_pos"][:, 0, :2] = pads[k].pos[:2]
        d["car_boost"][10:, 0] = 45.0                  # +12 at frame 10
        r = ReplayStateReconstructor(d)
        active, cooldown = r.pads_at(10)
        self.assertFalse(active[k])
        self.assertAlmostEqual(float(cooldown[k]), 4.0, places=3)
        self.assertEqual(int((~active).sum()), 1)
        active, _ = r.pads_at(10 + int(4.0 / DT) + 2)
        self.assertTrue(active[k])

    def test_full_refill_is_a_big_pad(self):
        pads = BoostPad.create_standard_pads()
        k = next(i for i, p in enumerate(pads) if p.is_big)
        d = make_pool(40)
        d["car_pos"][:, 0, :2] = pads[k].pos[:2]
        d["car_boost"][10:, 0] = 100.0
        active, cooldown = ReplayStateReconstructor(d).pads_at(20)
        self.assertFalse(active[k])
        self.assertAlmostEqual(float(cooldown[k]), 10.0 - 10 * DT, places=3)

    def test_kickoff_resets_pads(self):
        pads = BoostPad.create_standard_pads()
        k = next(i for i, p in enumerate(pads) if p.is_big)
        d = make_pool(60)
        d["car_pos"][:, 0, :2] = pads[k].pos[:2]
        d["car_boost"][10:, 0] = 100.0
        d["ball_pos"][30:] = [0.0, 0.0, 93.0]          # goal reset: kickoff pause at frame 30
        active, _ = ReplayStateReconstructor(d).pads_at(40)
        self.assertTrue(active.all())


class TestMechanics(unittest.TestCase):
    def _airborne_pool(self):
        d = make_pool(40)
        # Blue leaves the floor at frame 5 and stays high in the air.
        d["car_pos"][5:, 0, 2] = 400.0
        return d

    def test_grounded_car_holds_jump_not_flip(self):
        f = ReplayStateReconstructor(make_pool(10)).car_fields(5, 0)
        self.assertTrue(f["on_ground"])
        self.assertTrue(f["has_jump"])
        self.assertFalse(f["has_flip"])
        self.assertEqual(f["air_timer"], 0.0)

    def test_airborne_car_has_flip_and_counts_air_time(self):
        f = ReplayStateReconstructor(self._airborne_pool()).car_fields(20, 0)
        self.assertFalse(f["on_ground"])
        self.assertFalse(f["has_jump"])
        self.assertTrue(f["has_flip"])
        self.assertAlmostEqual(f["air_timer"], 16 * DT, places=4)

    def test_spin_kick_is_a_dodge_that_spends_the_flip(self):
        d = self._airborne_pool()
        d["car_ang_vel"][15:, 0] = [0.0, 550.0, 0.0]
        r = ReplayStateReconstructor(d)
        before, during, after = r.car_fields(14, 0), r.car_fields(20, 0), r.car_fields(39, 0)
        self.assertTrue(before["has_flip"])
        self.assertFalse(during["has_flip"])
        self.assertTrue(during["is_dodging"])
        self.assertAlmostEqual(during["flip_timer"], 6 * DT, places=4)
        self.assertFalse(after["is_dodging"])          # 25 frames > flip animation

    def test_touchdown_restores_jump(self):
        d = self._airborne_pool()
        d["car_ang_vel"][15:25, 0] = [0.0, 550.0, 0.0]
        d["car_pos"][30:, 0, 2] = 17.0
        f = ReplayStateReconstructor(d).car_fields(35, 0)
        self.assertTrue(f["on_ground"])
        self.assertTrue(f["has_jump"])
        self.assertFalse(f["is_dodging"])

    def test_unreplicated_frame_reads_last_replicated_state(self):
        d = self._airborne_pool()
        d["car_time"][21:, 0] = d["frame_time"][20]    # blue not replicated after frame 20
        d["car_ang_vel"][25:, 0] = [0.0, 550.0, 0.0]     # would be a dodge, but never replicated
        f = ReplayStateReconstructor(d).car_fields(30, 0)
        self.assertTrue(f["has_flip"])


class TestSyntheticOpponent(unittest.TestCase):
    def test_mirrored_stand_in_is_flagged_and_real_kickoff_is_not(self):
        d = make_pool(4)
        # Frame 0: parser's mirror of a lone car (yaw + pi, unwrapped).
        d["car_pos"][0, 1] = [1000.0, 2000.0, 17.0]
        d["car_rot"][0, 1] = [0.0, math.pi / 2 + math.pi, 0.0]
        # Frame 1: a real kickoff, positions mirrored but the orange yaw is a wrapped replicated angle.
        d["car_pos"][1, 0] = [-2048.0, -2560.0, 17.0]
        d["car_pos"][1, 1] = [2048.0, 2560.0, 17.0]
        d["car_rot"][1, 0] = [0.0, math.pi / 4, 0.0]
        d["car_rot"][1, 1] = [0.0, -3 * math.pi / 4, 0.0]
        r = ReplayStateReconstructor(d)
        self.assertTrue(r.synthetic_opponent[0])
        self.assertFalse(r.synthetic_opponent[1])


if __name__ == "__main__":
    unittest.main()
