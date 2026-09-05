"""
Unit Tests for Phase 3: Opponent Contract, Test Runner Truthfulness, and Replay Engine.
Verifies Fixes #4, #2, #10, and #1 replay rejection.
"""

import unittest
import os
import tempfile
import numpy as np
import torch

from env.physics_engine import CarState, BallState, RocketSimArena
from env.baseline_agent import NectoNextoOpponentBot
from utils.test_runner import get_cached_or_run_tests, CACHE_FILE
from utils.replay_parser import ReplayParser


class MockArenaForBot:
    def __init__(self, cars, ball=None):
        self.cars = cars
        self.ball = ball or BallState()
        self.boost_pads = []


class TestBaselineAndReplays(unittest.TestCase):
    def test_necto_and_nexto_observation_contract(self):
        """Fix #4: In both Necto and Nexto inputs, self is marked teammate, and Orange does not invert opponent/teammate."""
        bot = NectoNextoOpponentBot.__new__(NectoNextoOpponentBot)
        bot.device = torch.device("cpu")
        bot.prev_action = np.zeros(8, dtype=np.float32)

        # 2 players: Blue car (0) and Orange car (1)
        blue_car = CarState(id=0, team=0, pos=np.array([0, -2000, 17], dtype=np.float32))
        orange_car = CarState(id=1, team=1, pos=np.array([0, 2000, 17], dtype=np.float32))
        arena = MockArenaForBot([blue_car, orange_car])

        # 1. Test Necto inputs when Orange is the observing car
        q, kv, m = bot._build_necto_inputs(orange_car, arena)
        kv_np = kv.cpu().numpy()

        # In Necto: Entity 0 is ball, Entity 1 is Blue car, Entity 2 is Orange car (self)
        # Check Blue car from Orange perspective: is_opponent (index 2) must be 1.0, NOT is_teammate (index 1)
        blue_entity = kv_np[0, 1]
        self.assertEqual(blue_entity[0], 0.0, "Blue car is not self")
        self.assertEqual(blue_entity[1], 0.0, "Blue car is NOT teammate of Orange")
        self.assertEqual(blue_entity[2], 1.0, "Blue car MUST be opponent of Orange")

        # Check Orange car from Orange perspective: is_self=1.0 AND is_teammate=1.0
        orange_entity = kv_np[0, 2]
        self.assertEqual(orange_entity[0], 1.0, "Orange car is self")
        self.assertEqual(orange_entity[1], 1.0, "Orange car is its own teammate per EARL spec")
        self.assertEqual(orange_entity[2], 0.0, "Orange car is not opponent of self")

        # 2. Test Nexto inputs when Orange is the observing car
        q_nx, kv_nx, m_nx = bot._build_nexto_inputs(orange_car, arena)
        kv_nx_np = kv_nx.cpu().numpy()

        # In Nexto: Entities 0..N-1 are players
        # Entity 0 is Blue car, Entity 1 is Orange car
        blue_nx = kv_nx_np[0, 0]
        self.assertEqual(blue_nx[0], 0.0, "Blue car is not self in Nexto")
        self.assertEqual(blue_nx[1], 0.0, "Blue car is NOT teammate of Orange in Nexto")
        self.assertEqual(blue_nx[2], 1.0, "Blue car MUST be opponent of Orange in Nexto")

        orange_nx = kv_nx_np[0, 1]
        self.assertEqual(orange_nx[0], 1.0, "Orange car is self in Nexto")
        self.assertEqual(orange_nx[1], 1.0, "Orange car is its own teammate per EARL spec in Nexto")
        self.assertEqual(orange_nx[2], 0.0, "Orange car is not opponent in Nexto")

    def test_test_runner_not_run_when_cache_absent(self):
        """Fix #2: get_cached_or_run_tests() must return NOT_RUN and all_passed=False when cache is absent."""
        import utils.test_runner as tr
        old_mem_cache = tr._LATEST_TEST_RESULTS_CACHE
        tr._LATEST_TEST_RESULTS_CACHE = None

        cache_backup = None
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache_backup = f.read()
            os.remove(CACHE_FILE)

        try:
            res = get_cached_or_run_tests(force_refresh=False)
            self.assertEqual(res["status"], "NOT_RUN")
            self.assertEqual(res["total_tests"], 0)
            self.assertEqual(res["passed"], 0)
            self.assertFalse(res["all_passed"])
            self.assertEqual(len(res["subsystems"]), 0)
        finally:
            tr._LATEST_TEST_RESULTS_CACHE = old_mem_cache
            if cache_backup is not None:
                with open(CACHE_FILE, "w") as f:
                    f.write(cache_backup)

    def test_replay_ingestion_rejection(self):
        """Fix #1: Unparseable / dummy .replay files must be rejected, not converted to 50 synthetic frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_replay = os.path.join(tmpdir, "corrupt.replay")
            with open(dummy_replay, "w") as f:
                f.write("This is not a Rocket League replay file.")

            parser = ReplayParser(pool_path=os.path.join(tmpdir, "pool.npz"), demo_dir=tmpdir)
            res = parser.ingest_directory(directory=tmpdir)

            self.assertEqual(res["parsed_files"], 0, "Corrupt .replay file must be rejected (0 parsed)")
            self.assertEqual(res["total_frames"], 0, "Corrupt .replay file must yield 0 frames, not synthetic data")
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "pool.npz")), "Pool file must not be created from corrupt replay")

    def test_replay_parser_api_methods(self):
        """Fix #10: ReplayParser must support demo_dir, scan_demos, and ingest_directory default behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid .json dataset file
            dummy_json = os.path.join(tmpdir, "test_dataset.json")
            import json
            data = {
                "ball_pos": [[0.0, 0.0, 100.0]],
                "ball_vel": [[0.0, 0.0, 0.0]],
                "car_pos": [[[0.0, -1000.0, 17.0], [0.0, 1000.0, 17.0]]],
                "car_vel": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                "car_rot": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                "car_boost": [[33.0, 33.0]]
            }
            with open(dummy_json, "w") as f:
                json.dump(data, f)

            pool_file = os.path.join(tmpdir, "pool.npz")
            parser = ReplayParser(pool_path=pool_file, demo_dir=tmpdir)

            # Test scan_demos
            files = parser.scan_demos(max_replays=10, sort="newest")
            self.assertEqual(len(files), 1)
            self.assertEqual(os.path.basename(files[0]), "test_dataset.json")

            # Test ingest_directory with default directory=None (should use parser.demo_dir)
            res = parser.ingest_directory(sort="newest")
            self.assertEqual(res["parsed_files"], 1)
            self.assertEqual(res["total_frames"], 1)
            self.assertIn("elapsed_seconds", res)
            self.assertTrue(os.path.exists(pool_file))


if __name__ == "__main__":
    unittest.main()
