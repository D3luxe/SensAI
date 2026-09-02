"""
Unit and Integration Tests for RLGym-Tools State Setters and Replay Ingestion Engine.
"""

import os
import unittest
import numpy as np
import RocketSim as rsim

from utils.replay_parser import ReplayParser
from env.state_setters import (
    KickoffSetter, AerialScenarioSetter, WallPlaySetter,
    GoalieSaveSetter, ReplayStateSetter, WeightedScenarioSetter,
    CustomScenarioSetter
)
from utils.scenario_manager import (
    ScenarioManager,
    render_scenario_visual_guide,
    simulate_custom_scenario
)
from env.physics_engine import RocketSimArena


class TestScenariosAndReplays(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")

    def test_replay_parser_and_pool(self):
        pool_path = "data/replays/test_pool.npz"
        parser = ReplayParser(pool_path=pool_path)

        # Create synthetic dataset
        N = 20
        sample_dict = {
            "ball_pos": np.zeros((N, 3), dtype=np.float32),
            "ball_vel": np.ones((N, 3), dtype=np.float32),
            "car_pos": np.zeros((N, 2, 3), dtype=np.float32),
            "car_vel": np.zeros((N, 2, 3), dtype=np.float32),
            "car_rot": np.zeros((N, 2, 3), dtype=np.float32),
            "car_boost": np.full((N, 2), 50.0, dtype=np.float32)
        }
        parser.states_buffer = sample_dict
        parser.save_pool()

        # Check stats
        stats = parser.get_pool_stats()
        self.assertEqual(stats["total_frames"], N)

        # Check sample state
        sampled = parser.sample_state(num_cars=2)
        self.assertIsNotNone(sampled)
        self.assertEqual(sampled["ball_pos"].shape, (3,))
        self.assertEqual(sampled["car_pos"].shape, (2, 3))

        # Test clear pool
        self.assertTrue(parser.clear_pool())
        stats_cleared = parser.get_pool_stats()
        self.assertEqual(stats_cleared["total_frames"], 0)
        self.assertFalse(os.path.exists(pool_path))

        # Test zip ingestion
        import zipfile
        zip_test_path = "data/replays/test_replays.zip"
        raw_test_file = "data/replays/dummy.json"
        import json
        with open(raw_test_file, "w") as f:
            json.dump({
                "ball_pos": [[0,0,100], [10,10,100]],
                "ball_vel": [[0,0,0], [0,0,0]],
                "car_pos": [[[0,0,17],[0,0,17]], [[0,0,17],[0,0,17]]],
                "car_vel": [[[0,0,0],[0,0,0]], [[0,0,0],[0,0,0]]],
                "car_rot": [[[0,0,0],[0,0,0]], [[0,0,0],[0,0,0]]],
                "car_boost": [[50,50], [50,50]]
            }, f)

        with zipfile.ZipFile(zip_test_path, "w") as z:
            z.write(raw_test_file, arcname="nested/match_1.json")

        parser_zip = ReplayParser(pool_path=pool_path)
        count, frames = parser_zip.ingest_zip(zip_test_path)
        self.assertEqual(count, 1)
        self.assertEqual(frames, 2)
        self.assertEqual(parser_zip.get_pool_stats()["total_frames"], 2)

        # Cleanup
        parser_zip.clear_pool()
        if os.path.exists(zip_test_path):
            os.remove(zip_test_path)
        if os.path.exists(raw_test_file):
            os.remove(raw_test_file)

    def test_all_state_setters(self):
        rsim_arena = self.arena._rsim_arena

        setters = [
            ("Kickoff", KickoffSetter()),
            ("Aerial", AerialScenarioSetter()),
            ("Wall", WallPlaySetter()),
            ("Goalie", GoalieSaveSetter()),
            ("Replay", ReplayStateSetter()),
            ("Custom", CustomScenarioSetter())
        ]

        for name, setter in setters:
            setter.reset(rsim_arena, num_players=2)
            self.arena._sync_from_rsim()

            # Check that ball position is within legal field bounds
            self.assertTrue(abs(self.arena.ball.pos[0]) <= 4096.0, f"{name}: Ball X out of bounds")
            self.assertTrue(abs(self.arena.ball.pos[1]) <= 5120.0, f"{name}: Ball Y out of bounds")
            self.assertTrue(0.0 <= self.arena.ball.pos[2] <= 2044.0, f"{name}: Ball Z out of bounds")

            # Check car positions
            for car in self.arena.cars:
                self.assertTrue(abs(car.pos[0]) <= 4096.0, f"{name}: Car X out of bounds")
                self.assertTrue(abs(car.pos[1]) <= 5120.0, f"{name}: Car Y out of bounds")
                self.assertTrue(0.0 <= car.boost <= 100.0, f"{name}: Boost out of range")

    def test_custom_scenario_manager_and_visual_guide(self):
        test_cfg_path = "config/test_custom_scenarios.json"
        mgr = ScenarioManager(config_path=test_cfg_path)
        mgr.reset_to_defaults()

        scenarios = mgr.get_all_scenarios()
        self.assertGreaterEqual(len(scenarios), 4, "Must have at least 4 default presets")

        # Verify Opposing 1/3rd Bouncing Ball preset
        opp_third_sc = mgr.get_scenario("opposing_third_bouncing_ball")
        self.assertIsNotNone(opp_third_sc)
        self.assertIn("Opposing 1/3rd", opp_third_sc["name"])
        self.assertEqual(opp_third_sc["car"]["pos"][1], 1800.0)
        self.assertEqual(opp_third_sc["ball"]["pos"][1], 2600.0)

        # Test Save New Scenario
        new_sc = {
            "id": "unit_test_drill",
            "name": "Unit Test Practice Drill",
            "description": "Test drill for validation.",
            "enabled": True,
            "car": {"pos": [100.0, 500.0, 17.0], "yaw": 45.0, "vel": [200.0, 200.0, 0.0], "boost": 75.0},
            "ball": {"pos": [200.0, 800.0, 150.0], "vel": [0.0, 100.0, 0.0]},
            "opponent": {"mode": "goalie", "pos": [0.0, 4800.0, 17.0], "yaw": -90.0, "boost": 50.0},
            "variance": {"pos_jitter": 20.0, "vel_jitter": 20.0, "mirror_symmetry": True}
        }
        success, msg = mgr.save_scenario(new_sc)
        self.assertTrue(success)
        self.assertIsNotNone(mgr.get_scenario("unit_test_drill"))

        # Test Visual Guide Plot Generation
        fig = render_scenario_visual_guide(new_sc)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Test Delete Scenario
        del_success, del_msg = mgr.delete_scenario("unit_test_drill")
        self.assertTrue(del_success)
        self.assertIsNone(mgr.get_scenario("unit_test_drill"))

        if os.path.exists(test_cfg_path):
            os.remove(test_cfg_path)

    def test_weighted_scenario_setter(self):
        weighted_setter = WeightedScenarioSetter(
            kickoff_prob=0.15,
            replay_prob=0.15,
            aerial_prob=0.15,
            wall_prob=0.15,
            save_prob=0.15,
            turnaround_prob=0.10,
            custom_prob=0.15
        )
        rsim_arena = self.arena._rsim_arena

        sampled_scenarios = set()
        for _ in range(70):
            scenario_name = weighted_setter.reset(rsim_arena, num_players=2)
            sampled_scenarios.add(scenario_name)

        # Should sample multiple distinct scenarios including custom
        self.assertGreater(len(sampled_scenarios), 1, "Weighted setter must sample multiple scenario types")

    def test_arena_dynamic_scenario_weights(self):
        self.arena.set_scenario_weights({
            "aerial_prob": 1.0,
            "kickoff_prob": 0.0,
            "replay_prob": 0.0,
            "wall_prob": 0.0,
            "turnaround_prob": 0.0,
            "save_prob": 0.0,
            "custom_prob": 0.0
        })
        self.arena.reset(random_kickoff=True)
        # In pure aerial mode, ball should be high up in the air
        self.assertGreater(self.arena.ball.pos[2], 300.0, "100% Aerial scenario must spawn elevated ball")

        # Test 100% Custom Scenarios Mode
        self.arena.set_scenario_weights({
            "aerial_prob": 0.0,
            "kickoff_prob": 0.0,
            "replay_prob": 0.0,
            "wall_prob": 0.0,
            "turnaround_prob": 0.0,
            "save_prob": 0.0,
            "custom_prob": 1.0
        })
        self.arena.reset(random_kickoff=True)
        self.assertTrue(abs(self.arena.ball.pos[0]) <= 4096.0)
        self.assertTrue(abs(self.arena.cars[0].pos[0]) <= 4096.0)

    def test_behavioral_cloning_pretrainer(self):
        from agent.pretrainer import BehavioralCloningTrainer
        test_pool = "data/replays/test_bc_pool.npz"
        test_ckpt = "checkpoints/test_bc_model.pt"

        # Create synthetic replay frames
        parser = ReplayParser(pool_path=test_pool)
        N = 30
        parser.states_buffer = {
            "ball_pos": np.random.uniform(-1000, 1000, size=(N, 3)).astype(np.float32),
            "ball_vel": np.random.normal(0, 500, size=(N, 3)).astype(np.float32),
            "car_pos": np.zeros((N, 2, 3), dtype=np.float32),
            "car_vel": np.zeros((N, 2, 3), dtype=np.float32),
            "car_rot": np.zeros((N, 2, 3), dtype=np.float32),
            "car_boost": np.full((N, 2), 50.0, dtype=np.float32)
        }
        parser.save_pool()

        trainer = BehavioralCloningTrainer(pool_path=test_pool, checkpoint_path=test_ckpt)
        status = trainer.train(epochs=2, batch_size=16, lr=0.001)

        self.assertFalse(trainer.is_running())
        self.assertEqual(status["epoch"], 2)
        self.assertTrue(os.path.exists(test_ckpt))

        # Cleanup
        if os.path.exists(test_pool):
            os.remove(test_pool)
        if os.path.exists(test_ckpt):
            os.remove(test_ckpt)

    def test_inverse_dynamics_solver(self):
        from utils.inverse_dynamics import InverseDynamicsSolver

        # 1. Test Ground Throttle & Boost Extraction
        p_t = np.array([0.0, -3000.0, 17.0], dtype=np.float32)
        v_t = np.array([0.0, 500.0, 0.0], dtype=np.float32)
        r_t = np.array([0.0, np.pi / 2, 0.0], dtype=np.float32)  # Facing +Y
        b_t = 100.0

        p_next = np.array([0.0, -2980.0, 17.0], dtype=np.float32)
        v_next = np.array([0.0, 600.0, 0.0], dtype=np.float32)  # Forward accel
        r_next = np.array([0.0, np.pi / 2, 0.0], dtype=np.float32)
        b_next = 98.0  # Consumed boost

        act = InverseDynamicsSolver.solve_car_action(
            p_t, v_t, r_t, np.zeros(3), b_t, True,
            p_next, v_next, r_next, np.zeros(3), b_next, True,
            dt=1.0 / 30.0
        )
        self.assertGreater(act[0], 0.5, "Forward acceleration must yield positive throttle act[0] > 0.5")
        self.assertEqual(act[6], 1.0, "Boost consumption must yield act[6] = 1.0")

        # 2. Test Airborne Pitch Down Extraction (Front-Flip)
        p_air_t = np.array([0.0, 0.0, 500.0], dtype=np.float32)
        v_air_t = np.array([0.0, 1000.0, 0.0], dtype=np.float32)
        r_air_t = np.array([0.0, np.pi / 2, 0.0], dtype=np.float32)

        p_air_next = np.array([0.0, 33.0, 480.0], dtype=np.float32)
        v_air_next = np.array([0.0, 1500.0, -50.0], dtype=np.float32)
        r_air_next = np.array([-0.3, np.pi / 2, 0.0], dtype=np.float32)  # Pitch nose down

        act_air = InverseDynamicsSolver.solve_car_action(
            p_air_t, v_air_t, r_air_t, np.zeros(3), 50.0, False,
            p_air_next, v_air_next, r_air_next, np.zeros(3), 50.0, False,
            dt=1.0 / 30.0
        )
        self.assertLess(act_air[2], -0.2, "Pitching down must yield negative pitch act[2] < -0.2")

        # 3. Test Batch Extraction
        c_pos_seq = np.stack([p_t, p_next], axis=0)
        c_vel_seq = np.stack([v_t, v_next], axis=0)
        c_rot_seq = np.stack([r_t, r_next], axis=0)
        c_bst_seq = np.array([b_t, b_next], dtype=np.float32)

        batch_acts = InverseDynamicsSolver.batch_extract_actions(c_pos_seq, c_vel_seq, c_rot_seq, c_bst_seq)
        self.assertEqual(batch_acts.shape, (1, 8))
        self.assertGreater(batch_acts[0, 0], 0.5)


if __name__ == "__main__":
    unittest.main()


