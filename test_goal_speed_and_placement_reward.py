import unittest
import math
import numpy as np
from env.physics_engine import (
    RocketSimArena, CarState, BallState,
    GOAL_HALF_WIDTH, GOAL_HEIGHT, ARENA_EXTENT_Y
)
from env.rewards import GoalReward, CombinedReward, _norm3, _clip

try:
    import RocketSim as rsim
    ROCKETSIM_AVAILABLE = True
except ImportError:
    ROCKETSIM_AVAILABLE = False


class TestGoalSpeedAndPlacementReward(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2)
        self.arena.reset(random_kickoff=False)

    def test_speed_scaling_formula_and_continuity(self):
        """Verify two-stage piecewise linear speed scaling and its continuity across 0..3000 uu/s."""
        rew = GoalReward(goal_weight=5.0, concede_weight=-5.0, save_weight=2.0)
        rew.reset(self.arena)

        def eval_speed(v_mag):
            self.arena.last_goal_ball_pos = np.array([0.0, ARENA_EXTENT_Y, 100.0], dtype=np.float32)
            self.arena.last_goal_ball_vel = np.array([0.0, v_mag, 0.0], dtype=np.float32)
            # Car 0 scores, opponent car 1 is at midfield (open net -> s_placement = 1.0)
            self.arena.cars[0].team = 0
            self.arena.cars[1].team = 1
            self.arena.cars[1].pos = np.array([0.0, 0.0, 17.0], dtype=np.float32)
            # R = 5.0 * (1.0 + 0.5 * s_speed + 0.5 * 1.0) = 5.0 * (1.5 + 0.5 * s_speed)
            # s_speed = (R / 5.0 - 1.5) / 0.5
            r = rew.get_reward(self.arena.cars[0], self.arena, np.zeros(8), is_goal=True, scoring_team=0)
            return (r / 5.0 - 1.5) / 0.5

        # Key inflection points
        self.assertAlmostEqual(eval_speed(0.0), 0.00, places=4)
        self.assertAlmostEqual(eval_speed(500.0), 0.05, places=4)
        self.assertAlmostEqual(eval_speed(1000.0), 0.10, places=4)
        self.assertAlmostEqual(eval_speed(1750.0), 0.55, places=4)
        self.assertAlmostEqual(eval_speed(2500.0), 1.00, places=4)
        self.assertAlmostEqual(eval_speed(3200.0), 1.00, places=4)

        # Monotonicity test: strictly increasing from 0 to 2500
        speeds = np.linspace(0.0, 2500.0, 51)
        factors = [eval_speed(v) for v in speeds]
        for i in range(len(factors) - 1):
            self.assertGreater(factors[i + 1], factors[i], f"Speed factor must be strictly increasing at v={speeds[i]}")

    def test_open_net_early_exit(self):
        """When no defenders are within 2000 uu of the target goal, s_placement must be 1.0."""
        rew = GoalReward(goal_weight=5.0, concede_weight=-5.0, save_weight=2.0)
        rew.reset(self.arena)

        # Scoring team = 0 (Orange net at Y = +5120). Defender (team 1) at midfield Y = 0
        self.arena.cars[0].team = 0
        self.arena.cars[1].team = 1
        self.arena.cars[1].pos = np.array([0.0, 0.0, 17.0], dtype=np.float32)

        # Case 1: Slow roll into empty net (v = 0 -> s_speed = 0.0, s_placement = 1.0)
        self.arena.last_goal_ball_pos = np.array([0.0, ARENA_EXTENT_Y, 50.0], dtype=np.float32)
        self.arena.last_goal_ball_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        r_slow_open = rew.get_reward(self.arena.cars[0], self.arena, np.zeros(8), is_goal=True, scoring_team=0)
        # R = 5.0 * (1.0 + 0.5 * 0.0 + 0.5 * 1.0) = 7.50
        self.assertAlmostEqual(r_slow_open, 7.50, places=4)

        # Case 2: Supersonic blast into empty net (v = 2500 -> s_speed = 1.0, s_placement = 1.0)
        self.arena.last_goal_ball_vel = np.array([0.0, 2500.0, 0.0], dtype=np.float32)
        r_fast_open = rew.get_reward(self.arena.cars[0], self.arena, np.zeros(8), is_goal=True, scoring_team=0)
        # R = 5.0 * (1.0 + 0.5 * 1.0 + 0.5 * 1.0) = 10.00
        self.assertAlmostEqual(r_fast_open, 10.00, places=4)

    def test_defended_placement_geometry(self):
        """Test defended goal placement: center shot, top corner snipe, far-post shot, and ceiling/corner defenders."""
        rew = GoalReward(goal_weight=5.0, concede_weight=-5.0, save_weight=2.0)
        rew.reset(self.arena)

        self.arena.cars[0].team = 0
        self.arena.cars[1].team = 1
        # Set speed = 2500 so s_speed = 1.0, isolating s_placement:
        # R = 5.0 * (1.0 + 0.5 * 1.0 + 0.5 * s_placement) = 7.5 + 2.5 * s_placement
        # s_placement = (R - 7.5) / 2.5
        self.arena.last_goal_ball_vel = np.array([0.0, 2500.0, 0.0], dtype=np.float32)

        def eval_placement(ball_x, ball_z, def_x, def_y, def_z):
            self.arena.last_goal_ball_pos = np.array([ball_x, ARENA_EXTENT_Y, ball_z], dtype=np.float32)
            self.arena.cars[1].pos = np.array([def_x, def_y, def_z], dtype=np.float32)
            r = rew.get_reward(self.arena.cars[0], self.arena, np.zeros(8), is_goal=True, scoring_team=0)
            return (r - 7.5) / 2.5

        # Scenario A: Defender centered in net (0, 5100, 17), shot straight at defender (0, 50)
        s_center = eval_placement(0.0, 50.0, 0.0, 5100.0, 17.0)
        self.assertAlmostEqual(s_center, (50.0 - 17.0) / 1500.0, places=3)
        self.assertLess(s_center, 0.03)

        # Scenario B: Defender centered in net (0, 5100, 17), shot to top-right corner (892.755, 642.775)
        s_corner = eval_placement(GOAL_HALF_WIDTH, GOAL_HEIGHT, 0.0, 5100.0, 17.0)
        expected_d = math.hypot(GOAL_HALF_WIDTH, GOAL_HEIGHT - 17.0)
        self.assertAlmostEqual(s_corner, expected_d / 1500.0, places=3)
        self.assertGreater(s_corner, 0.70)

        # Scenario C: Defender at left post (-892.755, 5100, 17), shot to right post (892.755, 200)
        s_far_post = eval_placement(GOAL_HALF_WIDTH, 200.0, -GOAL_HALF_WIDTH, 5100.0, 17.0)
        self.assertAlmostEqual(s_far_post, 1.00, places=4)

        # Scenario D: Defender on ceiling (0, 5100, 2040)
        s_ceiling = eval_placement(0.0, 500.0, 0.0, 5100.0, 2040.0)
        self.assertAlmostEqual(s_ceiling, 1.00, places=4)

        # Scenario E: Defender wide in corner (2500, 5100, 17)
        s_wide = eval_placement(GOAL_HALF_WIDTH, 50.0, 2500.0, 5100.0, 17.0)
        self.assertAlmostEqual(s_wide, 1.00, places=4)

    def test_multi_defender_closest_distance(self):
        """In multi-car games, placement must be dictated by the closest defender in range."""
        arena_3v3 = RocketSimArena(num_players=6)
        arena_3v3.reset(random_kickoff=False)
        rew = GoalReward(goal_weight=5.0, concede_weight=-5.0, save_weight=2.0)
        rew.reset(arena_3v3)

        arena_3v3.cars[0].team = 0
        arena_3v3.cars[1].team = 0
        arena_3v3.cars[2].team = 0
        arena_3v3.cars[3].team = 1; arena_3v3.cars[3].pos = np.array([-800.0, 5100.0, 17.0], dtype=np.float32)
        arena_3v3.cars[4].team = 1; arena_3v3.cars[4].pos = np.array([800.0, 5100.0, 17.0], dtype=np.float32)
        arena_3v3.cars[5].team = 1; arena_3v3.cars[5].pos = np.array([0.0, 1000.0, 17.0], dtype=np.float32)

        arena_3v3.last_goal_ball_vel = np.array([0.0, 2500.0, 0.0], dtype=np.float32)

        # Shot right at Defender 1 (-800, 50)
        arena_3v3.last_goal_ball_pos = np.array([-800.0, ARENA_EXTENT_Y, 50.0], dtype=np.float32)
        r = rew.get_reward(arena_3v3.cars[0], arena_3v3, np.zeros(8), is_goal=True, scoring_team=0)
        s_placement = (r - 7.5) / 2.5
        self.assertLess(s_placement, 0.05, "Shot straight at Defender 1 should have near-zero placement reward")

        # Shot right through the middle gap between defenders (0, 300)
        arena_3v3.last_goal_ball_pos = np.array([0.0, ARENA_EXTENT_Y, 300.0], dtype=np.float32)
        r_mid = rew.get_reward(arena_3v3.cars[0], arena_3v3, np.zeros(8), is_goal=True, scoring_team=0)
        s_mid = (r_mid - 7.5) / 2.5
        expected_d = math.hypot(800.0, 300.0 - 17.0)
        self.assertAlmostEqual(s_mid, expected_d / 1500.0, places=3)

    def test_zero_sum_symmetry_and_independent_concede(self):
        """Verify strict zero-sum symmetry by default, and independent concede_weight scaling."""
        rew_default = GoalReward(goal_weight=5.0, concede_weight=-5.0, save_weight=2.0)
        rew_default.reset(self.arena)

        self.arena.cars[0].team = 0
        self.arena.cars[1].team = 1

        test_scenarios = [
            (np.array([0.0, ARENA_EXTENT_Y, 50.0]), np.array([0.0, 500.0, 0.0]), np.array([0.0, 5100.0, 17.0])),
            (np.array([800.0, ARENA_EXTENT_Y, 600.0]), np.array([0.0, 1800.0, 0.0]), np.array([0.0, 5100.0, 17.0])),
            (np.array([0.0, ARENA_EXTENT_Y, 100.0]), np.array([0.0, 2600.0, 0.0]), np.array([0.0, 0.0, 17.0])),
        ]

        for b_pos, b_vel, def_pos in test_scenarios:
            self.arena.last_goal_ball_pos = b_pos.astype(np.float32)
            self.arena.last_goal_ball_vel = b_vel.astype(np.float32)
            self.arena.cars[1].pos = def_pos.astype(np.float32)

            r_scored = rew_default.get_reward(self.arena.cars[0], self.arena, np.zeros(8), is_goal=True, scoring_team=0)
            r_concede = rew_default.get_reward(self.arena.cars[1], self.arena, np.zeros(8), is_goal=True, scoring_team=0)

            self.assertAlmostEqual(r_scored + r_concede, 0.0, places=5)
            self.assertGreaterEqual(r_scored, 5.0)
            self.assertLessEqual(r_scored, 10.0)

        # Independent concede_weight test
        rew_custom = GoalReward(goal_weight=5.0, concede_weight=-8.0, save_weight=2.0)
        rew_custom.reset(self.arena)

        self.arena.last_goal_ball_pos = np.array([0.0, ARENA_EXTENT_Y, 100.0], dtype=np.float32)
        self.arena.last_goal_ball_vel = np.array([0.0, 2500.0, 0.0], dtype=np.float32)
        self.arena.cars[1].pos = np.array([0.0, 0.0, 17.0], dtype=np.float32)

        r_custom_scored = rew_custom.get_reward(self.arena.cars[0], self.arena, np.zeros(8), is_goal=True, scoring_team=0)
        r_custom_concede = rew_custom.get_reward(self.arena.cars[1], self.arena, np.zeros(8), is_goal=True, scoring_team=0)

        self.assertAlmostEqual(r_custom_scored, 10.00, places=4)
        self.assertAlmostEqual(r_custom_concede, -16.00, places=4)

    @unittest.skipUnless(ROCKETSIM_AVAILABLE, "RocketSim required for callback & early break test")
    def test_ball_snapshot_and_early_break_on_rsim_goal(self):
        """Verify RocketSim triggers goal callback, snapshots ball state, and breaks step loop early."""
        arena = RocketSimArena(num_players=2)
        arena.reset(random_kickoff=False)

        # Position ball right at the goal mouth (Y = +5100), moving fast into net (+Y)
        b_state = arena._rsim_arena.ball.get_state()
        b_state.pos = rsim.Vec(100.0, 5100.0, 200.0)
        b_state.vel = rsim.Vec(0.0, 1800.0, 0.0)
        arena._rsim_arena.ball.set_state(b_state)
        arena._sync_from_rsim()

        # Step 8 sub-ticks
        is_goal, scoring_team = arena.step([np.zeros(8), np.zeros(8)], dt=8.0 / 120.0)

        self.assertTrue(is_goal, "Goal must be detected")
        self.assertEqual(scoring_team, 0, "Team 0 must score in Orange goal")
        self.assertIsNotNone(arena.last_goal_ball_pos, "Ball position must be snapshotted on goal")
        self.assertIsNotNone(arena.last_goal_ball_vel, "Ball velocity must be snapshotted on goal")

        # Verify snapshotted velocity is near 1800 uu/s
        snap_speed = _norm3(arena.last_goal_ball_vel)
        self.assertAlmostEqual(snap_speed, 1800.0, delta=100.0)
        # Verify snapshotted Y position crossed goal line (>= 5100)
        self.assertGreaterEqual(arena.last_goal_ball_pos[1], 5100.0)


if __name__ == "__main__":
    unittest.main()
