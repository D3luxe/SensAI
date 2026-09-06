"""
Targeted tests for Reward System Remediation.
Verifies fixes for goal step penalties, lateral defensive saves, PBRS wiggle pump,
mid-air barrel roll farming, wall hovering exploits, defensive sprint exemptions,
and joystick-only dodge impulse reconstruction.
"""

import math
import unittest
import numpy as np

from env.physics_engine import CarState, RocketSimArena, ARENA_EXTENT_X, ARENA_EXTENT_Y, GOAL_HALF_WIDTH
from env.rewards import (
    GoalReward, BallToGoalVelocityReward, PlayerToBallVelocityReward,
    TouchBallReward, JumpBridgeReward, BoostReward, PowerslideReward,
    AirRollRecoveryReward, CombinedReward, RewardManager,
    compute_opponent_threats, evaluate_clear_quality, OpponentThreat
)


class TestRewardRemediation(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena(num_players=2, game_mode="1v1")
        self.arena.reset(random_kickoff=False)

    def test_goal_step_no_penalty(self):
        """Test that scoring a goal grants positive reward and does NOT inflict negative progression or own-goal penalties."""
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 5000.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1500.0, 0.0], dtype=np.float32),
            ball_touches=1
        )
        self.arena.ball.pos = np.array([0.0, 5150.0, 100.0], dtype=np.float32)  # inside opponent net
        self.arena.ball.vel = np.array([0.0, 1500.0, 0.0], dtype=np.float32)
        self.arena.cars = [car]

        rew_touch = TouchBallReward(weight=1.2)
        rew_touch._prev_touches[0] = 0
        r_touch = rew_touch.get_reward(car, self.arena, np.zeros(8), is_goal=True, scoring_team=0)

        rew_prog = BallToGoalVelocityReward(weight=1.5)
        r_prog = rew_prog.get_reward(car, self.arena, np.zeros(8), is_goal=True, scoring_team=0)

        rew_pursuit = PlayerToBallVelocityReward(weight=0.6)
        r_pursuit = rew_pursuit.get_reward(car, self.arena, np.zeros(8), is_goal=True, scoring_team=0)

        self.assertGreater(r_touch, 2.0, f"Scoring touch must be strongly positive, got {r_touch}")
        self.assertEqual(r_prog, 0.0, f"Progression on goal step should be 0.0, got {r_prog}")
        self.assertEqual(r_pursuit, 0.0, f"Pursuit on goal step should be 0.0, got {r_pursuit}")

    def test_lateral_defensive_save_no_own_goal(self):
        """Test that deflecting a goal-bound shot laterally into the corner yields save reward and not own-goal penalty."""
        # Defending goal is at Y = -5120
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, -4800.0, 17.0], dtype=np.float32),
            vel=np.array([500.0, 0.0, 0.0], dtype=np.float32),
            ball_touches=1
        )
        self.arena.ball.pos = np.array([0.0, -4800.0, 93.0], dtype=np.float32)
        # Ball deflected laterally into corner: vx = 1200, vy = -50 (away from center net towards back corner)
        self.arena.ball.vel = np.array([1200.0, -50.0, 100.0], dtype=np.float32)
        self.arena.cars = [car]

        rew_goal = GoalReward(save_weight=12.0)
        rew_goal._prev_touches[0] = 0
        r_goal = rew_goal.get_reward(car, self.arena, np.zeros(8), False, None)

        rew_touch = TouchBallReward(weight=1.2)
        rew_touch._prev_touches[0] = 0
        r_touch = rew_touch.get_reward(car, self.arena, np.zeros(8), False, None)

        self.assertEqual(r_goal, 12.0, f"Lateral deflection in goal box should trigger save reward (12.0), got {r_goal}")
        self.assertGreater(r_touch, 0.5, f"Lateral defensive clear touch should yield positive reward, got {r_touch}")

    def test_strike_zone_no_wiggle_pump(self):
        """Test that oscillating back and forth inside the strike zone yields 0.0 net distance reward (no infinite pump)."""
        rew = PlayerToBallVelocityReward(weight=0.6)
        arena = RocketSimArena(num_players=1)
        arena.ball.pos = np.array([0.0, 1000.0, 93.0], dtype=np.float32)
        car = CarState(id=0, team=0, pos=np.array([0.0, 600.0, 17.0], dtype=np.float32), rot=np.array([0.0, 1.57, 0.0], dtype=np.float32))
        arena.cars = [car]
        rew.reset(arena)

        total_loop_reward = 0.0
        for _ in range(50):
            car.pos[1] = 650.0
            r1 = rew.get_reward(car, arena, np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), False, None)
            car.pos[1] = 600.0
            r2 = rew.get_reward(car, arena, np.array([-1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), False, None)
            total_loop_reward += (r1 + r2)

        self.assertLessEqual(total_loop_reward, 0.0,
                             msg=f"Wiggling back and forth must yield zero or negative net reward (no positive pump), got {total_loop_reward}")

    def test_no_midair_barrel_roll_farming(self):
        """Test that continuous midair 360 rolls are capped and cannot be farmed indefinitely."""
        rew = AirRollRecoveryReward(weight=0.10)
        arena = RocketSimArena(num_players=1)
        arena.ball.pos = np.array([0.0, 2000.0, 93.0], dtype=np.float32)
        car = CarState(id=0, team=0, pos=np.array([0.0, 0.0, 500.0], dtype=np.float32), vel=np.array([0.0, 500.0, 0.0], dtype=np.float32), rot=np.array([0.0, 1.57, 0.0], dtype=np.float32), on_ground=False)
        arena.cars = [car]
        rew.reset(arena)

        total_reward = 0.0
        steps_per_rev = 20
        for _ in range(10):
            for s in range(steps_per_rev):
                angle = (s / steps_per_rev) * 2 * math.pi
                car.rot[2] = angle
                r = rew.get_reward(car, arena, np.zeros(8, dtype=np.float32), False, None)
                total_reward += r

        self.assertLessEqual(total_reward, 0.15, f"Continuous midair barrel rolls must be strictly capped, got {total_reward}")

    def test_no_wall_hovering_farming(self):
        """Test that floating/hovering near the wall does NOT award continuous passive rewards."""
        rew = AirRollRecoveryReward(weight=0.10)
        arena = RocketSimArena(num_players=1)
        car = CarState(id=0, team=0, pos=np.array([ARENA_EXTENT_X - 100.0, 0.0, 500.0], dtype=np.float32), vel=np.array([0.0, 500.0, 0.0], dtype=np.float32), on_ground=False)
        car.rot_mat = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0]
        ], dtype=np.float32)

        arena.cars = [car]
        rew.reset(arena)

        total_wall_reward = 0.0
        for _ in range(30):
            r = rew.get_reward(car, arena, np.zeros(8, dtype=np.float32), False, None)
            total_wall_reward += r

        self.assertEqual(total_wall_reward, 0.0, f"Hovering parallel to wall must award 0.0 continuous reward, got {total_wall_reward}")

    def test_defensive_sprint_no_penalties(self):
        """Test that sprinting back to defend own net does not incur throttle or off-axis boost penalties."""
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 1000.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, -1400.0, 0.0], dtype=np.float32),
            rot=np.array([0.0, -math.pi / 2, 0.0], dtype=np.float32),
            boost=50.0,
            on_ground=True
        )
        self.arena.ball.pos = np.array([0.0, 2000.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([0.0, -1800.0, 0.0], dtype=np.float32)
        self.arena.cars = [car]

        action = np.zeros(8, dtype=np.float32)
        action[0] = 1.0
        action[6] = 1.0

        rew_boost = BoostReward(gain_weight=0.6, lose_weight=0.3)
        rew_boost.reset(self.arena)
        car.boost = 47.0
        r_boost = rew_boost.get_reward(car, self.arena, action, False, None)
        self.assertGreater(r_boost, -0.05, f"Boosting back on defense must not incur off-axis penalty, got {r_boost}")

        rew_pursuit = PlayerToBallVelocityReward(weight=0.6)
        rew_pursuit.reset(self.arena)
        r_pursuit = rew_pursuit.get_reward(car, self.arena, action, False, None)
        self.assertGreater(r_pursuit, -0.05, f"Throttling back on defense must not incur wrong_way_throttle_penalty, got {r_pursuit}")

    def test_joystick_dodge_no_roll_interference(self):
        """Test that dodge impulse is driven by joystick pitch/yaw and air roll has zero impact."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 100.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=False,
            has_flip=False
        )
        self.arena.ball.pos = np.array([0.0, 1000.0, 100.0], dtype=np.float32)

        rew._prev_on_ground[car.id] = False
        rew._prev_has_flip[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = car.vel.copy()

        action1 = np.zeros(8, dtype=np.float32)
        action1[2] = 1.0
        action1[5] = 1.0
        r1 = rew.get_reward(car, self.arena, action1, False, None)

        rew._prev_has_flip[car.id] = True
        action2 = np.zeros(8, dtype=np.float32)
        action2[2] = 1.0
        action2[4] = 1.0
        action2[5] = 1.0
        r2 = rew.get_reward(car, self.arena, action2, False, None)

        self.assertAlmostEqual(r1, r2, places=4, msg="Air roll input must not alter frontflip dodge reward")

    def test_attacking_rebound_tactical_dir(self):
        """Test that an attacker ahead of the ball in the offensive half is NOT marked wrong-side."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 4000.0, 100.0], dtype=np.float32),
            rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),
            on_ground=False,
            has_flip=False
        )
        self.arena.ball.pos = np.array([0.0, 3500.0, 100.0], dtype=np.float32)

        rew._prev_on_ground[car.id] = False
        rew._prev_has_flip[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = car.vel.copy()

        action = np.zeros(8, dtype=np.float32)
        action[2] = -1.0
        action[5] = 1.0
        r = rew.get_reward(car, self.arena, action, False, None)

        self.assertGreater(r, 0.15, f"Backflip dodge toward rebound in attacking half must be rewarded, got {r}")

    def test_opponent_threat_arrival_time_ordering(self):
        """Test that compute_opponent_threats orders opponents by arrival time ascending (most urgent first)."""
        car = CarState(id=0, team=0, pos=np.array([0.0, -3000.0, 17.0], dtype=np.float32))
        self.arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)

        # Opponent 1: Closer (dist = 1000) but slow/retreating (vel = +100 away from ball) -> high arrival time
        opp_slow = CarState(
            id=1, team=1,
            pos=np.array([0.0, 1000.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 100.0, 0.0], dtype=np.float32)  # moving away from ball (+Y)
        )

        # Opponent 2: Further away (dist = 1800) but charging at supersonic speed (vel = -2000 towards ball) -> arrival ~0.9s
        opp_fast = CarState(
            id=2, team=1,
            pos=np.array([0.0, 1800.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, -2000.0, 0.0], dtype=np.float32)  # charging towards ball (-Y)
        )

        self.arena.cars = [car, opp_slow, opp_fast]
        threats = compute_opponent_threats(car, self.arena)

        self.assertEqual(len(threats), 2)
        self.assertEqual(threats[0].opp.id, opp_fast.id, "Fast charging opponent must be ranked as most urgent threat")
        self.assertLess(threats[0].arrival_time, threats[1].arrival_time)
        self.assertLess(threats[0].arrival_time, 1.0)

    def test_jump_bridge_supersonic_challenge(self):
        """Test that an incoming supersonic opponent at 1200 uu triggers is_opponent_challenging and rewards 50/50 block jump."""
        rew = JumpBridgeReward(weight=0.35)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, -300.0, 17.0], dtype=np.float32),
            vel=np.zeros(3, dtype=np.float32),
            on_ground=False
        )
        car.vel[2] = 292.0  # Just lifted off
        self.arena.ball.pos = np.array([0.0, 0.0, 93.0], dtype=np.float32)

        # Supersonic opponent at 1200 uu charging at 2000 uu/s (arrival time = 0.6s < 0.85s)
        opp = CarState(
            id=1, team=1,
            pos=np.array([0.0, 1200.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, -2000.0, 0.0], dtype=np.float32)
        )
        self.arena.cars = [car, opp]

        rew._prev_on_ground[car.id] = True
        rew._prev_has_flip[car.id] = True
        rew._prev_touches[car.id] = 0
        rew._prev_vel[car.id] = np.zeros(3, dtype=np.float32)

        action = np.zeros(8, dtype=np.float32)
        action[5] = 1.0
        r = rew.get_reward(car, self.arena, action, False, None)
        self.assertGreater(r, 0.3, f"Supersonic incoming challenge must grant contested 50/50 liftoff reward, got {r}")

    def test_evaluate_clear_quality_edge_cases(self):
        """Test edge cases: empty opponents, demoed opponents, dead-ball pops, dangerous rebounds, and corner clears."""
        ball_pos = np.array([0.0, -4800.0, 93.0], dtype=np.float32)

        # 1. Empty opponents -> 1.0
        q_empty = evaluate_clear_quality(ball_pos, np.array([500.0, 500.0, 0.0], dtype=np.float32), car_team=0, arena=self.arena, opponents=[])
        self.assertEqual(q_empty, 1.0)

        # 2. Demoed opponent -> 1.0
        demoed_opp = CarState(id=1, team=1, pos=np.array([0.0, -3500.0, 17.0], dtype=np.float32), demoed=True)
        q_demoed = evaluate_clear_quality(ball_pos, np.array([0.0, 800.0, 0.0], dtype=np.float32), car_team=0, arena=self.arena, opponents=[demoed_opp])
        self.assertEqual(q_demoed, 1.0)

        # 3. Soft dead-ball pop (speed < 50) -> 1.0
        active_opp = CarState(id=1, team=1, pos=np.array([0.0, -3500.0, 17.0], dtype=np.float32))
        q_soft = evaluate_clear_quality(ball_pos, np.array([0.0, 30.0, 0.0], dtype=np.float32), car_team=0, arena=self.arena, opponents=[active_opp])
        self.assertEqual(q_soft, 1.0)

        # 4. Dangerous pass directly into oncoming opponent (cos_theta > 0.9, fast intercept) -> <= 0.75
        opp_charging = CarState(
            id=1, team=1,
            pos=np.array([0.0, -3500.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, -1000.0, 0.0], dtype=np.float32)
        )
        q_dangerous = evaluate_clear_quality(
            ball_pos,
            np.array([0.0, 600.0, 0.0], dtype=np.float32),  # ball cleared directly toward oncoming opponent
            car_team=0, arena=self.arena, opponents=[opp_charging]
        )
        self.assertLessEqual(q_dangerous, 0.75, f"Clearing directly into charging attacker must yield low multiplier, got {q_dangerous}")

        # 5. Safe corner clear (aimed away into corner: vx = 900, vy = 200, |x| > 2000) -> >= 1.15
        corner_ball = np.array([2200.0, -4800.0, 93.0], dtype=np.float32)
        q_safe_corner = evaluate_clear_quality(
            corner_ball,
            np.array([800.0, 200.0, 0.0], dtype=np.float32),
            car_team=0, arena=self.arena, opponents=[opp_charging]
        )
        self.assertGreaterEqual(q_safe_corner, 1.15, f"Safe corner clear must yield boosted multiplier, got {q_safe_corner}")

    def test_vertical_pop_save_trigger(self):
        """Test that a pure vertical pop on the goal line triggers the save reward in GoalReward."""
        rew = GoalReward(save_weight=12.0)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, -4900.0, 17.0], dtype=np.float32),
            ball_touches=1
        )
        self.arena.ball.pos = np.array([0.0, -4900.0, 150.0], dtype=np.float32)
        # Pure vertical pop: vz = 550, vy = 0, vx = 50
        self.arena.ball.vel = np.array([50.0, 0.0, 550.0], dtype=np.float32)
        self.arena.cars = [car]

        rew._prev_touches[car.id] = 0
        r_save = rew.get_reward(car, self.arena, np.zeros(8), False, None)
        self.assertGreaterEqual(r_save, 7.0, f"Vertical goal-line pop must trigger save reward, got {r_save}")

    def test_multi_car_pinch_clear_same_tick(self):
        """Test that in a 2v2 arena, simultaneous touches by two teammates on the same tick evaluate independently."""
        rew = GoalReward(save_weight=12.0)
        car1 = CarState(id=0, team=0, pos=np.array([-100.0, -4900.0, 17.0], dtype=np.float32), ball_touches=1)
        car2 = CarState(id=1, team=0, pos=np.array([100.0, -4900.0, 17.0], dtype=np.float32), ball_touches=1)
        opp = CarState(id=2, team=1, pos=np.array([0.0, -2000.0, 17.0], dtype=np.float32))

        self.arena.cars = [car1, car2, opp]
        self.arena.ball.pos = np.array([0.0, -4900.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([1000.0, 500.0, 200.0], dtype=np.float32)

        rew._prev_touches[car1.id] = 0
        rew._prev_touches[car2.id] = 0

        r1 = rew.get_reward(car1, self.arena, np.zeros(8), False, None)
        r2 = rew.get_reward(car2, self.arena, np.zeros(8), False, None)

        self.assertGreater(r1, 10.0)
        self.assertAlmostEqual(r1, r2, places=3, msg="Both pinching teammates must receive identical valid clearance rewards")

    def test_touch_reward_clear_urgency(self):
        """Test that a defensive clear under heavy opponent pressure earns higher clear bonus than unpressured."""
        rew = TouchBallReward(weight=1.2)
        car = CarState(
            id=0, team=0,
            pos=np.array([1500.0, -4500.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, 1000.0, 0.0], dtype=np.float32),
            ball_touches=1
        )
        self.arena.ball.pos = np.array([1500.0, -4500.0, 93.0], dtype=np.float32)
        self.arena.ball.vel = np.array([500.0, 1600.0, 200.0], dtype=np.float32)

        # Scenario A: Urgent pressure (opponent 800 uu away charging at 1500 uu/s -> arrival ~0.5s)
        opp_urgent = CarState(
            id=1, team=1,
            pos=np.array([1500.0, -3700.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, -1500.0, 0.0], dtype=np.float32)
        )
        self.arena.cars = [car, opp_urgent]
        rew._prev_touches[car.id] = 0
        r_urgent = rew.get_reward(car, self.arena, np.zeros(8), False, None)

        # Scenario B: Uncontested (opponent far away at opponent goal pos = [0, 4500, 17])
        opp_far = CarState(
            id=1, team=1,
            pos=np.array([0.0, 4500.0, 17.0], dtype=np.float32),
            vel=np.zeros(3, dtype=np.float32)
        )
        self.arena.cars = [car, opp_far]
        rew._prev_touches[car.id] = 0
        r_unpressured = rew.get_reward(car, self.arena, np.zeros(8), False, None)

        self.assertGreater(r_urgent, r_unpressured, f"Urgent defensive clear ({r_urgent}) must earn higher reward than unpressured clear ({r_unpressured})")

    def test_boost_retreat_exemption_threat_gating(self):
        """Test that BoostReward retreat exemption is active under threat and penalizes boost burn when unpressured."""
        rew = BoostReward(gain_weight=0.6, lose_weight=0.3)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, -600.0, 0.0], dtype=np.float32),  # Driving backwards toward defending goal
            rot=np.array([0.0, -math.pi / 2, 0.0], dtype=np.float32),  # Facing defending goal (-Y)
            boost=50.0,
            on_ground=True
        )
        # Ball in opponent half
        self.arena.ball.pos = np.array([0.0, 2000.0, 93.0], dtype=np.float32)
        rew._prev_boost[car.id] = 52.0  # Burned 2 boost

        action = np.zeros(8, dtype=np.float32)
        action[0] = 1.0  # throttle
        action[6] = 1.0  # boost

        # Scenario A: Opponent is threatening near ball (dist = 500, charging) -> retreat is EXEMPT
        opp_threat = CarState(
            id=1, team=1,
            pos=np.array([0.0, 2500.0, 17.0], dtype=np.float32),
            vel=np.array([0.0, -1200.0, 0.0], dtype=np.float32)
        )
        self.arena.cars = [car, opp_threat]
        r_threat_retreat = rew.get_reward(car, self.arena, action, False, None)

        # Scenario B: Opponent is far away in their own corner, no threat -> retreat is NOT exempt from off-axis boost waste
        rew._prev_boost[car.id] = 52.0  # Reset burned boost
        opp_passive = CarState(
            id=1, team=1,
            pos=np.array([3000.0, 4800.0, 17.0], dtype=np.float32),
            vel=np.zeros(3, dtype=np.float32)
        )
        self.arena.cars = [car, opp_passive]
        r_passive_retreat = rew.get_reward(car, self.arena, action, False, None)

        self.assertGreater(r_threat_retreat, r_passive_retreat,
                           f"Retreating under threat ({r_threat_retreat}) must incur less boost penalty than panic retreating without threat ({r_passive_retreat})")


if __name__ == "__main__":
    unittest.main()
