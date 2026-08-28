"""
Macro Potential-Based Reward Architecture for Rocket League Reinforcement Learning.
Engineered for clean macro game intelligence (scoring, ball progression, pursuit, and boost conservation)
modeled after competitive RLGym standards (Nexto, Necto, Element).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from env.physics_engine import (
    CarState, BallState, RocketSimArena,
    CAR_MAX_SPEED, BALL_MAX_SPEED, GOAL_HALF_WIDTH, GOAL_HEIGHT, ARENA_EXTENT_Y
)


class BaseReward:
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def reset(self, initial_state: RocketSimArena):
        pass

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        raise NotImplementedError


# ==============================================================================
# 1. MACRO MATCH EVENT (Goals, Concedes, Saves)
# ==============================================================================
class GoalReward(BaseReward):
    """
    Zero-sum match outcome reward.
    Rewards scoring goals (+10.0), penalizes conceding (-10.0),
    and rewards defensive saves/clears off the goal line (+3.0).
    """
    def __init__(self, goal_weight: float = 10.0, concede_weight: float = -10.0, save_weight: float = 3.0):
        super().__init__(goal_weight)
        self.concede_weight = concede_weight
        self.save_weight = save_weight
        self._prev_touches: Dict[int, int] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if is_goal and scoring_team is not None:
            return self.weight if car.team == scoring_team else self.concede_weight

        # Defensive Goal-Line Save & Clear
        defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_ball_to_net = abs(arena.ball.pos[1] - defending_y)
        dist_car_to_net = abs(car.pos[1] - defending_y)

        if dist_ball_to_net < 1200.0 and dist_car_to_net < 1500.0 and abs(arena.ball.pos[0]) < GOAL_HALF_WIDTH * 1.6:
            ball_vy_out = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
            prev_t = self._prev_touches.get(car.id, 0)
            if car.ball_touches > prev_t and ball_vy_out > 350.0:
                self._prev_touches[car.id] = car.ball_touches
                return self.save_weight

        self._prev_touches[car.id] = car.ball_touches
        return 0.0


# ==============================================================================
# 2. BALL-TO-GOAL PROGRESSION (Field Displacement)
# ==============================================================================
class BallToGoalVelocityReward(BaseReward):
    """
    Continuous Potential-Based Progression.
    Rewards ball velocity directed toward the opponent's net.
    Normalizes by BALL_MAX_SPEED (6000.0 uu/s).
    """
    def __init__(self, weight: float = 1.5):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
        target_pos = np.array([0.0, target_goal_y, 0.0], dtype=np.float32)

        ball_to_goal = target_pos - arena.ball.pos
        dist = float(np.linalg.norm(ball_to_goal))
        if dist < 1e-4:
            return 0.0

        unit_to_goal = ball_to_goal / dist
        ball_velocity_toward_goal = float(np.dot(arena.ball.vel, unit_to_goal))

        # Normalized progress: positive when moving toward opponent goal, negative toward own
        normalized_progress = ball_velocity_toward_goal / BALL_MAX_SPEED
        return self.weight * normalized_progress


# ==============================================================================
# 3. PLAYER-TO-BALL DISTANCE DELTA (Pursuit & Approach Potential)
# ==============================================================================
class PlayerToBallVelocityReward(BaseReward):
    """
    Necto / RLGym Potential-Based Distance Delta Approach Reward.
    Rewards closing the Euclidean distance gap to the ball:
        Phi(s) = -dist(car, ball) / 2000.0
        Reward = (prev_dist - curr_dist) / 2000.0
    When shadowing or circling a moving ball at constant distance: Reward is 0.0.
    Within close striking proximity (< 300 uu): Approach potential is saturated (0.0),
    forcing the agent to focus purely on strike direction (BallToGoal / TouchBall).
    """
    def __init__(self, weight: float = 0.15):
        super().__init__(weight)
        self._prev_dist: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_dist = {
            car.id: float(np.linalg.norm(initial_state.ball.pos - car.pos))
            for car in initial_state.cars
        }

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        curr_dist = float(np.linalg.norm(arena.ball.pos - car.pos))
        prev_dist = self._prev_dist.get(car.id, curr_dist)
        self._prev_dist[car.id] = curr_dist

        # If within striking proximity (< 300 uu), approach potential saturates to 0.0
        # so strike quality and goal trajectory dictate reward
        if curr_dist < 300.0:
            return 0.0

        # Distance gap delta (positive when closing distance, negative when retreating)
        delta_dist = (prev_dist - curr_dist) / 2000.0

        # Kickoff sprint multiplier
        is_kickoff = bool(abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and float(np.linalg.norm(arena.ball.vel)) < 100.0)
        if is_kickoff and delta_dist > 0.0:
            return self.weight * delta_dist * 2.0

        return self.weight * delta_dist


# ==============================================================================
# 4. BALL TOUCH & DIRECTIONALITY (Hit Quality)
# ==============================================================================
class TouchBallReward(BaseReward):
    """
    Atomic Ball Strike Quality.
    Rewarded at the exact moment of ball contact, scaled by touch speed and
    alignment directed towards the opponent's net.
    Includes an instant +2.5 bounty for winning the kickoff first touch.
    """
    def __init__(self, weight: float = 1.2):
        super().__init__(weight)
        self._prev_touches: Dict[int, int] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_touches.get(car.id, 0)
        curr = car.ball_touches
        self._prev_touches[car.id] = curr

        if curr > prev:
            # Touch occurred on this step
            ball_speed = float(np.linalg.norm(arena.ball.vel))
            power_factor = min(1.6, max(0.5, ball_speed / 1200.0))

            target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
            unit_to_goal = np.array([0.0, 1.0 if car.team == 0 else -1.0, 0.0], dtype=np.float32)

            if ball_speed > 1e-4:
                unit_ball_vel = arena.ball.vel / ball_speed
                goal_alignment = float(np.dot(unit_ball_vel, unit_to_goal))
                direction_multiplier = max(0.1, (goal_alignment + 1.0) / 2.0)
            else:
                direction_multiplier = 0.5

            # Kickoff first-touch race bounty
            is_kickoff_touch = bool(abs(arena.ball.pos[0]) < 200.0 and abs(arena.ball.pos[1]) < 200.0 and all(c.ball_touches <= 1 for c in arena.cars))
            kickoff_bounty = 2.5 if is_kickoff_touch else 0.0

            return (self.weight * power_factor * direction_multiplier) + kickoff_bounty

        return 0.0


# ==============================================================================
# 5. BOOST RETENTION & ECONOMY (Necto Sqrt-Potential Engine)
# ==============================================================================
class BoostReward(BaseReward):
    """
    Necto Potential-Based Boost Conservation & Pad Collection.
    Uses sqrt(boost) to weight low boost levels heavily, and gates ground-burning waste
    without penalizing aerial flight.
    """
    def __init__(self, gain_weight: float = 0.6, lose_weight: float = 0.3):
        super().__init__(gain_weight)
        self.gain_weight = gain_weight
        self.lose_weight = lose_weight
        self._prev_boost: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_boost = {car.id: float(np.clip(car.boost / 100.0, 0.0, 1.0)) for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_boost.get(car.id, float(np.clip(car.boost / 100.0, 0.0, 1.0)))
        curr = float(np.clip(car.boost / 100.0, 0.0, 1.0))
        self._prev_boost[car.id] = curr

        boost_diff = math.sqrt(curr) - math.sqrt(prev)
        is_kickoff = bool(abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and float(np.linalg.norm(arena.ball.vel)) < 100.0)

        if boost_diff >= 0:
            return self.gain_weight * boost_diff
        elif not is_kickoff and car.pos[2] < GOAL_HEIGHT:
            height_factor = max(0.0, 1.0 - (car.pos[2] / GOAL_HEIGHT))
            return self.lose_weight * boost_diff * height_factor

        return 0.0


# ==============================================================================
# COMBINED MACRO REWARD ENGINE & MANAGER
# ==============================================================================
class CombinedReward:
    """
    Unified Macro Potential-Based Reward Manager.
    Aggregates Macro Goal, Ball-to-Goal, Player-to-Ball, Touch Quality, and Boost Conservation.
    """
    def __init__(self, weights: Dict[str, float]):
        self.rewards: Dict[str, BaseReward] = {
            "goal": GoalReward(
                goal_weight=weights.get("goal_weight", 30.0),
                concede_weight=weights.get("concede_weight", -30.0),
                save_weight=weights.get("save_weight", 5.0)
            ),
            "ball_to_goal": BallToGoalVelocityReward(
                weight=weights.get("ball_to_goal_weight", 2.0)
            ),
            "player_to_ball": PlayerToBallVelocityReward(
                weight=weights.get("player_to_ball_weight", 0.15)
            ),
            "touch": TouchBallReward(
                weight=weights.get("touch_weight", 1.5)
            ),
            "boost": BoostReward(
                gain_weight=weights.get("boost_gain_weight", 0.5),
                lose_weight=weights.get("boost_lose_weight", 0.1)
            )
        }

    def reset(self, initial_state: RocketSimArena):
        for r in self.rewards.values():
            r.reset(initial_state)

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Dynamically update macro weights from UI or live config.
        """
        if "goal_weight" in new_weights and "goal" in self.rewards:
            self.rewards["goal"].weight = float(new_weights["goal_weight"])
        if "concede_weight" in new_weights and "goal" in self.rewards:
            self.rewards["goal"].concede_weight = float(new_weights["concede_weight"])
        if "save_weight" in new_weights and "goal" in self.rewards:
            self.rewards["goal"].save_weight = float(new_weights["save_weight"])

        if "ball_to_goal_weight" in new_weights and "ball_to_goal" in self.rewards:
            self.rewards["ball_to_goal"].weight = float(new_weights["ball_to_goal_weight"])

        if "player_to_ball_weight" in new_weights and "player_to_ball" in self.rewards:
            self.rewards["player_to_ball"].weight = float(new_weights["player_to_ball_weight"])

        if "touch_weight" in new_weights and "touch" in self.rewards:
            self.rewards["touch"].weight = float(new_weights["touch_weight"])

        if "boost_gain_weight" in new_weights and "boost" in self.rewards:
            self.rewards["boost"].gain_weight = float(new_weights["boost_gain_weight"])
        if "boost_lose_weight" in new_weights and "boost" in self.rewards:
            self.rewards["boost"].lose_weight = float(new_weights["boost_lose_weight"])

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        breakdown = {}
        for name, r in self.rewards.items():
            rew = float(r.get_reward(car, arena, action, is_goal, scoring_team))
            total += rew
            breakdown[name] = rew
        return float(total), breakdown


class RewardManager:
    """
    Standard RewardManager API wrapper for environment integrations.
    """
    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        self.combined = CombinedReward(reward_weights or {})

    def reset(self, initial_state: RocketSimArena):
        self.combined.reset(initial_state)

    def update_weights(self, new_weights: Dict[str, float]):
        self.combined.update_weights(new_weights)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> Tuple[float, Dict[str, float]]:
        return self.combined.get_reward(car, arena, action, is_goal, scoring_team)
