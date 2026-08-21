"""
Modular and dynamically configurable reward functions for Rocket League bot reinforcement learning.
Supports real-time weight adjustments without stopping training.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any, List, Optional
from env.physics_engine import CarState, BallState, RocketSimArena, CAR_MAX_SPEED, BALL_MAX_SPEED, GOAL_HALF_WIDTH, GOAL_HEIGHT, ARENA_EXTENT_Y


class BaseReward:
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def reset(self, initial_state: RocketSimArena):
        pass

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        raise NotImplementedError


class TouchBallReward(BaseReward):
    """
    Rewards making contact with the ball.
    Scales with aerial_flip_multi when jumping/flipping and awards kickoff_first_touch_bonus on kickoff.
    """
    def __init__(self, weight: float = 10.0, first_touch_bonus: float = 35.0, aerial_flip_multi: float = 2.5):
        super().__init__(weight)
        self.first_touch_bonus = first_touch_bonus
        self.aerial_flip_multi = aerial_flip_multi
        self._prev_touches: Dict[int, int] = {}
        self._first_touch_claimed: bool = False

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._first_touch_claimed = False

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_touches.get(car.id, 0)
        curr = car.ball_touches
        self._prev_touches[car.id] = curr
        if curr > prev:
            # First touch kickoff bounty
            first_bonus = self.first_touch_bonus if not self._first_touch_claimed else 0.0
            self._first_touch_claimed = True

            # Multiplier for jumping, dodging, or flipping into ANY ball (ground or air!)
            is_jump_or_flip = (not car.on_ground) or car.just_dodged or (car.pos[2] > 25.0)
            aerial_flip_mult = self.aerial_flip_multi if is_jump_or_flip else 1.0

            return (self.weight * aerial_flip_mult) + first_bonus
        return 0.0


class SpeedTowardBallReward(BaseReward):
    """
    Micro-scaled per-step reward for closing distance to the ball through the front bumper.
    """
    def __init__(self, weight: float = 0.05, dodge_rush_multi: float = 1.5):
        super().__init__(weight)
        self.dodge_rush_multi = dodge_rush_multi

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        car_to_ball = arena.ball.pos - car.pos
        dist = float(np.linalg.norm(car_to_ball))
        if dist < 1e-4:
            return 0.0
        unit_to_ball = car_to_ball / dist

        # Forward alignment multiplier ensures speed towards ball is prioritized through front bumper
        fwd = car.get_forward_vector()
        fwd_align = max(0.0, float(np.dot(fwd, unit_to_ball)))

        speed_toward = float(np.dot(car.vel, unit_to_ball))
        norm_speed = speed_toward / CAR_MAX_SPEED

        # Flip / Dodge forward boost: rewards front-flipping / speed-flipping directly toward the ball
        dodge_mult = self.dodge_rush_multi if (car.just_dodged and fwd_align > 0.5) else 1.0

        return self.weight * norm_speed * (0.3 + 0.7 * fwd_align) * dodge_mult


class FaceBallReward(BaseReward):
    """
    Rewards aligning the car nose directly towards the ball ONLY when actively moving fast toward it (>350 uu/s).
    Completely eliminates the standstill 'stare-from-the-midfield' exploit.
    """
    def __init__(self, weight: float = 0.02):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        car_to_ball = arena.ball.pos - car.pos
        dist = float(np.linalg.norm(car_to_ball))
        if dist < 1e-4:
            return 0.0
        unit_to_ball = car_to_ball / dist

        # Strict velocity gate: must be driving towards the ball (>350 uu/s) to earn alignment reward
        speed_toward = float(np.dot(car.vel, unit_to_ball))
        if speed_toward < 350.0:
            return 0.0

        fwd = car.get_forward_vector()
        alignment = max(0.0, float(np.dot(fwd, unit_to_ball)))
        norm_speed = min(1.0, speed_toward / CAR_MAX_SPEED)

        return self.weight * alignment * norm_speed


class BallVelocityToGoalReward(BaseReward):
    """
    Rewards propelling the ball toward the opponent net at high speed.
    Penalizes sending the ball towards defending net (own-goal prevention).
    """
    def __init__(self, weight: float = 0.08):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
        target_goal = np.array([0.0, target_goal_y, GOAL_HEIGHT * 0.5], dtype=np.float32)

        ball_to_goal = target_goal - arena.ball.pos
        dist = float(np.linalg.norm(ball_to_goal))
        if dist < 1e-4:
            return 0.0
        unit_ball_to_goal = ball_to_goal / dist

        ball_speed_toward_goal = float(np.dot(arena.ball.vel, unit_ball_to_goal))
        norm_ball_speed = ball_speed_toward_goal / BALL_MAX_SPEED

        # Symmetric reward/penalty
        return self.weight * norm_ball_speed


class GoalReward(BaseReward):
    """
    Major match-winning reward granted when scoring a goal, with optional power shot scaling.
    """
    def __init__(self, goal_weight: float = 100.0, concede_weight: float = -100.0, speed_multiplier: float = 1.5):
        super().__init__(goal_weight)
        self.concede_weight = concede_weight
        self.speed_multiplier = speed_multiplier

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if is_goal and scoring_team is not None:
            if car.team == scoring_team:
                ball_speed = float(np.linalg.norm(arena.ball.vel))
                speed_factor = 1.0 + (ball_speed / BALL_MAX_SPEED) * self.speed_multiplier
                return self.weight * speed_factor
            else:
                return self.concede_weight
        return 0.0


class SaveReward(BaseReward):
    """
    Major defensive reward for making a goal-line save / clear.
    """
    def __init__(self, weight: float = 50.0):
        super().__init__(weight)
        self._prev_touches: Dict[int, int] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_touches.get(car.id, 0)
        curr = car.ball_touches
        self._prev_touches[car.id] = curr
        if curr > prev:
            # Check if ball was heading towards car's defending goal net
            defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
            dist_to_defend = abs(arena.ball.pos[1] - defending_y)
            if dist_to_defend < 2000.0 and abs(arena.ball.pos[0]) < GOAL_HALF_WIDTH * 1.5:
                return self.weight
        return 0.0


class BoostManagementReward(BaseReward):
    """
    Rewards collecting boost pads (+delta boost gain).
    """
    def __init__(self, weight: float = 0.05):
        super().__init__(weight)
        self._prev_boost: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_boost = {car.id: car.boost for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_boost.get(car.id, 33.3)
        curr = car.boost
        self._prev_boost[car.id] = curr
        # Purely reward gaining boost from pads (+delta)
        if curr > prev:
            return (curr - prev) / 100.0 * self.weight
        return 0.0


class VelocityReward(BaseReward):
    """
    Rewards maintaining forward speed through the front bumper. Discourages reversing.
    """
    def __init__(self, weight: float = 0.02):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        fwd = car.get_forward_vector()
        fwd_speed = float(np.dot(car.vel, fwd))
        return self.weight * (fwd_speed / CAR_MAX_SPEED)


class AerialHeightReward(BaseReward):
    """
    Rewards aerial challenges ONLY when the ball is airborne (Z > 140 uu) and within challenging range.
    """
    def __init__(self, weight: float = 0.05):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        # Ball must be airborne (> 140 uu) to justify an aerial challenge
        if arena.ball.pos[2] > 140.0 and not car.on_ground and car.pos[2] > 30.0:
            car_to_ball = arena.ball.pos - car.pos
            dist = float(np.linalg.norm(car_to_ball))
            if dist < 2500.0:
                dist_factor = max(0.0, 1.0 - (dist / 2500.0))
                height_norm = min(1.0, (car.pos[2] - 17.0) / 400.0)
                flip_bonus = 1.5 if car.just_dodged else 1.0
                return self.weight * height_norm * flip_bonus * dist_factor
        return 0.0


class AlignedShotReward(BaseReward):
    """
    Rewards aligning and driving through the ball toward the opponent net.
    Strictly gates out standstill parking: car must be actively moving (>350 uu/s) toward the ball.
    """
    def __init__(self, weight: float = 0.05):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        car_speed = float(np.linalg.norm(car.vel))
        if car_speed < 350.0:
            return 0.0

        target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
        target_goal = np.array([0.0, target_goal_y, GOAL_HEIGHT * 0.5], dtype=np.float32)
        ball_to_goal = target_goal - arena.ball.pos
        norm_goal = np.linalg.norm(ball_to_goal)
        if norm_goal < 1e-4:
            return 0.0
        unit_ball_to_goal = ball_to_goal / norm_goal

        car_to_ball = arena.ball.pos - car.pos
        dist_to_ball = np.linalg.norm(car_to_ball)
        if dist_to_ball < 1e-4:
            return 0.0
        unit_car_to_ball = car_to_ball / dist_to_ball

        # Car must be driving toward the ball
        speed_toward_ball = float(np.dot(car.vel, unit_car_to_ball))
        if speed_toward_ball < 350.0:
            return 0.0

        alignment = float(np.dot(unit_car_to_ball, unit_ball_to_goal))
        if alignment > 0.0 and dist_to_ball < 2000.0:
            dist_factor = max(0.0, 1.0 - (dist_to_ball / 2000.0))
            norm_speed = min(1.0, speed_toward_ball / CAR_MAX_SPEED)
            return self.weight * alignment * dist_factor * norm_speed
        return 0.0


class KickoffReward(BaseReward):
    """
    Rewards rushing the ball at maximum speed specifically on kickoffs.
    """
    def __init__(self, weight: float = 0.05):
        super().__init__(weight)
        self._kickoff_ticks: Dict[int, int] = {}

    def reset(self, initial_state: RocketSimArena):
        self._kickoff_ticks = {car.id: 0 for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        ticks = self._kickoff_ticks.get(car.id, 0)
        self._kickoff_ticks[car.id] = ticks + 1

        is_kickoff = (abs(arena.ball.pos[0]) < 20.0 and abs(arena.ball.pos[1]) < 20.0 and np.linalg.norm(arena.ball.vel) < 80.0)
        if is_kickoff and ticks < 80:
            car_to_ball = arena.ball.pos - car.pos
            dist = float(np.linalg.norm(car_to_ball))
            if dist > 1e-4:
                unit_to_ball = car_to_ball / dist
                speed_toward = float(np.dot(car.vel, unit_to_ball))
                if speed_toward > 0:
                    return self.weight * (speed_toward / CAR_MAX_SPEED)
        return 0.0


class BehindBallReward(BaseReward):
    """
    Rewards staying between ball and defending goal; discourages over-committing.
    """
    def __init__(self, weight: float = 0.03):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        defending_goal_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_car_to_net = abs(car.pos[1] - defending_goal_y)
        dist_ball_to_net = abs(arena.ball.pos[1] - defending_goal_y)

        if dist_car_to_net < dist_ball_to_net:
            dist = float(np.linalg.norm(arena.ball.pos - car.pos))
            car_speed = float(np.linalg.norm(car.vel))
            if dist < 2500.0 and car_speed > 300.0:
                return self.weight
        return 0.0


class PossessionReward(BaseReward):
    """
    Rewards close-proximity dribbling and speed-matching when actively carrying the ball.
    Strictly gates out stationary ball/car parking to eliminate the standstill exploit.
    """
    def __init__(self, weight: float = 0.04):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        ball_speed = float(np.linalg.norm(arena.ball.vel))
        car_speed = float(np.linalg.norm(car.vel))
        
        # Ball and car must both be actively moving across the pitch (> 200 uu/s) to count as possession
        if ball_speed < 200.0 or car_speed < 200.0:
            return 0.0

        dist = float(np.linalg.norm(car.pos - arena.ball.pos))
        if dist < 350.0:
            rel_speed = float(np.linalg.norm(car.vel - arena.ball.vel))
            speed_match = max(0.0, 1.0 - (rel_speed / 1000.0))
            dist_factor = max(0.0, 1.0 - (dist / 350.0))
            return self.weight * dist_factor * (0.5 + 0.5 * speed_match)
        return 0.0


class DefensivePositionReward(BaseReward):
    """
    Rewards positioning on the direct line between defending net and ball when defending.
    """
    def __init__(self, weight: float = 0.03):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        is_defending = (arena.ball.pos[1] < 0) if car.team == 0 else (arena.ball.pos[1] > 0)
        if not is_defending:
            return 0.0

        defending_goal_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        net_pos = np.array([0.0, defending_goal_y, 0.0], dtype=np.float32)

        line_vec = arena.ball.pos - net_pos
        line_len = float(np.linalg.norm(line_vec))
        if line_len < 1e-4:
            return 0.0
        unit_line = line_vec / line_len

        car_from_net = car.pos - net_pos
        proj_dist = float(np.dot(car_from_net, unit_line))

        if 0.0 < proj_dist < line_len:
            perp_dist = float(np.linalg.norm(car_from_net - proj_dist * unit_line))
            alignment_score = max(0.0, 1.0 - (perp_dist / 1500.0))
            return self.weight * alignment_score
        return 0.0


class DemoBumpReward(BaseReward):
    """
    Rewards aggressive bumps and demolitions against opponents.
    """
    def __init__(self, weight: float = 15.0):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        opponents = [c for c in arena.cars if c.team != car.team]
        if not opponents:
            return 0.0

        reward = 0.0
        car_speed = float(np.linalg.norm(car.vel))
        for opp in opponents:
            dist = float(np.linalg.norm(car.pos - opp.pos))
            if dist < 180.0 and car_speed > 1600.0:
                reward += self.weight * (car_speed / CAR_MAX_SPEED)
        return reward


class BoostStealReward(BaseReward):
    """
    Rewards collecting big boost pads on the opponent's side of the pitch.
    """
    def __init__(self, weight: float = 10.0):
        super().__init__(weight)
        self._prev_boost: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_boost = {car.id: car.boost for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_boost.get(car.id, 33.3)
        curr = car.boost
        self._prev_boost[car.id] = curr

        if curr > prev + 50.0:  # Big pad pickup
            on_opp_half = (car.pos[1] > 0) if car.team == 0 else (car.pos[1] < 0)
            if on_opp_half:
                return self.weight
        return 0.0


class RewardManager:
    """
    Manages all reward functions and exposes dynamic runtime weight updates.
    Standardized so macro game events (Goals, Saves, Touches) dominate over continuous guidance.
    """
    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        weights = reward_weights or {}
        self.rewards = {
            # Flat Macro Events
            "goal": GoalReward(
                goal_weight=weights.get("goal_weight", 100.0),
                concede_weight=weights.get("concede_weight", -100.0),
                speed_multiplier=weights.get("goal_speed_multi", 1.5)
            ),
            "save": SaveReward(weights.get("save_weight", 50.0)),
            "touch_ball": TouchBallReward(
                weight=weights.get("touch_ball_weight", 10.0),
                first_touch_bonus=weights.get("kickoff_first_touch_bonus", 35.0),
                aerial_flip_multi=weights.get("touch_aerial_flip_multi", 2.5)
            ),
            "demo_bump": DemoBumpReward(weights.get("demo_bump_weight", 15.0)),
            "boost_steal": BoostStealReward(weights.get("boost_steal_weight", 10.0)),

            # Micro-Scaled Per-Step Guidance (~0.01 - 0.08 pts/step)
            "ball_vel_toward_goal": BallVelocityToGoalReward(weights.get("ball_vel_toward_goal_weight", 0.08)),
            "speed_toward_ball": SpeedTowardBallReward(
                weight=weights.get("speed_toward_ball_weight", 0.05),
                dodge_rush_multi=weights.get("dodge_rush_multi", 1.5)
            ),
            "kickoff": KickoffReward(weights.get("kickoff_weight", 0.05)),
            "face_ball": FaceBallReward(weights.get("face_ball_weight", 0.02)),
            "aligned_shot": AlignedShotReward(weights.get("aligned_shot_weight", 0.05)),
            "behind_ball": BehindBallReward(weights.get("behind_ball_weight", 0.03)),
            "possession": PossessionReward(weights.get("possession_weight", 0.04)),
            "defensive_position": DefensivePositionReward(weights.get("defensive_position_weight", 0.03)),
            "boost_management": BoostManagementReward(weights.get("boost_management_weight", 0.05)),
            "velocity": VelocityReward(weights.get("velocity_weight", 0.02)),
            "aerial_height": AerialHeightReward(weights.get("aerial_height_weight", 0.05)),
        }

    def reset(self, initial_state: RocketSimArena):
        for r in self.rewards.values():
            r.reset(initial_state)

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Dynamically update weights at runtime from GUI / live config.
        """
        mapping = {
            "touch_ball_weight": "touch_ball",
            "speed_toward_ball_weight": "speed_toward_ball",
            "ball_vel_toward_goal_weight": "ball_vel_toward_goal",
            "kickoff_weight": "kickoff",
            "face_ball_weight": "face_ball",
            "goal_weight": "goal",
            "save_weight": "save",
            "boost_management_weight": "boost_management",
            "velocity_weight": "velocity",
            "aerial_height_weight": "aerial_height",
            "aligned_shot_weight": "aligned_shot",
            "behind_ball_weight": "behind_ball",
            "possession_weight": "possession",
            "defensive_position_weight": "defensive_position",
            "demo_bump_weight": "demo_bump",
            "boost_steal_weight": "boost_steal",
        }

        for param_key, reward_name in mapping.items():
            if param_key in new_weights and reward_name in self.rewards:
                self.rewards[reward_name].weight = float(new_weights[param_key])

        # Goal specific params
        if "concede_weight" in new_weights and "goal" in self.rewards:
            self.rewards["goal"].concede_weight = float(new_weights["concede_weight"])
        if "goal_speed_multi" in new_weights and "goal" in self.rewards:
            self.rewards["goal"].speed_multiplier = float(new_weights["goal_speed_multi"])

        # Touch specific params
        if "kickoff_first_touch_bonus" in new_weights and "touch_ball" in self.rewards:
            self.rewards["touch_ball"].first_touch_bonus = float(new_weights["kickoff_first_touch_bonus"])
        if "touch_aerial_flip_multi" in new_weights and "touch_ball" in self.rewards:
            self.rewards["touch_ball"].aerial_flip_multi = float(new_weights["touch_aerial_flip_multi"])

        # Speed toward ball specific params
        if "dodge_rush_multi" in new_weights and "speed_toward_ball" in self.rewards:
            self.rewards["speed_toward_ball"].dodge_rush_multi = float(new_weights["dodge_rush_multi"])

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        breakdown = {}
        for name, r in self.rewards.items():
            rew = float(r.get_reward(car, arena, action, is_goal, scoring_team))
            total += rew
            breakdown[name] = rew
        return float(total), breakdown
