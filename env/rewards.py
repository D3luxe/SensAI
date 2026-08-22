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
    Scaled by hit power and aerial flip bonus.
    Rate-limited with a 0.25s cooldown to eliminate continuous contact / touch grinding exploits.
    """
    def __init__(self, weight: float = 10.0, first_touch_bonus: float = 35.0, aerial_flip_multi: float = 2.0):
        super().__init__(weight)
        self.first_touch_bonus = first_touch_bonus
        self.aerial_flip_multi = aerial_flip_multi
        self._prev_touches: Dict[int, int] = {}
        self._touch_cooldown: Dict[int, float] = {}
        self._first_touch_claimed: bool = False

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._touch_cooldown = {car.id: 0.0 for car in initial_state.cars}
        self._first_touch_claimed = False

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_touches.get(car.id, 0)
        curr = car.ball_touches
        self._prev_touches[car.id] = curr

        cd = self._touch_cooldown.get(car.id, 0.0)
        if cd > 0.0:
            self._touch_cooldown[car.id] = max(0.0, cd - (1.0 / 15.0))

        if curr > prev and self._touch_cooldown.get(car.id, 0.0) <= 0.0:
            self._touch_cooldown[car.id] = 0.25  # 250ms cooldown prevents grinding 15 touches per second

            # Kickoff First-Touch Bounty (controlled directly by the Kickoff First-Touch Bounty slider)
            first_bounty = self.first_touch_bonus if not self._first_touch_claimed else 0.0
            self._first_touch_claimed = True

            # Multiplier for jumping, dodging, or aerial hits
            is_jump_or_flip = (not car.on_ground) or car.just_dodged or (car.pos[2] > 25.0)
            aerial_flip_mult = self.aerial_flip_multi if is_jump_or_flip else 1.0

            # Power scaling: reward solid strikes over gentle grazing
            ball_speed = float(np.linalg.norm(arena.ball.vel))
            power_factor = 0.5 + 0.5 * min(1.0, ball_speed / 1500.0)

            # Base Hit Bounty + Kickoff First-Touch Bounty
            return (self.weight * aerial_flip_mult * power_factor) + first_bounty

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
    Rewards aligning the car nose directly towards the ball.
    Provides smooth continuous orientation gradient both at speed and from a dead stop.
    Eliminates standstill gridlocks and rewards tightening turns toward the ball.
    """
    def __init__(self, weight: float = 0.025):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        car_to_ball = arena.ball.pos - car.pos
        dist = float(np.linalg.norm(car_to_ball))
        if dist < 1e-4:
            return 0.0
        unit_to_ball = car_to_ball / dist

        fwd = car.get_forward_vector()
        alignment = float(np.dot(fwd, unit_to_ball))  # Range [-1.0, 1.0]

        if alignment > 0.0:
            # Proximity factor: stronger alignment incentive when closer to the ball
            dist_factor = 0.5 + 0.5 * max(0.0, 1.0 - (dist / 3000.0))
            return self.weight * (alignment ** 2) * dist_factor
        elif alignment < -0.4 and dist < 1500.0:
            # Gentle penalty for facing completely away from ball at close range (stops blind reversing / orbiting)
            return self.weight * alignment * 0.4

        return 0.0


class BallVelocityToGoalReward(BaseReward):
    """
    Rewards actively accelerating / propelling the ball toward the opponent net (delta velocity toward goal).
    Penalizes sending the ball towards defending net (own-goal prevention).
    Awards 0 reward for passive rolling / escorting without impact.
    """
    def __init__(self, weight: float = 0.08):
        super().__init__(weight)
        self._prev_vel_toward_goal: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_vel_toward_goal = {}
        for car in initial_state.cars:
            target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
            target_goal = np.array([0.0, target_goal_y, GOAL_HEIGHT * 0.5], dtype=np.float32)
            ball_to_goal = target_goal - initial_state.ball.pos
            dist = float(np.linalg.norm(ball_to_goal))
            if dist > 1e-4:
                unit = ball_to_goal / dist
                self._prev_vel_toward_goal[car.id] = float(np.dot(initial_state.ball.vel, unit))
            else:
                self._prev_vel_toward_goal[car.id] = 0.0

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
        target_goal = np.array([0.0, target_goal_y, GOAL_HEIGHT * 0.5], dtype=np.float32)

        ball_to_goal = target_goal - arena.ball.pos
        dist = float(np.linalg.norm(ball_to_goal))
        if dist < 1e-4:
            return 0.0
        unit_ball_to_goal = ball_to_goal / dist

        curr_vel_toward_goal = float(np.dot(arena.ball.vel, unit_ball_to_goal))
        prev_vel = self._prev_vel_toward_goal.get(car.id, curr_vel_toward_goal)
        self._prev_vel_toward_goal[car.id] = curr_vel_toward_goal

        # Delta velocity toward goal: only rewards actively accelerating the ball toward the net
        d_vel = curr_vel_toward_goal - prev_vel

        # If ball is accelerated toward opponent net
        if d_vel > 20.0:
            norm_d_vel = min(1.0, d_vel / 1500.0)
            return self.weight * norm_d_vel
        elif d_vel < -150.0 and curr_vel_toward_goal < 0.0:
            # Own-goal deflection penalty
            return -self.weight * 0.5

        return 0.0


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
    Strictly rate-limited with a 3.5s cooldown per defensive sequence (eliminates in-box touch farming).
    Requires the ball to be in the danger zone and actively cleared away from the net.
    """
    def __init__(self, weight: float = 50.0):
        super().__init__(weight)
        self._prev_touches: Dict[int, int] = {}
        self._save_cooldown: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._save_cooldown = {car.id: 0.0 for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_touches.get(car.id, 0)
        curr = car.ball_touches
        self._prev_touches[car.id] = curr

        # Update cooldown
        cd = self._save_cooldown.get(car.id, 0.0)
        if cd > 0.0:
            self._save_cooldown[car.id] = max(0.0, cd - (1.0 / 15.0))

        if curr > prev and self._save_cooldown.get(car.id, 0.0) <= 0.0:
            defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
            dist_to_defend = abs(arena.ball.pos[1] - defending_y)

            # Ball must be in defensive danger zone (< 2000 uu from defending goal)
            if dist_to_defend < 2000.0 and abs(arena.ball.pos[0]) < GOAL_HALF_WIDTH * 1.5:
                # The touch must propel the ball AWAY from the defending goal (positive Vy for Blue, negative Vy for Orange)
                ball_vy = arena.ball.vel[1]
                is_clearing = (ball_vy > 250.0) if car.team == 0 else (ball_vy < -250.0)

                if is_clearing:
                    self._save_cooldown[car.id] = 3.5
                    ball_speed = float(np.linalg.norm(arena.ball.vel))
                    power_scale = 1.0 + (ball_speed / BALL_MAX_SPEED) * 0.5
                    return self.weight * power_scale

        return 0.0


class SmallPadReward(BaseReward):
    """
    Rewards running over small boost pads (+12 boost) with a flat event bounty.
    Tracks both physical pad state transitions and boost delta to ensure high-tank pickups are rewarded.
    """
    def __init__(self, weight: float = 2.0):
        super().__init__(weight)
        self._prev_boost: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_boost = {car.id: car.boost for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_boost.get(car.id, car.boost)
        curr = car.boost
        self._prev_boost[car.id] = curr

        # Check proximity to any small pad that was just collected
        car_pos_2d = car.pos[:2]
        for pad in arena.boost_pads:
            if not pad.is_big and not pad.is_active and pad.cooldown_timer > (pad.respawn_time - 0.1):
                dist = float(np.linalg.norm(car_pos_2d - pad.pos[:2]))
                if dist < 180.0:
                    return self.weight

        # Fallback boost delta check for small pads (e.g. 5-30 boost increase or topping off tank)
        if (curr > prev + 5.0 and curr <= prev + 40.0) or (prev >= 88.0 and curr == 100.0 and prev < 100.0):
            return self.weight
        return 0.0


class BigPadReward(BaseReward):
    """
    Rewards collecting full boost orbs (+100 boost) with a flat event bounty.
    Tracks physical pad state transitions to ensure pickups at high boost tanks are reliably rewarded.
    """
    def __init__(self, weight: float = 5.0):
        super().__init__(weight)
        self._prev_boost: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_boost = {car.id: car.boost for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_boost.get(car.id, car.boost)
        curr = car.boost
        self._prev_boost[car.id] = curr

        # Check proximity to any big pad that was just collected
        car_pos_2d = car.pos[:2]
        for pad in arena.boost_pads:
            if pad.is_big and not pad.is_active and pad.cooldown_timer > (pad.respawn_time - 0.1):
                dist = float(np.linalg.norm(car_pos_2d - pad.pos[:2]))
                if dist < 250.0:
                    return self.weight

        # Fallback boost delta check for big pads
        if curr > prev + 40.0:
            return self.weight
        return 0.0


class SaveBoostReward(BaseReward):
    """
    Rewards maintaining healthy boost tank reserves using the concave sqrt(boost / 100) curve.
    Strictly motion-gated: only active when actively moving (> 250 uu/s) to eliminate standstill parking.
    """
    def __init__(self, weight: float = 0.02):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        car_speed = float(np.linalg.norm(car.vel))
        if car_speed < 250.0:
            return 0.0
        return math.sqrt(max(0.0, car.boost / 100.0)) * self.weight


class VelocityReward(BaseReward):
    """
    Rewards maintaining forward speed through the front bumper. Discourages reversing.
    When close to the ball (< 1200 uu), gates reward by ball alignment to eliminate high-speed donut orbiting.
    """
    def __init__(self, weight: float = 0.02):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        fwd = car.get_forward_vector()
        fwd_speed = float(np.dot(car.vel, fwd))
        if fwd_speed <= 0.0:
            return 0.0

        norm_speed = fwd_speed / CAR_MAX_SPEED

        # Orbiting elimination: if close to the ball (< 1200 uu), velocity must be directed generally toward the ball
        car_to_ball = arena.ball.pos - car.pos
        dist = float(np.linalg.norm(car_to_ball))
        if dist < 1200.0 and dist > 1e-4:
            unit_to_ball = car_to_ball / dist
            alignment = float(np.dot(fwd, unit_to_ball))
            if alignment < 0.2:
                # Driving fast tangent/perpendicular to close ball earns 0 (kills orbiting reward loop)
                return 0.0
            align_scale = max(0.2, alignment)
            return self.weight * norm_speed * align_scale

        return self.weight * norm_speed


class AerialHeightReward(BaseReward):
    """
    Rewards jumping and aerial challenges when the ball is genuinely airborne (Z > 180 uu) and approaching.
    Strictly gates ground jump feedback so bots cannot farm points bunny-hopping beside ground balls.
    """
    def __init__(self, weight: float = 0.05):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        # Ball must be genuinely airborne (> 180 uu - well above car height) to justify an aerial challenge
        if arena.ball.pos[2] > 180.0:
            car_to_ball = arena.ball.pos - car.pos
            dist = float(np.linalg.norm(car_to_ball))
            if dist < 2200.0 and dist > 1e-4:
                unit_to_ball = car_to_ball / dist
                speed_toward = float(np.dot(car.vel, unit_to_ball))
                dist_factor = max(0.0, 1.0 - (dist / 2200.0))
                
                # Case 1: Airborne flight tracking towards elevated ball
                if not car.on_ground and car.pos[2] > 35.0 and speed_toward > 100.0:
                    height_norm = min(1.0, (car.pos[2] - 17.0) / 400.0)
                    flip_bonus = 1.5 if car.just_dodged else 1.0
                    return self.weight * height_norm * flip_bonus * dist_factor
                
                # Case 2: Ground launch initiation (only when actively rushing an airborne ball at high speed)
                if car.on_ground and action[5] > 0.0 and speed_toward > 450.0 and dist < 1200.0:
                    return self.weight * 0.8 * dist_factor
        return 0.0


class GroundToAirSetupReward(BaseReward):
    """
    Rewards popping a ground ball upward into the air (self-pass setup).
    Triggers when a grounded car impacts a low ball and imparts strong upward vertical velocity (dz > +350 uu/s).
    """
    def __init__(self, weight: float = 8.0):
        super().__init__(weight)
        self._prev_ball_vz = 0.0
        self._setup_cooldown: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_ball_vz = initial_state.ball.vel[2]
        self._setup_cooldown = {car.id: 0.0 for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        curr_vz = arena.ball.vel[2]
        d_vz = curr_vz - self._prev_ball_vz
        self._prev_ball_vz = curr_vz

        cd = self._setup_cooldown.get(car.id, 0.0)
        if cd > 0.0:
            self._setup_cooldown[car.id] = max(0.0, cd - (1.0 / 15.0))

        # Must be on ground/low near ball, ball was low (Z < 180), and imparted vertical pop
        if cd <= 0.0 and car.pos[2] < 60.0 and arena.ball.pos[2] < 180.0:
            dist = float(np.linalg.norm(arena.ball.pos - car.pos))
            if dist < 220.0 and d_vz > 350.0:
                self._setup_cooldown[car.id] = 2.5
                pop_scale = min(1.5, d_vz / 600.0)
                return self.weight * pop_scale

        return 0.0


class WallAerialLaunchReward(BaseReward):
    """
    Rewards popping the ball off the sidewall and immediately launching off the wall into an aerial pursuit.
    """
    def __init__(self, weight: float = 12.0):
        super().__init__(weight)
        self._wall_touch_timer: Dict[int, float] = {}
        self._wall_cooldown: Dict[int, float] = {}
        self._prev_touches: Dict[int, int] = {}

    def reset(self, initial_state: RocketSimArena):
        self._wall_touch_timer = {car.id: 0.0 for car in initial_state.cars}
        self._wall_cooldown = {car.id: 0.0 for car in initial_state.cars}
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev_t = self._prev_touches.get(car.id, 0)
        curr_t = car.ball_touches
        self._prev_touches[car.id] = curr_t

        cd = self._wall_cooldown.get(car.id, 0.0)
        if cd > 0.0:
            self._wall_cooldown[car.id] = max(0.0, cd - (1.0 / 15.0))

        # Detect wall touch: car on sidewall (|x| > 3400, z > 250) touches ball
        if curr_t > prev_t and abs(car.pos[0]) > 3400.0 and car.pos[2] > 250.0:
            self._wall_touch_timer[car.id] = 0.6  # 600ms window to jump off wall

        t_timer = self._wall_touch_timer.get(car.id, 0.0)
        if t_timer > 0.0:
            self._wall_touch_timer[car.id] = max(0.0, t_timer - (1.0 / 15.0))
            # If bot jumps off wall during this window towards the ball into midfield
            if action[5] > 0.0 and cd <= 0.0:
                dist = float(np.linalg.norm(arena.ball.pos - car.pos))
                if dist < 800.0:
                    self._wall_cooldown[car.id] = 3.5
                    self._wall_touch_timer[car.id] = 0.0
                    return self.weight

        return 0.0


class AirDribbleCarryReward(BaseReward):
    """
    Rewards airborne velocity matching and ball carrying towards the opponent goal (Air-Dribble).
    Active when both car and ball are elevated (Z > 220 uu), in close proximity, and moving goal-bound.
    """
    def __init__(self, weight: float = 0.06):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        # Both car and ball must be genuinely airborne
        if not car.on_ground and car.pos[2] > 200.0 and arena.ball.pos[2] > 220.0:
            car_to_ball = arena.ball.pos - car.pos
            dist = float(np.linalg.norm(car_to_ball))
            if dist < 250.0:
                # Relative speed matching (soft touch / carry)
                rel_speed = float(np.linalg.norm(car.vel - arena.ball.vel))
                speed_match = max(0.0, 1.0 - (rel_speed / 450.0))
                
                # Goal direction alignment
                target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
                target_goal = np.array([0.0, target_goal_y, GOAL_HEIGHT * 0.5], dtype=np.float32)
                ball_to_goal = target_goal - arena.ball.pos
                norm_goal = np.linalg.norm(ball_to_goal)
                if norm_goal > 1e-4:
                    unit_goal = ball_to_goal / norm_goal
                    ball_vy_goal = float(np.dot(arena.ball.vel, unit_goal))
                    goal_factor = max(0.0, min(1.0, ball_vy_goal / 1200.0))
                    
                    dist_factor = max(0.0, 1.0 - (dist / 250.0))
                    height_factor = min(1.2, car.pos[2] / 500.0)
                    
                    return self.weight * speed_match * (0.3 + 0.7 * goal_factor) * dist_factor * height_factor

        return 0.0


class AlignedShotReward(BaseReward):
    """
    Major flat event reward granted strictly upon striking the ball on target toward the opponent net.
    Strictly rate-limited with a 3.5s cooldown per shot sequence (eliminates multi-touch dribble farming).
    """
    def __init__(self, weight: float = 25.0):
        super().__init__(weight)
        self._prev_touches: Dict[int, int] = {}
        self._shot_cooldown: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._shot_cooldown = {car.id: 0.0 for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_touches.get(car.id, 0)
        curr = car.ball_touches
        self._prev_touches[car.id] = curr

        # Update cooldown
        cd = self._shot_cooldown.get(car.id, 0.0)
        if cd > 0.0:
            self._shot_cooldown[car.id] = max(0.0, cd - (1.0 / 15.0))

        # Only evaluate on a fresh ball contact when not in shot cooldown
        if curr > prev and self._shot_cooldown.get(car.id, 0.0) <= 0.0:
            target_heading_positive = (car.team == 0)
            ball_vy = arena.ball.vel[1]
            ball_speed = float(np.linalg.norm(arena.ball.vel))

            # Must be an actual forward strike (> 600 uu/s) toward opponent half
            if ((target_heading_positive and ball_vy > 400.0) or (not target_heading_positive and ball_vy < -400.0)) and ball_speed > 600.0:
                is_goal_bound = False
                if hasattr(arena, "_rsim_arena") and arena._rsim_arena is not None:
                    is_goal_bound = arena._rsim_arena.is_ball_probably_going_in(max_time=3.0)
                else:
                    target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
                    target_goal = np.array([0.0, target_goal_y, GOAL_HEIGHT * 0.5], dtype=np.float32)
                    ball_to_goal = target_goal - arena.ball.pos
                    norm_goal = np.linalg.norm(ball_to_goal)
                    if norm_goal > 1e-4:
                        unit_to_goal = ball_to_goal / norm_goal
                        alignment = float(np.dot(arena.ball.vel / ball_speed, unit_to_goal))
                        is_goal_bound = (alignment > 0.7)

                if is_goal_bound:
                    # Lock cooldown for 3.5 seconds to ensure this shot is only rewarded ONCE per attempt
                    self._shot_cooldown[car.id] = 3.5
                    power_scale = 1.0 + (ball_speed / BALL_MAX_SPEED) * 0.5
                    return self.weight * power_scale

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
            # Require car to be generally centered/on-pitch (not riding high up on the sidewall when ball is on floor)
            if dist < 2500.0 and car_speed > 300.0 and car.pos[2] < 400.0:
                return self.weight
        return 0.0


class PossessionReward(BaseReward):
    """
    Layer 1: Tactical Time-to-Ball Space Dominance.
    Calculates time-to-ball (T_reach) for self vs nearest opponent.
    Rewards uncontested space when T_self < T_opp (teaches patience and field control; eliminates 50/50 side-by-side farming).
    """
    def __init__(self, weight: float = 0.04):
        super().__init__(weight)

    def _get_time_to_ball(self, car: CarState, ball_pos: np.ndarray) -> float:
        car_to_ball = ball_pos - car.pos
        dist = float(np.linalg.norm(car_to_ball))
        if dist < 1e-4:
            return 0.0
        unit_to_ball = car_to_ball / dist
        closing_speed = float(np.dot(car.vel, unit_to_ball))
        effective_speed = max(150.0, closing_speed + (500.0 if car.boost > 10.0 else 150.0))
        return dist / effective_speed

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        ball_speed = float(np.linalg.norm(arena.ball.vel))
        car_speed = float(np.linalg.norm(car.vel))
        if ball_speed < 150.0 and car_speed < 150.0:
            return 0.0

        t_self = self._get_time_to_ball(car, arena.ball.pos)

        # Find nearest opponent time-to-ball
        opponents = [c for c in arena.cars if c.team != car.team and not c.demoed]
        if not opponents:
            return self.weight * 0.5

        t_opp_min = min(self._get_time_to_ball(opp, arena.ball.pos) for opp in opponents)

        # Time-to-ball differential: T_opp - T_self
        delta_t = t_opp_min - t_self

        # If opponent reaches ball first or 50/50 contest (delta_t <= 0.05s), 0 possession reward
        if delta_t <= 0.05:
            return 0.0

        # Uncontested possession scaled by time cushion (up to 1.5s margin)
        time_cushion = min(1.0, (delta_t - 0.05) / 1.5)
        
        # Proximity scaling: higher when within reasonable playing distance (< 2500 uu)
        dist_to_ball = float(np.linalg.norm(arena.ball.pos - car.pos))
        dist_factor = max(0.0, 1.0 - (dist_to_ball / 2500.0))

        return self.weight * time_cushion * (0.4 + 0.6 * dist_factor)


class DribbleReward(BaseReward):
    """
    Layer 2: Mechanical Roof Carry & Close Bumper Dribbling.
    Rewards balancing the ball on the car roof or pushing it with precision speed-matching.
    """
    def __init__(self, weight: float = 0.04):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        ball_speed = float(np.linalg.norm(arena.ball.vel))
        car_speed = float(np.linalg.norm(car.vel))
        if ball_speed < 200.0 or car_speed < 200.0:
            return 0.0

        rel_pos = arena.ball.pos - car.pos
        horiz_dist = float(np.linalg.norm(rel_pos[:2]))
        vert_dist = rel_pos[2]

        # Ball must be close horizontally (< 180 uu) and above/on roof (15 < dz < 140 uu)
        if horiz_dist < 180.0 and 15.0 < vert_dist < 140.0:
            rel_vel = float(np.linalg.norm(car.vel - arena.ball.vel))
            speed_match = max(0.0, 1.0 - (rel_vel / 600.0))
            horiz_factor = max(0.0, 1.0 - (horiz_dist / 180.0))
            # Roof carry bonus (ball directly atop car Z)
            roof_bonus = 1.5 if (vert_dist > 35.0 and horiz_dist < 100.0) else 1.0
            return self.weight * horiz_factor * speed_match * roof_bonus

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

        on_opp_half = (car.pos[1] > 0) if car.team == 0 else (car.pos[1] < 0)
        if on_opp_half:
            car_pos_2d = car.pos[:2]
            for pad in arena.boost_pads:
                if pad.is_big and not pad.is_active and pad.cooldown_timer > (pad.respawn_time - 0.1):
                    pad_on_opp_half = (pad.pos[1] > 0) if car.team == 0 else (pad.pos[1] < 0)
                    if pad_on_opp_half:
                        dist = float(np.linalg.norm(car_pos_2d - pad.pos[:2]))
                        if dist < 250.0:
                            return self.weight

            if curr > prev + 40.0:
                return self.weight
        return 0.0


class InactivityPenaltyReward(BaseReward):
    """
    Escalating per-step penalty assessed when a bot sits stationary, wiggles, or hops in place without meaningful horizontal displacement.
    Eliminates mutual standstills, midfield staring, hopping traps, and parking equilibria.
    """
    def __init__(self, weight: float = 0.05, grace_steps: int = 15):
        super().__init__(weight)
        self.grace_steps = grace_steps
        self._idle_ticks: Dict[int, int] = {}
        self._prev_pos: Dict[int, np.ndarray] = {}

    def reset(self, initial_state: RocketSimArena):
        self._idle_ticks = {car.id: 0 for car in initial_state.cars}
        self._prev_pos = {car.id: car.pos.copy() for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        horiz_speed = float(np.linalg.norm(car.vel[:2]))
        ticks = self._idle_ticks.get(car.id, 0)
        
        prev_p = self._prev_pos.get(car.id, car.pos)
        horiz_disp = float(np.linalg.norm(car.pos[:2] - prev_p[:2]))
        self._prev_pos[car.id] = car.pos.copy()

        # Idling, oscillating, or hopping in place with low net horizontal speed/displacement
        if horiz_speed < 160.0 or horiz_disp < 10.0:
            ticks += 1
        else:
            ticks = max(0, ticks - 2)

        self._idle_ticks[car.id] = ticks

        if ticks > self.grace_steps:
            # Escalates up to 4x penalty as prolonged idling/hopping continues
            escalation = min(4.0, 1.0 + (ticks - self.grace_steps) / 30.0)
            return -self.weight * escalation
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
            "aligned_shot": AlignedShotReward(weights.get("aligned_shot_weight", 25.0)),
            "touch_ball": TouchBallReward(
                weight=weights.get("touch_ball_weight", 10.0),
                first_touch_bonus=weights.get("kickoff_first_touch_bonus", 35.0),
                aerial_flip_multi=weights.get("touch_aerial_flip_multi", 2.5)
            ),
            "small_pad": SmallPadReward(weights.get("small_pad_weight", 2.0)),
            "big_pad": BigPadReward(weights.get("big_pad_weight", 5.0)),
            "demo_bump": DemoBumpReward(weights.get("demo_bump_weight", 15.0)),
            "boost_steal": BoostStealReward(weights.get("boost_steal_weight", 10.0)),
            "ground_to_air_setup": GroundToAirSetupReward(weights.get("ground_to_air_setup_weight", 8.0)),
            "wall_aerial_launch": WallAerialLaunchReward(weights.get("wall_aerial_launch_weight", 12.0)),

            # Micro-Scaled Per-Step Guidance (~0.01 - 0.08 pts/step)
            "ball_vel_toward_goal": BallVelocityToGoalReward(weights.get("ball_vel_toward_goal_weight", 0.08)),
            "speed_toward_ball": SpeedTowardBallReward(
                weight=weights.get("speed_toward_ball_weight", 0.05),
                dodge_rush_multi=weights.get("dodge_rush_multi", 1.5)
            ),
            "kickoff": KickoffReward(weights.get("kickoff_weight", 0.05)),
            "face_ball": FaceBallReward(weights.get("face_ball_weight", 0.02)),
            "behind_ball": BehindBallReward(weights.get("behind_ball_weight", 0.03)),
            "possession": PossessionReward(weights.get("possession_weight", 0.04)),
            "dribble": DribbleReward(weights.get("dribble_weight", 0.04)),
            "air_dribble_carry": AirDribbleCarryReward(weights.get("air_dribble_carry_weight", 0.06)),
            "defensive_position": DefensivePositionReward(weights.get("defensive_position_weight", 0.03)),
            "save_boost": SaveBoostReward(weights.get("save_boost_weight", 0.02)),
            "velocity": VelocityReward(weights.get("velocity_weight", 0.02)),
            "aerial_height": AerialHeightReward(weights.get("aerial_height_weight", 0.05)),
            "inactivity_penalty": InactivityPenaltyReward(weights.get("inactivity_penalty_weight", 0.05)),
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
            "small_pad_weight": "small_pad",
            "big_pad_weight": "big_pad",
            "save_boost_weight": "save_boost",
            "velocity_weight": "velocity",
            "aerial_height_weight": "aerial_height",
            "aligned_shot_weight": "aligned_shot",
            "behind_ball_weight": "behind_ball",
            "possession_weight": "possession",
            "dribble_weight": "dribble",
            "air_dribble_carry_weight": "air_dribble_carry",
            "ground_to_air_setup_weight": "ground_to_air_setup",
            "wall_aerial_launch_weight": "wall_aerial_launch",
            "defensive_position_weight": "defensive_position",
            "demo_bump_weight": "demo_bump",
            "boost_steal_weight": "boost_steal",
            "inactivity_penalty_weight": "inactivity_penalty",
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
