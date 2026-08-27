"""
Consolidated 6-Module Hierarchical Reward System for Rocket League Bot Reinforcement Learning.
Engineered for maximum training efficiency (3,000+ SPS), instant xG credit assignment,
and mathematical orthogonality with zero inter-module deadlocks.
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
# MODULE 1: MATCH MACRO EVENTS
# ==============================================================================
class GoalReward(BaseReward):
    """
    Rewards scoring goals (+250) and penalizes conceding goals (-100).
    Includes a high-speed goal velocity multiplier and a dedicated Save bounty (+50).
    """
    def __init__(self, goal_weight: float = 250.0, concede_weight: float = -100.0, save_weight: float = 50.0, speed_multiplier: float = 1.5):
        super().__init__(goal_weight)
        self.concede_weight = concede_weight
        self.save_weight = save_weight
        self.speed_multiplier = speed_multiplier
        self._prev_saves: Dict[int, int] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_saves = {car.id: 0 for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if is_goal and scoring_team is not None:
            if car.team == scoring_team:
                ball_speed = float(np.linalg.norm(arena.ball.vel))
                norm_speed = min(1.0, ball_speed / BALL_MAX_SPEED)
                return self.weight * (1.0 + (self.speed_multiplier - 1.0) * norm_speed)
            else:
                return self.concede_weight

        # Check defensive goal line save (clearing ball off defending goal line)
        defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_ball_to_net = abs(arena.ball.pos[1] - defending_y)
        dist_car_to_net = abs(car.pos[1] - defending_y)
        if dist_ball_to_net < 800.0 and dist_car_to_net < 1000.0 and abs(arena.ball.pos[0]) < GOAL_HALF_WIDTH * 1.5:
            ball_vy_out = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
            if ball_vy_out > 400.0 and car.ball_touches > self._prev_saves.get(car.id, 0):
                self._prev_saves[car.id] = car.ball_touches
                return self.save_weight

        return 0.0


# ==============================================================================
# MODULE 2: BALL STRIKE & xG SHOT ENGINE (Atomic Touch Events)
# ==============================================================================
class BallStrikeReward(BaseReward):
    """
    Atomic Ball Strike Engine. Evaluates touch quality, power, xG shot trajectory,
    roof flicks, high aerial strikes, and anti-own-goal deflection at the exact instant of impact.
    """
    def __init__(
        self,
        weight: float = 12.0,
        xg_shot_bounty: float = 40.0,
        high_aerial_bounty: float = 25.0,
        flick_bounty: float = 30.0,
        directional_dodge_bounty: float = 15.0,
        first_touch_bonus: float = 35.0
    ):
        super().__init__(weight)
        self.xg_shot_bounty = xg_shot_bounty
        self.high_aerial_bounty = high_aerial_bounty
        self.flick_bounty = flick_bounty
        self.directional_dodge_bounty = directional_dodge_bounty
        self.first_touch_bonus = first_touch_bonus
        self._prev_touches: Dict[int, int] = {}
        self._touch_cooldown: Dict[int, float] = {}
        self._prev_carrying: Dict[int, bool] = {}
        self._first_touch_claimed: bool = False

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._touch_cooldown = {car.id: 0.0 for car in initial_state.cars}
        self._prev_carrying = {car.id: False for car in initial_state.cars}
        b_pos = initial_state.ball.pos
        b_vel = initial_state.ball.vel
        self._is_kickoff_episode = bool(abs(b_pos[0]) < 50.0 and abs(b_pos[1]) < 50.0 and float(np.linalg.norm(b_vel)) < 100.0)
        self._first_touch_claimed = False

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_touches.get(car.id, 0)
        curr = car.ball_touches
        self._prev_touches[car.id] = curr

        cd = self._touch_cooldown.get(car.id, 0.0)
        if cd > 0.0:
            self._touch_cooldown[car.id] = max(0.0, cd - (1.0 / 15.0))

        rel_pos = arena.ball.pos - car.pos
        horiz_dist = float(np.linalg.norm(rel_pos[:2]))
        vert_dist = rel_pos[2]
        was_carrying = self._prev_carrying.get(car.id, False)
        is_carrying = (horiz_dist < 180.0 and 15.0 < vert_dist < 140.0)
        self._prev_carrying[car.id] = is_carrying

        if curr > prev and self._touch_cooldown.get(car.id, 0.0) <= 0.0:
            self._touch_cooldown[car.id] = 0.25

            # 1. Anti-Own-Goal Deflection Check
            defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
            in_defensive_half = (arena.ball.pos[1] < 0.0) if car.team == 0 else (arena.ball.pos[1] > 0.0)
            ball_vy_to_net = -arena.ball.vel[1] if car.team == 0 else arena.ball.vel[1]
            if in_defensive_half and ball_vy_to_net > 300.0 and abs(arena.ball.pos[0]) < GOAL_HALF_WIDTH * 2.5:
                return -30.0

            # 2. Kickoff First Touch Bounty
            first_bounty = 0.0
            if getattr(self, "_is_kickoff_episode", True) and not self._first_touch_claimed:
                first_bounty = self.first_touch_bonus * (1.4 if car.boost >= 10.0 else 1.0)
            self._first_touch_claimed = True

            # 3. Base Hit Power & Bumper Alignment
            ball_speed = float(np.linalg.norm(arena.ball.vel))
            power_factor = 0.5 + 0.5 * min(1.0, ball_speed / 1500.0)

            car_to_ball = arena.ball.pos - car.pos
            dist = float(np.linalg.norm(car_to_ball))
            bumper_alignment = 1.0
            if dist > 1e-4:
                unit_to_ball = car_to_ball / dist
                fwd = car.get_forward_vector()
                align = float(np.dot(fwd, unit_to_ball))
                rear_align = float(np.dot(-fwd, unit_to_ball))
                bumper_alignment = 1.0 + 0.8 * max(0.0, max(align, rear_align))

            # 4. Instant xG Shot on Target Raycast Bounty
            xg_bounty = 0.0
            target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
            ball_vy_to_target = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
            if ball_vy_to_target > 500.0:
                dy = target_goal_y - arena.ball.pos[1]
                dt = dy / (arena.ball.vel[1] if abs(arena.ball.vel[1]) > 1e-4 else 1e-4)
                if 0.05 < dt < 2.5:
                    pred_x = arena.ball.pos[0] + arena.ball.vel[0] * dt
                    pred_z = arena.ball.pos[2] + arena.ball.vel[2] * dt + 0.5 * (-650.0) * (dt ** 2)
                    if abs(pred_x) < GOAL_HALF_WIDTH and 0.0 < pred_z < GOAL_HEIGHT:
                        shot_speed_factor = min(1.5, max(1.0, ball_speed / 1200.0))
                        xg_bounty = self.xg_shot_bounty * shot_speed_factor

            # 5. High Aerial Strike / Roof Flick / Dodge Bounty
            air_bounty = 0.0
            if car.pos[2] > 200.0 and arena.ball.pos[2] > 240.0 and not car.on_ground:
                air_bounty = self.high_aerial_bounty
            elif was_carrying and car.just_dodged and ball_speed > 700.0:
                air_bounty = self.flick_bounty
            elif car.just_dodged and (abs(action[2]) > 0.3 or abs(action[3]) > 0.3 or abs(action[4]) > 0.3):
                air_bounty = self.directional_dodge_bounty

            return (self.weight * power_factor * bumper_alignment) + first_bounty + xg_bounty + air_bounty

        return 0.0


# ==============================================================================
# MODULE 3: 3D LOCOMOTION & POWERSLIDE (Navigation Engine)
# ==============================================================================
class LocomotionReward(BaseReward):
    """
    Consolidated 3D Locomotion Engine. Replaces Velocity, FaceBall, and SpeedTowardBall.
    Provides smooth continuous forward momentum modulated by alignment to target/ball,
    with built-in powerslide cuts, side-on turn-in guidance, and boost acceleration rush.
    """
    def __init__(self, weight: float = 0.06, dodge_rush_multi: float = 1.6):
        super().__init__(weight)
        self.dodge_rush_multi = dodge_rush_multi
        self.prev_steer: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self.prev_steer.clear()

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        # Determine Navigation Target: Intercept Predicted Bounce when Ball is Airborne
        is_airborne = (arena.ball.pos[2] > 200.0 or abs(arena.ball.vel[2]) > 250.0)
        target_pos = arena.ball.pos.copy()
        if is_airborne and hasattr(arena, "get_predicted_ball_pos"):
            pred_pos = arena.get_predicted_ball_pos(60)  # 0.5s trajectory forecast
            if pred_pos is not None:
                if car.on_ground or car.pos[2] < 120.0:
                    # Ground car paths to landing bounce spot
                    target_pos = np.array([pred_pos[0], pred_pos[1], max(93.0, min(pred_pos[2], 220.0))], dtype=np.float32)
                else:
                    target_pos = pred_pos

        car_to_target = target_pos - car.pos
        dist = float(np.linalg.norm(car_to_target))
        if dist < 1e-4:
            return 0.0
        unit_to_target = car_to_target / dist

        fwd = car.get_forward_vector()
        fwd_align = float(np.dot(fwd, unit_to_target))
        rear_align = float(np.dot(-fwd, unit_to_target))
        best_align = max(fwd_align, rear_align)

        car_speed = float(np.linalg.norm(car.vel))
        fwd_speed = float(np.dot(car.vel, fwd))
        norm_speed = max(0.0, fwd_speed) / CAR_MAX_SPEED
        speed_toward = float(np.dot(car.vel, unit_to_target))

        # Wrong-Side Defensive Check: suppress forward drive if between ball and own net in defensive third
        defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_car_to_net = abs(car.pos[1] - defending_y)
        dist_ball_to_net = abs(arena.ball.pos[1] - defending_y)
        if dist_car_to_net > dist_ball_to_net and dist_ball_to_net < 3000.0:
            norm_speed *= 0.15

        # Precision Cross-Track Error & Collision Corridor
        fwd_cross = float(np.linalg.norm(np.cross(car_to_target, fwd)))
        is_on_collision_course = (fwd_cross < 110.0 and fwd_align > 0.7)

        # Tactical Powerslide Turnaround Bounty & Drag Penalty
        turn_bonus = 0.0
        handbrake_drag_penalty = 0.0
        is_powersliding = bool(action[7] > 0.5)

        if car.on_ground and car_speed > 300.0:
            right = car.get_right_vector()
            lat_align = float(np.dot(right, unit_to_target))
            is_turning_in = (action[1] > 0.2 and lat_align > 0.1) or (action[1] < -0.2 and lat_align < -0.1)

            # 1. Tactical Turnaround Cut: ONLY when facing away (fwd_align < 0.3) and executing a sharp cut into the target
            if is_powersliding and is_turning_in and fwd_align < 0.3:
                turn_bonus = 0.05 * (1.0 - max(-1.0, fwd_align))

            # 2. Handbrake Drag Penalty: Penalize dragging handbrake in straightaways or when already facing the target
            if is_powersliding:
                if fwd_align > 0.6 or abs(action[1]) < 0.2:
                    handbrake_drag_penalty = -0.04  # Actively breaks the constant drifting addiction

        # Close-Range Bumper Contact Lock & Fly-By Whiff Penalty
        contact_bonus = 0.0
        whiff_penalty = 0.0
        if dist < 500.0:
            if fwd_align > 0.85 and fwd_cross < 75.0 and speed_toward > 300.0:
                contact_bonus = 0.06 * min(1.5, speed_toward / 1000.0)  # Direct bumper impact lock
            elif car_speed > 600.0 and (fwd_cross > 120.0 or fwd_align < 0.3):
                whiff_penalty = -0.04  # Penalize flying past the ball without making contact

        # Boost Acceleration Rush: active incentive for boosting towards target
        boost_rush = 0.0
        if action[6] > 0.0 and car_speed < 2150.0 and speed_toward > 800.0 and best_align > 0.4:
            boost_rush = 0.05 * min(1.0, speed_toward / 1600.0)

        # Kickoff Sprint Acceleration Rush & Collision Aiming
        kickoff_bonus = 0.0
        is_center_ball = (abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and float(np.linalg.norm(arena.ball.vel)) < 80.0)
        if is_center_ball and dist > 220.0:
            # Steep angular aiming accuracy to center ball
            aim_error_rad = math.acos(max(-1.0, min(1.0, fwd_align)))
            aim_accuracy = max(0.0, 1.0 - (aim_error_rad / math.radians(10.0)))
            
            # Rush sprint payout scaled by angular aim lock
            if speed_toward > 400.0:
                kickoff_bonus += 0.12 * aim_accuracy * min(1.5, speed_toward / 1400.0)

        # Smooth Proportional Steering Alignment & Straight-Line Guidance (Active across all ground driving)
        steer_alignment_bonus = 0.0
        if car.on_ground and car_speed > 250.0 and fwd_align > 0.0:
            right = car.get_right_vector()
            lat_offset = float(np.dot(right, unit_to_target))
            target_steer = float(np.clip(lat_offset * 3.5, -1.0, 1.0))
            steer_match = max(0.0, 1.0 - abs(float(action[1]) - target_steer))
            steer_alignment_bonus = 0.06 * steer_match

        # High-Speed Steering Chattering & Oscillations Damping (Eliminates Left-Right Fishtailing)
        steer_chatter_penalty = 0.0
        curr_steer = float(action[1])
        prev_steer = self.prev_steer.get(car.id, None)
        if car.on_ground and car_speed > 700.0 and prev_steer is not None:
            steer_delta = abs(curr_steer - prev_steer)
            if steer_delta > 0.9:
                steer_chatter_penalty = -0.025 * steer_delta
        self.prev_steer[car.id] = curr_steer

        # Airborne Wheels-Down Recovery & Tumble Damping Engine
        recovery_bonus = 0.0
        tumble_penalty = 0.0
        if not car.on_ground and car.pos[2] > 35.0:
            up_vec = car.get_up_vector()
            upright_align = float(up_vec[2])  # Dot product with world +Z [0, 0, 1]
            
            # Wheels-Down landing alignment bounty when descending towards surface
            if car.vel[2] < -50.0:
                if upright_align > 0.6:
                    recovery_bonus = 0.05 * upright_align  # Clean 4-wheel touchdown reward
                elif upright_align < -0.2:
                    recovery_bonus = -0.04  # Penalize inverted roof collisions and flopping

            # Angular rate tumble damping: penalize high-rate uncontrolled spinning
            ang_speed = float(np.linalg.norm(car.ang_vel))
            if not car.just_dodged and ang_speed > 3.5:
                tumble_penalty = -0.03 * min(1.5, (ang_speed - 3.5) / 2.0)

        # Bounce Anticipation Bonus: rewards closing speed toward airborne intercept point
        bounce_anticipation = 0.0
        if is_airborne and dist < 1600.0 and speed_toward > 500.0 and best_align > 0.5:
            bounce_anticipation = 0.04 * min(1.0, speed_toward / 1400.0)

        # Anti-Orbit Centrifugal Penalty: Penalize high-speed tangential looping around the ball
        orbit_penalty = 0.0
        if dist < 2500.0 and car_speed > 600.0:
            radial_vel = max(0.0, speed_toward)
            tangential_vel = math.sqrt(max(0.0, car_speed ** 2 - radial_vel ** 2))
            if tangential_vel > 700.0 and radial_vel < 250.0:
                orbit_penalty = -0.035  # Active donut loop breaker

        dodge_mult = self.dodge_rush_multi if (car.just_dodged and best_align > 0.5) else 1.0

        # Strict alignment gating: zero forward driving reward if pointing away/perpendicular to target
        if fwd_align <= 0.0:
            align_factor = 0.0
        elif dist < 1600.0:
            precision_factor = max(0.1, 1.0 - (fwd_cross / 350.0))
            align_factor = (fwd_align ** 2) * precision_factor
        else:
            align_factor = fwd_align ** 2

        return (self.weight * norm_speed * align_factor * dodge_mult) + turn_bonus + boost_rush + kickoff_bonus + bounce_anticipation + orbit_penalty + contact_bonus + whiff_penalty + recovery_bonus + tumble_penalty + steer_chatter_penalty + handbrake_drag_penalty + steer_alignment_bonus


# ==============================================================================
# MODULE 4: CONTEXT-AWARE TACTICAL AERIALS
# ==============================================================================
class TacticalAerialReward(BaseReward):
    """
    Context-Aware Tactical Aerial Engine.
    Evaluates aerial climbs ONLY when feasible (boost >= 15 & beating/challenging opponent, or defending goal threat).
    Requires nose-to-target alignment (> 0.5) to eliminate backwards flailing.
    Provides built-in Boost-Tax Shield (+0.04) during active flight.
    """
    def __init__(self, weight: float = 0.08, air_carry_weight: float = 0.06):
        super().__init__(weight)
        self.air_carry_weight = air_carry_weight

    def _get_time_to_ball(self, c: CarState, b_pos: np.ndarray) -> float:
        d = float(np.linalg.norm(b_pos - c.pos))
        if d < 1e-4:
            return 0.0
        unit = (b_pos - c.pos) / d
        closing = float(np.dot(c.vel, unit))
        effective = max(200.0, closing + (500.0 if c.boost > 10.0 else 100.0))
        return d / effective

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        # Determine Aerial Intercept Target: Aim for the future meeting point in the air
        target_pos = arena.ball.pos
        if hasattr(arena, "get_predicted_ball_pos"):
            pred_pos = arena.get_predicted_ball_pos(45)  # ~0.375s ahead flight intercept
            if pred_pos is not None and pred_pos[2] > 160.0:
                target_pos = pred_pos

        if target_pos[2] > 280.0:
            car_to_target = target_pos - car.pos
            dist = float(np.linalg.norm(car_to_target))
            if 1e-4 < dist < 2800.0:
                unit_to_target = car_to_target / dist
                dist_factor = max(0.0, 1.0 - (dist / 2800.0))

                defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
                in_defensive_box = abs(car.pos[1] - defending_y) < 2200.0
                is_threat, threat_intensity, threat_z = arena.get_shot_threat(car.team) if hasattr(arena, "get_shot_threat") else (False, 0.0, 0.0)

                # Boost feasibility gate
                if car.boost < 15.0 and not is_threat and not in_defensive_box:
                    return 0.0

                # Opponent time-to-ball contest
                tactical_mult = 1.0
                if not is_threat:
                    t_self = self._get_time_to_ball(car, target_pos)
                    opponents = [c for c in arena.cars if c.team != car.team and not c.demoed]
                    if opponents:
                        t_opp_min = min(self._get_time_to_ball(opp, target_pos) for opp in opponents)
                        if t_opp_min < t_self - 0.8:
                            return 0.0  # Late overcommit whiff
                        if t_self <= t_opp_min + 0.2:
                            tactical_mult = 1.4

                # Airborne flight tracking & climb (Strict Nose-Alignment Gate)
                if not car.on_ground and car.pos[2] > 35.0:
                    fwd = car.get_forward_vector()
                    fwd_align = float(np.dot(fwd, unit_to_target))
                    if fwd_align < 0.4:
                        return 0.0  # Zero reward for backwards/upside-down uncontrolled tumbles

                    height_norm = min(1.3, math.sqrt(max(0.0, (car.pos[2] - 17.0) / 400.0))) * (fwd_align ** 2)
                    flip_bonus = 1.4 if car.just_dodged else 1.0
                    threat_bonus = 1.8 if (is_threat or in_defensive_box) else 1.0
                    boost_shield = 0.04 if car.boost > 5.0 else 0.0

                    # Air dribble carry bonus (close flight beside elevated ball)
                    air_carry = 0.0
                    if dist < 400.0 and car.pos[2] > 140.0 and target_pos[2] > 160.0:
                        rel_vel = float(np.linalg.norm(car.vel - arena.ball.vel))
                        carry_match = max(0.0, 1.0 - (rel_vel / 800.0))
                        air_carry = self.air_carry_weight * carry_match

                    return (self.weight * height_norm * flip_bonus * dist_factor * threat_bonus * tactical_mult) + boost_shield + air_carry

                # Ground launch initiation
                if car.on_ground and action[5] > 0.0 and dist < 1800.0:
                    return self.weight * 1.2 * dist_factor * tactical_mult

        return 0.0


# ==============================================================================
# MODULE 5: TACTICAL POSITIONING & 50/50s
# ==============================================================================
class TacticalPositionReward(BaseReward):
    """
    Consolidated Positioning & 50/50 Challenge Engine. Replaces BehindBall, DefensivePosition,
    Possession, and InactivityPenalty.
    Rewards goal-side rotation, active 50/50 challenges, and goalkeeper positioning while penalizing open-field idling.
    """
    def __init__(self, weight: float = 0.04, inactivity_weight: float = 0.05, grace_steps: int = 45):
        super().__init__(weight)
        self.inactivity_weight = inactivity_weight
        self.grace_steps = grace_steps
        self._idle_ticks: Dict[int, int] = {}
        self._prev_pos: Dict[int, np.ndarray] = {}

    def reset(self, initial_state: RocketSimArena):
        self._idle_ticks = {car.id: 0 for car in initial_state.cars}
        self._prev_pos = {car.id: car.pos.copy() for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        dist_to_ball = float(np.linalg.norm(arena.ball.pos - car.pos))
        defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_car_to_net = abs(car.pos[1] - defending_y)
        
        # Trajectory-Aware Defense: Account for both current and predicted ball position on clears
        pred_pos = arena.get_predicted_ball_pos(60) if hasattr(arena, "get_predicted_ball_pos") else None
        pred_ball_y = pred_pos[1] if pred_pos is not None else arena.ball.pos[1]
        dist_ball_to_net = min(abs(arena.ball.pos[1] - defending_y), abs(pred_ball_y - defending_y))
        in_goal_box = (dist_car_to_net < 1800.0) and (abs(car.pos[0]) < 1200.0)

        # 1. Contested 50/50 Challenge Engine
        opponents = [c for c in arena.cars if c.team != car.team and not c.demoed]
        if opponents and dist_to_ball < 900.0:
            opp_dist = min(float(np.linalg.norm(arena.ball.pos - opp.pos)) for opp in opponents)
            if opp_dist < 900.0:
                car_speed = float(np.linalg.norm(car.vel))
                unit_to_ball = (arena.ball.pos - car.pos) / max(1e-4, dist_to_ball)
                speed_toward = float(np.dot(car.vel, unit_to_ball))
                if speed_toward > 400.0 or action[5] > 0.0 or car.just_dodged:
                    return 0.06  # 50/50 Commitment Bonus
                elif car_speed < 180.0:
                    return -0.04  # 50/50 Hesitation Standoff Penalty

        # 2. Defensive Goal-Side Shadowing & Recovery
        pos_reward = 0.0
        if dist_car_to_net < dist_ball_to_net and car.pos[2] < 400.0:
            unit_to_defending_goal = np.array([0.0, -1.0 if car.team == 0 else 1.0, 0.0], dtype=np.float32)
            speed_retreating = float(np.dot(car.vel, unit_to_defending_goal))
            fwd_defending = float(np.dot(car.get_forward_vector(), unit_to_defending_goal))

            # Reward active goalkeeper stance or purposeful retreat into net
            if in_goal_box:
                pos_reward = self.weight * 0.5
            elif speed_retreating > 300.0 and fwd_defending > 0.3 and dist_ball_to_net < 4200.0:
                pos_reward = self.weight * min(1.0, speed_retreating / 1200.0)

        # 3. Open-Field Inactivity & Non-Progression Drain
        horiz_speed = float(np.linalg.norm(car.vel[:2]))
        ticks = self._idle_ticks.get(car.id, 0)
        prev_p = self._prev_pos.get(car.id, car.pos)
        horiz_disp = float(np.linalg.norm(car.pos[:2] - prev_p[:2]))
        self._prev_pos[car.id] = car.pos.copy()

        # If ball is stagnant in center or far away and car is just looping/stagnant, increment ticks
        is_closing = (float(np.dot(car.vel, (arena.ball.pos - car.pos))) > 200.0)
        if dist_to_ball < 1200.0 or in_goal_box:
            ticks = max(0, ticks - 3)
        elif horiz_speed < 160.0 or horiz_disp < 10.0 or not is_closing:
            ticks += 1
        else:
            ticks = max(0, ticks - 1)

        self._idle_ticks[car.id] = ticks
        if ticks > self.grace_steps:
            escalation = min(4.0, 1.0 + (ticks - self.grace_steps) / 30.0)
            return pos_reward - (self.inactivity_weight * escalation)

        return pos_reward


# ==============================================================================
# MODULE 6: BOOST ECONOMY & SHIELDS
# ==============================================================================
class BoostEconomyReward(BaseReward):
    """
    Consolidated Boost Economy Engine. Replaces SmallPad, BigPad, BoostSteal, and SaveBoost.
    Awards pad collection bounties (+6 small, +18 big, +10 steal) and low-boost routing (+0.03),
    penalizes supersonic waste (-0.025), and grants Attack Immunity during offensive commits.
    """
    def __init__(self, small_pad_weight: float = 6.0, big_pad_weight: float = 18.0, boost_steal_weight: float = 10.0, save_boost_weight: float = 0.02):
        super().__init__(save_boost_weight)
        self.small_pad_weight = small_pad_weight
        self.big_pad_weight = big_pad_weight
        self.boost_steal_weight = boost_steal_weight
        self._prev_boost: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_boost = {car.id: car.boost for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_boost.get(car.id, car.boost)
        curr = car.boost
        self._prev_boost[car.id] = curr

        # 1. Big Orb Pickup & Steal Bounty
        if curr > prev + 40.0:
            on_opp_half = (car.pos[1] > 0) if car.team == 0 else (car.pos[1] < 0)
            return self.boost_steal_weight if on_opp_half else self.big_pad_weight

        # 2. Small Pad Pickup Bounty
        if (curr > prev + 5.0 and curr <= prev + 40.0) or (prev >= 88.0 and curr == 100.0 and prev < 100.0):
            return self.small_pad_weight

        car_speed = float(np.linalg.norm(car.vel))

        # 3. Supersonic Boost Waste Penalty
        if action[6] > 0.0 and car_speed >= 2150.0 and car.on_ground:
            return -self.weight * 1.5

        # 4. Attack Immunity Shield: zero boost penalty while actively attacking or shooting
        car_to_ball = arena.ball.pos - car.pos
        dist = float(np.linalg.norm(car_to_ball))
        if dist > 1e-4:
            unit_to_ball = car_to_ball / dist
            speed_toward = float(np.dot(car.vel, unit_to_ball))
            if speed_toward > 600.0 or not car.on_ground:
                return 0.0  # Boost consumption is 100% free during attack/flight!

        # 5. Low-Boost Pad Routing Guidance
        if car.boost < 40.0 and car.on_ground and hasattr(arena, "_small_pad_pos_3d") and hasattr(arena, "_small_pad_active"):
            sm_act = arena._small_pad_active
            if sm_act.any():
                act_pos = arena._small_pad_pos_3d[sm_act]
                diff = act_pos[:, :2] - car.pos[:2]
                d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
                min_i = int(np.argmin(d2))
                dist_sm_sq = float(d2[min_i])
                if 1.0 < dist_sm_sq < 2250000.0:
                    dist_sm = math.sqrt(dist_sm_sq)
                    unit_sm = diff[min_i] / dist_sm
                    fwd_2d = car.get_forward_vector()[:2]
                    fwd_2d_norm = float(np.linalg.norm(fwd_2d))
                    if fwd_2d_norm > 1e-4:
                        align_sm = float(np.dot(fwd_2d / fwd_2d_norm, unit_sm))
                        if align_sm > 0.6:
                            pad_prox = max(0.0, 1.0 - (dist_sm / 1500.0))
                            return 0.03 * align_sm * pad_prox

        if car_speed < 250.0:
            return 0.0
        return math.sqrt(max(0.0, car.boost / 100.0)) * self.weight


# ==============================================================================
# REWARD MANAGER (Dynamic Runtime Live Config Interface)
# ==============================================================================
class RewardManager:
    """
    Manages the 6 core tournament-grade reward modules and exposes dynamic runtime weight updates.
    """
    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        weights = reward_weights or {}
        self.rewards = {
            "goal": GoalReward(
                goal_weight=weights.get("goal_weight", 250.0),
                concede_weight=weights.get("concede_weight", -100.0),
                save_weight=weights.get("save_weight", 50.0),
                speed_multiplier=weights.get("goal_speed_multi", 1.5)
            ),
            "ball_strike": BallStrikeReward(
                weight=weights.get("touch_ball_weight", 12.0),
                xg_shot_bounty=weights.get("aligned_shot_weight", 40.0),
                high_aerial_bounty=weights.get("high_aerial_bounty", 25.0),
                flick_bounty=weights.get("flick_bounty", 30.0),
                directional_dodge_bounty=weights.get("directional_dodge_bounty", 15.0),
                first_touch_bonus=weights.get("kickoff_first_touch_bonus", 35.0)
            ),
            "locomotion": LocomotionReward(
                weight=weights.get("speed_toward_ball_weight", 0.06),
                dodge_rush_multi=weights.get("dodge_rush_multi", 1.6)
            ),
            "aerial": TacticalAerialReward(
                weight=weights.get("aerial_height_weight", 0.08),
                air_carry_weight=weights.get("air_dribble_carry_weight", 0.06)
            ),
            "positioning": TacticalPositionReward(
                weight=weights.get("behind_ball_weight", 0.04),
                inactivity_weight=weights.get("inactivity_penalty_weight", 0.05)
            ),
            "boost_economy": BoostEconomyReward(
                small_pad_weight=weights.get("small_pad_weight", 6.0),
                big_pad_weight=weights.get("big_pad_weight", 18.0),
                boost_steal_weight=weights.get("boost_steal_weight", 10.0),
                save_boost_weight=weights.get("save_boost_weight", 0.02)
            )
        }

    def reset(self, initial_state: RocketSimArena):
        for r in self.rewards.values():
            r.reset(initial_state)

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Dynamically update weights at runtime from GUI / live config.
        """
        if "goal_weight" in new_weights and "goal" in self.rewards:
            self.rewards["goal"].weight = float(new_weights["goal_weight"])
        if "concede_weight" in new_weights and "goal" in self.rewards:
            self.rewards["goal"].concede_weight = float(new_weights["concede_weight"])
        if "save_weight" in new_weights and "goal" in self.rewards:
            self.rewards["goal"].save_weight = float(new_weights["save_weight"])
        if "goal_speed_multi" in new_weights and "goal" in self.rewards:
            self.rewards["goal"].speed_multiplier = float(new_weights["goal_speed_multi"])

        if "touch_ball_weight" in new_weights and "ball_strike" in self.rewards:
            self.rewards["ball_strike"].weight = float(new_weights["touch_ball_weight"])
        if "aligned_shot_weight" in new_weights and "ball_strike" in self.rewards:
            self.rewards["ball_strike"].xg_shot_bounty = float(new_weights["aligned_shot_weight"])
        if "high_aerial_bounty" in new_weights and "ball_strike" in self.rewards:
            self.rewards["ball_strike"].high_aerial_bounty = float(new_weights["high_aerial_bounty"])
        if "flick_bounty" in new_weights and "ball_strike" in self.rewards:
            self.rewards["ball_strike"].flick_bounty = float(new_weights["flick_bounty"])
        if "directional_dodge_bounty" in new_weights and "ball_strike" in self.rewards:
            self.rewards["ball_strike"].directional_dodge_bounty = float(new_weights["directional_dodge_bounty"])
        if "kickoff_first_touch_bonus" in new_weights and "ball_strike" in self.rewards:
            self.rewards["ball_strike"].first_touch_bonus = float(new_weights["kickoff_first_touch_bonus"])

        if "speed_toward_ball_weight" in new_weights and "locomotion" in self.rewards:
            self.rewards["locomotion"].weight = float(new_weights["speed_toward_ball_weight"])
        if "dodge_rush_multi" in new_weights and "locomotion" in self.rewards:
            self.rewards["locomotion"].dodge_rush_multi = float(new_weights["dodge_rush_multi"])

        if "aerial_height_weight" in new_weights and "aerial" in self.rewards:
            self.rewards["aerial"].weight = float(new_weights["aerial_height_weight"])
        if "air_dribble_carry_weight" in new_weights and "aerial" in self.rewards:
            self.rewards["aerial"].air_carry_weight = float(new_weights["air_dribble_carry_weight"])

        if "behind_ball_weight" in new_weights and "positioning" in self.rewards:
            self.rewards["positioning"].weight = float(new_weights["behind_ball_weight"])
        if "inactivity_penalty_weight" in new_weights and "positioning" in self.rewards:
            self.rewards["positioning"].inactivity_weight = float(new_weights["inactivity_penalty_weight"])

        if "small_pad_weight" in new_weights and "boost_economy" in self.rewards:
            self.rewards["boost_economy"].small_pad_weight = float(new_weights["small_pad_weight"])
        if "big_pad_weight" in new_weights and "boost_economy" in self.rewards:
            self.rewards["boost_economy"].big_pad_weight = float(new_weights["big_pad_weight"])
        if "boost_steal_weight" in new_weights and "boost_economy" in self.rewards:
            self.rewards["boost_economy"].boost_steal_weight = float(new_weights["boost_steal_weight"])
        if "save_boost_weight" in new_weights and "boost_economy" in self.rewards:
            self.rewards["boost_economy"].weight = float(new_weights["save_boost_weight"])

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        breakdown = {}
        for name, r in self.rewards.items():
            rew = float(r.get_reward(car, arena, action, is_goal, scoring_team))
            total += rew
            breakdown[name] = rew
        return float(total), breakdown
