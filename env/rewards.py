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
    CAR_MAX_SPEED, BALL_MAX_SPEED, GOAL_HALF_WIDTH, GOAL_HEIGHT, ARENA_EXTENT_X, ARENA_EXTENT_Y
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
    Rewards scoring goals (+30.0), penalizes conceding (-30.0),
    and rewards defensive saves/clears off the goal line (+8.0).
    """
    def __init__(self, goal_weight: float = 30.0, concede_weight: float = -30.0, save_weight: float = 12.0):
        super().__init__(goal_weight)
        self.concede_weight = concede_weight
        self.save_weight = save_weight
        self._prev_touches: Dict[int, int] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if is_goal and scoring_team is not None:
            self._prev_touches[car.id] = car.ball_touches
            return self.weight if car.team == scoring_team else self.concede_weight

        # Defensive Goal-Line Save & Clear
        defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_ball_to_net = abs(arena.ball.pos[1] - defending_y)
        dist_car_to_net = abs(car.pos[1] - defending_y)

        if dist_ball_to_net < 1400.0 and dist_car_to_net < 1600.0 and abs(arena.ball.pos[0]) < GOAL_HALF_WIDTH * 1.6:
            ball_vy_out = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
            ball_vx_mag = abs(arena.ball.vel[0])
            prev_t = self._prev_touches.get(car.id, 0)
            if car.ball_touches > prev_t and (ball_vy_out > 150.0 or (ball_vx_mag > 350.0 and ball_vy_out >= -100.0)):
                self._prev_touches[car.id] = car.ball_touches
                return self.save_weight

        self._prev_touches[car.id] = car.ball_touches
        return 0.0


# ==============================================================================
# 2. BALL-TO-GOAL PROGRESSION (Field Displacement & On-Target Trajectory)
# ==============================================================================
class BallToGoalVelocityReward(BaseReward):
    """
    Continuous Potential-Based Progression with Goal Opening Targeting.
    Rewards ball velocity directed toward the opponent's goal opening (X in [-GOAL_HALF_WIDTH, +GOAL_HALF_WIDTH]).
    Heavily bonuses on-target trajectories that enter the net (1.6x), while dampening
    wide shots that roll into the backwall/corner beside the goal.
    Applies an asymmetric 1.5x penalty when ball velocity is directed towards the defending net.
    """
    def __init__(self, weight: float = 1.5):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if is_goal:
            return 0.0

        target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
        # If ball has already crossed the target endline into the net opening, do not invert vector
        if (car.team == 0 and arena.ball.pos[1] >= target_goal_y) or (car.team == 1 and arena.ball.pos[1] <= target_goal_y):
            return 0.0

        target_x = float(np.clip(arena.ball.pos[0], -GOAL_HALF_WIDTH * 0.8, GOAL_HALF_WIDTH * 0.8))
        target_pos = np.array([target_x, target_goal_y, GOAL_HEIGHT * 0.35], dtype=np.float32)

        ball_to_goal = target_pos - arena.ball.pos
        dist = float(np.linalg.norm(ball_to_goal))
        if dist < 1e-4:
            return 0.0

        unit_to_goal = ball_to_goal / dist
        ball_velocity_toward_goal = float(np.dot(arena.ball.vel, unit_to_goal))

        # Asymmetric penalty for advancing ball toward defending net
        if ball_velocity_toward_goal < 0.0:
            normalized_progress = (ball_velocity_toward_goal / BALL_MAX_SPEED) * 1.5
            return self.weight * normalized_progress

        # On-Target Trajectory & Backwall Miss Multiplier:
        # If ball is moving downfield into attacking half, calculate where its trajectory intersects the opponent endline
        vy_forward = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
        ball_y_forward = arena.ball.pos[1] if car.team == 0 else -arena.ball.pos[1]
        on_target_mult = 1.0
        if vy_forward > 50.0:
            delta_y = abs(target_goal_y - arena.ball.pos[1])
            dt = delta_y / vy_forward
            x_impact = arena.ball.pos[0] + arena.ball.vel[0] * dt
            if abs(x_impact) <= GOAL_HALF_WIDTH:
                # Shot is directly on target into the net opening!
                on_target_mult = 1.6
            elif ball_y_forward > 1000.0:
                # Ball is in attacking half and heading wide into the backwall/corner
                # Strictly eliminate progression reward so the bot cannot farm pushing wide
                miss_dist = abs(x_impact) - GOAL_HALF_WIDTH
                if miss_dist > 400.0:
                    on_target_mult = 0.0
                else:
                    on_target_mult = max(0.0, 1.0 - (miss_dist / 400.0))
            elif abs(x_impact) > GOAL_HALF_WIDTH * 1.3 and ball_y_forward > 0.0:
                # Midfield wide trajectory dampening
                miss_factor = min(1.0, (abs(x_impact) - GOAL_HALF_WIDTH) / 1500.0)
                on_target_mult = max(0.20, 1.0 - (0.80 * miss_factor))

        normalized_progress = (ball_velocity_toward_goal / BALL_MAX_SPEED) * on_target_mult
        return self.weight * normalized_progress


# ==============================================================================
# 3. PLAYER-TO-BALL DISTANCE DELTA & AERIAL INTERCEPT (Pursuit & Pacing)
# ==============================================================================
class PlayerToBallVelocityReward(BaseReward):
    """
    Necto / RLGym Potential-Based Distance Delta Approach & Aerial Intercept Reward.
    - Grounded Ball (Z < 300): Evaluates 2D horizontal distance delta so jumping for flips does not register an artificial penalty.
    - Elevated Aerial Ball (Z >= 300): Evaluates true 3D intercept distance, rewarding climbing velocity in the air and dampening floor-circling underneath floating balls.
    - Strike Zone Pacing (< 450 uu): Seamlessly transitions from downfield rush to strike-zone velocity matching.
    - Anti-Overshoot Penalty: Punishes blasting past the ball along the attack axis without touching it.
    - Deceleration / Braking Incentive: Rewards braking (throttle < 0) when closing dangerously fast on a slow ball from behind.
    - Wrong-Side & Own-Goal Guard: Suppresses velocity matching and pursuit rewards when driving behind the ball towards own net, and eliminates distance-delta cliffs when peeling away.
    """
    def __init__(self, weight: float = 0.6):
        super().__init__(weight)
        self._prev_dist: Dict[int, float] = {}
        self._prev_touches: Dict[int, int] = {}
        self._was_in_strike_zone: Dict[int, bool] = {}

    def _calc_dist(self, car_pos: np.ndarray, ball_pos: np.ndarray) -> float:
        # If BOTH car and ball are near pitch floor (car Z < 200, ball Z < 300), evaluate horizontal (X, Y) distance
        # so low ground flips / wavedashes do not incur an artificial vertical distance penalty.
        # If car is elevated on the wall or in the air (Z >= 200), strictly evaluate full 3D distance.
        if ball_pos[2] < 300.0 and car_pos[2] < 200.0:
            return float(np.linalg.norm(ball_pos[:2] - car_pos[:2]))
        return float(np.linalg.norm(ball_pos - car_pos))

    def reset(self, initial_state: RocketSimArena):
        self._prev_dist = {
            car.id: self._calc_dist(car.pos, initial_state.ball.pos)
            for car in initial_state.cars
        }
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._was_in_strike_zone = {car.id: False for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if is_goal:
            return 0.0

        curr_dist = self._calc_dist(car.pos, arena.ball.pos)
        prev_dist = self._prev_dist.get(car.id, curr_dist)
        self._prev_dist[car.id] = curr_dist

        prev_t = self._prev_touches.get(car.id, car.ball_touches)
        self._prev_touches[car.id] = car.ball_touches

        # Unit alignment vector to ball (properly normalized in 3D)
        car_to_ball = arena.ball.pos - car.pos
        dist_3d = float(np.linalg.norm(car_to_ball))
        unit_to_ball = car_to_ball / max(1e-4, dist_3d)
        fwd_alignment = float(np.dot(car.get_forward_vector(), unit_to_ball))

        # Defensive coordinate context
        defend_goal_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_car_to_defend = abs(car.pos[1] - defend_goal_y)
        dist_ball_to_defend = abs(arena.ball.pos[1] - defend_goal_y)
        car_vy_defend = -car.vel[1] if car.team == 0 else car.vel[1]  # >0 when moving towards own goal
        is_wrong_side = bool(dist_car_to_defend > dist_ball_to_defend + 50.0)

        # Kickoff sprint multiplier & anti-peel penalty (guarantees full-throttle rush on kickoff)
        is_kickoff = bool(abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and arena.ball.pos[2] < 120.0 and float(np.linalg.norm(arena.ball.vel)) < 100.0)
        if is_kickoff:
            delta_dist = (prev_dist - curr_dist) / 2000.0
            if fwd_alignment < -0.20 and float(action[0]) > 0.30:
                # Car is actively peeling away backwards from the kickoff ball
                return self.weight * -1.5
            fwd_speed_to_ball = max(0.0, float(np.dot(car.vel, unit_to_ball)))
            vel_toward_ball = (fwd_speed_to_ball / 2300.0) * 0.30 * max(0.0, fwd_alignment)
            kickoff_mult = 3.0 if delta_dist > 0.0 else 2.5
            return self.weight * (delta_dist * kickoff_mult + vel_toward_ball)

        # ── General Open Play ─────────────────────────────────────────────────
        ball_z = float(arena.ball.pos[2])
        is_elevated_aerial = (ball_z > 350.0)

        is_on_ceiling = bool((car.pos[2] > 1750.0 and car.on_ground) or car.pos[2] > 1900.0)
        is_on_wall = bool((abs(car.pos[0]) > 3450.0 or abs(car.pos[1]) > 4450.0) and car.pos[2] > 200.0 and car.on_ground)

        # 1. Anti-Overshoot Penalty & Strike Zone Tracking
        overshoot_penalty = 0.0
        in_strike = (curr_dist < 400.0)
        was_strike = self._was_in_strike_zone.get(car.id, False)
        self._was_in_strike_zone[car.id] = in_strike

        if was_strike and not in_strike and car.ball_touches == prev_t and fwd_alignment < -0.15:
            # Car was in the strike zone and flew past the ball without touching it!
            # Heavy penalty (-0.40), escalated if overshooting at supersonic speeds
            car_spd = float(np.linalg.norm(car.vel))
            overshoot_penalty = -0.40 if car_spd < 1800.0 else -0.60

        # Ceiling Exploit Prevention:
        # If car is riding the ceiling and the ball is below it, eliminate distance closure rewards
        # and penalize burning boost to sprint across the ceiling
        ceiling_penalty = 0.0
        if is_on_ceiling and ball_z < car.pos[2] - 200.0:
            raw_delta_dist = min(0.0, (prev_dist - curr_dist) / 2000.0)
            if action[6] > 0.0:
                ceiling_penalty = -0.25
        else:
            raw_delta_dist = (prev_dist - curr_dist) / 2000.0

        # 2. Distance Delta with Strike Zone Pacing
        # Downfield (> 450 uu): 100% distance closure rewarded
        # Inside strike zone (< 450 uu): Paces approach so car doesn't blindly barrel past ball
        strike_pacing = min(1.0, max(0.20, (curr_dist - 150.0) / 300.0))
        delta_dist = raw_delta_dist * strike_pacing

        # Wall-Crawling Dampening:
        # 1. When ball is elevated in the air, dampen wall-crawling so leaping off into an aerial is preferred.
        # 2. When ball has bounced away from the wall into the infield (lateral separation),
        #    or car is climbing higher than the ball on the wall (car_z > ball_z + 200), heavily dampen wall driving.
        is_ball_infield = False
        is_car_above_ball = False
        if is_on_wall:
            is_ball_infield = bool(abs(arena.ball.pos[0]) < 2800.0) if abs(car.pos[0]) > 3450.0 else bool(abs(arena.ball.pos[1]) < 3800.0)
            is_car_above_ball = bool(car.pos[2] > ball_z + 200.0)
            if is_elevated_aerial or is_ball_infield or is_car_above_ball:
                delta_dist *= 0.15

        # If car is moving in reverse, executing a half-flip, or executing an active dodge/speedflip towards target,
        # evaluate horizontal travel velocity alignment rather than car nose forward vector:
        car_fwd_vel = float(np.dot(car.vel[:2], car.get_forward_vector()[:2]))
        car_horiz_speed = float(np.linalg.norm(car.vel[:2]))
        travel_unit_h = car.vel[:2] / max(1e-4, car_horiz_speed)
        travel_align_to_ball = float(np.dot(travel_unit_h, unit_to_ball[:2]))

        # Front flips, diagonal speedflips, and dodges temporarily pitch the nose away while rocketing forward:
        is_dodging_toward_ball = bool(car.just_dodged and car_horiz_speed > 250.0 and travel_align_to_ball > 0.2)
        is_reversing_to_target = (car_fwd_vel < -100.0 or delta_dist > 0.0) and float(np.dot(car.vel[:2], unit_to_ball[:2])) > 100.0

        if fwd_alignment < 0.0 and delta_dist > 0.0 and not (is_reversing_to_target or is_dodging_toward_ball):
            delta_dist = delta_dist * max(0.0, fwd_alignment + 1.0) * 0.2

        # 3. Strike-Zone Velocity Matching & Brake Incentives (< 450 uu)
        vel_matching_bonus = 0.0
        brake_incentive = 0.0
        wrong_side_push_penalty = 0.0

        if curr_dist < 450.0 and fwd_alignment > 0.2:
            car_speed = float(np.linalg.norm(car.vel))
            ball_speed = float(np.linalg.norm(arena.ball.vel))
            rel_speed = float(np.linalg.norm(car.vel - arena.ball.vel))
            
            # Gated Velocity Matching:
            # Velocity matching should only reward pacing a moving ball (ball_speed > 250 and car_speed > 200),
            # preventing continuous reward farming when car is merely nose-pushing a grounded ball
            # or sitting stationary near a stopped ball.
            is_ground_pushing = bool(curr_dist < 180.0 and ball_z < 130.0 and car.on_ground)

            if not (is_wrong_side and car_vy_defend > 100.0):
                if not is_ground_pushing and ball_speed > 250.0 and car_speed > 200.0:
                    vel_matching_bonus = 0.30 * max(0.0, 1.0 - (rel_speed / 700.0))
            else:
                # Car is pushing ball toward own net: apply wrong-side push penalty
                wrong_side_push_penalty = -0.30 * max(0.0, car_vy_defend / 1500.0) * max(0.0, fwd_alignment)

            # If closing dangerously fast (> 1000 uu/s) on a slower ball (< 700 uu/s), reward braking to pace arrival
            # When scrambling defensively towards own half, braking before arriving at ball prevents blowing past it
            if car_speed > 900.0 and (ball_speed < 700.0 or is_wrong_side) and action[0] < -0.05:
                brake_incentive = 0.30 * min(1.0, -action[0])

        # 4. Projected Velocity Toward Ball (Airborne Climbing vs Ground Traversal)
        vel_toward_ball = 0.0
        if is_elevated_aerial:
            if not car.on_ground:
                # Airborne flight: Directly reward 3D closing velocity toward high ball!
                air_climb_speed = float(np.dot(car.vel, unit_to_ball))
                if air_climb_speed > 0.0:
                    vel_toward_ball = (air_climb_speed / 2300.0) * 0.40 * max(0.0, fwd_alignment)
                elif air_climb_speed < -100.0 and curr_dist > 300.0:
                    # Penalize actively flying away from elevated aerial ball in mid-air
                    vel_toward_ball = (air_climb_speed / 2300.0) * 0.25
            else:
                vel_toward_ball = 0.0
        else:
            # Grounded or low ball: Gate downfield rush when pushing towards defending goal
            if not (is_wrong_side and car_vy_defend > 100.0):
                speed_taper = min(1.0, max(0.0, (curr_dist - 180.0) / 320.0))
                fwd_speed_to_ball = max(0.0, float(np.dot(car.vel, unit_to_ball)))
                effective_alignment = travel_align_to_ball if is_dodging_toward_ball else fwd_alignment
                vel_toward_ball = (fwd_speed_to_ball / 2300.0) * 0.20 * max(0.0, effective_alignment) * speed_taper
                if is_on_wall and (is_ball_infield or is_car_above_ball or is_elevated_aerial):
                    vel_toward_ball *= 0.15

        # 5. Wrong-Way Ground Rush Penalty (Incentivizes coasting/braking/turning when facing away)
        # Exempt when rotating back on defense toward defending net, OR when actively steering into a turn/cut!
        wrong_way_throttle_penalty = 0.0
        is_retreating_to_defend = bool(car_vy_defend > 100.0)
        is_actively_steering = bool(abs(float(action[1])) >= 0.30)
        if car.on_ground and fwd_alignment < -0.3 and float(action[0]) > 0.4 and not is_retreating_to_defend and not is_actively_steering:
            wrong_way_throttle_penalty = -0.15 * float(action[0]) * abs(fwd_alignment)

        total_reward = self.weight * (
            delta_dist + vel_toward_ball + vel_matching_bonus + brake_incentive +
            overshoot_penalty + ceiling_penalty + wrong_side_push_penalty + wrong_way_throttle_penalty
        )
        return float(total_reward)


# ==============================================================================
# 4. BALL TOUCH & CONTEXTUAL DIRECTIONALITY (Power Shots vs Controlled Catches)
# ==============================================================================
class TouchBallReward(BaseReward):
    """
    Context-Aware Ball Strike & Possession Quality.
    Rewarded at the exact moment of ball contact, adapting intelligently to tactical context:
      1. Tactical Boom / Shot on Net: Booming strikes scaled heavily by goal alignment.
      2. Possession & Control Catch: Soft touches and pops that keep the ball under close control.
      3. Defensive Saves / Clears: Rewarded when clearing the ball out of the defensive sector.
      4. Own-Goal Touch Guard: Strictly penalizes touches that project the ball towards the defending net.
      5. Vertical Aerials: Heavy height scaling (up to 2.5x) and airborne bonuses for aerial challenges.
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
            # Height scaling: Ground touch (Z=93) = 1.0x, High Aerial touch (Z=1500) = 2.5x
            ball_z = float(arena.ball.pos[2])
            height_multiplier = 1.0 + 1.5 * max(0.0, min(1.0, (ball_z - 150.0) / 1850.0))

            # Aerial airborne touch bonus (car airborne contesting high ball)
            airborne_bonus = 1.2 if (not car.on_ground and ball_z > 350.0) else 0.0

            # Kickoff first-touch race bounty
            is_kickoff_touch = bool(abs(arena.ball.pos[0]) < 200.0 and abs(arena.ball.pos[1]) < 200.0 and arena.ball.pos[2] < 150.0 and all(c.ball_touches <= 1 for c in arena.cars))
            kickoff_bounty = 1.0 if is_kickoff_touch else 0.0

            # Explicit Goal Scoring Touch:
            # Reward scoring strikes generously; never penalize the shot that enters the net
            if is_goal:
                if scoring_team == car.team:
                    return (self.weight * 2.5 * height_multiplier) + airborne_bonus + kickoff_bounty
                else:
                    return 0.0

            # Touch occurred on this step
            ball_speed = float(np.linalg.norm(arena.ball.vel))
            # Target goal opening rather than pure +Y direction
            target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
            defend_goal_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y

            target_x = float(np.clip(arena.ball.pos[0], -GOAL_HALF_WIDTH * 0.75, GOAL_HALF_WIDTH * 0.75))
            target_pos = np.array([target_x, target_goal_y, GOAL_HEIGHT * 0.35], dtype=np.float32)

            ball_to_net = target_pos - arena.ball.pos
            unit_to_goal = ball_to_net / max(1e-4, float(np.linalg.norm(ball_to_net)))

            goal_alignment = 0.0
            if ball_speed > 1e-4:
                unit_ball_vel = arena.ball.vel / ball_speed
                goal_alignment = float(np.dot(unit_ball_vel, unit_to_goal))

            # Defensive sector clear / save check:
            # Recognizes forward clears (vy_out > 80.0) or lateral corner pinches/clears (|vx| > 350.0)
            dist_ball_to_defend = abs(arena.ball.pos[1] - defend_goal_y)
            ball_vy_out = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
            ball_vx_mag = abs(arena.ball.vel[0])
            is_defensive_clear = bool(
                dist_ball_to_defend < 2400.0 and (
                    ball_vy_out > 80.0 or (ball_vx_mag > 350.0 and ball_vy_out > -100.0)
                )
            )

            # --- CASE 1: Ball hit directed toward opponent half / goal ---
            if goal_alignment >= 0.0:
                # On-Target Trajectory Bonus: Check if touch velocity produces a direct shot into the net
                vy_forward = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
                if vy_forward > 80.0:
                    delta_y = abs(target_goal_y - arena.ball.pos[1])
                    dt = delta_y / vy_forward
                    x_impact = arena.ball.pos[0] + arena.ball.vel[0] * dt
                    if abs(x_impact) <= GOAL_HALF_WIDTH:
                        # Direct shot on target into the net opening!
                        goal_alignment = max(goal_alignment, 0.7) + 0.35

                direction_multiplier = 1.0 + (min(1.0, goal_alignment) * 1.5)  # 1.0x -> 2.5x

                # Dual-Path Context Evaluator:
                rel_speed = float(np.linalg.norm(car.vel - arena.ball.vel))
                is_gentle_ground_push = bool(car.on_ground and ball_z < 130.0 and rel_speed < 150.0)

                power_bonus = 0.0
                if goal_alignment > 0.4 and not is_gentle_ground_push:
                    power_bonus = min(1.2, max(ball_speed, rel_speed) / 1800.0)

                if is_gentle_ground_push:
                    control_bonus = 0.0
                    base_touch = 0.25
                else:
                    control_bonus = max(0.0, 1.0 - (rel_speed / 600.0)) * 0.8
                    clear_bonus = 0.5 if is_defensive_clear else 0.0
                    base_touch = 0.8 + clear_bonus

                tactical_bonus = max(power_bonus, control_bonus)
                return (self.weight * (base_touch + tactical_bonus) * direction_multiplier * height_multiplier) + airborne_bonus + kickoff_bounty

            # --- CASE 2: Ball hit directed backward toward defending half / goal ---
            else:
                if is_defensive_clear:
                    # Lateral pinch / side clear out of defensive third
                    return self.weight * 0.8 * height_multiplier + airborne_bonus
                else:
                    # Direct touch toward own goal: Strictly penalized to prevent own-goal dribbling
                    penalty_scale = max(0.3, abs(goal_alignment))
                    return -self.weight * 1.5 * penalty_scale * height_multiplier

        return 0.0


# ==============================================================================
# 5. JUMP MOMENTUM BRIDGE (50/50 Blocks, Aerial Takeoffs & Tactical Traversal)
# ==============================================================================
class JumpBridgeReward(BaseReward):
    """
    Context-Aware Jump & Momentum Bridge:
      1. 50/50 Challenge Jump (dist <= 650 uu):
         Rewards single-jump liftoff regardless of car facing angle, allowing front, side,
         and rear center-mass absorption blocks. Grants an Airborne Challenge Completion Bonus
         if the ball is intercepted before landing.
      2. Aerial & Wall Launch (ball.pos[2] > 250 uu or Wall zone):
         Requires forward alignment with the high ball (3.0x multiplier) or wall closing velocity.
      3. Open-Field Traversal (dist > 650 uu, ball grounded):
         Raw ground liftoff is unrewarded (eliminates bunny-hop farming). Instead, rewards
         tactically aligned flips and wavedash speed impulses (delta_v > 0) towards the active
         objective (ball/goal when attacking, defensive third when retreating).
    """
    def __init__(self, weight: float = 0.35):
        super().__init__(weight)
        self._prev_on_ground: Dict[int, bool] = {}
        self._prev_has_flip: Dict[int, bool] = {}
        self._prev_touches: Dict[int, int] = {}
        self._prev_vel: Dict[int, np.ndarray] = {}
        self._prev_pos_z: Dict[int, float] = {}
        self._challenge_jump_active: Dict[int, bool] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_on_ground = {car.id: car.on_ground for car in initial_state.cars}
        self._prev_has_flip = {car.id: car.has_flip for car in initial_state.cars}
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._prev_vel = {car.id: car.vel.copy() for car in initial_state.cars}
        self._prev_pos_z = {car.id: float(car.pos[2]) for car in initial_state.cars}
        self._challenge_jump_active = {car.id: False for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev_ground = self._prev_on_ground.get(car.id, car.on_ground)
        self._prev_on_ground[car.id] = car.on_ground

        prev_flip = self._prev_has_flip.get(car.id, car.has_flip)
        self._prev_has_flip[car.id] = car.has_flip

        prev_touch = self._prev_touches.get(car.id, car.ball_touches)
        self._prev_touches[car.id] = car.ball_touches

        prev_vel = self._prev_vel.get(car.id, car.vel)
        self._prev_vel[car.id] = car.vel.copy()

        prev_pos_z = self._prev_pos_z.get(car.id, float(car.pos[2]))
        self._prev_pos_z[car.id] = float(car.pos[2])

        car_to_ball = arena.ball.pos - car.pos
        dist = float(np.linalg.norm(car_to_ball))
        unit_to_ball = car_to_ball / max(1e-4, dist)
        forward_alignment = float(np.dot(car.get_forward_vector(), unit_to_ball))
        takeoff_closing_vel = float(np.dot(car.vel, unit_to_ball))
        ball_z = float(arena.ball.pos[2])

        # Defensive & tactical context
        defend_goal_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_car_to_defend = abs(car.pos[1] - defend_goal_y)
        dist_ball_to_defend = abs(arena.ball.pos[1] - defend_goal_y)
        # Car is only on the 'wrong side' requiring defensive retreat if the ball is in or entering the defensive half.
        # When ball is in the attacking half, offensive rebounds and centering redirects remain the target.
        is_ball_in_defensive_half = bool(dist_ball_to_defend < ARENA_EXTENT_Y)
        is_wrong_side = bool(is_ball_in_defensive_half and (dist_car_to_defend > dist_ball_to_defend + 100.0))

        # Tactical vector (toward shadow intercept position when retreating, toward ball when attacking/contesting)
        # Instead of retreating blindly to the goal-line (which leads to overshooting and panicking),
        # retreat to a shadow-defense position between the ball and net (ball.pos[1] +/- 700 uu).
        if is_wrong_side:
            shadow_offset = -700.0 if car.team == 0 else 700.0
            shadow_target_y = float(arena.ball.pos[1] + shadow_offset)
            # Bound within defending goal line and pitch bounds:
            if car.team == 0:
                shadow_target_y = max(-ARENA_EXTENT_Y + 200.0, min(0.0, shadow_target_y))
            else:
                shadow_target_y = min(ARENA_EXTENT_Y - 200.0, max(0.0, shadow_target_y))

            retreat_vec = np.array([float(arena.ball.pos[0]) * 0.5 - car.pos[0], shadow_target_y - car.pos[1], 0.0], dtype=np.float32)
            tactical_dir = retreat_vec / max(1e-4, float(np.linalg.norm(retreat_vec)))
        else:
            tactical_dir = unit_to_ball

        reward = 0.0

        # ── 1. Takeoff Transition (Ground -> Air) ─────────────────────────────
        if prev_ground and not car.on_ground and car.vel[2] > 80.0:
            is_on_wall_zone = bool(abs(car.pos[0]) > 3400.0 or abs(car.pos[1]) > 4400.0)
            is_aerial_ball = bool(ball_z > 250.0)
            car_boost = float(car.boost)

            # 1a. Close-Quarters Strike Liftoff & 50/50 Challenge (dist <= 500 uu, closing toward ball)
            opponents = [c for c in arena.cars if c.team != car.team and not c.demoed]
            opp_dist_to_ball = min([float(np.linalg.norm(c.pos - arena.ball.pos)) for c in opponents], default=9999.0)
            is_contested_5050 = bool(dist <= 450.0 and opp_dist_to_ball <= 650.0 and ball_z < 220.0 and car.pos[2] < 150.0)
            is_strike_liftoff = bool(dist <= 500.0 and ball_z < 250.0 and car.pos[2] < 150.0 and forward_alignment > 0.20 and takeoff_closing_vel > 150.0)

            if is_contested_5050 or is_strike_liftoff:
                self._challenge_jump_active[car.id] = True
                bonus_scale = 1.2 if is_contested_5050 else (0.8 + 0.4 * forward_alignment)
                reward += self.weight * bonus_scale

            # 1b. Wall Takeoff / Air Dribble Pop / Wall Bang Setup
            elif is_on_wall_zone and car.pos[2] > 200.0 and (takeoff_closing_vel > 150.0 or forward_alignment > 0.15):
                # Close-proximity wall strike / dodge setup (dist <= 450 uu): rewarded for all boost levels
                if dist <= 450.0:
                    reward += self.weight * max(0.2, forward_alignment) * 2.5
                elif car_boost >= 30.0:
                    # Air dribble carry setup into open pitch: requires >= 30 boost
                    reward += self.weight * max(0.2, forward_alignment) * 3.5

            # 1c. Aerial Floor Launch (Ball elevated in air)
            elif is_aerial_ball and car.pos[2] < 300.0 and forward_alignment > 0.15:
                # Moderate ball (Z <= 450 uu): double-jump or pop reachable with minimal boost
                if ball_z <= 450.0:
                    reward += self.weight * forward_alignment * 2.0
                # High aerial ball (Z > 450 uu): requires >= 30 boost to fly
                elif car_boost >= 30.0:
                    reward += self.weight * forward_alignment * 3.0
                elif car_boost < 20.0:
                    # Hopeless floor takeoff under high ball with low/no boost
                    reward += -0.15

            # 1d. Open-field ground traversal (dist > 650 uu, ball grounded)
            # Liftoff alone gets 0.0 (prevents open-field bunny hopping)

        # ── 2. Airborne 50/50 Challenge Completion Bonus ──────────────────────
        if not car.on_ground and self._challenge_jump_active.get(car.id, False):
            if car.ball_touches > prev_touch:
                # Intercepted/blocked ball during 50/50 jump window!
                reward += self.weight * 1.5
                self._challenge_jump_active[car.id] = False
        elif car.on_ground:
            self._challenge_jump_active[car.id] = False

        # ── 3. Airborne Dodge / Flip & Traversal Impulse ──────────────────────
        pitch_input = float(action[2])
        yaw_input = float(action[3])
        stick_deflection = max(abs(pitch_input), abs(yaw_input))

        # Joystick-Only Dodge Impulse Reconstruction:
        # In Rocket League, flip/dodge direction is governed solely by joystick pitch (action[2]) and yaw (action[3]).
        # Throttle/brake and air roll have NO influence on dodge impulse direction.
        fwd_vec = car.get_forward_vector()
        right_vec = car.get_right_vector()
        dodge_dir_local = np.array([
            1.0 if pitch_input > 0.3 else (-1.0 if pitch_input < -0.3 else 0.0),
            1.0 if yaw_input > 0.3 else (-1.0 if yaw_input < -0.3 else 0.0),
            0.0
        ], dtype=np.float32)
        dodge_norm = float(np.linalg.norm(dodge_dir_local))
        if dodge_norm > 1e-4:
            dodge_impulse_world = (dodge_dir_local[0] * fwd_vec + dodge_dir_local[1] * right_vec) / dodge_norm
            dodge_align = float(np.dot(dodge_impulse_world[:2], tactical_dir[:2]))
        else:
            dodge_align = 0.0

        car_speed_horiz = float(np.linalg.norm(car.vel[:2]))
        car_fwd_speed = float(np.dot(car.vel[:2], fwd_vec[:2]))
        if not car.on_ground and prev_flip and not car.has_flip:
            # Traversal flips downfield require forward or backward movement (speed > 350 uu/s)
            # to prevent spamming flips in place from a stop or slow turn.
            # Close strike-zone dodges (dist <= 650 uu) allow stationary challenge 50/50 jumps.
            is_open_field = bool(dist > 650.0)
            has_traversal_speed = bool(car_speed_horiz > 350.0)
            # Backflip traversal restriction: do NOT reward backflips for travel when already driving forward (car_fwd_speed > 250)
            # (prevents panic backflipping when rushing back or overshooting). Backflips for traversal are only for half-flips from reverse.
            is_forward_backflip = bool(is_open_field and pitch_input < -0.3 and car_fwd_speed > 250.0)
            if stick_deflection >= 0.30 and dodge_align > 0.25 and not is_forward_backflip:
                if (not is_open_field) or has_traversal_speed:
                    reward += self.weight * dodge_align * (0.4 + 0.3 * stick_deflection)
            elif ball_z > 350.0 and forward_alignment > 0.30:
                # Double jump for high aerial balls
                reward += self.weight * forward_alignment * 0.4

        # ── 4. Wavedash & Speed Impulse on Touchdown / Flip Acceleration ─────
        # Rewards speed increases (delta_v > 0) along tactical vector resulting from flips/wavedashes
        tactical_speed_curr = float(np.dot(car.vel[:2], tactical_dir[:2]))
        tactical_speed_prev = float(np.dot(prev_vel[:2], tactical_dir[:2]))
        delta_tactical_speed = tactical_speed_curr - tactical_speed_prev

        # Explicit Wavedash Detection:
        # A wavedash occurs when dodging while very low to the turf (prev_pos_z < 55 uu)
        # and immediately contacting turf, slamming the flip impulse into ground acceleration (> 120 uu/s)
        is_wavedash = bool(car.on_ground and prev_pos_z < 55.0 and car.just_dodged and delta_tactical_speed > 120.0)
        if is_wavedash:
            reward += self.weight * 1.5 * min(1.0, delta_tactical_speed / 400.0)
        else:
            is_landing_or_dodge = bool((not prev_ground and car.on_ground) or car.just_dodged)
            if is_landing_or_dodge and delta_tactical_speed > 60.0 and tactical_speed_curr > 500.0:
                speed_factor = min(1.0, tactical_speed_curr / 2200.0)
                impulse_factor = min(1.0, delta_tactical_speed / 400.0)
                reward += self.weight * 0.8 * impulse_factor * speed_factor

        return float(reward)


# ==============================================================================
# 8. BOOST RETENTION & ECONOMY (Necto Sqrt-Potential Engine)
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

        # Suspend all boost collection rewards and usage penalties during active kickoff (until ball is first touched/moving)
        is_kickoff = bool(abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and arena.ball.pos[2] < 120.0 and float(np.linalg.norm(arena.ball.vel)) < 100.0)
        if is_kickoff:
            return 0.0

        boost_diff = math.sqrt(curr) - math.sqrt(prev)

        if boost_diff > 1e-5:
            # ── 1. Boost Pad Collection Event Bonus & Low-Boost Hunger ─────────
            # Heavy low-boost hunger: picking up pads when near zero boost is critical for mobility & defense
            hunger_mult = 1.0 + 2.0 * max(0.0, 1.0 - prev)
            base_gain = self.gain_weight * boost_diff * hunger_mult

            # Discrete pad collection event bonus:
            # Small pad (+12 boost): +0.45 * hunger
            # Big orb (+100 boost): +1.20 * hunger
            is_big_pad = bool((curr - prev) > 0.50)
            pickup_bonus = (1.20 if is_big_pad else 0.45) * max(0.5, 1.0 - prev)
            return float(base_gain + pickup_bonus)

        elif boost_diff < -1e-5:
            # ── 2. Boost Usage, Waste & Negative Momentum Penalties ─────────────
            height_factor = max(0.2, 1.0 - (car.pos[2] / GOAL_HEIGHT))
            loss_rew = self.lose_weight * boost_diff * height_factor

            # Supersonic boost waste penalty: burning boost when already at max speed (>= 2150 uu/s)
            speed = float(np.linalg.norm(car.vel))
            if speed >= 2150.0 and action[6] > 0.0:
                loss_rew -= 0.35 if car.on_ground else 0.20

            # Ceiling and vertical climb boost waste penalty: burning boost along ceiling or climbing vertically away from a lower ball
            is_climbing_above_ball = bool(car.vel[2] > 100.0 and car.pos[2] > arena.ball.pos[2] + 200.0)
            if (car.pos[2] > 1750.0 or is_climbing_above_ball) and action[6] > 0.0 and arena.ball.pos[2] < car.pos[2] - 200.0:
                loss_rew -= 0.25

            fwd_vec = car.get_forward_vector()
            fwd_speed = float(np.dot(car.vel, fwd_vec))

            # Reverse-Momentum Boost Waste Penalty:
            # Burning boost when the car's 3D momentum opposes its forward nose direction (fwd_speed < -150 uu/s).
            # Attempting to use boost as an emergency airbrake against reverse momentum wastes massive boost
            # while floating helplessly; the player should coast to ground contact and powerslide/brake instead.
            if action[6] > 0.0 and fwd_speed < -150.0:
                rev_waste_scale = min(1.0, abs(fwd_speed) / 1200.0)
                loss_rew -= 0.35 * rev_waste_scale if not car.on_ground else 0.20 * rev_waste_scale

            # Off-axis boost waste penalty: burning boost when facing away from ball on ground (causes wide orbiting)
            # Only exempt when genuinely boosting in forward retreat direction toward defending net
            defend_goal_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
            car_vy_defend = -car.vel[1] if car.team == 0 else car.vel[1]
            is_retreating_to_defend = bool(car_vy_defend > 100.0 and fwd_speed > 100.0)

            car_to_ball = arena.ball.pos - car.pos
            dist_to_ball = float(np.linalg.norm(car_to_ball))

            if not is_retreating_to_defend:
                if car.on_ground and action[6] > 0.0:
                    if dist_to_ball > 300.0:
                        unit_to_ball = car_to_ball / dist_to_ball
                        fwd_align = float(np.dot(fwd_vec, unit_to_ball))
                        if fwd_align < 0.10:
                            loss_rew -= 0.15 * (1.0 - fwd_align)

                # Airborne off-trajectory boost waste penalty:
                # Burning boost while airborne when car's 3D momentum is moving away from or past the ball
                # (Exempt during active flip-cancels and half-flip recoveries)
                elif not car.on_ground and action[6] > 0.0:
                    is_recovering_halfflip = bool(car.just_dodged or (float(action[2]) > 0.4 and abs(float(action[4])) > 0.2))
                    if dist_to_ball > 250.0 and not is_recovering_halfflip:
                        unit_to_ball = car_to_ball / dist_to_ball
                        closing_vel = float(np.dot(car.vel, unit_to_ball))
                        fwd_align = float(np.dot(fwd_vec, unit_to_ball))
                        if closing_vel < -100.0 or (closing_vel < 100.0 and fwd_align < 0.20):
                            loss_rew -= 0.30 * min(1.0, max(0.2, -closing_vel / 1000.0 if closing_vel < 0 else 0.5))

            return loss_rew
        else:
            # ── 3. Continuous Transit Pad Approach & Alignment Shaping ──────────
            # When low on boost (< 65) and driving on the ground, reward steering toward and routing
            # through active boost pads along the travel path, eliminating straight-line pad skipping.
            if car.on_ground and car.boost < 65.0 and hasattr(arena, "_small_pad_pos_3d") and hasattr(arena, "_small_pad_active"):
                sm_act = arena._small_pad_active
                sm_poses = arena._small_pad_pos_3d
                cpx, cpy = float(car.pos[0]), float(car.pos[1])
                min_sm_d2 = 550.0 * 550.0
                min_sm_idx = -1
                for p_idx in range(len(sm_act)):
                    if sm_act[p_idx]:
                        dx = float(sm_poses[p_idx, 0]) - cpx
                        dy = float(sm_poses[p_idx, 1]) - cpy
                        d2 = dx * dx + dy * dy
                        if d2 < min_sm_d2:
                            min_sm_d2 = d2
                            min_sm_idx = p_idx

                if min_sm_idx >= 0:
                    pad_dist = math.sqrt(min_sm_d2)
                    dx = float(sm_poses[min_sm_idx, 0]) - cpx
                    dy = float(sm_poses[min_sm_idx, 1]) - cpy
                    pad_dir_x = dx / max(1e-4, pad_dist)
                    pad_dir_y = dy / max(1e-4, pad_dist)
                    fwd = car.get_forward_vector()
                    pad_align = fwd[0] * pad_dir_x + fwd[1] * pad_dir_y
                    speed_to_pad = car.vel[0] * pad_dir_x + car.vel[1] * pad_dir_y

                    if pad_align > 0.25 and speed_to_pad > 150.0:
                        boost_hunger = (65.0 - car.boost) / 65.0
                        prox = 1.0 - (pad_dist / 550.0)
                        speed_fac = min(1.0, speed_to_pad / 1000.0)
                        return float(0.20 * boost_hunger * pad_align * prox * speed_fac)

            return 0.0


# ==============================================================================
# 9. POWERSLIDE & DRIFT TURN-AROUND REWARD (Tight Hairpin Cuts & Snap Pivoting)
# ==============================================================================
class PowerslideReward(BaseReward):
    """
    Rewards active handbrake / powerslide usage during sharp ground turnarounds.
    Incentivizes pressing handbrake (action[7] > 0.0) when the ball is off-axis (fwd_alignment < 0.7)
    proportional to turning rate toward the ball, enabling tight U-turns and cuts without orbiting.
    """
    def __init__(self, weight: float = 0.30):
        super().__init__(weight)
        self._prev_alignment: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_alignment = {}
        for car in initial_state.cars:
            d = initial_state.ball.pos - car.pos
            dist = float(np.linalg.norm(d))
            if dist > 1e-4:
                self._prev_alignment[car.id] = float(np.dot(car.get_forward_vector(), d / dist))
            else:
                self._prev_alignment[car.id] = 1.0

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        car_to_ball = arena.ball.pos - car.pos
        dist = float(np.linalg.norm(car_to_ball))
        if dist < 1e-4:
            return 0.0

        unit_to_ball = car_to_ball / dist
        fwd_alignment = float(np.dot(car.get_forward_vector(), unit_to_ball))
        prev_align = self._prev_alignment.get(car.id, fwd_alignment)
        self._prev_alignment[car.id] = fwd_alignment

        # Active on ground during sharp off-axis cuts (fwd_alignment < 0.60, steer > 0.25, handbrake active)
        # Enables low-speed cut turns and U-turns (speed > 50.0 uu/s) when pivoting rapidly toward target
        speed = float(np.linalg.norm(car.vel))
        steer_mag = abs(float(action[1]))
        yaw_rate = abs(float(car.ang_vel[2])) if hasattr(car, "ang_vel") else 0.0
        handbrake_active = float(action[7]) > 0.0

        if car.on_ground and fwd_alignment < 0.60 and steer_mag > 0.25 and speed > 50.0 and handbrake_active:
            alignment_rate = max(0.0, fwd_alignment - prev_align)
            handbrake_intensity = max(0.0, float(action[7]))
            # Rapid pivoting (high yaw velocity) or positive heading alignment progression:
            if yaw_rate > 1.2 or alignment_rate > 0.02:
                pivot_efficiency = min(1.0, max(alignment_rate * 5.0, yaw_rate / 3.5))
                turn_bonus = pivot_efficiency * (0.6 + 0.4 * steer_mag) * handbrake_intensity
                return self.weight * turn_bonus

        return 0.0


# ==============================================================================
# 10. AIR-ROLL ORIENTATION & LANDING RECOVERY REWARD
# ==============================================================================
class AirRollRecoveryReward(BaseReward):
    """
    Incentivizes 3D air-roll recoveries, clean landing orientations, and aerial attitude control:
      1. Active Roll & Inversion Recovery (delta_up > 0): Rewards active rotation toward wheels-down,
         scaled up to 2.5x when rotating from an inverted (wheels up) state.
      2. Active Yaw & Momentum Heading Recovery (delta_heading > 0): Rewards active rotation aligning
         the nose with horizontal travel velocity, scaled up to 2.5x when rotating from flying backwards.
      3. Disorientation-Gated Touchdown: Rewards landing on 4 wheels with forward momentum retention after
         any aerial disorientation or 50/50 collision bounce, without rewarding ground bunny hops.
      4. Wall Landing Recovery: When airborne near sidewall/backwall, rewards aligning car up-vector with wall normal.
      5. Aerial Challenge Attitude Control: When closing toward elevated balls, rewards matching roll alignment.
    """
    def __init__(self, weight: float = 0.10):
        super().__init__(weight)
        self._prev_up_z: Dict[int, float] = {}
        self._prev_heading: Dict[int, float] = {}
        self._airborne_ticks: Dict[int, int] = {}
        self._was_disoriented: Dict[int, bool] = {}
        self._airborne_recovery_total: Dict[int, float] = {}
        self._wall_landed: Dict[int, bool] = {}
        self._halfflip_cancel_executed: Dict[int, bool] = {}
        self._halfflip_cancel_total: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_up_z = {car.id: float(car.get_up_vector()[2]) for car in initial_state.cars}
        self._prev_heading = {car.id: 1.0 for car in initial_state.cars}
        self._airborne_ticks = {car.id: 0 for car in initial_state.cars}
        self._was_disoriented = {car.id: False for car in initial_state.cars}
        self._airborne_recovery_total = {car.id: 0.0 for car in initial_state.cars}
        self._wall_landed = {car.id: False for car in initial_state.cars}
        self._halfflip_cancel_executed = {car.id: False for car in initial_state.cars}
        self._halfflip_cancel_total = {car.id: 0.0 for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if car.on_ground:
            self._airborne_ticks[car.id] = 0
            self._prev_up_z[car.id] = 1.0
            self._prev_heading[car.id] = 1.0
            self._was_disoriented[car.id] = False
            self._airborne_recovery_total[car.id] = 0.0
            self._wall_landed[car.id] = False
            self._halfflip_cancel_executed[car.id] = False
            self._halfflip_cancel_total[car.id] = 0.0
            return 0.0

        air_ticks = self._airborne_ticks.get(car.id, 0) + 1
        self._airborne_ticks[car.id] = air_ticks

        up = car.get_up_vector()
        up_z = float(up[2])
        prev_up_z = self._prev_up_z.get(car.id, up_z)
        self._prev_up_z[car.id] = up_z

        car_z = float(car.pos[2])
        vel_z = float(car.vel[2])

        # Horizontal flight velocity & forward heading alignment
        v_horiz = car.vel[:2]
        speed_horiz = float(np.linalg.norm(v_horiz))
        fwd_h = car.get_forward_vector()[:2]
        fwd_norm = float(np.linalg.norm(fwd_h))

        if speed_horiz > 250.0 and fwd_norm > 1e-4:
            unit_vel_h = v_horiz / speed_horiz
            unit_fwd_h = fwd_h / fwd_norm
            curr_heading = float(np.dot(unit_fwd_h, unit_vel_h))
        else:
            curr_heading = 1.0

        prev_heading = self._prev_heading.get(car.id, curr_heading)
        self._prev_heading[car.id] = curr_heading

        # Track if car was genuinely knocked off-axis / inverted during this airborne sequence
        if up_z < 0.3 or curr_heading < -0.2:
            self._was_disoriented[car.id] = True

        car_to_ball = arena.ball.pos - car.pos
        dist_to_ball = float(np.linalg.norm(car_to_ball))
        ball_z = float(arena.ball.pos[2])

        # Aerial engagement check (protects inverted flight during air dribbles / flip resets)
        is_aerial_engagement = bool(ball_z > 350.0 and dist_to_ball < 450.0)

        total_reward = 0.0

        # ── 1. Active 3D Disorientation Recovery (Roll & Yaw) ────────────────
        # Only active when the car was genuinely knocked off-axis, inverted, or executed a flip turnaround
        is_recovering = bool(self._was_disoriented.get(car.id, False))
        pitch_input = float(action[2])
        roll_input = float(action[4])
        is_active_halfflip_cancel = bool(air_ticks <= 18 and (pitch_input > 0.3 or abs(roll_input) > 0.25))

        if not is_aerial_engagement and is_recovering:
            urgency = min(1.0, max(0.4, (800.0 - car_z) / 600.0))
            rec_spent = self._airborne_recovery_total.get(car.id, 0.0)
            rec_budget = max(0.0, 0.80 - rec_spent)

            # 1a. Active Roll & Inversion Recovery (delta_up > 0)
            delta_up = up_z - prev_up_z
            if delta_up > 0.0 and prev_up_z < 0.90:
                # Inversion multiplier: rotating from wheels-up (prev_up_z < 0) yields up to 2.0x reward
                inversion_mult = 1.0 + max(0.0, -prev_up_z) * 1.0
                roll_rec = min(rec_budget, (delta_up * 1.5) * inversion_mult * urgency)
                total_reward += roll_rec
                rec_budget = max(0.0, rec_budget - roll_rec)
                self._airborne_recovery_total[car.id] = rec_spent + roll_rec

            # 1b. Active Yaw & Momentum Heading Recovery (delta_heading > 0)
            delta_heading = curr_heading - prev_heading
            if delta_heading > 0.0 and prev_heading < 0.90 and speed_horiz > 250.0:
                heading_inversion_mult = 1.0 + max(0.0, -prev_heading) * 1.0
                yaw_rec = min(rec_budget, (delta_heading * 1.0) * heading_inversion_mult * urgency)
                total_reward += yaw_rec
                self._airborne_recovery_total[car.id] = self._airborne_recovery_total.get(car.id, 0.0) + yaw_rec

            # 1c. Dedicated Half-Flip Flip-Cancel & Air-Roll Bonus Budget
            # Has its own independent budget (0.60) so passive delta_up cannot starve the active cancel!
            cancel_spent = self._halfflip_cancel_total.get(car.id, 0.0)
            cancel_budget = max(0.0, 0.60 - cancel_spent)
            if air_ticks <= 18 and up_z < 0.40 and speed_horiz > 250.0:
                step_cancel_reward = 0.0
                if pitch_input > 0.3:
                    c_rew = min(cancel_budget, (pitch_input * 0.30) * urgency)
                    step_cancel_reward += c_rew
                    cancel_budget = max(0.0, cancel_budget - c_rew)
                    self._halfflip_cancel_executed[car.id] = True
                if abs(roll_input) > 0.25:
                    r_rew = min(cancel_budget, (abs(roll_input) * 0.30) * urgency)
                    step_cancel_reward += r_rew
                    cancel_budget = max(0.0, cancel_budget - r_rew)
                total_reward += step_cancel_reward
                self._halfflip_cancel_total[car.id] = cancel_spent + step_cancel_reward

            # Reset disorientation once upright attitude is fully restored
            if up_z > 0.85 and curr_heading > 0.85:
                self._was_disoriented[car.id] = False

            # ── 2. Touchdown Alignment (Evaluated as a single impulse near ground contact) ──
            if car_z < 60.0 and vel_z < -50.0:
                if up_z > 0.70:
                    total_reward += (up_z * 0.5)
                    if speed_horiz > 300.0 and curr_heading > 0.30:
                        total_reward += (curr_heading * 0.5)
                        # 180° Turnaround Half-Flip Completion Bonus:
                        # Massive +1.50 completion bonus when bot executes a flip-cancel turnaround
                        # and lands cleanly facing forward in the direction of flight!
                        if self._halfflip_cancel_executed.get(car.id, False) and curr_heading > 0.60 and speed_horiz > 400.0:
                            total_reward += 1.50
                # Consume disorientation so touchdown reward only fires once per landing
                self._was_disoriented[car.id] = False
                self._halfflip_cancel_executed[car.id] = False
                self._halfflip_cancel_total[car.id] = 0.0

                # Upside down landing crash penalty (forgiven during active flip-cancels & quick half-flip air-rolls)
                if up_z < 0.0 and not is_active_halfflip_cancel:
                    total_reward += (up_z * 0.5) * urgency

        # ── 3. Wall Landing Recovery (Airborne near side or back wall) ────────
        # Single-shot landing impulse when approaching wall with wheels oriented toward wall
        if not self._wall_landed.get(car.id, False) and car_z > 200.0:
            dist_x_wall = ARENA_EXTENT_X - abs(car.pos[0])
            dist_y_wall = ARENA_EXTENT_Y - abs(car.pos[1])
            if dist_x_wall < 180.0:
                wall_norm_x = -math.copysign(1.0, car.pos[0])
                closing_to_wall = float(car.vel[0] * -wall_norm_x)
                wall_align = float(up[0] * wall_norm_x)
                if closing_to_wall > 100.0 and wall_align > 0.5:
                    total_reward += max(0.0, wall_align) * 0.4
                    self._wall_landed[car.id] = True
            elif dist_y_wall < 180.0:
                wall_norm_y = -math.copysign(1.0, car.pos[1])
                closing_to_wall = float(car.vel[1] * -wall_norm_y)
                wall_align = float(up[1] * wall_norm_y)
                if closing_to_wall > 100.0 and wall_align > 0.5:
                    total_reward += max(0.0, wall_align) * 0.4
                    self._wall_landed[car.id] = True

        # ── 4. Aerial Challenge Attitude Control (Ball elevated > 350 uu) ─────
        # Only reward attitude alignment if car is actively closing toward elevated ball in flight
        if ball_z > 350.0 and car_z > 200.0 and not car.on_ground:
            if dist_to_ball > 1e-4:
                unit_to_ball = car_to_ball / dist_to_ball
                fwd_align = float(np.dot(car.get_forward_vector(), unit_to_ball))
                closing_vel = float(np.dot(car.vel, unit_to_ball))
                if fwd_align > 0.4 and closing_vel > 200.0:
                    upright_bonus = max(0.0, up_z) * 0.2
                    total_reward += (fwd_align * 0.3 + upright_bonus) * 0.5

        return self.weight * total_reward


# ==============================================================================
# COMBINED MACRO REWARD ENGINE & MANAGER
# ==============================================================================
class CombinedReward:
    """
    Unified Macro Potential-Based Reward Manager.
    Aggregates Macro Goal, Ball-to-Goal, Player-to-Ball, Speed/Flip, Face-Ball, Jump-Bridge, Touch Quality, Boost, Powerslide, and Air Roll Recovery.
    """
    def __init__(self, weights: Dict[str, float]):
        self.rewards: Dict[str, BaseReward] = {
            "goal": GoalReward(
                goal_weight=weights.get("goal_weight", 30.0),
                concede_weight=weights.get("concede_weight", -30.0),
                save_weight=weights.get("save_weight", 12.0)
            ),
            "ball_to_goal": BallToGoalVelocityReward(
                weight=weights.get("ball_to_goal_weight", 1.5)
            ),
            "player_to_ball": PlayerToBallVelocityReward(
                weight=weights.get("player_to_ball_weight", 0.6)
            ),
            "jump_bridge": JumpBridgeReward(
                weight=weights.get("jump_bridge_weight", 0.35)
            ),
            "touch": TouchBallReward(
                weight=weights.get("touch_weight", 1.2)
            ),
            "boost": BoostReward(
                gain_weight=weights.get("boost_gain_weight", 0.6),
                lose_weight=weights.get("boost_lose_weight", 0.3)
            ),
            "powerslide": PowerslideReward(
                weight=weights.get("powerslide_weight", 0.20)
            ),
            "air_roll_recovery": AirRollRecoveryReward(
                weight=weights.get("air_roll_recovery_weight", 0.10)
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

        if "powerslide_weight" in new_weights and "powerslide" in self.rewards:
            self.rewards["powerslide"].weight = float(new_weights["powerslide_weight"])

        if "jump_bridge_weight" in new_weights and "jump_bridge" in self.rewards:
            self.rewards["jump_bridge"].weight = float(new_weights["jump_bridge_weight"])

        if "air_roll_recovery_weight" in new_weights and "air_roll_recovery" in self.rewards:
            self.rewards["air_roll_recovery"].weight = float(new_weights["air_roll_recovery_weight"])

        if "touch_weight" in new_weights and "touch" in self.rewards:
            self.rewards["touch"].weight = float(new_weights["touch_weight"])

        if "boost_gain_weight" in new_weights and "boost" in self.rewards:
            self.rewards["boost"].gain_weight = float(new_weights["boost_gain_weight"])
        if "boost_lose_weight" in new_weights and "boost" in self.rewards:
            self.rewards["boost"].lose_weight = float(new_weights["boost_lose_weight"])

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int], include_breakdown: bool = True) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        breakdown = {} if include_breakdown else None
        for name, r in self.rewards.items():
            rew = float(r.get_reward(car, arena, action, is_goal, scoring_team))
            total += rew
            if include_breakdown:
                breakdown[name] = rew

        # Handbrake Economy Regularization:
        # Penalize dragging handbrake while driving forward on straightaways or gentle curves
        if car.on_ground and float(action[7]) > 0.10:
            fwd = car.get_forward_vector()
            fwd_speed = float(car.vel[0] * fwd[0] + car.vel[1] * fwd[1] + car.vel[2] * fwd[2])
            steer_mag = abs(float(action[1]))
            car_to_ball = arena.ball.pos - car.pos
            dist = float(np.linalg.norm(car_to_ball))
            fwd_align = float(np.dot(fwd, car_to_ball / max(1e-4, dist)))
            if fwd_speed > 300.0 and (steer_mag < 0.40 or fwd_align > 0.50):
                pen = -0.15 * float(action[7]) * min(1.0, fwd_speed / 1500.0)
                total += pen
                if include_breakdown:
                    breakdown["handbrake_penalty"] = pen

        return float(total), breakdown if breakdown is not None else {}


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

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int], include_breakdown: bool = True) -> Tuple[float, Dict[str, float]]:
        return self.combined.get_reward(car, arena, action, is_goal, scoring_team, include_breakdown=include_breakdown)


