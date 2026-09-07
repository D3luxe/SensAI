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
    CAR_MAX_SPEED, BALL_MAX_SPEED, GOAL_HALF_WIDTH, GOAL_HEIGHT, ARENA_EXTENT_X, ARENA_EXTENT_Y, ARENA_HEIGHT_Z,
    WALL_BOUNCE_VX_THRESHOLD, WALL_BOUNCE_VY_THRESHOLD
)


class BaseReward:
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def reset(self, initial_state: RocketSimArena):
        pass

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        raise NotImplementedError


class OpponentThreat:
    __slots__ = ("opp", "dist", "closing_speed", "arrival_time")

    def __init__(self, opp: CarState, dist: float, closing_speed: float, arrival_time: float):
        self.opp = opp
        self.dist = dist
        self.closing_speed = closing_speed
        self.arrival_time = arrival_time


def compute_trajectory_arrival_time(
    pos: np.ndarray,
    vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """
    Canonical kinematics helper.
    Computes (arrival_time, dist, closing_speed) along line-of-sight.
    """
    rel_pos = target_pos - pos
    dist = float(np.linalg.norm(rel_pos))
    if dist < 1e-4:
        return 0.0, 0.0, 0.0
    unit_dir = rel_pos / dist
    if target_vel is not None:
        rel_vel = vel - target_vel
        closing_speed = float(np.dot(rel_vel, unit_dir))
    else:
        closing_speed = float(np.dot(vel, unit_dir))

    if closing_speed > 50.0:
        arrival = dist / closing_speed
    else:
        # Fallback when closing speed <= 50 uu/s:
        # Assumes vehicle can accelerate toward target from baseline speed
        speed = float(np.linalg.norm(vel))
        arrival = dist / max(50.0, speed * 0.35 + 100.0)
    return arrival, dist, closing_speed


def compute_car_arrival_time(
    car: CarState,
    target_pos: np.ndarray,
    target_vel: Optional[np.ndarray] = None
) -> float:
    """Computes arrival time in seconds for car relative to target_pos."""
    arrival, _, _ = compute_trajectory_arrival_time(car.pos, car.vel, target_pos, target_vel)
    return arrival


def compute_opponent_threats(
    car: CarState,
    arena: RocketSimArena,
    target_pos: Optional[np.ndarray] = None,
    opponents: Optional[List[CarState]] = None
) -> List[OpponentThreat]:
    """
    Computes threat arrival time for active (non-demoed) opponents relative to
    target_pos (defaults to arena.ball.pos).
    If opponents list is provided, it is used directly (ideal for unit testing);
    otherwise, filters [c for c in arena.cars if c.team != car.team and not c.demoed].
    Returns list of OpponentThreat sorted by arrival_time ascending (most urgent first).
    """
    if opponents is None:
        opponents = [c for c in arena.cars if c.team != car.team and not c.demoed]

    ref_pos = arena.ball.pos if target_pos is None else target_pos

    threats: List[OpponentThreat] = []
    for opp in opponents:
        if opp.demoed or opp.id == car.id or opp.team == car.team:
            continue
        arrival, dist, closing_speed = compute_trajectory_arrival_time(opp.pos, opp.vel, ref_pos)
        threats.append(OpponentThreat(opp, dist, closing_speed, arrival))

    threats.sort(key=lambda t: t.arrival_time)
    return threats


def evaluate_clear_quality(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    car_team: int,
    arena: Optional[RocketSimArena] = None,
    opponents: Optional[List[CarState]] = None,
    min_mult: float = 0.6,
    max_mult: float = 1.3
) -> float:
    """
    Evaluates the quality of a clearance strike based on active opponent threat arrival,
    angular alignment, safe pocket targeting, and breakout potential.
    Returns quality multiplier clamped strictly to [min_mult, max_mult].
    """
    if opponents is None and arena is not None:
        opponents = [c for c in arena.cars if c.team != car_team and not c.demoed]
    elif opponents is None:
        opponents = []

    active_opps = [c for c in opponents if not c.demoed and c.team != car_team]
    if not active_opps:
        return 1.0

    ball_speed = float(np.linalg.norm(ball_vel))
    if ball_speed < 50.0:
        return 1.0

    eff_vel = ball_vel.copy()
    defending_y = -ARENA_EXTENT_Y if car_team == 0 else ARENA_EXTENT_Y
    ball_vy_out = eff_vel[1] if car_team == 0 else -eff_vel[1]
    dist_to_defend = abs(ball_pos[1] - defending_y)

    # Lookahead for backboard pinch / rebounds into defensive wall
    if dist_to_defend < 600.0 and ball_vy_out < 0.0 and arena is not None and hasattr(arena, "get_predicted_ball_pos"):
        pred_pos = arena.get_predicted_ball_pos(20)
        if pred_pos is not None:
            pred_disp = pred_pos - ball_pos
            pred_dt = 20 / 120.0  # 20 ticks at 120Hz
            pred_vel = pred_disp / max(1e-4, pred_dt)
            pred_vy_out = pred_vel[1] if car_team == 0 else -pred_vel[1]
            if pred_vy_out > 100.0:
                eff_vel = pred_vel
                ball_vy_out = pred_vy_out
                ball_speed = float(np.linalg.norm(eff_vel))

    unit_ball_vel = eff_vel / max(1e-4, ball_speed)

    # 1. Opponent Threat & Direct Rebound Alignment Penalty
    max_danger_penalty = 0.0
    all_opps_in_defensive_half = True
    opp_fwd_y_positions = []

    for opp in active_opps:
        ball_to_opp = opp.pos - ball_pos
        d_to_opp = max(1e-4, float(np.linalg.norm(ball_to_opp)))
        unit_to_opp = ball_to_opp / d_to_opp

        cos_theta = float(np.dot(unit_ball_vel, unit_to_opp))

        rel_closing = float(np.dot(eff_vel - opp.vel, unit_to_opp))
        if rel_closing > 50.0:
            t_intercept = d_to_opp / rel_closing
        else:
            t_intercept = d_to_opp / max(50.0, float(np.linalg.norm(opp.vel)) * 0.3)

        if cos_theta > 0.4 and t_intercept < 1.5:
            align_factor = min(1.0, (cos_theta - 0.4) / 0.5)
            time_factor = min(1.0, max(0.2, (1.5 - t_intercept) / 0.7))
            danger = 0.40 * align_factor * time_factor
            if danger > max_danger_penalty:
                max_danger_penalty = danger

        opp_vy_fwd = opp.pos[1] if car_team == 0 else -opp.pos[1]
        opp_fwd_y_positions.append(opp_vy_fwd)
        dist_opp_to_defend = abs(opp.pos[1] - defending_y)
        if dist_opp_to_defend >= ARENA_EXTENT_Y:
            all_opps_in_defensive_half = False

    # 2. Safe Pocket / Corner Targeting Bonus
    pocket_bonus = 0.0
    ball_vx_mag = abs(eff_vel[0])
    if abs(ball_pos[0]) > 1800.0 or (ball_vx_mag > 350.0 and (eff_vel[0] * ball_pos[0]) > 0):
        if ball_vy_out >= 0.0 and ball_vx_mag > 350.0:
            pocket_bonus = 0.15

    # 3. Beat-the-Press / Breakout Bonus
    breakout_bonus = 0.0
    ball_fwd_y = ball_pos[1] if car_team == 0 else -ball_pos[1]
    if ball_vy_out > 750.0 and opp_fwd_y_positions:
        max_opp_fwd_y = max(opp_fwd_y_positions)
        if ball_fwd_y > max_opp_fwd_y or (all_opps_in_defensive_half and ball_vy_out > 1000.0):
            breakout_bonus = 0.15

    raw_mult = 1.0 - max_danger_penalty + pocket_bonus + breakout_bonus
    return float(np.clip(raw_mult, min_mult, max_mult))


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
            prev_t = self._prev_touches.get(car.id, car.ball_touches)
            if car.ball_touches > prev_t and (
                ball_vy_out > 150.0 or
                (ball_vx_mag > 350.0 and ball_vy_out >= -100.0) or
                (arena.ball.vel[2] > 400.0 and ball_vy_out >= -50.0)
            ):
                self._prev_touches[car.id] = car.ball_touches
                clear_quality = evaluate_clear_quality(
                    arena.ball.pos, arena.ball.vel, car.team, arena=arena
                )
                return self.save_weight * clear_quality

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
            z_impact = arena.ball.pos[2] + arena.ball.vel[2] * dt + 0.5 * (-650.0) * (dt ** 2)
            is_crossbar_miss = bool(z_impact > GOAL_HEIGHT + 35.0)
            if abs(x_impact) <= GOAL_HALF_WIDTH and not is_crossbar_miss:
                # Shot is directly on target into the net opening!
                on_target_mult = 1.6
            elif ball_y_forward > 1000.0:
                # Ball is in attacking half and heading wide into the backwall/corner or high over crossbar
                # Strictly eliminate progression reward so the bot cannot farm pushing wide
                miss_dist = max(abs(x_impact) - GOAL_HALF_WIDTH, (z_impact - GOAL_HEIGHT) if is_crossbar_miss else 0.0)
                if miss_dist > 400.0:
                    on_target_mult = 0.0
                else:
                    on_target_mult = max(0.0, 1.0 - (miss_dist / 400.0))
            elif (abs(x_impact) > GOAL_HALF_WIDTH * 1.3 or is_crossbar_miss) and ball_y_forward > 0.0:
                # Midfield wide/high trajectory dampening
                miss_val = max(abs(x_impact) - GOAL_HALF_WIDTH, (z_impact - GOAL_HEIGHT) if is_crossbar_miss else 0.0)
                miss_factor = min(1.0, miss_val / 1500.0)
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
        self._prev_car_touches: Dict[int, int] = {}
        self._was_in_strike_zone: Dict[int, bool] = {}

    def _calc_dist(self, car_pos: np.ndarray, ball_pos: np.ndarray) -> float:
        # If BOTH car and ball are near pitch floor (car Z < 200, ball Z < 300), evaluate horizontal (X, Y) distance
        # so low ground flips / wavedashes do not incur an artificial vertical distance penalty.
        # If car is elevated on the wall or in the air (Z >= 200), strictly evaluate full 3D distance.
        if ball_pos[2] < 300.0 and car_pos[2] < 200.0:
            return float(np.linalg.norm(ball_pos[:2] - car_pos[:2]))
        return float(np.linalg.norm(ball_pos - car_pos))

    def _get_target_pos(self, car_pos: np.ndarray, arena: RocketSimArena, is_kickoff: bool) -> np.ndarray:
        """
        Computes the tactical target point for distance delta and alignment.
        When the ball has significant velocity (> 300 uu/s) and has future trajectory,
        blends the target toward predicted future position so reading wall bounces
        and intercepting rebounds yields positive rewards instead of penalties.
        Inside the close strike zone (< 400 uu) or when ball is slow, seamlessly
        locks to the instantaneous ball position for accurate touches.
        """
        target_pos = arena.ball.pos
        if is_kickoff:
            return target_pos

        ball_speed = float(np.linalg.norm(arena.ball.vel))
        if ball_speed > 300.0 and hasattr(arena, "get_predicted_ball_pos"):
            bx, by = float(arena.ball.pos[0]), float(arena.ball.pos[1])
            bvx, bvy = float(arena.ball.vel[0]), float(arena.ball.vel[1])

            # Detect impending wall or backboard rebound:
            # Ball moving fast toward sidewall (|vx| > 500, heading outward, |x| > 2500)
            # or fast toward backboard (|vy| > 600, heading outward, |y| > 3500)
            is_sidewall_bounce = (abs(bvx) > WALL_BOUNCE_VX_THRESHOLD and (bvx * bx) > 0.0 and abs(bx) > 2500.0)
            is_backboard_bounce = (abs(bvy) > WALL_BOUNCE_VY_THRESHOLD and (bvy * by) > 0.0 and abs(by) > 3500.0)

            # Use 1.5s (180 ticks) post-bounce slice for wall bounces; 0.5s (60 ticks) for standard play
            slice_ticks = 180 if (is_sidewall_bounce or is_backboard_bounce) else 60
            pred_pos = arena.get_predicted_ball_pos(slice_ticks)

            if pred_pos is not None:
                raw_ball_dist = self._calc_dist(car_pos, arena.ball.pos)
                proximity_factor = min(1.0, max(0.0, (raw_ball_dist - 250.0) / 350.0))
                speed_factor = min(1.0, max(0.0, (ball_speed - 300.0) / 1200.0))
                # For impending wall bounces, allow stronger blend toward the rebound point
                max_blend = 0.85 if (is_sidewall_bounce or is_backboard_bounce) else 0.65
                blend = max_blend * proximity_factor * speed_factor
                blended = (1.0 - blend) * arena.ball.pos + blend * pred_pos

                # Clamp within arena bounds to prevent numerical overshoot
                target_pos = np.array([
                    float(np.clip(blended[0], -ARENA_EXTENT_X + 100.0, ARENA_EXTENT_X - 100.0)),
                    float(np.clip(blended[1], -ARENA_EXTENT_Y + 100.0, ARENA_EXTENT_Y - 100.0)),
                    float(np.clip(blended[2], 93.0, ARENA_HEIGHT_Z - 100.0))
                ], dtype=np.float32)
        return target_pos

    def reset(self, initial_state: RocketSimArena):
        is_kickoff = bool(
            abs(initial_state.ball.pos[0]) < 50.0 and
            abs(initial_state.ball.pos[1]) < 50.0 and
            initial_state.ball.pos[2] < 120.0 and
            float(np.linalg.norm(initial_state.ball.vel)) < 100.0
        )
        self._prev_dist = {
            car.id: self._calc_dist(car.pos, self._get_target_pos(car.pos, initial_state, is_kickoff))
            for car in initial_state.cars
        }
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._prev_car_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._was_in_strike_zone = {car.id: False for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if is_goal:
            return 0.0

        # Kickoff sprint multiplier & anti-peel penalty (guarantees full-throttle rush on kickoff)
        is_kickoff = bool(
            abs(arena.ball.pos[0]) < 50.0 and
            abs(arena.ball.pos[1]) < 50.0 and
            arena.ball.pos[2] < 120.0 and
            float(np.linalg.norm(arena.ball.vel)) < 100.0
        )

        target_pos = self._get_target_pos(car.pos, arena, is_kickoff)
        curr_dist = self._calc_dist(car.pos, target_pos)
        prev_dist = self._prev_dist.get(car.id, curr_dist)
        self._prev_dist[car.id] = curr_dist

        prev_t = self._prev_touches.get(car.id, car.ball_touches)
        self._prev_touches[car.id] = car.ball_touches

        raw_ball_dist = self._calc_dist(car.pos, arena.ball.pos)

        # Unit alignment vector to tactical target (properly normalized in 3D)
        car_to_ball = target_pos - car.pos
        dist_3d = float(np.linalg.norm(car_to_ball))
        unit_to_ball = car_to_ball / max(1e-4, dist_3d)
        fwd_alignment = float(np.dot(car.get_forward_vector(), unit_to_ball))

        # Defensive coordinate context
        defend_goal_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_car_to_defend = abs(car.pos[1] - defend_goal_y)
        dist_ball_to_defend = abs(arena.ball.pos[1] - defend_goal_y)
        car_vy_defend = -car.vel[1] if car.team == 0 else car.vel[1]  # >0 when moving towards own goal
        is_wrong_side = bool(dist_car_to_defend > dist_ball_to_defend + 50.0)

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

        horiz_ball_dist = float(np.linalg.norm(arena.ball.pos[:2] - car.pos[:2]))
        eff_dist = min(curr_dist, horiz_ball_dist) if (car.on_ground and ball_z < 650.0) else curr_dist

        # 1. Anti-Overshoot Penalty & Strike Zone Tracking
        overshoot_penalty = 0.0
        in_strike = (raw_ball_dist < 400.0) or (car.on_ground and horiz_ball_dist < 380.0 and ball_z < 650.0)
        was_strike = self._was_in_strike_zone.get(car.id, False)
        self._was_in_strike_zone[car.id] = in_strike

        opp_touched = any(
            c.ball_touches > self._prev_car_touches.get(c.id, c.ball_touches)
            for c in arena.cars if c.team != car.team
        )
        self._prev_car_touches = {c.id: c.ball_touches for c in arena.cars}

        car_fwd_spd = float(np.dot(car.vel, car.get_forward_vector())) if not car.on_ground else float(np.dot(car.vel[:2], car.get_forward_vector()[:2]))
        if was_strike and not in_strike and car.ball_touches == prev_t and fwd_alignment < -0.15 and not opp_touched and car_fwd_spd > 150.0:
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

        # When an opponent touches/clears the ball away, distance increased due to the opponent's strike;
        # clamp the delta so the bot is not penalized with an artificial distance penalty cliff for an external hit.
        if opp_touched and raw_delta_dist < 0.0:
            raw_delta_dist = max(-0.08, raw_delta_dist)

        # 2. Distance Delta with Strike Zone Pacing
        # Downfield (> 450 uu): 100% distance closure rewarded
        # Inside strike zone (< 450 uu): Paces approach so car doesn't blindly barrel past ball
        strike_pacing = min(1.0, max(0.20, (eff_dist - 150.0) / 300.0))
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
        # Protects the ENTIRE airborne flight of a dodge/flip (not just the initial tick):
        is_in_flip_flight = bool(not car.on_ground and not car.has_flip and car_horiz_speed > 300.0 and travel_align_to_ball > 0.35)
        is_dodging_toward_ball = bool((car.just_dodged or is_in_flip_flight) and car_horiz_speed > 250.0 and travel_align_to_ball > 0.35)
        is_traveling_toward_ball = bool(travel_align_to_ball > 0.35 and car_horiz_speed > 100.0)

        if fwd_alignment < 0.0 and delta_dist > 0.0 and not is_dodging_toward_ball and not is_traveling_toward_ball:
            delta_dist = delta_dist * max(0.0, fwd_alignment + 1.0) * 0.2

        fwd_vec = car.get_forward_vector()
        right_vec = car.get_right_vector()
        local_x = float(np.dot(car_to_ball[:2], fwd_vec[:2]))
        local_y = float(np.dot(car_to_ball[:2], right_vec[:2]))
        is_roof_carry = bool(
            car.on_ground and
            curr_dist < 195.0 and
            118.0 <= ball_z <= 225.0 and
            abs(local_x) < 80.0 and
            abs(local_y) < 60.0
        )
        is_ground_pushing = bool(raw_ball_dist < 180.0 and ball_z < 130.0 and car.on_ground)

        # 3. Strike-Zone Velocity Matching, Arrival Pacing & Anti-Overshoot (< 500 uu)
        vel_matching_bonus = 0.0
        pacing_penalty = 0.0
        wrong_side_push_penalty = 0.0
        dribble_boost_penalty = 0.0

        self_tti, dist_to_ball, closing_spd = compute_trajectory_arrival_time(car.pos, car.vel, arena.ball.pos, arena.ball.vel)
        threats = compute_opponent_threats(car, arena)
        opp_tti = threats[0].arrival_time if threats else 999.0
        delta_t = opp_tti - self_tti  # > 0: bot arrives first

        in_strike_zone = (raw_ball_dist < 500.0) or (car.on_ground and horiz_ball_dist < 500.0 and ball_z < 650.0)
        if in_strike_zone and fwd_alignment > 0.2:
            car_speed = float(np.linalg.norm(car.vel))
            ball_speed = float(np.linalg.norm(arena.ball.vel))
            car_speed_2d = float(np.linalg.norm(car.vel[:2]))
            ball_speed_2d = float(np.linalg.norm(arena.ball.vel[:2]))
            rel_speed = float(np.linalg.norm(car.vel - arena.ball.vel))
            rel_speed_2d = float(np.linalg.norm(car.vel[:2] - arena.ball.vel[:2]))

            effective_car_speed = car_speed_2d if (car.on_ground and ball_z < 650.0) else car_speed
            effective_ball_speed = ball_speed_2d if (car.on_ground and ball_z < 650.0) else ball_speed
            effective_rel_speed = rel_speed_2d if (car.on_ground and ball_z < 650.0) else rel_speed

            # Gated Velocity Matching:
            # Velocity matching should only reward pacing a moving ball (ball_speed > 250 and car_speed > 200),
            # preventing continuous reward farming when car is merely nose-pushing a grounded ball
            # or sitting stationary near a stopped ball.
            if not (is_wrong_side and car_vy_defend > 100.0):
                if not is_ground_pushing and effective_ball_speed > 250.0 and effective_car_speed > 200.0:
                    vel_matching_bonus = 0.30 * max(0.0, 1.0 - (effective_rel_speed / 700.0))

                # Kinetic Arrival Velocity Pacing Envelope:
                # When closing toward the ball on the ground, evaluate required approach pacing:
                # Pure outcome-driven penalty avoidance: overspeeding incurs pacing_penalty,
                # decelerating to desired speed brings penalty to 0.0. No positive per-tick hovering bounties.
                if car.on_ground and not is_roof_carry and not is_ground_pushing:
                    safe_speed_margin = max(150.0, (min(curr_dist, 500.0) / 500.0) * 650.0)
                    desired_speed = effective_ball_speed + safe_speed_margin

                    # Pacing penalty is mutually exclusive with overshoot penalty (approach vs aftermath)
                    if overshoot_penalty == 0.0:
                        if self_tti < 0.40 and effective_car_speed > desired_speed:
                            excess = (effective_car_speed - desired_speed) / 800.0
                            pacing_penalty = -0.35 * min(1.0, max(0.0, excess))
            else:
                # Car is pushing ball toward own net: apply wrong-side push penalty
                wrong_side_push_penalty = -0.30 * max(0.0, car_vy_defend / 1500.0) * max(0.0, fwd_alignment)

            # Dribble Proximity Pacing & Anti-Overshoot:
            is_close_approach = bool(raw_ball_dist < 350.0 or (horiz_ball_dist < 350.0 and ball_z < 650.0))
            if is_close_approach and car.on_ground and not is_roof_carry:
                if effective_car_speed > effective_ball_speed + 150.0 and float(action[6]) > 0.0:
                    dribble_boost_penalty = -0.30 * float(action[6])

            # Hard Anti-Stacking Floor:
            # Clamps combined strike-zone approach penalties to a maximum floor of -0.50
            total_approach_penalties = overshoot_penalty + pacing_penalty + dribble_boost_penalty
            if total_approach_penalties < -0.50:
                scale = -0.50 / total_approach_penalties
                overshoot_penalty *= scale
                pacing_penalty *= scale
                dribble_boost_penalty *= scale

        # 4. Projected Velocity Toward Ball (Airborne Climbing vs Ground Traversal)
        vel_toward_ball = 0.0
        if is_elevated_aerial:
            if not car.on_ground:
                # Airborne flight: Evaluate true 3D closing velocity toward high ball relative to ball motion
                car_closing_proj = float(np.dot(car.vel, unit_to_ball))
                rel_closing_proj = float(np.dot(car.vel - arena.ball.vel, unit_to_ball))
                ball_proj = float(np.dot(arena.ball.vel, unit_to_ball))

                # When ball is moving away along line of sight (ball_proj > 100), car must outpace it to close distance:
                if ball_proj > 100.0:
                    effective_air_speed = max(0.0, rel_closing_proj)
                else:
                    # Floating or incoming ball: reward flight momentum toward intercept, guided by relative closure
                    effective_air_speed = max(0.0, car_closing_proj) if rel_closing_proj >= 0.0 else max(0.0, rel_closing_proj)

                if effective_air_speed > 0.0:
                    vel_toward_ball = (effective_air_speed / 2300.0) * 0.40 * max(0.0, fwd_alignment)
                elif car_closing_proj < -100.0 and curr_dist > 300.0:
                    # Penalize actively flying away from elevated aerial ball in mid-air
                    vel_toward_ball = (car_closing_proj / 2300.0) * 0.25
            else:
                vel_toward_ball = 0.0
        else:
            # Grounded or low ball: Gate downfield rush when pushing towards defending goal
            if not (is_wrong_side and car_vy_defend > 100.0):
                fwd_speed_to_ball = max(0.0, float(np.dot(car.vel, unit_to_ball)))
                eff_ball_spd = float(np.linalg.norm(arena.ball.vel[:2])) if car.on_ground else float(np.linalg.norm(arena.ball.vel))
                # Prevent nose-push farming when merely rolling behind ball at matching speed:
                if is_ground_pushing and fwd_speed_to_ball <= eff_ball_spd + 50.0:
                    vel_toward_ball = 0.0
                else:
                    speed_taper = min(1.0, max(0.35, (eff_dist - 180.0) / 320.0))
                    effective_alignment = max(fwd_alignment, travel_align_to_ball) if (is_dodging_toward_ball or is_traveling_toward_ball) else max(0.0, fwd_alignment)
                    vel_toward_ball = (fwd_speed_to_ball / 2300.0) * 0.20 * max(0.0, effective_alignment) * speed_taper
                    if is_on_wall and (is_ball_infield or is_car_above_ball or is_elevated_aerial):
                        vel_toward_ball *= 0.15

        # 5. Turnaround Incentive, Lateral Flank Pocket, and Overshoot Resolution
        turnaround_reward = 0.0
        roof_carry_reward = 0.0
        if car.on_ground:
            fwd_vec = car.get_forward_vector()
            right_vec = car.get_right_vector()
            local_x = float(np.dot(car_to_ball[:2], fwd_vec[:2]))
            local_y = float(np.dot(car_to_ball[:2], right_vec[:2]))
            car_fwd_speed = float(np.dot(car.vel[:2], fwd_vec[:2]))
            ball_fwd_speed = float(np.dot(arena.ball.vel[:2], fwd_vec[:2]))
            rel_fwd_speed = car_fwd_speed - ball_fwd_speed

            throttle = float(action[0])
            steer = float(action[1])
            boost = float(action[6])
            handbrake = float(action[7])

            # A. Lateral Flank / Pocket Control (ball rolling alongside car: doors / fenders):
            is_lateral_pocket = bool(
                curr_dist < 320.0 and
                abs(local_x) < 140.0 and
                50.0 < abs(local_y) < 240.0 and
                ball_z < 200.0
            )

            lateral_slip = abs(float(np.dot(car.vel[:2], right_vec[:2])))
            forward_strike_vel = float(np.dot(car.vel[:2], fwd_vec[:2]))
            yaw_rate = abs(float(car.ang_vel[2])) if hasattr(car, "ang_vel") else 0.0

            if is_lateral_pocket:
                # 1. Hook Cut / Lateral Pop: Steering directly into the ball
                steer_into_ball = bool(steer * local_y > 0.15)
                if steer_into_ball:
                    # Pure Physics-Driven Two-Stage Cut Mechanic:
                    # Phase 1: Initiation Angular Redirection (dist > 180 uu or off-angle fwd_alignment < 0.65)
                    # Phase 2: Tire Bite & Strike Drive (dist <= 180 uu and fwd_alignment >= 0.65)
                    is_strike_window = bool(curr_dist <= 180.0 and fwd_alignment >= 0.65)
                    if is_strike_window:
                        if lateral_slip < 80.0 and forward_strike_vel > 250.0:
                            # Tires gripping turf with forward momentum: loads suspension for solid pop
                            grip_factor = 1.0 - (lateral_slip / 80.0)
                            fwd_factor = min(1.0, forward_strike_vel / 1200.0)
                            cut_bonus = 0.30 * min(1.0, abs(steer)) + 0.25 * grip_factor * fwd_factor
                        elif lateral_slip > 150.0:
                            # Drifting sideways into the ball on ice: penalize lateral tire slip
                            cut_bonus = -0.20 * min(1.0, (lateral_slip - 100.0) / 400.0)
                        else:
                            cut_bonus = 0.20 * min(1.0, abs(steer))
                    else:
                        # Initiation: Reward rapid angular yaw rotation toward the ball
                        yaw_bonus = 0.15 * min(1.0, yaw_rate / 2.5)
                        cut_bonus = 0.30 * min(1.0, abs(steer)) + yaw_bonus
                    turnaround_reward += cut_bonus

                # 2. Downfield Speed Matching / Escort in Pocket:
                # Both moving downfield: reward matching the ball's pace so the bot can carry it on its hip
                if car_fwd_speed > 150.0 and ball_fwd_speed > 150.0:
                    pacing_bonus = 0.25 * max(0.0, 1.0 - min(1.0, abs(rel_fwd_speed) / 400.0))
                    turnaround_reward += pacing_bonus

                # 3. Penalize racing ahead and abandoning pocket without cutting:
                if rel_fwd_speed > 250.0 and forward_strike_vel > 200.0 and not steer_into_ball:
                    turnaround_reward -= 0.25 * min(1.0, (rel_fwd_speed - 250.0) / 400.0)

            # B. Close-Proximity Overshoot & Rear Bumper Resolution (ball behind center of mass on turf or bounce, not in pocket or on roof):
            elif not is_roof_carry and (curr_dist < 300.0 or (horiz_ball_dist < 300.0 and ball_z < 650.0)) and local_x < 0.0 and ball_z < 650.0:
                # Speed-Differential Aware Overshoot Resolution:
                # 1. Car outrunning trailing ball downfield:
                # Penalize widening the gap away from the trailing ball downfield:
                if rel_fwd_speed > 150.0 and car_fwd_speed > 150.0:
                    turnaround_reward = -0.25 * min(1.0, (rel_fwd_speed - 150.0) / 400.0)

                # Active steering or rotation to swing around the ball:
                # Gate: require actual vehicle speed > 100 to prevent stationary spinning exploits
                # Rewarded for physical yaw rotation rate, scaled by steering deflection
                car_speed_for_steer = float(np.linalg.norm(car.vel[:2]))
                steer_mag = abs(steer)
                if car_speed_for_steer > 100.0 and steer_mag > 0.15:
                    rot_mult = 0.5 + 0.5 * min(1.0, yaw_rate / 2.5)
                    turnaround_reward += +0.25 * rot_mult * steer_mag

            elif fwd_alignment < -0.25:
                # Downfield ball-behind: reward rapid turnaround rotation to reorient toward ball
                steer_mag = abs(steer)
                if steer_mag > 0.20:
                    rot_mult = 0.5 + 0.5 * min(1.0, yaw_rate / 2.5)
                    turnaround_reward += +0.20 * rot_mult * steer_mag

            # C. Roof Dribble Carry & Velcro Settling (Seer/Nexto Architecture):
            if is_roof_carry:
                target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
                target_goal_dir = np.array([0.0, 1.0 if car.team == 0 else -1.0, 0.0], dtype=np.float32)
                car_to_goal_vel = float(np.dot(car.vel[:2], target_goal_dir[:2]))

                # Anti-Circling Goal Projection (Seer/Nexto Guard):
                # Only reward carrying the ball when advancing downfield toward the opponent net
                if car_to_goal_vel > 50.0:
                    goal_progress = min(1.0, max(0.2, car_to_goal_vel / 1400.0))
                    center_score = max(0.0, 1.0 - (abs(local_x) / 80.0 * 0.5 + abs(local_y) / 60.0 * 0.5))

                    # Velcro Settling Bonus: dampening vertical ball bounce on roof for stable flicks
                    rel_vz = abs(float(arena.ball.vel[2] - car.vel[2]))
                    velcro_bonus = 0.25 * max(0.0, 1.0 - (rel_vz / 120.0))

                    # Velocity Synchronization
                    rel_horiz_speed = float(np.linalg.norm(car.vel[:2] - arena.ball.vel[:2]))
                    sync_bonus = 0.25 * max(0.0, 1.0 - (rel_horiz_speed / 250.0))

                    roof_carry_reward = 0.40 * center_score * goal_progress + velcro_bonus + sync_bonus

        total_reward = self.weight * (
            delta_dist + vel_toward_ball + vel_matching_bonus + pacing_penalty + dribble_boost_penalty +
            overshoot_penalty + ceiling_penalty + wrong_side_push_penalty + turnaround_reward + roof_carry_reward
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
        prev = self._prev_touches.get(car.id, car.ball_touches)
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
                    return self.weight * (2.5 * height_multiplier + airborne_bonus + kickoff_bounty)
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

            clear_quality = 1.0
            clear_urgency = 1.0
            if is_defensive_clear:
                threats = compute_opponent_threats(car, arena)
                if threats:
                    arr_time = threats[0].arrival_time
                    if arr_time < 0.7:
                        clear_urgency = 1.2
                    elif arr_time > 2.5:
                        clear_urgency = 0.7
                    else:
                        clear_urgency = 1.0
                else:
                    clear_urgency = 1.0

                clear_quality = evaluate_clear_quality(
                    arena.ball.pos, arena.ball.vel, car.team, arena=arena
                )

            # --- CASE 1: Ball hit directed toward opponent half / goal ---
            if goal_alignment >= 0.0:
                # On-Target Trajectory Bonus: Check if touch velocity produces a direct shot into the net
                vy_forward = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
                if vy_forward > 80.0:
                    delta_y = abs(target_goal_y - arena.ball.pos[1])
                    dt = delta_y / vy_forward
                    x_impact = arena.ball.pos[0] + arena.ball.vel[0] * dt
                    z_impact = arena.ball.pos[2] + arena.ball.vel[2] * dt + 0.5 * (-650.0) * (dt ** 2)
                    is_crossbar_miss = bool(z_impact > GOAL_HEIGHT + 35.0)
                    if abs(x_impact) <= GOAL_HALF_WIDTH and not is_crossbar_miss:
                        # Direct shot on target into the net opening!
                        goal_alignment = max(goal_alignment, 0.7) + 0.35

                direction_multiplier = 1.0 + (min(1.0, goal_alignment) * 1.5)  # 1.0x -> 2.5x

                # Dual-Path Context Evaluator:
                rel_speed = float(np.linalg.norm(car.vel - arena.ball.vel))
                is_gentle_ground_push = bool(car.on_ground and ball_z < 130.0 and rel_speed < 150.0)

                # Power and directional strike bonus:
                # Rewards solid impact velocity transferred into the ball toward the opponent net
                fwd_vec = car.get_forward_vector()
                effective_rel_strike = abs(float(np.dot(car.vel[:2] - arena.ball.vel[:2], fwd_vec[:2]))) if car.on_ground else rel_speed
                power_bonus = 0.0
                if goal_alignment > 0.2 and not is_gentle_ground_push:
                    power_bonus = min(1.5, max(ball_speed, effective_rel_strike) / 1500.0)

                if is_defensive_clear:
                    clear_bonus = 0.5 * clear_urgency * max(0.0, (clear_quality - 0.5) / 0.5)
                else:
                    clear_bonus = 0.0

                # Soft Possession Catch Bonus:
                # When uncontested (opponent threat arrival > 1.2s or no threats) on a grounded/low ball,
                # reward cushioning the ball (rel_speed < 350.0 uu/s) into an immediate dribble/carry
                # rather than blasting it away uncontrollably.
                soft_catch_bonus = 0.0
                if car.on_ground and ball_z < 200.0 and not is_defensive_clear:
                    threats = compute_opponent_threats(car, arena)
                    opp_arr = threats[0].arrival_time if threats else 999.0
                    self_arr, _, _ = compute_trajectory_arrival_time(car.pos, car.vel, arena.ball.pos, arena.ball.vel)
                    delta_t = opp_arr - self_arr
                    if delta_t > 0.60 and rel_speed < 350.0:
                        soft_catch_bonus = 0.80 * max(0.0, 1.0 - (rel_speed / 350.0))

                base_touch = 0.25 if is_gentle_ground_push else (0.8 + clear_bonus + soft_catch_bonus)

                # Physical Lateral Tire Slip Dampening on Ground Contact:
                # When striking a grounded ball, if the car is sliding laterally across the turf
                # (high lateral slip), the strike lacks traction and glances weakly.
                # Solid strikes with wheels gripping the pitch transfer full impulse into the ball.
                if car.on_ground and ball_z < 180.0:
                    contact_lateral_slip = abs(float(np.dot(car.vel[:2], car.get_right_vector()[:2])))
                    if contact_lateral_slip > 80.0:
                        slip_factor = max(0.2, 1.0 - (contact_lateral_slip - 80.0) / 300.0)
                        base_touch *= slip_factor
                        power_bonus *= slip_factor
                        kickoff_bounty *= slip_factor

                return self.weight * ((base_touch + power_bonus) * direction_multiplier * height_multiplier + airborne_bonus + kickoff_bounty)

            # --- CASE 2: Ball hit directed backward toward defending half / goal ---
            else:
                if is_defensive_clear:
                    # Lateral pinch / side clear out of defensive third
                    if car.on_ground and ball_z < 180.0:
                        contact_lateral_slip = abs(float(np.dot(car.vel[:2], car.get_right_vector()[:2])))
                        clear_base = 0.8 * clear_quality * max(0.4, 1.0 - (contact_lateral_slip / 500.0))
                    else:
                        clear_base = 0.8 * clear_quality
                    return self.weight * (clear_base * height_multiplier + airborne_bonus)
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
        self._halfflip_in_progress: Dict[int, bool] = {}
        self._halfflip_cancel_executed: Dict[int, bool] = {}
        self._halfflip_roll_executed: Dict[int, bool] = {}
        self._flick_window_active: Dict[int, bool] = {}
        self._prev_ball_vel: Dict[int, np.ndarray] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_on_ground = {car.id: car.on_ground for car in initial_state.cars}
        self._prev_has_flip = {car.id: car.has_flip for car in initial_state.cars}
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._prev_vel = {car.id: car.vel.copy() for car in initial_state.cars}
        self._prev_pos_z = {car.id: float(car.pos[2]) for car in initial_state.cars}
        self._challenge_jump_active = {car.id: False for car in initial_state.cars}
        self._halfflip_in_progress = {car.id: False for car in initial_state.cars}
        self._halfflip_cancel_executed = {car.id: False for car in initial_state.cars}
        self._halfflip_roll_executed = {car.id: False for car in initial_state.cars}
        self._flick_window_active = {car.id: False for car in initial_state.cars}
        self._prev_ball_vel = {car.id: initial_state.ball.vel.copy() for car in initial_state.cars}

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

        # Opponent challenge detection:
        threats = compute_opponent_threats(car, arena)
        is_opponent_challenging = False
        if threats:
            most_urgent = threats[0]
            self_arr = compute_car_arrival_time(car, arena.ball.pos, arena.ball.vel)
            delta_t = most_urgent.arrival_time - self_arr
            is_opponent_challenging = bool(
                most_urgent.dist <= 650.0 or
                (most_urgent.closing_speed > 100.0 and most_urgent.arrival_time < 0.85) or
                (abs(delta_t) < 0.30 and self_arr < 0.60)
            )

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

        fwd_vec = car.get_forward_vector()
        right_vec = car.get_right_vector()
        local_x = float(np.dot(car_to_ball[:2], fwd_vec[:2]))
        local_y = float(np.dot(car_to_ball[:2], right_vec[:2]))
        car_speed_horiz = float(np.linalg.norm(car.vel[:2]))
        car_fwd_speed = float(np.dot(car.vel[:2], fwd_vec[:2]))
        pitch_input = float(action[2])
        yaw_input = float(action[3])
        stick_deflection = max(abs(pitch_input), abs(yaw_input))

        # ── 1. Takeoff Transition (Ground -> Air) ─────────────────────────────
        if prev_ground and not car.on_ground and car.vel[2] > 80.0:
            is_on_wall_zone = bool(abs(car.pos[0]) > 3400.0 or abs(car.pos[1]) > 4400.0)
            is_aerial_ball = bool(ball_z > 250.0)
            car_boost = float(car.boost)

            # 1a. Close-Quarters Strike Liftoff, 50/50 Challenge, and Flick Pop Setup:
            is_contested_5050 = bool(dist <= 450.0 and is_opponent_challenging and ball_z < 220.0 and car.pos[2] < 150.0)
            is_flick_liftoff = bool(dist <= 260.0 and 115.0 <= ball_z <= 280.0 and car.pos[2] < 150.0 and abs(local_x) < 90.0 and abs(local_y) < 70.0)
            is_strike_liftoff = bool(dist <= 500.0 and ball_z < 250.0 and car.pos[2] < 150.0 and forward_alignment > 0.20 and takeoff_closing_vel > 150.0 and pitch_input >= -0.10)

            if is_contested_5050 or is_flick_liftoff or is_strike_liftoff:
                if is_contested_5050:
                    self._challenge_jump_active[car.id] = True
                    bonus_scale = 1.2
                elif is_flick_liftoff:
                    self._flick_window_active[car.id] = True
                    bonus_scale = 1.5
                else:
                    self._challenge_jump_active[car.id] = True
                    bonus_scale = (0.8 + 0.4 * forward_alignment)
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
                ball_retreat_vel = float(np.dot(arena.ball.vel, unit_to_ball))
                # Do not reward jumping off the floor after a ball that is already screaming away downfield (> 1100 uu/s)
                if ball_retreat_vel < 1100.0:
                    # Moderate ball (Z <= 450 uu): double-jump or pop reachable with minimal boost
                    if ball_z <= 450.0:
                        reward += self.weight * forward_alignment * 2.0
                    # High aerial ball (Z > 450 uu): requires >= 30 boost to fly
                    elif car_boost >= 30.0:
                        reward += self.weight * forward_alignment * 3.0
                    elif car_boost < 20.0:
                        # Hopeless floor takeoff under high ball with low/no boost
                        reward += -0.15
                elif car_boost >= 30.0 and forward_alignment > 0.4:
                    # Chasing a fast-moving ball requires high commitment; damp takeoff bonus
                    reward += self.weight * forward_alignment * 0.8

            # 1d. Open-field ground traversal & downfield sprint (dist > 650 uu, ball grounded)
            # Rewards initiating forward traversal liftoff when sprinting downfield (neutral or forward pitch)
            elif not is_aerial_ball and not is_on_wall_zone and dist > 650.0 and forward_alignment > 0.40:
                if car_fwd_speed > 400.0 and pitch_input >= -0.10:
                    is_kickoff = bool(abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and arena.ball.pos[2] < 120.0)
                    liftoff_mult = 1.30 if is_kickoff else 0.80
                    reward += self.weight * liftoff_mult * forward_alignment

        # ── 2. Airborne 50/50 Challenge Completion Bonus ──────────────────────
        if not car.on_ground and self._challenge_jump_active.get(car.id, False):
            if car.ball_touches > prev_touch:
                # Intercepted/blocked ball during 50/50 jump window!
                reward += self.weight * 1.5
                self._challenge_jump_active[car.id] = False
        elif car.on_ground:
            self._challenge_jump_active[car.id] = False
            self._flick_window_active[car.id] = False

        # ── 3. Airborne Dodge / Flip & Traversal Impulse ──────────────────────
        is_executing_dodge = bool(not car.on_ground and prev_flip and not car.has_flip)

        # Joystick-Only Dodge Impulse Reconstruction:
        # In Rocket League, flip/dodge direction is governed solely by joystick pitch (action[2]) and yaw (action[3]).
        # Throttle/brake and air roll have NO influence on dodge impulse direction.
        dodge_dir_local = np.array([
            1.0 if pitch_input > 0.25 else (-1.0 if pitch_input < -0.25 else 0.0),
            1.0 if yaw_input > 0.25 else (-1.0 if yaw_input < -0.25 else 0.0),
            0.0
        ], dtype=np.float32)
        dodge_norm = float(np.linalg.norm(dodge_dir_local))
        if dodge_norm > 1e-4:
            dodge_impulse_world = (dodge_dir_local[0] * fwd_vec + dodge_dir_local[1] * right_vec) / dodge_norm
            dodge_align = float(np.dot(dodge_impulse_world[:2], tactical_dir[:2]))
        else:
            dodge_align = 0.0

        is_flick_active = bool(self._flick_window_active.get(car.id, False) or (dist < 260.0 and 115.0 <= ball_z <= 320.0 and abs(local_x) < 100.0 and abs(local_y) < 80.0))

        is_5050_backflip = bool(
            is_executing_dodge and pitch_input < -0.20 and dist <= 450.0 and is_opponent_challenging
            and car_fwd_speed < 200.0 and forward_alignment < 0.20
        )
        is_uncontested_dribble_backflip = bool(is_executing_dodge and pitch_input < -0.20 and dist <= 450.0 and not is_opponent_challenging and not is_flick_active and forward_alignment < -0.20)
        is_halfflip_candidate = bool(is_executing_dodge and pitch_input < -0.20 and forward_alignment < -0.20 and (dist > 450.0 or is_wrong_side))
        is_forward_backflip = bool(is_executing_dodge and pitch_input < -0.20 and not is_5050_backflip and not is_flick_active and not is_halfflip_candidate and (car_fwd_speed > 100.0 or forward_alignment > 0.15))

        # Strict Backflip Penalization & Half-Flip Initiation Tracking:
        # 1. If opponent is challenging within 50/50 distance or bot is in a flick setup, backflipping is permitted.
        # 2. Backflips while moving forward, facing the ball, or overshooting an uncontested dribble are strictly penalized.
        # 3. Only backward dodges when facing away from the target downfield (dist > 450 or retreating) represent valid Half-Flip attempts.
        if is_executing_dodge and pitch_input < -0.20:
            if is_5050_backflip:
                self._challenge_jump_active[car.id] = True
            elif is_flick_active:
                pass  # Free backflip flick / scoop execution
            elif is_halfflip_candidate:
                self._halfflip_in_progress[car.id] = True
                self._halfflip_cancel_executed[car.id] = False
                self._halfflip_roll_executed[car.id] = False
            elif is_forward_backflip or is_uncontested_dribble_backflip:
                reward -= self.weight * 0.80  # Strict penalty against forward backflips and uncontested dribble overshoot backflips

        # Active Half-Flip In-Flight Shaping (Flip Cancel & Air Roll):
        # Once an intended half-flip is initiated, reward pushing pitch forward to cancel
        # and rolling onto wheels, guiding the agent to discover the full mechanics.
        if not car.on_ground and self._halfflip_in_progress.get(car.id, False):
            if pitch_input > 0.25:
                self._halfflip_cancel_executed[car.id] = True
                reward += self.weight * 0.35 * min(1.0, pitch_input)
            if abs(float(action[4])) > 0.20:
                self._halfflip_roll_executed[car.id] = True
                reward += self.weight * 0.30 * min(1.0, abs(float(action[4])))

        if is_executing_dodge:
            is_open_field = bool(dist > 650.0)
            has_traversal_speed = bool(car_speed_horiz > 350.0)
            is_bad_backflip = bool(is_forward_backflip or is_uncontested_dribble_backflip)

            # Dedicated Forward & Diagonal Traversal Flip Incentive:
            # Forward flip (pitch > 0.25) or diagonal speed-flip (pitch > 0.15, |yaw| > 0.15)
            is_forward_flip = bool(pitch_input > 0.25)
            is_diagonal_flip = bool(pitch_input > 0.15 and abs(yaw_input) > 0.15)
            is_forward_or_diagonal = bool((is_forward_flip or is_diagonal_flip) and forward_alignment > 0.30)

            if stick_deflection >= 0.25 and dodge_align > 0.20 and not is_bad_backflip:
                if (not is_open_field) or has_traversal_speed:
                    reward += self.weight * dodge_align * (0.5 + 0.3 * stick_deflection)

                    if is_forward_or_diagonal:
                        speed_progression = min(1.0, max(0.2, car_fwd_speed / 1800.0))
                        diag_bonus = 0.50 if is_diagonal_flip else 0.25
                        reward += self.weight * (0.8 * speed_progression + diag_bonus) * forward_alignment

                    # Kickoff Speed-Flip / Dodge Bounty:
                    is_kickoff = bool(abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and arena.ball.pos[2] < 120.0 and float(np.linalg.norm(arena.ball.vel)) < 100.0)
                    if is_kickoff and dist > 800.0 and (is_forward_flip or is_diagonal_flip):
                        reward += self.weight * 1.50
            elif ball_z > 350.0 and forward_alignment > 0.30:
                # Double jump for high aerial balls
                reward += self.weight * forward_alignment * 0.4

        # ── 3b. Flick Launch Impulse & Goal Acceleration Bonus (Seer/Nexto Architecture) ──
        target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
        target_x = float(np.clip(arena.ball.pos[0], -GOAL_HALF_WIDTH * 0.8, GOAL_HALF_WIDTH * 0.8))
        target_net_pos = np.array([target_x, target_goal_y, GOAL_HEIGHT * 0.35], dtype=np.float32)
        ball_to_net = target_net_pos - arena.ball.pos
        net_dist = float(np.linalg.norm(ball_to_net))
        unit_to_goal = (ball_to_net / max(1e-4, net_dist)) if net_dist > 1e-4 else np.array([0.0, 1.0 if car.team == 0 else -1.0, 0.0], dtype=np.float32)

        if (is_flick_active or dist < 260.0) and car.ball_touches > prev_touch and (is_executing_dodge or car.just_dodged):
            prev_b_vel = self._prev_ball_vel.get(car.id, arena.ball.vel)
            exit_speed_goal = float(np.dot(arena.ball.vel, unit_to_goal))
            delta_v_goal = float(np.dot(arena.ball.vel - prev_b_vel, unit_to_goal))

            if exit_speed_goal > 650.0 and delta_v_goal > 150.0:
                # Genuine explosive flick on target net!
                flick_power = min(3.0, (exit_speed_goal / 600.0) + (delta_v_goal / 500.0))
                reward += self.weight * 1.5 * flick_power
                self._flick_window_active[car.id] = False

        self._prev_ball_vel[car.id] = arena.ball.vel.copy()

        # ── 4. Wavedash & Speed Impulse on Touchdown / Flip Acceleration ─────
        # Rewards speed increases (delta_v > 0) along tactical vector resulting from flips/wavedashes
        tactical_speed_curr = float(np.dot(car.vel[:2], tactical_dir[:2]))
        tactical_speed_prev = float(np.dot(prev_vel[:2], tactical_dir[:2]))
        delta_tactical_speed = tactical_speed_curr - tactical_speed_prev

        # Half-Flip Touchdown Verification:
        # If the bot initiated a half-flip, evaluate upon landing if it completed the turn!
        if (not prev_ground and car.on_ground) and self._halfflip_in_progress.get(car.id, False):
            up = car.get_up_vector()
            up_z = float(up[2])
            # Completed half-flip: wheels down (up_z > 0.60) AND heading inverted forward toward target (forward_alignment > 0.20)
            if up_z > 0.60 and forward_alignment > 0.20:
                # 🏆 PRO HALF-FLIP COMPLETED!
                reward += self.weight * 1.80
            else:
                # ❌ FAILED / NAKED 360° BACKFLIP:
                # The car spun 360 degrees and landed still facing backward, or crashed!
                reward -= self.weight * 0.80
                # Suppress touchdown speed impulse so naked backflips cannot farm speed reward!
                delta_tactical_speed = 0.0

            self._halfflip_in_progress[car.id] = False
            self._halfflip_cancel_executed[car.id] = False
            self._halfflip_roll_executed[car.id] = False

        # Explicit Wavedash Detection:
        # A wavedash occurs when dodging while very low to the turf (prev_pos_z < 55 uu or low airborne)
        # and immediately contacting turf, slamming the flip impulse into ground acceleration (> 120 uu/s)
        was_airborne_low = bool(not prev_ground and prev_pos_z < 55.0)
        did_dodge_or_flip = bool(car.just_dodged or (prev_flip and not car.has_flip))
        is_wavedash = bool(car.on_ground and (was_airborne_low or prev_pos_z < 55.0) and did_dodge_or_flip and delta_tactical_speed > 120.0)
        if is_wavedash:
            reward += self.weight * 1.5 * min(1.0, delta_tactical_speed / 400.0)
        else:
            is_landing_or_dodge = bool((not prev_ground and car.on_ground) or did_dodge_or_flip)
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
            # Strictly gate: if gain_weight is zero, pad collection contributes 0.0
            if self.gain_weight <= 1e-6:
                return 0.0
            # Heavy low-boost hunger: picking up pads when near zero boost is critical for mobility & defense
            hunger_mult = 1.0 + 2.0 * max(0.0, 1.0 - prev)
            base_gain = self.gain_weight * boost_diff * hunger_mult

            # Discrete pad collection event bonus:
            # Small pad (+12 boost): +0.45 * hunger
            # Big orb (+100 boost): +1.20 * hunger
            is_big_pad = bool((curr - prev) > 0.50)
            pickup_bonus = (1.20 if is_big_pad else 0.45) * max(0.5, 1.0 - prev)
            return float(base_gain + self.gain_weight * pickup_bonus)

        elif boost_diff < -1e-5:
            # ── 2. Boost Usage, Waste & Negative Momentum Penalties ─────────────
            # Strictly gate: if lose_weight is zero, boost usage penalties contribute 0.0
            if self.lose_weight <= 1e-6:
                return 0.0
            height_factor = max(0.2, 1.0 - (car.pos[2] / GOAL_HEIGHT))
            loss_rew = self.lose_weight * boost_diff * height_factor

            # Supersonic boost waste penalty: burning boost when already at max speed (>= 2150 uu/s)
            speed = float(np.linalg.norm(car.vel))
            if speed >= 2150.0 and action[6] > 0.0:
                loss_rew -= 0.35 if car.on_ground else 0.20

            # Strike-zone overspeed boost waste penalty: burning boost when closing on ball too fast
            self_arr = compute_car_arrival_time(car, arena.ball.pos, arena.ball.vel)
            ball_speed = float(np.linalg.norm(arena.ball.vel))
            if car.on_ground and action[6] > 0.0 and self_arr < 0.35 and speed > ball_speed + 200.0:
                loss_rew -= 0.25

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
            dist_ball_to_defend = abs(arena.ball.pos[1] - defend_goal_y)

            threats = compute_opponent_threats(car, arena)
            threat_active = bool(
                not threats or threats[0].arrival_time < 1.8 or dist_ball_to_defend < 3000.0
            )
            is_retreating_to_defend = bool(car_vy_defend > 100.0 and fwd_speed > 100.0 and threat_active)

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
            # When low on boost, reward steering toward and routing through active boost pads
            # along the travel path, eliminating straight-line pad skipping.
            if car.on_ground:
                cpx, cpy = float(car.pos[0]), float(car.pos[1])
                fwd = car.get_forward_vector()

                # 3a. Strategic Big Orb Transit Shaping (gated at boost < 50.0, 1200 uu search radius)
                if car.boost < 50.0 and hasattr(arena, "_big_pad_pos_3d") and hasattr(arena, "_big_pad_active"):
                    bg_act = arena._big_pad_active
                    bg_poses = arena._big_pad_pos_3d
                    min_bg_d2 = 1200.0 * 1200.0
                    min_bg_idx = -1
                    for p_idx in range(len(bg_act)):
                        if bg_act[p_idx]:
                            dx = float(bg_poses[p_idx, 0]) - cpx
                            dy = float(bg_poses[p_idx, 1]) - cpy
                            d2 = dx * dx + dy * dy
                            if d2 < min_bg_d2:
                                min_bg_d2 = d2
                                min_bg_idx = p_idx

                    if min_bg_idx >= 0:
                        pad_dist = math.sqrt(min_bg_d2)
                        dx = float(bg_poses[min_bg_idx, 0]) - cpx
                        dy = float(bg_poses[min_bg_idx, 1]) - cpy
                        pad_dir_x = dx / max(1e-4, pad_dist)
                        pad_dir_y = dy / max(1e-4, pad_dist)
                        pad_align = fwd[0] * pad_dir_x + fwd[1] * pad_dir_y
                        speed_to_pad = car.vel[0] * pad_dir_x + car.vel[1] * pad_dir_y

                        if pad_align > 0.20 and speed_to_pad > 150.0:
                            boost_hunger = (50.0 - car.boost) / 50.0
                            if car.boost < 20.0:
                                boost_hunger = min(1.5, boost_hunger * 1.3)
                            prox = 1.0 - (pad_dist / 1200.0)
                            speed_fac = min(1.0, speed_to_pad / 1000.0)
                            return float(self.gain_weight * 0.40 * boost_hunger * pad_align * prox * speed_fac)

                # 3b. Transit Small Pad Shaping (gated at boost < 65.0, 550 uu search radius)
                if car.boost < 65.0 and hasattr(arena, "_small_pad_pos_3d") and hasattr(arena, "_small_pad_active"):
                    sm_act = arena._small_pad_active
                    sm_poses = arena._small_pad_pos_3d
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
                        pad_align = fwd[0] * pad_dir_x + fwd[1] * pad_dir_y
                        speed_to_pad = car.vel[0] * pad_dir_x + car.vel[1] * pad_dir_y

                        if pad_align > 0.25 and speed_to_pad > 150.0:
                            boost_hunger = (65.0 - car.boost) / 65.0
                            prox = 1.0 - (pad_dist / 550.0)
                            speed_fac = min(1.0, speed_to_pad / 1000.0)
                            return float(self.gain_weight * 0.20 * boost_hunger * pad_align * prox * speed_fac)

            return 0.0


# ==============================================================================
# 9. POWERSLIDE & DRIFT TURN-AROUND REWARD (Tight Hairpin Cuts & Snap Pivoting)
# ==============================================================================
class PowerslideReward(BaseReward):
    """
    Rewards tight, responsive ground turnarounds and snap cuts toward the ball.
    Purely outcome-driven based on yaw velocity and positive heading alignment rate when off-axis,
    gated with proximity and self-TTI arrival suppression to prevent drift-skating into the ball.
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

        # Proximity & Imminent Arrival Suppression:
        # Powerslide / drift cuts are strictly for macro turnarounds and pivots at distance.
        # Drift-skating into the strike zone causes uncontrollable lateral slide and overshoots.
        self_tti = compute_car_arrival_time(car, arena.ball.pos, arena.ball.vel)
        if dist < 300.0 or self_tti < 0.40:
            return 0.0

        # Outcome-driven turning performance on ground (fwd_alignment < 0.60, steer > 0.25, speed > 50 uu/s)
        # Eliminates explicit button-checking (action[7] > 0) so any effective turnaround is rewarded,
        # while straightaway powersliding is penalized by CombinedReward's economy penalty.
        speed = float(np.linalg.norm(car.vel))
        steer_mag = abs(float(action[1]))
        yaw_rate = abs(float(car.ang_vel[2])) if hasattr(car, "ang_vel") else 0.0

        if car.on_ground and fwd_alignment < 0.60 and steer_mag > 0.25 and speed > 50.0:
            alignment_rate = max(0.0, fwd_alignment - prev_align)
            # Rapid pivoting (high yaw velocity) or positive heading alignment progression:
            if yaw_rate > 1.2 or alignment_rate > 0.02:
                pivot_efficiency = min(1.0, max(alignment_rate * 5.0, yaw_rate / 3.5))
                turn_bonus = pivot_efficiency * (0.6 + 0.4 * steer_mag)
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
        self._prev_on_ground: Dict[int, bool] = {}
        self._airborne_ticks: Dict[int, int] = {}
        self._was_disoriented: Dict[int, bool] = {}
        self._disoriented_this_flight: Dict[int, bool] = {}
        self._airborne_recovery_total: Dict[int, float] = {}
        self._wall_landed: Dict[int, bool] = {}
        self._halfflip_cancel_executed: Dict[int, bool] = {}
        self._halfflip_cancel_total: Dict[int, float] = {}
        self._takeoff_heading: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_up_z = {car.id: float(car.get_up_vector()[2]) for car in initial_state.cars}
        self._prev_heading = {car.id: 1.0 for car in initial_state.cars}
        self._prev_on_ground = {car.id: car.on_ground for car in initial_state.cars}
        self._airborne_ticks = {car.id: 0 for car in initial_state.cars}
        self._was_disoriented = {car.id: False for car in initial_state.cars}
        self._disoriented_this_flight = {car.id: False for car in initial_state.cars}
        self._airborne_recovery_total = {car.id: 0.0 for car in initial_state.cars}
        self._wall_landed = {car.id: False for car in initial_state.cars}
        self._halfflip_cancel_executed = {car.id: False for car in initial_state.cars}
        self._halfflip_cancel_total = {car.id: 0.0 for car in initial_state.cars}
        self._takeoff_heading = {car.id: 1.0 for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev_ground = self._prev_on_ground.get(car.id, car.on_ground)
        self._prev_on_ground[car.id] = car.on_ground

        up = car.get_up_vector()
        up_z = float(up[2])
        car_z = float(car.pos[2])

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

        if car.on_ground and prev_ground:
            self._airborne_ticks[car.id] = 0
            self._prev_up_z[car.id] = 1.0
            self._prev_heading[car.id] = 1.0
            self._was_disoriented[car.id] = False
            self._disoriented_this_flight[car.id] = False
            self._airborne_recovery_total[car.id] = 0.0
            self._wall_landed[car.id] = False
            self._halfflip_cancel_executed[car.id] = False
            self._halfflip_cancel_total[car.id] = 0.0
            self._takeoff_heading[car.id] = 1.0
            return 0.0

        if prev_ground and not car.on_ground:
            if speed_horiz > 80.0 and fwd_norm > 1e-4:
                self._takeoff_heading[car.id] = float(np.dot(fwd_h / fwd_norm, v_horiz / speed_horiz))
            else:
                car_to_b = arena.ball.pos[:2] - car.pos[:2]
                d_b = float(np.linalg.norm(car_to_b))
                self._takeoff_heading[car.id] = float(np.dot(fwd_h / max(1e-4, fwd_norm), car_to_b / d_b)) if d_b > 1e-4 else 1.0

        if not car.on_ground:
            air_ticks = self._airborne_ticks.get(car.id, 0) + 1
            self._airborne_ticks[car.id] = air_ticks
        else:
            air_ticks = self._airborne_ticks.get(car.id, 0)

        prev_up_z = self._prev_up_z.get(car.id, up_z)
        self._prev_up_z[car.id] = up_z

        vel_z = float(car.vel[2])

        prev_heading = self._prev_heading.get(car.id, curr_heading)
        self._prev_heading[car.id] = curr_heading

        # Track if car was genuinely knocked off-axis / inverted during this airborne sequence
        if up_z < 0.3 or curr_heading < -0.2:
            self._was_disoriented[car.id] = True
            self._disoriented_this_flight[car.id] = True

        car_to_ball = arena.ball.pos - car.pos
        dist_to_ball = float(np.linalg.norm(car_to_ball))
        ball_z = float(arena.ball.pos[2])

        # Aerial engagement check (protects steep climbing & inverted flight during aerials / air dribbles / flip resets)
        unit_to_ball = car_to_ball / max(1e-4, dist_to_ball)
        fwd_align_to_ball = float(np.dot(car.get_forward_vector(), unit_to_ball))
        is_aerial_engagement = bool(ball_z > 350.0 and (dist_to_ball < 450.0 or (fwd_align_to_ball > 0.35 and dist_to_ball < 1500.0)))

        total_reward = 0.0

        # ── 1. Active 3D Disorientation Recovery (Roll & Yaw) ────────────────
        # Only active when the car was genuinely knocked off-axis, inverted, or executed a flip turnaround
        is_recovering = bool(self._was_disoriented.get(car.id, False))
        pitch_input = float(action[2])
        roll_input = float(action[4])
        is_active_halfflip_cancel = bool(air_ticks <= 18 and (pitch_input > 0.3 or abs(roll_input) > 0.25))

        ang_vel = car.ang_vel if hasattr(car, "ang_vel") and car.ang_vel is not None else np.zeros(3, dtype=np.float32)
        # Roll rate: rotation around the car's longitudinal (forward) axis
        roll_rate = float(np.dot(car.get_forward_vector(), ang_vel))
        total_ang_speed = float(np.linalg.norm(ang_vel))

        if not is_aerial_engagement and is_recovering and not car.on_ground:
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

            # 1b. Roll Rate Damping & Settling (D-term):
            # As the car approaches flat attitude (up_z > 0.75), damp angular velocity to prevent rotational overshoot.
            is_touchdown = bool((car_z < 60.0 and vel_z < -50.0) or (not prev_ground and car.on_ground))
            if up_z > 0.75 and not is_touchdown:
                abs_roll = abs(roll_rate)
                # Require both roll rate AND total angular velocity to be controlled (eliminates pitch-tumble blindspot)
                if abs_roll < 1.0 and total_ang_speed < 1.8 and (prev_up_z < 0.90 or delta_up > 0.01):
                    # Stabilized attitude bonus: reward arresting roll velocity near flat
                    settle_bonus = min(rec_budget, (1.0 - abs_roll) * 0.15 * urgency)
                    total_reward += settle_bonus
                    rec_budget = max(0.0, rec_budget - settle_bonus)
                    self._airborne_recovery_total[car.id] = rec_spent + settle_bonus
                elif abs_roll > 2.2 and up_z > 0.85:
                    # Excess rotational inertia penalty: penalize violent spin that will blow past upright
                    excess_spin = min(1.0, (abs_roll - 2.2) / 2.5)
                    total_reward -= excess_spin * 0.15 * urgency

            # 1c. Active Yaw & Momentum Heading Recovery (delta_heading > 0)
            delta_heading = curr_heading - prev_heading
            if delta_heading > 0.0 and prev_heading < 0.90 and speed_horiz > 250.0:
                heading_inversion_mult = 1.0 + max(0.0, -prev_heading) * 1.0
                yaw_rec = min(rec_budget, (delta_heading * 1.0) * heading_inversion_mult * urgency)
                total_reward += yaw_rec
                self._airborne_recovery_total[car.id] = self._airborne_recovery_total.get(car.id, 0.0) + yaw_rec

            # 1d. Dedicated Half-Flip Flip-Cancel & Air-Roll Bonus Budget
            # Has its own independent budget (0.60) so passive delta_up cannot starve the active cancel!
            cancel_spent = self._halfflip_cancel_total.get(car.id, 0.0)
            cancel_budget = max(0.0, 0.80 - cancel_spent)
            if air_ticks <= 18 and up_z < 0.50 and speed_horiz > 200.0:
                step_cancel_reward = 0.0
                if pitch_input > 0.25:
                    c_rew = min(cancel_budget, (pitch_input * 0.40) * urgency)
                    step_cancel_reward += c_rew
                    cancel_budget = max(0.0, cancel_budget - c_rew)
                    self._halfflip_cancel_executed[car.id] = True
                if abs(roll_input) > 0.20:
                    r_rew = min(cancel_budget, (abs(roll_input) * 0.40) * urgency)
                    step_cancel_reward += r_rew
                    cancel_budget = max(0.0, cancel_budget - r_rew)
                total_reward += step_cancel_reward
                self._halfflip_cancel_total[car.id] = cancel_spent + step_cancel_reward

            # Conclude in-flight roll/yaw recovery once upright attitude is restored AND total angular velocity has settled
            if up_z > 0.88 and curr_heading > 0.85 and total_ang_speed < 1.5:
                self._was_disoriented[car.id] = False

        # Upright Roll Input Suppression:
        # Prevents continuous Gaussian action noise or persistent roll holding from rolling off-axis once upright
        if not car.on_ground and up_z > 0.90 and not is_active_halfflip_cancel and not is_aerial_engagement:
            if abs(roll_input) > 0.15:
                total_reward -= (abs(roll_input) - 0.15) * 0.10

        # ── 2. Touchdown Alignment (Evaluated as a single impulse near ground contact) ──
        if (car_z < 60.0 and vel_z < -50.0) or (not prev_ground and car.on_ground):
            had_disorientation = bool(self._was_disoriented.get(car.id, False) or self._disoriented_this_flight.get(car.id, False) or self._halfflip_cancel_executed.get(car.id, False))
            if had_disorientation:
                if up_z > 0.85:
                    total_reward += (up_z * 0.5)
                    # Touchdown spin penalty: landing with high rotational speed causes the car to bounce onto its roof
                    if total_ang_speed > 1.5:
                        spin_excess = min(1.0, (total_ang_speed - 1.5) / 2.5)
                        total_reward -= spin_excess * 0.30
                    if speed_horiz > 300.0 and curr_heading > 0.30:
                        total_reward += (curr_heading * 0.5)
                        # 180° Turnaround Half-Flip Completion Bonus:
                        if self._halfflip_cancel_executed.get(car.id, False) and curr_heading > 0.60 and speed_horiz > 400.0 and self._takeoff_heading.get(car.id, 1.0) < -0.20:
                            total_reward += 1.50
                elif up_z < 0.65 and not is_active_halfflip_cancel:
                    urgency = min(1.0, max(0.4, (800.0 - car_z) / 600.0))
                    # Continuous monotonic penalty: from 0.0 at up_z=0.65, to -0.40 at up_z=0.0 (door), to -0.50 at up_z=-1.0 (inverted roof)
                    if up_z >= 0.0:
                        door_crash = -0.40 * (1.0 - up_z / 0.65)
                    else:
                        door_crash = -0.40 + (up_z * 0.10)
                    total_reward += door_crash * urgency

                # Consume disorientation so touchdown reward only fires once per landing
                self._was_disoriented[car.id] = False
                self._disoriented_this_flight[car.id] = False
                self._halfflip_cancel_executed[car.id] = False
                self._halfflip_cancel_total[car.id] = 0.0

        if car.on_ground:
            self._airborne_ticks[car.id] = 0
            self._prev_up_z[car.id] = 1.0
            self._prev_heading[car.id] = 1.0
            self._was_disoriented[car.id] = False
            self._disoriented_this_flight[car.id] = False
            self._airborne_recovery_total[car.id] = 0.0
            self._wall_landed[car.id] = False
            self._halfflip_cancel_executed[car.id] = False
            self._halfflip_cancel_total[car.id] = 0.0
            self._takeoff_heading[car.id] = 1.0

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
                car_approach_vel = float(np.dot(car.vel, unit_to_ball))
                rel_closing_vel = float(np.dot(car.vel - arena.ball.vel, unit_to_ball))
                # Must be flying toward ball and not hopelessly losing ground to a receding ball:
                if fwd_align > 0.4 and car_approach_vel > 150.0 and rel_closing_vel > -150.0:
                    # Upright bonus: reward upright orientation for shallow aerials, exempt steep climbs (fwd[2] > 0.5)
                    upright_bonus = max(0.0, up_z) * 0.2 if car.get_forward_vector()[2] < 0.5 else 0.1
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

        # Strike-Zone Lateral Slip Regularization:
        # Penalize sliding sideways into the ball in the immediate strike zone instead of biting the turf with traction
        if car.on_ground:
            car_to_ball = arena.ball.pos - car.pos
            dist = float(np.linalg.norm(car_to_ball))
            fwd = car.get_forward_vector()
            fwd_align = float(np.dot(fwd, car_to_ball / max(1e-4, dist)))
            lateral_slip = abs(float(np.dot(car.vel[:2], car.get_right_vector()[:2])))
            if dist < 220.0 and fwd_align > 0.50 and arena.ball.pos[2] < 180.0 and lateral_slip > 150.0:
                slip_pen = -0.20 * min(1.0, (lateral_slip - 100.0) / 400.0)
                total += slip_pen
                if include_breakdown:
                    breakdown["lateral_slip_penalty"] = slip_pen

        # Handbrake Economy Regularization:
        # Penalize dragging handbrake while driving forward on straightaways or gentle curves
        if car.on_ground and float(action[7]) > 0.10:
            fwd = car.get_forward_vector()
            fwd_speed = float(car.vel[0] * fwd[0] + car.vel[1] * fwd[1] + car.vel[2] * fwd[2])
            steer_mag = abs(float(action[1]))
            car_to_ball = arena.ball.pos - car.pos
            dist = float(np.linalg.norm(car_to_ball))
            fwd_align = float(np.dot(fwd, car_to_ball / max(1e-4, dist)))
            if fwd_speed > 300.0 and (steer_mag < 0.25 or (fwd_align > 0.65 and steer_mag < 0.40)):
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


