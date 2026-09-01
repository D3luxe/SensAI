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
# 2. BALL-TO-GOAL PROGRESSION (Field Displacement & On-Target Trajectory)
# ==============================================================================
class BallToGoalVelocityReward(BaseReward):
    """
    Continuous Potential-Based Progression with Goal Opening Targeting.
    Rewards ball velocity directed toward the opponent's goal opening (X in [-GOAL_HALF_WIDTH, +GOAL_HALF_WIDTH]).
    Heavily bonuses on-target trajectories that enter the net (1.6x), while dampening
    wide shots that roll into the backwall/corner beside the goal.
    """
    def __init__(self, weight: float = 1.5):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
        target_x = float(np.clip(arena.ball.pos[0], -GOAL_HALF_WIDTH * 0.8, GOAL_HALF_WIDTH * 0.8))
        target_pos = np.array([target_x, target_goal_y, GOAL_HEIGHT * 0.35], dtype=np.float32)

        ball_to_goal = target_pos - arena.ball.pos
        dist = float(np.linalg.norm(ball_to_goal))
        if dist < 1e-4:
            return 0.0

        unit_to_goal = ball_to_goal / dist
        ball_velocity_toward_goal = float(np.dot(arena.ball.vel, unit_to_goal))

        # On-Target Trajectory & Backwall Miss Multiplier:
        # If ball is moving downfield, calculate where its trajectory intersects the opponent endline
        vy_forward = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
        on_target_mult = 1.0
        if vy_forward > 50.0:
            delta_y = abs(target_goal_y - arena.ball.pos[1])
            dt = delta_y / vy_forward
            x_impact = arena.ball.pos[0] + arena.ball.vel[0] * dt
            if abs(x_impact) <= GOAL_HALF_WIDTH:
                # Shot is directly on target into the net opening!
                on_target_mult = 1.6
            elif abs(x_impact) > GOAL_HALF_WIDTH * 1.3 and abs(arena.ball.pos[1]) > 2000.0:
                # Ball is in attacking half and heading wide into the backwall/corner
                # Dampen reward so the bot is forced to cut the ball inward towards the goal opening
                miss_factor = min(1.0, (abs(x_impact) - GOAL_HALF_WIDTH) / 1500.0)
                on_target_mult = max(0.15, 1.0 - (0.75 * miss_factor))

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
    """
    def __init__(self, weight: float = 0.15):
        super().__init__(weight)
        self._prev_dist: Dict[int, float] = {}
        self._prev_touches: Dict[int, int] = {}
        self._was_in_strike_zone: Dict[int, bool] = {}

    def _calc_dist(self, car_pos: np.ndarray, ball_pos: np.ndarray) -> float:
        # If ball is grounded (Z < 300), evaluate horizontal (X, Y) distance
        # so jumping / flipping never incurs an artificial vertical distance penalty
        if ball_pos[2] < 300.0:
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
        curr_dist = self._calc_dist(car.pos, arena.ball.pos)
        prev_dist = self._prev_dist.get(car.id, curr_dist)
        self._prev_dist[car.id] = curr_dist

        prev_t = self._prev_touches.get(car.id, car.ball_touches)
        self._prev_touches[car.id] = car.ball_touches

        # Unit alignment vector to ball
        car_to_ball = arena.ball.pos - car.pos
        unit_to_ball = car_to_ball / max(1e-4, curr_dist)
        fwd_alignment = float(np.dot(car.get_forward_vector(), unit_to_ball))

        # Kickoff sprint multiplier & anti-peel penalty (guarantees full-throttle rush on kickoff)
        is_kickoff = bool(abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and float(np.linalg.norm(arena.ball.vel)) < 100.0)
        if is_kickoff:
            delta_dist = (prev_dist - curr_dist) / 2000.0
            if abs(car.pos[0]) > 1200.0 and abs(car.pos[1]) > 3200.0:
                return -1.5
            fwd_speed_to_ball = max(0.0, float(np.dot(car.vel, unit_to_ball)))
            vel_toward_ball = (fwd_speed_to_ball / 2300.0) * 0.30 * max(0.0, fwd_alignment)
            kickoff_mult = 3.0 if delta_dist > 0.0 else 2.5
            return (self.weight * delta_dist * kickoff_mult) + vel_toward_ball

        # ── General Open Play ─────────────────────────────────────────────────
        ball_z = float(arena.ball.pos[2])
        is_elevated_aerial = (ball_z > 350.0)

        # 1. Anti-Overshoot Penalty & Strike Zone Tracking
        overshoot_penalty = 0.0
        in_strike = (curr_dist < 400.0)
        was_strike = self._was_in_strike_zone.get(car.id, False)
        self._was_in_strike_zone[car.id] = in_strike

        if was_strike and not in_strike and car.ball_touches == prev_t and fwd_alignment < -0.15:
            # Car was in the strike zone and flew past the ball without touching it!
            overshoot_penalty = -0.30

        # 2. Distance Delta with Strike Zone Pacing
        raw_delta_dist = (prev_dist - curr_dist) / 2000.0

        # Downfield (> 450 uu): 100% distance closure rewarded
        # Inside strike zone (< 450 uu): Paces approach so car doesn't blindly barrel past ball
        strike_pacing = min(1.0, max(0.15, (curr_dist - 150.0) / 300.0))
        delta_dist = raw_delta_dist * strike_pacing

        if curr_dist <= 300.0:
            if delta_dist < 0.0 and fwd_alignment > 0.0:
                delta_dist = 0.0
        elif fwd_alignment < 0.0 and delta_dist > 0.0:
            delta_dist = delta_dist * max(0.0, fwd_alignment + 1.0) * 0.2

        # 3. Strike-Zone Velocity Matching & Brake Incentives (< 450 uu)
        vel_matching_bonus = 0.0
        brake_incentive = 0.0

        if curr_dist < 450.0 and fwd_alignment > 0.2:
            rel_speed = float(np.linalg.norm(car.vel - arena.ball.vel))
            # Reward matching velocity with the ball inside the strike zone
            vel_matching_bonus = 0.15 * max(0.0, 1.0 - (rel_speed / 900.0))

            # If closing dangerously fast (> 1100 uu/s) on a slower ball (< 600 uu/s), reward braking to pace arrival
            car_speed = float(np.linalg.norm(car.vel))
            ball_speed = float(np.linalg.norm(arena.ball.vel))
            if car_speed > 1100.0 and ball_speed < 600.0 and action[0] < -0.05:
                brake_incentive = 0.12 * min(1.0, -action[0])

        # 4. Projected Velocity Toward Ball (Airborne Climbing vs Ground Traversal)
        vel_toward_ball = 0.0
        if is_elevated_aerial:
            if not car.on_ground:
                # Airborne flight: Directly reward 3D closing velocity toward high ball!
                air_climb_speed = max(0.0, float(np.dot(car.vel, unit_to_ball)))
                vel_toward_ball = (air_climb_speed / 2300.0) * 0.35 * max(0.0, fwd_alignment)
            else:
                # Car is staying grounded while ball is floating high in the air
                # Dampen grounded speed reward so bot doesn't exploit circling on floor underneath ball
                fwd_speed_to_ball = max(0.0, float(np.dot(car.vel, unit_to_ball)))
                vel_toward_ball = (fwd_speed_to_ball / 2300.0) * 0.05 * max(0.0, fwd_alignment)
        else:
            # Grounded or low ball: Distance-gated downfield rush
            speed_taper = min(1.0, max(0.0, (curr_dist - 180.0) / 320.0))
            fwd_speed_to_ball = max(0.0, float(np.dot(car.vel, unit_to_ball)))
            vel_toward_ball = (fwd_speed_to_ball / 2300.0) * 0.20 * max(0.0, fwd_alignment) * speed_taper

        total_reward = (self.weight * delta_dist) + vel_toward_ball + vel_matching_bonus + brake_incentive + overshoot_penalty
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
      3. Vertical Aerials: Heavy height scaling (up to 2.5x) and airborne bonuses for aerial challenges.
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
            # Target goal opening rather than pure +Y direction
            target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
            target_x = float(np.clip(arena.ball.pos[0], -GOAL_HALF_WIDTH * 0.75, GOAL_HALF_WIDTH * 0.75))
            target_pos = np.array([target_x, target_goal_y, GOAL_HEIGHT * 0.35], dtype=np.float32)

            ball_to_net = target_pos - arena.ball.pos
            unit_to_goal = ball_to_net / max(1e-4, float(np.linalg.norm(ball_to_net)))

            goal_alignment = 0.0
            if ball_speed > 1e-4:
                unit_ball_vel = arena.ball.vel / ball_speed
                goal_alignment = float(np.dot(unit_ball_vel, unit_to_goal))

            # On-Target Trajectory Bonus: Check if touch velocity produces a direct shot into the net
            vy_forward = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
            if vy_forward > 80.0:
                delta_y = abs(target_goal_y - arena.ball.pos[1])
                dt = delta_y / vy_forward
                x_impact = arena.ball.pos[0] + arena.ball.vel[0] * dt
                if abs(x_impact) <= GOAL_HALF_WIDTH:
                    # Direct shot on target into the net opening!
                    goal_alignment = max(goal_alignment, 0.7) + 0.35

            # Directional multiplier: heavy boost for hits toward opponent net
            if goal_alignment >= 0.0:
                direction_multiplier = 1.0 + (min(1.0, goal_alignment) * 1.5)  # 1.0x -> 2.5x
            else:
                direction_multiplier = max(0.1, (goal_alignment + 1.0) * 0.5)

            # Dual-Path Context Evaluator:
            # Context A: Tactical Boom (Power shot on net / clearing hit)
            power_bonus = 0.0
            if goal_alignment > 0.4:
                power_bonus = min(1.0, ball_speed / 2000.0)

            # Context B: Possession & Controlled Catch (Soft touch / pop that keeps car and ball close)
            rel_speed = float(np.linalg.norm(car.vel - arena.ball.vel))
            control_bonus = max(0.0, 1.0 - (rel_speed / 600.0)) * 0.8

            # Take the best tactical execution (either booming shot on target or surgical possession catch)
            tactical_bonus = max(power_bonus, control_bonus)

            # Height scaling: Ground touch (Z=93) = 1.0x, High Aerial touch (Z=1500) = 2.5x
            ball_z = float(arena.ball.pos[2])
            height_multiplier = 1.0 + 1.5 * max(0.0, min(1.0, (ball_z - 150.0) / 1850.0))

            # Aerial airborne touch bonus (car airborne contesting high ball)
            airborne_bonus = 1.2 if (not car.on_ground and ball_z > 350.0) else 0.0

            # Kickoff first-touch race bounty
            is_kickoff_touch = bool(abs(arena.ball.pos[0]) < 200.0 and abs(arena.ball.pos[1]) < 200.0 and all(c.ball_touches <= 1 for c in arena.cars))
            kickoff_bounty = 1.0 if is_kickoff_touch else 0.0

            base_touch = 0.8
            return (self.weight * (base_touch + tactical_bonus) * direction_multiplier * height_multiplier) + airborne_bonus + kickoff_bounty

        return 0.0


# ==============================================================================
# 5. SPEED & FLIP MOMENTUM (Supersonic Traversal on Low Boost)
# ==============================================================================
class SpeedReward(BaseReward):
    """
    Velocity Projection & Supersonic Traversal Reward.
    Rewards car forward speed directed toward the ball.
    Incentivizes speed-flipping and forward dodges to break the 1400 uu/s ground drive limit
    up to supersonic (2200 - 2300 uu/s) even when boost is empty.
    Strictly gates on positive forward vehicle velocity (zero reward for reversing across the pitch).
    """
    def __init__(self, weight: float = 0.35):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        car_to_ball = arena.ball.pos - car.pos
        dist = float(np.linalg.norm(car_to_ball))
        if dist < 1e-4:
            return 0.0

        # Only reward FORWARD driving speed (car moving in the direction it is facing)
        fwd_speed = float(np.dot(car.vel, car.get_forward_vector()))
        if fwd_speed <= 50.0:
            return 0.0

        unit_to_ball = car_to_ball / dist
        vel_toward_ball = float(np.dot(car.vel, unit_to_ball))

        # Positive normalized forward speed towards ball with strike-zone taper
        speed_taper = min(1.0, max(0.0, (dist - 180.0) / 320.0))
        norm_vel = (max(0.0, vel_toward_ball) / CAR_MAX_SPEED) * speed_taper

        # Supersonic bonus (downfield traversal only)
        supersonic_bonus = (0.15 * speed_taper) if car.is_supersonic else 0.0

        return self.weight * (norm_vel + supersonic_bonus)


# ==============================================================================
# 6. FACE BALL POTENTIAL DELTA (PBRS Nose Alignment without Per-Tick Farming)
# ==============================================================================
class FaceBallReward(BaseReward):
    """
    Potential-Based Alignment Delta Reward (PBRS).
    Rewards the rate of angular convergence toward the ball (alignment_next - alignment_prev).
    Yields strictly 0.0 reward when maintaining heading or driving straight, completely preventing
    per-tick reward farming while providing a direct gradient to rotate the nose toward the ball.
    """
    def __init__(self, weight: float = 0.0):
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
        curr_alignment = float(np.dot(car.get_forward_vector(), unit_to_ball))
        prev_align = self._prev_alignment.get(car.id, curr_alignment)
        self._prev_alignment[car.id] = curr_alignment

        # Pure potential difference delta: (align_t+1 - align_t)
        # Strictly 0.0 when driving straight; positive when rotating toward ball; negative when yawing away.
        delta_alignment = curr_alignment - prev_align
        return self.weight * delta_alignment


# ==============================================================================
# 7. JUMP MOMENTUM BRIDGE (Eliminates First-Jump Latency Barrier)
# ==============================================================================
class JumpBridgeReward(BaseReward):
    """
    Eliminates the initial jump latency barrier when initiating dodges/aerials toward the ball.
    Provides an immediate transition incentive on:
      1. Ground -> Air Takeoff: on the exact frame the car jumps off the ground toward the ball (2.0x on aerials).
      2. Airborne Dodge / Flip: on the exact frame an airborne front-flip/dodge is triggered toward the ball.
    """
    def __init__(self, weight: float = 0.35):
        super().__init__(weight)
        self._prev_on_ground: Dict[int, bool] = {}
        self._prev_has_flip: Dict[int, bool] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_on_ground = {car.id: car.on_ground for car in initial_state.cars}
        self._prev_has_flip = {car.id: car.has_flip for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev_ground = self._prev_on_ground.get(car.id, car.on_ground)
        self._prev_on_ground[car.id] = car.on_ground

        prev_flip = self._prev_has_flip.get(car.id, car.has_flip)
        self._prev_has_flip[car.id] = car.has_flip

        car_to_ball = arena.ball.pos - car.pos
        dist = float(np.linalg.norm(car_to_ball))
        if dist > 200.0:
            unit_to_ball = car_to_ball / dist
            forward_alignment = float(np.dot(car.get_forward_vector(), unit_to_ball))

            # 1. Takeoff Transition (Ground -> Air)
            if prev_ground and not car.on_ground and car.vel[2] > 100.0 and forward_alignment > 0.1:
                aerial_mult = 3.0 if arena.ball.pos[2] > 250.0 else 1.2
                return self.weight * forward_alignment * aerial_mult

            # 2. Dodge / Flip Transition (Airborne Flip toward the ball)
            if not car.on_ground and prev_flip and not car.has_flip and forward_alignment > 0.2:
                return self.weight * forward_alignment * 1.5

        return 0.0


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
        is_kickoff = bool(abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and float(np.linalg.norm(arena.ball.vel)) < 100.0)
        if is_kickoff:
            return 0.0

        boost_diff = math.sqrt(curr) - math.sqrt(prev)

        if boost_diff >= 0:
            return self.gain_weight * boost_diff
        else:
            height_factor = max(0.2, 1.0 - (car.pos[2] / GOAL_HEIGHT))
            loss_rew = self.lose_weight * boost_diff * height_factor

            # Supersonic boost waste penalty: burning boost when already at max speed (>= 2100 uu/s)
            speed = float(np.linalg.norm(car.vel))
            if speed >= 2100.0 and action[6] > 0.0:
                loss_rew -= 0.15

            # Off-axis boost waste penalty: burning boost when facing away from ball on ground (causes wide orbiting)
            if car.on_ground and action[6] > 0.0:
                car_to_ball = arena.ball.pos - car.pos
                dist_to_ball = float(np.linalg.norm(car_to_ball))
                if dist_to_ball > 300.0:
                    unit_to_ball = car_to_ball / dist_to_ball
                    fwd_align = float(np.dot(car.get_forward_vector(), unit_to_ball))
                    if fwd_align < 0.10:
                        loss_rew -= 0.15 * (1.0 - fwd_align)

            return loss_rew

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

        # Active on ground when ball is off-axis and handbrake is applied
        if car.on_ground and fwd_alignment < 0.75 and float(action[7]) > 0.0:
            alignment_rate = max(0.0, fwd_alignment - prev_align)
            steer_mag = abs(float(action[1]))
            handbrake_intensity = max(0.0, float(action[7]))
            turn_bonus = alignment_rate * 5.0 * (0.5 + 0.5 * steer_mag) * handbrake_intensity
            return self.weight * turn_bonus

        return 0.0


# ==============================================================================
# COMBINED MACRO REWARD ENGINE & MANAGER
# ==============================================================================
class CombinedReward:
    """
    Unified Macro Potential-Based Reward Manager.
    Aggregates Macro Goal, Ball-to-Goal, Player-to-Ball, Speed/Flip, Face-Ball, Jump-Bridge, Touch Quality, and Boost.
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
            "speed": SpeedReward(
                weight=weights.get("speed_weight", 0.0)
            ),
            "face_ball": FaceBallReward(
                weight=weights.get("face_ball_weight", 0.0)
            ),
            "jump_bridge": JumpBridgeReward(
                weight=weights.get("jump_bridge_weight", 0.35)
            ),
            "touch": TouchBallReward(
                weight=weights.get("touch_weight", 1.5)
            ),
            "boost": BoostReward(
                gain_weight=weights.get("boost_gain_weight", 0.5),
                lose_weight=weights.get("boost_lose_weight", 0.1)
            ),
            "powerslide": PowerslideReward(
                weight=weights.get("powerslide_weight", 0.30)
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

        if "speed_weight" in new_weights and "speed" in self.rewards:
            self.rewards["speed"].weight = float(new_weights["speed_weight"])

        if "face_ball_weight" in new_weights and "face_ball" in self.rewards:
            self.rewards["face_ball"].weight = float(new_weights["face_ball_weight"])

        if "powerslide_weight" in new_weights and "powerslide" in self.rewards:
            self.rewards["powerslide"].weight = float(new_weights["powerslide_weight"])

        if "jump_bridge_weight" in new_weights and "jump_bridge" in self.rewards:
            self.rewards["jump_bridge"].weight = float(new_weights["jump_bridge_weight"])

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

        # Handbrake Economy Regularization:
        # Penalize holding handbrake while driving forward on straightaways
        if car.on_ground and float(action[7]) > 0.2 and abs(float(action[1])) < 0.2:
            fwd_speed = float(np.dot(car.vel, car.get_forward_vector()))
            if fwd_speed > 300.0:
                pen = -0.10 * float(action[7])
                total += pen
                breakdown["handbrake_penalty"] = pen

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
