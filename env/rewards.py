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
    def __init__(
        self,
        weight: float = 10.0,
        first_touch_bonus: float = 35.0,
        aerial_flip_multi: float = 2.0,
        directional_dodge_bounty: float = 15.0,
        kickoff_boost_eff_multi: float = 1.4
    ):
        super().__init__(weight)
        self.first_touch_bonus = first_touch_bonus
        self.aerial_flip_multi = aerial_flip_multi
        self.directional_dodge_bounty = directional_dodge_bounty
        self.kickoff_boost_eff_multi = kickoff_boost_eff_multi
        self._prev_touches: Dict[int, int] = {}
        self._touch_cooldown: Dict[int, float] = {}
        self._first_touch_claimed: bool = False

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._touch_cooldown = {car.id: 0.0 for car in initial_state.cars}
        # Strictly gate First-Touch bounty to authentic center-court kickoffs (Ball at 0,0 and motionless)
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

        if curr > prev and self._touch_cooldown.get(car.id, 0.0) <= 0.0:
            self._touch_cooldown[car.id] = 0.25  # 250ms cooldown prevents grinding 15 touches per second

            # Kickoff First-Touch Bounty (strictly awarded on authentic center-court kickoffs only)
            first_bounty = 0.0
            if getattr(self, "_is_kickoff_episode", True) and not self._first_touch_claimed:
                boost_eff_multi = self.kickoff_boost_eff_multi if car.boost >= 10.0 else 1.0
                first_bounty = self.first_touch_bonus * boost_eff_multi
            self._first_touch_claimed = True

            # Multiplier and bounties for jumping, dodging, and authentic high aerial strikes
            is_airborne_touch = (not car.on_ground) and (car.pos[2] > 120.0) and (arena.ball.pos[2] > 160.0)
            is_high_aerial = is_airborne_touch and (car.pos[2] > 200.0) and (arena.ball.pos[2] > 240.0)
            is_dodge = car.just_dodged or (not car.on_ground and car.pos[2] > 25.0)

            aerial_flip_mult = 1.0
            high_aerial_bounty = 0.0
            if is_high_aerial:
                aerial_flip_mult = self.aerial_flip_multi * 1.5
                high_aerial_bounty = 20.0
            elif is_airborne_touch:
                aerial_flip_mult = self.aerial_flip_multi
                high_aerial_bounty = 10.0
            elif is_dodge:
                aerial_flip_mult = max(1.2, self.aerial_flip_multi * 0.6)

            # Power scaling: reward solid strikes over gentle grazing
            ball_speed = float(np.linalg.norm(arena.ball.vel))
            power_factor = 0.5 + 0.5 * min(1.0, ball_speed / 1500.0)

            # Impact Alignment: reward hitting the ball squarely with front or rear bumper (backflip hits)
            car_to_ball = arena.ball.pos - car.pos
            dist = float(np.linalg.norm(car_to_ball))
            bumper_alignment = 1.0
            directional_bounty = 0.0
            if dist > 1e-4:
                unit_to_ball = car_to_ball / dist
                fwd = car.get_forward_vector()
                align = float(np.dot(fwd, unit_to_ball))
                rear_align = float(np.dot(-fwd, unit_to_ball))
                best_impact = max(align, rear_align)
                bumper_alignment = 1.0 + 0.8 * max(0.0, best_impact)

                # Directional Dodge Strike Bounty: diagonal, side, and backflip strikes
                if car.just_dodged and (abs(action[3]) > 0.1 or abs(action[4]) > 0.1 or action[2] > 0.5):
                    directional_bounty = self.directional_dodge_bounty

            # Anti-Own-Goal Trajectory Penalty:
            # If in defensive half and hit propels ball rapidly towards own defending net
            defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
            in_defensive_half = (arena.ball.pos[1] < 0.0) if car.team == 0 else (arena.ball.pos[1] > 0.0)
            ball_vy_to_net = -arena.ball.vel[1] if car.team == 0 else arena.ball.vel[1]
            if in_defensive_half and ball_vy_to_net > 300.0 and abs(arena.ball.pos[0]) < GOAL_HALF_WIDTH * 2.5:
                # Severe own-goal deflection penalty
                return -30.0

            # Base Hit Bounty + Kickoff First-Touch + High Aerial Bounty + Directional Dodge Bounty
            return (self.weight * aerial_flip_mult * power_factor * bumper_alignment) + first_bounty + high_aerial_bounty + directional_bounty

        return 0.0


class SpeedTowardBallReward(BaseReward):
    """
    Micro-scaled per-step reward for closing distance to the ball through the front bumper.
    Includes anti-overshoot mechanics to encourage powersliding/decelerating on wide-angle approaches.
    Dampens reward when on the wrong side of the ball in the defensive third to prevent own-goal pushing.
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

        # Both forward nose alignment and rear alignment are rewarded (enables backflips and reverse pursuits)
        fwd = car.get_forward_vector()
        fwd_align = max(0.0, float(np.dot(fwd, unit_to_ball)))
        rear_align = max(0.0, float(np.dot(-fwd, unit_to_ball)))
        best_align = max(fwd_align, rear_align)

        speed_toward = float(np.dot(car.vel, unit_to_ball))
        norm_speed = speed_toward / CAR_MAX_SPEED

        # Wrong-Side Defensive Check: if car is between the ball and opponent net in defensive half,
        # suppress forward rush to force car to rotate around to the goal side instead of pushing into net
        defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_car_to_net = abs(car.pos[1] - defending_y)
        dist_ball_to_net = abs(arena.ball.pos[1] - defending_y)
        if dist_car_to_net > dist_ball_to_net and dist_ball_to_net < 3200.0:
            norm_speed *= 0.15

        # Anti-Overshoot Dynamics: when close to ball (< 900 uu) and angle is wide
        if dist < 900.0 and best_align < 0.7:
            car_speed = float(np.linalg.norm(car.vel))
            if car_speed > 1500.0:
                overshoot_ratio = min(1.0, (car_speed - 1500.0) / (CAR_MAX_SPEED - 1500.0))
                norm_speed *= max(0.1, 1.0 - 0.8 * overshoot_ratio)
            # Powerslide bonus for tightening turn into ball
            if action[7] > 0.5:
                best_align = min(1.0, best_align + 0.3)

        # Flip / Dodge boost: rewards front-flipping, speed-flipping, and backflipping directly toward the ball
        dodge_mult = self.dodge_rush_multi if (car.just_dodged and best_align > 0.5) else 1.0

        return self.weight * norm_speed * (0.3 + 0.7 * best_align) * dodge_mult


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
            # Suppress alignment farming during pure tangential orbiting around ball without closing in
            if dist < 650.0:
                speed_closing = float(np.dot(car.vel, unit_to_ball))
                car_speed = float(np.linalg.norm(car.vel))
                if speed_closing < 80.0 and car_speed > 250.0:
                    # Car is driving around the ball in a circle (tangential orbit) - penalize orbit
                    return -self.weight * 0.5

            # Proximity factor: stronger alignment incentive when closer to the ball
            dist_factor = 0.5 + 0.5 * max(0.0, 1.0 - (dist / 3000.0))
            return self.weight * (alignment ** 2) * dist_factor
        elif alignment < -0.4 and dist < 1500.0:
            # Check if car is actively rotating back towards defending net
            defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
            net_vec = np.array([0.0, defending_y, 0.0], dtype=np.float32) - car.pos
            net_dist = float(np.linalg.norm(net_vec))
            if net_dist > 1e-4:
                speed_to_net = float(np.dot(car.vel, net_vec / net_dist))
                if speed_to_net > 300.0:
                    # Car is retreating to defense / net - exempt from facing-away penalty
                    return 0.0

            # Gentle penalty for facing completely away from ball at close range when NOT retreating
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
    Major defensive reward for making goal-line saves and defensive sidewall/corner clears.
    Strictly rate-limited with a 2.5s cooldown per defensive sequence.
    Rewards clearing the ball out of the defensive danger zone or hooking it to the sidewall away from net.
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

        if curr > prev and cd <= 0.0:
            defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
            dist_to_defend = abs(arena.ball.pos[1] - defending_y)
            in_defensive_half = (arena.ball.pos[1] < 0.0) if car.team == 0 else (arena.ball.pos[1] > 0.0)

            ball_vy = arena.ball.vel[1]
            ball_vx = arena.ball.vel[0]
            ball_speed = float(np.linalg.norm(arena.ball.vel))
            
            is_upfield_clear = (ball_vy > 250.0) if car.team == 0 else (ball_vy < -250.0)
            is_sidewall_clear = (abs(ball_vx) > 380.0) and (is_upfield_clear or abs(ball_vy) < 250.0)

            # Case 1: Critical Goal-Line Danger Zone Save (< 2200 uu from defending goal)
            if dist_to_defend < 2200.0 and abs(arena.ball.pos[0]) < GOAL_HALF_WIDTH * 2.0:
                if is_upfield_clear or is_sidewall_clear:
                    self._save_cooldown[car.id] = 2.5
                    power_scale = 1.0 + (ball_speed / BALL_MAX_SPEED) * 0.5
                    return self.weight * power_scale

            # Case 2: Defensive Half Sidewall / Hook Clear (teaches cutting across rolling balls rather than stuttering)
            elif in_defensive_half and (is_upfield_clear or is_sidewall_clear) and ball_speed > 350.0:
                self._save_cooldown[car.id] = 2.5
                power_scale = 0.6 + min(0.4, ball_speed / 2000.0)
                return self.weight * 0.6 * power_scale

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
    Strictly penalizes burning boost while already at supersonic speed (>= 2150 uu/s).
    """
    def __init__(self, weight: float = 0.02):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        car_speed = float(np.linalg.norm(car.vel))

        # Supersonic Boost Waste Penalty: burning boost at max speed wastes 100% of fuel for 0 speed gain
        if action[6] > 0.0 and car_speed >= 2150.0 and car.on_ground:
            return -self.weight * 1.5

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
    Context-Aware Tactical Aerial Challenge Reward.
    Rewards launching and climbing into aerials ONLY when it is the best course of action:
    1. Boost Level Feasibility: requires car.boost >= 15.0 unless defending immediate shot in net.
    2. Tactical Time-to-Ball Advantage: rewards beating/challenging opponent; suppresses late overcommit whiffs.
    3. Ball Height Window: ball genuinely airborne (Z > 200 uu).
    4. Boost-Tax Shield: compensates SaveBoost loss during active high-percentage flight.
    """
    def __init__(self, weight: float = 0.05):
        super().__init__(weight)

    def _get_time_to_ball(self, c: CarState, b_pos: np.ndarray) -> float:
        d = float(np.linalg.norm(b_pos - c.pos))
        if d < 1e-4:
            return 0.0
        unit = (b_pos - c.pos) / d
        closing = float(np.dot(c.vel, unit))
        effective = max(200.0, closing + (500.0 if c.boost > 10.0 else 100.0))
        return d / effective

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        # Ball must be genuinely airborne (> 200 uu) to justify an aerial challenge
        if arena.ball.pos[2] > 200.0:
            car_to_ball = arena.ball.pos - car.pos
            dist = float(np.linalg.norm(car_to_ball))
            if 1e-4 < dist < 2800.0:
                unit_to_ball = car_to_ball / dist
                speed_toward = float(np.dot(car.vel, unit_to_ball))
                dist_factor = max(0.0, 1.0 - (dist / 2800.0))

                # Check defending goal threat or defensive box
                defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
                in_defensive_box = abs(car.pos[1] - defending_y) < 2200.0
                is_threat, threat_intensity, threat_z = arena.get_shot_threat(car.team) if hasattr(arena, "get_shot_threat") else (False, 0.0, 0.0)

                # Context Check 1: Boost Level Feasibility
                has_sufficient_boost = (car.boost >= 15.0) or is_threat or in_defensive_box
                if not has_sufficient_boost and not in_defensive_box:
                    return 0.0

                # Context Check 2: Player Proximity & Time-to-Ball Advantage
                tactical_multiplier = 1.0
                if not is_threat:
                    t_self = self._get_time_to_ball(car, arena.ball.pos)
                    opponents = [c for c in arena.cars if c.team != car.team and not c.demoed]
                    if opponents:
                        t_opp_min = min(self._get_time_to_ball(opp, arena.ball.pos) for opp in opponents)
                        # If opponent will beat us by > 0.8s, jumping late is a reckless whiff -> 0 reward
                        if t_opp_min < t_self - 0.8:
                            return 0.0
                        # If we beat or contest opponent in the air -> tactical bonus
                        if t_self <= t_opp_min + 0.2:
                            tactical_multiplier = 1.4

                # Case 1: Airborne flight tracking & climb towards elevated ball
                if not car.on_ground and car.pos[2] > 35.0:
                    # Concave takeoff curve gives immediate reinforcement upon launch
                    height_norm = min(1.3, math.sqrt(max(0.0, (car.pos[2] - 17.0) / 400.0)))
                    flip_bonus = 1.4 if car.just_dodged else 1.0
                    threat_bonus = 1.8 if (is_threat or in_defensive_box) else 1.0

                    # Boost-tax shield (+0.04 pts/step) to offset SaveBoost loss during valid aerial flight
                    boost_shield = 0.04 if car.boost > 5.0 else 0.0
                    return (self.weight * height_norm * flip_bonus * dist_factor * threat_bonus * tactical_multiplier) + boost_shield

                # Case 2: Ground launch initiation (requires forward speed unless in defensive goal box)
                min_launch_speed = 0.0 if (is_threat or in_defensive_box) else 350.0
                if car.on_ground and action[5] > 0.0 and (speed_toward > min_launch_speed or in_defensive_box) and dist < 1800.0:
                    return self.weight * 1.2 * dist_factor * tactical_multiplier
        return 0.0


class GroundToAirSetupReward(BaseReward):
    """
    Rewards popping a ground ball upward into the air (self-pass setup).
    Triggers when a grounded car impacts a low ball and imparts upward vertical velocity (d_vz > +250 uu/s).
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

        # Must be on ground/low near ball, ball was low (Z < 200), and imparted vertical pop
        if cd <= 0.0 and car.pos[2] < 80.0 and arena.ball.pos[2] < 220.0:
            dist = float(np.linalg.norm(arena.ball.pos - car.pos))
            if dist < 280.0 and d_vz > 250.0:
                self._setup_cooldown[car.id] = 2.0
                pop_scale = min(1.5, max(0.5, d_vz / 500.0))
                return self.weight * pop_scale

        return 0.0


class WallAerialLaunchReward(BaseReward):
    """
    Rewards popping the ball off the sidewall and launching off the wall into an aerial pursuit.
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

        # Detect wall touch: car on/near wall (|x| > 3000 or |y| > 4500, z > 140) touches ball
        is_on_wall = (abs(car.pos[0]) > 3000.0 or abs(car.pos[1]) > 4500.0) and car.pos[2] > 140.0
        if curr_t > prev_t and is_on_wall:
            self._wall_touch_timer[car.id] = 0.8  # 800ms window to jump off wall

        t_timer = self._wall_touch_timer.get(car.id, 0.0)
        if t_timer > 0.0:
            self._wall_touch_timer[car.id] = max(0.0, t_timer - (1.0 / 15.0))
            # If bot jumps or launches into the air off the wall towards the ball
            if (action[5] > 0.0 or not car.on_ground) and cd <= 0.0:
                dist = float(np.linalg.norm(arena.ball.pos - car.pos))
                if dist < 1000.0:
                    self._wall_cooldown[car.id] = 3.0
                    self._wall_touch_timer[car.id] = 0.0
                    return self.weight

        return 0.0


class AirDribbleCarryReward(BaseReward):
    """
    Rewards airborne velocity matching and ball carrying towards the opponent goal (Air-Dribble).
    Active when both car and ball are elevated (Z > 140 uu), in close proximity (< 450 uu), and moving goal-bound.
    """
    def __init__(self, weight: float = 0.06):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        # Both car and ball must be elevated in the air (> 140 uu)
        if not car.on_ground and car.pos[2] > 140.0 and arena.ball.pos[2] > 160.0:
            car_to_ball = arena.ball.pos - car.pos
            dist = float(np.linalg.norm(car_to_ball))
            if dist < 450.0:
                # Relative speed matching (soft touch / control)
                rel_speed = float(np.linalg.norm(car.vel - arena.ball.vel))
                speed_match = max(0.2, 1.0 - (rel_speed / 600.0))
                
                # Goal direction alignment
                target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
                target_goal = np.array([0.0, target_goal_y, GOAL_HEIGHT * 0.5], dtype=np.float32)
                ball_to_goal = target_goal - arena.ball.pos
                norm_goal = np.linalg.norm(ball_to_goal)
                if norm_goal > 1e-4:
                    unit_goal = ball_to_goal / norm_goal
                    ball_vy_goal = float(np.dot(arena.ball.vel, unit_goal))
                    goal_factor = max(0.0, min(1.0, ball_vy_goal / 1000.0))
                    
                    dist_factor = max(0.0, 1.0 - (dist / 450.0))
                    height_factor = min(1.3, car.pos[2] / 400.0)
                    
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

            # Must be an actual forward strike (> 450 uu/s) toward opponent half
            if ((target_heading_positive and ball_vy > 300.0) or (not target_heading_positive and ball_vy < -300.0)) and ball_speed > 450.0:
                target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
                target_goal = np.array([0.0, target_goal_y, GOAL_HEIGHT * 0.5], dtype=np.float32)
                ball_to_goal = target_goal - arena.ball.pos
                norm_goal = np.linalg.norm(ball_to_goal)
                is_goal_bound = False
                if norm_goal > 1e-4:
                    unit_to_goal = ball_to_goal / norm_goal
                    alignment = float(np.dot(arena.ball.vel / ball_speed, unit_to_goal))
                    is_goal_bound = (alignment > 0.65)

                if is_goal_bound:
                    # Lock cooldown for 2.5 seconds to ensure this shot is only rewarded ONCE per attempt
                    self._shot_cooldown[car.id] = 2.5
                    power_scale = 1.0 + (ball_speed / BALL_MAX_SPEED) * 0.5
                    return self.weight * power_scale

        return 0.0


class KickoffReward(BaseReward):
    """
    Rewards rushing the ball at maximum speed specifically on kickoffs.
    Includes flip acceleration bounties and inline boost pad routing bonuses.
    """
    def __init__(self, weight: float = 0.05, flip_bounty: float = 15.0, pad_bounty: float = 10.0):
        super().__init__(weight)
        self.flip_bounty = flip_bounty
        self.pad_bounty = pad_bounty
        self._kickoff_ticks: Dict[int, int] = {}
        self._kickoff_flip_claimed: Dict[int, bool] = {}
        self._kickoff_pad_claimed: Dict[int, bool] = {}
        self._prev_boost: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._kickoff_ticks = {car.id: 0 for car in initial_state.cars}
        self._kickoff_flip_claimed = {car.id: False for car in initial_state.cars}
        self._kickoff_pad_claimed = {car.id: False for car in initial_state.cars}
        self._prev_boost = {car.id: car.boost for car in initial_state.cars}

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
                fwd = car.get_forward_vector()
                align = max(0.0, float(np.dot(fwd, unit_to_ball)))
                
                reward = 0.0
                if speed_toward > 0:
                    reward += self.weight * (speed_toward / CAR_MAX_SPEED) * (align ** 2)

                # Kickoff Inline Small Pad Route Bounty
                prev_b = self._prev_boost.get(car.id, car.boost)
                self._prev_boost[car.id] = car.boost
                if not self._kickoff_pad_claimed.get(car.id, False) and (car.boost > prev_b + 5.0):
                    self._kickoff_pad_claimed[car.id] = True
                    reward += self.pad_bounty

                # Kickoff Speed Flip Acceleration Bounty
                if not self._kickoff_flip_claimed.get(car.id, False) and car.just_dodged and align > 0.65 and speed_toward > 900.0:
                    self._kickoff_flip_claimed[car.id] = True
                    reward += self.flip_bounty

                return reward
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
    Layer 2: Mechanical Roof Carry & Roof Flick Strike Engine.
    Rewards balancing the ball on the car roof and executing powerful flicks into the opponent net.
    """
    def __init__(self, weight: float = 0.04, flick_bounty: float = 30.0):
        super().__init__(weight)
        self.flick_bounty = flick_bounty
        self._prev_carrying: Dict[int, bool] = {}
        self._flick_cooldown: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_carrying = {car.id: False for car in initial_state.cars}
        self._flick_cooldown = {car.id: 0.0 for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        cd = self._flick_cooldown.get(car.id, 0.0)
        if cd > 0.0:
            self._flick_cooldown[car.id] = max(0.0, cd - (1.0 / 15.0))

        rel_pos = arena.ball.pos - car.pos
        horiz_dist = float(np.linalg.norm(rel_pos[:2]))
        vert_dist = rel_pos[2]
        was_carrying = self._prev_carrying.get(car.id, False)

        is_carrying = (horiz_dist < 180.0 and 15.0 < vert_dist < 140.0)
        self._prev_carrying[car.id] = is_carrying

        # Roof Flick Detection: was carrying on roof and executes a jump/dodge that launches ball forward/upward
        if was_carrying and car.just_dodged and cd <= 0.0:
            target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
            target_dir = 1.0 if target_goal_y > 0 else -1.0
            ball_vy_forward = arena.ball.vel[1] * target_dir
            ball_vz = arena.ball.vel[2]
            ball_speed = float(np.linalg.norm(arena.ball.vel))
            
            if (ball_vy_forward > 500.0 or ball_vz > 400.0) and ball_speed > 800.0:
                self._flick_cooldown[car.id] = 2.5
                flick_power = min(1.5, max(1.0, ball_speed / 1500.0))
                return self.flick_bounty * flick_power

        ball_speed = float(np.linalg.norm(arena.ball.vel))
        car_speed = float(np.linalg.norm(car.vel))
        if ball_speed < 200.0 or car_speed < 200.0:
            return 0.0

        if is_carrying:
            rel_vel = float(np.linalg.norm(car.vel - arena.ball.vel))
            speed_match = max(0.0, 1.0 - (rel_vel / 600.0))
            horiz_factor = max(0.0, 1.0 - (horiz_dist / 180.0))
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
            
            # Backflip retreat recovery bonus (accelerating backwards into defense)
            net_vec = net_pos - car.pos
            net_dist = float(np.linalg.norm(net_vec))
            if net_dist > 1e-4 and car.just_dodged and action[2] > 0.5:
                unit_to_net = net_vec / net_dist
                speed_to_net = float(np.dot(car.vel, unit_to_net))
                if speed_to_net > 400.0:
                    return self.weight * alignment_score + 0.08

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

        # Ball proximity and orientation check: allow patient ball-control and bounce pacing
        dist_to_ball = float(np.linalg.norm(arena.ball.pos - car.pos))
        fwd = car.get_forward_vector()
        unit_to_ball = (arena.ball.pos - car.pos) / max(1e-4, dist_to_ball)
        align = float(np.dot(fwd, unit_to_ball))

        # Goal box check: allow patient goalkeeping in net
        defending_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_to_defend_net = abs(car.pos[1] - defending_y)
        in_goal_box = (dist_to_defend_net < 1800.0) and (abs(car.pos[0]) < 1200.0)

        if (dist_to_ball < 700.0 and align > 0.4) or in_goal_box:
            # Within 700 uu of ball and facing it, OR holding goalkeeper stance in goal box: exempt
            ticks = max(0, ticks - 3)
        elif horiz_speed < 160.0 or horiz_disp < 10.0:
            # Idling, oscillating, or hopping in place with low net horizontal speed/displacement
            ticks += 1
        else:
            ticks = max(0, ticks - 2)

        self._idle_ticks[car.id] = ticks

        if ticks > self.grace_steps:
            # Escalates up to 4x penalty as prolonged idling/hopping continues
            escalation = min(4.0, 1.0 + (ticks - self.grace_steps) / 30.0)
            return -self.weight * escalation
        return 0.0


class WallFaceplantPenalty(BaseReward):
    """
    Penalizes slamming into perimeter sidewalls/backboards at high speed when the ball
    is rebounding high above or behind the car without contact.
    """
    def __init__(self, weight: float = 0.04):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        is_near_wall_x = abs(car.pos[0]) > 3900.0
        is_near_wall_y = abs(car.pos[1]) > 4950.0

        if (is_near_wall_x or is_near_wall_y) and car.on_ground:
            speed_into_wall = 0.0
            if is_near_wall_x:
                speed_into_wall = car.vel[0] * np.sign(car.pos[0])
            elif is_near_wall_y:
                speed_into_wall = car.vel[1] * np.sign(car.pos[1])

            if speed_into_wall > 400.0 and arena.ball.pos[2] > 200.0:
                dist_to_ball = float(np.linalg.norm(arena.ball.pos - car.pos))
                if dist_to_ball > 350.0:
                    penalty_scale = min(1.0, speed_into_wall / 1500.0)
                    return -self.weight * penalty_scale

        return 0.0


class BounceInterceptReward(BaseReward):
    """
    Rewards anticipating ball bounce landing points and wall rebounds (0.5s trajectory forecasting).
    """
    def __init__(self, weight: float = 0.04):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        future_pos = arena.get_predicted_ball_pos(60) if hasattr(arena, "get_predicted_ball_pos") else None
        if future_pos is None:
            return 0.0

        # If ball is high, fast, or bouncing
        if arena.ball.pos[2] > 200.0 or abs(arena.ball.vel[2]) > 250.0 or abs(arena.ball.vel[0]) > 600.0:
            car_to_future = future_pos - car.pos
            dist = float(np.linalg.norm(car_to_future))
            if 1e-4 < dist < 3000.0:
                unit = car_to_future / dist
                speed_toward_future = float(np.dot(car.vel, unit))
                if speed_toward_future > 250.0:
                    fwd = car.get_forward_vector()
                    align = max(0.0, float(np.dot(fwd, unit)))
                    return self.weight * (speed_toward_future / CAR_MAX_SPEED) * (align ** 1.5)

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
                aerial_flip_multi=weights.get("touch_aerial_flip_multi", 2.5),
                directional_dodge_bounty=weights.get("directional_dodge_bounty", 15.0),
                kickoff_boost_eff_multi=weights.get("kickoff_boost_eff_multi", 1.4)
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
            "kickoff": KickoffReward(
                weight=weights.get("kickoff_weight", 0.05),
                flip_bounty=weights.get("kickoff_flip_bounty", 15.0),
                pad_bounty=weights.get("kickoff_pad_bounty", 10.0)
            ),
            "face_ball": FaceBallReward(weights.get("face_ball_weight", 0.02)),
            "behind_ball": BehindBallReward(weights.get("behind_ball_weight", 0.03)),
            "possession": PossessionReward(weights.get("possession_weight", 0.04)),
            "dribble": DribbleReward(
                weight=weights.get("dribble_weight", 0.04),
                flick_bounty=weights.get("flick_bounty", 30.0)
            ),
            "air_dribble_carry": AirDribbleCarryReward(weights.get("air_dribble_carry_weight", 0.06)),
            "defensive_position": DefensivePositionReward(weights.get("defensive_position_weight", 0.03)),
            "save_boost": SaveBoostReward(weights.get("save_boost_weight", 0.02)),
            "velocity": VelocityReward(weights.get("velocity_weight", 0.02)),
            "aerial_height": AerialHeightReward(weights.get("aerial_height_weight", 0.05)),
            "bounce_intercept": BounceInterceptReward(weights.get("bounce_intercept_weight", 0.04)),
            "wall_faceplant_penalty": WallFaceplantPenalty(weights.get("wall_faceplant_penalty_weight", 0.04)),
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
            "bounce_intercept_weight": "bounce_intercept",
            "wall_faceplant_penalty_weight": "wall_faceplant_penalty",
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
        if "directional_dodge_bounty" in new_weights and "touch_ball" in self.rewards:
            self.rewards["touch_ball"].directional_dodge_bounty = float(new_weights["directional_dodge_bounty"])
        if "kickoff_boost_eff_multi" in new_weights and "touch_ball" in self.rewards:
            self.rewards["touch_ball"].kickoff_boost_eff_multi = float(new_weights["kickoff_boost_eff_multi"])

        # Kickoff specific params
        if "kickoff_flip_bounty" in new_weights and "kickoff" in self.rewards:
            self.rewards["kickoff"].flip_bounty = float(new_weights["kickoff_flip_bounty"])
        if "kickoff_pad_bounty" in new_weights and "kickoff" in self.rewards:
            self.rewards["kickoff"].pad_bounty = float(new_weights["kickoff_pad_bounty"])

        # Speed toward ball specific params
        if "dodge_rush_multi" in new_weights and "speed_toward_ball" in self.rewards:
            self.rewards["speed_toward_ball"].dodge_rush_multi = float(new_weights["dodge_rush_multi"])

        # Dribble flick specific params
        if "flick_bounty" in new_weights and "dribble" in self.rewards:
            self.rewards["dribble"].flick_bounty = float(new_weights["flick_bounty"])

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        breakdown = {}
        for name, r in self.rewards.items():
            rew = float(r.get_reward(car, arena, action, is_goal, scoring_team))
            total += rew
            breakdown[name] = rew
        return float(total), breakdown
