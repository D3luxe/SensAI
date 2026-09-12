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
    WALL_BOUNCE_VX_THRESHOLD, WALL_BOUNCE_VY_THRESHOLD, BALL_RADIUS, GRAVITY,
    PREDICTION_HORIZON_TICKS, PREDICTION_HORIZON_S, PREDICTION_TICK_RATE
)

# Clean goal opening clearance thresholds accounting for physical ball sphere radius (91.25 uu)
EFFECTIVE_GOAL_HALF_WIDTH = GOAL_HALF_WIDTH - BALL_RADIUS  # 801.505 uu (clean clearance inside posts)
EFFECTIVE_GOAL_HEIGHT = GOAL_HEIGHT - BALL_RADIUS          # 551.525 uu (clean clearance below crossbar)



def _clip(x, lo: float, hi: float) -> float:
    """Scalar clamp.

    np.clip on a scalar costs about 1.6us of dispatch against 0.03us for two
    comparisons, and this runs on every parsed action of every car every step.
    """
    x = float(x)
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return x


def _norm2(v) -> float:
    """Magnitude of the XY components of a vector.

    Hand-rolled rather than np.linalg.norm: these run tens of times per environment
    step on 2- and 3-element vectors, where numpy's dispatch overhead is several times
    the arithmetic. Results differ from np.linalg.norm only by float32 rounding
    (relative error < 1e-7) because numpy computes the norm of a float32 array in
    float32 while this accumulates in double precision.
    """
    a = float(v[0]); b = float(v[1])
    return math.sqrt(a * a + b * b)


def _norm3(v) -> float:
    """Magnitude of a 3D vector. See _norm2 for why this is not np.linalg.norm."""
    a = float(v[0]); b = float(v[1]); c = float(v[2])
    return math.sqrt(a * a + b * b + c * c)


def unit_horiz(v: np.ndarray) -> np.ndarray:
    """
    Returns the unit horizontal (XY) direction of a 3D vector.
    Truncating a 3D basis vector to [:2] does NOT yield a unit vector: a car pitched
    60 degrees nose-up has |fwd[:2]| = 0.5, which silently halves every alignment and
    local-frame distance computed from it. All horizontal projections must normalize.
    """
    n = _norm2(v)
    if n < 1e-4:
        return np.zeros(2, dtype=np.float32)
    return (v[:2] / n).astype(np.float32)


# Approximate distance from the car's centre of mass to the plane its wheels rest on.
CAR_SURFACE_CLEARANCE = 60.0

# World-space inward normal of the pitch floor.
FLOOR_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def compute_landing_surface_normal(car: CarState, horizon: float = 1.4) -> np.ndarray:
    """
    Predicts which surface an airborne car is about to arrive at and returns that surface's
    inward normal (the direction the car's wheels must point to land cleanly on it).

    Orientation rewards that hard-code world +Z teach the car to put its wheels toward the pitch
    floor even while it is flying at a side wall or backboard, which lands it on its door and
    scrubs off all momentum. Rocket League surfaces are drivable in every orientation, so the
    correct recovery target is the normal of whichever surface the car actually reaches first.

    Ballistic for the floor (gravity acts on Z), linear for the walls (nothing accelerates the car
    horizontally while coasting). Returns FLOOR_NORMAL when no wall is reached inside the horizon,
    which keeps ordinary open-field recoveries behaving exactly as before.
    """
    px, py, pz = float(car.pos[0]), float(car.pos[1]), float(car.pos[2])
    vx, vy, vz = float(car.vel[0]), float(car.vel[1]), float(car.vel[2])

    # Time to the floor plane under gravity.
    floor_z = CAR_SURFACE_CLEARANCE
    t_floor = float("inf")
    if pz > floor_z:
        # Solve 0.5*g*t^2 + vz*t + (pz - floor_z) = 0 for the positive root.
        a = 0.5 * GRAVITY
        disc = vz * vz - 4.0 * a * (pz - floor_z)
        if disc >= 0.0:
            sq = math.sqrt(disc)
            for root in ((-vz - sq) / (2.0 * a), (-vz + sq) / (2.0 * a)):
                if 0.0 < root < t_floor:
                    t_floor = root
    else:
        t_floor = 0.0

    best_t = float("inf")
    best_normal = None

    limit_x = ARENA_EXTENT_X - CAR_SURFACE_CLEARANCE
    limit_y = ARENA_EXTENT_Y - CAR_SURFACE_CLEARANCE

    if vx > 1.0 and px < limit_x:
        t = (limit_x - px) / vx
        if t < best_t:
            best_t, best_normal = t, np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    elif vx < -1.0 and px > -limit_x:
        t = (-limit_x - px) / vx
        if t < best_t:
            best_t, best_normal = t, np.array([1.0, 0.0, 0.0], dtype=np.float32)

    if vy > 1.0 and py < limit_y:
        t = (limit_y - py) / vy
        if t < best_t:
            best_t, best_normal = t, np.array([0.0, -1.0, 0.0], dtype=np.float32)
    elif vy < -1.0 and py > -limit_y:
        t = (-limit_y - py) / vy
        if t < best_t:
            best_t, best_normal = t, np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # A car already pressed against a wall counts as arriving there now, even with little
    # closing speed: it is on the wall surface and must stay oriented to it.
    if abs(px) > limit_x - 60.0 and pz > 100.0:
        w = np.array([-math.copysign(1.0, px), 0.0, 0.0], dtype=np.float32)
        if 0.0 < best_t:
            best_t, best_normal = 0.0, w
    elif abs(py) > limit_y - 60.0 and pz > 100.0:
        w = np.array([0.0, -math.copysign(1.0, py), 0.0], dtype=np.float32)
        if 0.0 < best_t:
            best_t, best_normal = 0.0, w

    if best_normal is not None and best_t <= horizon and best_t < t_floor:
        return best_normal
    return FLOOR_NORMAL


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
    dist = _norm3(rel_pos)
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
        speed = _norm3(vel)
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


def compute_effective_alignment(
    car: CarState,
    target_dir: np.ndarray,
    min_speed: float = 150.0
) -> float:
    """
    Computes effective alignment toward target_dir.
    Evaluates max(nose_alignment, vel_alignment) when traveling or airborne,
    so that flips, dodges, and flicks (where nose pitches 360 degrees) do not
    suffer false negative alignment penalties while momentum is heading to target.

    Safeguards:
      1. Stationary liftoffs: If horizontal velocity is low (< min_speed), pure vertical liftoff
         does not count as traveling toward the target (evaluates true nose heading).
      2. On ground: lateral tire slip > 100 uu/s gates velocity alignment to prevent drift farming.
      3. Sustained high aerial flight without dodge (Z > 350, not dodging): requires nose alignment for boost thrusters.
    """
    fwd_vec = car.get_forward_vector()
    fwd_align = float(np.dot(fwd_vec, target_dir))
    car_horiz_speed = _norm2(car.vel)
    car_speed = _norm3(car.vel)

    # If car has no horizontal travel momentum, vertical jumping doesn't make it travel to target
    if car_horiz_speed <= min_speed and car.pos[2] < 200.0:
        return fwd_align

    if car_speed <= min_speed:
        return fwd_align

    vel_dir = car.vel / max(1e-4, car_speed)
    vel_align = float(np.dot(vel_dir, target_dir))

    if not car.on_ground:
        # If in active dodge/flip or low flight: evaluate max to decouple 360 pitch rotation
        is_flipping = bool(car.just_dodged or getattr(car, "is_dodging", False) or (not car.has_flip and car.pos[2] < 350.0))
        if is_flipping:
            return max(fwd_align, vel_align)
        # Sustained high aerial flight: blend toward nose alignment for thruster propulsion
        if car.pos[2] > 350.0:
            return 0.70 * fwd_align + 0.30 * max(0.0, vel_align)
        return max(fwd_align, vel_align)
    else:
        # On ground: verify wheels are not in uncontrolled lateral slide (drift farming guard)
        right_vec = car.get_right_vector()
        lateral_slip = abs(float(np.dot(car.vel[:2], right_vec[:2])))
        if lateral_slip > 100.0:
            slip_penalty = min(1.0, (lateral_slip - 100.0) / 300.0)
            gated_vel_align = vel_align * (1.0 - 0.75 * slip_penalty)
            return max(fwd_align, gated_vel_align)
        return max(fwd_align, vel_align)



# Candidate lookahead slices (ticks at 120 Hz) for the intercept solver. Coarse on purpose:
# each rung costs an arrival-time solve, and this runs for every body every step. 60 and 180 are
# already requested by the observation builder, so those two come back from the arena's cache.
INTERCEPT_SLICE_LADDER = (0, 30, 60, 120, 180)

# Depth goalside of the ball that a beaten defender should be shadowing toward.
SHADOW_OFFSET_Y = 700.0

# Rung used to decide whether the engine's trajectory can be trusted at all. Shared with the
# ladder so the probe never costs an extra trajectory integration on the pure-Python fallback.
_TRUST_PROBE_TICKS = 30


def threat_safety_multiplier(threat_intensity: float) -> float:
    """
    Fraction of a boost detour still affordable with a shot developing on our own net.

    Both boost-pathing gates used to switch off at `threat_intensity < 0.40`, which had two
    problems. A detour that was fully affordable one tick became entirely forbidden the next, and
    the threshold's real meaning was silently tied to the threat decay ramp: widening that ramp
    from 1.0s to 3.0s moved the cutoff from 0.6s to 1.8s of time-to-goal without anyone touching
    the number.

    Scaling the budget by the complement of intensity fixes both. The budget fades out smoothly as
    the threat closes, and it is expressed in units of intensity rather than in units of the ramp,
    so changing the horizon no longer retunes the gate behind your back.

    Clamped at both ends. The lower clamp stops a maximal threat from turning the budget negative;
    the upper one stops an out-of-range intensity from handing out a budget larger than the
    unthreatened maximum, which is the only way this could ever loosen a gate rather than tighten
    it.
    """
    return _clip(1.0 - float(threat_intensity), 0.0, 1.0)


def predictions_trustworthy(arena: RocketSimArena) -> bool:
    """Whether the engine's ball trajectory actually describes the ball the rewards can see.

    The trajectory comes from the physics backend's own ball, so any caller that writes
    arena.ball.* directly without pushing the change down gets a prediction for a different ball
    -- typically the one sitting at kickoff. Chasing that is worse than not predicting at all.

    Compares the nearest rung against a straight-line extrapolation of the live ball. Even a full
    reversal off a wall only deviates by twice the distance travelled, so anything beyond that
    means the prediction is describing some other ball. Cached per step: the answer depends only
    on the ball, while the solver runs once per body.
    """
    step = getattr(arena, "step_count", -1)
    if getattr(arena, "_pred_trust_step", None) == step:
        return arena._pred_trust
    arena._pred_trust_step = step

    trusted = True
    probe = arena.get_predicted_ball_pos(_TRUST_PROBE_TICKS)
    if probe is None:
        trusted = False
    else:
        probe_t = _TRUST_PROBE_TICKS / PREDICTION_TICK_RATE
        bp, bv = arena.ball.pos, arena.ball.vel
        dx = float(probe[0]) - (float(bp[0]) + float(bv[0]) * probe_t)
        dy = float(probe[1]) - (float(bp[1]) + float(bv[1]) * probe_t)
        dz = float(probe[2]) - (float(bp[2]) + float(bv[2]) * probe_t)
        tolerance = 2.0 * _norm3(bv) * probe_t + 300.0
        trusted = bool(dx * dx + dy * dy + dz * dz <= tolerance * tolerance)

    arena._pred_trust = trusted
    return trusted


def solve_intercept_point(
    pos: np.ndarray,
    vel: np.ndarray,
    arena: RocketSimArena,
    ladder: Tuple[int, ...] = INTERCEPT_SLICE_LADDER
) -> Tuple[np.ndarray, float]:
    """
    Finds where along the ball's predicted trajectory a body at (pos, vel) can first meet it.

    Aiming a fixed distance into the future -- 0.5s for open play, 1.5s for a detected wall
    rebound -- is only correct when the chaser happens to need exactly that long to arrive. Every
    other time it aims at a point the car reaches early or late, which is what makes bounce reads
    and arrival timing look mistimed. Instead walk the trajectory outward and take the earliest
    slice the body can actually reach, which is the definition of an intercept.

    Returns (intercept_pos, intercept_time_seconds). Falls back to the furthest slice when the
    ball outruns the body entirely, and to the live ball position when no usable prediction exists.
    """
    if not hasattr(arena, "get_predicted_ball_pos") or not predictions_trustworthy(arena):
        return arena.ball.pos, 0.0

    fallback_pos = arena.ball.pos
    fallback_t = 0.0
    for slice_ticks in ladder:
        slice_t = slice_ticks / PREDICTION_TICK_RATE
        pred = arena.ball.pos if slice_ticks == 0 else arena.get_predicted_ball_pos(slice_ticks)
        if pred is None:
            continue
        fallback_pos, fallback_t = pred, slice_t
        arrival, _, _ = compute_trajectory_arrival_time(pos, vel, pred)
        if arrival <= slice_t:
            # Reachable with time to spare: this is the earliest meeting point on the trajectory.
            return pred, slice_t

    # Ball outruns the chaser across the whole ladder; chase the furthest point considered.
    return fallback_pos, fallback_t


def cached_intercept_point(arena: RocketSimArena, body_id: int, pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, float]:
    """Per-step memo around solve_intercept_point, keyed by body id.

    Several terms need the same body's intercept within one step -- the pursuit target, the
    arrival-timing term, the boost-routing gate, and the opponent race -- and the ladder walk
    costs a handful of arrival-time solves each. Positions and velocities are frozen for the
    duration of a step, so one answer per body per step is exactly equivalent.
    """
    step = getattr(arena, "step_count", -1)
    if getattr(arena, "_intercept_cache_step", None) != step:
        arena._intercept_cache_step = step
        arena._intercept_cache = {}
    cache = arena._intercept_cache
    hit = cache.get(body_id)
    if hit is None:
        hit = solve_intercept_point(pos, vel, arena)
        cache[body_id] = hit
    return hit


def compute_opponent_threats(
    car: CarState,
    arena: RocketSimArena,
    target_pos: Optional[np.ndarray] = None,
    opponents: Optional[List[CarState]] = None,
    use_intercept: bool = False
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

    # use_intercept times each opponent to its OWN intercept point on the ball's predicted
    # trajectory instead of to where the ball happens to be right now. Use it for genuine races
    # ("who reaches the ball first"), where timing everyone to the live ball misreads a bouncing
    # ball: an opponent driving to the bounce spot reads as slow and one trailing reads as fast.
    #
    # Leave it off for pressure heuristics ("is someone contesting me"), which are proximity
    # questions. Those must not care that a ball the bot just flicked away at 1600 uu/s is no
    # longer interceptable -- the challenger standing on top of the bot is still a challenger.
    solve_per_opponent = bool(use_intercept and target_pos is None and hasattr(arena, "get_predicted_ball_pos"))
    ref_pos = arena.ball.pos if target_pos is None else target_pos

    threats: List[OpponentThreat] = []
    for opp in opponents:
        if opp.demoed or opp.id == car.id or opp.team == car.team:
            continue
        opp_ref = cached_intercept_point(arena, opp.id, opp.pos, opp.vel)[0] if solve_per_opponent else ref_pos
        arrival, dist, closing_speed = compute_trajectory_arrival_time(opp.pos, opp.vel, opp_ref)
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

    ball_speed = _norm3(ball_vel)
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
            pred_dt = 20 / PREDICTION_TICK_RATE  # 20 ticks at 120Hz
            pred_vel = pred_disp / max(1e-4, pred_dt)
            pred_vy_out = pred_vel[1] if car_team == 0 else -pred_vel[1]
            if pred_vy_out > 100.0:
                eff_vel = pred_vel
                ball_vy_out = pred_vy_out
                ball_speed = _norm3(eff_vel)

    unit_ball_vel = eff_vel / max(1e-4, ball_speed)

    # 1. Opponent Threat & Direct Rebound Alignment Penalty
    max_danger_penalty = 0.0
    all_opps_in_defensive_half = True
    opp_fwd_y_positions = []

    for opp in active_opps:
        ball_to_opp = opp.pos - ball_pos
        d_to_opp = max(1e-4, _norm3(ball_to_opp))
        unit_to_opp = ball_to_opp / d_to_opp

        cos_theta = float(np.dot(unit_ball_vel, unit_to_opp))

        rel_closing = float(np.dot(eff_vel - opp.vel, unit_to_opp))
        if rel_closing > 50.0:
            t_intercept = d_to_opp / rel_closing
        else:
            t_intercept = d_to_opp / max(50.0, _norm3(opp.vel) * 0.3)

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
    return _clip(raw_mult, min_mult, max_mult)


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
        self._opp_touches: Dict[int, int] = {}
        self._self_touched_since_opp: Dict[int, bool] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._opp_touches = {c.id: c.ball_touches for c in initial_state.cars}
        self._self_touched_since_opp = {car.id: False for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if is_goal and scoring_team is not None:
            self._prev_touches[car.id] = car.ball_touches
            return self.weight if car.team == scoring_team else self.concede_weight

        # Anti-farming gate: deny the save when THIS car is the one that last put the ball in
        # motion. Without it the save is farmable: nudge the ball off-target toward your own
        # half (explicitly unpenalized by TouchBallReward CASE 2), then "clear" it for
        # save_weight. An opponent touch clears the flag, so a genuine incoming shot always
        # pays; a threat the car created for itself does not. Note this is deliberately weaker
        # than requiring an opponent touch outright, which would deny the first save of an
        # episode and every save off a wall bounce or a loose ball.
        for c in arena.cars:
            if c.team != car.team and c.ball_touches > self._opp_touches.get(c.id, c.ball_touches):
                self._self_touched_since_opp[car.id] = False
        self._opp_touches = {c.id: c.ball_touches for c in arena.cars}

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
                if self._self_touched_since_opp.get(car.id, False):
                    return 0.0
                self._self_touched_since_opp[car.id] = True
                clear_quality = evaluate_clear_quality(
                    arena.ball.pos, arena.ball.vel, car.team, arena=arena
                )
                return self.save_weight * clear_quality

        # Any own touch — including the off-target nudge toward our own half that starts the
        # farming loop — marks this car as the author of the ball's current trajectory.
        if car.ball_touches > self._prev_touches.get(car.id, car.ball_touches):
            self._self_touched_since_opp[car.id] = True
        self._prev_touches[car.id] = car.ball_touches
        return 0.0


# ==============================================================================
# 2. BALL-TO-GOAL PROGRESSION (Field Displacement & On-Target Trajectory)
# ==============================================================================
# Endline projection shared by BallToGoalVelocityReward and OwnGoalThreatReward. Both used to
# carry their own copy of the same ballistic block, with the same gravity constant and the same
# goal geometry, and disagree about what to do with the result.
_ENDLINE_SCAN_STRIDE = 10
# Asking for a tick past the end of the prediction array falls through get_predicted_ball_pos's
# out-of-range path into a separate pure-Python simulation, whose state does not line up with the
# slices around it -- estimating a velocity across that seam produced a continuation pointing
# somewhere the ball never goes. Stay inside the array.
_ENDLINE_MAX_SLICE = PREDICTION_HORIZON_TICKS
# A shot missing the opening by more than this earns no placement bonus at all. Two ball radii
# is the width of the graze band the old branch ladder covered, expressed once.
ON_TARGET_FALLOFF = 2.0 * BALL_RADIUS
ON_TARGET_BONUS = 0.6

# Strike scoring: goalward velocity ADDED to the ball by the contact. Reference is the impulse a
# solid strike imparts; the cap matches the combined ceiling of the power and impulse bonuses this
# replaced, so a maximal strike is worth what it was worth before the consolidation.
STRIKE_IMPULSE_REF = 1000.0
STRIKE_IMPULSE_CAP = 2.3


def project_ball_to_endline(arena: RocketSimArena, target_goal_y: float) -> Optional[Tuple[float, float]]:
    """
    Where the ball would cross the plane y = target_goal_y, as (x_impact, z_impact).
    Returns None when it never gets there.

    Walks the RocketSim predicted trajectory first. Rocket League is a cage, and a bare parabola
    has no ceiling at 2044 uu and no side walls at +/-4096 uu: a lob that bounces off the ceiling
    down into the net projects to an impact height above the arena, and an angled shot toward a
    corner yields an impact x outside the world. The predictor models all of that. It spans
    PREDICTION_HORIZON_S, so beyond the horizon this continues ballistically from the last
    predicted slice, which is safe because a ball that far from the endline is nowhere near the
    backboard and gravity dominates its arc.
    """
    sign = 1.0 if target_goal_y > 0.0 else -1.0
    ball_pos = arena.ball.pos
    if sign * (float(ball_pos[1]) - target_goal_y) >= 0.0:
        return None  # already past the plane

    prev_pos = np.asarray(ball_pos, dtype=np.float32)
    prev_t = 0.0
    have_pred = bool(hasattr(arena, "get_predicted_ball_pos") and predictions_trustworthy(arena))

    if have_pred:
        # Track the closest approach to the plane as well as an outright crossing. A shot that
        # strikes the back wall or the backboard beside the opening never reaches y = goal_y at
        # all -- it stops a ball radius short and rebounds -- so waiting for a crossing would
        # miss every wide or over-the-bar shot and fall through to a continuation computed from
        # post-bounce state, which points somewhere meaningless.
        best_fwd = sign * float(prev_pos[1])
        best_pos = prev_pos
        turned_back = False
        prev_prev = prev_pos

        for s in range(_ENDLINE_SCAN_STRIDE, _ENDLINE_MAX_SLICE + 1, _ENDLINE_SCAN_STRIDE):
            pred = arena.get_predicted_ball_pos(s)
            if pred is None:
                break
            if sign * (float(pred[1]) - target_goal_y) >= 0.0:
                # Bracketed: linear interpolation inside one stride is well under a ball radius
                # of error at any speed the ball can reach.
                y0, y1 = float(prev_pos[1]), float(pred[1])
                span = y1 - y0
                frac = 1.0 if abs(span) < 1e-6 else _clip((target_goal_y - y0) / span, 0.0, 1.0)
                x_i = float(prev_pos[0]) + frac * (float(pred[0]) - float(prev_pos[0]))
                z_i = float(prev_pos[2]) + frac * (float(pred[2]) - float(prev_pos[2]))
                return x_i, max(BALL_RADIUS, z_i)

            fwd = sign * float(pred[1])
            if fwd > best_fwd:
                best_fwd, best_pos = fwd, np.asarray(pred, dtype=np.float32)
            elif fwd < best_fwd - 1.0:
                turned_back = True
                break

            prev_prev = prev_pos
            prev_pos = np.asarray(pred, dtype=np.float32)
            prev_t = s / PREDICTION_TICK_RATE

        if turned_back:
            # Deepest point reached is where it met the wall. That is the impact point.
            return float(best_pos[0]), max(BALL_RADIUS, float(best_pos[2]))

    # Beyond the prediction horizon (or with no usable prediction): ballistic continuation from
    # wherever the trajectory left off, using the velocity across the last two scanned slices so
    # the continuation starts from post-bounce state rather than the ball's launch velocity.
    if prev_t > 0.0:
        vel = (prev_pos - prev_prev) / (_ENDLINE_SCAN_STRIDE / PREDICTION_TICK_RATE)
    else:
        vel = arena.ball.vel

    vy = float(vel[1])
    if sign * vy <= 50.0:
        return None  # not travelling toward that endline fast enough to arrive

    dt = (target_goal_y - float(prev_pos[1])) / vy
    if dt <= 0.0:
        return None
    x_i = float(prev_pos[0]) + float(vel[0]) * dt
    z_i = float(prev_pos[2]) + float(vel[2]) * dt + 0.5 * (-650.0) * (dt ** 2)
    return x_i, max(BALL_RADIUS, z_i)


def on_target_factor(arena: RocketSimArena, target_goal_y: float) -> float:
    """
    Placement quality of the ball's current trajectory against one goal mouth, in [0, 1].
    1.0 is cleanly inside the opening; it falls off smoothly to 0 two ball radii outside, with
    post and crossbar treated identically rather than through separate branches.
    """
    proj = project_ball_to_endline(arena, target_goal_y)
    if proj is None:
        return 0.0
    x_i, z_i = proj
    miss = max(
        0.0,
        abs(x_i) - EFFECTIVE_GOAL_HALF_WIDTH,
        z_i - EFFECTIVE_GOAL_HEIGHT,
    )
    return 1.0 - _clip(miss / ON_TARGET_FALLOFF, 0.0, 1.0)


class BallToGoalVelocityReward(BaseReward):
    """
    Symmetric potential-based progression with goal-opening targeting.

    Rewards ball velocity projected toward the opponent's goal, and charges the same rate for
    losing that ground, so the term telescopes: accumulated reward tracks where the ball ended
    up rather than the path it took there.

    On-target placement is a BONUS on that rate, in [1.0, 1.6] -- never a suppressor. The old
    multiplier spanned [0.0, 1.6] and returned exactly 0.0 for any shot projecting above the
    crossbar, which zeroed the whole progression term rather than a bonus: a ball leaving the
    boot at 3000 uu/s scored what a motionless ball scored. Paired with a flat 1.5x penalty on
    the rebound, elevating the ball anywhere in the attacking third was net-negative, which
    foreclosed pops into aerial finishes, air dribbles and ceiling shots. The reverse multiplier
    is gone; own-goal danger is handled by OwnGoalThreatReward, which is localized to the
    defensive third where it belongs.

    The bonus is applied to a rate rather than added per step deliberately. A flat additive
    on-target term integrates over time-in-flight, so a floating shot that takes 60 steps to
    arrive would out-earn a blast that takes 10, biasing the policy toward slow saveable shots.
    Scaling the rate keeps the integral over a flight invariant to shot speed.
    """
    def __init__(self, weight: float = 1.5):
        super().__init__(weight)
        # Team of whoever last touched the ball; None until the first touch of the point.
        # See _refresh_authorship for why this is gated.
        self._last_touch_team: Optional[int] = None
        self._touch_counts: Dict[int, int] = {}

    def reset(self, initial_state: RocketSimArena):
        self._last_touch_team = None
        self._touch_counts = {c.id: int(getattr(c, "ball_touches", 0)) for c in initial_state.cars}

    def _refresh_authorship(self, car: CarState, arena: RocketSimArena) -> None:
        """Track which team last touched the ball.

        ball_touches is cumulative, so only an increase since the previous observation means a
        new touch. This is called once per car per step; the second call of a step sees no
        increase and correctly leaves the stored owner alone.

        The acting car is scanned alongside arena.cars rather than assumed to be among them.
        A car passed in that the arena does not list could otherwise never author a touch, and
        would be permanently locked out of its own progression reward.
        """
        seen = set()
        for c in list(arena.cars) + [car]:
            if id(c) in seen:
                continue
            seen.add(id(c))
            n = int(getattr(c, "ball_touches", 0))
            if n > self._touch_counts.get(c.id, 0):
                self._last_touch_team = int(c.team)
            self._touch_counts[c.id] = n

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        self._refresh_authorship(car, arena)
        if is_goal:
            return 0.0

        # Authorship gate. This term is meant to price what the CAR did to the ball, but the
        # ball moves for reasons the car had no part in: an opponent's clear, a whiff, a bounce
        # off our own backboard. Measured at iteration 7600, steps where the opponent was the
        # last toucher contributed -3.17 per episode and steps before anyone had touched
        # contributed +1.33, against +1.03 for steps the bot actually owned. Three quarters of
        # the term's magnitude was being assigned for events outside the policy's control,
        # which is noise in the advantage estimate and nothing else.
        #
        # The gate is symmetric: when the opponent owns the ball, this term pays nothing AND
        # charges nothing. Suppressing only the positive side would leave a one-way penalty
        # stream the policy cannot switch off, which is strictly worse than either option.
        # Conceding is still priced, at full strength, by the terminal concede reward.
        if self._last_touch_team is None or self._last_touch_team != car.team:
            return 0.0

        target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
        # If ball has already crossed the target endline into the net opening, do not invert vector
        if (car.team == 0 and arena.ball.pos[1] >= target_goal_y) or (car.team == 1 and arena.ball.pos[1] <= target_goal_y):
            return 0.0

        target_x = _clip(arena.ball.pos[0], -GOAL_HALF_WIDTH * 0.8, GOAL_HALF_WIDTH * 0.8)
        target_pos = np.array([target_x, target_goal_y, GOAL_HEIGHT * 0.35], dtype=np.float32)

        ball_to_goal = target_pos - arena.ball.pos
        dist = _norm3(ball_to_goal)
        if dist < 1e-4:
            return 0.0

        unit_to_goal = ball_to_goal / dist
        ball_velocity_toward_goal = float(np.dot(arena.ball.vel, unit_to_goal))
        normalized_progress = ball_velocity_toward_goal / BALL_MAX_SPEED

        # Losing ground costs exactly what gaining it pays. No reverse multiplier: the old 1.5x
        # applied anywhere on the pitch, so a rebound off the OPPONENT's backboard was priced as
        # an own-goal threat. OwnGoalThreatReward handles real own-goal danger where it occurs.
        if ball_velocity_toward_goal <= 0.0:
            return self.weight * normalized_progress

        # Placement bonus on the progression rate, floored at 1.0 so a fast goalward ball is
        # never worth what a motionless one is worth.
        placement = on_target_factor(arena, target_goal_y)
        return self.weight * normalized_progress * (1.0 + ON_TARGET_BONUS * placement)


class OwnGoalThreatReward(BaseReward):
    """
    Localized deterrent against driving the ball at your own net.

    This used to live inside BallToGoalVelocityReward as a flat 1.5x multiplier on reverse
    progression, applied at every point on the pitch. That priced any ball travelling back
    toward your half as an own-goal threat, including a rebound off the OPPONENT's backboard
    roughly 10000 uu from your net, which taught the bot never to use the backboard at all.

    Own-goal danger is a defensive-third concern, so the term activates only there, only when
    the ball is actually travelling at your net, and only in proportion to how much of a shot
    the trajectory really is. A ball rolling into your own corner at speed is not a threat and
    is not charged as one.
    """
    DEFENSIVE_THIRD_DEPTH = 3400.0

    def __init__(self, weight: float = 2.0):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if is_goal:
            return 0.0

        defend_goal_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_to_defend = abs(float(arena.ball.pos[1]) - defend_goal_y)
        if dist_to_defend > self.DEFENSIVE_THIRD_DEPTH:
            return 0.0

        # Ball speed along the axis into our own net.
        vy_defend = -float(arena.ball.vel[1]) if car.team == 0 else float(arena.ball.vel[1])
        if vy_defend <= 0.0:
            return 0.0

        placement = on_target_factor(arena, defend_goal_y)
        if placement <= 0.0:
            return 0.0

        # Closer to the net is harder to recover from.
        urgency = 0.5 + _clip(1.0 - (dist_to_defend / self.DEFENSIVE_THIRD_DEPTH), 0.0, 1.0)
        return -self.weight * (vy_defend / BALL_MAX_SPEED) * placement * urgency


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
    def __init__(self, weight: float = 0.6, boost_pathing_threshold: float = 50.0):
        super().__init__(weight)
        self.boost_pathing_threshold = float(boost_pathing_threshold)
        self._prev_pos: Dict[int, np.ndarray] = {}
        self._prev_target: Dict[int, np.ndarray] = {}
        # Optional single-step override of the previous potential, keyed by car id and consumed
        # on read. Empty during normal rollouts (reset() does not populate it); it exists so a
        # caller can pin an exact distance delta without simulating two consecutive steps.
        self._prev_dist: Dict[int, float] = {}
        self._prev_touches: Dict[int, int] = {}
        self._prev_car_touches: Dict[int, int] = {}
        self._was_in_strike_zone: Dict[int, bool] = {}
        self._prev_timing_err: Dict[int, float] = {}
        self._reread_ticks: Dict[int, int] = {}

    def _calc_dist(self, car_pos: np.ndarray, ball_pos: np.ndarray) -> float:
        # If BOTH car and ball are near pitch floor (car Z < 150, ball Z < 300), evaluate horizontal (X, Y) distance
        # so low ground flips / wavedashes do not incur an artificial vertical distance penalty.
        # Smoothly blend between 2D and 3D distance between car Z=150 and Z=350 to avoid metric cliffs.
        if ball_pos[2] < 300.0:
            d2 = _norm2(ball_pos - car_pos)
            if car_pos[2] <= 150.0:
                return d2
            d3 = _norm3(ball_pos - car_pos)
            if car_pos[2] >= 350.0:
                return d3
            alpha = (float(car_pos[2]) - 150.0) / 200.0
            return float((1.0 - alpha) * d2 + alpha * d3)
        return _norm3(ball_pos - car_pos)

    def _get_target_pos(self, car_pos: np.ndarray, arena: RocketSimArena, is_kickoff: bool, car_vel: Optional[np.ndarray] = None, car_id: int = -1, car_team: int = 0) -> np.ndarray:
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

        ball_speed = _norm3(arena.ball.vel)
        # A near-stationary ball's intercept point is its current position, so skip the solver.
        if ball_speed <= 150.0 or not hasattr(arena, "get_predicted_ball_pos"):
            return self._apply_shadow_retarget(target_pos, car_pos, arena, car_team)

        chaser_vel = car_vel if car_vel is not None else np.zeros(3, dtype=np.float32)
        pred_pos, _ = cached_intercept_point(arena, car_id, car_pos, chaser_vel)
        if pred_pos is None:
            return self._apply_shadow_retarget(target_pos, car_pos, arena, car_team)

        # Blend on proximity alone. The solver already collapses to the live ball position when
        # the intercept is immediate, so the old speed_factor damping only weakened correct
        # lookahead on slower balls. Inside the strike zone we still lock to the true ball so
        # contact geometry stays exact.
        raw_ball_dist = self._calc_dist(car_pos, arena.ball.pos)
        blend = min(1.0, max(0.0, (raw_ball_dist - 250.0) / 350.0))
        blended = (1.0 - blend) * arena.ball.pos + blend * pred_pos

        # Clamp within arena bounds to prevent numerical overshoot
        clamped = np.array([
            _clip(blended[0], -ARENA_EXTENT_X + 100.0, ARENA_EXTENT_X - 100.0),
            _clip(blended[1], -ARENA_EXTENT_Y + 100.0, ARENA_EXTENT_Y - 100.0),
            _clip(blended[2], 93.0, ARENA_HEIGHT_Z - 100.0)
        ], dtype=np.float32)
        return self._apply_shadow_retarget(clamped, car_pos, arena, car_team)

    def _apply_shadow_retarget(self, target_pos: np.ndarray, car_pos: np.ndarray, arena: RocketSimArena, car_team: int) -> np.ndarray:
        """
        Slides the tactical target toward a shadow point when the ball is goalside of the car.

        The distance potential used to aim at the ball unconditionally. The wrong-side flag was
        computed and then spent only on suppressing velocity-matching and pursuit bonuses, never
        on the delta itself, so a bot doing the correct 1v1 thing -- turning and retreating toward
        its back post to delay an attacker instead of diving from a bad angle -- was charged for
        every uu of ground it gave up. A persistent per-step cost against a delayed and uncertain
        benefit is the shape of pressure that produces desperate challenges.

        Retargeting rather than nulling: nulling removes the penalty and the guidance together,
        while a shadow point keeps a gradient pointing where a defender should actually go. Same
        construction JumpBridgeReward already uses, including its clamp keeping the point on the
        playable pitch rather than inside the net structure.

        The offset is blended in over 250 uu of relative depth. Switching it on a boolean would
        teleport the target by 700 uu in one tick, and inside a distance delta that does not even
        surface as a visible spike: the decomposition clips target motion to the ball's own travel
        budget, so the potential would quietly swallow most of the transition instead.
        """
        defend_goal_y = -ARENA_EXTENT_Y if car_team == 0 else ARENA_EXTENT_Y
        dist_car_to_defend = abs(float(car_pos[1]) - defend_goal_y)
        dist_ball_to_defend = abs(float(arena.ball.pos[1]) - defend_goal_y)

        # 0 while the car is goalside of the ball, ramping to 1 once the ball is 300 uu behind it.
        wrong_t = _clip((dist_car_to_defend - dist_ball_to_defend - 50.0) / 250.0, 0.0, 1.0)

        # Being goalside of the ball is not the same as being beaten. Two guards, both ramped:
        #
        # Proximity: with the ball within reach the car is contesting, not retreating, and pulling
        # its target off the ball would starve every strike-zone term that depends on aiming at
        # it. Only a ball that has genuinely got past and away calls for shadowing.
        raw_dist = self._calc_dist(car_pos, arena.ball.pos)
        wrong_t *= _clip((raw_dist - 600.0) / 600.0, 0.0, 1.0)

        # Field position: deep in the attacking half, "farther from our own goal than the ball"
        # is ordinary attacking posture, not a defender who has been beaten. Full weight in our
        # own half, fading out across the attacking half.
        ball_y = float(arena.ball.pos[1])
        ball_depth = -ball_y if car_team == 0 else ball_y  # > 0 when the ball is in our half
        wrong_t *= _clip((ball_depth + 2000.0) / 2000.0, 0.0, 1.0)

        if wrong_t <= 0.0:
            return target_pos

        shadow_offset = -SHADOW_OFFSET_Y if car_team == 0 else SHADOW_OFFSET_Y
        shadow_y = float(arena.ball.pos[1]) + shadow_offset
        if car_team == 0:
            shadow_y = max(-ARENA_EXTENT_Y + 200.0, min(0.0, shadow_y))
        else:
            shadow_y = min(ARENA_EXTENT_Y - 200.0, max(0.0, shadow_y))

        shadow = np.array([
            float(arena.ball.pos[0]) * 0.5,
            shadow_y,
            BALL_RADIUS,
        ], dtype=np.float32)
        return ((1.0 - wrong_t) * np.asarray(target_pos, dtype=np.float32) + wrong_t * shadow).astype(np.float32)

    def reset(self, initial_state: RocketSimArena):
        is_kickoff = bool(
            abs(initial_state.ball.pos[0]) < 50.0 and
            abs(initial_state.ball.pos[1]) < 50.0 and
            initial_state.ball.pos[2] < 120.0 and
            _norm3(initial_state.ball.vel) < 100.0
        )
        self._prev_dist = {}
        self._prev_pos = {car.id: car.pos.copy() for car in initial_state.cars}
        self._prev_target = {
            car.id: np.asarray(self._get_target_pos(car.pos, initial_state, is_kickoff, car.vel, car.id, car.team), dtype=np.float32).copy()
            for car in initial_state.cars
        }
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._prev_car_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._was_in_strike_zone = {car.id: False for car in initial_state.cars}
        self._prev_timing_err = {}
        self._reread_ticks = {car.id: 0 for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if is_goal:
            return 0.0

        # Kickoff sprint multiplier & anti-peel penalty (guarantees full-throttle rush on kickoff)
        is_kickoff = bool(
            abs(arena.ball.pos[0]) < 50.0 and
            abs(arena.ball.pos[1]) < 50.0 and
            arena.ball.pos[2] < 120.0 and
            _norm3(arena.ball.vel) < 100.0
        )

        target_pos = self._get_target_pos(car.pos, arena, is_kickoff, car.vel, car.id, car.team)
        curr_dist = self._calc_dist(car.pos, target_pos)

        # Stationary-potential decomposition.
        #
        # Storing last step's distance and differencing it against this step's makes the shaping
        # non-stationary: the tactical target is a blend toward a predicted intercept, so it moves
        # discontinuously whenever the ball bounces, an opponent touches it, or the blend factor
        # crosses one of its proximity/speed ramps. That injected a free +/- delta unrelated to any
        # action the bot took -- and the opponent-touch guard below only ever clamped the negative
        # side, so an opponent knocking the ball toward the bot paid out.
        #
        # Split the delta into the two halves that produce it and hold the other endpoint fixed for
        # each, so both are measured against a single consistent potential:
        #   car-motion   : how much the car closed on THIS step's target
        #   target-motion: how much the target moved relative to THIS step's car position
        # The car half is always trustworthy. The target half is clamped to the distance the ball
        # could physically have travelled this step, which absorbs prediction-blend jumps while
        # preserving genuine credit/debit for a ball rolling toward or away from the bot.
        prev_car_pos = self._prev_pos.get(car.id)
        prev_target = self._prev_target.get(car.id)
        self._prev_pos[car.id] = car.pos.copy()
        self._prev_target[car.id] = np.asarray(target_pos, dtype=np.float32).copy()

        if car.id in self._prev_dist:
            prev_dist = float(self._prev_dist.pop(car.id))
        elif prev_car_pos is None or prev_target is None:
            prev_dist = curr_dist
        else:
            car_motion_delta = self._calc_dist(prev_car_pos, target_pos) - curr_dist
            target_motion_delta = self._calc_dist(car.pos, prev_target) - curr_dist

            step_dt = float(getattr(arena, "last_step_dt", 8.0 / 120.0))
            # Generous budget: ball speed plus a floor for a near-stationary ball that is
            # nonetheless about to be struck. Anything beyond this is a target discontinuity.
            ball_travel_budget = _norm3(arena.ball.vel) * step_dt + 60.0
            target_motion_delta = _clip(target_motion_delta, -ball_travel_budget, ball_travel_budget)

            # Both halves are kept deliberately. Crediting only the car's motion against the
            # CURRENT target was tried and is worse: a car following a fleeing ball at a fixed
            # gap would then be paid every step for ground it never gained. With both terms a
            # car and ball moving together cancel to exactly zero, which is the correct price
            # for a carry.
            prev_dist = curr_dist + car_motion_delta + target_motion_delta

        prev_t = self._prev_touches.get(car.id, car.ball_touches)
        self._prev_touches[car.id] = car.ball_touches

        raw_ball_dist = self._calc_dist(car.pos, arena.ball.pos)

        # Unit alignment vector to tactical target (properly normalized in 3D)
        car_to_ball = target_pos - car.pos
        dist_3d = _norm3(car_to_ball)
        unit_to_ball = car_to_ball / max(1e-4, dist_3d)
        fwd_alignment = compute_effective_alignment(car, unit_to_ball)

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

        # Precision Wall Geometry Detection:
        # Ball is physically on or hugging sidewall or backboard (excluding active open goal net)
        is_in_goal_mouth = bool(abs(arena.ball.pos[0]) < 900.0 and ball_z < 650.0)
        is_ball_on_sidewall = bool(abs(arena.ball.pos[0]) > 3650.0 and ball_z > 140.0)
        is_ball_on_backboard = bool(abs(arena.ball.pos[1]) > 4700.0 and ball_z > 140.0 and not is_in_goal_mouth)
        # Ball must be genuinely climbing the curved ramp, not merely rolling on the floor near it.
        # A ball at rest sits at z = BALL_RADIUS = 91.25, so the previous z > 80.0 gate was
        # satisfied by every grounded ball within 750 uu of a sidewall / 770 uu of a backboard.
        # That forced is_elevated_aerial false across a large slice of the pitch, enabled the
        # wall-climb multipliers for ordinary corner play, and disabled the grounded approach
        # pacing envelope and dribble boost penalty there. 160.0 matches the 140.0 used by the
        # flat-wall predicates while clearing a resting ball's radius with margin.
        is_ball_on_curve = bool((abs(arena.ball.pos[0]) > 3350.0 or abs(arena.ball.pos[1]) > 4350.0) and not is_in_goal_mouth and ball_z > 160.0)
        is_ball_on_wall = bool(is_ball_on_sidewall or is_ball_on_backboard or is_ball_on_curve)
        # Elevated aerial is strictly an open-air floating ball infield away from arena walls
        is_elevated_aerial = bool(ball_z > 350.0 and not is_ball_on_wall)

        is_on_ceiling = bool((car.pos[2] > 1750.0 and car.on_ground) or car.pos[2] > 1900.0)
        up_tilt = abs(float(car.rot_mat[2, 2])) if hasattr(car, "rot_mat") and car.rot_mat is not None else 1.0
        is_on_wall_curve = bool((abs(car.pos[0]) > 3300.0 or abs(car.pos[1]) > 4300.0) and car.on_ground and (car.pos[2] > 55.0 or up_tilt < 0.92))
        is_on_wall = bool(((abs(car.pos[0]) > 3450.0 or abs(car.pos[1]) > 4450.0) and car.pos[2] > 200.0 and car.on_ground) or is_on_wall_curve)

        horiz_ball_dist = _norm2(arena.ball.pos - car.pos)
        eff_dist = min(curr_dist, horiz_ball_dist) if (car.on_ground and ball_z < 650.0) else curr_dist

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

        opp_touched = any(
            c.ball_touches > self._prev_car_touches.get(c.id, c.ball_touches)
            for c in arena.cars if c.team != car.team
        )
        bot_touched = bool(car.ball_touches > self._prev_car_touches.get(car.id, car.ball_touches))
        self._prev_car_touches = {c.id: c.ball_touches for c in arena.cars}

        # An opponent strike displaces the ball by an amount the bot did not cause, in whichever
        # direction the strike happened to send it, so the delta it produces is not credit or
        # blame either way. Clamp both sides to the same budget.
        #
        # This clamped only the negative side before. The class's own potential decomposition
        # above was written to fix exactly this asymmetry for target motion, and this reintroduced
        # it on a different input: an opponent knocking the ball toward the bot paid the full
        # positive delta for distance the bot did nothing to close.
        #
        # `not bot_touched` matters in 1v1, where a large share of contact is a simultaneous 50/50
        # with both cars registering a touch within a frame or two. That is not an external
        # displacement to be clamped away -- it is an outcome the bot half-caused, and it should
        # be paid or charged on the resulting trajectory.
        if opp_touched and not bot_touched:
            raw_delta_dist = _clip(raw_delta_dist, -0.08, 0.08)

        # 1. Anti-Overshoot Penalty & Strike Zone Tracking
        overshoot_penalty = 0.0
        in_strike = (raw_ball_dist < 400.0) or (car.on_ground and horiz_ball_dist < 380.0 and ball_z < 650.0)
        was_strike = self._was_in_strike_zone.get(car.id, False)
        self._was_in_strike_zone[car.id] = in_strike

        car_fwd_spd = float(np.dot(car.vel, car.get_forward_vector())) if not car.on_ground else float(np.dot(car.vel[:2], car.get_forward_vector()[:2]))
        is_active_flip = bool(car.just_dodged or getattr(car, "is_dodging", False) or (not car.on_ground and not car.has_flip))
        car_vel_toward_ball = float(np.dot(car.vel, unit_to_ball))
        if was_strike and not in_strike and car.ball_touches == prev_t and not opp_touched and car_fwd_spd > 150.0:
            if not is_active_flip and fwd_alignment < -0.15:
                # Ground / non-flip overshoot where car drove away
                car_spd = _norm3(car.vel)
                overshoot_penalty = -0.40 if car_spd < 1800.0 else -0.60
            elif is_active_flip and car_vel_toward_ball < -100.0 and raw_delta_dist < -0.05:
                # Body momentum actually sailing away after flip without touching
                overshoot_penalty = -0.30

        # 2. Distance Delta with Strike Zone Pacing
        # Downfield (> 450 uu): 100% distance closure rewarded
        # Inside strike zone (< 450 uu): Paces approach so car doesn't blindly barrel past ball
        strike_pacing = min(1.0, max(0.20, (eff_dist - 150.0) / 300.0))
        delta_dist = raw_delta_dist * strike_pacing

        # Kinematic TTI-Detour Pathing Gate with Continuous Boost Urgency Gradient:
        # The boost-pathing penalty relief that used to live here is gone.
        #
        # It cut the charge for moving AWAY from the ball by up to 85% whenever the car was
        # low on boost, which broke the one property this term depends on. Approaching paid
        # full and retreating charged a fraction, so oscillating toward and away from the ball
        # closed net positive and could be repeated all episode. Measured on the policy at
        # iteration 2320 the term paid 0.0628/step and charged 0.0154/step, a 4:1 ratio, worth
        # about 22 per episode against a goal's 30 -- while ball_to_goal, which telescopes
        # properly, netted 2.3. The relief was active nearly always, because the policy had
        # learned to stop boosting (2.2% of steps) and so was permanently 'low on boost'.
        #
        # The intent was to avoid punishing a detour through a boost pad. That belongs in the
        # boost potential, which now prices pad routing symmetrically and telescopes, so there
        # is nothing left for this to forgive. Distance shaping is now strictly symmetric:
        # ground gained and ground lost cost exactly the same.

        # Wall-Crawling & Wall Pursuit Dynamics:
        # 1. When ball has bounced away from the wall into the infield (lateral separation), heavily dampen wall driving.
        # 2. When car is climbing away higher than the ball (car_z > ball_z + 200 and car_vz > -50), dampen wall driving.
        # 3. When tracking a ball on the wall, award active distance progression bonus (1.25x).
        # Both factors ramp rather than switch. A hard boolean multiplying a potential derivative
        # destroys telescoping and leaves a seam at the boundary: one uu of ball travel across
        # abs(ball.x) == 2800 used to swing this term by more than eight times, so the accumulated
        # reward depended on which side of an invisible line the ball sat rather than on where the
        # car actually went. Same technique the class already uses for the 2D-to-3D distance blend
        # between car heights of 150 and 350.
        is_ball_infield = False
        is_car_above_ball = False
        if is_on_wall:
            if abs(car.pos[0]) > 3450.0:
                # On a side wall: how far infield the ball has bounced, ramped over 2600..3000.
                infield_t = _clip((3000.0 - abs(float(arena.ball.pos[0]))) / 400.0, 0.0, 1.0)
            else:
                # On a backboard: same ramp against the y wall, 3600..4000.
                infield_t = _clip((4000.0 - abs(float(arena.ball.pos[1]))) / 400.0, 0.0, 1.0)

            # Climbing away above the ball, ramped over 200 uu of separation and 50 uu/s of climb.
            above_t = _clip((float(car.pos[2]) - ball_z - 200.0) / 200.0, 0.0, 1.0)
            above_t *= _clip((float(car.vel[2]) + 50.0) / 100.0, 0.0, 1.0)

            damp_t = max(infield_t, above_t)
            is_ball_infield = bool(infield_t > 0.5)
            is_car_above_ball = bool(above_t > 0.5)

            if damp_t > 0.0:
                delta_dist *= 1.0 + damp_t * (0.15 - 1.0)
            elif is_ball_on_wall and fwd_alignment > 0.20:
                # Ramp the pursuit bonus in over the alignment threshold too.
                track_t = _clip((fwd_alignment - 0.20) / 0.20, 0.0, 1.0)
                delta_dist *= 1.0 + 0.25 * track_t

        # If car is moving in reverse, executing a half-flip, or executing an active dodge/speedflip towards target,
        # evaluate horizontal travel velocity alignment rather than car nose forward vector:
        car_fwd_vel = float(np.dot(car.vel[:2], car.get_forward_vector()[:2]))
        car_horiz_speed = _norm2(car.vel)
        travel_unit_h = car.vel[:2] / max(1e-4, car_horiz_speed)
        travel_align_to_ball = float(np.dot(travel_unit_h, unit_to_ball[:2]))

        # Airborne flight: front flips, speedflips, and airborne half-flips rocket toward ball with nose off-axis:
        is_in_flip_flight = bool(not car.on_ground and not car.has_flip and car_horiz_speed > 300.0 and travel_align_to_ball > 0.35)
        is_dodging_toward_ball = bool((car.just_dodged or is_in_flip_flight) and car_horiz_speed > 250.0 and travel_align_to_ball > 0.35)
        # Genuine airborne half-flip in mid-air (car in reverse flight toward target):
        is_airborne_half_flip = bool(not car.on_ground and car_fwd_vel < -150.0 and travel_align_to_ball > 0.35)
        # Ground forward traversal toward ball:
        is_forward_traveling = bool(car_fwd_vel > 80.0 and travel_align_to_ball > 0.35 and car_horiz_speed > 100.0)
        is_traveling_toward_ball = bool(is_forward_traveling or not car.on_ground) and (travel_align_to_ball > 0.35 and car_horiz_speed > 100.0)

        if fwd_alignment < 0.0 and delta_dist > 0.0 and not is_dodging_toward_ball and not is_airborne_half_flip:
            # On wheels on the turf, driving in reverse toward a trailing ball receives damped delta_dist (0.2x)
            # to strongly incentivize executing an angular turnaround (powerslide cut or half-flip)
            delta_dist = delta_dist * max(0.0, fwd_alignment + 1.0) * 0.2

        fwd_vec = car.get_forward_vector()
        right_vec = car.get_right_vector()
        up_vec = car.get_up_vector()
        local_x = float(np.dot(car_to_ball[:2], unit_horiz(fwd_vec)))
        local_y = float(np.dot(car_to_ball[:2], unit_horiz(right_vec)))
        local_z = float(np.dot(arena.ball.pos - car.pos, up_vec))
        is_roof_carry = bool(
            car.on_ground and
            curr_dist < 210.0 and
            110.0 <= local_z <= 170.0 and
            -30.0 <= local_x <= 65.0 and
            abs(local_y) < 50.0
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

        is_opponent_challenging = False
        if threats:
            most_urgent = threats[0]
            is_opponent_challenging = bool(
                most_urgent.dist <= 650.0 or
                (most_urgent.closing_speed > 100.0 and most_urgent.arrival_time < 0.85) or
                (abs(delta_t) < 0.30 and self_tti < 0.60)
            )

        in_strike_zone = (raw_ball_dist < 500.0) or (car.on_ground and horiz_ball_dist < 500.0 and ball_z < 650.0)
        if in_strike_zone:
            car_speed = _norm3(car.vel)
            ball_speed = _norm3(arena.ball.vel)
            car_speed_2d = _norm2(car.vel)
            ball_speed_2d = _norm2(arena.ball.vel)
            rel_speed = _norm3(car.vel - arena.ball.vel)
            rel_speed_2d = _norm2(car.vel - arena.ball.vel)

            effective_car_speed = car_speed_2d if (car.on_ground and ball_z < 650.0) else car_speed
            effective_ball_speed = ball_speed_2d if (car.on_ground and ball_z < 650.0) else ball_speed
            effective_rel_speed = rel_speed_2d if (car.on_ground and ball_z < 650.0) else rel_speed

            # 3a. Forward Strike-Zone Velocity Matching & Arrival Pacing
            if fwd_alignment > 0.2 and not (is_wrong_side and car_vy_defend > 100.0):
                if not is_ground_pushing and not is_roof_carry and effective_ball_speed > 250.0 and effective_car_speed > 200.0:
                    # vel_matching_bonus removed with the rest of the additive income.
                    # It paid up to 0.30/step for matching the ball's speed inside the strike
                    # zone, which is the shape that rewards shadowing a ball instead of
                    # hitting it. The pacing and overshoot penalties below still discourage
                    # barrelling through it, and they only subtract.
                    pass

                # Kinetic Arrival Velocity Pacing Envelope:
                # When closing toward the ball on the ground, evaluate required approach pacing:
                # Pure outcome-driven penalty avoidance: overspeeding incurs pacing_penalty,
                # decelerating to desired speed brings penalty to 0.0. No positive per-tick hovering bounties.
                # Strictly for grounded open-field dribble pacing (exempt on walls where climbing momentum is required).
                if car.on_ground and not is_roof_carry and not is_ground_pushing and not is_on_wall and not is_ball_on_wall and (ball_z < 250.0 or (ball_z < 500.0 and arena.ball.vel[2] < -100.0)):
                    safe_speed_margin = max(150.0, (min(curr_dist, 500.0) / 500.0) * 650.0)
                    desired_speed = effective_ball_speed + safe_speed_margin

                    # Pacing penalty is mutually exclusive with overshoot penalty (approach vs aftermath)
                    if overshoot_penalty == 0.0:
                        if self_tti < 0.40 and effective_car_speed > desired_speed:
                            excess = (effective_car_speed - desired_speed) / 800.0
                            pacing_penalty = -0.35 * min(1.0, max(0.0, excess))

            # 3b. Trajectory & Time-To-Intercept (TTI) Defending Net Threat Evaluation:
            # Differentiates safe defensive plays (corner wraps, backboard clears, recoverable touches)
            # and dangerous unrecoverable own-goal threats. Evaluates regardless of car facing angle!
            if is_wrong_side or car_vy_defend > 50.0:
                ball_vy_defend = -arena.ball.vel[1] if car.team == 0 else arena.ball.vel[1]
                dist_ball_to_defend = abs(arena.ball.pos[1] - defend_goal_y)

                if ball_vy_defend > 150.0:
                    dt_defend = dist_ball_to_defend / ball_vy_defend
                    # Threat horizon capped at 6.5s to comfortably cover full-pitch shots while avoiding singularities
                    if dt_defend <= 6.5:
                        x_defend_impact = arena.ball.pos[0] + arena.ball.vel[0] * dt_defend
                        z_defend_impact = max(BALL_RADIUS, arena.ball.pos[2] + arena.ball.vel[2] * dt_defend + 0.5 * (-650.0) * (dt_defend ** 2))
                        is_on_defend_net = bool(abs(x_defend_impact) <= EFFECTIVE_GOAL_HALF_WIDTH and z_defend_impact <= EFFECTIVE_GOAL_HEIGHT + 50.0)

                        if is_on_defend_net:
                            # Ball is heading on-target into defending net opening!
                            impact_pos = np.array([x_defend_impact, defend_goal_y, min(GOAL_HEIGHT * 0.5, z_defend_impact)], dtype=np.float32)
                            car_tti_defend, _, _ = compute_trajectory_arrival_time(car.pos, car.vel, impact_pos)

                            is_unrecoverable = bool(car_tti_defend >= dt_defend - 0.15)
                            is_active_own_net_push = bool(car_vy_defend > 80.0 and (raw_ball_dist < 250.0 or horiz_ball_dist < 250.0))

                            if is_unrecoverable or is_active_own_net_push:
                                threat_severity = max(0.4, min(1.0, ball_vy_defend / 1200.0))
                                proximity_scale = max(0.5, 1.0 - (dist_ball_to_defend / 3500.0))
                                wrong_side_push_penalty = -0.50 * threat_severity * proximity_scale
                elif is_wrong_side and car_vy_defend > 150.0 and (raw_ball_dist < 200.0 or horiz_ball_dist < 200.0):
                    # Proximity goal-mouth push toward defending net inside red zone
                    if dist_ball_to_defend < 1800.0:
                        wrong_side_push_penalty = -0.35 * (car_vy_defend / 1500.0)

            # Dribble Proximity Pacing & Anti-Overshoot:
            is_close_approach = bool(raw_ball_dist < 350.0 or (horiz_ball_dist < 350.0 and ball_z < 650.0))
            if is_close_approach and car.on_ground and not is_roof_carry and not is_on_wall and not is_ball_on_wall and ball_z < 250.0:
                if effective_car_speed > effective_ball_speed + 150.0 and float(action[6]) > 0.0:
                    dribble_boost_penalty = -0.30 * float(action[6])

            # Hard Anti-Stacking Floor:
            # Clamps combined strike-zone approach penalties to a maximum floor of -0.60
            total_approach_penalties = overshoot_penalty + pacing_penalty + dribble_boost_penalty + wrong_side_push_penalty
            if total_approach_penalties < -0.60:
                scale = -0.60 / total_approach_penalties
                overshoot_penalty *= scale
                pacing_penalty *= scale
                dribble_boost_penalty *= scale
                wrong_side_push_penalty *= scale

        # Sections 4, 5, the strike-timing term and the boost-ahead term used to live here.
        # All four were removed: every one of them paid positive-only, per-step income on top
        # of a distance potential that already measured the same thing symmetrically.
        #
        #   vel_toward_ball    up to 0.40/step for moving at the ball, with nothing charged
        #                      for moving away. This double-counted delta_dist: one symmetric
        #                      copy that telescopes, one asymmetric copy that does not.
        #   turnaround_reward  cut, pacing and rotation bonuses for executing a pivot.
        #   roof_carry_reward  carry, velcro and sync bonuses for balancing the ball on the
        #                      roof -- income for holding possession rather than using it,
        #                      which prices dribbling above flicking or shooting.
        #   timing_reward      strike-window arrival shaping.
        #   boost_ahead_reward a second, non-telescoping copy of the boost-pathing incentive
        #                      that BoostReward's transit potential already provides, and
        #                      provides symmetrically.
        #
        # Measured on the policy at iteration 2320, this class paid about 18.5 per episode
        # while a pure distance potential at this weight is bounded near 1.25, because a car
        # cannot close more ground than the pitch is long. Roughly 93% of the term was these
        # streams rather than distance.
        #
        # What remains is the distance potential plus the one-sided penalties below it, which
        # cannot be farmed because they only ever subtract. The policy was seeded from human
        # replays, so it already knows how to cut, dribble and strike; how it uses the ball is
        # now decided by touch, ball_to_goal and goal rather than by hand-written per-step pay.
        #
        # The kickoff branch keeps its own vel_toward_ball term and its own early return. It
        # is bounded to the opening seconds and kickoffs are the one phase already working.

        # The distance potential, plus penalties that can only subtract. Nothing here pays
        # positive income per step, so the term telescopes: over any path that returns to its
        # starting distance the delta sums to zero, and what survives is ground actually
        # gained. That bounds the whole class near 1.25 per episode at weight 0.5.
        total_reward = self.weight * (
            delta_dist + pacing_penalty + dribble_boost_penalty +
            overshoot_penalty + ceiling_penalty + wrong_side_push_penalty
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
        self._prev_ball_vel: Dict[int, np.ndarray] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._prev_ball_vel = {car.id: initial_state.ball.vel.copy() for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev_b_vel = self._prev_ball_vel.get(car.id, arena.ball.vel.copy())
        self._prev_ball_vel[car.id] = arena.ball.vel.copy()

        prev = self._prev_touches.get(car.id, car.ball_touches)
        curr = car.ball_touches
        self._prev_touches[car.id] = curr

        if curr > prev:
            # Height scaling: Ground touch (Z=93) = 1.0x, High Aerial touch (Z=1500) = 2.5x
            ball_z = float(arena.ball.pos[2])
            height_multiplier = 1.0 + 1.5 * max(0.0, min(1.0, (ball_z - 150.0) / 1850.0))

            # Aerial airborne touch bonus: rewards leaving the turf to intercept bouncing or aerial balls cleanly
            airborne_bonus = 1.2 * min(1.0, max(0.4, (ball_z - 120.0) / 300.0)) if (not car.on_ground and ball_z > 140.0) else 0.0

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
            ball_speed = _norm3(arena.ball.vel)
            # Target goal opening rather than pure +Y direction
            target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
            defend_goal_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y

            target_x = _clip(arena.ball.pos[0], -GOAL_HALF_WIDTH * 0.75, GOAL_HALF_WIDTH * 0.75)
            target_pos = np.array([target_x, target_goal_y, GOAL_HEIGHT * 0.35], dtype=np.float32)

            ball_to_net = target_pos - arena.ball.pos
            unit_to_goal = ball_to_net / max(1e-4, _norm3(ball_to_net))

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
                # On-Target Trajectory Bonus: Check if touch velocity produces a direct shot into
                # the net. This used to carry its own copy of the endline projection, with the
                # same gravity constant and goal geometry as BallToGoalVelocityReward's copy and
                # a different set of thresholds on the result. Both now read the shared helper,
                # which also means both inherit its arena-aware trajectory rather than a bare
                # parabola that ignores the ceiling and side walls.
                vy_forward = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
                if vy_forward > 80.0:
                    placement = on_target_factor(arena, target_goal_y)
                    if placement >= 1.0:
                        # Direct shot on target into the net opening!
                        goal_alignment = max(goal_alignment, 0.7) + 0.35
                    elif placement > 0.0:
                        # Near-post / crossbar grazing shot: moderate alignment without clean goal bonus
                        goal_alignment = max(goal_alignment, 0.45)

                direction_multiplier = 1.0 + (min(1.0, goal_alignment) * 1.5)  # 1.0x -> 2.5x

                # Dual-Path Context Evaluator:
                rel_speed = _norm3(car.vel - arena.ball.vel)
                is_gentle_ground_push = bool(car.on_ground and ball_z < 130.0 and rel_speed < 150.0)

                # Dedicated Wall Strike Bonus:
                # Rewards solid wall contact (pops, pinches, passes, and strikes along/off the wall)
                is_wall_touch = bool(car.pos[2] > 200.0 and (abs(car.pos[0]) > 3400.0 or abs(car.pos[1]) > 4400.0))
                wall_strike_bonus = (0.60 * min(1.5, max(0.4, ball_speed / 1000.0))) if is_wall_touch else 0.0

                # Directional Kinetic Impulse Transfer:
                #
                # Single strike estimator. This used to be three: a power bonus on the ball's
                # absolute speed, this impulse bonus on the same collision projected toward the
                # net, and the direction multiplier they were both multiplied by. Power and
                # direction multiply out to roughly speed-times-alignment, which is the quantity
                # impulse measures directly, so three knobs moved one number and tuning any of
                # them silently rescaled the others.
                #
                # Impulse is the right one to keep. Absolute ball speed pays for a ball that was
                # already travelling fast: tapping a rocket that is on its way to the net scored
                # nearly as well as striking it, though the bot did almost nothing. Velocity
                # ADDED along the goal direction is what the car is actually responsible for.
                #
                # Critically this does not cross the possession fork. A catch matches ball speed,
                # so its impulse is near zero by construction, and an impulse-only score would
                # pay hugely for booming the ball at a wall and near nothing for a soft catch or
                # a roof carry -- boom-ball, conceding possession on every touch. Possession is
                # scored by base_touch through soft_catch_bonus and the gentle-push path, both
                # untouched here, and strike is gated off entirely for a gentle ground push.
                delta_v_vec = arena.ball.vel - prev_b_vel
                delta_v_goal = float(np.dot(delta_v_vec, unit_to_goal))
                strike_impulse = 0.0
                if goal_alignment > 0.15 and not is_gentle_ground_push:
                    strike_impulse = _clip(delta_v_goal / STRIKE_IMPULSE_REF, 0.0, STRIKE_IMPULSE_CAP)

                if is_defensive_clear:
                    clear_bonus = 0.5 * clear_urgency * max(0.0, (clear_quality - 0.5) / 0.5)
                else:
                    clear_bonus = 0.0

                # Soft Possession Catch Bonus:
                # When uncontested (opponent threat arrival > 1.2s or no threats) on a grounded/low ball,
                # reward cushioning the ball (rel_speed < 350.0 uu/s) into an immediate dribble/carry
                # rather than blasting it away uncontrollably.
                soft_catch_bonus = 0.0
                if car.on_ground and ball_z < 200.0 and not is_defensive_clear and not is_wall_touch:
                    threats = compute_opponent_threats(car, arena)
                    opp_arr = threats[0].arrival_time if threats else 999.0
                    self_arr, _, _ = compute_trajectory_arrival_time(car.pos, car.vel, arena.ball.pos, arena.ball.vel)
                    delta_t = opp_arr - self_arr
                    if delta_t > 0.60 and rel_speed < 350.0:
                        soft_catch_bonus = 0.80 * max(0.0, 1.0 - (rel_speed / 350.0))

                base_touch = 0.25 if is_gentle_ground_push else (0.8 + clear_bonus + soft_catch_bonus + wall_strike_bonus)

                # Physical Lateral Tire Slip Dampening on Ground Contact:
                # When striking a grounded ball, if the car is sliding laterally across the turf
                # (high lateral slip), the strike lacks traction and glances weakly.
                # Solid strikes with wheels gripping the pitch transfer full impulse into the ball.
                if car.on_ground and ball_z < 180.0:
                    contact_lateral_slip = abs(float(np.dot(car.vel[:2], car.get_right_vector()[:2])))
                    if contact_lateral_slip > 80.0:
                        slip_factor = max(0.2, 1.0 - (contact_lateral_slip - 80.0) / 300.0)
                        base_touch *= slip_factor
                        strike_impulse *= slip_factor
                        kickoff_bounty *= slip_factor

                return self.weight * ((base_touch + strike_impulse) * direction_multiplier * height_multiplier + airborne_bonus + kickoff_bounty)

            # --- CASE 2: Ball hit directed backward toward defending half / goal ---
            else:
                if is_defensive_clear:
                    # Lateral pinch / side clear out of defensive third
                    unit_clear_y = 1.0 if car.team == 0 else -1.0
                    delta_v_vec = arena.ball.vel - prev_b_vel
                    delta_v_clear = float(delta_v_vec[1] * unit_clear_y)
                    clear_impulse = min(0.50, max(0.0, delta_v_clear / 1500.0))
                    wall_clear_bonus = 0.40 if (car.pos[2] > 200.0 and (abs(car.pos[0]) > 3400.0 or abs(car.pos[1]) > 4400.0)) else 0.0
                    if car.on_ground and ball_z < 180.0:
                        contact_lateral_slip = abs(float(np.dot(car.vel[:2], car.get_right_vector()[:2])))
                        clear_base = (0.8 + wall_clear_bonus) * clear_quality * max(0.4, 1.0 - (contact_lateral_slip / 500.0))
                    else:
                        clear_base = (0.8 + wall_clear_bonus) * clear_quality
                    return self.weight * ((clear_base + clear_impulse) * height_multiplier + airborne_bonus)
                else:
                    # Trajectory & Time-To-Intercept (TTI) Defending Net Threat Evaluation:
                    # Differentiates safe back-passes, corner rolls, and wall wraps from unrecoverable own-goal shots.
                    ball_vy_defend = -arena.ball.vel[1] if car.team == 0 else arena.ball.vel[1]
                    dist_ball_to_defend = abs(arena.ball.pos[1] - defend_goal_y)
                    is_threatening_own_net = False
                    is_unrecoverable_own_shot = False

                    if ball_vy_defend > 150.0:
                        dt_defend = dist_ball_to_defend / ball_vy_defend
                        if dt_defend <= 6.5:
                            x_defend_impact = arena.ball.pos[0] + arena.ball.vel[0] * dt_defend
                            z_defend_impact = max(BALL_RADIUS, arena.ball.pos[2] + arena.ball.vel[2] * dt_defend + 0.5 * (-650.0) * (dt_defend ** 2))
                            if abs(x_defend_impact) <= EFFECTIVE_GOAL_HALF_WIDTH and z_defend_impact <= EFFECTIVE_GOAL_HEIGHT + 50.0:
                                is_threatening_own_net = True
                                impact_pos = np.array([x_defend_impact, defend_goal_y, min(GOAL_HEIGHT * 0.5, z_defend_impact)], dtype=np.float32)
                                car_tti_defend, _, _ = compute_trajectory_arrival_time(car.pos, car.vel, impact_pos)
                                is_unrecoverable_own_shot = bool(car_tti_defend >= dt_defend - 0.15)

                    if is_unrecoverable_own_shot:
                        # Direct unrecoverable touch into own goal: strictly penalized
                        penalty_scale = max(0.5, abs(goal_alignment))
                        return -self.weight * 1.5 * penalty_scale * height_multiplier
                    elif is_threatening_own_net:
                        # Ball directed on-target into own net, but car may still reach it: mild discouragement
                        return -self.weight * 0.3 * abs(goal_alignment)
                    else:
                        # Safe touch toward own half (corner reset, backboard wrap, soft possession catch):
                        # Completely unpenalized! Allow strategic reset and possession play.
                        rel_speed = _norm3(car.vel - arena.ball.vel)
                        return self.weight * 0.10 if (car.on_ground and rel_speed < 300.0) else 0.0

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
        self._prev_has_double_jumped: Dict[int, bool] = {}
        self._prev_touches: Dict[int, int] = {}
        self._prev_vel: Dict[int, np.ndarray] = {}
        self._prev_pos_z: Dict[int, float] = {}
        self._challenge_jump_active: Dict[int, bool] = {}
        self._halfflip_in_progress: Dict[int, bool] = {}
        self._halfflip_cancel_executed: Dict[int, bool] = {}
        self._halfflip_roll_executed: Dict[int, bool] = {}
        self._flick_window_active: Dict[int, bool] = {}
        self._dodge_strike_ticks: Dict[int, int] = {}
        self._prev_ball_vel: Dict[int, np.ndarray] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_on_ground = {car.id: car.on_ground for car in initial_state.cars}
        self._prev_has_flip = {car.id: car.has_flip for car in initial_state.cars}
        self._prev_has_double_jumped = {car.id: getattr(car, "has_double_jumped", False) for car in initial_state.cars}
        self._prev_touches = {car.id: car.ball_touches for car in initial_state.cars}
        self._prev_vel = {car.id: car.vel.copy() for car in initial_state.cars}
        self._prev_pos_z = {car.id: float(car.pos[2]) for car in initial_state.cars}
        self._challenge_jump_active = {car.id: False for car in initial_state.cars}
        self._halfflip_in_progress = {car.id: False for car in initial_state.cars}
        self._halfflip_cancel_executed = {car.id: False for car in initial_state.cars}
        self._halfflip_roll_executed = {car.id: False for car in initial_state.cars}
        self._flick_window_active = {car.id: False for car in initial_state.cars}
        self._dodge_strike_ticks = {car.id: 0 for car in initial_state.cars}
        self._prev_ball_vel = {car.id: initial_state.ball.vel.copy() for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        # Strict zero gate. Every payout below is `self.weight * multiplier`, so a zero weight
        # already yields zero, but this term is the most expensive reward in the set: it solves
        # opponent arrival times and runs several flip classifiers on every step for every car.
        # Returning early keeps that off the hot path once the weight is annealed or zeroed,
        # rather than computing a full analysis and multiplying the answer by nothing.
        if self.weight <= 1e-6:
            self._prev_on_ground[car.id] = car.on_ground
            self._prev_has_flip[car.id] = car.has_flip
            self._prev_has_double_jumped[car.id] = getattr(car, "has_double_jumped", False)
            self._prev_touches[car.id] = car.ball_touches
            self._prev_vel[car.id] = car.vel.copy()
            self._prev_pos_z[car.id] = float(car.pos[2])
            self._prev_ball_vel[car.id] = arena.ball.vel.copy()
            return 0.0

        prev_ground = self._prev_on_ground.get(car.id, car.on_ground)
        self._prev_on_ground[car.id] = car.on_ground

        prev_flip = self._prev_has_flip.get(car.id, car.has_flip)
        self._prev_has_flip[car.id] = car.has_flip

        curr_double_jump = getattr(car, "has_double_jumped", False)
        prev_double_jump = self._prev_has_double_jumped.get(car.id, curr_double_jump)
        self._prev_has_double_jumped[car.id] = curr_double_jump

        prev_touch = self._prev_touches.get(car.id, car.ball_touches)
        self._prev_touches[car.id] = car.ball_touches

        prev_vel = self._prev_vel.get(car.id, car.vel)
        self._prev_vel[car.id] = car.vel.copy()

        prev_pos_z = self._prev_pos_z.get(car.id, float(car.pos[2]))
        self._prev_pos_z[car.id] = float(car.pos[2])

        car_to_ball = arena.ball.pos - car.pos
        dist = _norm3(car_to_ball)
        unit_to_ball = car_to_ball / max(1e-4, dist)
        forward_alignment = compute_effective_alignment(car, unit_to_ball)
        takeoff_closing_vel = float(np.dot(car.vel, unit_to_ball))
        ball_z = float(arena.ball.pos[2])

        # Defensive & tactical context
        defend_goal_y = -ARENA_EXTENT_Y if car.team == 0 else ARENA_EXTENT_Y
        dist_car_to_defend = abs(car.pos[1] - defend_goal_y)
        dist_ball_to_defend = abs(arena.ball.pos[1] - defend_goal_y)
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

        if is_wrong_side:
            shadow_offset = -700.0 if car.team == 0 else 700.0
            shadow_target_y = float(arena.ball.pos[1] + shadow_offset)
            if car.team == 0:
                shadow_target_y = max(-ARENA_EXTENT_Y + 200.0, min(0.0, shadow_target_y))
            else:
                shadow_target_y = min(ARENA_EXTENT_Y - 200.0, max(0.0, shadow_target_y))

            retreat_vec = np.array([float(arena.ball.pos[0]) * 0.5 - car.pos[0], shadow_target_y - car.pos[1], 0.0], dtype=np.float32)
            tactical_dir = retreat_vec / max(1e-4, _norm3(retreat_vec))
        else:
            tactical_dir = unit_to_ball

        reward = 0.0

        fwd_vec = car.get_forward_vector()
        right_vec = car.get_right_vector()
        local_x = float(np.dot(car_to_ball[:2], unit_horiz(fwd_vec)))
        local_y = float(np.dot(car_to_ball[:2], unit_horiz(right_vec)))
        car_speed_horiz = _norm2(car.vel)
        car_fwd_speed = float(np.dot(car.vel[:2], unit_horiz(fwd_vec)))

        # Horizontal-frame validity.
        #
        # Every quantity above projects the car's basis onto the XY plane. For a car driving up a
        # wall the nose is near-vertical, so fwd[:2] is a tiny residual that unit_horiz either
        # zeroes or renormalizes back to unit length -- amplifying orientation noise into
        # confident-looking headings. The flip classifiers downstream then fire on garbage.
        # Require the nose to be at least ~20 degrees off vertical before trusting them.
        horiz_frame_valid = bool(_norm2(fwd_vec) > 0.35)
        pitch_input = float(action[2])
        yaw_input = float(action[3])
        stick_deflection = max(abs(pitch_input), abs(yaw_input))

        is_executing_dodge = bool(not car.on_ground and prev_flip and not car.has_flip)

        # Track dodge execution window for strike contact:
        if is_executing_dodge or car.just_dodged:
            self._dodge_strike_ticks[car.id] = 3
        elif self._dodge_strike_ticks.get(car.id, 0) > 0:
            self._dodge_strike_ticks[car.id] -= 1
        if car.on_ground:
            self._dodge_strike_ticks[car.id] = 0

        # ── 1. Takeoff Transition (Ground -> Air) ─────────────────────────────
        # Takeoff detection along the surface normal, not world +Z.
        #
        # A jump imparts CAR_JUMP_INITIAL_VEL along the car's up vector. On the flat wall that is
        # entirely horizontal and on the upper curved ramp it is mostly horizontal, so the old
        # car.vel[2] > 80.0 gate needed up[2] > ~0.27 and never fired there. That made the whole
        # takeoff block -- including branch 1b, the wall takeoff / air-dribble pop bonus -- dead
        # code in exactly the situations it was written for. Project the launch velocity onto the
        # car's own up vector so a wall jump registers with the same threshold a floor jump does.
        takeoff_normal_speed = float(np.dot(car.vel, car.get_up_vector()))
        if prev_ground and not car.on_ground and (car.vel[2] > 80.0 or takeoff_normal_speed > 80.0):
            is_on_wall_zone = bool(abs(car.pos[0]) > 3400.0 or abs(car.pos[1]) > 4400.0)
            is_on_wall_curve = bool((abs(car.pos[0]) > 3300.0 or abs(car.pos[1]) > 4300.0) and car.pos[2] > 60.0)
            is_aerial_ball = bool(ball_z > 200.0)
            car_boost = float(car.boost)

            # 1a. Close-Quarters Strike Liftoff, 50/50 Challenge, and Flick Pop Setup:
            up_vec = car.get_up_vector()
            local_z = float(np.dot(car_to_ball, up_vec))
            is_contested_5050 = bool(dist <= 450.0 and is_opponent_challenging and ball_z < 220.0 and car.pos[2] < 150.0)
            is_flick_liftoff = bool(dist <= 260.0 and 110.0 <= local_z <= 175.0 and car.pos[2] < 150.0 and -30.0 <= local_x <= 65.0 and abs(local_y) < 55.0)
            is_strike_liftoff = bool(dist <= 750.0 and ball_z < 450.0 and car.pos[2] < 150.0 and forward_alignment > 0.15 and takeoff_closing_vel > -50.0)

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
            elif (is_on_wall_zone or is_on_wall_curve) and (takeoff_closing_vel > 150.0 or forward_alignment > 0.15 or car.vel[2] > 250.0):
                if dist <= 450.0:
                    reward += self.weight * max(0.2, forward_alignment) * 2.5
                elif car_boost >= 30.0:
                    reward += self.weight * max(0.2, forward_alignment) * 3.5

            # 1c. Aerial Floor Launch (Ball elevated in air)
            elif is_aerial_ball and car.pos[2] < 300.0 and forward_alignment > 0.15:
                ball_retreat_vel = float(np.dot(arena.ball.vel, unit_to_ball))
                if ball_retreat_vel < 1100.0:
                    # Reachable ball (Z <= 500 uu): double-jump or pop reachable with 0 boost!
                    if ball_z <= 500.0:
                        reward += self.weight * forward_alignment * 2.0
                    # High aerial ball (Z > 500 uu): requires boost to climb, but NO penalty for low boost
                    elif car_boost >= 30.0:
                        reward += self.weight * forward_alignment * 3.0
                    elif car_boost >= 15.0:
                        reward += self.weight * forward_alignment * 1.5
                elif car_boost >= 30.0 and forward_alignment > 0.4:
                    reward += self.weight * forward_alignment * 0.8

            # 1d. Open-field ground traversal & downfield sprint (dist > 650 uu, ball grounded)
            elif not is_aerial_ball and not is_on_wall_zone and dist > 650.0 and forward_alignment > 0.40:
                if car_fwd_speed > 400.0:
                    is_kickoff = bool(abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and arena.ball.pos[2] < 120.0)
                    liftoff_mult = 1.30 if is_kickoff else 0.80
                    reward += self.weight * liftoff_mult * forward_alignment

        # ── 1e. Neutral Double-Jump Impulse (Fast Aerial Vertical Kick) ─────
        is_neutral_double_jump = bool(
            not car.on_ground and curr_double_jump and not prev_double_jump
        )
        if not is_neutral_double_jump and is_executing_dodge and stick_deflection <= 0.08 and not car.just_dodged:
            is_neutral_double_jump = True

        if is_neutral_double_jump:
            car_vz = float(car.vel[2])
            ball_rel_z = ball_z - car.pos[2]
            if ball_z > 200.0 and ball_rel_z > 40.0 and forward_alignment > 0.10:
                vz_factor = min(1.0, max(0.25, car_vz / 600.0))
                reward += self.weight * 1.2 * vz_factor * max(0.4, forward_alignment)

        # ── 2. Airborne 50/50 Challenge Completion Bonus ──────────────────────
        if not car.on_ground and self._challenge_jump_active.get(car.id, False):
            if car.ball_touches > prev_touch:
                reward += self.weight * 1.5
                self._challenge_jump_active[car.id] = False
        elif car.on_ground:
            self._challenge_jump_active[car.id] = False
            self._flick_window_active[car.id] = False

        # ── 3. Airborne Dodge / Flip & Traversal Impulse ──────────────────────
        # delta_v is measured across a full env step (tick_skip ticks). If the car struck the
        # ball during that window the measurement is dodge_impulse + collision_impulse and its
        # direction is meaningless -- a solid forward flip into the ball reads as a BACKFLIP.
        # Fall back to the commanded stick direction whenever contact occurred this step.
        dodge_delta_v = car.vel - prev_vel
        dodge_delta_v_mag = _norm2(dodge_delta_v)
        touched_this_step = bool(car.ball_touches > prev_touch)
        dv_trustworthy = bool(dodge_delta_v_mag > 80.0 and not touched_this_step)

        dodge_dir_local = np.array([
            1.0 if pitch_input > 0.25 else (-1.0 if pitch_input < -0.25 else 0.0),
            1.0 if yaw_input > 0.25 else (-1.0 if yaw_input < -0.25 else 0.0),
            0.0
        ], dtype=np.float32)
        dodge_norm = _norm3(dodge_dir_local)

        if dv_trustworthy:
            dodge_impulse_world = dodge_delta_v[:2] / max(1e-4, dodge_delta_v_mag)
            dodge_align = float(np.dot(dodge_impulse_world, tactical_dir[:2]))
        elif dodge_norm > 1e-4:
            dodge_impulse_world_full = (dodge_dir_local[0] * fwd_vec[:2] + dodge_dir_local[1] * right_vec[:2]) / dodge_norm
            dodge_align = float(np.dot(dodge_impulse_world_full, tactical_dir[:2]))
        else:
            dodge_align = 0.0

        up_vec = car.get_up_vector()
        local_z = float(np.dot(car_to_ball, up_vec))
        is_flick_active = bool(self._flick_window_active.get(car.id, False) or (dist < 320.0 and 90.0 <= local_z <= 240.0 and -45.0 <= local_x <= 85.0 and abs(local_y) < 75.0))

        fwd_h = unit_horiz(fwd_vec)
        nose_align_ball = float(np.dot(fwd_h, unit_horiz(unit_to_ball)))
        nose_align_tactical = float(np.dot(fwd_h, unit_horiz(tactical_dir)))

        dodge_car_fwd_align = (float(np.dot(dodge_delta_v[:2], fwd_h)) / dodge_delta_v_mag) if dv_trustworthy else 0.0
        is_dodge_backward = bool(pitch_input < -0.20 or (dv_trustworthy and dodge_car_fwd_align < -0.30))

        # Fast aerial pitch-up recognition: tilting nose up under elevated ball is an aerial attempt, not bad backflip
        is_fast_aerial_attempt = bool(
            is_executing_dodge and is_dodge_backward
            and ball_z > 140.0 and (ball_z > car.pos[2] + 60.0)
            and forward_alignment > 0.10 and car.pos[2] > 25.0
        )
        is_5050_backflip = bool(
            is_executing_dodge and is_dodge_backward and dist <= 450.0 and is_opponent_challenging
            and car_fwd_speed < 200.0 and nose_align_ball < 0.20
        )
        # The three classes below are decided entirely from horizontal nose/heading projections,
        # and two of them carry penalties. On a wall those projections are degenerate (see
        # horiz_frame_valid), and a nose-up pitch off the wall toward the ball is a legitimate
        # manoeuvre rather than a wasteful backflip, so leave all three unset there.
        is_halfflip_candidate = bool(
            horiz_frame_valid
            and is_executing_dodge and (is_dodge_backward or (dodge_align > 0.15 and nose_align_tactical < -0.20))
            and nose_align_tactical < -0.20 and (dist > 450.0 or is_wrong_side)
        )
        is_uncontested_dribble_backflip = bool(
            horiz_frame_valid
            and is_executing_dodge and is_dodge_backward and dist <= 450.0
            and not is_opponent_challenging and not is_flick_active and nose_align_ball < -0.20
            and not is_halfflip_candidate
        )
        is_forward_backflip = bool(
            horiz_frame_valid
            and is_executing_dodge and is_dodge_backward
            and not is_5050_backflip and not is_flick_active and not is_halfflip_candidate 
            and not is_fast_aerial_attempt
            and (car_fwd_speed > 100.0 or nose_align_tactical > 0.15)
        )

        if is_executing_dodge and (is_dodge_backward or is_halfflip_candidate):
            if is_5050_backflip:
                self._challenge_jump_active[car.id] = True
            elif is_flick_active or is_fast_aerial_attempt:
                pass  # Free backflip flick / scoop or fast aerial pitch attempt (no penalty)
            elif is_halfflip_candidate:
                self._halfflip_in_progress[car.id] = True
                self._halfflip_cancel_executed[car.id] = False
                self._halfflip_roll_executed[car.id] = False
            elif is_forward_backflip or is_uncontested_dribble_backflip:
                # Full-strength discouragement of genuinely wasteful backflips. The audit's
                # concern here was misclassification, not magnitude: a forward flip that
                # connected had its impulse reversed by the collision and was scored as a
                # backflip. That is handled upstream by dv_trustworthy, which falls back to
                # stick input on any contact step, so this branch now only sees real backflips.
                reward -= self.weight * 0.80

        # Active Half-Flip In-Flight Shaping:
        if not car.on_ground and self._halfflip_in_progress.get(car.id, False):
            if not self._halfflip_cancel_executed.get(car.id, False):
                up_z = float(car.get_up_vector()[2])
                ang_vel = car.ang_vel if hasattr(car, "ang_vel") and car.ang_vel is not None else np.zeros(3, dtype=np.float32)
                pitch_rate = abs(float(np.dot(ang_vel, right_vec)))
                if up_z < 0.20 and pitch_rate < 2.5:
                    self._halfflip_cancel_executed[car.id] = True
                    reward += self.weight * 0.35 * max(0.0, 1.0 - (pitch_rate / 2.5))

        if is_executing_dodge:
            is_open_field = bool(dist > 650.0)
            has_traversal_speed = bool(car_speed_horiz > 350.0)
            is_bad_backflip = bool(is_forward_backflip or is_uncontested_dribble_backflip)

            is_forward_flip = bool(pitch_input > 0.25 or (dv_trustworthy and dodge_align > 0.35))
            is_diagonal_flip = bool((pitch_input > 0.15 and abs(yaw_input) > 0.15) or (dv_trustworthy and 0.20 < abs(float(np.dot(dodge_delta_v[:2] / max(1e-4, dodge_delta_v_mag), unit_horiz(right_vec)))) < 0.85))
            is_forward_or_diagonal = bool((is_forward_flip or is_diagonal_flip) and (forward_alignment > 0.30 or nose_align_tactical > 0.30))

            effective_deflection = max(stick_deflection, min(1.0, dodge_delta_v_mag / 500.0) if dv_trustworthy else 0.0)
            if (effective_deflection >= 0.25 or dv_trustworthy) and dodge_align > 0.20 and not is_bad_backflip:
                if (not is_open_field) or has_traversal_speed:
                    # Traversal flip lock-in horizon check:
                    # When chasing downfield, ensure the ball lead is large enough that the 0.65s flip animation does not overshoot.
                    # Exempt when retreating on defense (is_wrong_side) so defensive recovery flips to gain speed are never tapered!
                    ball_flee_vel = float(np.dot(arena.ball.vel, unit_to_ball))
                    is_chasing_downfield = bool(car_fwd_speed > 250.0 and car_fwd_speed > ball_flee_vel + 100.0)
                    flip_horizon = (car_fwd_speed + 450.0) * 0.65
                    effective_ball_dist = dist + max(0.0, ball_flee_vel) * 0.65

                    overshoot_taper = 1.0
                    if not is_wrong_side and is_chasing_downfield and dist < 750.0 and not is_flick_active:
                        if effective_ball_dist < flip_horizon - 100.0:
                            overshoot_taper = max(0.0, (effective_ball_dist - 200.0) / max(1.0, flip_horizon - 300.0))

                    reward += self.weight * dodge_align * (0.5 + 0.3 * effective_deflection) * overshoot_taper

                    if is_forward_or_diagonal:
                        speed_progression = min(1.0, max(0.2, car_fwd_speed / 1800.0))
                        diag_bonus = 0.50 if is_diagonal_flip else 0.25
                        reward += self.weight * (0.8 * speed_progression + diag_bonus) * max(forward_alignment, nose_align_tactical) * overshoot_taper

                    is_kickoff = bool(abs(arena.ball.pos[0]) < 50.0 and abs(arena.ball.pos[1]) < 50.0 and arena.ball.pos[2] < 120.0 and _norm3(arena.ball.vel) < 100.0)
                    if is_kickoff and dist > 800.0 and (is_forward_flip or is_diagonal_flip):
                        reward += self.weight * 1.50
            elif ball_z > 250.0 and forward_alignment > 0.20:
                reward += self.weight * forward_alignment * 0.5

        # ── 3b. Flick Launch Impulse & Goal Acceleration Bonus ────────────────
        # 3b and 3c below are both keyed on the same touch counter and were previously sequential
        # `if` blocks, so a flick that also registered as a dodge strike collected both bounties
        # for one collision. They are mutually exclusive now: the flick is the more specific
        # classification and takes precedence.
        flick_bounty_paid = False
        target_goal_y = ARENA_EXTENT_Y if car.team == 0 else -ARENA_EXTENT_Y
        target_x = _clip(arena.ball.pos[0], -GOAL_HALF_WIDTH * 0.8, GOAL_HALF_WIDTH * 0.8)
        target_net_pos = np.array([target_x, target_goal_y, GOAL_HEIGHT * 0.35], dtype=np.float32)
        ball_to_net = target_net_pos - arena.ball.pos
        net_dist = _norm3(ball_to_net)
        unit_to_goal = (ball_to_net / max(1e-4, net_dist)) if net_dist > 1e-4 else np.array([0.0, 1.0 if car.team == 0 else -1.0, 0.0], dtype=np.float32)

        if (is_flick_active or dist < 260.0) and car.ball_touches > prev_touch and (is_executing_dodge or car.just_dodged):
            prev_b_vel = self._prev_ball_vel.get(car.id, arena.ball.vel)
            exit_speed_goal = float(np.dot(arena.ball.vel, unit_to_goal))
            delta_v_goal = float(np.dot(arena.ball.vel - prev_b_vel, unit_to_goal))

            if exit_speed_goal > 600.0 and delta_v_goal > 100.0:
                flick_power = min(3.5, (exit_speed_goal / 600.0) + (delta_v_goal / 400.0))
                tactical_mult = 1.0
                if is_opponent_challenging:
                    tactical_mult = 1.50
                elif threats and threats[0].arrival_time < 1.20:
                    tactical_mult = 1.40
                elif abs(target_goal_y - arena.ball.pos[1]) < 2800.0:
                    tactical_mult = 1.25

                # Coefficient cut 3.5 -> 1.3 -> 0.55. The first cut was sized against the nominal
                # goal weight, which is the wrong yardstick: at gamma 0.99 and tick_skip 8 the
                # policy runs 15 steps/s, so a goal three seconds out is worth ~19, not 30. The
                # bounty is also certain the instant contact registers while the goal is only a
                # probability -- at a 30% conversion rate the expected discounted goal was worth
                # LESS than the contact bounty, so farming contact was the better trade and the
                # value function was right to prefer it. 0.55 lands the peak near 2.0, roughly a
                # tenth of a discounted goal, which is shaping rather than competition.
                reward += self.weight * 0.55 * flick_power * tactical_mult
                self._flick_window_active[car.id] = False
                flick_bounty_paid = True

        # ── 3c. Outcome-Driven Dodge Strike & Aerial Interception Bounty ──────
        if car.ball_touches > prev_touch:
            is_dodge_strike = bool(
                self._dodge_strike_ticks.get(car.id, 0) > 0 or 
                car.just_dodged or 
                (not car.on_ground and prev_flip and not car.has_flip)
            )
            if is_dodge_strike:
                # A flick already paid for this collision in 3b. Consume the dodge window so the
                # next touch is scored fresh, but do not pay a second bounty for one impact.
                if not flick_bounty_paid:
                    vy_forward = arena.ball.vel[1] if car.team == 0 else -arena.ball.vel[1]
                    prev_b_vel = self._prev_ball_vel.get(car.id, arena.ball.vel)
                    delta_v_vec = arena.ball.vel - prev_b_vel
                    delta_v_mag = _norm3(delta_v_vec)

                    if vy_forward > -100.0 or delta_v_mag > 300.0:
                        power_factor = min(1.5, max(0.3, delta_v_mag / 800.0))
                        fwd_factor = max(0.2, (vy_forward + 500.0) / 1500.0)
                        # 1.5 -> 1.2 for the same discounted-return reason as the flick
                        # coefficient above; peak lands near 1.5 at the live weight.
                        reward += self.weight * 1.2 * power_factor * min(1.2, fwd_factor)
                self._dodge_strike_ticks[car.id] = 0
            elif not car.on_ground and ball_z > 250.0:
                reward += self.weight * 1.0 * min(1.5, (ball_z - 150.0) / 500.0)

        self._prev_ball_vel[car.id] = arena.ball.vel.copy()

        # ── 4. Wavedash & Speed Impulse on Touchdown / Flip Acceleration ─────
        # Normalize the tactical direction before projecting. Truncating a 3D unit vector to [:2]
        # scales the measured speed by the direction's own horizontal fraction, so chasing a ball
        # high overhead silently reported near-zero traversal speed and suppressed the impulse term.
        tactical_dir_h = unit_horiz(tactical_dir)
        tactical_speed_curr = float(np.dot(car.vel[:2], tactical_dir_h))
        tactical_speed_prev = float(np.dot(prev_vel[:2], tactical_dir_h))
        delta_tactical_speed = tactical_speed_curr - tactical_speed_prev

        if (not prev_ground and car.on_ground) and self._halfflip_in_progress.get(car.id, False):
            up = car.get_up_vector()
            up_z = float(up[2])
            nose_align_target = float(np.dot(car.get_forward_vector()[:2], tactical_dir[:2]))
            if up_z > 0.60 and (forward_alignment > 0.20 or nose_align_target > 0.20):
                reward += self.weight * 1.80
            else:
                reward -= self.weight * 0.80
                delta_tactical_speed = 0.0

            self._halfflip_in_progress[car.id] = False
            self._halfflip_cancel_executed[car.id] = False
            self._halfflip_roll_executed[car.id] = False

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
def _nearest_active_pad_dist(active, poses, cpx: float, cpy: float, radius: float):
    """
    Planar distance to the nearest ACTIVE pad within `radius`, or None if there is none.

    Ignores respawning pads deliberately: a pad on cooldown is not boost, so steering at one
    should not read as progress.
    """
    best2 = radius * radius
    found = False
    for i in range(len(active)):
        if not active[i]:
            continue
        dx = float(poses[i, 0]) - cpx
        dy = float(poses[i, 1]) - cpy
        d2 = dx * dx + dy * dy
        if d2 < best2:
            best2 = d2
            found = True
    return math.sqrt(best2) if found else None


class BoostReward(BaseReward):
    """
    Potential-Based Boost Conservation & Pad Collection.

    The conservation term is an exact potential difference, charged at the same rate in
    both directions, so any path that returns to the boost level it started from sums to
    zero. What survives is a pure function of the boost the car ended up holding.

    On top of that sit situational waste penalties (supersonic burn, ceiling climbs,
    reverse-momentum airbraking, off-axis orbiting) scaled by lose_weight, and a transit
    shaping stream for routing through pads while low, scaled by gain_weight.
    """
    def __init__(self, gain_weight: float = 0.6, lose_weight: float = 0.3):
        super().__init__(gain_weight)
        self.gain_weight = gain_weight
        self.lose_weight = lose_weight
        self._prev_boost: Dict[int, float] = {}
        self._prev_transit: Dict[int, float] = {}

    @staticmethod
    def _potential(b: float) -> float:
        """
        Phi(b) = 3*sqrt(b) - (2/3)*b^1.5, for b in [0, 1].

        This is the exact antiderivative of the hunger-weighted sqrt shaping this class
        used to pay only on the way up. That term was `(1 + 2*(1 - b)) * d(sqrt(b))`,
        while the way down was charged a different and smaller expression at a different
        weight, so a burn-and-refill cycle closed net positive and could be repeated for
        the whole episode. Substituting u = sqrt(b) turns the differential into
        (3 - 2*u^2) du, which integrates to 3u - (2/3)u^3.

        Charging Phi(curr) - Phi(prev) in both directions reproduces the identical
        low-boost hunger gradient -- the slope is still steepest near empty -- while
        telescoping exactly. Phi(0) = 0 and Phi(1) = 7/3, so the conservation term over
        an entire episode is bounded by (7/3) * gain_weight no matter how many pads the
        car drives through.
        """
        u = math.sqrt(max(0.0, min(1.0, b)))
        return 3.0 * u - (2.0 / 3.0) * u * u * u

    def reset(self, initial_state: RocketSimArena):
        self._prev_boost = {car.id: _clip(car.boost / 100.0, 0.0, 1.0) for car in initial_state.cars}
        self._prev_transit = {car.id: self._transit_potential(car, initial_state) for car in initial_state.cars}

    def _transit_potential(self, car: CarState, arena: RocketSimArena) -> float:
        """
        Psi(s): how well placed the car is to pick up boost it needs, as a pure function of
        state -- distance to the nearest active pad, scaled by how empty the tank is.

        This replaces a per-step INCOME stream. The old term paid
        `0.40 * hunger * align * prox * speed` on every step the car was closing on a pad,
        which had no counterpart when it turned away, so circling in and out of pad range
        while low on boost paid roughly 3.0 per approach and could be repeated all episode.

        Charging the DIFFERENCE in Psi instead preserves exactly the behaviour that term was
        bought for -- routing through pads rather than skipping them still pays, step by
        step, in proportion to distance actually closed -- while making it a guide with zero
        net payout. Approaching a pad and turning away refunds precisely what it paid, and
        collecting one zeroes Psi (the pad goes inactive and the tank is full), which cancels
        the approach. All the durable value then comes from Phi, where it belongs.

        The explicit alignment and speed gates the rate term needed are gone: a potential
        difference already measures alignment times speed, because that is what closing
        distance is. The height taper stays, and in this form its old hazard is gone too --
        jumping near a pad costs Psi but landing refunds it, so there is no longer a standing
        opportunity cost for leaving the turf.
        """
        if self.gain_weight <= 1e-6:
            return 0.0
        # Only shaped near the turf; decays to zero by 250 uu so it never rewards an aerial
        # "pad approach" the car cannot actually complete.
        cz = float(car.pos[2])
        if not (car.on_ground or cz < 250.0):
            return 0.0
        air_taper = 1.0 if car.on_ground else max(0.0, 1.0 - (cz - 17.0) / 233.0)

        cpx, cpy = float(car.pos[0]), float(car.pos[1])
        best = 0.0

        # Big orbs: worth going further out of the way for, so a wider search radius.
        if car.boost < 50.0 and hasattr(arena, "_big_pad_pos_3d") and hasattr(arena, "_big_pad_active"):
            # Urgency below 20 boost ramps in continuously rather than switching on at the
            # threshold. As a rate term the old `if boost < 20: hunger *= 1.3` was merely a
            # step in the income; as a potential it became a payment for crossing 20 going
            # down, which made the combined potential non-monotone in boost right there --
            # burning a sliver of boost next to a pad paid 0.074. The ramp reaches the same
            # 1.3x at empty with no discontinuity anywhere.
            hunger = (50.0 - car.boost) / 50.0
            hunger *= 1.0 + 0.3 * _clip((20.0 - car.boost) / 20.0, 0.0, 1.0)
            hunger = min(1.5, hunger)
            d = _nearest_active_pad_dist(arena._big_pad_active, arena._big_pad_pos_3d, cpx, cpy, 1200.0)
            if d is not None:
                best = max(best, 0.40 * hunger * (1.0 - d / 1200.0))

        if car.boost < 65.0 and hasattr(arena, "_small_pad_pos_3d") and hasattr(arena, "_small_pad_active"):
            hunger = (65.0 - car.boost) / 65.0
            d = _nearest_active_pad_dist(arena._small_pad_active, arena._small_pad_pos_3d, cpx, cpy, 550.0)
            if d is not None:
                best = max(best, 0.20 * hunger * (1.0 - d / 550.0))

        return float(self.gain_weight * best * air_taper)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev = self._prev_boost.get(car.id, _clip(car.boost / 100.0, 0.0, 1.0))
        curr = _clip(car.boost / 100.0, 0.0, 1.0)
        self._prev_boost[car.id] = curr

        # Transit potential advances on EVERY step, including kickoff and including steps
        # where the tank did not change. Skipping the update on any step would drop that
        # step's difference on the floor, and a shaping term that silently forgets an
        # interval is no longer telescoping.
        psi_now = self._transit_potential(car, arena)
        psi_delta = psi_now - self._prev_transit.get(car.id, psi_now)
        self._prev_transit[car.id] = psi_now

        boost_diff = math.sqrt(curr) - math.sqrt(prev)

        # ── 1. Symmetric Boost Conservation Potential ────────────────────────
        # Paid on gain and charged at exactly the same rate on loss, so burning boost and
        # refilling it nets zero. Three things used to break that symmetry and each one is
        # deliberately gone:
        #   - the discrete pad pickup bonus, which was pure event income with no counterpart
        #     on the way down (a big orb from empty paid 4.62, more than advancing the ball
        #     the full length of the pitch). Phi already prices a big pad far above a small
        #     one, because a big pad moves b much further, so the distinction survives.
        #   - the low-boost hunger multiplier applied only to gains. It is now baked into the
        #     shape of Phi and therefore applies in both directions.
        #   - the height_factor discount on losses, which charged aerial boost at a fifth of
        #     what refilling it paid. A successful aerial is still free in net terms: the
        #     boost it spends costs exactly what refuelling refunds. Encouraging aerials is
        #     AirRollRecoveryReward's job, not a discount on the fuel gauge.
        # Gated on gain_weight alone. lose_weight now scales only the situational waste
        # penalties below; letting it scale the conservation term at a different value from
        # gain_weight is precisely what made cycling profitable.
        if self.gain_weight <= 1e-6:
            potential_delta = 0.0
        else:
            potential_delta = self.gain_weight * (self._potential(curr) - self._potential(prev))

        # Everything shaped is now a potential difference, so the two are simply summed.
        shaping = potential_delta + psi_delta

        # The kickoff gate applies to the situational PENALTIES only. It used to suppress the
        # whole term, which let the kickoff burn go uncharged while the refill afterwards was
        # still paid -- worth about 1.76 of free reward per kickoff at the current weights.
        # Neutrality already delivers what the gate was for: boost spent on a kickoff costs
        # exactly what refuelling refunds, so there is nothing left to forgive.
        is_kickoff = bool(
            abs(arena.ball.pos[0]) < 50.0
            and abs(arena.ball.pos[1]) < 50.0
            and arena.ball.pos[2] < 120.0
            and _norm3(arena.ball.vel) < 100.0
        )

        if boost_diff > 1e-5:
            return float(shaping)

        elif boost_diff < -1e-5:
            # ── 2. Boost Usage, Waste & Negative Momentum Penalties ─────────────
            # Strictly gate: if lose_weight is zero, only the potentials remain
            if self.lose_weight <= 1e-6 or is_kickoff:
                return float(shaping)
            loss_rew = shaping

            # Supersonic boost waste penalty: burning boost when already at max speed (>= 2150 uu/s)
            speed = _norm3(car.vel)
            # NOTE: all situational penalties below are scaled by flat_scale so the knob is
            # monotonic. Previously they were flat constants ~30x the weighted potential term,
            # which made boost_lose_weight a no-op everywhere except exactly 0.0. flat_scale is
            # normalized against the 0.3 class default so that at the default weight the
            # penalties keep the magnitudes they were originally tuned with.
            flat_scale = self.lose_weight / 0.3
            if speed >= 2150.0 and action[6] > 0.0:
                loss_rew -= flat_scale * (0.35 if car.on_ground else 0.20)

            # Strike-zone overspeed boost waste penalty: burning boost when closing on ball too fast
            self_arr = compute_car_arrival_time(car, arena.ball.pos, arena.ball.vel)
            ball_speed = _norm3(arena.ball.vel)
            if car.on_ground and action[6] > 0.0 and self_arr < 0.35 and speed > ball_speed + 200.0:
                loss_rew -= flat_scale * 0.25

            # Ceiling and vertical climb boost waste penalty: burning boost along ceiling or climbing vertically away from a lower ball
            is_climbing_above_ball = bool(car.vel[2] > 100.0 and car.pos[2] > arena.ball.pos[2] + 200.0)
            if (car.pos[2] > 1750.0 or is_climbing_above_ball) and action[6] > 0.0 and arena.ball.pos[2] < car.pos[2] - 200.0:
                loss_rew -= flat_scale * 0.25

            fwd_vec = car.get_forward_vector()
            fwd_speed = float(np.dot(car.vel, fwd_vec))

            # Reverse-Momentum Boost Waste Penalty:
            # Burning boost when the car's 3D momentum opposes its forward nose direction (fwd_speed < -150 uu/s).
            # Attempting to use boost as an emergency airbrake against reverse momentum wastes massive boost
            # while floating helplessly; the player should coast to ground contact and powerslide/brake instead.
            if action[6] > 0.0 and fwd_speed < -150.0:
                rev_waste_scale = min(1.0, abs(fwd_speed) / 1200.0)
                loss_rew -= flat_scale * ((0.35 if not car.on_ground else 0.20) * rev_waste_scale)

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
            dist_to_ball = _norm3(car_to_ball)

            if not is_retreating_to_defend:
                if car.on_ground and action[6] > 0.0:
                    if dist_to_ball > 300.0:
                        unit_to_ball = car_to_ball / dist_to_ball
                        fwd_align = float(np.dot(fwd_vec, unit_to_ball))
                        if fwd_align < 0.10:
                            loss_rew -= flat_scale * 0.15 * (1.0 - fwd_align)

                # Airborne off-trajectory boost waste penalty:
                # Burning boost while airborne when car's 3D momentum is moving away from or past the ball
                elif not car.on_ground and action[6] > 0.0:
                    is_active_dodge = bool(car.just_dodged or getattr(car, "is_dodging", False))
                    # Physical thruster-momentum penalty during flips:
                    # Thrusters apply force along nose. Penalize if nose points backward against momentum
                    # or steeply down into the pitch (slamming car into turf).
                    if is_active_dodge:
                        car_fwd_proj = float(np.dot(fwd_vec, car.vel))
                        is_thruster_braking = bool(car_fwd_proj < -100.0)
                        horiz_speed = _norm2(car.vel)
                        is_thruster_ground_smash = bool(fwd_vec[2] < -0.40 and car.pos[2] < 250.0 and horiz_speed < 800.0)
                        if is_thruster_braking or is_thruster_ground_smash:
                            loss_rew -= flat_scale * 0.30
                    else:
                        is_recovering_halfflip = bool(float(action[2]) > 0.4 and abs(float(action[4])) > 0.2)
                        if dist_to_ball > 250.0 and not is_recovering_halfflip:
                            unit_to_ball = car_to_ball / dist_to_ball
                            closing_vel = float(np.dot(car.vel, unit_to_ball))
                            eff_align = compute_effective_alignment(car, unit_to_ball)
                            if closing_vel < -100.0 or (closing_vel < 100.0 and eff_align < 0.20):
                                loss_rew -= flat_scale * 0.30 * min(1.0, max(0.2, -closing_vel / 1000.0 if closing_vel < 0 else 0.5))

            return loss_rew
        else:
            # ── 3. Transit Pad Routing ─────────────────────────────────
            # The tank did not change this step, so only the transit potential can have
            # moved, and it is already folded into `shaping`. This branch used to carry a
            # per-step income stream for closing on a pad; see _transit_potential for why
            # that is now a difference instead.
            return float(shaping)


# ==============================================================================
# 9. POWERSLIDE & DRIFT TURN-AROUND REWARD (Tight Hairpin Cuts & Snap Pivoting)
# ==============================================================================
class PowerslideReward(BaseReward):
    """
    Rewards tight, responsive ground turnarounds and snap cuts toward the ball.
    Outcome-driven on positive heading alignment rate when off-axis, gated with proximity and
    self-TTI arrival suppression to prevent drift-skating into the ball, and capped by a
    per-activation budget so a single turnaround cannot be extended into an income stream.

    This is a bootstrapping term. Once the policy can pivot, the velocity potential in
    PlayerToBallVelocityReward carries turn-radius learning on its own, so powerslide_weight is
    expected to be annealed toward zero rather than held at its starting value.
    """
    # Most of a turnaround's value is delivered in the first few tenths of a second. The cap is
    # expressed pre-weight so it tracks powerslide_weight as that weight is annealed down.
    ACTIVATION_BUDGET = 1.0

    def __init__(self, weight: float = 0.30):
        super().__init__(weight)
        self._prev_alignment: Dict[int, float] = {}
        self._activation_spent: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_alignment = {}
        self._activation_spent = {car.id: 0.0 for car in initial_state.cars}
        for car in initial_state.cars:
            d = initial_state.ball.pos - car.pos
            dist = _norm3(d)
            if dist > 1e-4:
                self._prev_alignment[car.id] = float(np.dot(car.get_forward_vector(), d / dist))
            else:
                self._prev_alignment[car.id] = 1.0

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        car_to_ball = arena.ball.pos - car.pos
        dist = _norm3(car_to_ball)
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
        speed = _norm3(car.vel)
        steer_mag = abs(float(action[1]))

        is_turning = bool(car.on_ground and fwd_alignment < 0.60 and steer_mag > 0.25 and speed > 50.0)
        if not is_turning:
            # Leaving the turning state closes the activation and restores the budget. Note the
            # early returns above (degenerate distance, proximity/TTI suppression) deliberately
            # do NOT reset it: they suppress payout inside the strike zone, and treating that as
            # the end of an activation would refund the budget mid-turn.
            self._activation_spent[car.id] = 0.0
            return 0.0

        # Heading alignment progression only.
        #
        # This previously also qualified on `yaw_rate > 1.2` alone and took the larger of the two
        # as its efficiency factor. Nothing in that path required the car to be turning toward
        # anything, so sustained yaw with the stick held paid out on its own: a car circling more
        # than 300 uu from the ball sits under the 0.60 alignment gate for roughly two thirds of
        # every lap and collected on each of those steps, burning no boost. The handbrake economy
        # penalty in CombinedReward could not offset it either, since that fires only when steer
        # is UNDER 0.25, and the distance potential nets to zero around a closed loop.
        alignment_rate = max(0.0, fwd_alignment - prev_align)
        if alignment_rate <= 0.02:
            return 0.0

        pivot_efficiency = min(1.0, alignment_rate * 5.0)
        turn_bonus = pivot_efficiency * (0.6 + 0.4 * steer_mag)

        # Per-activation budget. alignment_rate only ever credits gains and never charges for
        # alignment lost, so weaving the nose in and out still collects on every upswing. The cap
        # bounds what one continuous turn can pay regardless of how the heading oscillates inside
        # it. Same pattern as AirRollRecoveryReward's per-flight budgets.
        spent = self._activation_spent.get(car.id, 0.0)
        remaining = max(0.0, self.ACTIVATION_BUDGET - spent)
        turn_bonus = min(turn_bonus, remaining)
        if turn_bonus <= 0.0:
            return 0.0
        self._activation_spent[car.id] = spent + turn_bonus

        return self.weight * turn_bonus


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
        self._prev_surface_align: Dict[int, float] = {}
        self._prev_heading: Dict[int, float] = {}
        self._prev_on_ground: Dict[int, bool] = {}
        self._airborne_ticks: Dict[int, int] = {}
        self._was_disoriented: Dict[int, bool] = {}
        self._disoriented_this_flight: Dict[int, bool] = {}
        # Did the car put ITSELF off-axis this flight, by dodging? See the detector below.
        self._self_dodged_this_flight: Dict[int, bool] = {}
        self._airborne_recovery_total: Dict[int, float] = {}
        self._wall_landed: Dict[int, bool] = {}
        self._halfflip_cancel_executed: Dict[int, bool] = {}
        self._halfflip_cancel_total: Dict[int, float] = {}
        self._takeoff_heading: Dict[int, float] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_surface_align = {car.id: float(car.get_up_vector()[2]) for car in initial_state.cars}
        self._prev_heading = {car.id: 1.0 for car in initial_state.cars}
        self._prev_on_ground = {car.id: car.on_ground for car in initial_state.cars}
        self._airborne_ticks = {car.id: 0 for car in initial_state.cars}
        self._was_disoriented = {car.id: False for car in initial_state.cars}
        self._disoriented_this_flight = {car.id: False for car in initial_state.cars}
        self._self_dodged_this_flight = {car.id: False for car in initial_state.cars}
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
        speed_horiz = _norm2(v_horiz)
        fwd_h = car.get_forward_vector()[:2]
        fwd_norm = _norm2(fwd_h)

        if speed_horiz > 250.0 and fwd_norm > 1e-4:
            unit_vel_h = v_horiz / speed_horiz
            unit_fwd_h = fwd_h / fwd_norm
            curr_heading = float(np.dot(unit_fwd_h, unit_vel_h))
        else:
            curr_heading = 1.0

        if car.on_ground and prev_ground:
            self._airborne_ticks[car.id] = 0
            self._prev_surface_align[car.id] = 1.0
            self._prev_heading[car.id] = 1.0
            self._was_disoriented[car.id] = False
            self._disoriented_this_flight[car.id] = False
            self._self_dodged_this_flight[car.id] = False
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
                d_b = _norm2(car_to_b)
                self._takeoff_heading[car.id] = float(np.dot(fwd_h / max(1e-4, fwd_norm), car_to_b / d_b)) if d_b > 1e-4 else 1.0

        if not car.on_ground:
            air_ticks = self._airborne_ticks.get(car.id, 0) + 1
            self._airborne_ticks[car.id] = air_ticks
        else:
            air_ticks = self._airborne_ticks.get(car.id, 0)

        # Surface-relative attitude.
        #
        # Recovery used to be scored against world +Z, which paid the car to point its wheels at
        # the pitch floor even while flying at a side wall or backboard -- landing it on its door.
        # Score against the normal of the surface the car is actually about to arrive at instead.
        # compute_landing_surface_normal returns FLOOR_NORMAL whenever no wall is reached first,
        # so open-field recoveries are unchanged and surface_align degenerates to up_z there.
        landing_normal = compute_landing_surface_normal(car) if not car.on_ground else FLOOR_NORMAL
        is_wall_landing = bool(landing_normal[2] < 0.5)
        surface_align = float(np.dot(up, landing_normal))

        prev_surface_align = self._prev_surface_align.get(car.id, surface_align)
        self._prev_surface_align[car.id] = surface_align

        vel_z = float(car.vel[2])

        prev_heading = self._prev_heading.get(car.id, curr_heading)
        self._prev_heading[car.id] = curr_heading

        # Track if car was genuinely knocked off-axis / inverted during this airborne sequence.
        # Gated by air_ticks >= 3 to filter out single-tick suspension micro-hops on curved ramps.
        # Measured against the surface the car is heading to, so a car correctly rolled onto its
        # side to meet a wall is not flagged as disoriented (its old up_z would have read ~0).
        #
        # A dodge the car CHOSE does not count. This detector reads pure geometry -- on its back,
        # or travelling backwards -- and a backflip satisfies both, so the car could invert itself
        # for free and then be paid 0.315 for righting what it had just broken. With jump_bridge
        # zeroed, that loop became the last standing reason to flip: at iteration 1129 the policy
        # was airborne 71% of the time, never reversed (0.0% reverse throttle), and drew 2.55 per
        # episode from this term with 91% of it earned in the air.
        #
        # Order is handled naturally. A car knocked off-axis BEFORE it dodges has already had the
        # flag set, so dodging to recover from a real bump still earns. Only a flight whose
        # disorientation began with the car's own dodge is excluded.
        if car.just_dodged or getattr(car, "is_dodging", False):
            self._self_dodged_this_flight[car.id] = True
        if air_ticks >= 3 and not self._self_dodged_this_flight.get(car.id, False):
            if surface_align < 0.30 or curr_heading < -0.20:
                self._was_disoriented[car.id] = True
                self._disoriented_this_flight[car.id] = True

        car_to_ball = arena.ball.pos - car.pos
        dist_to_ball = _norm3(car_to_ball)
        ball_z = float(arena.ball.pos[2])

        # Aerial engagement check (protects steep climbing & inverted flight during aerials / air dribbles / flip resets)
        unit_to_ball = car_to_ball / max(1e-4, dist_to_ball)
        fwd_align_to_ball = float(np.dot(car.get_forward_vector(), unit_to_ball))
        is_aerial_engagement = bool(
            (ball_z > 350.0 and (dist_to_ball < 450.0 or (fwd_align_to_ball > 0.35 and dist_to_ball < 1500.0))) or
            (not car.on_ground and dist_to_ball < 400.0)
        )

        total_reward = 0.0

        # ── 1. Active 3D Disorientation Recovery (Roll & Yaw) ────────────────
        # Only active when the car was genuinely knocked off-axis, inverted, or executed a flip turnaround
        # The half-flip cancel block (1d) has its own trigger and its own budget, and a half-flip
        # is a dodge by construction. Gating it behind _was_disoriented would have killed it
        # outright once self-inflicted dodges stopped setting that flag.
        is_halfflip_candidate = bool(self._takeoff_heading.get(car.id, 1.0) < -0.20)
        # Generic recovery income: only for a disorientation the car did not cause itself.
        pay_generic_recovery = bool(self._was_disoriented.get(car.id, False))
        # The outer gate additionally opens for a half-flip so block 1d stays reachable, but
        # 1a/1b/1c below each re-check pay_generic_recovery. Without that, is_halfflip_candidate
        # is true of ANY backward-moving flight -- which is exactly what a backflip is -- and
        # the whole branch would reopen for the behaviour this is meant to stop.
        is_recovering = pay_generic_recovery or is_halfflip_candidate
        roll_input = float(action[4])
        is_active_halfflip_cancel = bool(air_ticks <= 18 and self._halfflip_cancel_executed.get(car.id, False))

        ang_vel = car.ang_vel if hasattr(car, "ang_vel") and car.ang_vel is not None else np.zeros(3, dtype=np.float32)
        # Roll rate: rotation around the car's longitudinal (forward) axis
        roll_rate = float(np.dot(car.get_forward_vector(), ang_vel))
        total_ang_speed = _norm3(ang_vel)

        if not is_aerial_engagement and is_recovering and not car.on_ground:
            urgency = min(1.0, max(0.4, (800.0 - car_z) / 600.0))
            rec_spent = self._airborne_recovery_total.get(car.id, 0.0)
            rec_budget = max(0.0, 0.80 - rec_spent)

            # 1a. Active Roll & Inversion Recovery (delta_up > 0), measured toward the landing surface
            delta_up = surface_align - prev_surface_align
            if pay_generic_recovery and delta_up > 0.0 and prev_surface_align < 0.90:
                # Inversion multiplier: rotating from wheels-away (prev align < 0) yields up to 2.0x reward
                inversion_mult = 1.0 + max(0.0, -prev_surface_align) * 1.0
                roll_rec = min(rec_budget, (delta_up * 1.5) * inversion_mult * urgency)
                total_reward += roll_rec
                rec_budget = max(0.0, rec_budget - roll_rec)
                # Accumulate against the running total, not the value read at the top of the step:
                # 1b used to overwrite this with rec_spent + settle_bonus, dropping the roll credit
                # from the ledger and letting the flight exceed its 0.80 recovery budget.
                self._airborne_recovery_total[car.id] = self._airborne_recovery_total.get(car.id, 0.0) + roll_rec

            # 1b. Roll Rate Damping & Settling (D-term):
            # As the car approaches flat attitude (up_z > 0.75), damp angular velocity to prevent rotational overshoot.
            is_touchdown = bool((car_z < 60.0 and vel_z < -50.0) or (not prev_ground and car.on_ground))
            if pay_generic_recovery and surface_align > 0.75 and not is_touchdown:
                abs_roll = abs(roll_rate)
                # Require both roll rate AND total angular velocity to be controlled (eliminates pitch-tumble blindspot)
                if abs_roll < 1.0 and total_ang_speed < 1.8 and (prev_surface_align < 0.90 or delta_up > 0.01):
                    # Stabilized attitude bonus: reward arresting roll velocity near flat
                    settle_bonus = min(rec_budget, (1.0 - abs_roll) * 0.15 * urgency)
                    total_reward += settle_bonus
                    rec_budget = max(0.0, rec_budget - settle_bonus)
                    self._airborne_recovery_total[car.id] = self._airborne_recovery_total.get(car.id, 0.0) + settle_bonus
                elif abs_roll > 2.2 and surface_align > 0.85:
                    # Excess rotational inertia penalty: penalize violent spin that will blow past upright
                    excess_spin = min(1.0, (abs_roll - 2.2) / 2.5)
                    total_reward -= excess_spin * 0.15 * urgency

            # 1c. Active Yaw & Momentum Heading Recovery (delta_heading > 0)
            delta_heading = curr_heading - prev_heading
            if pay_generic_recovery and delta_heading > 0.0 and prev_heading < 0.90 and speed_horiz > 250.0:
                heading_inversion_mult = 1.0 + max(0.0, -prev_heading) * 1.0
                yaw_rec = min(rec_budget, (delta_heading * 1.0) * heading_inversion_mult * urgency)
                total_reward += yaw_rec
                self._airborne_recovery_total[car.id] = self._airborne_recovery_total.get(car.id, 0.0) + yaw_rec

            # 1d. Dedicated Half-Flip Flip-Cancel & Air-Roll Bonus Budget (Pure Outcome-Driven Angular Kinematics)
            cancel_spent = self._halfflip_cancel_total.get(car.id, 0.0)
            cancel_budget = max(0.0, 0.80 - cancel_spent)
            is_halfflip_flight = bool(self._takeoff_heading.get(car.id, 1.0) < -0.20)
            if is_halfflip_flight and air_ticks <= 18 and up_z < 0.50 and speed_horiz > 200.0:
                step_cancel_reward = 0.0
                pitch_rate = abs(float(np.dot(ang_vel, car.get_right_vector())))
                roll_rate_mag = abs(roll_rate)
                # Physical flip-cancel: pitch tumble arrested while the dodge is still running.
                #
                # This used to credit a cancel on `pitch_rate < 2.5` alone. A car with no pitch
                # rotation whatsoever satisfies that, so any backward-moving flight -- a plain
                # backflip, or simply leaving the ground while reversing -- read as a perfectly
                # executed half-flip and collected 0.40. Nothing required the flip to have been
                # started, let alone cancelled.
                #
                # A cancel is the conjunction of two things: a dodge is in progress, and its
                # pitch rotation has already been arrested. An UNCANCELLED backflip fails the
                # second for the length of the animation and the first once the animation ends,
                # so it never satisfies both at once. RocketSim owns is_dodging, so this reads
                # the simulator's own notion of the flip rather than inferring one.
                is_cancelling_now = bool(getattr(car, "is_dodging", False) and pitch_rate < 2.5)
                if not self._halfflip_cancel_executed.get(car.id, False) and is_cancelling_now:
                    self._halfflip_cancel_executed[car.id] = True
                    c_rew = min(cancel_budget, (0.40 * max(0.0, 1.0 - pitch_rate / 2.5)) * urgency)
                    step_cancel_reward += c_rew
                    cancel_budget = max(0.0, cancel_budget - c_rew)
                # Physical roll-upright: rolling around forward vector toward wheels down.
                # Gated on the cancel having actually been credited, so the roll bonus is the
                # second half of a genuine half-flip rather than a payout for any backward
                # flight that happens to be rotating.
                if self._halfflip_cancel_executed.get(car.id, False) and (delta_up > 0.0 or roll_rate_mag > 0.8):
                    roll_metric = min(1.0, max(delta_up / 0.05, roll_rate_mag / 3.0))
                    r_rew = min(cancel_budget, (0.40 * roll_metric) * urgency)
                    step_cancel_reward += r_rew
                    cancel_budget = max(0.0, cancel_budget - r_rew)
                total_reward += step_cancel_reward
                self._halfflip_cancel_total[car.id] = cancel_spent + step_cancel_reward

            # Conclude in-flight roll/yaw recovery once upright attitude is restored AND total angular velocity has settled
            if surface_align > 0.88 and curr_heading > 0.85 and total_ang_speed < 1.5:
                self._was_disoriented[car.id] = False

        # Upright Roll Input Suppression:
        # Prevents continuous Gaussian action noise or persistent roll holding from rolling off-axis once upright
        if not car.on_ground and surface_align > 0.90 and not is_active_halfflip_cancel and not is_aerial_engagement:
            if abs(roll_input) > 0.15:
                total_reward -= (abs(roll_input) - 0.15) * 0.10

        # ── 2. Touchdown Alignment (Evaluated as a single impulse near ground contact) ──
        if (car_z < 60.0 and vel_z < -50.0) or (not prev_ground and car.on_ground):
            # A flight the car put itself into does not qualify. _was_disoriented and
            # _disoriented_this_flight are already withheld for self-inflicted dodges; the
            # half-flip cancel remains a genuine outcome and still opens the touchdown reward.
            had_disorientation = bool(self._was_disoriented.get(car.id, False) or self._disoriented_this_flight.get(car.id, False) or self._halfflip_cancel_executed.get(car.id, False))
            if had_disorientation:
                # Multi-Surface Landing Evaluator:
                # Evaluate alignment against floor normal [0,0,1] and nearest wall normal
                floor_align = up_z
                if car_z > 150.0 or abs(car.pos[0]) > 3500.0 or abs(car.pos[1]) > 4500.0:
                    is_near_side = bool(abs(car.pos[0]) > 3400.0)
                    is_near_back = bool(abs(car.pos[1]) > 4400.0)
                    wall_nx = -math.copysign(1.0, car.pos[0]) if is_near_side else 0.0
                    wall_ny = -math.copysign(1.0, car.pos[1]) if is_near_back else 0.0
                    wall_align = float(up[0] * wall_nx + up[1] * wall_ny)
                    # surface_align included so a car that landed on the wall it was predicted to
                    # reach still scores as wheels-down even when it is inside the fixed
                    # |x| > 3400 / |y| > 4400 bands but the nearest-wall guess picked the other axis.
                    best_landing_align = max(floor_align, wall_align, surface_align)
                else:
                    best_landing_align = floor_align

                if best_landing_align > 0.0:
                    # Positive continuous landing gradient for any wheels-down landing
                    total_reward += (best_landing_align * 0.5)
                    # Touchdown spin penalty: landing with high rotational speed causes the car to bounce onto its roof
                    if total_ang_speed > 1.5:
                        spin_excess = min(1.0, (total_ang_speed - 1.5) / 2.5)
                        total_reward -= spin_excess * 0.30
                    if speed_horiz > 300.0 and curr_heading > 0.30:
                        total_reward += (curr_heading * 0.5)
                        # 180° Turnaround Half-Flip Completion Bonus:
                        if self._halfflip_cancel_executed.get(car.id, False) and curr_heading > 0.60 and speed_horiz > 400.0 and self._takeoff_heading.get(car.id, 1.0) < -0.20:
                            total_reward += 1.50
                elif not is_active_halfflip_cancel:
                    # best_landing_align <= 0.0: Door or roof crash.
                    # Damp (do not waive) the charge for landings that conclude a close ball
                    # engagement: the in-flight recovery reward is already suppressed there by
                    # is_aerial_engagement, so charging the full crash would be one-sided.
                    # Landing on the roof stays negative at any distance from the ball.
                    # At best_landing_align = 0.0 (flat on door): -0.15 * urgency
                    # Scales strictly monotonically to -0.40 * urgency at best_landing_align = -1.0 (inverted roof)
                    engagement_scale = 0.35 if dist_to_ball < 400.0 else 1.0
                    urgency = min(1.0, max(0.4, (800.0 - car_z) / 600.0))
                    door_crash = -0.15 + (best_landing_align * 0.25)
                    total_reward += door_crash * urgency * engagement_scale

                # Consume disorientation so touchdown reward only fires once per landing
                self._was_disoriented[car.id] = False
                self._disoriented_this_flight[car.id] = False
                self._halfflip_cancel_executed[car.id] = False
                self._halfflip_cancel_total[car.id] = 0.0

        if car.on_ground:
            self._airborne_ticks[car.id] = 0
            self._prev_surface_align[car.id] = 1.0
            self._prev_heading[car.id] = 1.0
            self._was_disoriented[car.id] = False
            self._disoriented_this_flight[car.id] = False
            self._self_dodged_this_flight[car.id] = False
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
                    # Measured against the landing surface: demanding world-upright here fought the
                    # correct attitude for challenging a ball up on a wall.
                    upright_bonus = max(0.0, surface_align) * 0.2 if car.get_forward_vector()[2] < 0.5 else 0.1
                    total_reward += (fwd_align * 0.3 + upright_bonus) * 0.5

        return self.weight * total_reward


class JumpCostReward(BaseReward):
    """
    A flat fee charged once, on the frame the car leaves the ground under its own jump.

    Why this exists. The three binary action channels are Bernoulli, and an entropy bonus
    pulls a Bernoulli with no reward gradient toward probability 0.5. Boost and handbrake are
    both priced -- boost by its usage and waste penalties, handbrake by the economy penalty --
    and both sit far from 0.5 in the learned policy. Jump was priced only by JumpBridgeReward,
    and once that weight went to zero nothing charged for pressing it at all. Measured at
    iteration 5360, raw P(jump) was 0.377 airborne and 0.613 one step after landing, against
    0.042 for boost. The policy was flipping a coin on every frame a jump was available, and a
    coin flip at 15 Hz looks exactly like re-jumping the instant the wheels touch down.

    Deliberately unconditional: no distance or ball-height gate. A gate would create a cliff
    at its own boundary, which is the class of defect that produced most of the trouble in
    this reward set. A jump that leads to a touch, a clear or a goal absorbs the fee easily;
    a jump taken because the channel had no opinion does not.

    Charged strictly on the rising edge of takeoff. Not per airborne step, so holding jump for
    full height costs nothing extra, and not on the dodge, so flips and aerial manoeuvres are
    not taxed a second time. It can only ever subtract, so unlike the reward it replaces it
    cannot be farmed.
    """

    def __init__(self, weight: float = 0.025):
        super().__init__(weight)
        self._prev_on_ground: Dict[int, bool] = {}

    def reset(self, initial_state: RocketSimArena):
        self._prev_on_ground = {car.id: bool(car.on_ground) for car in initial_state.cars}

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        prev_ground = self._prev_on_ground.get(car.id, bool(car.on_ground))
        self._prev_on_ground[car.id] = bool(car.on_ground)

        if self.weight <= 1e-9:
            return 0.0

        # Leaving the ground is not enough on its own: a car can drive off a ramp or the lip
        # of the wall without spending anything. has_jump is consumed by an actual jump and
        # stays available otherwise, so it separates a takeoff from a roll-off.
        took_off = bool(prev_ground and not car.on_ground and not car.has_jump)
        return -self.weight if took_off else 0.0


class TimeCostReward(BaseReward):
    """
    A flat cost charged every step, to make time worth something.

    WHY THIS EXISTS, and why it replaces three mechanic-specific terms.

    Every shaping term here is potential-based, which is what stops them being farmed, and it
    also makes them blind to time. player_to_ball pays (prev_dist - curr_dist) / 2000 per step,
    so an approach telescopes to (start_dist - end_dist) / 2000 -- a quantity in which elapsed
    time does not appear at all. Measured directly: the same approach driven at full throttle
    with boost took 27.9 steps and paid 4.3688, and at half throttle took 48.0 steps and paid
    4.3912. Arriving 1.3 seconds later paid very slightly MORE.

    Discounting does not cover this. At gamma 0.995 a one-second delay is nominally worth 7.2%,
    but a potential pays continuously along the path rather than at the end, so a slower car
    collects the same total only slightly later. The discounted totals across those runs came
    out 3.56, 3.76 and 3.63, not even ordered by speed.

    So no mechanic whose only benefit is SAVING TIME could ever be worth a point: not the
    half-flip, not the powerslide turnaround, not preserving speed through a crooked landing
    (measured at 400 uu/s of speed saved, worth nothing when speed is not priced). The bot was
    not failing to learn them. It had correctly learned they were worthless, and reversing at
    600 uu/s collected exactly what a half-flip and a 1400 uu/s drive collected.

    HOW IT WORKS. Every episode in this environment ends in a goal -- measured across 124
    consecutive episodes, zero reached the 600-step limit -- so episode length is entirely
    determined by how long someone takes to score. A flat per-step cost therefore prices
    exactly that: dawdling costs, decisiveness pays, and every mechanic that shortens the path
    to a goal inherits value from it without needing a term of its own.

    ON ENDING THE EPISODE THE WRONG WAY. A living cost does give the policy a reason to end
    the episode, and conceding ends it too. The arithmetic is what makes that safe rather than
    the intent: at 0.01 per step, throwing a goal to end a 180-step episode 100 steps early
    saves 1.0 and costs 30. Keep this weight small enough that the trade stays that lopsided.
    A full 600-step episode costs 6.0 against a 30.0 concede.

    Deliberately unconditional, including on the terminal step. A gate would create a region
    where stalling is free, which is the whole behaviour this is meant to price.
    """

    def __init__(self, weight: float = 0.01):
        super().__init__(weight)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if self.weight <= 1e-9:
            return 0.0
        return -self.weight


class SpinCostReward(BaseReward):
    """
    A per-step fee on airborne angular speed above a deadband.

    Why this exists. Pitch, yaw and roll had no price on them at all. Measured across
    iterations 7780 to 9700, the policy held mean |pitch| of 0.72 with 37% of airborne steps
    past 0.9, and the car spun at a mean 3.66 rad/s against a 5.5 rad/s physics cap, with 75%
    of airborne steps above 2 rad/s. A car tumbling at that rate cannot land upright or aim,
    and more than half of all episodes ended with the bot never touching the ball.

    The counterfactual is what settled it. Replaying the same policy with pitch, yaw and roll
    forced to zero produced 51% more touches, 154% more ball progression and flipped goals from
    net negative to net positive. The air control the policy had learned was worse than none.

    AirRollRecoveryReward was paying for the tumble -- about +0.07 per flight for exactly the
    input mix the policy used -- but zeroing that term for 2000 iterations moved angular
    velocity only from 3.85 to 3.66. It funded the behaviour without causing it. What remained
    was that nothing opposed the rotational channels in either direction.

    Three choices worth recording, because each rules out a worse version:

    1. It prices the RESULT, not the input. Charging for stick deflection taxes a precise
       recovery exactly as hard as a barrel roll, and a car recovering from a real bump must
       apply large input while spinning fast. Charging for angular speed instead means a
       recovery pays only while it is still spinning and stops paying the moment it settles.

    2. All three axes together, via the magnitude of the angular velocity vector. Pitch is the
       saturated channel today, but pricing pitch alone just invites the same tumble to be
       rebuilt out of yaw and roll.

    3. A deadband, so ordinary air control is free. Only rotation faster than a car needs for
       deliberate manoeuvring is charged, and the charge grows with the excess.

    Dodges are exempt. A dodge rotates the car hard by design and the driver cannot stop it
    mid-flip, so charging it would be charging for the dodge itself -- which JumpCostReward
    already does, once, at takeoff. Taxing it again here would price flips twice.
    """

    # RocketSim clamps car angular velocity here; the excess is normalised against it so the
    # per-step charge is at most `weight`.
    ANG_VEL_CAP = 5.5

    def __init__(self, weight: float = 0.03, deadband: float = 2.0):
        super().__init__(weight)
        self.deadband = float(deadband)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool, scoring_team: Optional[int]) -> float:
        if self.weight <= 1e-9 or car.on_ground:
            return 0.0

        # A dodge is a scripted rotation the car cannot arrest; JumpCostReward already charged
        # for initiating it.
        if car.just_dodged or getattr(car, "is_dodging", False):
            return 0.0

        ang_vel = getattr(car, "ang_vel", None)
        if ang_vel is None:
            return 0.0
        speed = _norm3(ang_vel)
        excess = speed - self.deadband
        if excess <= 0.0:
            return 0.0

        span = max(1e-6, self.ANG_VEL_CAP - self.deadband)
        return -self.weight * min(1.0, excess / span)


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
            "own_goal_threat": OwnGoalThreatReward(
                weight=weights.get("own_goal_threat_weight", 0.0)
            ),
            "player_to_ball": PlayerToBallVelocityReward(
                weight=weights.get("player_to_ball_weight", 0.6),
                boost_pathing_threshold=weights.get("boost_pathing_threshold", 50.0)
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
            ),
            "jump_cost": JumpCostReward(
                weight=weights.get("jump_cost_weight", 0.025)
            ),
            "time_cost": TimeCostReward(
                weight=weights.get("time_cost_weight", 0.01)
            ),
            "spin_cost": SpinCostReward(
                weight=weights.get("spin_cost_weight", 0.03),
                deadband=weights.get("spin_cost_deadband", 2.0)
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

        if "own_goal_threat_weight" in new_weights and "own_goal_threat" in self.rewards:
            self.rewards["own_goal_threat"].weight = float(new_weights["own_goal_threat_weight"])

        if "player_to_ball_weight" in new_weights and "player_to_ball" in self.rewards:
            self.rewards["player_to_ball"].weight = float(new_weights["player_to_ball_weight"])
        if "boost_pathing_threshold" in new_weights and "player_to_ball" in self.rewards:
            self.rewards["player_to_ball"].boost_pathing_threshold = float(new_weights["boost_pathing_threshold"])

        if "powerslide_weight" in new_weights and "powerslide" in self.rewards:
            self.rewards["powerslide"].weight = float(new_weights["powerslide_weight"])

        if "jump_cost_weight" in new_weights and "jump_cost" in self.rewards:
            self.rewards["jump_cost"].weight = float(new_weights["jump_cost_weight"])

        if "time_cost_weight" in new_weights and "time_cost" in self.rewards:
            self.rewards["time_cost"].weight = float(new_weights["time_cost_weight"])

        if "spin_cost_weight" in new_weights and "spin_cost" in self.rewards:
            self.rewards["spin_cost"].weight = float(new_weights["spin_cost_weight"])
        if "spin_cost_deadband" in new_weights and "spin_cost" in self.rewards:
            self.rewards["spin_cost"].deadband = float(new_weights["spin_cost_deadband"])

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
            dist = _norm3(car_to_ball)
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
            dist = _norm3(car_to_ball)
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


