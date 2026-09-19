"""
Reward v3: five terms, state and outcome only. docs/reward_v3_spec.md is the specification; every
formula here is written out in full there, and nothing in this file may differ from it.

  T1 goal            +goal_reward to the scorers, -goal_reward * (1 - aggression_bias) to the conceders
  T2 ball_position   potential  (|b - own| - |b - opp|) / 10240,           weight ball_position_weight
  T3 touch           on a touch  min(1, |dv_ball| / 2300),                  weight touch_weight
  T4 closeness       potential  -|b - c| / 12000,                          weight closeness_weight (annealed)
  T5 boost           potential  sqrt(boost / 100),                         weight boost_weight

Rules this file keeps (spec section 2):
  - No term reads the action vector (R2). get_reward accepts it only to share RewardManager's signature.
  - Every dense term pays w * (gamma * phi(s') - phi(s)), with phi(s') = 0 on a goal step (R4), so any
    loop through the same states sums to zero and nothing here can be farmed by repetition.
  - No gates, clips or special cases beyond what the spec writes down (R6).

v3 is frozen once its run starts (config/reward_versions/v3.json, test_reward_versions_frozen.py).
A change is v4, in a new file.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from env.physics_engine import ARENA_EXTENT_Y, GOAL_HEIGHT, CarState, RocketSimArena

# Code-side defaults; config/reward_versions/v3.json carries the values the run uses, and matches these.
REWARD_V3_DEFAULTS: Dict[str, float] = {
    "goal_reward": 10.0,
    "aggression_bias": 0.25,
    "ball_position_weight": 5.0,
    "touch_weight": 0.5,
    "closeness_weight": 1.0,
    "boost_weight": 1.0,
    "gamma": 0.995,
}

GOAL_CENTRE_Z = GOAL_HEIGHT / 2.0
GOAL_SEPARATION = 2.0 * ARENA_EXTENT_Y   # 10240: the largest |b - own| - |b - opp| can be
TOUCH_FULL_DV = 2300.0                   # ball speed change that earns the full touch reward
CLOSENESS_SCALE = 12000.0                # roughly the pitch diagonal


def goal_centres(team: int) -> Tuple[np.ndarray, np.ndarray]:
    """(own, opponent) goal centres for a team: blue (0) defends -y, orange (1) defends +y."""
    own_y = -ARENA_EXTENT_Y if team == 0 else ARENA_EXTENT_Y
    return (np.array([0.0, own_y, GOAL_CENTRE_Z], dtype=np.float64),
            np.array([0.0, -own_y, GOAL_CENTRE_Z], dtype=np.float64))


def ball_position_potential(car: CarState, arena: RocketSimArena) -> float:
    """T2: -1 with the ball in our goal mouth, +1 in theirs. Exactly zero-sum between the teams."""
    own, opp = goal_centres(car.team)
    b = np.asarray(arena.ball.pos, dtype=np.float64)
    return float((np.linalg.norm(b - own) - np.linalg.norm(b - opp)) / GOAL_SEPARATION)


def closeness_potential(car: CarState, arena: RocketSimArena) -> float:
    """T4: minus the 3D car-to-ball distance, normalised."""
    d = np.asarray(arena.ball.pos, dtype=np.float64) - np.asarray(car.pos, dtype=np.float64)
    return -float(np.linalg.norm(d)) / CLOSENESS_SCALE


def boost_potential(car: CarState, arena: RocketSimArena) -> float:
    """T5: concave in boost held, so a pad is worth more when low than when nearly full."""
    return math.sqrt(min(100.0, max(0.0, float(car.boost))) / 100.0)


# name -> (potential, weight key). Evaluation order is the breakdown order.
POTENTIAL_TERMS: Dict[str, Tuple[Callable[[CarState, RocketSimArena], float], str]] = {
    "ball_position": (ball_position_potential, "ball_position_weight"),
    "closeness": (closeness_potential, "closeness_weight"),
    "boost": (boost_potential, "boost_weight"),
}
TERM_NAMES = ("goal", "ball_position", "touch", "closeness", "boost")


class RewardV3:
    """The five v3 terms for every car of one arena. Per-car state is keyed by car id."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.w: Dict[str, float] = dict(REWARD_V3_DEFAULTS)
        self.update_weights(weights or {})
        self._phi: Dict[Tuple[str, int], float] = {}
        self._touches: Dict[int, int] = {}
        self._ball_vel: Dict[int, np.ndarray] = {}

    def update_weights(self, new_weights: Dict[str, float]):
        """Unknown keys are ignored, so a shared weights dict can carry other versions' keys."""
        for key in REWARD_V3_DEFAULTS:
            if key in new_weights:
                self.w[key] = float(new_weights[key])

    def reset(self, initial_state: RocketSimArena):
        self._phi.clear()
        for car in initial_state.cars:
            for name, (phi, _) in POTENTIAL_TERMS.items():
                self._phi[(name, car.id)] = phi(car, initial_state)
            self._touches[car.id] = car.ball_touches
            self._ball_vel[car.id] = np.array(initial_state.ball.vel, dtype=np.float64)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool,
                   scoring_team: Optional[int], include_breakdown: bool = True) -> Tuple[float, Dict[str, float]]:
        del action  # R2: the reward is a function of the state transition alone
        terms: Dict[str, float] = {}

        # T1 goal
        goal = 0.0
        if is_goal and scoring_team is not None:
            goal = self.w["goal_reward"] if car.team == scoring_team \
                else -self.w["goal_reward"] * (1.0 - self.w["aggression_bias"])
        terms["goal"] = goal

        # T3 touch: judged on how hard the ball was hit, not where it went (T2 judges that)
        ball_vel = np.array(arena.ball.vel, dtype=np.float64)
        prev_vel = self._ball_vel.get(car.id, ball_vel)
        prev_touches = self._touches.get(car.id, car.ball_touches)
        touch = 0.0
        if car.ball_touches > prev_touches:
            touch = self.w["touch_weight"] * min(1.0, float(np.linalg.norm(ball_vel - prev_vel)) / TOUCH_FULL_DV)
        self._touches[car.id] = car.ball_touches
        self._ball_vel[car.id] = ball_vel

        # T2, T4, T5: potentials, phi(terminal) = 0 on a goal step
        gamma = self.w["gamma"]
        for name, (phi, weight_key) in POTENTIAL_TERMS.items():
            key = (name, car.id)
            new = 0.0 if is_goal else phi(car, arena)
            old = self._phi.get(key, new)
            self._phi[key] = new
            terms[name] = self.w[weight_key] * (gamma * new - old)
        terms["touch"] = touch

        total = float(sum(terms.values()))
        return total, ({n: float(terms[n]) for n in TERM_NAMES} if include_breakdown else {})


class RewardManagerV3:
    """RewardManager's interface over RewardV3, so the env drives either version the same way."""

    version = "v3"

    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        self.reward = RewardV3(reward_weights)

    def reset(self, initial_state: RocketSimArena):
        self.reward.reset(initial_state)

    def update_weights(self, new_weights: Dict[str, float]):
        self.reward.update_weights(new_weights)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool,
                   scoring_team: Optional[int], include_breakdown: bool = True) -> Tuple[float, Dict[str, float]]:
        return self.reward.get_reward(car, arena, action, is_goal, scoring_team, include_breakdown)
