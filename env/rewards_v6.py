"""
Reward v6: v5 plus one term, T6 align ball goal. docs/reward_v6_align_spec.md is the specification;
every formula here is written out in full there (v3's in docs/reward_v3_spec.md), and nothing in
this file may differ from them. v5 is v3's terms at gamma 0.9977; gamma is a setting, not code.

  T1 goal            +goal_reward to the scorers, -goal_reward * (1 - aggression_bias) to the conceders
  T2 ball_position   potential  (|b - own| - |b - opp|) / 10240,           weight ball_position_weight
  T3 touch           on a touch  min(1, |dv_ball| / 2300),                  weight touch_weight
  T4 closeness       potential  -|b - c| / 12000,                          weight closeness_weight (0 in v6)
  T5 boost           potential  sqrt(boost / 100),                         weight boost_weight
  T6 align           potential  0.5 cos(b - c, c - own) + 0.5 cos(b - c, opp - c),  weight align_weight

T6 is Seer's Align Ball Goal (eq. 3.7) with the attacking half as rlgym-tools writes it; the paper
prints cos(c - b, opp - c), which scores the ideal attacking position -1. See the spec, section 2.

T1-T5 are v3's own functions, imported from env/rewards_v3.py (frozen, and hashed into v6's identity
too). The rules of v3 still hold: no term reads the action (R2), every dense term pays
w * (gamma * phi(s') - phi(s)) with phi(s') = 0 on a goal step (R4), and there are no gates (R6).

v6 is frozen once its run starts (config/reward_versions/v6.json, test_reward_versions_frozen.py).
A change is v7, in a new file.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from env.physics_engine import CarState, RocketSimArena
from env.rewards_v3 import (
    REWARD_V3_DEFAULTS, TOUCH_FULL_DV,
    ball_position_potential, boost_potential, closeness_potential, goal_centres,
)

# Code-side defaults; config/reward_versions/v6.json carries the values the run uses, and matches these.
REWARD_V6_DEFAULTS: Dict[str, float] = {
    **REWARD_V3_DEFAULTS,
    "closeness_weight": 0.0,   # retired since v4
    "align_weight": 1.0,
    "gamma": 0.9977,           # v5's horizon
}

# Below this length a direction is meaningless. A numerical guard, not a gate (spec section 2):
# a car's centre cannot reach the ball's centre or a goal centre in play.
MIN_LENGTH = 1.0


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < MIN_LENGTH or nv < MIN_LENGTH:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def align_potential(car: CarState, arena: RocketSimArena) -> float:
    """
    T6: +1 goal-side and behind the ball, -1 caught upfield of it.

    The defending half asks whether the car is on the line from its own net to the ball; the
    attacking half whether the ball is on the car's line to the opponent's net.
    """
    own, opp = goal_centres(car.team)
    b = np.asarray(arena.ball.pos, dtype=np.float64)
    c = np.asarray(car.pos, dtype=np.float64)
    return 0.5 * _cos(b - c, c - own) + 0.5 * _cos(b - c, opp - c)


# name -> (potential, weight key). Evaluation order is the breakdown order.
POTENTIAL_TERMS: Dict[str, Tuple[Callable[[CarState, RocketSimArena], float], str]] = {
    "ball_position": (ball_position_potential, "ball_position_weight"),
    "closeness": (closeness_potential, "closeness_weight"),
    "boost": (boost_potential, "boost_weight"),
    "align": (align_potential, "align_weight"),
}
TERM_NAMES = ("goal", "ball_position", "touch", "closeness", "boost", "align")


class RewardV6:
    """The six v6 terms for every car of one arena. Per-car state is keyed by car id."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.w: Dict[str, float] = dict(REWARD_V6_DEFAULTS)
        self.update_weights(weights or {})
        self._phi: Dict[Tuple[str, int], float] = {}
        self._touches: Dict[int, int] = {}
        self._ball_vel: Dict[int, np.ndarray] = {}

    def update_weights(self, new_weights: Dict[str, float]):
        """Unknown keys are ignored, so a shared weights dict can carry other versions' keys."""
        for key in REWARD_V6_DEFAULTS:
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

        # T2, T4, T5, T6: potentials, phi(terminal) = 0 on a goal step
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


class RewardManagerV6:
    """RewardManager's interface over RewardV6, so the env drives any version the same way."""

    version = "v6"

    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        self.reward = RewardV6(reward_weights)

    def reset(self, initial_state: RocketSimArena):
        self.reward.reset(initial_state)

    def update_weights(self, new_weights: Dict[str, float]):
        self.reward.update_weights(new_weights)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool,
                   scoring_team: Optional[int], include_breakdown: bool = True) -> Tuple[float, Dict[str, float]]:
        return self.reward.get_reward(car, arena, action, is_goal, scoring_team, include_breakdown)
