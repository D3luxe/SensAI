"""
Reward v4: v3's five terms plus one, T6 ball race. docs/reward_v4_spec.md is the specification;
every formula here is written out in full there (v3's in docs/reward_v3_spec.md), and nothing in
this file may differ from them.

  T1 goal            +goal_reward to the scorers, -goal_reward * (1 - aggression_bias) to the conceders
  T2 ball_position   potential  (|b - own| - |b - opp|) / 10240,           weight ball_position_weight
  T3 touch           on a touch  min(1, |dv_ball| / 2300),                  weight touch_weight
  T4 closeness       potential  -|b - c| / 12000,                          weight closeness_weight (0 in v4)
  T5 boost           potential  sqrt(boost / 100),                         weight boost_weight
  T6 race            potential  (|b - o| - |b - c|) / 12000,               weight race_weight
                     o = the nearest opponent car

T1-T5 are v3's own functions, imported from env/rewards_v3.py (frozen, and hashed into v4's identity
too). The rules of v3 still hold: no term reads the action (R2), every dense term pays
w * (gamma * phi(s') - phi(s)) with phi(s') = 0 on a goal step (R4), and there are no gates (R6).

v4 is frozen once its run starts (config/reward_versions/v4.json, test_reward_versions_frozen.py).
A change is v5, in a new file.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from env.physics_engine import CarState, RocketSimArena
from env.rewards_v3 import (
    REWARD_V3_DEFAULTS, TOUCH_FULL_DV, CLOSENESS_SCALE,
    ball_position_potential, boost_potential, closeness_potential,
)

# Code-side defaults; config/reward_versions/v4.json carries the values the run uses, and matches these.
REWARD_V4_DEFAULTS: Dict[str, float] = {
    **REWARD_V3_DEFAULTS,
    "closeness_weight": 0.0,   # retired: v3 ended with it at zero and v4 does not restart it
    "race_weight": 1.0,
}

RACE_SCALE = CLOSENESS_SCALE   # the same normalisation as T4


def race_potential(car: CarState, arena: RocketSimArena) -> float:
    """
    T6: how much nearer the ball this car is than the nearest opponent, normalised. In 1v1 the two
    cars' values are exact negatives. With no opponent on the pitch there is no race, and it is 0.
    """
    b = np.asarray(arena.ball.pos, dtype=np.float64)
    opp = [float(np.linalg.norm(b - np.asarray(o.pos, dtype=np.float64))) for o in arena.cars if o.team != car.team]
    if not opp:
        return 0.0
    own = float(np.linalg.norm(b - np.asarray(car.pos, dtype=np.float64)))
    return (min(opp) - own) / RACE_SCALE


# name -> (potential, weight key). Evaluation order is the breakdown order.
POTENTIAL_TERMS: Dict[str, Tuple[Callable[[CarState, RocketSimArena], float], str]] = {
    "ball_position": (ball_position_potential, "ball_position_weight"),
    "closeness": (closeness_potential, "closeness_weight"),
    "boost": (boost_potential, "boost_weight"),
    "race": (race_potential, "race_weight"),
}
TERM_NAMES = ("goal", "ball_position", "touch", "closeness", "boost", "race")


class RewardV4:
    """The six v4 terms for every car of one arena. Per-car state is keyed by car id."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.w: Dict[str, float] = dict(REWARD_V4_DEFAULTS)
        self.update_weights(weights or {})
        self._phi: Dict[Tuple[str, int], float] = {}
        self._touches: Dict[int, int] = {}
        self._ball_vel: Dict[int, np.ndarray] = {}

    def update_weights(self, new_weights: Dict[str, float]):
        """Unknown keys are ignored, so a shared weights dict can carry other versions' keys."""
        for key in REWARD_V4_DEFAULTS:
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


class RewardManagerV4:
    """RewardManager's interface over RewardV4, so the env drives any version the same way."""

    version = "v4"

    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        self.reward = RewardV4(reward_weights)

    def reset(self, initial_state: RocketSimArena):
        self.reward.reset(initial_state)

    def update_weights(self, new_weights: Dict[str, float]):
        self.reward.update_weights(new_weights)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool,
                   scoring_team: Optional[int], include_breakdown: bool = True) -> Tuple[float, Dict[str, float]]:
        return self.reward.get_reward(car, arena, action, is_goal, scoring_team, include_breakdown)
