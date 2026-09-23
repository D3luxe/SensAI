"""
Reward v8: v5's terms with T5 boost turned from a potential into a ratchet.
docs/reward_v8_boost_spec.md is the specification; every formula here is written out in full there
(v3's in docs/reward_v3_spec.md), and nothing in this file may differ from them.

  T1 goal            +goal_reward to the scorers, -goal_reward * (1 - aggression_bias) to the conceders
  T2 ball_position   potential  (|b - own| - |b - opp|) / 10240,           weight ball_position_weight
  T3 touch           on a touch  min(1, |dv_ball| / 2300),                  weight touch_weight
  T4 closeness       potential  -|b - c| / 12000,                          weight closeness_weight (0 since v4)
  T5 boost           potential  sqrt(boost / 100),                         weight boost_weight (0 in v8)
  T7 boost_gain      on a rise   max(0, sqrt(b'/100) - sqrt(b/100)),        weight boost_gain_weight

WHY T5 IS REPLACED RATHER THAN RE-WEIGHTED.

Potential-based shaping telescopes: over an episode ending in a goal, where phi(terminal) = 0, a
potential term's total discounted contribution is exactly -w * phi(s0) -- a constant fixed by the
start state and independent of the policy. That is what makes a potential unfarmable, and it is
also a proof that T5 can never create an incentive to collect boost. It only ever moved credit
around. Measured: big-pad pickups sat at ~0.5/min across the whole of v5, v6 and v7 regardless of
what else changed, while humans in the replay pool take 4.1/min and collect 411 boost/min against
the bot's 173.

T7 pays the POSITIVE part of the same quantity and does not charge the negative part. That one
asymmetry is the whole change, and it is what makes T7 non-telescoping and therefore able to move
the policy at all. Because it is not a potential it carries no gamma: it prices a transition, not
a difference of state values.

Three properties follow from reusing T5's concave sqrt rather than raw boost:
  - path independence. sqrt telescopes across a monotonic rise, so 0 -> 100 pays exactly 1.0
    whether it is taken in one big pad or eight small ones. Nothing is gained by splitting a
    pickup, and there is no incentive to nibble.
  - refuelling from empty is what pays. 0 -> 30 pays 0.55; 70 -> 100 pays 0.16. The bot is paid
    for the pickup that ends a low-boost retreat, which is the failure the eval actually records
    (retreat_starts_low_boost_pct ~60%).
  - it is bounded per refill. A full tank is worth boost_gain_weight and no more.

ON FARMING, which is the real risk and is deliberately not suppressed by a gate. Spending is free,
so collect -> dump -> collect pays every cycle. Three things bound it: pads respawn on a timer, a
car that is cycling pads is not near the ball, and the weight is sized so a full refill is worth
2.0 against a goal's 10.0 and an average episode's goal term of about -6.4. If the bots farm,
touches_per_min falls -- that is the canary named in the spec, not a formula that prevents it.

T1-T5 are v3's own functions, imported from env/rewards_v3.py (frozen, and hashed into v8's
identity too). The rules of v3 still hold: no term reads the action (R2), every POTENTIAL term
pays w * (gamma * phi(s') - phi(s)) with phi(s') = 0 on a goal step (R4), and there are no gates
(R6). T7 is not a potential, so R4 does not apply to it; max(0, .) is the term's definition, not a
gate on some other quantity.

v8 is frozen once its run starts (config/reward_versions/v8.json, test_reward_versions_frozen.py).
A change is v9, in a new file.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from env.physics_engine import CarState, RocketSimArena
from env.rewards_v3 import (
    REWARD_V3_DEFAULTS, TOUCH_FULL_DV,
    ball_position_potential, boost_potential, closeness_potential,
)

# Code-side defaults; config/reward_versions/v8.json carries the values the run uses, and matches these.
REWARD_V8_DEFAULTS: Dict[str, float] = {
    **REWARD_V3_DEFAULTS,
    "closeness_weight": 0.0,    # retired since v4
    "boost_weight": 0.0,        # T5 retired in favour of T7; kept so the breakdown stays comparable
    "boost_gain_weight": 2.0,
    "gamma": 0.9977,            # v5's horizon
}

# name -> (potential, weight key). Evaluation order is the breakdown order.
POTENTIAL_TERMS: Dict[str, Tuple[Callable[[CarState, RocketSimArena], float], str]] = {
    "ball_position": (ball_position_potential, "ball_position_weight"),
    "closeness": (closeness_potential, "closeness_weight"),
    "boost": (boost_potential, "boost_weight"),
}
TERM_NAMES = ("goal", "ball_position", "touch", "closeness", "boost", "boost_gain")


class RewardV8:
    """The v8 terms for every car of one arena. Per-car state is keyed by car id."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.w: Dict[str, float] = dict(REWARD_V8_DEFAULTS)
        self.update_weights(weights or {})
        self._phi: Dict[Tuple[str, int], float] = {}
        self._boost_phi: Dict[int, float] = {}
        self._touches: Dict[int, int] = {}
        self._ball_vel: Dict[int, np.ndarray] = {}

    def update_weights(self, new_weights: Dict[str, float]):
        """Unknown keys are ignored, so a shared weights dict can carry other versions' keys."""
        for key in REWARD_V8_DEFAULTS:
            if key in new_weights:
                self.w[key] = float(new_weights[key])

    def reset(self, initial_state: RocketSimArena):
        self._phi.clear()
        self._boost_phi.clear()
        for car in initial_state.cars:
            for name, (phi, _) in POTENTIAL_TERMS.items():
                self._phi[(name, car.id)] = phi(car, initial_state)
            # T7 measures from the start state, so the first step cannot bank a phantom pickup.
            self._boost_phi[car.id] = boost_potential(car, initial_state)
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

        # T7 boost_gain: the rise in T5's potential, paid without the fall. Not a potential, so it
        # carries no gamma and is not zeroed on a goal step -- a pickup on that step is a real one.
        cur_boost = boost_potential(car, arena)
        prev_boost = self._boost_phi.get(car.id, cur_boost)
        self._boost_phi[car.id] = cur_boost
        terms["boost_gain"] = self.w["boost_gain_weight"] * max(0.0, cur_boost - prev_boost)

        total = float(sum(terms.values()))
        return total, ({n: float(terms[n]) for n in TERM_NAMES} if include_breakdown else {})


class RewardManagerV8:
    """RewardManager's interface over RewardV8, so the env drives any version the same way."""

    version = "v8"

    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        self.reward = RewardV8(reward_weights)

    def reset(self, initial_state: RocketSimArena):
        self.reward.reset(initial_state)

    def update_weights(self, new_weights: Dict[str, float]):
        self.reward.update_weights(new_weights)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool,
                   scoring_team: Optional[int], include_breakdown: bool = True) -> Tuple[float, Dict[str, float]]:
        return self.reward.get_reward(car, arena, action, is_goal, scoring_team, include_breakdown)
