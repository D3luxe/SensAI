"""
Reward v9: v8's terms with the boost ratchet T7 replaced by the boost-edge state bonus T8.
docs/reward_v9_boost_spec.md is the specification; every formula here is written out in full there
(v3's in docs/reward_v3_spec.md), and nothing in this file may differ from them.

  T1 goal            +goal_reward to the scorers, -goal_reward * (1 - aggression_bias) to the conceders
  T2 ball_position   potential  (|b - own| - |b - opp|) / 10240,           weight ball_position_weight
  T3 touch           on a touch  min(1, |dv_ball| / 2300),                  weight touch_weight
  T4 closeness       potential  -|b - c| / 12000,                          weight closeness_weight (0 since v4)
  T5 boost           potential  sqrt(boost / 100),                         weight boost_weight (0 since v8)
  T7 boost_gain      on a rise   max(0, sqrt(b'/100) - sqrt(b/100)),        weight boost_gain_weight (0 in v9)
  T8 boost_edge      every step  sqrt(b_self/100) - sqrt(b_opp/100),        weight boost_edge_weight

WHY T7 IS REPLACED RATHER THAN RE-SHAPED.

v8 priced the wrong object. The replay study that motivated it measured boost DIFFERENTIAL -- a
state, how much boost you are holding relative to the opponent -- as the strongest predictor of
scoring next (AUC 0.618, +0.212 per standard deviation in a joint fit). v8 translated that into a
reward for ACQUIRING boost, and the policy did exactly what it was asked:

    at 200M, pooled, v8 against v7 at the same step count
    boost collected / min   216  vs  187      the transition we paid for went up
    boost spent / min       291  vs  242      and so did the spending
    mean tank               9.1  vs  16.3     the state we cared about went DOWN
    empty %                63.2  vs  33.4
    retreat starts low     77.1  vs  46.9     the exact failure T7 was built to fix, 30 points worse

T7 paid the rise and never charged the fall, so a collect -> dump -> collect loop pays every cycle,
and the concave sqrt makes refilling from empty the single most valuable transition. Running the
tank dry is therefore the profitable state to be in. The bot found that loop and committed to it.
The v8 spec named touches_per_min as the canary for farming and it did fire (5.92 vs 6.79), but it
was written for the wrong failure: the fear was a car wandering the map cycling pads, and small-pad
pickups actually FELL (10.9 vs 11.7). It never needed to wander. Dumping boost is free and instant,
so any pickup pays full concave value without going anywhere.

Discounting the fall instead of ignoring it -- w * (max(0, dphi) - k * max(0, -dphi)) for k < 1 --
was considered and rejected. It scales the exploit by (1 - k) without changing its shape: a cycle
still pays in proportion to the phi-amplitude traversed, and under a concave phi that amplitude is
still largest at the bottom of the tank, so the premium on running empty survives. It also cannot
be made unfarmable: at k = 1 it telescopes into a potential and goes inert, which is where v5 was.

T8 prices the state directly, and prices it as a DIFFERENTIAL for one specific reason: the two
cars' contributions sum to exactly zero on every step, so a dense always-on bonus cannot become a
survival bonus. Nothing is gained by extending an episode, because whatever one car banks per step
the other pays. Spending boost now costs immediately; letting the opponent refuel costs too, which
is what boost starving is in real 1v1.

Three properties follow:
  - no cycle pays. There is no trajectory of one's own boost that returns to its starting value and
    has earned anything, because the term reads the level rather than the change.
  - it is not a potential, so it can move the policy. The sum over an episode of w * (phi_self -
    phi_opp) depends on the whole trajectory, not on the endpoints.
  - concavity now says the right thing. With sqrt, the first 30 boost is worth more than the last
    30, so the bot is paid most for not being empty -- a level, not a refill.

ON SIZING, which is the risk this time. T8 is paid EVERY step, unlike T3 (per touch) and T1
(per goal), so its total depends on the step rate: tick_skip 8 at 120 Hz, 15 steps/s, and about
17.4 s (~261 steps) per episode at current goal rates. At boost_edge_weight 0.01 an entire episode
held at full tank against an empty opponent is worth 2.61, about a quarter of a goal; a realistic
sustained edge of 0.2 in phi is worth ~0.52 per episode, the same order as T3's ~0.37. If tick_skip
ever changes, this weight is no longer calibrated -- test_rewards_v9.py pins it at 8.

T1-T7 are v8's and v3's own functions, imported and hashed into v9's identity. The rules of v3 still
hold: no term reads the action (R2), every POTENTIAL term pays w * (gamma * phi(s') - phi(s)) with
phi(s') = 0 on a goal step (R4), and there are no gates (R6). T8 is not a potential, so R4 does not
apply to it and it is paid on the goal step like any other; it is a plain continuous function of the
state, so there is nothing in it to gate.

v9 is frozen once its run starts (config/reward_versions/v9.json, test_reward_versions_frozen.py).
A change is v10, in a new file.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from env.physics_engine import CarState, RocketSimArena
from env.rewards_v3 import (
    REWARD_V3_DEFAULTS, TOUCH_FULL_DV,
    ball_position_potential, boost_potential, closeness_potential,
)

# Code-side defaults; config/reward_versions/v9.json carries the values the run uses, and matches these.
REWARD_V9_DEFAULTS: Dict[str, float] = {
    **REWARD_V3_DEFAULTS,
    "closeness_weight": 0.0,     # retired since v4
    "boost_weight": 0.0,         # T5 retired in v8; kept so the breakdown stays comparable
    "boost_gain_weight": 0.0,    # T7 retired in favour of T8; kept for the same reason
    "boost_edge_weight": 0.01,   # per step, calibrated to tick_skip 8 (15 steps/s)
    "gamma": 0.9977,             # v5's horizon
}

# name -> (potential, weight key). Evaluation order is the breakdown order.
POTENTIAL_TERMS: Dict[str, Tuple[Callable[[CarState, RocketSimArena], float], str]] = {
    "ball_position": (ball_position_potential, "ball_position_weight"),
    "closeness": (closeness_potential, "closeness_weight"),
    "boost": (boost_potential, "boost_weight"),
}
TERM_NAMES = ("goal", "ball_position", "touch", "closeness", "boost", "boost_gain", "boost_edge")


def boost_edge(car: CarState, arena: RocketSimArena) -> float:
    """
    T8: this car's boost potential less the opposing team's mean boost potential.

    Zero-sum over the two cars of a 1v1, which is what stops a dense per-step bonus from paying for
    staying alive. With no opponent on the field there is no edge to price, so the term is 0 rather
    than this car's own level -- that would be a survival bonus.
    """
    opponents = [c for c in arena.cars if c.team != car.team]
    if not opponents:
        return 0.0
    opp_phi = sum(boost_potential(c, arena) for c in opponents) / len(opponents)
    return boost_potential(car, arena) - opp_phi


class RewardV9:
    """The v9 terms for every car of one arena. Per-car state is keyed by car id."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.w: Dict[str, float] = dict(REWARD_V9_DEFAULTS)
        self.update_weights(weights or {})
        self._phi: Dict[Tuple[str, int], float] = {}
        self._boost_phi: Dict[int, float] = {}
        self._touches: Dict[int, int] = {}
        self._ball_vel: Dict[int, np.ndarray] = {}

    def update_weights(self, new_weights: Dict[str, float]):
        """Unknown keys are ignored, so a shared weights dict can carry other versions' keys."""
        for key in REWARD_V9_DEFAULTS:
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

        # T7 boost_gain: v8's ratchet, retired at weight 0. The state is still tracked so the term
        # stays in the breakdown and a later version can price it without changing this file.
        cur_boost = boost_potential(car, arena)
        prev_boost = self._boost_phi.get(car.id, cur_boost)
        self._boost_phi[car.id] = cur_boost
        terms["boost_gain"] = self.w["boost_gain_weight"] * max(0.0, cur_boost - prev_boost)

        # T8 boost_edge: the level, not the change, and relative to the opponent so that the two
        # cars' bonuses cancel every step. Not a potential: no gamma, and not zeroed on a goal step.
        terms["boost_edge"] = self.w["boost_edge_weight"] * boost_edge(car, arena)

        total = float(sum(terms.values()))
        return total, ({n: float(terms[n]) for n in TERM_NAMES} if include_breakdown else {})


class RewardManagerV9:
    """RewardManager's interface over RewardV9, so the env drives any version the same way."""

    version = "v9"

    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        self.reward = RewardV9(reward_weights)

    def reset(self, initial_state: RocketSimArena):
        self.reward.reset(initial_state)

    def update_weights(self, new_weights: Dict[str, float]):
        self.reward.update_weights(new_weights)

    def get_reward(self, car: CarState, arena: RocketSimArena, action: np.ndarray, is_goal: bool,
                   scoring_team: Optional[int], include_breakdown: bool = True) -> Tuple[float, Dict[str, float]]:
        return self.reward.get_reward(car, arena, action, is_goal, scoring_team, include_breakdown)
