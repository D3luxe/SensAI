"""
TrueSkill Bayesian Rating and Evaluation Engine for Rocket League Bots.
Provides fast headless match simulation, golden-goal overtime, team-symmetry pairings,
persistent leaderboard tracking, and visualization.
"""

from __future__ import annotations
import os
import time
import json
import random
import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, Callable, Generator
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import trueskill
except ImportError:
    import math

    class _FallbackRating:
        def __init__(self, mu: float = 25.0, sigma: float = 8.333333):
            self.mu = float(mu)
            self.sigma = float(sigma)

        def __repr__(self):
            return f"Rating(mu={self.mu:.3f}, sigma={self.sigma:.3f})"

    class _FallbackTrueSkill:
        def __init__(
            self,
            mu: float = 25.0,
            sigma: float = 8.333333,
            beta: float = 4.166667,
            tau: float = 0.083333,
            draw_probability: float = 0.05
        ):
            self.mu = mu
            self.sigma = sigma
            self.beta = beta
            self.tau = tau
            self.draw_probability = draw_probability

        def create_rating(self, mu: Optional[float] = None, sigma: Optional[float] = None):
            return _FallbackRating(
                mu=self.mu if mu is None else mu,
                sigma=self.sigma if sigma is None else sigma
            )

        def rate_1vs1(self, r1: _FallbackRating, r2: _FallbackRating, drawn: bool = False):
            beta = self.beta
            c = math.sqrt(2.0 * beta * beta + r1.sigma * r1.sigma + r2.sigma * r2.sigma)

            def phi(x):
                return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

            def Phi(x):
                return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

            p_draw = max(1e-5, min(0.99, self.draw_probability))
            y = p_draw
            a = 0.147
            t_log = math.log(1.0 - y * y)
            erfinv_val = math.copysign(
                math.sqrt(-2.0 / (math.pi * a) - t_log / 2.0 + math.sqrt((2.0 / (math.pi * a) + t_log / 2.0) ** 2 - t_log / a)),
                y
            )
            eps = math.sqrt(2.0) * beta * (erfinv_val * math.sqrt(2.0))

            t = (r1.mu - r2.mu) / c
            if not drawn:
                denom = max(1e-9, Phi(t - eps / c))
                v = phi(t - eps / c) / denom
                w = v * (v + t - eps / c)
            else:
                denom = max(1e-9, Phi(eps / c - t) - Phi(-eps / c - t))
                v = (phi(-eps / c - t) - phi(eps / c - t)) / denom
                w = v * v + ((eps / c - t) * phi(eps / c - t) - (-eps / c - t) * phi(-eps / c - t)) / denom

            new_mu1 = r1.mu + (r1.sigma * r1.sigma / c) * v
            new_mu2 = r2.mu - (r2.sigma * r2.sigma / c) * v
            new_var1 = r1.sigma * r1.sigma * max(1e-6, 1.0 - (r1.sigma * r1.sigma / (c * c)) * w)
            new_var2 = r2.sigma * r2.sigma * max(1e-6, 1.0 - (r2.sigma * r2.sigma / (c * c)) * w)

            return _FallbackRating(new_mu1, math.sqrt(new_var1)), _FallbackRating(new_mu2, math.sqrt(new_var2))

    class _TrueskillModule:
        Rating = _FallbackRating
        TrueSkill = _FallbackTrueSkill

        @staticmethod
        def rate_1vs1(r1, r2, drawn: bool = False, env=None):
            active_env = env if env is not None else ts_env
            return active_env.rate_1vs1(r1, r2, drawn=drawn)

    trueskill = _TrueskillModule()

from env.physics_engine import RocketSimArena
from utils import rating_fit
from env.baseline_agent import (
    BaseOpponent, BaselineChaser, CheckpointOpponentBot, NectoNextoOpponentBot, create_opponent_bot
)

DEFAULT_LEADERBOARD_PATH = "logs/trueskill_leaderboard.json"

# Ranking gate. Sigma decides whether a rating may be ranked at all; mu decides where.
DEFAULT_ELIGIBILITY_SIGMA = 1.5
DEFAULT_MIN_RANKED_MATCHES = 24

# Series after which a converged rating stops moving. Past this point further play makes
# the number worse, not better, and the reason is the schedule rather than the sample.
#
# Against a fixed, well-measured opponent more series is monotonically good: the true
# spread of mu falls 1.30 -> 0.89 between 30 and 64 series, then only 0.89 -> 0.59 across
# the next 336. But an established model does not face a fixed opponent. It faces a
# stream of fresh challengers, each entering at sigma 8.33 and replaced after ~3 series.
# Simulating that schedule with every player's true skill held identical (n=300 runs):
#
#     challengers   series   true spread of mu   reported sigma
#          10          30          0.36               0.97
#          25          75          0.53               0.95
#          50         150          0.61               0.94
#         100         300          0.70               0.94
#
# The rating random-walks with roughly the square root of reigns served, because a fresh
# challenger's enormous uncertainty pushes variance into the established side instead of
# pulling information out of it. Sigma flatlines around 0.94 and stops telling the truth:
# it reports 0.94 while the actual dispersion is 0.70 and still climbing, against an
# elite pool whose entire measured spread is 1.51 mu.
#
# 64 is where the precision curve flattens and before the walk dominates.
DEFAULT_RATING_LOCK_MATCHES = 64

# Match-length defaults, chosen from measurement rather than intuition (n=102 per arm,
# pinned near-peer pairings, decisive results per minute of compute):
#
#   reg 400 / ot 200 fixed    53.9% draws   1.33 s/game   20.7/min   baseline
#   reg 400 / ot 600 random   28.4% draws   1.69 s/game   25.5/min   +23%
#   reg 600 / ot 600 random   16.7% draws   1.92 s/game   26.1/min   +26%
#   reg 600 / ot 200 fixed    51.0% draws   1.73 s/game   17.0/min   -18%
#
# The fourth arm isolates the surprise: longer regulation on its own does essentially
# nothing for draws (51.0% vs 53.9%) while costing 30% more per game. The overtime
# change does all the work. Regulation 600 is then chosen for validity, not throughput
# -- it matches the 600-step training horizon, so the policy is graded on the horizon
# it was optimised for, and it is throughput-neutral once overtime is fixed.
# Per-episode step cap. Episodes end at the first goal, so this is a backstop for a
# stalemate rather than a match clock; it is generous because it is rarely reached.
DEFAULT_EVAL_MAX_STEPS = 3000

# An episode with no ball contact for this many steps is dead and ends level, so two
# passive policies cannot burn the full cap doing nothing.
DEFAULT_NO_TOUCH_STEPS = 500

# Series shape. Best-of-9, first to 5.
DEFAULT_SERIES_LENGTH = 9
DEFAULT_SERIES_WINS_NEEDED = 5

# TrueSkill Global Configuration
# draw_probability is calibrated to the rate actually observed in headless evaluation
# (~24% of games end level even with golden-goal overtime, because two similar policies
# frequently stall). Declaring 0.05 here made every draw a large surprise to the model,
# which dragged mu toward the opponent and shrank sigma faster than the evidence warranted
# -- the mechanism behind the whole veteran population collapsing into a narrow mu band.
# A rated encounter is a best-of-9 series, first to 5, and each episode inside it ends
# at the first goal. A series is decisive by construction: it can only tie if the whole
# nine play out level, which is rare. That is a structural fix for the draw problem
# rather than the parameter tuning it replaces -- fixed-length matches scored on goals
# drew 54% of the time between near-peers, and no amount of clock adjustment got that
# below ~17%.
# Measured, not inherited: 2 of 30 series between near-peer checkpoints ended level
# (6.7%). The reference implementation declares 0.01, but its episodes cannot end
# scoreless the way ours can -- a step cap or a no-touch timeout leaves an episode
# undecided here, which lets nine of them split evenly. Near-peer pairings are also the
# pessimistic case and the ones the league actually schedules.
DEFAULT_DRAW_PROBABILITY = 0.07

# The mpmath backend, not the default one. TrueSkill's default solver raises
# FloatingPointError once the two mu are far enough apart -- measured here, anything past
# a gap of ~229 mu against a sigma-0.5 anchor. That is not hypothetical: the league ran to
# mu 518-521 against anchors at 30-34.7, a gap of ~485, so every anchor grading raised and
# was swallowed by the caller's except. The anchor could not correct the drift precisely
# when the drift was worst, which is what let it keep going.
#
# mpmath computes the same update in arbitrary precision (520 vs 34.7 resolves to a 13.6
# mu correction instead of an exception). It costs 394 us against 65 us per update, which
# is 0.01% of the ~3 s best-of-9 series that produces the update.
_TS_KWARGS = dict(
    mu=25.0,
    sigma=25.0 / 3.0,     # ~8.333
    beta=25.0 / 6.0,      # ~4.167
    tau=25.0 / 300.0,     # ~0.0833
    draw_probability=DEFAULT_DRAW_PROBABILITY,
)
try:
    ts_env = trueskill.TrueSkill(backend="mpmath", **_TS_KWARGS)
except Exception:   # mpmath missing, or the fallback shim is in use
    ts_env = trueskill.TrueSkill(**_TS_KWARGS)

# The one pinned rating, which fixes the scale. See utils/rating_fit.py.
#
# Until 2026-09-21 this was a ladder of pinned bots -- heuristic 15, BC baseline 14.5, Necto 30,
# Nexto 34.7 -- solved against by online TrueSkill. Necto wins every series against this
# league, so pinning it tied the scale to a reference no result could reach, and the
# population drifted to mu 41 while losing to it 35-1. The ladder's values also came from a
# 2026 round robin whose field no longer exists.
#
# Now exactly one rating is pinned: the v3 king, which the league plays close enough that every
# series against it carries information, and which never trains. mu 25 is TrueSkill's default,
# so the number reads as "relative to the v3 king". Necto, Nexto, the heuristic and the BC
# baseline are still reference opponents (is_anchor), but they are fitted like anyone else: their
# results place them on the scale and can no longer move it.
#
# Pin more than one only with values measured against each other in the same fit. Two pins
# that disagree with the data bend every rating between them, which is the failure this
# replaced. (A ladder that pins each new king at its fitted value was simulated too, and
# inflates: the king is chosen for rating high, so its pinned value is a winner's-curse number.)
ANCHOR_CALIBRATION: Dict[str, float] = {
    "v3_iter198000": 25.0,
}
# Display only: a pinned rating is exact in the fit. Kept non-zero because callers divide by
# and subtract multiples of sigma.
ANCHOR_SIGMA = 0.5


def get_anchor_calibration(model_spec: str) -> Optional[Tuple[float, float]]:
    """Returns the (mu, sigma) this rating is pinned at, or None if it is fitted."""
    key = os.path.basename(str(model_spec).strip().strip('"').strip("'")).lower()
    if str(model_spec).strip().lower() in ("heuristic", "baseline", "baselinechaser"):
        key = "heuristic"
    for name, mu in ANCHOR_CALIBRATION.items():
        if name in key:
            return mu, ANCHOR_SIGMA
    return None


@dataclass
class ModelRating:
    """TrueSkill rating and match statistics for a model or bot."""
    name: str
    path: str
    mu: float = 25.0
    sigma: float = 8.333333
    conservative_rating: float = 0.0  # mu - 3 * sigma
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    goals_for: int = 0
    goals_against: int = 0
    goal_diff: int = 0
    win_rate: float = 0.0
    is_anchor: bool = False
    # Locked ratings keep playing but stop absorbing updates -- a checkpoint that has
    # earned the same treatment the calibrated anchors get. Recorded on the rating rather
    # than recomputed from config, so raising or lowering the cap later cannot silently
    # re-open a model that was already frozen, and the leaderboard stays auditable.
    rating_locked: bool = False
    locked_at_matches: int = 0
    # Calibration series -- those played against a fixed anchor purely to bind the scale.
    # Counted separately because the anchors are saturated against this population (Necto
    # takes ~93% of points off any checkpoint we have), so folding them into points_rate
    # puts the promotion floor out of reach however good a contender is. That is the bug
    # that got the anchors dropped from grading in the first place, and dropping them is
    # what let mu drift to 520. Separating the two lets an anchor move mu without touching
    # the gate. They are still in wins/losses/matches_played, which are the raw record.
    calibration_matches: int = 0
    calibration_wins: int = 0
    calibration_draws: int = 0
    # Centre of this rating's prior in the batch fit: the predecessor's mu for a seeded
    # checkpoint, the default otherwise. Only matters until the data outweigh it.
    prior_mu: float = 25.0
    last_updated: str = ""

    def update_conservative(self):
        self.conservative_rating = round(self.mu - 3.0 * self.sigma, 2)
        self.goal_diff = self.goals_for - self.goals_against
        self.win_rate = round((self.wins / max(1, self.matches_played)) * 100.0, 1)

    @property
    def points_rate(self) -> float:
        """
        Percentage of available points taken, scoring a draw as half a win.

        Raw win_rate is the wrong gate for this environment: draws are ~24% of games, so
        an undefeated 1W-0L-3D contender scores 25% on win_rate and fails a 45-50% floor
        despite never having lost. Points rate is the standard fix and is what promotion,
        demotion and pool tiebreaks should use.
        """
        played = self.matches_played - self.calibration_matches
        wins = self.wins - self.calibration_wins
        draws = self.draws - self.calibration_draws
        if played <= 0:     # nothing but calibration so far: no peer evidence either way
            return 0.0
        return round(((wins + 0.5 * draws) / played) * 100.0, 1)

    def to_trueskill_rating(self) -> trueskill.Rating:
        return ts_env.create_rating(mu=self.mu, sigma=self.sigma)

    def from_trueskill_rating(self, rating: trueskill.Rating):
        self.mu = round(float(rating.mu), 3)
        self.sigma = round(float(rating.sigma), 3)
        self.last_updated = datetime.datetime.now().isoformat()
        self.update_conservative()


def get_model_display_name(model_spec: str) -> str:
    """Generates a clean human-readable name from a model path or identifier."""
    clean = model_spec.strip().strip('"').strip("'")
    if clean.lower() in ("heuristic", "baseline", "baselinechaser", "none"):
        return "Baseline Chaser (Heuristic)"
    base = os.path.basename(clean)
    if "latest_model" in base:
        return "Latest Policy (latest_model.pt)"
    if "pretrained_baseline" in base:
        return "Pretrained Baseline (BC)"
    if "necto" in base:
        return "Necto (EARL TorchScript)"
    if "nexto" in base:
        return "Nexto (EARL TorchScript)"
    if base.endswith(".pt"):
        return base[:-3]
    return base


def simulate_headless_episode(
    blue_bot: BaseOpponent,
    orange_bot: BaseOpponent,
    max_steps: int = DEFAULT_EVAL_MAX_STEPS,
    no_touch_steps: int = DEFAULT_NO_TOUCH_STEPS,
    dt: float = 1.0 / 15.0,
    arena: Optional[RocketSimArena] = None
) -> Dict[str, Any]:
    """
    One episode, ending at the first goal.

    A fixed-length match keeps simulating after the outcome is effectively settled and
    then reports a draw whenever the goals happen to be level. Ending at the first goal
    spends the compute only up to the decision, and hands the series a clean win or loss.

    Returns result 1 (blue), -1 (orange) or 0 (no goal: stalemate or no-touch timeout).
    """
    if arena is None:
        arena = RocketSimArena(num_players=2, game_mode="1v1")

    # random_kickoff=False selects the KickoffSetter, which picks one of five standard
    # spawn locations at random and mirrors it exactly for the other team. Randomised
    # for variety, symmetric so neither side is handed an advantage.
    arena.reset(random_kickoff=False)

    bot_mask = [
        isinstance(blue_bot, NectoNextoOpponentBot),
        isinstance(orange_bot, NectoNextoOpponentBot)
    ]

    prev_touches = sum(c.ball_touches for c in arena.cars)
    last_touch_step = 0
    result = 0
    steps = 0

    for step in range(max_steps):
        steps = step + 1
        act0 = blue_bot.get_action(arena.cars[0], arena)
        act1 = orange_bot.get_action(arena.cars[1], arena)
        goal, scoring_team = arena.step([act0, act1], dt=dt, bot_mask=bot_mask)

        touches = sum(c.ball_touches for c in arena.cars)
        if touches != prev_touches:
            prev_touches = touches
            last_touch_step = step

        if goal:
            result = 1 if scoring_team == 0 else -1
            break

        if step - last_touch_step >= no_touch_steps:
            break

    return {
        "result": result,
        "steps": steps,
        "blue_touches": arena.cars[0].ball_touches,
        "orange_touches": arena.cars[1].ball_touches,
        "no_touch_timeout": (steps - last_touch_step) >= no_touch_steps and result == 0,
    }


def simulate_headless_series(
    bot_a: BaseOpponent,
    bot_b: BaseOpponent,
    series_length: int = DEFAULT_SERIES_LENGTH,
    wins_needed: int = DEFAULT_SERIES_WINS_NEEDED,
    max_steps: int = DEFAULT_EVAL_MAX_STEPS,
    no_touch_steps: int = DEFAULT_NO_TOUCH_STEPS,
    dt: float = 1.0 / 15.0
) -> Dict[str, Any]:
    """
    A best-of-N series between two bots, stopping as soon as one reaches wins_needed.

    Sides alternate every episode, so a series is symmetric even when the arena or the
    policies carry a side preference. This is the unit that produces exactly one
    TrueSkill update, which is what makes that update decisive and worth its cost.
    """
    arena = RocketSimArena(num_players=2, game_mode="1v1")
    a_score = b_score = 0
    episodes = []

    for i in range(series_length):
        a_is_blue = (i % 2 == 0)
        blue, orange = (bot_a, bot_b) if a_is_blue else (bot_b, bot_a)
        ep = simulate_headless_episode(
            blue_bot=blue, orange_bot=orange, max_steps=max_steps,
            no_touch_steps=no_touch_steps, dt=dt, arena=arena
        )
        raw = ep["result"]
        # Translate blue/orange back into a/b.
        if raw == 0:
            outcome = 0
        elif (raw == 1) == a_is_blue:
            outcome = 1
        else:
            outcome = -1

        if outcome > 0:
            a_score += 1
        elif outcome < 0:
            b_score += 1
        ep["outcome_for_a"] = outcome
        episodes.append(ep)

        if a_score >= wins_needed or b_score >= wins_needed:
            break

    score_diff = a_score - b_score
    return {
        "a_score": a_score,
        "b_score": b_score,
        "score_diff": score_diff,
        "winner": "a" if score_diff > 0 else ("b" if score_diff < 0 else "draw"),
        "episodes_played": len(episodes),
        "total_steps": sum(e["steps"] for e in episodes),
        "episodes": episodes,
    }


class TrueSkillEvaluator:
    """
    Manages TrueSkill ratings, tournament matchmaking, persistence, and reporting.
    """
    # Ranking gate, mirrored from the league config so the standings table and the
    # LeagueManager's King/Elite Pool selection cannot disagree about who outranks whom.
    eligibility_sigma: float = DEFAULT_ELIGIBILITY_SIGMA
    min_ranked_matches: int = DEFAULT_MIN_RANKED_MATCHES
    rating_lock_matches: int = DEFAULT_RATING_LOCK_MATCHES

    def is_rating_frozen(self, rec: ModelRating) -> bool:
        """
        Whether this rating is pinned rather than fitted: only the ANCHOR_CALIBRATION entries.

        Reference bots (is_anchor) are fitted now, and the lock is retired, so every other
        rating is re-solved from the whole results log after each pairing.
        """
        return get_anchor_calibration(rec.path) is not None

    def maybe_lock_rating(self, rec: ModelRating) -> bool:
        """
        Retired 2026-09-21; never locks. Kept so older callers and saved records still load.

        The lock stopped an established rating random-walking under a stream of sigma-8
        newcomers in the online update. The batch fit weighs every result once against all
        the others, so there is no walk to stop, and a frozen rating is exactly what let the
        v5 league ratchet away from its anchor.
        """
        return False

    def is_rank_eligible(self, rec: ModelRating) -> bool:
        """
        Whether a rating has converged enough to be ranked on raw mu against peers.

        Anchors are always eligible. Their mu is declared by ANCHOR_CALIBRATION and their
        sigma is pinned, so there is nothing to converge; holding them to a match count
        they can never satisfy would sort a calibrated reference below every checkpoint
        that cleared the gate.
        """
        if rec.is_anchor:
            return True
        return rec.sigma <= self.eligibility_sigma and rec.matches_played >= self.min_ranked_matches

    def ranking_key(self, rec: ModelRating) -> Tuple[int, float, float]:
        """Descending sort key: eligible models by mu, provisional ones by mu - 3*sigma."""
        if self.is_rank_eligible(rec):
            return (1, rec.mu, rec.points_rate)
        return (0, rec.conservative_rating, rec.points_rate)

    def __init__(self, leaderboard_path: str = DEFAULT_LEADERBOARD_PATH,
                 results_path: Optional[str] = None):
        self.leaderboard_path = leaderboard_path
        # The results log is named after its leaderboard, so a test's scratch board gets a
        # scratch log rather than appending to the live one.
        self.results_path = results_path or (os.path.splitext(leaderboard_path)[0] + ".results.jsonl")
        self.ratings: Dict[str, ModelRating] = {}
        self.match_history: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = rating_fit.load_results(self.results_path)
        self.load_leaderboard()

    def get_or_create_rating(self, model_spec: str, is_anchor: bool = False) -> ModelRating:
        """Retrieves or creates a rating record for a given model path or identifier."""
        norm_key = os.path.normpath(model_spec.strip().strip('"').strip("'")).replace("\\", "/")
        name = get_model_display_name(norm_key)

        # Exact key first. The name and basename fallbacks exist so "x.pt" and "checkpoints/x.pt"
        # reach one record, but they must not join two different files: checkpoint numbering
        # restarts from each run's start checkpoint, so a new run's checkpoint_iter_222200.pt
        # shares its name with the archived one of the run before, and would otherwise inherit
        # that checkpoint's record and have its own games scored against it.
        if norm_key in self.ratings:
            return self.ratings[norm_key]
        for k, r in self.ratings.items():
            if r.name == name or os.path.basename(k) == os.path.basename(norm_key):
                if (os.path.exists(k) and os.path.exists(norm_key)
                        and os.path.abspath(k) != os.path.abspath(norm_key)):
                    continue
                return r

        # A pinned rating is a reference whoever registers it first.
        calib = get_anchor_calibration(norm_key)
        is_anchor = is_anchor or calib is not None
        record = ModelRating(
            name=name,
            path=norm_key,
            mu=calib[0] if calib else 25.0,
            sigma=calib[1] if calib else 25.0 / 3.0,
            is_anchor=is_anchor,
            last_updated=datetime.datetime.now().isoformat()
        )
        record.update_conservative()
        self.ratings[norm_key] = record
        return record

    def apply_anchor_calibration(self) -> int:
        """
        Re-pins every anchor record to its calibrated (mu, sigma) from ANCHOR_CALIBRATION.

        Anchors are fixed reference points, so their rating is a declaration rather than
        something learned. Re-applying on every load keeps the scale stable even if an
        older leaderboard on disk carries the uncalibrated values, and makes the ladder a
        single source of truth that editing the table is enough to change.
        """
        changed = 0
        for rec in self.ratings.values():
            calib = get_anchor_calibration(rec.path) or get_anchor_calibration(rec.name)
            if not calib:
                continue
            rec.is_anchor = True
            if abs(rec.mu - calib[0]) > 1e-6 or abs(rec.sigma - calib[1]) > 1e-6:
                rec.mu, rec.sigma = calib
                rec.update_conservative()
                changed += 1
        return changed

    def load_leaderboard(self, path: Optional[str] = None):
        """Loads persistent ratings and match history from JSON."""
        target_path = path or self.leaderboard_path
        if not os.path.exists(target_path):
            return

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.ratings.clear()
            for k, v in data.get("ratings", {}).items():
                v = {k2: v2 for k2, v2 in v.items() if k2 in ModelRating.__dataclass_fields__}
                record = ModelRating(**v)
                record.rating_locked = False      # the lock is retired; see maybe_lock_rating
                record.update_conservative()
                self.ratings[k] = record
            self.match_history = data.get("history", [])
            self.apply_anchor_calibration()
        except Exception as e:
            print(f"[TrueSkill] Warning: Could not load leaderboard from {target_path}: {e}")

    def save_leaderboard(self, path: Optional[str] = None):
        """Saves ratings and history to JSON atomically using a temporary file."""
        target_path = path or self.leaderboard_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        tmp_path = target_path + ".tmp"

        payload = {
            "version": "1.0",
            "last_updated": datetime.datetime.now().isoformat(),
            "ratings": {k: asdict(v) for k, v in self.ratings.items()},
            "history": self.match_history[-500:]  # Keep last 500 matches
        }

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, target_path)
        except Exception as e:
            print(f"[TrueSkill] Error saving leaderboard to {target_path}: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def evaluate_pairing(
        self,
        model_a_path: str,
        model_b_path: str,
        series_per_pair: int = 1,
        max_steps: int = DEFAULT_EVAL_MAX_STEPS,
        no_touch_steps: int = DEFAULT_NO_TOUCH_STEPS,
        series_length: int = DEFAULT_SERIES_LENGTH,
        wins_needed: int = DEFAULT_SERIES_WINS_NEEDED,
        device: str = "cpu",
        calibration: bool = False,
        **legacy
    ) -> List[Dict[str, Any]]:
        """
        Plays `series_per_pair` best-of-N series and applies ONE rating update per series.

        Rating a whole series rather than each episode is the point: a series outcome is
        decisive and carries far more information than a single short episode, so each
        update moves the rating on real evidence instead of on one lucky goal. Sides
        alternate inside the series, so no side symmetry handling is needed here.

        `matches_per_pair` is accepted as a legacy alias and interpreted as a series
        count, so older callers keep working rather than silently playing 9x the games.
        """
        # Retired kwargs from the fixed-length-match era. A series ends at the first goal
        # and cannot tie except across all nine episodes, so overtime has no meaning now.
        for retired in ("enable_overtime", "max_ot_steps", "ot_random_kickoff"):
            legacy.pop(retired, None)
        if "matches_per_pair" in legacy and legacy["matches_per_pair"] is not None:
            # Old callers passed game counts; treat two games as one series, minimum one.
            series_per_pair = max(1, int(legacy["matches_per_pair"]) // 2)
            legacy.pop("matches_per_pair")
        if legacy:
            raise TypeError(f"evaluate_pairing got unexpected keyword(s): {sorted(legacy)}")
        series_per_pair = max(1, int(series_per_pair))

        record_a = self.get_or_create_rating(model_a_path)
        record_b = self.get_or_create_rating(model_b_path)

        bot_a = create_opponent_bot(model_a_path, device=device)
        bot_b = create_opponent_bot(model_b_path, device=device)

        results = []
        for _ in range(series_per_pair):
            res = simulate_headless_series(
                bot_a=bot_a, bot_b=bot_b,
                series_length=series_length, wins_needed=wins_needed,
                max_steps=max_steps, no_touch_steps=no_touch_steps
            )

            record_a.matches_played += 1
            record_b.matches_played += 1
            record_a.goals_for += res["a_score"]
            record_a.goals_against += res["b_score"]
            record_b.goals_for += res["b_score"]
            record_b.goals_against += res["a_score"]

            drawn = (res["score_diff"] == 0)

            # The win/loss tallies are the raw record; the rating comes from the fit below.
            if drawn:
                record_a.draws += 1
                record_b.draws += 1
            elif res["score_diff"] > 0:
                record_a.wins += 1
                record_b.losses += 1
            else:
                record_b.wins += 1
                record_a.losses += 1

            # Calibration series are tallied apart so the promotion gate stays a peer-only
            # judgement. In the fit they are ordinary results: that is what ties the scale.
            if calibration:
                for rec in (record_a, record_b):
                    if self.is_rating_frozen(rec):
                        continue
                    rec.calibration_matches += 1
                    if drawn:
                        rec.calibration_draws += 1
                    elif (rec is record_a) == (res["score_diff"] > 0):
                        rec.calibration_wins += 1

            self.record_series(record_a.path, record_b.path, res,
                               kind="calibration" if calibration else "league", refit=False)

            res["model_a"] = record_a.name
            res["model_b"] = record_b.name
            res["a_mu"] = record_a.mu
            res["b_mu"] = record_b.mu
            res["winner_name"] = (
                record_a.name if res["score_diff"] > 0
                else (record_b.name if res["score_diff"] < 0 else "Draw")
            )
            results.append(res)
            self.match_history.append(res)

        # One solve per pairing, after its series: every rating on the board can move,
        # because a result between A and B is also evidence about everyone they have met.
        self.refit()
        for res in results:
            res["a_mu"] = record_a.mu
            res["b_mu"] = record_b.mu
        self.save_leaderboard()
        return results

    def record_series(self, a_path: str, b_path: str, res: Dict[str, Any],
                      kind: str = "league", refit: bool = True) -> None:
        """
        Adds one played series to the results log, and by default re-solves the ratings.

        Everything that plays a series should come through here, including the benchmark
        runs against Necto and Nexto, which touch no tallies but are data the fit can use.
        """
        a = self._key(a_path)
        b = self._key(b_path)
        self.results.append(rating_fit.append_result(
            self.results_path, a, b, int(res.get("a_score", 0)), int(res.get("b_score", 0)),
            kind, datetime.datetime.now().isoformat()))
        if refit:
            self.refit()

    @staticmethod
    def _key(model_spec: str) -> str:
        return os.path.normpath(str(model_spec).strip().strip('"').strip("'")).replace("\\", "/")

    def refit(self) -> None:
        """
        Re-solves every rating from the whole results log; see utils/rating_fit.py.

        Pinned ratings are held, everyone else who has played is fitted, and a record that
        has never played keeps whatever it was created or seeded with.
        """
        if not self.results:
            return
        by_key = {self._key(k): r for k, r in self.ratings.items()}
        pinned = {}
        for k, r in by_key.items():
            calib = get_anchor_calibration(r.path)
            if calib:
                pinned[k] = calib[0]
        fitted = rating_fit.fit_ratings(
            self.results, pinned,
            prior_mu={k: r.prior_mu for k, r in by_key.items()},
            start={k: r.mu for k, r in by_key.items()},
        )
        now = datetime.datetime.now().isoformat()
        for k, (mu, sd) in fitted.items():
            r = by_key.get(k)
            if r is None or k in pinned:
                continue
            r.mu = round(mu, 3)
            r.sigma = round(sd, 3)
            r.last_updated = now
        for r in self.ratings.values():
            r.update_conservative()

    def run_tournament(
        self,
        model_paths: List[str],
        series_per_pair: int = 1,
        max_steps: int = DEFAULT_EVAL_MAX_STEPS,
        no_touch_steps: int = DEFAULT_NO_TOUCH_STEPS,
        device: str = "cpu",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Executes a round-robin tournament across all provided models.
        Yields status dict after each pairing for real-time progress streaming in CLI and Gradio.
        """
        # Deduplicate models
        models = list(dict.fromkeys([os.path.normpath(p).replace("\\", "/") for p in model_paths if p]))
        if len(models) < 2:
            return

        # Generate all unique combinations
        pairings: List[Tuple[str, str]] = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                pairings.append((models[i], models[j]))

        # Shuffle pairings to eliminate path-dependent ordering bias
        random.shuffle(pairings)
        total_pairings = len(pairings)

        for idx, (mA, mB) in enumerate(pairings):
            status_msg = f"Matchup {idx+1}/{total_pairings}: {get_model_display_name(mA)} vs {get_model_display_name(mB)}"
            if progress_callback:
                progress_callback(idx, total_pairings, status_msg)

            results = self.evaluate_pairing(
                model_a_path=mA,
                model_b_path=mB,
                series_per_pair=series_per_pair,
                max_steps=max_steps,
                no_touch_steps=no_touch_steps,
                device=device
            )

            yield {
                "pairing_index": idx + 1,
                "total_pairings": total_pairings,
                "model_a": get_model_display_name(mA),
                "model_b": get_model_display_name(mB),
                "results": results,
                "leaderboard_df": self.get_leaderboard_dataframe()
            }

    def get_leaderboard_dataframe(self, ascii_safe: bool = False, include_anchors: bool = False) -> pd.DataFrame:
        """
        Ranked standings of the checkpoints.

        Anchors are excluded by default. They are fixed reference points whose mu is
        declared rather than earned, so ranking them alongside checkpoints invites the
        reading that a checkpoint is "beating Nexto" when it has merely been assigned a
        higher number. Their values belong in a legend, not in the table.
        """
        mu_col = "Rating (mu)" if ascii_safe else "Rating (μ)"
        sigma_col = "Uncertainty (sigma)" if ascii_safe else "Uncertainty (σ)"
        plus_minus = "+/-" if ascii_safe else "±"

        necto_col = "vs Necto"
        cols = [
            "Rank", "Model", "Confidence", mu_col, necto_col, sigma_col,
            "Conservative Score", "Points", "Win Rate", "Record (W-L-D)", "Goal Diff",
            "Matches"
        ]

        if not self.ratings:
            return pd.DataFrame(columns=cols)

        # Rank-eligible models first, ordered by raw mu; provisional models after,
        # ordered by their conservative lower bound. Sorting the table by mu - 3*sigma
        # would contradict the LeagueManager, which no longer penalises a converged
        # rating for the sample size that earned it.
        # Ephemeral moving files like latest_model.pt are excluded from ranked TrueSkill standings
        records = [
            r for r in self.ratings.values()
            if "latest_model" not in r.path.lower() and "latest_model" not in r.name.lower()
            and (include_anchors or not r.is_anchor)
        ]
        records.sort(key=lambda r: (self.ranking_key(r), r.matches_played), reverse=True)

        rows = []
        for rank, r in enumerate(records, start=1):
            rows.append({
                "Rank": f"#{rank}",
                "Model": r.name,
                "Confidence": "Ranked" if self.is_rank_eligible(r) else "Provisional",
                mu_col: f"{r.mu:.2f}",
                necto_col: ("n/a" if self.vs_necto(r) is None else f"{self.vs_necto(r):+.1f}"),
                sigma_col: f"{plus_minus}{r.sigma:.2f}",
                "Conservative Score": f"{r.conservative_rating:.2f}",
                "Points": f"{r.points_rate:.1f}%",
                "Win Rate": f"{r.win_rate:.1f}%",
                "Record (W-L-D)": f"{r.wins}-{r.losses}-{r.draws}",
                "Goal Diff": f"{r.goal_diff:+d}",
                "Matches": r.matches_played
            })

        return pd.DataFrame(rows)

    def necto_reference_mu(self) -> Optional[float]:
        """The Necto anchor's mu, or None if it is not on the board."""
        for r in self.ratings.values():
            if r.is_anchor and "necto" in os.path.basename(r.path).lower()                     and "nexto" not in os.path.basename(r.path).lower():
                return r.mu
        return None

    def vs_necto(self, rec: ModelRating) -> Optional[float]:
        """
        A rating's mu relative to Necto's, which is the only number on this board with a
        fixed meaning.

        Raw mu is relative to whoever a model actually played, and this league plays
        mostly itself, so mu drifts with the population and is not comparable across
        eras. Reporting it alone is how the board came to show checkpoints at mu 518
        against an anchored Necto at 30 while those same checkpoints lost to Necto 35-1.
        Anchored to Necto the number stays interpretable: negative means the eval suite
        should show us losing to it, and it cannot drift without the anchor drifting.
        """
        ref = self.necto_reference_mu()
        return None if ref is None else rec.mu - ref

    def get_anchor_ratings(self) -> List[ModelRating]:
        """The calibrated reference ladder, strongest first, for the legend."""
        anchors = [r for r in self.ratings.values() if r.is_anchor]
        anchors.sort(key=lambda r: r.mu, reverse=True)
        return anchors

    def render_leaderboard_plot(self, max_models: int = 25) -> plt.Figure:
        """
        Generates a clean, dark-themed horizontal bar chart showing TrueSkill ratings
        with ±2σ (95% confidence interval) error bars.
        Caps display to top-performing models (and anchors) to ensure readability
        and prevent exceeding browser/WebP 16,383px height constraints.
        """
        if not self.ratings:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
            fig.patch.set_facecolor("#1a202c")
            ax.set_facecolor("#2d3748")
            ax.text(0.5, 0.5, "No TrueSkill evaluation matches recorded yet.",
                    color="#e2e8f0", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return fig

        # Sort descending first to select the top models
        records = [
            r for r in self.ratings.values()
            if "latest_model" not in r.path.lower() and "latest_model" not in r.name.lower()
        ]
        records.sort(
            key=lambda r: (r.conservative_rating, r.win_rate, r.mu),
            reverse=True
        )

        # Select top models and ensure active anchors are included
        top_candidates = records[:max_models]
        anchors = [r for r in records if r.is_anchor and r not in top_candidates]
        selected_records = top_candidates + anchors

        # Sort ascending for horizontal bar chart (highest at top)
        selected_records.sort(
            key=lambda r: (r.conservative_rating, r.win_rate, r.mu),
            reverse=False
        )

        names = [r.name for r in selected_records]
        mus = [r.mu for r in selected_records]
        # 2 * sigma error bars represent the 95% Bayesian confidence interval
        sigmas = [2.0 * r.sigma for r in selected_records]

        fig_height = min(14.0, max(4.5, 0.42 * len(selected_records)))
        fig, ax = plt.subplots(figsize=(10, fig_height), dpi=100)
        fig.patch.set_facecolor("#1a202c")
        ax.set_facecolor("#2d3748")

        y_pos = np.arange(len(selected_records))

        # Color gradient: top models blue/cyan, baseline/heuristic orange
        colors = ["#4299e1" if not r.is_anchor else "#ed8936" for r in selected_records]

        bars = ax.barh(y_pos, mus, xerr=sigmas, height=0.55, color=colors,
                       alpha=0.85, edgecolor="#bee3f8", capsize=4, error_kw={"ecolor": "#cbd5e0", "linewidth": 1.5})

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, color="#e2e8f0", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#cbd5e0")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#4a5568")
        ax.spines["bottom"].set_color("#4a5568")
        ax.grid(axis="x", linestyle=":", alpha=0.3, color="#718096")
        ax.axvline(25.0, color="#ecc94b", linestyle="--", linewidth=1.2, alpha=0.7, label="Baseline Starting μ (25.0)")

        ax.set_xlabel("TrueSkill Rating (μ ± 2σ Confidence Interval)", color="#e2e8f0", fontsize=11, fontweight="bold")
        title_suffix = f"(Top {len(selected_records)} Standings)" if len(records) > max_models else ""
        ax.set_title(f"SensAI TrueSkill Leaderboard {title_suffix}", color="white", fontsize=13, fontweight="bold", pad=12)

        # Value annotations
        for idx, r in enumerate(selected_records):
            text = f" μ={r.mu:.1f} (score: {r.conservative_rating:.1f}) [{r.wins}W-{r.losses}L]"
            ax.annotate(text, (r.mu + 2.0 * r.sigma + 0.5, idx),
                        color="#bee3f8", fontsize=8, va="center", ha="left")

        ax.legend(loc="lower right", facecolor="#1a202c", edgecolor="#4a5568", labelcolor="white")
        plt.tight_layout()
        return fig

    def reset_leaderboard(self):
        """Wipes all ratings, match history, and league promotion state."""
        self.ratings.clear()
        self.match_history.clear()
        if os.path.exists(self.leaderboard_path):
            try:
                os.remove(self.leaderboard_path)
            except Exception:
                pass
        # Clean up corresponding league_state.json if it exists
        league_state_file = os.path.join(os.path.dirname(self.leaderboard_path), "league_state.json")
        if os.path.exists(league_state_file):
            try:
                os.remove(league_state_file)
            except Exception:
                pass
