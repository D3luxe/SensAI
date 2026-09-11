"""
League Manager for SenseiBot.
Manages automated Bayesian grading of saved policy checkpoints, King-of-the-Hill tracking,
elite pool curation, and stratified vectorized environment distribution (Option B).
"""

from __future__ import annotations
import re
import os
import math
import random
import json
import datetime
from typing import Dict, Any, List, Optional, Set, Tuple

from utils.trueskill_evaluator import (
    TrueSkillEvaluator, ModelRating, get_model_display_name, DEFAULT_LEADERBOARD_PATH,
    DEFAULT_EVAL_MAX_STEPS, DEFAULT_NO_TOUCH_STEPS,
    DEFAULT_SERIES_LENGTH, DEFAULT_SERIES_WINS_NEEDED,
    simulate_headless_series
)

# Checkpoint files are named by the iteration that produced them, and that ordering is
# what lets a new arrival inherit its predecessor's rating.
_CHECKPOINT_ITER_RE = re.compile(r"checkpoint_iter_(\d+)", re.I)


def snap_tiers_to_worker_slices(
    num_envs: int,
    n_training: int,
    n_self_play: int,
    n_king: int,
    n_pool: int,
    quantum: int,
) -> Tuple[int, int, int, int]:
    """
    Rounds each tier to a whole env-worker slice, returning the adjusted counts.

    A worker owns a contiguous run of environments and a rollout step does not finish
    until every worker returns, so a tier ending mid-slice leaves one worker holding two
    opponent models. That worker pays an unbatched forward pass on every step and sets
    the pace for all the environments, not just its own.

    Ratios land off-boundary easily. At 128 environments a fixed-opponent share of 0.20
    asks for 25.6, and rounding that to 26 pushed five of sixteen workers off their
    boundary. Snapping moves a tier by at most half a slice, a far smaller distortion of
    the requested mix than the throughput cost of honouring the ratio exactly.

    `quantum` is pool_group_size, which is already required to equal environments per
    worker because the pool tier depends on that same equality.
    """
    quantum = max(1, int(quantum))
    if quantum <= 1 or num_envs % quantum:
        return n_training, n_self_play, n_king, n_pool

    # With fewer slices than active tiers, snapping cannot give every tier a whole one
    # and would silently delete the losers: at 4 environments and a quantum of 4 the
    # King would round up to all four and self-play would vanish. There is also nothing
    # to gain, because a single worker holding a mixed set is not a straddle that any
    # rounding can fix. Honour the requested mix instead.
    active = sum(1 for n in (n_training, n_self_play, n_king, n_pool) if n > 0)
    if num_envs // quantum < active:
        return n_training, n_self_play, n_king, n_pool

    def snap(n: int) -> int:
        # A tier that was asked for at all keeps at least one whole slice.
        return max(quantum, int(round(n / quantum)) * quantum) if n > 0 else 0

    n_training = snap(n_training)
    n_king = snap(n_king)
    n_pool = (n_pool // quantum) * quantum

    # Self-play absorbs the remainder, which stays slice-aligned because every other
    # tier is and num_envs divides evenly. Give slices back in order of what the league
    # can most afford to lose if rounding overshot the budget.
    leftover = num_envs - n_training - n_king - n_pool
    while leftover < 0:
        if n_pool >= quantum:
            n_pool -= quantum
        elif n_king >= quantum:
            n_king -= quantum
        elif n_training >= quantum:
            n_training -= quantum
        else:
            break
        leftover += quantum
    return n_training, max(0, leftover), n_king, n_pool


class LeagueManager:
    """
    Coordinates Bayesian TrueSkill evaluation and dynamic league self-play.
    
    Stratification Tiers:
    - Tier 1: Pure Self-Play (Current Learner vs Current Learner)
    - Tier 2: King of the Hill (Current Learner vs Highest Conservative TrueSkill Checkpoint)
    - Tier 3: Historical Diversity Pool (Current Learner vs Randomly sampled elite checkpoint / Anchor)
    """

    def __init__(
        self,
        evaluator: Optional[TrueSkillEvaluator] = None,
        config: Optional[Dict[str, Any]] = None,
        leaderboard_path: str = DEFAULT_LEADERBOARD_PATH
    ):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.self_play_ratio = float(self.config.get("self_play_ratio", 0.50))
        self.king_ratio = float(self.config.get("king_ratio", 0.25))
        self.pool_ratio = float(self.config.get("pool_ratio", 0.25))
        self.max_pool_size = int(self.config.get("max_pool_size", 10))
        # Environments per distinct pool opponent within one refresh. Every distinct model
        # in the pool costs its own forward pass per rollout step, so spreading the pool
        # one-env-thin turns the pool tier into a pile of batch-size-1 inferences. Grouping
        # keeps the same long-run opponent diversity (the pool cycle advances each refresh)
        # at a fraction of the inference overhead.
        self.pool_group_size = max(1, int(self.config.get("pool_group_size", 4)))

        # Explicit, always-on training opponents. These are guaranteed a share of the
        # environments rather than competing for round-robin slots in the elite pool,
        # where a fixed reference ends up at whatever percentage the pool size implies.
        #
        # The ratio squashes the standard split rather than carving out of one tier: at
        # 0.10, ten percent of environments face this list and the remaining ninety are
        # divided 50/25/25 between self-play, King and pool. At 0.0 (the default) the
        # behaviour is exactly as before.
        self.training_opponents: List[str] = [
            self._normalize_path(x) for x in self.config.get("training_opponents", []) if x
        ]
        self.training_opponent_ratio = float(self.config.get(
            "training_opponent_ratio",
            # Migration: this control used to be a single model plus a weight.
            self.config.get("baseline_opponent_ratio", 0.0)
        ))
        self.training_opponent_ratio = max(0.0, min(1.0, self.training_opponent_ratio))
        self._training_opp_cycle_idx: int = 0
        self.protect_top_k = int(self.config.get("protect_top_k", 20))
        self.hall_of_fame_size = int(self.config.get("hall_of_fame_size", 20))
        self.hall_of_fame_min_matches = int(self.config.get("hall_of_fame_min_matches", 16))
        self.hall_of_fame_max_sigma = float(self.config.get("hall_of_fame_max_sigma", 2.5))
        # Debut games per opponent. Raised from 2 because the checkpoint interval moved
        # from 20 to 200 iterations: grading events became 10x rarer, so holding the
        # per-event budget fixed silently cut total evaluation throughput 10x. There is
        # room for it -- a grading event now costs ~20% of the ~175s window between
        # checkpoints, against ~70-100% under the old interval.
        self.eval_series_per_grade = int(self.config.get("eval_series_per_grade", 1))
        self.eval_max_steps = int(self.config.get("eval_max_steps", DEFAULT_EVAL_MAX_STEPS))
        self.eval_no_touch_steps = int(self.config.get("eval_no_touch_steps", DEFAULT_NO_TOUCH_STEPS))
        self.series_length = int(self.config.get("series_length", DEFAULT_SERIES_LENGTH))
        self.series_wins_needed = int(self.config.get("series_wins_needed", DEFAULT_SERIES_WINS_NEEDED))

        # Evaluator
        self.evaluator = evaluator or TrueSkillEvaluator(leaderboard_path=leaderboard_path)

        # League State Persistence Path
        default_state_path = (
            os.path.join(os.path.dirname(leaderboard_path), "league_state.json")
            if leaderboard_path != DEFAULT_LEADERBOARD_PATH
            else "logs/league_state.json"
        )
        self.league_state_path = self.config.get("league_state_path", default_state_path)

        # Anchors
        # The BC pretrained baseline is deliberately NOT an anchor. At 46W-4536L it is
        # beaten trivially by every checkpoint, so games against it carry almost no
        # information while consuming half of each debut's evaluation budget. It remains
        # available as a manual regression tripwire, just not as a rating reference.
        # Nexto is deliberately absent. It beats Necto 100% of series and loses 100% to
        # every checkpoint measured across a 63,000 iteration span, while Necto beats
        # those same checkpoints 100% of the time. That is a closed cycle, and no single
        # scalar rating can hold all three legs at once, so any value Nexto is pinned at
        # is wrong against someone. Worse, it is pinned above the field, so every win a
        # checkpoint takes off it would inflate the whole population.
        #
        # It stays a benchmark instead: played, reported, never rated. See
        # benchmark_opponents below.
        default_anchors = [
            "checkpoints/necto-model.pt",
            "heuristic"
        ]
        self.anchor_candidates = self.config.get("anchors", default_anchors)
        self.active_anchors: List[str] = []

        # Played for reporting only; results never touch a rating. This is where a
        # reference goes when it is genuinely strong but does not sit on one scale with
        # the models being rated.
        self.benchmark_opponents: List[str] = [
            self._normalize_path(x)
            for x in self.config.get("benchmark_opponents", ["checkpoints/nexto-model.pt"])
            if x
        ]
        self.benchmark_interval = int(self.config.get("benchmark_interval", 25))
        self.benchmark_series = int(self.config.get("benchmark_series", 2))
        # Both of these are restored from the league state file by load_league_state.
        # The counter has to survive the process boundary: grading runs in a fresh child
        # process per checkpoint (see agent.ppo._grade_in_subprocess), which rebuilds this
        # manager from disk, so a counter that lived only in memory was always 1 when it
        # reached the interval check and `1 % 25` never fired. Benchmarks had therefore
        # never run once, and benchmark_results was empty on every live board.
        self._benchmark_counter = 0
        self.benchmark_results: Dict[str, Any] = {}
        # One entry per benchmark event, so the reading becomes a curve. Capped because
        # this rides in the league state file, which the UI reloads on every refresh.
        self.benchmark_history: List[Dict[str, Any]] = []
        self.benchmark_history_cap = int(self.config.get("benchmark_history_cap", 300))

        # Opponents a gauntlet trial is split across. More than one because a long run
        # against a single opponent measures the pair rather than the population.
        self.gauntlet_opponents = max(1, int(self.config.get("gauntlet_opponents", 3)))

        # Gauntlet Contender Queue Configuration
        # Absolute skill floors do not survive a rescaling of the ladder. 25.5 was
        # calibrated when every rating converged near 30 on the old scale; against the
        # calibrated anchors the checkpoint population sits far lower, so a fixed floor
        # evicts contenders that are stronger than the reigning King. Kept only as a
        # legacy backstop, and no longer consulted directly -- see _contender_mu_floor.
        # How far below the King a contender may sit before the skill floor bites. The
        # floor is only applied once the King's own rating is established; on a fresh
        # leaderboard nothing is established, so there is no floor to breach.
        self.contender_mu_margin = float(self.config.get("contender_mu_margin", 4.0))
        # Minimum series before the points floor is applied. At the grace period of 6 a
        # points rate is 2-3 results wide and demotes on noise.
        self.min_matches_for_points_floor = int(self.config.get(
            "min_matches_for_points_floor", 12
        ))
        # Points rate (draw = half a win), not raw win rate. See ModelRating.points_rate.
        #
        # 40 rather than 45. The floor sits under an even contest, and a contender is
        # matched against the pool it is trying to join, so its true rate is near 50%.
        # Placing the floor five points under that put ordinary variance across it; ten
        # points under, and only when the interval clears it, asks the contender to be
        # measurably worse than the field rather than merely unlucky.
        self.min_contender_points_rate = float(self.config.get(
            "min_contender_points_rate",
            self.config.get("min_contender_win_rate", 40.0)
        ))
        # Consecutive series losses before the knockout fires.
        #
        # 4 sounds decisive and is not. Between evenly matched models a run of four is an
        # ordinary thing to see: across the 48 series of a full gauntlet it happens to
        # about 80% of contenders, so as a signal it carried almost no information and
        # did most of the evicting. 8 is genuinely unlikely for a contender worth keeping,
        # around 8%, while a checkpoint losing four fifths of its series still exits fast
        # because the points floor catches it by series 12.
        self.max_consecutive_losses = int(self.config.get("max_consecutive_losses", 8))
        self.grace_period_matches = int(self.config.get("grace_period_matches", 6))
        # Also the minimum count for rank-eligibility, in SERIES. Measured convergence
        # against the calibrated ladder (200 runs/point, draw rate 6.7%): 24 series
        # leaves mean sigma at 1.54 with only 17% clearing the 1.5 gate, while 30 series
        # reaches 1.37 and 99.5%. Setting this below where sigma actually crosses is what
        # produced un-rankable graduates before.
        self.target_eval_matches = int(self.config.get("target_eval_matches", 30))
        # Budget ceiling, in series. Headroom above target so a slow-converging
        # contender is not evicted un-ranked the moment it reaches the target.
        self.max_contender_matches = int(self.config.get("max_contender_matches", 48))
        # Series after which a converged rating is frozen. Sits above
        # max_contender_matches on purpose: the Gauntlet always finishes before the lock
        # can bite, so a contender is never frozen mid-trial. See
        # TrueSkillEvaluator.maybe_lock_rating for why the cap exists at all.
        self.rating_lock_matches = int(self.config.get("rating_lock_matches", 64))
        # Kept only for the extended-trial check. Graduation itself is gated on
        # rank-eligibility now: a separate 1.8 threshold sitting next to an
        # eligibility_sigma of 1.5 meant a contender could satisfy graduation while
        # still being unrankable, then stop receiving games -- the original lock-out,
        # one step further down the pipeline.
        self.target_eval_sigma = float(self.config.get("target_eval_sigma", 1.8))
        # Every Nth gauntlet step is spent on a King title bout instead of a contender
        # trial. At 2 that was half the (now much rarer) steps, spent re-measuring the
        # model whose rating is already the most certain. 3 leaves two thirds for
        # contenders, which is where the uncertainty actually is.
        self.king_challenge_frequency = int(self.config.get("king_challenge_frequency", 3))
        self.max_active_contenders = int(self.config.get("max_active_contenders", 3))
        # Games per gauntlet trial, split across the trial's opponents. Simulated sigma
        # convergence against the calibrated ladder (200 runs per point) puts the
        # sigma <= 1.5 eligibility gate at ~24 games: 16 games leaves mean sigma at 1.75
        # and 0% eligible, 24 games reaches 95%. At 12 per trial a contender converges
        # in a debut plus two trials.
        self.contender_series_per_step = int(self.config.get("contender_series_per_step", 3))
        # Retention: a newly minted checkpoint must survive on disk long enough for the
        # evaluator to reach it. Without this tier a recency window can delete a
        # checkpoint before it is ever measured, which silently biases the league toward
        # whatever happened to be graded first.
        self.max_provisional = int(self.config.get("max_provisional", 30))
        self.eligibility_sigma = float(self.config.get("eligibility_sigma", 1.5))

        # Gauntlet State & Sports Ticker Event History
        self.contender_queue: List[str] = []
        self.contender_consecutive_losses: Dict[str, int] = {}
        self.event_history: List[Dict[str, Any]] = []

        # League state
        self.king_of_the_hill: Optional[str] = None
        self.previous_king_of_the_hill: Optional[str] = None
        self.elite_pool: List[str] = []
        self._temporal_cycle_idx: int = 0
        self._pool_cycle_idx: int = 0
        self._gauntlet_step_counter: int = 0

        # Keep the standings table and the King/Elite Pool selection on one gate, so the
        # leaderboard the user reads cannot rank models differently from the league.
        self.evaluator.eligibility_sigma = self.eligibility_sigma
        self.evaluator.min_ranked_matches = self.target_eval_matches
        self.evaluator.rating_lock_matches = self.rating_lock_matches

        self.load_league_state()
        self._init_anchors()
        self.refresh_pool()

    def _normalize_path(self, path: str) -> str:
        if not path:
            return ""
        clean = path.strip().strip('"').strip("'")
        if clean.lower() in ("heuristic", "baseline", "baselinechaser", "none"):
            return "heuristic"
        return os.path.normpath(clean).replace("\\", "/")

    def _init_anchors(self):
        """Discovers and registers valid baseline anchor models."""
        self.active_anchors.clear()
        for cand in self.anchor_candidates:
            norm = self._normalize_path(cand)
            if norm == "heuristic":
                self.active_anchors.append("heuristic")
                r = self.evaluator.get_or_create_rating("heuristic", is_anchor=True)
                r.is_anchor = True
            elif os.path.exists(norm):
                self.active_anchors.append(norm)
                r = self.evaluator.get_or_create_rating(norm, is_anchor=True)
                r.is_anchor = True
            else:
                # Check inside checkpoints/
                alt = os.path.join("checkpoints", os.path.basename(norm)).replace("\\", "/")
                if os.path.exists(alt):
                    self.active_anchors.append(alt)
                    r = self.evaluator.get_or_create_rating(alt, is_anchor=True)
                    r.is_anchor = True

        if not self.active_anchors:
            self.active_anchors.append("heuristic")
            r = self.evaluator.get_or_create_rating("heuristic", is_anchor=True)
            r.is_anchor = True

    def _record_event(
        self,
        event_type: str,
        model_name: str,
        detail: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Records an event to the sports ticker transaction history (capped at 30 items)."""
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": event_type,  # 'promotion', 'demotion', 'admission', 'preemption', 'coronation'
            "model": model_name,
            "detail": detail,
            **(extra or {})
        }
        self.event_history.append(event)
        if len(self.event_history) > 30:
            self.event_history = self.event_history[-30:]
        self.save_league_state()

    def _contender_mu_floor(self) -> Optional[float]:
        """
        The skill floor a contender must stay above, or None when there is not enough
        established rating to define one.

        Relative to the King rather than absolute. A fixed threshold assumes the scale
        is fixed, and ours is not: recalibrating the anchors moved the whole checkpoint
        population, after which a floor of 25.5 sat *above* a King rated 23.26 and
        evicted contenders for outranking him.

        Returns None while the King is still provisional, which is the fresh-leaderboard
        case: nothing has been measured well enough for "too weak" to mean anything, and
        demoting on a noisy mu just starves the queue.
        """
        if not self.king_of_the_hill:
            return None
        king_rec = self.evaluator.ratings.get(self._normalize_path(self.king_of_the_hill))
        if not king_rec or not self._is_rank_eligible(king_rec):
            return None
        return king_rec.mu - self.contender_mu_margin

    def _is_rank_eligible(self, rec: Optional[ModelRating]) -> bool:
        """
        Whether a model's rating has converged enough to be ranked against its peers.

        A single scalar cannot both protect against small-sample flukes and rank
        established models: mu - k*sigma does the first job by permanently penalising the
        second. Splitting them, sigma becomes a gate and mu becomes the ranking, so an
        incumbent's extra few hundred matches stop buying it rank.
        """
        if not rec:
            return False
        # Anchors are calibrated, not measured; see TrueSkillEvaluator.is_rank_eligible.
        if rec.is_anchor:
            return True
        return rec.sigma <= self.eligibility_sigma and rec.matches_played >= self.target_eval_matches

    def _ranking_key(self, rec: Optional[ModelRating]) -> Tuple[int, float, float]:
        """
        Sort key for King and Elite Pool selection, descending.

        Eligible models always outrank provisional ones and are ordered by raw mu.
        Provisional models fall back to the conservative lower bound, which keeps an
        untested debut from vaulting to the top on four lucky games.
        """
        if not rec:
            return (0, -1e9, -1e9)
        if self._is_rank_eligible(rec):
            return (1, rec.mu, rec.points_rate)
        return (0, self._compute_competitive_score(rec), rec.points_rate)

    def _compute_competitive_score(self, rec: Optional[ModelRating]) -> float:
        """
        Fallback ordering for models that are not yet rank-eligible.

        Below target_eval_matches this is the conservative lower bound mu - 3*sigma, so a
        lucky four-game debut cannot vault the table; at or above it, mu - 2*sigma. Note
        that rank-eligible models are ordered by raw mu instead, via _ranking_key -- this
        score only separates provisional models from each other.
        """
        if not rec:
            return 0.0
        if rec.matches_played < self.target_eval_matches:
            return round(rec.mu - 3.0 * rec.sigma, 2)
        return round(rec.mu - 2.0 * rec.sigma, 2)

    def get_contender_queue_details(self) -> List[Dict[str, Any]]:
        """Provides rich metadata for each active contender in queue for the UI."""
        details = []
        king_rec = self.evaluator.ratings.get(self.king_of_the_hill) if self.king_of_the_hill else None
        king_mu = king_rec.mu if king_rec else 25.0

        for path in self.contender_queue:
            rec = self.evaluator.ratings.get(path)
            if not rec:
                continue
            consec_losses = self.contender_consecutive_losses.get(path, 0)

            is_title_contender = (
                (rec.mu >= king_mu - 1.0 or rec.points_rate >= 50.0)
                and rec.matches_played < self.max_contender_matches
                and rec.sigma > self.target_eval_sigma
            )
            target = self.max_contender_matches if (is_title_contender or rec.matches_played > self.target_eval_matches) else self.target_eval_matches
            progress_pct = min(100.0, round((rec.matches_played / max(1, target)) * 100.0, 1))

            # Status determination
            if rec.matches_played == 0:
                status = "Awaiting First Match"
            elif consec_losses >= self.max_consecutive_losses - 1:
                status = "⚠️ Knockout Danger"
            elif is_title_contender and rec.matches_played >= self.target_eval_matches:
                status = "🏆 Title Bout (Extended Trial)"
            elif rec.matches_played >= self.grace_period_matches:
                status = "Elimination Window Active"
            else:
                status = "Grace Period Active"

            details.append({
                "name": rec.name,
                "path": path,
                "mu": round(rec.mu, 2),
                "sigma": round(rec.sigma, 2),
                "conservative_score": round(rec.conservative_rating, 2),
                "competitive_score": self._compute_competitive_score(rec),
                "matches_played": rec.matches_played,
                "target_matches": target,
                "progress_pct": progress_pct,
                "win_rate": round(rec.win_rate, 1),
                "points_rate": rec.points_rate,
                "record": f"{rec.wins}W-{rec.losses}L-{rec.draws}D",
                "consecutive_losses": consec_losses,
                "max_consecutive_losses": self.max_consecutive_losses,
                "status": status
            })
        return details

    def get_elite_pool_details(self) -> List[Dict[str, Any]]:
        """Returns structured metadata for active Elite Pool models for UI display."""
        details = []
        for rank, path in enumerate(self.elite_pool, start=1):
            norm_p = self._normalize_path(path)
            rec = self.evaluator.ratings.get(norm_p) or self.evaluator.ratings.get(path)
            name = get_model_display_name(norm_p)
            is_king = (norm_p == self.king_of_the_hill or path == self.king_of_the_hill)
            if rec:
                details.append({
                    "rank": rank,
                    "name": name,
                    "path": norm_p,
                    "mu": round(rec.mu, 2),
                    "sigma": round(rec.sigma, 2),
                    "conservative_score": round(rec.conservative_rating, 2),
                    "competitive_score": self._compute_competitive_score(rec),
                    "win_rate": round(rec.win_rate, 1),
                    "points_rate": rec.points_rate,
                "points_rate": rec.points_rate,
                    "record": f"{rec.wins}W-{rec.losses}L-{rec.draws}D",
                    "matches_played": rec.matches_played,
                    "is_anchor": rec.is_anchor,
                    "is_king": is_king
                })
            else:
                details.append({
                    "rank": rank,
                    "name": name,
                    "path": norm_p,
                    "mu": 25.0,
                    "sigma": 8.33,
                    "conservative_score": 0.0,
                    "competitive_score": 0.0,
                    "win_rate": 0.0,
                    "record": "0W-0L-0D",
                    "matches_played": 0,
                    "is_anchor": True,
                    "is_king": is_king
                })
        return details

    def save_league_state(self, path: Optional[str] = None):
        """Atomically saves contender queue, elite pool, consecutive losses, and sports ticker event history."""
        target_path = path or self.league_state_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        tmp_path = target_path + ".tmp"

        payload = {
            "version": "1.0",
            "last_updated": datetime.datetime.now().isoformat(),
            "king_of_the_hill": self.king_of_the_hill,
            "elite_pool": self.elite_pool,
            "elite_pool_details": self.get_elite_pool_details(),
            "contender_queue": self.contender_queue,
            "contender_consecutive_losses": self.contender_consecutive_losses,
            "event_history": self.event_history[-30:],
            "contenders": self.get_contender_queue_details(),
            "benchmark_counter": self._benchmark_counter,
            "benchmark_results": self.benchmark_results,
            "benchmark_history": self.benchmark_history[-self.benchmark_history_cap:],
        }
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, target_path)
        except Exception as e:
            print(f"[League Manager] Warning: Could not save league state to {target_path}: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def load_league_state(self, path: Optional[str] = None):
        """Loads contender queue, elite pool, consecutive losses, and event history safely."""
        target_path = path or self.league_state_path
        if not os.path.exists(target_path):
            return
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded_queue = data.get("contender_queue", [])
            self.contender_queue = [
                p for p in loaded_queue
                if p == "heuristic" or os.path.exists(p)
            ]
            loaded_elite = data.get("elite_pool", [])
            self.elite_pool = [
                p for p in loaded_elite
                if p == "heuristic" or os.path.exists(p)
            ]
            self.contender_consecutive_losses = data.get("contender_consecutive_losses", {})
            self.event_history = data.get("event_history", [])[-30:]
            self.previous_king_of_the_hill = data.get("king_of_the_hill", None)
            try:
                self._benchmark_counter = int(data.get("benchmark_counter", 0))
            except (TypeError, ValueError):
                self._benchmark_counter = 0
            loaded_bench = data.get("benchmark_results")
            if isinstance(loaded_bench, dict):
                self.benchmark_results = loaded_bench
            loaded_hist = data.get("benchmark_history")
            if isinstance(loaded_hist, list):
                self.benchmark_history = [
                    e for e in loaded_hist
                    if isinstance(e, dict) and isinstance(e.get("results"), dict)
                ][-self.benchmark_history_cap:]
        except Exception as e:
            print(f"[League Manager] Warning: Could not load league state from {target_path}: {e}")

    def refresh_pool(self):
        """
        Scans current ratings, filters out deleted models, and identifies:
        1. King of the Hill (highest competitive score: mu - 2*sigma for established, mu - 3*sigma for debuts)
        2. Elite Pool (top max_pool_size models)
        Detects coronation events when a new King ascends.
        """
        old_king = self.king_of_the_hill

        valid_models: List[Tuple[str, float, ModelRating]] = []

        for key, rec in self.evaluator.ratings.items():
            norm_key = self._normalize_path(key)
            # Ephemeral rolling files like latest_model.pt are never eligible for King or elite pool
            if "latest_model" in norm_key.lower():
                continue
            comp_score = self._compute_competitive_score(rec)
            # Anchors are valid
            if rec.is_anchor or norm_key == "heuristic":
                valid_models.append((norm_key, comp_score, rec))
            elif os.path.exists(norm_key):
                valid_models.append((norm_key, comp_score, rec))

        if not valid_models:
            # Nothing rated at all: no King, and no pool. The pool tier folds back into
            # self-play rather than falling back to the anchors -- see below for why a
            # reference never belongs in the pool.
            self.king_of_the_hill = None
            self.elite_pool = []
            return

        # Sort descending: rank-eligible models first (ordered by raw mu), provisional
        # models after (ordered by their conservative lower bound).
        valid_models.sort(
            key=lambda item: (self._ranking_key(item[2]), item[2].matches_played),
            reverse=True
        )

        # The King is always a checkpoint, never an anchor.
        #
        # An anchor is a fixed external reference, not a competitor: it cannot be
        # dethroned, it does not improve, and crowning it points a quarter of all
        # environments at an opponent chosen for being a stable yardstick rather than for
        # being a useful sparring partner. On a freshly reset leaderboard Nexto took the
        # crown on its declared mu of 34.7 and drew 31% of environments, against a policy
        # that scores about 19% on it.
        checkpoints_only = [m for m in valid_models if not m[2].is_anchor and m[0] != "heuristic"]
        if checkpoints_only:
            self.king_of_the_hill = checkpoints_only[0][0]
        else:
            # No rated checkpoint yet. Hold the throne empty rather than seating an
            # anchor; get_stratified_distribution falls back for the King tier.
            self.king_of_the_hill = None

        # Check coronation (King handoff)
        if old_king and self.king_of_the_hill and old_king != self.king_of_the_hill:
            old_name = get_model_display_name(old_king)
            new_name = get_model_display_name(self.king_of_the_hill)
            new_rec = self.evaluator.ratings.get(self.king_of_the_hill)
            new_score = f"{self._compute_competitive_score(new_rec):.2f}" if new_rec else "N/A"
            self._record_event(
                event_type="coronation",
                model_name=new_name,
                detail=f"Dethroned '{old_name}' to claim King of the Hill (Score: {new_score})",
                extra={"old_king": old_name, "new_king": new_name}
            )

        # Populate the elite pool with checkpoints only.
        #
        # An anchor is a yardstick, not a sparring partner. It exists to give a new
        # checkpoint a calibrated reference to be measured against, and grading does that
        # job in full. Letting one into the pool hands it a share of training
        # environments it was never meant to have: Nexto's declared mu is the highest on
        # the ladder, so ranking alone seated it at the top of the pool permanently,
        # while heuristic and the BC baseline sat below every real checkpoint and taught
        # the policy nothing it had not already outgrown.
        #
        # It is also the most expensive opponent to run. The pool hands each model a
        # contiguous run of pool_group_size environments, which is one subprocess worker,
        # and a rollout step does not finish until every worker does. One worker drawing
        # Nexto costs 3.02 ms per step against 2.44 ms for a checkpoint, and all 128
        # environments pay that difference, not just the eight facing it.
        #
        # Deliberate exposure to a reference model is a separate control: the
        # training_opponents list gives it a guaranteed, explicitly chosen share.
        pool_paths = []
        for path, _, rec in valid_models:
            if rec.is_anchor or path == "heuristic":
                continue
            if path not in pool_paths:
                pool_paths.append(path)
            if len(pool_paths) >= self.max_pool_size:
                break

        self.elite_pool = pool_paths

    def _checkpoint_iteration(self, path: str) -> Optional[int]:
        """The training iteration a checkpoint file came from, or None if unnamed."""
        m = _CHECKPOINT_ITER_RE.search(os.path.basename(str(path)))
        return int(m.group(1)) if m else None

    def seed_from_predecessor(self, ckpt_path: str, rec: ModelRating) -> bool:
        """
        Starts a brand-new checkpoint at the previous checkpoint's mu.

        Seer's method (Neville & Walo, sec. 4.1): a new agent's sigma comes from the
        environment but its mu is initialised to the final mu of the agent before it.
        Consecutive checkpoints are 200 iterations apart and genuinely similar, so the
        predecessor is a far better guess than the global default of 25.0.

        Seeding at 25.0 was actively harmful here. The field measures at roughly 22 from
        both ends of the anchor ladder, so every arrival entered three mu above the
        population and the handful of anchor games it played could not pull it back. The
        cluster was held up by its own seeding rather than by any result.

        Only mu is inherited. Sigma stays at the default, because nothing about the
        predecessor's certainty transfers to a model that has not played.

        The predecessor must itself have cleared the grace period. A rating drawn from one
        or two series is barely a measurement: on this leaderboard the abandoned debuts
        left over from before the debut skip range from mu 18.89 to 22.92 between
        neighbouring saves, so inheriting one would pass that noise straight to the new
        checkpoint and call it a starting point. Seer's agents each had about twenty games
        behind them before the next one seeded from their final mu. Where no predecessor
        qualifies, the default rating stands and the gauntlet does the work.
        """
        if rec.matches_played > 0:
            return False
        n = self._checkpoint_iteration(ckpt_path)
        if n is None:
            return False
        best_iter, best_rec = None, None
        for path, other in self.evaluator.ratings.items():
            if other is rec or other.is_anchor:
                continue
            if other.matches_played < self.grace_period_matches:
                continue
            k = self._checkpoint_iteration(path)
            if k is None or k >= n:
                continue
            if best_iter is None or k > best_iter:
                best_iter, best_rec = k, other
        if best_rec is None:
            return False
        rec.mu = best_rec.mu
        rec.update_conservative()
        return True

    def nearest_rated_peers(
        self, rec: ModelRating, exclude: str = "", limit: int = 3
    ) -> List[str]:
        """
        The rated checkpoints closest to this one in mu.

        Seer picks the matchup with the highest probability of a draw when evaluating a
        new agent, for the reason that information per series peaks near even odds. Here
        it is also the only workable choice, because no anchor is anywhere near a draw
        against a current checkpoint. Measured over five checkpoints spanning 63,000
        iterations, four series each: the heuristic and the BC baseline lost every series,
        Necto won every series, Nexto lost every series. All four are saturated, so none
        of them can tell an early checkpoint from a late one.

        Peers can. The same probe found the checkpoint round robin cleanly transitive,
        with no cycles, which is what makes a scalar rating meaningful over this set.
        """
        candidates: List[Tuple[float, str]] = []
        for path, other in self.evaluator.ratings.items():
            norm = self._normalize_path(path)
            if norm == exclude or norm == "heuristic" or other.is_anchor:
                continue
            if "latest_model" in norm.lower() or not os.path.exists(norm):
                continue
            if other.matches_played < self.grace_period_matches:
                continue
            candidates.append((abs(other.mu - rec.mu), norm))
        candidates.sort(key=lambda c: c[0])
        return [path for _, path in candidates[:max(1, limit)]]

    def run_benchmarks(self, subject_path: str, device: str = "cpu") -> Dict[str, Any]:
        """
        Plays the benchmark references and records the score without rating anything.

        This is where a reference goes when it is strong but not on one scale with the
        models under test. Nexto beats Necto every series and loses to every checkpoint
        every series, so it cannot be ranked alongside them, but knowing how the current
        best does against it is still worth having. Nothing here calls the evaluator, so
        no rating moves.
        """
        results: Dict[str, Any] = {}
        subject = self._normalize_path(subject_path)
        if not self.benchmark_opponents or not os.path.exists(subject):
            return results
        try:
            from env.baseline_agent import create_opponent_bot
        except Exception:
            return results
        for opp in self.benchmark_opponents:
            if opp != "heuristic" and not os.path.exists(opp):
                continue
            try:
                a = create_opponent_bot(subject, continuous_actions=True)
                b = create_opponent_bot(opp, continuous_actions=True)
                wins = goals_for = goals_against = episodes = 0
                for _ in range(max(1, self.benchmark_series)):
                    r = simulate_headless_series(
                        a, b,
                        series_length=self.series_length,
                        wins_needed=self.series_wins_needed,
                        max_steps=self.eval_max_steps,
                        no_touch_steps=self.eval_no_touch_steps,
                    )
                    wins += 1 if r.get("winner") == "a" else 0
                    goals_for += int(r.get("a_score", 0))
                    goals_against += int(r.get("b_score", 0))
                    episodes += int(r.get("episodes_played", 0))
                played = max(1, self.benchmark_series)
                results[get_model_display_name(opp)] = {
                    "series": played,
                    "series_won": wins,
                    "points_rate": round(100.0 * wins / played, 1),
                    "goals": f"{goals_for}-{goals_against}",
                    # Goal margin per episode is the readout that carries a gradient.
                    # Series win rate against these references is pinned at the rails --
                    # checkpoints take 100% off the heuristic and the BC baseline and won
                    # 9 of 96 off Necto across iterations 160000-190200 -- while the
                    # margin over that same span moved -0.64 to -0.25. Recording the
                    # episode count is what makes the margin recoverable later.
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "episodes": episodes,
                    "margin_per_episode": round((goals_for - goals_against) / episodes, 3) if episodes else 0.0,
                    "subject": get_model_display_name(subject),
                    "at": datetime.datetime.now().isoformat(),
                }
            except Exception as e:
                print(f"[League Manager] Benchmark against {opp} failed: {e}")
        if results:
            self.benchmark_results = results
            self._append_benchmark_history(subject, results)
        return results

    def _append_benchmark_history(self, subject: str, results: Dict[str, Any]):
        """
        Keeps one entry per benchmark event so the score becomes a curve.

        Without this the benchmark is a single latest reading: run_benchmarks replaces
        benchmark_results wholesale, and the per-iteration history log is 327 MB of
        per-iteration rows, far too coarse a haystack to recover a value that fires once
        every few thousand iterations. A capped list on the league state is small, is
        already loaded by the UI on every refresh, and is the only thing that can answer
        "is this improving" rather than "where is it now".

        Keyed on training iteration rather than wall clock, because restarts and pauses
        make elapsed time a poor x axis for a training curve.
        """
        iteration = self._checkpoint_iteration(subject)
        entry = {
            "at": datetime.datetime.now().isoformat(),
            "iteration": iteration,
            "subject": get_model_display_name(subject),
            "results": {
                name: {
                    "series": res.get("series", 0),
                    "series_won": res.get("series_won", 0),
                    "goals_for": res.get("goals_for", 0),
                    "goals_against": res.get("goals_against", 0),
                    "episodes": res.get("episodes", 0),
                    "margin_per_episode": res.get("margin_per_episode", 0.0),
                }
                for name, res in results.items()
            },
        }
        # An entry with no iteration cannot be placed on the axis, so it is reported in
        # benchmark_results but kept out of the series.
        if iteration is None:
            return
        self.benchmark_history.append(entry)
        if len(self.benchmark_history) > self.benchmark_history_cap:
            self.benchmark_history = self.benchmark_history[-self.benchmark_history_cap:]

    def _nearest_anchor(self, rec: Optional[ModelRating], exclude: str = "") -> Optional[str]:
        """
        Returns the active anchor whose calibrated mu sits closest to `rec`'s.

        With the ladder spanning heuristic (15) to Nexto (38), picking a fixed anchor
        means most contenders are graded against a reference far from their own level,
        where the outcome is a foregone conclusion and the rating update is negligible.
        """
        target_mu = rec.mu if rec else 25.0
        best, best_gap = None, None
        for cand in self.active_anchors:
            norm_c = self._normalize_path(cand)
            if norm_c == exclude:
                continue
            if not (os.path.exists(norm_c) or norm_c == "heuristic"):
                continue
            anchor_rec = self.evaluator.ratings.get(norm_c)
            anchor_mu = anchor_rec.mu if anchor_rec else 25.0
            gap = abs(anchor_mu - target_mu)
            if best_gap is None or gap < best_gap:
                best, best_gap = norm_c, gap
        return best

    def can_admit_a_newcomer(self) -> bool:
        """
        Whether a checkpoint saved right now could enter the gauntlet at all.

        Knowable before any matches are played, which is the point. A fresh checkpoint
        always carries the highest sigma in the queue, so the only open question is
        whether the most settled contender is rank-eligible and can therefore be
        preempted. If it is not, admission is impossible and the debut buys nothing.

        Without this, every save was graded and thrown away: ten an hour spending two
        series each, leaving records stranded at sigma 5.1 that never joined the pool
        either, while the contenders those series could have measured sat idle.
        """
        if len(self.contender_queue) < self.max_active_contenders:
            return True
        rated = [
            r for r in (self.evaluator.ratings.get(p) for p in self.contender_queue)
            if r is not None
        ]
        if not rated:
            return True
        return self._is_rank_eligible(min(rated, key=lambda r: r.sigma))

    def _admit_contender(self, ckpt_path: str):
        """Admits or preempts into the active Gauntlet contender queue based on raw skill (mu)."""
        if ckpt_path in self.contender_queue:
            return

        rec = self.evaluator.ratings.get(ckpt_path)
        if not rec:
            return

        if len(self.contender_queue) < self.max_active_contenders:
            self.contender_queue.append(ckpt_path)
            self.contender_consecutive_losses[ckpt_path] = 0
            print(f"[League Manager] [Gauntlet Admission] Admitted '{rec.name}' to Contender Queue (mu={rec.mu:.2f}, sigma={rec.sigma:.2f}, Pts={rec.points_rate:.1f}%)")
            self._record_event(
                event_type="admission",
                model_name=rec.name,
                detail=f"Admitted to Gauntlet Contender Queue (μ={rec.mu:.2f}, σ={rec.sigma:.2f}, Pts={rec.points_rate:.1f}%)",
                extra={"mu": rec.mu, "sigma": rec.sigma, "points_rate": rec.points_rate}
            )
        else:
            # Preemption. The queue is full, so admitting this checkpoint means evicting
            # one that is already part-way through its trial.
            #
            # Comparing on mu made this pathological: mu is most inflated exactly when a
            # rating is least trustworthy, so a checkpoint fresh off four games routinely
            # showed mu 32+ and evicted a contender twelve games deep that had regressed
            # toward its true value. The queue thrashed and essentially nobody completed a
            # trial. Evict on *convergence* instead: the contender closest to being
            # rank-eligible has the least left to gain from the remaining budget, and a
            # newcomer only displaces it if the newcomer is itself less measured.
            contender_ratings = [(p, self.evaluator.ratings.get(p)) for p in self.contender_queue]
            valid_contenders = [c for c in contender_ratings if c[1] is not None]
            if valid_contenders:
                most_settled_path, most_settled_rec = min(valid_contenders, key=lambda c: c[1].sigma)
                if rec.sigma > most_settled_rec.sigma and self._is_rank_eligible(most_settled_rec):
                    self.contender_queue.remove(most_settled_path)
                    self.contender_consecutive_losses.pop(most_settled_path, None)
                    self.contender_queue.append(ckpt_path)
                    self.contender_consecutive_losses[ckpt_path] = 0
                    print(f"[League Manager] [Gauntlet Preemption] '{rec.name}' (sigma={rec.sigma:.2f}) replaced settled '{most_settled_rec.name}' (sigma={most_settled_rec.sigma:.2f}) in Gauntlet Queue")
                    self._record_event(
                        event_type="preemption",
                        model_name=rec.name,
                        detail=f"Replaced converged '{most_settled_rec.name}' in Gauntlet Queue (σ={rec.sigma:.2f} > {most_settled_rec.sigma:.2f})",
                        extra={"admitted": rec.name, "bumped": most_settled_rec.name, "sigma": rec.sigma}
                    )
                else:
                    print(f"[League Manager] [Gauntlet] Queue full of un-converged contenders; '{rec.name}' not admitted this round.")

    def step_king_title_bout(self, device: str = "cpu") -> Optional[Dict[str, Any]]:
        """
        Executes a 2-game head-to-head title bout between the reigning King of the Hill
        and the top-ranked non-King challenger in the Elite Pool.
        Ensures established elite checkpoints continue refining their TrueSkill ratings
        and prevents stagnant, uncontested incumbent reigns.
        """
        if not self.enabled or not self.king_of_the_hill:
            return None

        # Clean elite pool of missing files
        valid_challengers = [
            p for p in self.elite_pool
            if p != self.king_of_the_hill
            and p != "heuristic"
            and os.path.exists(p)
            and not getattr(self.evaluator.ratings.get(self._normalize_path(p)), "is_anchor", False)
        ]
        if not valid_challengers:
            return None

        # A bout between two frozen ratings changes nothing on either side, so it is the
        # one place the series cap actually reclaims compute. Prefer a challenger whose
        # rating can still move, and if the King is frozen too, skip the bout entirely
        # and hand the step back to the Gauntlet where the budget does buy something.
        king_frozen = self.evaluator.is_rating_frozen(
            self.evaluator.ratings.get(self._normalize_path(self.king_of_the_hill))
            or ModelRating(name="", path="")
        )
        if king_frozen:
            live = [
                p for p in valid_challengers
                if not self.evaluator.is_rating_frozen(
                    self.evaluator.ratings.get(self._normalize_path(p)) or ModelRating(name="", path="")
                )
            ]
            if not live:
                return None
            valid_challengers = live

        # Pick top challenger by the same rule that decides the pool, so the title bout
        # and the ranking cannot disagree about who the best challenger is.
        challenger_path = max(
            valid_challengers,
            key=lambda p: self._ranking_key(
                self.evaluator.ratings.get(self._normalize_path(p))
                or ModelRating(name=p, path=p)
            )
        )
        norm_challenger = self._normalize_path(challenger_path)
        norm_king = self._normalize_path(self.king_of_the_hill)
        c_rec = self.evaluator.ratings.get(norm_challenger)
        k_rec = self.evaluator.ratings.get(norm_king)
        if not c_rec or not k_rec:
            return None

        print(f"[League Manager] [King Title Bout] '{c_rec.name}' (Score: {self._compute_competitive_score(c_rec):.2f}) challenging King '{k_rec.name}' in 2-game title match...")
        try:
            self.evaluator.evaluate_pairing(
                model_a_path=norm_challenger,
                model_b_path=norm_king,
                series_per_pair=1,
                max_steps=self.eval_max_steps,
                no_touch_steps=self.eval_no_touch_steps,
                series_length=self.series_length,
                wins_needed=self.series_wins_needed,
                device=device
            )
        except Exception as e:
            print(f"[League Manager] Warning: Title bout error {norm_challenger} vs {norm_king}: {e}")
            return None

        old_king = self.king_of_the_hill
        self.refresh_pool()
        self.save_league_state()

        if self.king_of_the_hill and self.king_of_the_hill != old_king:
            print(f"[League Manager] [King Title Bout] Coronation! '{get_model_display_name(self.king_of_the_hill)}' dethroned '{get_model_display_name(old_king)}'!")
            return {"status": "coronation", "new_king": self.king_of_the_hill, "old_king": old_king}
        return {"status": "defended", "king": self.king_of_the_hill, "challenger": norm_challenger}

    def step_contender_gauntlet(self, device: str = "cpu") -> Optional[Dict[str, Any]]:
        """
        Advances the Gauntlet promotion/demotion trials by running matches for the top active contender.
        Evaluates early demotion (after grace period) and graduation (once target matches reached).
        Also coordinates periodic King Title Bouts against top Elite Pool challengers.
        """
        if not self.enabled:
            return None

        self._gauntlet_step_counter += 1

        # Clean queue of any non-existent files
        self.contender_queue = [p for p in self.contender_queue if os.path.exists(p)]

        # If contender queue is empty, run an Elite King Title Bout directly!
        if not self.contender_queue:
            return self.step_king_title_bout(device=device)

        # If contender queue has items, periodically (every king_challenge_frequency steps) run a title bout
        if self._gauntlet_step_counter % self.king_challenge_frequency == 0:
            self.step_king_title_bout(device=device)

        # Re-check queue in case title bout altered state or queue is empty
        if not self.contender_queue:
            return None

        # Pick the contender we know least about, not the one that currently looks best.
        # Scheduling by mu spends the budget re-confirming a leader while the genuinely
        # unmeasured sit idle; scheduling by sigma maximises information per match and is
        # what lets a contender actually converge to rank-eligibility.
        contender_path = max(
            self.contender_queue,
            key=lambda p: getattr(self.evaluator.ratings.get(p), "sigma", 0.0)
        )
        rec = self.evaluator.ratings.get(contender_path)
        if not rec:
            self.contender_queue.remove(contender_path)
            return None

        # Gauntlet pairing: the nearest-rated peers, not an anchor.
        #
        # Half of every trial used to go to the nearest anchor, which was always Necto,
        # which takes 93.4% of points off checkpoints over 311 recorded series. That put
        # the ceiling on a trial at roughly 28% for a contender exactly as good as the
        # King, against a floor of 40%, so no contender could pass however good it was.
        #
        # A probe over five checkpoints spanning 63,000 iterations found every anchor
        # saturated: heuristic and the BC baseline lost every series, Necto won every
        # series, Nexto lost every series. None of them distinguishes an early checkpoint
        # from a late one. Peers do, and the same probe found the checkpoint round robin
        # cleanly transitive.
        opponents = self.nearest_rated_peers(
            rec, exclude=contender_path, limit=self.gauntlet_opponents
        )

        # Fallbacks, in descending order of how much the result will tell us. Early in a
        # run there may be no rated peer yet.
        if not opponents:
            king = self.king_of_the_hill
            if king and king != contender_path and "latest_model" not in king.lower():
                opponents = [king]
        if not opponents:
            near = self._nearest_anchor(rec, exclude=contender_path)
            if near:
                opponents = [near]

        if not opponents:
            return None

        print(f"[League Manager] [Gauntlet Trial] Testing contender '{rec.name}' against {len(opponents)} opponent(s)...")

        # The streak is counted series by series, in the order they were played.
        #
        # It used to be inferred from the trial's totals: add every loss when a trial lost
        # more than it won, subtract the wins otherwise. That is a running loss surplus,
        # not a streak, and it scales with the trial size. At three series per trial the
        # increments were small enough to pass for one. At twenty-four it evicted
        # checkpoint_iter_181600, a contender 34 series deep at the top of the field, for
        # "19 consecutive losses" it never had. A 10W-14L trial would have added 14.
        #
        # evaluate_pairing returns one entry per series, in order, so the real run is
        # available and there is no need to infer anything. A loss extends it and anything
        # else ends it, which is what the name says and what max_consecutive_losses of 8
        # was calibrated against.
        streak = self.contender_consecutive_losses.get(contender_path, 0)
        for opp in opponents:
            try:
                series_results = self.evaluator.evaluate_pairing(
                    model_a_path=contender_path,
                    model_b_path=opp,
                    series_per_pair=max(1, self.contender_series_per_step // len(opponents)),
                    max_steps=self.eval_max_steps,
                    no_touch_steps=self.eval_no_touch_steps,
                    series_length=self.series_length,
                    wins_needed=self.series_wins_needed,
                    device=device
                )
                for res in (series_results or []):
                    # The contender is always model_a here, so a negative score_diff is
                    # its loss. A draw is not a loss, so it ends the run too.
                    streak = streak + 1 if res.get("score_diff", 0) < 0 else 0
            except Exception as e:
                print(f"[League Manager] Warning: Gauntlet trial error {contender_path} vs {opp}: {e}")

        self.contender_consecutive_losses[contender_path] = streak

        # Refresh rating after matches
        rec = self.evaluator.ratings.get(contender_path, rec)
        consec_losses = streak

        king_rec = self.evaluator.ratings.get(self.king_of_the_hill) if self.king_of_the_hill else None
        king_mu = king_rec.mu if king_rec else 25.0

        is_title_contender = (
            (rec.mu >= king_mu - 1.0 or rec.points_rate >= 50.0)
            and rec.matches_played < self.max_contender_matches
            and rec.sigma > self.target_eval_sigma
        )

        # 1. Check Demotion first (only after grace period matches).
        #
        # Order matters, and it used to be the other way round. Graduation sat behind two
        # absolute preconditions, mu >= 25.5 and a points rate above the demotion floor,
        # which opened a band where a contender could neither graduate nor be demoted. It
        # then drew gauntlet trials forever, spending evaluation budget on a rating that
        # had already settled and starving newer checkpoints of measurement. With the
        # King at mu 24.09 that band was every contender between 20.09 and 25.50, and one
        # record reached 91 matches inside it.
        #
        # Deciding demotion first and graduating on measurement alone makes the two
        # exhaustive: a contender either leaves on evidence or leaves once measured.
        should_demote = False
        demote_reason = ""
        if rec.matches_played >= self.grace_period_matches:
            mu_floor = self._contender_mu_floor()
            # Demote on skill only when the rating is confidently below the floor, not
            # when a noisy point estimate dips under it. At 6 series sigma is ~3.5, so
            # comparing bare mu to a threshold decides on noise: 159600 was evicted at
            # mu 24.34 +/- 3.41 against a floor of 25.50, which its interval straddled.
            if mu_floor is not None and (rec.mu + rec.sigma) < mu_floor:
                should_demote = True
                demote_reason = (
                    f"Skill floor breached (mu={rec.mu:.2f} +/- {rec.sigma:.2f} "
                    f"confidently below {mu_floor:.2f})"
                )
            elif rec.matches_played >= self.min_matches_for_points_floor:
                # Same standard of evidence as the skill floor, for the same reason. A
                # points rate over 12 series carries a standard error near 14 points, so
                # comparing it bare to a floor a few points under even odds evicts on
                # variance. Simulated over a full 48-series gauntlet, an exactly average
                # contender was thrown out 87% of the time and one genuinely better than
                # the field, at 60%, still 58% of the time. Requiring the interval to
                # clear the floor brings those to 17% and 4%, while a weak checkpoint at
                # 25% is still caught by series 12.
                rate = max(0.0, min(1.0, rec.points_rate / 100.0))
                spread = math.sqrt(
                    max(rate * (1.0 - rate), 0.01) / max(1, rec.matches_played)
                ) * 100.0
                if rec.points_rate + spread < self.min_contender_points_rate:
                    should_demote = True
                    demote_reason = (
                        f"Points rate below floor ({rec.points_rate:.1f}% +/- {spread:.1f} "
                        f"confidently below {self.min_contender_points_rate:.1f}%)"
                    )
                elif consec_losses >= self.max_consecutive_losses:
                    should_demote = True
                    demote_reason = (
                        f"Loss streak knockout ({consec_losses} >= "
                        f"{self.max_consecutive_losses} consecutive losses)"
                    )
            elif consec_losses >= self.max_consecutive_losses:
                should_demote = True
                demote_reason = (
                    f"Loss streak knockout ({consec_losses} >= "
                    f"{self.max_consecutive_losses} consecutive losses)"
                )

        if should_demote:
            print(f"[League Manager] [Gauntlet Demotion] '{rec.name}' evicted from contender queue. Reason: {demote_reason}")
            self.contender_queue.remove(contender_path)
            self.contender_consecutive_losses.pop(contender_path, None)
            self.refresh_pool()
            self._record_event(
                event_type="demotion",
                model_name=rec.name,
                detail=f"Evicted from Gauntlet: {demote_reason}",
                extra={"reason": demote_reason, "matches": rec.matches_played}
            )
            return {"status": "demoted", "model": rec.name, "reason": demote_reason}

        # 2. Check Graduation, on measurement alone.
        #
        # Anything that deserved to leave on skill has already left above, so the only
        # remaining question is whether the rating is worth ranking. Which contenders are
        # actually good is decided by the elite pool, which takes the top max_pool_size by
        # mu. The gauntlet's job is to find out, not to judge.
        #
        # Graduating on a looser sigma than the ranking gate produced a checkpoint that
        # was out of the queue, below the pool, and therefore never played again, its
        # sigma frozen forever just short of the threshold.
        should_graduate = (
            self._is_rank_eligible(rec)
            # Out of budget. It leaves un-ranked and sits as provisional until the pool
            # has room, rather than blocking the queue indefinitely.
            or rec.matches_played >= self.max_contender_matches
        )

        if should_graduate:
            print(f"[League Manager] [Gauntlet Graduation] '{rec.name}' graduated with established rating: mu={rec.mu:.2f}, sigma={rec.sigma:.2f}, Score={self._compute_competitive_score(rec):.2f} over {rec.matches_played} matches!")
            self.contender_queue.remove(contender_path)
            self.contender_consecutive_losses.pop(contender_path, None)
            self.refresh_pool()
            self._record_event(
                event_type="promotion",
                model_name=rec.name,
                detail=f"Graduated Gauntlet to Elite Pool! (mu: {rec.mu:.2f}, Pts: {rec.points_rate:.1f}% in {rec.matches_played} matches)",
                extra={"mu": rec.mu, "matches": rec.matches_played, "points_rate": rec.points_rate}
            )
            return {"status": "graduated", "model": rec.name, "score": self._compute_competitive_score(rec)}

        target = self.max_contender_matches if is_title_contender else self.target_eval_matches
        print(f"[League Manager] [Gauntlet Progress] '{rec.name}' now at {rec.matches_played}/{target} matches (mu={rec.mu:.2f}, sigma={rec.sigma:.2f}, Comp Score={self._compute_competitive_score(rec):.2f})")
        self.refresh_pool()
        self.save_league_state()
        return {"status": "progress", "model": rec.name, "matches": rec.matches_played, "score": self._compute_competitive_score(rec)}

    def grade_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> ModelRating:
        """
        Automatically grades a newly saved checkpoint in fast headless matches against:
        1. The reigning King of the Hill.
        2. A reference anchor baseline.
        
        Updates ratings in logs/trueskill_leaderboard.json and updates King-of-the-Hill status.
        """
        norm_ckpt = self._normalize_path(checkpoint_path)
        if "latest_model" in norm_ckpt.lower():
            return self.evaluator.get_or_create_rating(norm_ckpt)

        if not os.path.exists(norm_ckpt):
            print(f"[League Manager] Warning: Checkpoint not found on disk: {checkpoint_path}")
            return self.evaluator.get_or_create_rating(norm_ckpt)

        # Ensure model rating record exists
        rec = self.evaluator.get_or_create_rating(norm_ckpt)

        # A new arrival starts where its predecessor finished, not at the global default.
        if self.seed_from_predecessor(norm_ckpt, rec):
            print(
                f"[League Manager] Seeded '{rec.name}' at mu={rec.mu:.2f} from the "
                f"preceding checkpoint rather than the default."
            )

        # Grade only what could actually enter the gauntlet.
        #
        # A debut costs two series and exists to decide admission. With the queue full of
        # contenders that are not yet rank-eligible, nothing can be preempted, so the
        # answer is already no and the matches are spent for nothing. Skipping returns
        # that budget to the contenders being measured, and stops the leaderboard filling
        # with records stranded at their debut sigma.
        #
        # Anchors and the rolling latest_model are handled above; this only ever skips a
        # checkpoint, and the next save after a slot frees is graded normally. Since the
        # newest save is the one that takes the free slot, skipping costs nothing but the
        # rating of a model that was about to be superseded anyway.
        if not rec.is_anchor and not self.can_admit_a_newcomer():
            print(
                f"[League Manager] Gauntlet queue full and none of it converged; "
                f"deferring the debut of '{rec.name}' rather than spending "
                f"{self.eval_series_per_grade * 2} series on a rating that cannot enter."
            )
            return rec

        # Determine benchmark opponents
        opponents_to_test: List[str] = []
        if self.king_of_the_hill and self.king_of_the_hill != norm_ckpt and "latest_model" not in self.king_of_the_hill.lower():
            opponents_to_test.append(self.king_of_the_hill)

        # A nearest-rated peer rather than a nearest anchor. The debut no longer has to
        # establish an absolute position, because the checkpoint inherits its
        # predecessor's mu; it has to place this checkpoint against the current field,
        # and only a peer can do that. Every anchor is saturated against the population.
        for peer in self.nearest_rated_peers(rec, exclude=norm_ckpt, limit=2):
            if peer not in opponents_to_test:
                opponents_to_test.append(peer)
            if len(opponents_to_test) >= 2:
                break

        # Nothing rated yet: fall back to the ladder so the first checkpoints of a run
        # still get placed somewhere.
        if not opponents_to_test:
            near = self._nearest_anchor(rec, exclude=norm_ckpt)
            opponents_to_test.append(near or "heuristic")

        print(f"[League Manager] Auto-Grading Checkpoint '{rec.name}' against {len(opponents_to_test)} opponent(s)...")

        for opp in opponents_to_test:
            opp_name = get_model_display_name(opp)
            print(f"  -> Matchup: {rec.name} vs {opp_name} (best-of-{self.series_length} x{self.eval_series_per_grade})")
            try:
                self.evaluator.evaluate_pairing(
                    model_a_path=norm_ckpt,
                    model_b_path=opp,
                    series_per_pair=self.eval_series_per_grade,
                    max_steps=self.eval_max_steps,
                    no_touch_steps=self.eval_no_touch_steps,
                    series_length=self.series_length,
                    wins_needed=self.series_wins_needed,
                    device=device
                )
            except Exception as e:
                print(f"[League Manager] Error evaluating {norm_ckpt} vs {opp}: {e}")

        # Refresh state
        self.refresh_pool()

        # Benchmarks are rating-free and only run occasionally, so a reference that
        # cannot be ranked alongside the field is still tracked.
        self._benchmark_counter += 1
        if (self.benchmark_opponents and self.benchmark_interval > 0
                and self._benchmark_counter % self.benchmark_interval == 0):
            subject = self.king_of_the_hill or norm_ckpt
            for name, res in (self.run_benchmarks(subject) or {}).items():
                print(
                    f"[League Manager] Benchmark: {res['subject']} vs {name} -> "
                    f"{res['series_won']}/{res['series']} series, goals {res['goals']} "
                    f"(reported only, no rating changed)"
                )
        # Persist the counter even when no benchmark ran, so the next child process picks
        # up where this one left off rather than restarting the interval.
        self.save_league_state()

        king_name = get_model_display_name(self.king_of_the_hill) if self.king_of_the_hill else "None"
        print(f"[League Manager] Grading complete for {rec.name}: mu={rec.mu:.2f} (Score: {rec.conservative_rating:.2f}). King of the Hill: {king_name}")

        # Check qualification for Gauntlet Promotion Queue
        if not rec.is_anchor and norm_ckpt != "heuristic" and "latest_model" not in norm_ckpt.lower():
            # Admission is relative for the same reason demotion is: a fixed mu bar
            # assumes a fixed scale. With no established King the bar is the default
            # starting rating, so a debut that has not actively gone backwards competes.
            floor = self._contender_mu_floor()
            admit_mu = floor if floor is not None else (25.0 - self.contender_mu_margin)
            if rec.mu >= admit_mu and rec.points_rate >= 33.4:
                self._admit_contender(norm_ckpt)

        return rec

    def get_hall_of_fame(self) -> List[ModelRating]:
        """
        Returns the strictly bounded Top-K all-time best non-anchor checkpoints,
        gated by sample size and confidence (matches >= min_matches, sigma <= max_sigma).
        Anchors are excluded so that neural network checkpoints receive all K slots.
        """
        historical_ckpts = [
            r for r in self.evaluator.ratings.values()
            if not r.is_anchor and r.path != "heuristic" and "latest_model" not in r.path.lower()
        ]
        # Gate by sample size and Bayesian confidence to prevent debut flukes
        qualified = [
            r for r in historical_ckpts
            if r.matches_played >= self.hall_of_fame_min_matches and r.sigma <= self.hall_of_fame_max_sigma
        ]
        # If fewer qualified than K, backfill with best available non-anchors
        if len(qualified) < self.hall_of_fame_size:
            unqualified = [r for r in historical_ckpts if r not in qualified]
            unqualified.sort(key=lambda r: (self._ranking_key(r), r.matches_played), reverse=True)
            qualified.extend(unqualified[:(self.hall_of_fame_size - len(qualified))])

        qualified.sort(key=lambda r: (self._ranking_key(r), r.matches_played), reverse=True)
        return qualified[:self.hall_of_fame_size]

    def get_protected_checkpoint_paths(self) -> Set[str]:
        """
        Returns a set of canonicalized file paths for top-K checkpoints, active
        Gauntlet contenders, and Top-K Hall of Fame models that must NEVER be pruned.
        Strictly bounded by K to prevent exponential disk file accumulation.
        Uses os.path.normcase to prevent case/slash mismatch bugs on Windows.
        """
        self.refresh_pool()
        protected = set()

        def add_canonical(p: str):
            if p and p != "heuristic" and os.path.exists(p):
                abs_p = os.path.abspath(p)
                protected.add(abs_p)
                protected.add(os.path.normcase(abs_p))

        # 1. Protect current King of the Hill
        if self.king_of_the_hill:
            add_canonical(self.king_of_the_hill)

        # 2. Protect bounded Top-K Hall of Fame Checkpoints (Sample-size gated)
        for hof_rec in self.get_hall_of_fame():
            add_canonical(hof_rec.path)

        # 3. Protect top models in Elite Pool (Anchors do not consume checkpoint quota)
        count = 0
        for path in self.elite_pool:
            norm_p = self._normalize_path(path)
            rec = self.evaluator.ratings.get(norm_p)
            if rec and rec.is_anchor:
                continue
            if path != "heuristic" and os.path.exists(path):
                add_canonical(path)
                count += 1
                if count >= self.protect_top_k:
                    break

        # 4. Protect all active Gauntlet contenders
        for c_path in self.contender_queue:
            add_canonical(c_path)

        # 5. Protect the provisional tier: the newest checkpoints whose rating has not yet
        # converged (sigma above the eligibility threshold). Bounded by max_provisional so
        # this cannot grow without limit; a checkpoint that ages out of the window has had
        # its chance to be measured and is free to be pruned.
        provisional = []
        for norm_p, rec in self.evaluator.ratings.items():
            if rec.is_anchor or norm_p == "heuristic" or "latest_model" in norm_p.lower():
                continue
            if rec.sigma <= self.eligibility_sigma:
                continue
            it = self._checkpoint_iteration(rec.path)
            if it is None or not os.path.exists(rec.path):
                continue
            provisional.append((it, rec.path))

        provisional.sort(reverse=True)
        for _, path in provisional[:self.max_provisional]:
            add_canonical(path)

        return protected

    @staticmethod
    def _checkpoint_iteration(path: str) -> Optional[int]:
        """Parses the iteration number out of a checkpoint_iter_*.pt path, or None."""
        m = re.search(r"checkpoint_iter_(\d+)", str(path).replace("\\", "/"))
        return int(m.group(1)) if m else None

    def get_stratified_distribution(
        self,
        num_envs: int,
        self_play_ratio: Optional[float] = None,
        king_ratio: Optional[float] = None,
        pool_ratio: Optional[float] = None
    ) -> List[Optional[str]]:
        """
        Computes the opponent bot assignment for each vectorized environment (0..num_envs-1).
        - None = Pure Self-Play (Current Learner vs Current Learner)
        - str path = Current Learner vs Specified Opponent Bot
        """
        if not self.enabled:
            return [None] * num_envs

        self.refresh_pool()

        # Handle num_envs < 4 via Temporal Alternation
        if num_envs < 4:
            self._temporal_cycle_idx = (self._temporal_cycle_idx + 1) % 3
            if self._temporal_cycle_idx == 0:
                # Self-Play
                return [None] * num_envs
            elif self._temporal_cycle_idx == 1:
                # King of the Hill, or the pool when the throne is empty.
                king = self.king_of_the_hill
                if not king:
                    king = random.choice(self.elite_pool) if self.elite_pool else "heuristic"
                return [king] * num_envs
            else:
                # Pool
                pool_choice = random.choice(self.elite_pool) if self.elite_pool else "heuristic"
                return [pool_choice] * num_envs

        # Normalize ratios
        sp_r = self.self_play_ratio if self_play_ratio is None else self_play_ratio
        k_r = self.king_ratio if king_ratio is None else king_ratio
        p_r = self.pool_ratio if pool_ratio is None else pool_ratio

        total_r = sp_r + k_r + p_r
        if total_r > 1e-6:
            sp_r /= total_r
            k_r /= total_r
            p_r /= total_r
        else:
            sp_r, k_r, p_r = 0.50, 0.25, 0.25

        # Explicit training opponents take their share off the top; whatever remains is
        # divided by the standard ratios. So the list squashes the default split rather
        # than competing inside one tier, and a ratio of 0.0 reproduces the old
        # behaviour exactly.
        live_training_opps = [
            op for op in self.training_opponents
            if op == "heuristic" or os.path.exists(op)
        ]
        n_training = 0
        if live_training_opps and self.training_opponent_ratio > 0.0:
            n_training = min(num_envs, int(round(num_envs * self.training_opponent_ratio)))

        remaining = max(0, num_envs - n_training)
        n_self_play = max(1, int(round(remaining * sp_r))) if remaining else 0
        n_king = max(1, int(round(remaining * k_r))) if remaining else 0
        n_pool = max(0, remaining - n_self_play - n_king)

        # Adjust in case rounding exceeded the remaining budget
        while (n_self_play + n_king + n_pool) > remaining:
            if n_self_play > 1:
                n_self_play -= 1
            elif n_king > 1:
                n_king -= 1
            else:
                n_pool = max(0, n_pool - 1)

        # No King means no rated checkpoint yet, which means the pool holds nothing but
        # anchors. Handing them half the environments is worse than useless: the ladder
        # spans the BC baseline, which loses 100-0 to everything, and Nexto, which beats
        # the current policy four times out of five. Neither is a useful sparring
        # partner, and between them they would supply the entire training signal.
        #
        # Fall back to pure self-play for both tiers until the league has real data. The
        # explicit training-opponent share is kept, because that is a deliberate choice
        # by the operator rather than a default the league picked on its own.
        if not self.king_of_the_hill:
            n_self_play += n_king + n_pool
            n_king = 0
            n_pool = 0

        # Snap every tier boundary onto a whole env-worker slice.
        #
        # A worker owns a contiguous run of environments, and a rollout step does not
        # finish until every worker returns. So a tier that ends mid-slice leaves one
        # worker holding two opponent models, paying an unbatched forward pass on every
        # step, and setting the pace for all the environments rather than just its own.
        #
        # Ratios land off-boundary easily. At 128 environments a fixed-opponent share of
        # 0.20 asks for 25.6, and rounding that to 26 pushed five of sixteen workers off
        # their boundary, with every environment paying for it. Rounding to the quantum
        # moves a tier by at most half a slice, which is a far smaller distortion of the
        # requested mix than the throughput it costs to honour the ratio exactly.
        #
        # pool_group_size is the quantum because it is already required to equal
        # environments per worker; the pool tier depends on that same equality.
        n_training, n_self_play, n_king, n_pool = snap_tiers_to_worker_slices(
            num_envs, n_training, n_self_play, n_king, n_pool, self.pool_group_size
        )

        assignments: List[Optional[str]] = []

        # 0. Tier 0: Explicit training opponents, split evenly.
        #
        # Allocated by even division rather than by pool_group_size runs: with 6 envs and
        # two opponents, contiguous runs of 4 would hand one of them 4 and the other 2.
        # Each opponent's block is still contiguous, so opponent inference stays batched.
        # Any remainder rotates between refreshes so it does not always favour the same
        # entry.
        if n_training and live_training_opps:
            k = len(live_training_opps)
            base, extra = divmod(n_training, k)
            rotated = [
                live_training_opps[(self._training_opp_cycle_idx + i) % k] for i in range(k)
            ]
            for i, opp in enumerate(rotated):
                assignments.extend([opp] * (base + (1 if i < extra else 0)))
            if extra:
                self._training_opp_cycle_idx = (self._training_opp_cycle_idx + extra) % k

        # 1. Tier 1: Self-Play (None)
        assignments.extend([None] * n_self_play)

        # 2. Tier 2: King of the Hill. n_king is zero when the throne is empty.
        if n_king and self.king_of_the_hill:
            assignments.extend([self.king_of_the_hill] * n_king)

        # 3. Tier 3: Historical Diversity Pool (round-robin rotation through elite pool and anchors)
        # Anything on the explicit training-opponent list is excluded here: it already
        # holds a guaranteed share, and letting it also draw pool slots would make its
        # real exposure the ratio PLUS whatever the round-robin hands it. Keeping the
        # two disjoint is what makes the ratio mean what it says -- 0.10 across two
        # listed opponents is 5% each, not 5% plus pool spillover.
        explicit = set(live_training_opps)
        candidate_pool = [m for m in dict.fromkeys(self.elite_pool) if m not in explicit]

        # Contiguous runs of `pool_group_size` envs per model, rather than one env each:
        # batches the opponent forward passes and keeps each subprocess env worker facing
        # only a couple of distinct models. The cycle index still advances, so successive
        # refreshes walk the whole pool.
        # An empty pool means no rated checkpoint is available to spar with. Self-play is
        # the right fallback: it is always well matched, and it costs less per step than
        # any opponent model.
        if not candidate_pool:
            assignments.extend([None] * n_pool)
            n_pool = 0

        assigned = 0
        while assigned < n_pool:
            choice = candidate_pool[self._pool_cycle_idx % len(candidate_pool)]
            self._pool_cycle_idx = (self._pool_cycle_idx + 1) % len(candidate_pool)
            run = min(self.pool_group_size, n_pool - assigned)
            assignments.extend([choice] * run)
            assigned += run

        return assignments

    def get_telemetry(self) -> Dict[str, Any]:
        """Provides high-level league metrics for streaming to JSON logs and Gradio UI."""
        king_rec = self.evaluator.ratings.get(self.king_of_the_hill) if self.king_of_the_hill else None
        return {
            "league_enabled": self.enabled,
            "king_of_the_hill": get_model_display_name(self.king_of_the_hill) if self.king_of_the_hill else "None",
            "king_score": round(king_rec.conservative_rating, 2) if king_rec else 0.0,
            "king_mu": round(king_rec.mu, 2) if king_rec else 25.0,
            "king_sigma": round(king_rec.sigma, 2) if king_rec else 8.33,
            "elite_pool_size": len(self.elite_pool),
            "protected_checkpoints_count": len(self.get_protected_checkpoint_paths()),
            "benchmark_results": self.benchmark_results,
            "contender_queue_count": len(self.contender_queue),
            "contenders": self.get_contender_queue_details(),
            "recent_events": self.event_history[-5:]
        }
