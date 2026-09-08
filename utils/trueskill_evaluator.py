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
from env.baseline_agent import (
    BaseOpponent, BaselineChaser, CheckpointOpponentBot, NectoNextoOpponentBot, create_opponent_bot
)

DEFAULT_LEADERBOARD_PATH = "logs/trueskill_leaderboard.json"

# TrueSkill Global Configuration
# Rocket League 1v1 typically has low draw rates when overtime is enabled
ts_env = trueskill.TrueSkill(
    mu=25.0,
    sigma=25.0 / 3.0,     # ~8.333
    beta=25.0 / 6.0,      # ~4.167
    tau=25.0 / 300.0,     # ~0.0833
    draw_probability=0.05
)


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
    last_updated: str = ""

    def update_conservative(self):
        self.conservative_rating = round(self.mu - 3.0 * self.sigma, 2)
        self.goal_diff = self.goals_for - self.goals_against
        self.win_rate = round((self.wins / max(1, self.matches_played)) * 100.0, 1)

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


def simulate_headless_match(
    blue_bot: BaseOpponent,
    orange_bot: BaseOpponent,
    max_steps: int = 400,
    enable_overtime: bool = True,
    max_ot_steps: int = 200,
    dt: float = 1.0 / 15.0
) -> Dict[str, Any]:
    """
    Simulates a fast headless 1v1 match between Blue and Orange bots without rendering overhead.
    Correctly applies bot_mask for external TorchScript bots and resolves ties with sudden-death overtime.
    """
    arena = RocketSimArena(num_players=2, game_mode="1v1")
    arena.reset(random_kickoff=False)

    # External bots (Necto/Nexto) must bypass SensAI jump sequencer via bot_mask
    bot_mask = [
        isinstance(blue_bot, NectoNextoOpponentBot),
        isinstance(orange_bot, NectoNextoOpponentBot)
    ]

    blue_goals = 0
    orange_goals = 0
    ot_steps_taken = 0
    went_to_overtime = False

    # 1. Regulation Period
    for step in range(max_steps):
        act0 = blue_bot.get_action(arena.cars[0], arena)
        act1 = orange_bot.get_action(arena.cars[1], arena)

        goal, scoring_team = arena.step([act0, act1], dt=dt, bot_mask=bot_mask)
        if goal:
            if scoring_team == 0:
                blue_goals += 1
            else:
                orange_goals += 1
            arena.reset(random_kickoff=True)

    # 2. Sudden-Death Golden Goal Overtime (if tied)
    if enable_overtime and blue_goals == orange_goals:
        went_to_overtime = True
        arena.reset(random_kickoff=False)
        for ot_step in range(max_ot_steps):
            ot_steps_taken = ot_step + 1
            act0 = blue_bot.get_action(arena.cars[0], arena)
            act1 = orange_bot.get_action(arena.cars[1], arena)

            goal, scoring_team = arena.step([act0, act1], dt=dt, bot_mask=bot_mask)
            if goal:
                if scoring_team == 0:
                    blue_goals += 1
                else:
                    orange_goals += 1
                break

    # Outcome
    if blue_goals > orange_goals:
        winner = "blue"
    elif orange_goals > blue_goals:
        winner = "orange"
    else:
        winner = "draw"

    return {
        "blue_goals": blue_goals,
        "orange_goals": orange_goals,
        "blue_touches": arena.cars[0].ball_touches,
        "orange_touches": arena.cars[1].ball_touches,
        "winner": winner,
        "overtime": went_to_overtime,
        "overtime_steps": ot_steps_taken,
        "total_steps": max_steps + ot_steps_taken
    }


class TrueSkillEvaluator:
    """
    Manages TrueSkill ratings, tournament matchmaking, persistence, and reporting.
    """
    def __init__(self, leaderboard_path: str = DEFAULT_LEADERBOARD_PATH):
        self.leaderboard_path = leaderboard_path
        self.ratings: Dict[str, ModelRating] = {}
        self.match_history: List[Dict[str, Any]] = []
        self.load_leaderboard()

    def get_or_create_rating(self, model_spec: str, is_anchor: bool = False) -> ModelRating:
        """Retrieves or creates a rating record for a given model path or identifier."""
        norm_key = os.path.normpath(model_spec.strip().strip('"').strip("'")).replace("\\", "/")
        name = get_model_display_name(norm_key)

        # Match by key or by display name
        for k, r in self.ratings.items():
            if k == norm_key or r.name == name or os.path.basename(k) == os.path.basename(norm_key):
                return r

        record = ModelRating(
            name=name,
            path=norm_key,
            mu=25.0,
            sigma=25.0 / 3.0,
            is_anchor=is_anchor,
            last_updated=datetime.datetime.now().isoformat()
        )
        record.update_conservative()
        self.ratings[norm_key] = record
        return record

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
                record = ModelRating(**v)
                record.update_conservative()
                self.ratings[k] = record
            self.match_history = data.get("history", [])
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
        matches_per_pair: int = 2,
        max_steps: int = 400,
        enable_overtime: bool = True,
        device: str = "cpu"
    ) -> List[Dict[str, Any]]:
        """
        Runs an even number of matches between two models, alternating Blue and Orange
        sides to guarantee zero spawn or side bias.
        """
        # Ensure even number of matches for perfect side symmetry
        if matches_per_pair < 2:
            matches_per_pair = 2
        elif matches_per_pair % 2 != 0:
            matches_per_pair += 1

        record_a = self.get_or_create_rating(model_a_path)
        record_b = self.get_or_create_rating(model_b_path)

        bot_a = create_opponent_bot(model_a_path, device=device)
        bot_b = create_opponent_bot(model_b_path, device=device)

        match_results = []

        for m_idx in range(matches_per_pair):
            # Alternating sides: Game 0 (A Blue, B Orange), Game 1 (B Blue, A Orange)...
            if m_idx % 2 == 0:
                blue_bot, orange_bot = bot_a, bot_b
                blue_rec, orange_rec = record_a, record_b
                blue_is_a = True
            else:
                blue_bot, orange_bot = bot_b, bot_a
                blue_rec, orange_rec = record_b, record_a
                blue_is_a = False

            res = simulate_headless_match(
                blue_bot=blue_bot,
                orange_bot=orange_bot,
                max_steps=max_steps,
                enable_overtime=enable_overtime
            )

            # Update scores
            bg, og = res["blue_goals"], res["orange_goals"]
            blue_rec.goals_for += bg
            blue_rec.goals_against += og
            orange_rec.goals_for += og
            orange_rec.goals_against += bg

            blue_rec.matches_played += 1
            orange_rec.matches_played += 1

            r_blue = blue_rec.to_trueskill_rating()
            r_orange = orange_rec.to_trueskill_rating()

            if res["winner"] == "blue":
                blue_rec.wins += 1
                orange_rec.losses += 1
                new_blue, new_orange = trueskill.rate_1vs1(r_blue, r_orange)
                if not blue_rec.is_anchor:
                    blue_rec.from_trueskill_rating(new_blue)
                if not orange_rec.is_anchor:
                    orange_rec.from_trueskill_rating(new_orange)
            elif res["winner"] == "orange":
                orange_rec.wins += 1
                blue_rec.losses += 1
                new_orange, new_blue = trueskill.rate_1vs1(r_orange, r_blue)
                if not orange_rec.is_anchor:
                    orange_rec.from_trueskill_rating(new_orange)
                if not blue_rec.is_anchor:
                    blue_rec.from_trueskill_rating(new_blue)
            else:
                blue_rec.draws += 1
                orange_rec.draws += 1
                new_blue, new_orange = trueskill.rate_1vs1(r_blue, r_orange, drawn=True)
                if not blue_rec.is_anchor:
                    blue_rec.from_trueskill_rating(new_blue)
                if not orange_rec.is_anchor:
                    orange_rec.from_trueskill_rating(new_orange)

            blue_rec.update_conservative()
            orange_rec.update_conservative()

            res["blue_name"] = blue_rec.name
            res["orange_name"] = orange_rec.name
            res["blue_mu"] = blue_rec.mu
            res["orange_mu"] = orange_rec.mu
            res["winner_name"] = blue_rec.name if res["winner"] == "blue" else (orange_rec.name if res["winner"] == "orange" else "Draw")
            match_results.append(res)
            self.match_history.append(res)

        self.save_leaderboard()
        return match_results

    def run_tournament(
        self,
        model_paths: List[str],
        matches_per_pair: int = 2,
        max_steps: int = 400,
        enable_overtime: bool = True,
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
                matches_per_pair=matches_per_pair,
                max_steps=max_steps,
                enable_overtime=enable_overtime,
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

    def get_leaderboard_dataframe(self, ascii_safe: bool = False) -> pd.DataFrame:
        """Returns a ranked Pandas DataFrame of all models."""
        mu_col = "Rating (mu)" if ascii_safe else "Rating (μ)"
        sigma_col = "Uncertainty (sigma)" if ascii_safe else "Uncertainty (σ)"
        plus_minus = "+/-" if ascii_safe else "±"

        cols = [
            "Rank", "Model", mu_col, sigma_col, "Conservative Score",
            "Win Rate", "Record (W-L-D)", "Goal Diff", "Matches"
        ]

        if not self.ratings:
            return pd.DataFrame(columns=cols)

        # Sort by conservative rating (mu - 3*sigma) descending, then by win rate
        # Ephemeral moving files like latest_model.pt are excluded from ranked TrueSkill standings
        records = [
            r for r in self.ratings.values()
            if "latest_model" not in r.path.lower() and "latest_model" not in r.name.lower()
        ]
        records.sort(
            key=lambda r: (r.conservative_rating, r.win_rate, r.mu),
            reverse=True
        )

        rows = []
        for rank, r in enumerate(records, start=1):
            rows.append({
                "Rank": f"#{rank}",
                "Model": r.name,
                mu_col: f"{r.mu:.2f}",
                sigma_col: f"{plus_minus}{r.sigma:.2f}",
                "Conservative Score": f"{r.conservative_rating:.2f}",
                "Win Rate": f"{r.win_rate:.1f}%",
                "Record (W-L-D)": f"{r.wins}-{r.losses}-{r.draws}",
                "Goal Diff": f"{r.goal_diff:+d}",
                "Matches": r.matches_played
            })

        return pd.DataFrame(rows)

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
