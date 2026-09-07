"""
League Manager for SenseiBot.
Manages automated Bayesian grading of saved policy checkpoints, King-of-the-Hill tracking,
elite pool curation, and stratified vectorized environment distribution (Option B).
"""

from __future__ import annotations
import os
import random
from typing import Dict, Any, List, Optional, Set, Tuple

from utils.trueskill_evaluator import (
    TrueSkillEvaluator, ModelRating, get_model_display_name, DEFAULT_LEADERBOARD_PATH
)


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
        self.protect_top_k = int(self.config.get("protect_top_k", 5))
        self.eval_matches_per_grade = int(self.config.get("eval_matches_per_grade", 2))
        self.eval_max_steps = int(self.config.get("eval_max_steps", 400))

        # Evaluator
        self.evaluator = evaluator or TrueSkillEvaluator(leaderboard_path=leaderboard_path)

        # Anchors
        default_anchors = [
            "checkpoints/pretrained_baseline.pt",
            "checkpoints/necto-model.pt",
            "heuristic"
        ]
        self.anchor_candidates = self.config.get("anchors", default_anchors)
        self.active_anchors: List[str] = []

        # League state
        self.king_of_the_hill: Optional[str] = None
        self.elite_pool: List[str] = []
        self._temporal_cycle_idx: int = 0

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

    def refresh_pool(self):
        """
        Scans current ratings, filters out deleted models, and identifies:
        1. King of the Hill (highest conservative score: mu - 3*sigma)
        2. Elite Pool (top max_pool_size models)
        """
        valid_models: List[Tuple[str, float, ModelRating]] = []

        for key, rec in self.evaluator.ratings.items():
            norm_key = self._normalize_path(key)
            # Anchors are valid
            if rec.is_anchor or norm_key == "heuristic":
                valid_models.append((norm_key, rec.conservative_rating, rec))
            elif os.path.exists(norm_key):
                valid_models.append((norm_key, rec.conservative_rating, rec))

        if not valid_models:
            self.king_of_the_hill = self.active_anchors[0] if self.active_anchors else "heuristic"
            self.elite_pool = [self.king_of_the_hill]
            return

        # Sort descending: highest conservative rating first, then win rate, then mu
        valid_models.sort(
            key=lambda item: (item[2].conservative_rating, item[2].win_rate, item[2].mu),
            reverse=True
        )

        # Prioritize non-anchor checkpoints for King of the Hill if any exist with reasonable rating
        checkpoints_only = [m for m in valid_models if not m[2].is_anchor and m[0] != "heuristic"]
        if checkpoints_only:
            self.king_of_the_hill = checkpoints_only[0][0]
        else:
            self.king_of_the_hill = valid_models[0][0]

        # Populate elite pool with top models (mix of checkpoints and anchors)
        pool_paths = []
        for path, _, _ in valid_models:
            if path not in pool_paths:
                pool_paths.append(path)
            if len(pool_paths) >= self.max_pool_size:
                break

        self.elite_pool = pool_paths

    def grade_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> ModelRating:
        """
        Automatically grades a newly saved checkpoint in fast headless matches against:
        1. The reigning King of the Hill.
        2. A reference anchor baseline.
        
        Updates ratings in logs/trueskill_leaderboard.json and updates King-of-the-Hill status.
        """
        norm_ckpt = self._normalize_path(checkpoint_path)
        if not os.path.exists(norm_ckpt):
            print(f"[League Manager] Warning: Checkpoint not found on disk: {checkpoint_path}")
            return self.evaluator.get_or_create_rating(norm_ckpt)

        # Ensure model rating record exists
        rec = self.evaluator.get_or_create_rating(norm_ckpt)

        # Determine benchmark opponents
        opponents_to_test: List[str] = []
        if self.king_of_the_hill and self.king_of_the_hill != norm_ckpt:
            opponents_to_test.append(self.king_of_the_hill)

        # Pick best anchor not already included
        for anc in self.active_anchors:
            if anc != norm_ckpt and anc not in opponents_to_test:
                opponents_to_test.append(anc)
                break

        # If no opponents found, test against heuristic
        if not opponents_to_test:
            opponents_to_test.append("heuristic")

        print(f"[League Manager] Auto-Grading Checkpoint '{rec.name}' against {len(opponents_to_test)} opponent(s)...")

        for opp in opponents_to_test:
            opp_name = get_model_display_name(opp)
            print(f"  -> Matchup: {rec.name} vs {opp_name} ({self.eval_matches_per_grade} games)")
            try:
                self.evaluator.evaluate_pairing(
                    model_a_path=norm_ckpt,
                    model_b_path=opp,
                    matches_per_pair=self.eval_matches_per_grade,
                    max_steps=self.eval_max_steps,
                    enable_overtime=True,
                    device=device
                )
            except Exception as e:
                print(f"[League Manager] Error evaluating {norm_ckpt} vs {opp}: {e}")

        # Refresh state
        self.refresh_pool()
        king_name = get_model_display_name(self.king_of_the_hill) if self.king_of_the_hill else "None"
        print(f"[League Manager] Grading complete for {rec.name}: mu={rec.mu:.2f} (Score: {rec.conservative_rating:.2f}). King of the Hill: {king_name}")

        return rec

    def get_protected_checkpoint_paths(self) -> Set[str]:
        """
        Returns a set of normalized file paths for top-K checkpoints that must
        NEVER be pruned by rolling checkpoint cleanups.
        """
        self.refresh_pool()
        protected = set()

        if self.king_of_the_hill and os.path.exists(self.king_of_the_hill):
            protected.add(os.path.abspath(self.king_of_the_hill))

        count = 0
        for path in self.elite_pool:
            if path != "heuristic" and os.path.exists(path):
                protected.add(os.path.abspath(path))
                count += 1
                if count >= self.protect_top_k:
                    break

        return protected

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
                # King of the Hill
                king = self.king_of_the_hill or (self.active_anchors[0] if self.active_anchors else "heuristic")
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

        n_self_play = max(1, int(round(num_envs * sp_r)))
        n_king = max(1, int(round(num_envs * k_r)))
        n_pool = max(0, num_envs - n_self_play - n_king)

        # Adjust in case rounding exceeded num_envs
        while (n_self_play + n_king + n_pool) > num_envs:
            if n_self_play > 1:
                n_self_play -= 1
            elif n_king > 1:
                n_king -= 1
            else:
                n_pool = max(0, n_pool - 1)

        assignments: List[Optional[str]] = []

        # 1. Tier 1: Self-Play (None)
        assignments.extend([None] * n_self_play)

        # 2. Tier 2: King of the Hill
        king_bot = self.king_of_the_hill or (self.active_anchors[0] if self.active_anchors else "heuristic")
        assignments.extend([king_bot] * n_king)

        # 3. Tier 3: Historical Diversity Pool (sampled from elite pool and anchors)
        candidate_pool = list(dict.fromkeys(self.elite_pool + self.active_anchors))
        if not candidate_pool:
            candidate_pool = ["heuristic"]

        for _ in range(n_pool):
            choice = random.choice(candidate_pool)
            assignments.append(choice)

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
            "protected_checkpoints_count": len(self.get_protected_checkpoint_paths())
        }
