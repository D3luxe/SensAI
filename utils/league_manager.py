"""
League Manager for SenseiBot.
Manages automated Bayesian grading of saved policy checkpoints, King-of-the-Hill tracking,
elite pool curation, and stratified vectorized environment distribution (Option B).
"""

from __future__ import annotations
import os
import random
import json
import datetime
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
        self.protect_top_k = int(self.config.get("protect_top_k", 20))
        self.hall_of_fame_size = int(self.config.get("hall_of_fame_size", 20))
        self.hall_of_fame_min_matches = int(self.config.get("hall_of_fame_min_matches", 16))
        self.hall_of_fame_max_sigma = float(self.config.get("hall_of_fame_max_sigma", 2.5))
        self.eval_matches_per_grade = int(self.config.get("eval_matches_per_grade", 2))
        self.eval_max_steps = int(self.config.get("eval_max_steps", 400))

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
        default_anchors = [
            "checkpoints/pretrained_baseline.pt",
            "checkpoints/necto-model.pt",
            "checkpoints/nexto-model.pt",
            "heuristic"
        ]
        self.anchor_candidates = self.config.get("anchors", default_anchors)
        self.active_anchors: List[str] = []

        # Gauntlet Contender Queue Configuration
        self.min_contender_mu = float(self.config.get("min_contender_mu", 25.5))
        self.min_contender_win_rate = float(self.config.get("min_contender_win_rate", 45.0))
        self.max_consecutive_losses = int(self.config.get("max_consecutive_losses", 4))
        self.grace_period_matches = int(self.config.get("grace_period_matches", 6))
        self.target_eval_matches = int(self.config.get("target_eval_matches", 16))
        self.max_contender_matches = int(self.config.get("max_contender_matches", 32))
        self.target_eval_sigma = float(self.config.get("target_eval_sigma", 1.8))
        self.king_challenge_frequency = int(self.config.get("king_challenge_frequency", 2))
        self.max_active_contenders = int(self.config.get("max_active_contenders", 3))
        self.contender_eval_matches_per_step = int(self.config.get("contender_eval_matches_per_step", 2))

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

    def _compute_competitive_score(self, rec: Optional[ModelRating]) -> float:
        """
        Computes the effective competitive ranking score for King and Elite Pool determination.
        - Models with < target_eval_matches (16): uses conservative lower bound mu - 3*sigma
          to enforce debut protection and prevent unearned ascension on small-sample flukes.
        - Established models (>= 16 matches): uses mu - 2*sigma (95% confidence lower bound)
          to eliminate the sample-size penalty trap where incumbents with 800+ matches
          block vastly superior challengers with 16-32 matches.
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
                (rec.mu >= king_mu - 1.0 or rec.win_rate >= 50.0)
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
            "contenders": self.get_contender_queue_details()
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
            self.king_of_the_hill = self.active_anchors[0] if self.active_anchors else "heuristic"
            self.elite_pool = [self.king_of_the_hill]
            return

        # Sort descending: highest competitive score first, then conservative rating, then win rate, then mu
        valid_models.sort(
            key=lambda item: (item[1], item[2].conservative_rating, item[2].win_rate, item[2].mu),
            reverse=True
        )

        # Prioritize non-anchor checkpoints for King of the Hill if any exist with reasonable rating
        checkpoints_only = [m for m in valid_models if not m[2].is_anchor and m[0] != "heuristic"]
        if checkpoints_only:
            self.king_of_the_hill = checkpoints_only[0][0]
        else:
            self.king_of_the_hill = valid_models[0][0]

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

        # Populate elite pool with top models (mix of checkpoints and anchors)
        pool_paths = []
        for path, _, _ in valid_models:
            if path not in pool_paths:
                pool_paths.append(path)
            if len(pool_paths) >= self.max_pool_size:
                break

        self.elite_pool = pool_paths

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
            print(f"[League Manager] [Gauntlet Admission] Admitted '{rec.name}' to Contender Queue (mu={rec.mu:.2f}, WR={rec.win_rate:.1f}%)")
            self._record_event(
                event_type="admission",
                model_name=rec.name,
                detail=f"Admitted to Gauntlet Contender Queue (μ={rec.mu:.2f}, WR={rec.win_rate:.1f}%)",
                extra={"mu": rec.mu, "win_rate": rec.win_rate}
            )
        else:
            # Check preemption: replace the lowest mu contender if new contender has higher mu
            contender_ratings = [(p, self.evaluator.ratings.get(p)) for p in self.contender_queue]
            valid_contenders = [c for c in contender_ratings if c[1] is not None]
            if valid_contenders:
                lowest_path, lowest_rec = min(valid_contenders, key=lambda c: c[1].mu)
                if rec.mu > lowest_rec.mu:
                    self.contender_queue.remove(lowest_path)
                    self.contender_consecutive_losses.pop(lowest_path, None)
                    self.contender_queue.append(ckpt_path)
                    self.contender_consecutive_losses[ckpt_path] = 0
                    print(f"[League Manager] [Gauntlet Preemption] '{rec.name}' (mu={rec.mu:.2f}) replaced '{lowest_rec.name}' (mu={lowest_rec.mu:.2f}) in Gauntlet Queue")
                    self._record_event(
                        event_type="preemption",
                        model_name=rec.name,
                        detail=f"Replaced '{lowest_rec.name}' in Gauntlet Queue (μ={rec.mu:.2f} > {lowest_rec.mu:.2f})",
                        extra={"admitted": rec.name, "bumped": lowest_rec.name, "mu": rec.mu}
                    )

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

        # Pick top challenger by competitive score
        challenger_path = max(
            valid_challengers,
            key=lambda p: self._compute_competitive_score(
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
                matches_per_pair=2,
                max_steps=self.eval_max_steps,
                enable_overtime=True,
                device=device
            )
        except Exception as e:
            print(f"[League Manager] Warning: Title bout error {norm_challenger} vs {norm_king}: {e}")
            return None

        old_king = self.king_of_the_hill
        self.refresh_pool()
        self.save_league_state()

        if self.king_of_the_hill != old_king:
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

        # Pick contender with highest raw skill (mu)
        contender_path = max(
            self.contender_queue,
            key=lambda p: getattr(self.evaluator.ratings.get(p), "mu", 0.0)
        )
        rec = self.evaluator.ratings.get(contender_path)
        if not rec:
            self.contender_queue.remove(contender_path)
            return None

        # Standardized Gauntlet pairing: 1 match vs reference anchor, 1 match vs King
        ref_anchor = None
        for cand in ["checkpoints/necto-model.pt", "checkpoints/nexto-model.pt"] + self.active_anchors:
            norm_c = self._normalize_path(cand)
            if (os.path.exists(norm_c) or norm_c == "heuristic") and norm_c != contender_path:
                ref_anchor = norm_c
                break

        opponents = []
        if ref_anchor:
            opponents.append(ref_anchor)

        king = self.king_of_the_hill
        if king and king != contender_path and king not in opponents and "latest_model" not in king.lower():
            opponents.append(king)
        elif not opponents and self.active_anchors:
            opponents.append(self.active_anchors[0])

        if not opponents:
            return None

        prev_wins = rec.wins
        prev_losses = rec.losses

        print(f"[League Manager] [Gauntlet Trial] Testing contender '{rec.name}' against {len(opponents)} opponent(s)...")
        for opp in opponents:
            try:
                self.evaluator.evaluate_pairing(
                    model_a_path=contender_path,
                    model_b_path=opp,
                    matches_per_pair=max(2, self.contender_eval_matches_per_step // len(opponents)),
                    max_steps=self.eval_max_steps,
                    enable_overtime=True,
                    device=device
                )
            except Exception as e:
                print(f"[League Manager] Warning: Gauntlet trial error {contender_path} vs {opp}: {e}")

        # Refresh rating after matches
        rec = self.evaluator.ratings.get(contender_path, rec)
        new_wins = rec.wins - prev_wins
        new_losses = rec.losses - prev_losses

        if new_losses > new_wins:
            self.contender_consecutive_losses[contender_path] = self.contender_consecutive_losses.get(contender_path, 0) + new_losses
        elif new_wins > 0:
            self.contender_consecutive_losses[contender_path] = max(0, self.contender_consecutive_losses.get(contender_path, 0) - new_wins)

        consec_losses = self.contender_consecutive_losses.get(contender_path, 0)

        king_rec = self.evaluator.ratings.get(self.king_of_the_hill) if self.king_of_the_hill else None
        king_mu = king_rec.mu if king_rec else 25.0

        is_title_contender = (
            (rec.mu >= king_mu - 1.0 or rec.win_rate >= 50.0)
            and rec.matches_played < self.max_contender_matches
            and rec.sigma > self.target_eval_sigma
        )

        # 1. Check Graduation (if qualified by skill and matches)
        should_graduate = False
        if rec.mu >= self.min_contender_mu and rec.win_rate >= self.min_contender_win_rate:
            if rec.matches_played >= self.max_contender_matches or rec.sigma <= self.target_eval_sigma:
                should_graduate = True
            elif rec.matches_played >= self.target_eval_matches and not is_title_contender:
                should_graduate = True

        if should_graduate:
            print(f"[League Manager] [Gauntlet Graduation] '{rec.name}' graduated with established rating: mu={rec.mu:.2f}, sigma={rec.sigma:.2f}, Score={self._compute_competitive_score(rec):.2f} over {rec.matches_played} matches!")
            self.contender_queue.remove(contender_path)
            self.contender_consecutive_losses.pop(contender_path, None)
            self.refresh_pool()
            self._record_event(
                event_type="promotion",
                model_name=rec.name,
                detail=f"Graduated Gauntlet to Elite Pool! (Score: {self._compute_competitive_score(rec):.2f}, WR: {rec.win_rate:.1f}% in {rec.matches_played} matches)",
                extra={"score": self._compute_competitive_score(rec), "matches": rec.matches_played, "win_rate": rec.win_rate}
            )
            return {"status": "graduated", "model": rec.name, "score": self._compute_competitive_score(rec)}

        # 2. Check Demotion (Only after grace period matches)
        should_demote = False
        demote_reason = ""
        if rec.matches_played >= self.grace_period_matches:
            if rec.mu < self.min_contender_mu:
                should_demote = True
                demote_reason = f"Skill floor breached (mu={rec.mu:.2f} < {self.min_contender_mu:.2f})"
            elif rec.win_rate < self.min_contender_win_rate:
                should_demote = True
                demote_reason = f"Win rate dropped below floor (WR={rec.win_rate:.1f}% < {self.min_contender_win_rate:.1f}%)"
            elif consec_losses >= self.max_consecutive_losses:
                should_demote = True
                demote_reason = f"Loss streak knockout ({consec_losses} >= {self.max_consecutive_losses} consecutive losses)"

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

        # Determine benchmark opponents
        opponents_to_test: List[str] = []
        if self.king_of_the_hill and self.king_of_the_hill != norm_ckpt and "latest_model" not in self.king_of_the_hill.lower():
            opponents_to_test.append(self.king_of_the_hill)

        # Pick best anchor not already included
        for anc in self.active_anchors:
            if anc != norm_ckpt and anc not in opponents_to_test and "latest_model" not in anc.lower():
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

        # Check qualification for Gauntlet Promotion Queue
        if not rec.is_anchor and norm_ckpt != "heuristic" and "latest_model" not in norm_ckpt.lower():
            if rec.mu >= 26.0 and rec.win_rate >= 50.0:
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
            unqualified.sort(
                key=lambda r: (r.conservative_rating, r.win_rate, r.mu, r.matches_played),
                reverse=True
            )
            qualified.extend(unqualified[:(self.hall_of_fame_size - len(qualified))])

        qualified.sort(
            key=lambda r: (r.conservative_rating, r.win_rate, r.mu, r.matches_played),
            reverse=True
        )
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

        # 3. Tier 3: Historical Diversity Pool (round-robin rotation through elite pool and anchors)
        candidate_pool = list(dict.fromkeys(self.elite_pool + self.active_anchors))
        if not candidate_pool:
            candidate_pool = ["heuristic"]

        for _ in range(n_pool):
            choice = candidate_pool[self._pool_cycle_idx % len(candidate_pool)]
            self._pool_cycle_idx = (self._pool_cycle_idx + 1) % len(candidate_pool)
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
            "protected_checkpoints_count": len(self.get_protected_checkpoint_paths()),
            "contender_queue_count": len(self.contender_queue),
            "contenders": self.get_contender_queue_details(),
            "recent_events": self.event_history[-5:]
        }
