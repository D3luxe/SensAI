"""
Vectorized PPO Trainer for Rocket League Agents with live dynamic parameter updating and metric streaming.
"""

from __future__ import annotations
import os
import sys
import time
import json
import yaml
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List, Set

from env.rocket_env import VectorizedRocketEnv
from env.observations import OBS_MIRROR_MASK_NP, ACT_MIRROR_MASK_NP
from agent.models import ActorCritic
from utils.league_manager import LeagueManager


def _league_grade_entry(ckpt_path: str, league_cfg: Dict[str, Any], project_root: str):
    """
    Child-process entry point for TrueSkill grading.

    Rebuilds a LeagueManager from disk (its constructor loads the leaderboard and league
    state), grades the checkpoint, advances one gauntlet trial, and exits. Both files are
    written atomically via os.replace, so the parent can keep reading them meanwhile.
    """
    import sys
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    from utils.league_manager import LeagueManager
    try:
        lm = LeagueManager(config=league_cfg)
        lm.grade_checkpoint(ckpt_path, device="cpu")
        lm.step_contender_gauntlet(device="cpu")
    except Exception as e:
        print(f"[League Grading] Failed for {ckpt_path}: {e}")


class PPOTrainer:
    def __init__(
        self,
        config_path: str = "config/default_config.yaml",
        live_config_path: str = "config/live_config.json",
        device: Optional[str] = None
    ):
        self.config_path = config_path
        self.live_config_path = live_config_path

        # Load YAML config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        hp = self.config.get("hyperparameters", {})
        env_cfg = self.config.get("environment", {})
        model_cfg = self.config.get("model", {})
        rew_cfg = self.config.get("rewards", {})
        log_cfg = self.config.get("logging", {})

        self.lr = float(hp.get("learning_rate", 3e-4))
        self.weight_decay = float(hp.get("weight_decay", 1e-4))
        self.gamma = float(hp.get("gamma", 0.99))
        self.gae_lambda = float(hp.get("gae_lambda", 0.95))
        self.clip_range = float(hp.get("clip_range", 0.2))
        self.ent_coef = float(hp.get("ent_coef", 0.01))
        self.vf_coef = float(hp.get("vf_coef", 0.5))
        self.max_grad_norm = float(hp.get("max_grad_norm", 0.5))
        # Rotational (Pitch/Yaw/Roll) exploration ceiling anneal. See ActorCritic.log_std_max.
        self.rot_log_std_ceiling_final = float(hp.get("rot_log_std_ceiling_final", -1.4))
        self.rot_log_std_anneal_iters = int(hp.get("rot_log_std_anneal_iters", 4000))
        self._rot_anneal_start_iter = None
        self._rot_anneal_start_ceiling = None
        self._rot_ceiling_last_logged = None
        self.batch_size = int(hp.get("batch_size", 2048))
        self.mini_batch_size = int(hp.get("mini_batch_size", 256))
        self.n_epochs = int(hp.get("n_epochs", 4))
        self.total_timesteps = int(hp.get("total_timesteps", 1_000_000))
        # Crash-recovery autosave: rewrites latest_model.pt in place. Cheap, frequent,
        # and never creates a new file, so it does not feed the league.
        self.autosave_interval = max(1, int(log_cfg.get("autosave_interval", 20)))
        # League population: how often a new numbered checkpoint is minted and graded.
        # Two checkpoints a few iterations apart are near-identical networks that noisy
        # evaluation cannot separate, so this is deliberately much coarser than autosave.
        self.checkpoint_interval = int(log_cfg.get("checkpoint_interval", hp.get("checkpoint_interval", 200)))
        # Permanent historical spine: one checkpoint every `archive_stride` iterations is
        # retained forever, so old eras survive regardless of current ranking. 0 disables.
        self.archive_stride = max(0, int(log_cfg.get("archive_stride", 5000)))
        # Legacy recency cap. Retention is now the union of the provisional/ranked/archive
        # tiers; this is only consulted when no league manager is attached.
        self.max_checkpoints_to_keep = int(log_cfg.get("max_checkpoints_to_keep", 50))

        self.num_envs = int(env_cfg.get("num_envs", 16))
        self.tick_skip = int(env_cfg.get("tick_skip", 8))
        self.max_episode_steps = int(env_cfg.get("max_episode_steps", 1500))
        self.game_mode = str(env_cfg.get("game_mode", "1v1"))
        self.self_play = bool(env_cfg.get("self_play", True))
        self.continuous_actions = bool(model_cfg.get("continuous_actions", True))
        self.use_layer_norm = bool(model_cfg.get("use_layer_norm", True))
        self.activation = str(model_cfg.get("activation", "leaky_relu"))

        self.save_dir = log_cfg.get("save_dir", "checkpoints")
        self.log_dir = log_cfg.get("log_dir", "logs")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Enable hardware Flush-To-Zero (FTZ) & Denormals-Are-Zero (DAZ)
        # Prevents costly x86 microcode assist traps on decaying weights & Adam moments
        try:
            torch.set_flush_denormal(True)
        except Exception:
            pass

        # Hardware Thread Optimization for PyTorch
        num_threads = int(env_cfg.get("num_threads", 8))
        total_cores = os.cpu_count() or 24
        try:
            torch.set_num_threads(min(total_cores, num_threads))
            torch.set_num_interop_threads(min(4, num_threads))
            print(f"[Hardware Optimizer] PyTorch threads set to {min(total_cores, num_threads)} (Interop: {min(4, num_threads)} | FTZ/DAZ active).")
        except Exception as e:
            print(f"[Hardware Optimizer] PyTorch thread setup note: {e}")


        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[PPO Trainer] Initialized on device: {self.device}")

        # Pre-Flight Physics & Controls Verification Pipeline
        try:
            from test_physics_and_controls import verify_physics_and_controls_pipeline
            verify_physics_and_controls_pipeline(verbose=False)
        except Exception as e:
            raise RuntimeError(f"[PPO Trainer] Pre-Flight Physics Verification Failed: {e}") from e

        # Initialize Vectorized Environment
        self.baseline_opponent_ratio = float(env_cfg.get("baseline_opponent_ratio", 0.25))
        self.baseline_opponent_type = str(env_cfg.get("baseline_opponent_type", "heuristic"))

        # Rollout collection is pure-Python bound and cannot be threaded past the GIL,
        # so distribute environments across worker processes when asked.
        self.num_env_workers = int(env_cfg.get("num_env_workers", 1))
        env_kwargs = dict(
            num_envs=self.num_envs,
            game_mode=self.game_mode,
            tick_skip=self.tick_skip,
            max_episode_steps=self.max_episode_steps,
            reward_weights=rew_cfg,
            continuous_actions=self.continuous_actions,
            self_play=self.self_play,
            baseline_opponent_ratio=self.baseline_opponent_ratio,
            baseline_opponent_type=self.baseline_opponent_type
        )
        if self.num_env_workers > 1:
            from env.subproc_vec_env import SubprocVectorizedRocketEnv
            self.env = SubprocVectorizedRocketEnv(num_workers=self.num_env_workers, **env_kwargs)
        else:
            self.env = VectorizedRocketEnv(**env_kwargs)

        # Initialize League Manager (Stratified Vectorized League Self-Play)
        league_cfg = self.config.get("league", {})
        self.league_manager = LeagueManager(config=league_cfg)
        if self.league_manager.enabled:
            strat_assignments = self.league_manager.get_stratified_distribution(self.num_envs)
            self.env.set_stratified_opponents(strat_assignments)
            print(f"[PPO Trainer] Stratified League Self-Play active across {self.num_envs} envs (King: {self.league_manager.king_of_the_hill})")

        sc_cfg = self.config.get("scenarios", {})
        if sc_cfg:
            self.env.update_scenarios(sc_cfg)

        self.obs_dim = self.env.obs_dim
        self.act_dim = self.env.act_dim
        self.num_agents_per_env = self.env.num_players_per_env
        self.total_actors = self.num_envs * self.num_agents_per_env

        # Steps per rollout
        self.num_steps = max(1, self.batch_size // self.total_actors)

        # Initialize Actor-Critic Model
        self.agent = ActorCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            actor_hidden_dims=model_cfg.get("actor_hidden_dims", [256, 256, 128]),
            critic_hidden_dims=model_cfg.get("critic_hidden_dims", [256, 256, 128]),
            activation=self.activation,
            continuous_actions=self.continuous_actions,
            use_layer_norm=self.use_layer_norm,
            use_action_masking=bool(model_cfg.get("use_action_masking", True)),
            handbrake_height_buffer=float(model_cfg.get("handbrake_height_buffer", 120.0))
        ).to(self.device)
        # Switch to AdamW with decoupled weight decay (0.0 for standard PPO stability, preventing LayerNorm decay collapse)
        self.optimizer = optim.AdamW(self.agent.parameters(), lr=self.lr, eps=1e-5, weight_decay=0.0)

        # Left-Right Mirror Augmentation Masks (Strict Bilateral Symmetry)
        self.obs_mirror_mask = torch.tensor(OBS_MIRROR_MASK_NP, dtype=torch.float32, device=self.device)
        self.act_mirror_mask = torch.tensor(ACT_MIRROR_MASK_NP, dtype=torch.float32, device=self.device)

        # Tensorboard
        self.writer = None
        if log_cfg.get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.log_dir)
            except Exception as e:
                print(f"[PPO Trainer] TensorBoard disabled: {e}")

        # Decaying Behavioral Cloning Regularization
        ppo_cfg = self.config.get("ppo", {})
        self.bc_regularization_weight = float(ppo_cfg.get("bc_regularization_weight", 0.5))
        self.bc_decay_steps = int(ppo_cfg.get("bc_decay_steps", 30_000_000))
        self._bc_dataset_loaded = False
        self.bc_obs_tensor = None
        self.bc_act_tensor = None

        # State tracking
        self.global_step = 0
        self.iteration = 0
        self.last_live_config_mtime = 0.0

        # Background league grading handoff (see _submit_league_grading)
        self._league_proc = None

    def _league_job_running(self) -> bool:
        return self._league_proc is not None and self._league_proc.is_alive()

    def _submit_league_grading(self, ckpt_path: str):
        """
        Grades a freshly saved checkpoint in a separate process.

        Grading plays full evaluation matches, which is pure-Python environment code.
        Running it on a background *thread* traded a hard stall for sustained GIL
        contention with the training loop, costing about 30% of throughput for as long
        as the job ran. A separate process shares no interpreter lock and lands on one
        of the cores the env workers are not using. Only one job runs at a time.
        """
        import multiprocessing as mp

        if self._league_job_running():
            print("[PPO Trainer] League grading still in flight; skipping this interval.")
            return

        ctx = mp.get_context("spawn")
        self._league_proc = ctx.Process(
            target=_league_grade_entry,
            args=(ckpt_path, self.config.get("league", {}),
                  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            daemon=True,
        )
        self._league_proc.start()

    def _apply_pending_league_update(self):
        """
        Picks up the results of a finished grading process.

        The child persists ratings and league state to disk, so the parent re-reads
        those files rather than shipping objects back across the process boundary.
        """
        proc = self._league_proc
        if proc is None or proc.is_alive():
            return
        self._league_proc = None
        proc.join(timeout=5.0)

        try:
            self.league_manager.evaluator.load_leaderboard()
            self.league_manager.load_league_state()
            self.league_manager.refresh_pool()
            # Retention runs now, not during grading, so it cannot delete a checkpoint
            # while the child is playing matches against it.
            self.cleanup_old_checkpoints(max_to_keep=self.max_checkpoints_to_keep)
            strat = self.league_manager.get_stratified_distribution(self.num_envs)
            self.env.set_stratified_opponents(strat)
            print("[PPO Trainer] Applied refreshed league stratification from background grading.")
        except Exception as e:
            print(f"[PPO Trainer] Could not apply league grading results: {e}")

    def _ensure_bc_dataset(self):
        if self._bc_dataset_loaded:
            return
        self._bc_dataset_loaded = True
        try:
            from agent.pretrainer import BehavioralCloningTrainer
            bc_trainer = BehavioralCloningTrainer()
            obs, act = bc_trainer.generate_pretrain_dataset(max_samples=20000)
            if len(obs) > 0:
                self.bc_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                self.bc_act_tensor = torch.tensor(act, dtype=torch.float32, device=self.device)
                print(f"[PPO Trainer] Decaying BC Regularization: Loaded {len(obs):,} human replay frames (Weight: {self.bc_regularization_weight}, Decay Horizon: {self.bc_decay_steps:,} steps)")
        except Exception as e:
            print(f"[PPO Trainer] Warning: Could not initialize BC dataset: {e}")

    def _step_rot_log_std_anneal(self):
        """
        Walk the Pitch/Yaw/Roll exploration ceiling from wherever the policy currently sits
        down to rot_log_std_ceiling_final over rot_log_std_anneal_iters iterations.

        These axes are masked while grounded, so they see policy gradient on only the
        airborne fraction of steps while the entropy bonus pushes on all of them. That
        asymmetry parks them at the ceiling, and at the ground ceiling of -0.7 that is
        sigma ~0.50 of analog noise on every aerial tick. Stepping the ceiling down
        gradually lets the already-trained mean actually be executed without yanking the
        action distribution out from under the value function in one iteration.
        """
        agent = self.agent
        if not self.continuous_actions or not hasattr(agent, "set_rot_log_std_ceiling"):
            return

        if self._rot_anneal_start_iter is None:
            self._rot_anneal_start_iter = self.iteration
            self._rot_anneal_start_ceiling = float(agent.log_std_ceiling_rot)

        start_c = float(self._rot_anneal_start_ceiling)
        final_c = float(self.rot_log_std_ceiling_final)
        span = max(1, int(self.rot_log_std_anneal_iters))
        progress = min(1.0, max(0.0, (self.iteration - self._rot_anneal_start_iter) / span))
        target = start_c + (final_c - start_c) * progress

        applied = agent.set_rot_log_std_ceiling(target)
        moved = self._rot_ceiling_last_logged is None or abs(applied - self._rot_ceiling_last_logged) >= 0.05
        finished = progress >= 1.0 and self._rot_ceiling_last_logged != applied
        if moved or finished:
            sigma = float(math.exp(applied))
            print(f"[PPO Trainer] Rotational exploration ceiling: log_std {applied:.3f} (sigma {sigma:.3f}) "
                  f"| anneal {progress * 100.0:.0f}% toward {final_c:.2f}")
            self._rot_ceiling_last_logged = applied

    def check_live_config(self):
        """
        Dynamically reload hyperparameters and reward weights from live_config.json.
        """
        if not os.path.exists(self.live_config_path):
            return

        try:
            mtime = os.path.getmtime(self.live_config_path)
            if mtime > self.last_live_config_mtime:
                self.last_live_config_mtime = mtime
                with open(self.live_config_path, "r") as f:
                    live = json.load(f)

                # Update learning rate
                if "learning_rate" in live and float(live["learning_rate"]) != self.lr:
                    self.lr = float(live["learning_rate"])
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.lr
                    print(f"[Live Config] Learning rate updated to: {self.lr}")

                # Update entropy coef & clip range
                if "ent_coef" in live:
                    self.ent_coef = float(live["ent_coef"])
                if "clip_range" in live:
                    self.clip_range = float(live["clip_range"])
                if "rot_log_std_ceiling_final" in live:
                    target = float(live["rot_log_std_ceiling_final"])
                    if target != self.rot_log_std_ceiling_final:
                        self.rot_log_std_ceiling_final = target
                        # Re-anchor so a retarget anneals from where the policy is now
                        self._rot_anneal_start_iter = None
                        print(f"[Live Config] Rotational log_std ceiling target updated to: {target}")
                if "rot_log_std_anneal_iters" in live:
                    self.rot_log_std_anneal_iters = int(live["rot_log_std_anneal_iters"])

                # Update Behavioral Cloning (BC) replay regularization
                if "bc_regularization_weight" in live:
                    self.bc_regularization_weight = float(live["bc_regularization_weight"])
                if "bc_decay_steps" in live:
                    self.bc_decay_steps = int(live["bc_decay_steps"])

                # Update rewards
                if "rewards" in live and isinstance(live["rewards"], dict):
                    self.env.update_reward_weights(live["rewards"])
                    print(f"[Live Config] Reward weights dynamically updated.")

                # Update scenario distributions
                if "scenarios" in live and isinstance(live["scenarios"], dict):
                    self.env.update_scenarios(live["scenarios"])
                    print(f"[Live Config] Scenario distributions dynamically updated.")

                # Update baseline opponent ratio & bot type
                ratio_changed = False
                type_changed = False
                if "baseline_opponent_ratio" in live:
                    new_ratio = float(live["baseline_opponent_ratio"])
                    if abs(new_ratio - getattr(self, "baseline_opponent_ratio", 0.25)) > 1e-4:
                        self.baseline_opponent_ratio = new_ratio
                        ratio_changed = True
                
                new_opp_type = live.get("baseline_opponent_type", live.get("baseline_opponent_model", None))
                if new_opp_type is not None and str(new_opp_type) != getattr(self, "baseline_opponent_type", "heuristic"):
                    self.baseline_opponent_type = str(new_opp_type)
                    type_changed = True

                if ratio_changed or type_changed:
                    if not (hasattr(self, "league_manager") and self.league_manager.enabled):
                        self.env.update_baseline_opponent(self.baseline_opponent_ratio, self.baseline_opponent_type)
                        print(f"[Live Config] Opponent bot dynamically updated: Ratio={self.baseline_opponent_ratio:.2f}, Type='{self.baseline_opponent_type}'")

                # Update League Manager dynamic parameters
                if hasattr(self, "league_manager"):
                    league_changed = False
                    if "league_enabled" in live and bool(live["league_enabled"]) != self.league_manager.enabled:
                        self.league_manager.enabled = bool(live["league_enabled"])
                        league_changed = True
                    if "self_play_ratio" in live and abs(float(live["self_play_ratio"]) - self.league_manager.self_play_ratio) > 1e-4:
                        self.league_manager.self_play_ratio = float(live["self_play_ratio"])
                        league_changed = True
                    if "king_ratio" in live and abs(float(live["king_ratio"]) - self.league_manager.king_ratio) > 1e-4:
                        self.league_manager.king_ratio = float(live["king_ratio"])
                        league_changed = True
                    if "pool_ratio" in live and abs(float(live["pool_ratio"]) - self.league_manager.pool_ratio) > 1e-4:
                        self.league_manager.pool_ratio = float(live["pool_ratio"])
                        league_changed = True
                    if "training_opponents" in live:
                        new_list = [
                            self.league_manager._normalize_path(x)
                            for x in (live["training_opponents"] or []) if x
                        ]
                        if new_list != self.league_manager.training_opponents:
                            self.league_manager.training_opponents = new_list
                            league_changed = True
                    if "training_opponent_ratio" in live:
                        new_r = max(0.0, min(1.0, float(live["training_opponent_ratio"])))
                        if abs(new_r - self.league_manager.training_opponent_ratio) > 1e-4:
                            self.league_manager.training_opponent_ratio = new_r
                            league_changed = True

                    if league_changed:
                        if self.league_manager.enabled:
                            strat_assignments = self.league_manager.get_stratified_distribution(self.num_envs)
                            self.env.set_stratified_opponents(strat_assignments)
                            fixed = self.league_manager.training_opponents
                            fixed_note = (
                                f", Fixed={len(fixed)}@{self.league_manager.training_opponent_ratio:.2f}"
                                if fixed and self.league_manager.training_opponent_ratio > 0 else ""
                            )
                            print(f"[Live Config] Stratified League reconfigured: SP={self.league_manager.self_play_ratio:.2f}, King={self.league_manager.king_ratio:.2f}, Pool={self.league_manager.pool_ratio:.2f}{fixed_note}")
                        else:
                            self.env.update_baseline_opponent(self.baseline_opponent_ratio, self.baseline_opponent_type)
                            print(f"[Live Config] Stratified League disabled. Reverted to static baseline ratio.")

                # Update action masking parameters
                if "use_action_masking" in live:
                    self.agent.use_action_masking = bool(live["use_action_masking"])
                if "handbrake_height_buffer" in live:
                    self.agent.handbrake_height_buffer = float(live["handbrake_height_buffer"])
                    self.agent.height_buffer_norm = float(live["handbrake_height_buffer"]) / 2044.0
                if "torch_num_threads" in live:
                    try:
                        torch.set_num_threads(int(live["torch_num_threads"]))
                    except Exception:
                        pass

                # Check manual save checkpoint trigger
                if live.get("save_checkpoint_requested", False):
                    ckpt_path = os.path.join(self.save_dir, f"manual_checkpoint_step_{self.global_step}.pt")
                    self.save_checkpoint(ckpt_path)
                    print(f"[Live Config] Manual checkpoint saved to {ckpt_path}")
                    # Clear trigger flag
                    live["save_checkpoint_requested"] = False
                    with open(self.live_config_path, "w") as fw:
                        json.dump(live, fw, indent=2)

                # Handle pause
                while live.get("paused", False):
                    print("[Live Config] Training is PAUSED. Waiting to resume...")
                    time.sleep(1.0)
                    with open(self.live_config_path, "r") as fr:
                        live = json.load(fr)
                    if not live.get("paused", False):
                        print("[Live Config] Resuming training!")
                        break
        except Exception as e:
            print(f"[Live Config Error] {e}")

    def save_checkpoint(self, path: str):
        data = {
            "iteration": self.iteration,
            "global_step": self.global_step,
            "model_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "continuous_actions": self.continuous_actions,
            "use_layer_norm": self.use_layer_norm,
            "activation": self.activation,
            "rot_anneal_start_iter": self._rot_anneal_start_iter,
            "rot_anneal_start_ceiling": self._rot_anneal_start_ceiling,
        }
        # Atomic save on Windows: write to .tmp file then replace with retry to avoid file lock conflict (Error 1224)
        tmp_path = path + f".tmp.{os.getpid()}"
        try:
            torch.save(data, tmp_path)
            for attempt in range(10):
                try:
                    if os.path.exists(path):
                        os.replace(tmp_path, path)
                    else:
                        os.rename(tmp_path, path)
                    break
                except (PermissionError, OSError):
                    time.sleep(0.05)
            else:
                try:
                    torch.save(data, path)
                except Exception as e:
                    print(f"[PPO Trainer] Warning: Checkpoint save fallback failed: {e}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _archive_checkpoint_paths(self, ckpts: List[str]) -> Set[str]:
        """
        The permanent historical spine: the earliest surviving checkpoint in each
        `archive_stride`-wide iteration bucket. Bucketing rather than exact-multiple
        matching means the spine survives even when the checkpoint interval does not
        divide the stride, or when the interval is changed mid-run.
        """
        if self.archive_stride <= 0:
            return set()
        best: Dict[int, tuple] = {}
        for path in ckpts:
            it = self._checkpoint_iteration(path)
            if it is None:
                continue
            bucket = int(it) // self.archive_stride
            if bucket not in best or it < best[bucket][0]:
                best[bucket] = (it, path)
        return {p for _, p in best.values()}

    @staticmethod
    def _checkpoint_iteration(file_path: str) -> Optional[int]:
        """Parses the iteration number out of a checkpoint_iter_*.pt filename."""
        try:
            fname = os.path.basename(file_path)
            return int(fname.replace("checkpoint_iter_", "").replace(".pt", ""))
        except Exception:
            return None

    def cleanup_old_checkpoints(self, max_to_keep: Optional[int] = None):
        """
        Prunes numbered checkpoints down to the union of three retention tiers:

        1. Provisional / ranked / contender  - supplied by the league manager, which knows
           which checkpoints are still being measured and which currently rank.
        2. Archive - one checkpoint per `archive_stride` iterations, kept forever.
        3. Recency - the newest `max_to_keep`, used only as a backstop when no league
           manager is attached (the league tier already covers un-graded arrivals).

        Never touches latest_model.pt or manually named saves.
        """
        import glob
        pattern = os.path.join(self.save_dir, "checkpoint_iter_*.pt")
        ckpts = glob.glob(pattern)
        if not ckpts:
            return

        keep: Set[str] = set()

        def mark(path: str):
            abs_p = os.path.abspath(path)
            keep.add(abs_p)
            keep.add(os.path.normcase(abs_p))

        league = getattr(self, "league_manager", None)
        if league:
            # Already canonicalized (abspath + normcase) by the league manager.
            keep |= league.get_protected_checkpoint_paths()
        else:
            limit = max_to_keep if max_to_keep is not None else self.max_checkpoints_to_keep
            if limit <= 0:
                return
            newest = sorted(
                ckpts,
                key=lambda p: (self._checkpoint_iteration(p) is None, self._checkpoint_iteration(p) or os.path.getmtime(p)),
                reverse=True
            )[:limit]
            for path in newest:
                mark(path)

        for path in self._archive_checkpoint_paths(ckpts):
            mark(path)

        for old_file in ckpts:
            if os.path.normcase(os.path.abspath(old_file)) in keep:
                continue
            try:
                os.remove(old_file)
                print(f"[PPO Trainer] Rolling Cleanup: Removed old checkpoint {os.path.basename(old_file)}")
            except Exception as e:
                print(f"[PPO Trainer] Warning: Could not delete old checkpoint {old_file}: {e}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            print(f"[PPO Trainer] Checkpoint not found at {path}")
            return
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            saved_state = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            saved_state = checkpoint
        else:
            saved_state = checkpoint
        model_state = self.agent.state_dict()

        # Seamless dimension expansion migration (e.g. 64 -> 70 obs_dim, 19 -> 24 act_dim)
        migrated = False
        for k in list(saved_state.keys()):
            if k in model_state:
                saved_param = saved_state[k]
                curr_param = model_state[k]
                if saved_param.shape != curr_param.shape:
                    migrated = True
                    curr_param = curr_param.clone()
                    curr_param.zero_()
                    slices = tuple(slice(0, min(s, c)) for s, c in zip(saved_param.shape, curr_param.shape))
                    curr_param[slices] = saved_param[slices]
                    model_state[k] = curr_param
                else:
                    model_state[k] = saved_param

        if migrated:
            self.agent.load_state_dict(model_state)
            self.iteration = checkpoint.get("iteration", 0)
            self.global_step = checkpoint.get("global_step", 0)
            self._rot_anneal_start_iter = checkpoint.get("rot_anneal_start_iter")
            self._rot_anneal_start_ceiling = checkpoint.get("rot_anneal_start_ceiling")
            self.agent.debias_symmetric_actions()
            print(f"[PPO Trainer] Successfully migrated weights to new dimensions (Obs: {self.obs_dim}, Act: {self.act_dim}) from {path} (Iter: {self.iteration})")
            return

        self.agent.load_state_dict(saved_state)
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass
        self.iteration = checkpoint.get("iteration", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self._rot_anneal_start_iter = checkpoint.get("rot_anneal_start_iter")
        self._rot_anneal_start_ceiling = checkpoint.get("rot_anneal_start_ceiling")
        self.agent.debias_symmetric_actions()

        # Sanitize any legacy subnormal floating-point numbers in weights and optimizer states
        # to prevent x86 microcode assist performance traps
        with torch.no_grad():
            cleaned_subnormals = 0
            for p in self.agent.parameters():
                sub = (p.abs() > 0) & (p.abs() < 1.17549435e-38)
                if sub.any():
                    cleaned_subnormals += int(sub.sum().item())
                    p[sub] = 0.0
            if hasattr(self, "optimizer") and self.optimizer is not None:
                for state in self.optimizer.state.values():
                    for sk, sv in state.items():
                        if torch.is_tensor(sv) and sv.is_floating_point():
                            sub = (sv.abs() > 0) & (sv.abs() < 1.17549435e-38)
                            if sub.any():
                                cleaned_subnormals += int(sub.sum().item())
                                sv[sub] = 0.0
            if cleaned_subnormals > 0:
                print(f"[PPO Trainer] Performance Sanitizer: Flushed {cleaned_subnormals:,} legacy subnormal numbers to true 0.0.")

        print(f"[PPO Trainer] Loaded checkpoint from {path} (Iteration: {self.iteration}, Step: {self.global_step})")

    def train(self, max_iterations: Optional[int] = None):
        """
        Main PPO optimization loop.
        """
        print("[PPO Trainer] Starting training loop...")
        obs_raw = self.env.reset()  # shape: (num_envs, num_players, obs_dim)
        obs_tensor = torch.tensor(obs_raw, dtype=torch.float32, device=self.device).reshape(-1, self.obs_dim)
        done_tensor = torch.zeros(self.total_actors, dtype=torch.float32, device=self.device)

        # Buffers for rollout storage
        obs_buf = torch.zeros((self.num_steps, self.total_actors, self.obs_dim), device=self.device)
        act_buf = torch.zeros((self.num_steps, self.total_actors, self.act_dim if self.continuous_actions else 1), device=self.device)
        logp_buf = torch.zeros((self.num_steps, self.total_actors), device=self.device)
        rew_buf = torch.zeros((self.num_steps, self.total_actors), device=self.device)
        done_buf = torch.zeros((self.num_steps, self.total_actors), device=self.device)
        val_buf = torch.zeros((self.num_steps, self.total_actors), device=self.device)

        start_time = time.time()

        while True:
            self.iteration += 1
            if max_iterations and self.iteration > max_iterations:
                print(f"[PPO Trainer] Reached maximum requested iterations: {max_iterations}")
                break

            # 1. Dynamic live parameter check
            self.check_live_config()
            self._step_rot_log_std_anneal()

            # Install any league update produced by a finished background grading job
            self._apply_pending_league_update()

            # Dynamic Stratified League Rotation (Resample Pool Opponents every 5 iterations).
            # Skipped while a grading child owns the league files: computing a distribution
            # refreshes the pool, which can record a coronation event and write league
            # state, racing the child's own write. The rotation is opportunistic, and the
            # results of grading bring a fresh distribution with them anyway.
            if (hasattr(self, "league_manager") and self.league_manager.enabled
                    and self.iteration % 5 == 0 and not self._league_job_running()):
                strat_assignments = self.league_manager.get_stratified_distribution(self.num_envs)
                self.env.set_stratified_opponents(strat_assignments)

            iter_start_time = time.time()
            episode_rewards_list = []
            episode_touches_list = []
            episode_goals_list = []
            rollout_touches_total = 0

            # 2. Collect Rollout
            for step in range(self.num_steps):
                self.global_step += self.total_actors
                obs_buf[step] = obs_tensor
                done_buf[step] = done_tensor

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(obs_tensor)
                    val_buf[step] = value.flatten()

                act_buf[step] = action if self.continuous_actions else action.unsqueeze(-1)
                logp_buf[step] = logprob

                # Environment step
                act_np = action.detach().cpu().numpy().reshape(self.num_envs, self.num_agents_per_env, -1) if self.continuous_actions else action.detach().cpu().numpy().reshape(self.num_envs, self.num_agents_per_env)

                next_obs, rews, dones, infos = self.env.step(act_np)

                rew_tensor = torch.from_numpy(rews).float().flatten().to(self.device)
                rew_buf[step] = rew_tensor

                for info in infos:
                    rollout_touches_total += int(info.get("step_touches", 0))
                    if info.get("done", False) or info.get("is_goal", False) or info.get("step", 0) >= self.max_episode_steps:
                        episode_rewards_list.extend(info.get("episode_rewards", []))
                        episode_touches_list.extend(info.get("episode_touches", []))
                        episode_goals_list.append(sum(info.get("episode_goals", [0, 0])))

                obs_tensor = torch.from_numpy(next_obs).float().reshape(-1, self.obs_dim).to(self.device)
                done_tensor = torch.from_numpy(dones).float().flatten().to(self.device)

            # 3. Generalized Advantage Estimation (GAE)
            with torch.no_grad():
                next_value = self.agent.get_value(obs_tensor).reshape(1, -1)
                advantages = torch.zeros_like(rew_buf)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - done_tensor
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - done_buf[t + 1]
                        nextvalues = val_buf[t + 1]
                    delta = rew_buf[t] + self.gamma * nextvalues * nextnonterminal - val_buf[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + val_buf

            # Flatten rollout tensors for mini-batch updates
            b_obs = obs_buf.reshape(-1, self.obs_dim)
            b_logprobs = logp_buf.reshape(-1)
            b_actions = act_buf.reshape(-1, self.act_dim if self.continuous_actions else 1)
            if not self.continuous_actions:
                b_actions = b_actions.squeeze(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = val_buf.reshape(-1)

            # Filter only policy learner actors (excludes heuristic baseline bot trajectories)
            learner_mask_1d = self.env.get_learner_mask()
            full_learner_mask = np.tile(learner_mask_1d, self.num_steps)
            if not full_learner_mask.all():
                learner_indices = torch.tensor(np.where(full_learner_mask)[0], device=self.device)
                b_obs = b_obs[learner_indices]
                b_logprobs = b_logprobs[learner_indices]
                b_actions = b_actions[learner_indices]
                b_advantages = b_advantages[learner_indices]
                b_returns = b_returns[learner_indices]
                b_values = b_values[learner_indices]

            # 4. PPO Mini-Batch Optimization
            total_samples = b_obs.shape[0]
            b_inds = np.arange(total_samples)
            clipfracs = []

            pg_losses = []
            v_losses = []
            entropy_losses = []
            bc_losses = []

            for epoch in range(self.n_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, total_samples, self.mini_batch_size):
                    end = start + self.mini_batch_size
                    mb_inds = b_inds[start:end]

                    mb_o = b_obs[mb_inds]
                    mb_a = b_actions[mb_inds]
                    mb_lp = b_logprobs[mb_inds]
                    mb_adv = b_advantages[mb_inds]
                    mb_ret = b_returns[mb_inds]
                    mb_val = b_values[mb_inds]

                    aug_o = mb_o
                    aug_a = mb_a
                    aug_lp = mb_lp
                    aug_adv = mb_adv
                    aug_ret = mb_ret
                    aug_val = mb_val

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        aug_o, aug_a
                    )
                    logratio = newlogprob - aug_lp
                    ratio = logratio.exp()

                    with torch.no_grad():
                        clipfracs += [((ratio - 1.0).abs() > self.clip_range).float().mean().item()]

                    # Advantage normalization
                    norm_adv = (aug_adv - aug_adv.mean()) / (aug_adv.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -norm_adv * ratio
                    pg_loss2 = -norm_adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - aug_ret) ** 2
                    v_clipped = aug_val + torch.clamp(
                        newvalue - aug_val,
                        -self.clip_range,
                        self.clip_range,
                    )
                    v_loss_clipped = (v_clipped - aug_ret) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # Entropy loss (normalized per-channel for continuous Gaussian actions)
                    dim_scale = float(self.act_dim) if self.continuous_actions else 1.0
                    entropy_loss = entropy.mean() / dim_scale

                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    # Behavioral Cloning (BC) Regularization with persistent floor anchor
                    # Prevents catastrophic forgetting of core mechanical dodges and kickoffs during extended RL self-play
                    min_bc_floor = 0.05 * self.bc_regularization_weight
                    decay_factor = max(0.0, 1.0 - (self.global_step / max(1, self.bc_decay_steps)))
                    current_bc_weight = float(max(min_bc_floor, self.bc_regularization_weight * decay_factor))
                    if current_bc_weight > 1e-4:
                        self._ensure_bc_dataset()
                        if self.bc_obs_tensor is not None and len(self.bc_obs_tensor) > 0:
                            n_bc = len(self.bc_obs_tensor)
                            bc_sample_size = min(len(mb_inds), n_bc)
                            bc_idx = torch.randint(0, n_bc, (bc_sample_size,), device=self.device)
                            sample_bc_o = self.bc_obs_tensor[bc_idx]
                            sample_bc_a = self.bc_act_tensor[bc_idx]

                            # Fast Actor-only forward pass (skips Critic network)
                            feat = self.agent.actor_backbone(sample_bc_o)
                            pred_bc_continuous = torch.tanh(self.agent.actor_mean(feat))
                            pred_bc_binary = self.agent.actor_binary(feat)

                            target_bc_continuous = sample_bc_a[:, :5]
                            target_bc_binary = (sample_bc_a[:, 5:] > 0.0).float()

                            bc_cont_loss = nn.functional.smooth_l1_loss(pred_bc_continuous, target_bc_continuous)
                            bc_bin_loss = nn.functional.binary_cross_entropy_with_logits(pred_bc_binary, target_bc_binary)
                            bc_loss = bc_cont_loss + 0.5 * bc_bin_loss

                            loss = loss + current_bc_weight * bc_loss
                            bc_losses.append(bc_loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    # In-place guard for exploration standard deviation parameter
                    if self.continuous_actions and hasattr(self.agent, "actor_log_std"):
                        self.agent.actor_log_std.data.clamp_(min=-2.5, max=-0.5)

                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    entropy_losses.append(entropy_loss.item())

            # 5. Metrics Compilation & Fast Vectorized Behavioral Telemetry
            mean_ep_rew = float(np.mean(episode_rewards_list)) if episode_rewards_list else float(rew_buf.mean().item() * self.num_steps)
            mean_touches = float(np.mean(episode_touches_list)) if episode_touches_list else float(rollout_touches_total)
            total_goals = int(sum(episode_goals_list)) if episode_goals_list else 0
            mean_pg_loss = float(np.mean(pg_losses))
            mean_v_loss = float(np.mean(v_losses))
            mean_entropy = float(np.mean(entropy_losses))
            sps = int(self.total_actors * self.num_steps / (time.time() - iter_start_time))

            # Fast zero-copy telemetry directly from rollout tensors
            act_np = b_actions.detach().cpu().numpy()
            obs_np = b_obs.detach().cpu().numpy()

            thr_col = act_np[:, 0]
            str_col = act_np[:, 1]
            jmp_col = act_np[:, 5]
            bst_col = act_np[:, 6]
            hnd_col = act_np[:, 7]

            pos_x = obs_np[:, 0]
            pos_y = obs_np[:, 1]
            boost_amt = np.clip(obs_np[:, 18], 0.0, 1.0) * 100.0
            on_ground_flag = obs_np[:, 19]

            telemetry = {
                "throttle_forward_pct": round(float(np.mean(thr_col > 0.2) * 100.0), 1),
                "throttle_reverse_pct": round(float(np.mean(thr_col < -0.2) * 100.0), 1),
                "throttle_coast_pct": round(float(np.mean((thr_col >= -0.2) & (thr_col <= 0.2)) * 100.0), 1),
                "steer_left_pct": round(float(np.mean(str_col < -0.2) * 100.0), 1),
                "steer_right_pct": round(float(np.mean(str_col > 0.2) * 100.0), 1),
                "steer_straight_pct": round(float(np.mean(np.abs(str_col) <= 0.2) * 100.0), 1),
                "jump_rate_pct": round(float(np.mean(jmp_col > 0.0) * 100.0), 1),
                "boost_rate_pct": round(float(np.mean(bst_col > 0.0) * 100.0), 1),
                "handbrake_rate_pct": round(float(np.mean(hnd_col > 0.0) * 100.0), 1),
                "ground_time_pct": round(float(np.mean(on_ground_flag > 0.5) * 100.0), 1),
                "air_time_pct": round(float(np.mean(on_ground_flag <= 0.5) * 100.0), 1),
                "corner_zone_pct": round(float(np.mean((np.abs(pos_x) > 0.65) & (np.abs(pos_y) > 0.70)) * 100.0), 1),
                "defensive_third_pct": round(float(np.mean(pos_y < -0.33) * 100.0), 1),
                "midfield_third_pct": round(float(np.mean(np.abs(pos_y) <= 0.33) * 100.0), 1),
                "offensive_third_pct": round(float(np.mean(pos_y > 0.33) * 100.0), 1),
                "mean_boost_tank": round(float(np.mean(boost_amt)), 1),
                "zero_boost_pct": round(float(np.mean(boost_amt < 1.0) * 100.0), 1),
            }

            # Stream metrics to JSON for Gradio UI to consume
            metrics_payload = {
                "iteration": self.iteration,
                "global_step": self.global_step,
                "mean_reward": round(mean_ep_rew, 3),
                "policy_loss": round(mean_pg_loss, 5),
                "value_loss": round(mean_v_loss, 5),
                "entropy": round(mean_entropy, 4),
                "ball_touches": round(mean_touches, 2),
                "total_touches": rollout_touches_total,
                "goals": total_goals,
                "sps": sps,
                "learning_rate": self.lr,
                "elapsed_time": round(time.time() - start_time, 1),
                "timestamp": time.time(),
                "telemetry": telemetry
            }

            # Add League Manager telemetry
            if hasattr(self, "league_manager") and self.league_manager:
                metrics_payload["league"] = self.league_manager.get_telemetry()

            metrics_file = os.path.join(self.log_dir, "metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics_payload, f, indent=2)

            # Append to metrics history
            history_file = os.path.join(self.log_dir, "history.jsonl")
            with open(history_file, "a") as f:
                f.write(json.dumps(metrics_payload) + "\n")

            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar("charts/mean_reward", mean_ep_rew, self.global_step)
                self.writer.add_scalar("losses/policy_loss", mean_pg_loss, self.global_step)
                self.writer.add_scalar("losses/value_loss", mean_v_loss, self.global_step)
                self.writer.add_scalar("losses/entropy", mean_entropy, self.global_step)
                self.writer.add_scalar("charts/sps", sps, self.global_step)

            # Console output
            print(
                f"[Iter {self.iteration:04d} | Step {self.global_step:07d}] "
                f"Reward: {mean_ep_rew:+.2f} | "
                f"Policy Loss: {mean_pg_loss:.4f} | "
                f"Value Loss: {mean_v_loss:.4f} | "
                f"Entropy: {mean_entropy:.3f} | "
                f"Touches: {rollout_touches_total} ({mean_touches:.1f}/ep) | "
                f"Goals: {total_goals} | "
                f"SPS: {sps}"
            )

            # Crash-recovery autosave. Overwrites one file, mints nothing, grades nothing.
            if self.iteration % self.autosave_interval == 0:
                self.save_checkpoint(os.path.join(self.save_dir, "latest_model.pt"))

            # League checkpoint: a new numbered file that enters the rating population.
            if self.iteration % self.checkpoint_interval == 0:
                ckpt_path = os.path.join(self.save_dir, f"checkpoint_iter_{self.iteration}.pt")
                self.save_checkpoint(ckpt_path)

                # Automated TrueSkill Bayesian Grading & League Promotion.
                # Grading plays several full evaluation matches and used to block the
                # training loop for seconds at a time; it now runs on a background thread
                # and its results are picked up at the top of a later iteration.
                if hasattr(self, "league_manager") and self.league_manager.enabled:
                    self._submit_league_grading(ckpt_path)
                else:
                    self.cleanup_old_checkpoints(max_to_keep=self.max_checkpoints_to_keep)
                print(f"[PPO Trainer] Saved checkpoint to {ckpt_path} (retaining provisional + ranked + archive tiers)")

        if self.writer:
            self.writer.close()
        print("[PPO Trainer] Training finished.")
