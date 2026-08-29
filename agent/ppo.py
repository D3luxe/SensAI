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
from typing import Dict, Any, Optional

from env.rocket_env import VectorizedRocketEnv
from env.observations import OBS_MIRROR_MASK_NP, ACT_MIRROR_MASK_NP
from agent.models import ActorCritic


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
        self.batch_size = int(hp.get("batch_size", 2048))
        self.mini_batch_size = int(hp.get("mini_batch_size", 256))
        self.n_epochs = int(hp.get("n_epochs", 4))
        self.total_timesteps = int(hp.get("total_timesteps", 1_000_000))
        self.checkpoint_interval = int(log_cfg.get("checkpoint_interval", hp.get("checkpoint_interval", 20)))
        self.max_checkpoints_to_keep = int(log_cfg.get("max_checkpoints_to_keep", 5))

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

        # Hardware Thread & CPU Core Optimization (8 P-Cores + 8 E-Cores for Arrow Lake / Core Ultra 9)
        num_threads = int(env_cfg.get("num_threads", 16))
        total_cores = os.cpu_count() or 24
        if total_cores >= 16:
            try:
                import ctypes
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                mask_16 = 0xFFFF  # Cores 0 through 15 (8 Lion Cove P-Cores + 8 Skymont E-Cores)
                ctypes.windll.kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask_16))
                torch.set_num_threads(num_threads)
                torch.set_num_interop_threads(min(4, num_threads))
                print(f"[Hardware Optimizer] Configured Process Affinity to Cores 0-15 (8 P-Cores + 8 E-Cores | {num_threads} PyTorch Threads) for peak throughput & sustained boost.")
            except Exception as e:
                torch.set_num_threads(num_threads)
                print(f"[Hardware Optimizer] PyTorch threads set to {num_threads} (Affinity note: {e})")
        else:
            torch.set_num_threads(min(total_cores, num_threads))

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
        self.env = VectorizedRocketEnv(
            num_envs=self.num_envs,
            game_mode=self.game_mode,
            tick_skip=self.tick_skip,
            max_episode_steps=self.max_episode_steps,
            reward_weights=rew_cfg,
            continuous_actions=self.continuous_actions,
            self_play=self.self_play,
            baseline_opponent_ratio=self.baseline_opponent_ratio
        )

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
            use_layer_norm=self.use_layer_norm
        ).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5, weight_decay=self.weight_decay)

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

                # Update rewards
                if "rewards" in live and isinstance(live["rewards"], dict):
                    self.env.update_reward_weights(live["rewards"])
                    print(f"[Live Config] Reward weights dynamically updated.")

                # Update scenario distributions
                if "scenarios" in live and isinstance(live["scenarios"], dict):
                    self.env.update_scenarios(live["scenarios"])
                    print(f"[Live Config] Scenario distributions dynamically updated.")

                # Update baseline opponent ratio
                if "baseline_opponent_ratio" in live:
                    new_ratio = float(live["baseline_opponent_ratio"])
                    if abs(new_ratio - getattr(self, "baseline_opponent_ratio", 0.25)) > 1e-4:
                        self.baseline_opponent_ratio = new_ratio
                        self.env.update_baseline_ratio(self.baseline_opponent_ratio)
                        print(f"[Live Config] Baseline opponent ratio dynamically updated to: {self.baseline_opponent_ratio:.2f}")

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

    def cleanup_old_checkpoints(self, max_to_keep: Optional[int] = None):
        """
        Keeps only the latest `max_to_keep` numbered checkpoints (checkpoint_iter_*.pt)
        and deletes older numbered checkpoints. Never touches latest_model.pt or manual saves.
        """
        limit = max_to_keep if max_to_keep is not None else self.max_checkpoints_to_keep
        if limit <= 0:
            return

        import glob
        pattern = os.path.join(self.save_dir, "checkpoint_iter_*.pt")
        ckpts = glob.glob(pattern)
        if len(ckpts) > limit:
            def get_ckpt_iteration(file_path: str):
                try:
                    fname = os.path.basename(file_path)
                    num_str = fname.replace("checkpoint_iter_", "").replace(".pt", "")
                    return int(num_str)
                except Exception:
                    return os.path.getmtime(file_path)

            sorted_ckpts = sorted(ckpts, key=get_ckpt_iteration, reverse=True)
            to_remove = sorted_ckpts[limit:]
            for old_file in to_remove:
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
                    slices = tuple(slice(0, min(s, c)) for s, c in zip(saved_param.shape, curr_param.shape))
                    curr_param[slices] = saved_param[slices]
                    model_state[k] = curr_param
                else:
                    model_state[k] = saved_param

        if migrated:
            self.agent.load_state_dict(model_state)
            self.iteration = checkpoint.get("iteration", 0)
            self.global_step = checkpoint.get("global_step", 0)
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
        self.agent.debias_symmetric_actions()
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
                act_np = action.cpu().numpy().reshape(self.num_envs, self.num_agents_per_env, -1)
                if not self.continuous_actions:
                    act_np = action.cpu().numpy().reshape(self.num_envs, self.num_agents_per_env)

                next_obs, rews, dones, infos = self.env.step(act_np)

                rew_tensor = torch.tensor(rews, dtype=torch.float32, device=self.device).flatten()
                rew_buf[step] = rew_tensor

                for info in infos:
                    rollout_touches_total += int(info.get("step_touches", 0))
                    if info.get("is_goal", False) or info.get("step", 0) >= self.max_episode_steps:
                        episode_rewards_list.extend(info.get("episode_rewards", []))
                        episode_touches_list.extend(info.get("episode_touches", []))
                        episode_goals_list.append(sum(info.get("episode_goals", [0, 0])))

                obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).reshape(-1, self.obs_dim)
                done_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).flatten()

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
                        # Calculate approx_kl for monitoring
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

                    # Decaying Behavioral Cloning (BC) Regularization
                    current_bc_weight = float(self.bc_regularization_weight * max(0.0, 1.0 - (self.global_step / max(1, self.bc_decay_steps))))
                    if current_bc_weight > 1e-4:
                        self._ensure_bc_dataset()
                        if self.bc_obs_tensor is not None and len(self.bc_obs_tensor) > 0:
                            n_bc = len(self.bc_obs_tensor)
                            bc_sample_size = min(len(mb_inds), n_bc)
                            bc_idx = torch.randint(0, n_bc, (bc_sample_size,), device=self.device)
                            sample_bc_o = self.bc_obs_tensor[bc_idx]
                            sample_bc_a = self.bc_act_tensor[bc_idx]

                            pred_bc_act, _, _, _ = self.agent.get_action_and_value(sample_bc_o)
                            bc_loss = nn.functional.smooth_l1_loss(pred_bc_act, sample_bc_a)
                            loss = loss + current_bc_weight * bc_loss
                            bc_losses.append(bc_loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    entropy_losses.append(entropy_loss.item())

            # 5. Metrics Compilation & Behavioral Telemetry
            mean_ep_rew = float(np.mean(episode_rewards_list)) if episode_rewards_list else float(rew_buf.mean().item() * self.num_steps)
            mean_touches = float(np.mean(episode_touches_list)) if episode_touches_list else float(rollout_touches_total)
            total_goals = int(sum(episode_goals_list)) if episode_goals_list else 0
            mean_pg_loss = float(np.mean(pg_losses))
            mean_v_loss = float(np.mean(v_losses))
            mean_entropy = float(np.mean(entropy_losses))
            sps = int(self.total_actors * self.num_steps / (time.time() - iter_start_time))

            # Compute rollout behavioral telemetry using actual parsed actions
            if self.continuous_actions:
                act_np = self.env.envs[0].action_parser.parse_actions(b_actions.cpu().numpy())
            else:
                act_indices = b_actions.cpu().numpy().astype(int)
                act_np = self.env.envs[0].action_parser.parse_actions(act_indices)
            obs_np = b_obs.cpu().numpy()

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
                "goals": total_goals,
                "sps": sps,
                "learning_rate": self.lr,
                "elapsed_time": round(time.time() - start_time, 1),
                "timestamp": time.time(),
                "telemetry": telemetry
            }

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
                f"Touches: {mean_touches:.1f} | "
                f"Goals: {total_goals} | "
                f"SPS: {sps}"
            )

            # Auto-save checkpoints with rolling retention
            if self.iteration % self.checkpoint_interval == 0:
                ckpt_path = os.path.join(self.save_dir, f"checkpoint_iter_{self.iteration}.pt")
                latest_path = os.path.join(self.save_dir, "latest_model.pt")
                self.save_checkpoint(ckpt_path)
                self.save_checkpoint(latest_path)
                self.cleanup_old_checkpoints(max_to_keep=self.max_checkpoints_to_keep)
                print(f"[PPO Trainer] Saved checkpoint to {ckpt_path} (Preserving latest {self.max_checkpoints_to_keep} checkpoints)")

        if self.writer:
            self.writer.close()
        print("[PPO Trainer] Training finished.")
