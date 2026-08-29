"""
Supervised Behavioral Cloning & Playstyle Replication Engine for SensAI.
Pretrains ActorCritic neural policy on human match replay states (kickoffs, speed-flips,
aerials, powerslides, and saves) before PPO reinforcement learning fine-tuning.
"""

from __future__ import annotations
import os
import time
import math
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, Callable

from agent.models import ActorCritic, OBS_MIRROR_MASK_NP, ACT_MIRROR_MASK_NP
from env.observations import DefaultObservationBuilder
from env.physics_engine import CarState, BallState, BoostPad, ARENA_EXTENT_X, ARENA_EXTENT_Y
from utils.replay_parser import ReplayParser, get_default_demo_dir
from utils.inverse_dynamics import InverseDynamicsSolver


class MockArenaForObs:
    def __init__(self, ball: BallState, cars: list[CarState]):
        self.ball = ball
        self.cars = cars
        self.boost_pads = BoostPad.create_standard_pads()

    def get_shot_threat(self, team: int) -> Tuple[bool, float, float]:
        return False, 0.0, 0.0

    def get_predicted_ball_pos(self, ticks_ahead: int) -> np.ndarray:
        dt = (ticks_ahead / 120.0)
        px = self.ball.pos[0] + self.ball.vel[0] * dt
        py = self.ball.pos[1] + self.ball.vel[1] * dt
        pz = max(93.0, self.ball.pos[2] + self.ball.vel[2] * dt + 0.5 * (-650.0) * (dt ** 2))
        return np.array([px, py, pz], dtype=np.float32)


class BehavioralCloningTrainer:
    """
    Supervised Imitation Learning Pretrainer.
    Converts raw replay frames into observation-action pairs and trains ActorCritic weights.
    """
    def __init__(
        self,
        pool_path: str = "data/replays/replays_pool.npz",
        checkpoint_path: str = "checkpoints/latest_model.pt",
        device: str = "cpu"
    ):
        self.pool_path = pool_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.obs_builder = DefaultObservationBuilder(symmetric=True)
        self._is_running = False
        self._stop_requested = False
        self._thread: Optional[threading.Thread] = None
        self.status = {
            "running": False,
            "epoch": 0,
            "total_epochs": 0,
            "loss": 0.0,
            "action_accuracy": 0.0,
            "progress_pct": 0.0,
            "message": "Ready to pretrain."
        }

    def is_running(self) -> bool:
        return self._is_running

    def request_stop(self):
        self._stop_requested = True
        self.status["message"] = "Stopping pretraining..."

    def generate_pretrain_dataset(self, parser: Optional[ReplayParser] = None, max_samples: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        if parser is None:
            parser = ReplayParser(pool_path=self.pool_path)
        return self.generate_expert_dataset(parser, max_samples)

    def generate_expert_dataset(self, parser: ReplayParser, max_samples: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds (N, 74) observation and (N, 8) expert action training pairs from replay pool.
        """
        if parser.states_buffer is None:
            parser.load_pool()
        if parser.states_buffer is None:
            return np.zeros((0, 74), dtype=np.float32), np.zeros((0, 8), dtype=np.float32)

        data = parser.states_buffer
        total_frames = len(data["ball_pos"])
        if total_frames == 0:
            return np.zeros((0, 74), dtype=np.float32), np.zeros((0, 8), dtype=np.float32)

        indices = np.arange(total_frames)
        if total_frames > max_samples:
            indices = np.random.choice(total_frames, size=max_samples, replace=False)

        obs_list = []
        act_list = []

        for idx in indices:
            b_pos = data["ball_pos"][idx]
            b_vel = data["ball_vel"][idx]
            c_pos = data["car_pos"][idx]
            c_vel = data["car_vel"][idx]
            c_rot = data["car_rot"][idx]
            c_bst = data["car_boost"][idx]

            ball = BallState(pos=b_pos, vel=b_vel)

            # Extract for each car in frame
            num_cars = c_pos.shape[0] if c_pos.ndim > 1 else 1
            for car_idx in range(min(2, num_cars)):
                car_p = c_pos[car_idx] if c_pos.ndim > 1 else c_pos
                car_v = c_vel[car_idx] if c_vel.ndim > 1 else c_vel
                car_r = c_rot[car_idx] if c_rot.ndim > 1 else c_rot
                car_b = c_bst[car_idx] if c_bst.ndim > 0 else c_bst

                car = CarState(
                    id=car_idx,
                    team=car_idx % 2,
                    pos=car_p,
                    vel=car_v,
                    rot=car_r,
                    boost=float(car_b),
                    on_ground=(car_p[2] < 25.0)
                )

                arena = MockArenaForObs(ball, [car])
                obs_vec = self.obs_builder.build_obs(car, arena)

                # ── True Human Action Extraction via Inverse Dynamics ───────────
                expert_act = None
                if idx < total_frames - 1:
                    # Check if next frame is part of the same continuous match sequence
                    c_pos_next = data["car_pos"][idx + 1]
                    c_vel_next = data["car_vel"][idx + 1]
                    c_rot_next = data["car_rot"][idx + 1]
                    c_bst_next = data["car_boost"][idx + 1]

                    c_p_next = c_pos_next[car_idx] if c_pos_next.ndim > 1 else c_pos_next
                    c_v_next = c_vel_next[car_idx] if c_vel_next.ndim > 1 else c_vel_next
                    c_r_next = c_rot_next[car_idx] if c_rot_next.ndim > 1 else c_rot_next
                    c_b_next = c_bst_next[car_idx] if c_bst_next.ndim > 0 else c_bst_next

                    # If delta distance < 200 uu, frames are continuous
                    if float(np.linalg.norm(c_p_next - car_p)) < 250.0:
                        expert_act = InverseDynamicsSolver.solve_car_action(
                            car_p, car_v, car_r, np.zeros(3, dtype=np.float32), float(car_b), bool(car_p[2] < 25.0),
                            c_p_next, c_v_next, c_r_next, np.zeros(3, dtype=np.float32), float(c_b_next), bool(c_p_next[2] < 25.0),
                            dt=1.0 / 30.0
                        )

                if expert_act is None:
                    # Analytical pursuit controller fallback for isolated boundary frames
                    local_ball_x = float(obs_vec[34])
                    local_ball_y = float(obs_vec[35])
                    local_ball_z = float(obs_vec[36])

                    if local_ball_x < -0.3:
                        steer = 1.0 if local_ball_y < 0 else -1.0
                        throttle = 1.0
                        handbrake = 1.0 if (abs(local_ball_y) > 0.6 and car.on_ground) else -1.0
                    else:
                        steer = float(np.clip(-local_ball_y * 6.0, -1.0, 1.0))
                        throttle = 1.0
                        handbrake = -1.0

                    pitch, yaw, roll, jump, boost = 0.0, 0.0, 0.0, -1.0, -1.0
                    if car.on_ground:
                        if abs(local_ball_y) < 0.2 and local_ball_x > 0.3 and car.boost > 5.0:
                            boost = 1.0
                        if (abs(ball.pos[0]) < 50.0 and abs(ball.pos[1]) < 50.0) and 0.4 < local_ball_x < 1.1:
                            jump = 1.0
                    else:
                        pitch = float(np.clip(local_ball_z * 3.0, -1.0, 1.0))
                        yaw = steer
                        if local_ball_z > 0.15 and car.boost > 10.0:
                            boost = 1.0

                    expert_act = np.array([throttle, steer, pitch, yaw, roll, jump, boost, handbrake], dtype=np.float32)

                # Add direct sample
                obs_list.append(obs_vec)
                act_list.append(expert_act)

                # Add exact bilateral mirrored sample to enforce 100% left-right symmetry
                obs_mirr = obs_vec * OBS_MIRROR_MASK_NP
                act_mirr = expert_act * ACT_MIRROR_MASK_NP
                obs_list.append(obs_mirr)
                act_list.append(act_mirr)

        return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 512,
        lr: float = 0.001,
        base_checkpoint: Optional[str] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes fast supervised behavioral cloning on replay dataset.
        """
        self._is_running = True
        self._stop_requested = False

        # Set max CPU threads for accelerated training
        torch.set_num_threads(os.cpu_count() or 16)

        parser = ReplayParser(pool_path=self.pool_path)
        self.status["message"] = "Extracting observation-action pairs from replay dataset..."
        if progress_cb:
            progress_cb(self.status)

        obs_data, act_data = self.generate_expert_dataset(parser, max_samples=5000)
        if len(obs_data) == 0:
            self._is_running = False
            self.status["message"] = "Error: Replay pool is empty. Ingest .replay files first."
            if progress_cb:
                progress_cb(self.status)
            return self.status

        # Initialize or load model
        model = ActorCritic(obs_dim=74, act_dim=8, continuous_actions=True, use_layer_norm=True).to(self.device)
        if base_checkpoint and os.path.exists(base_checkpoint):
            try:
                ckpt = torch.load(base_checkpoint, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                elif isinstance(ckpt, dict):
                    model.load_state_dict(ckpt, strict=False)
            except Exception as e:
                print(f"[Pretrainer] Warning: Could not load base checkpoint: {e}")

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        obs_tensor = torch.tensor(obs_data, dtype=torch.float32, device=self.device)
        act_tensor = torch.tensor(act_data, dtype=torch.float32, device=self.device)

        dataset_size = len(obs_tensor)
        num_batches = max(1, dataset_size // batch_size)

        self.status["total_epochs"] = epochs
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            if self._stop_requested:
                self.status["message"] = f"Pretraining stopped by user at epoch {epoch-1}."
                break

            model.train()
            perm = torch.randperm(dataset_size)
            epoch_loss = 0.0
            epoch_acc = 0.0

            for b in range(num_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                b_obs = obs_tensor[idx]
                b_act = act_tensor[idx]

                optimizer.zero_grad()
                pred_act, _, _, _ = model.get_action_and_value(b_obs, deterministic=True)
                loss = criterion(pred_act, b_act)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                # Sign accuracy for steer and throttle
                steer_acc = (torch.sign(pred_act[:, 1]) == torch.sign(b_act[:, 1])).float().mean().item()
                epoch_acc += steer_acc

            mean_loss = epoch_loss / num_batches
            mean_acc = (epoch_acc / num_batches) * 100.0

            self.status["epoch"] = epoch
            self.status["loss"] = round(mean_loss, 4)
            self.status["action_accuracy"] = round(mean_acc, 1)
            self.status["progress_pct"] = round((epoch / epochs) * 100.0, 1)
            self.status["message"] = f"Epoch {epoch}/{epochs} | Loss: {mean_loss:.4f} | Steer Accuracy: {mean_acc:.1f}%"

            if progress_cb and (epoch % 5 == 0 or epoch == epochs):
                progress_cb(self.status)

        # Save pretrained model & baseline
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        payload = {
            "model_state_dict": model.state_dict(),
            "obs_dim": 74,
            "act_dim": 8,
            "continuous_actions": True,
            "continuous": True,
            "use_layer_norm": True,
            "pretrained": True,
            "pretrain_samples": dataset_size,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(payload, self.checkpoint_path)
        baseline_path = os.path.join(os.path.dirname(self.checkpoint_path), "pretrained_baseline.pt")
        torch.save(payload, baseline_path)

        elapsed = round(time.time() - start_time, 1)
        self._is_running = False
        self.status["running"] = False
        self.status["message"] = f"✅ Pretraining finished in {elapsed}s ({dataset_size:,} frames). Baseline model saved to {self.checkpoint_path}!"

        if progress_cb:
            progress_cb(self.status)

        return self.status

    def start_async(
        self,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 0.001,
        base_checkpoint: Optional[str] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Runs pretraining in background thread."""
        if self._is_running:
            return
        self.status["running"] = True
        self._thread = threading.Thread(
            target=self.train,
            kwargs={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "base_checkpoint": base_checkpoint,
                "progress_cb": progress_cb
            },
            daemon=True
        )
        self._thread.start()
