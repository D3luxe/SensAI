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
from pathlib import Path

from agent.models import ActorCritic, OBS_MIRROR_MASK_NP, ACT_MIRROR_MASK_NP
from env.observations import DefaultObservationBuilder
from env.physics_engine import CarState, BallState, BoostPad, ARENA_EXTENT_X, ARENA_EXTENT_Y
from utils.replay_parser import ReplayParser, get_default_demo_dir, is_fresh_car_update, find_car_transition, REPLAY_ANG_VEL_SCALE
from utils.replay_state import ReplayStateReconstructor
from utils.inverse_dynamics import InverseDynamicsSolver
from utils.surface_contact import is_on_surface
from bot import rotation_to_rot_mat

try:
    import RocketSim as rsim
except ImportError:
    rsim = None


class MockArenaForObs:
    """
    Arena-shaped view of one replay frame (or hand-built state) for the observation builder.

    Ball prediction and shot threat are answered by RocketSim, the source the training arena uses;
    the straight-line extrapolation this used to do disagreed with the environment on every bounce
    and never reported a threat. Pad state is whatever the caller reconstructed, all active if none.
    """
    _SHARED_PADS = None
    _SM_PAD_INDICES = None
    _BG_PAD_INDICES = None
    _SMALL_PAD_POS_3D = None
    _BIG_PAD_POS_3D = None
    _SMALL_PAD_ACTIVE = None
    _BIG_PAD_ACTIVE = None
    _SIM = None

    def __init__(self, ball: BallState, cars: list[CarState], pad_active: Optional[np.ndarray] = None, pad_cooldown: Optional[np.ndarray] = None):
        if MockArenaForObs._SHARED_PADS is None:
            MockArenaForObs._SHARED_PADS = BoostPad.create_standard_pads()
            MockArenaForObs._SM_PAD_INDICES = np.array([i for i, p in enumerate(MockArenaForObs._SHARED_PADS) if not p.is_big], dtype=int)
            MockArenaForObs._BG_PAD_INDICES = np.array([i for i, p in enumerate(MockArenaForObs._SHARED_PADS) if p.is_big], dtype=int)
            MockArenaForObs._SMALL_PAD_POS_3D = np.array([MockArenaForObs._SHARED_PADS[i].pos for i in MockArenaForObs._SM_PAD_INDICES], dtype=np.float32)
            MockArenaForObs._BIG_PAD_POS_3D = np.array([MockArenaForObs._SHARED_PADS[i].pos for i in MockArenaForObs._BG_PAD_INDICES], dtype=np.float32)
            MockArenaForObs._SMALL_PAD_ACTIVE = np.array([MockArenaForObs._SHARED_PADS[i].is_active for i in MockArenaForObs._SM_PAD_INDICES], dtype=bool)
            MockArenaForObs._BIG_PAD_ACTIVE = np.array([MockArenaForObs._SHARED_PADS[i].is_active for i in MockArenaForObs._BG_PAD_INDICES], dtype=bool)

        self.ball = ball
        self.cars = cars
        self._sm_pad_indices = MockArenaForObs._SM_PAD_INDICES
        self._bg_pad_indices = MockArenaForObs._BG_PAD_INDICES
        self._small_pad_pos_3d = MockArenaForObs._SMALL_PAD_POS_3D
        self._big_pad_pos_3d = MockArenaForObs._BIG_PAD_POS_3D
        if pad_active is None:
            self.boost_pads = MockArenaForObs._SHARED_PADS
            self._small_pad_active = MockArenaForObs._SMALL_PAD_ACTIVE
            self._big_pad_active = MockArenaForObs._BIG_PAD_ACTIVE
        else:
            pad_active = np.asarray(pad_active, dtype=bool)
            self.boost_pads = [
                BoostPad(pos=p.pos, is_big=p.is_big, is_active=bool(a), cooldown_timer=float(cd))
                for p, a, cd in zip(MockArenaForObs._SHARED_PADS, pad_active, pad_cooldown)
            ]
            self._small_pad_active = pad_active[MockArenaForObs._SM_PAD_INDICES]
            self._big_pad_active = pad_active[MockArenaForObs._BG_PAD_INDICES]
        self._sim_synced = False

    @classmethod
    def _shared_sim(cls):
        if cls._SIM is None:
            from env.physics_engine import RocketSimArena
            cls._SIM = RocketSimArena(num_players=2)
        return cls._SIM

    def _sync_sim(self):
        """Loads this frame's ball into the shared RocketSim arena and drops its per-step caches."""
        sim = MockArenaForObs._shared_sim()
        if not self._sim_synced:
            pos = np.asarray(self.ball.pos, dtype=np.float32)
            vel = np.asarray(self.ball.vel, dtype=np.float32)
            sim.ball = BallState(pos=pos.copy(), vel=vel.copy())
            if sim._use_rsim and rsim is not None:
                bs = sim._rsim_arena.ball.get_state()
                bs.pos = rsim.Vec(float(pos[0]), float(pos[1]), float(pos[2]))
                bs.vel = rsim.Vec(float(vel[0]), float(vel[1]), float(vel[2]))
                bs.ang_vel = rsim.Vec(0.0, 0.0, 0.0)
                sim._rsim_arena.ball.set_state(bs)
            sim.step_count += 1
            sim._cached_rsim_preds = None
            sim._cached_threat = {}
            self._sim_synced = True
        return sim

    def get_shot_threat(self, team: int) -> Tuple[bool, float, float]:
        return self._sync_sim().get_shot_threat(team)

    def get_predicted_ball_pos(self, ticks_ahead: int) -> np.ndarray:
        return self._sync_sim().get_predicted_ball_pos(ticks_ahead)



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

    def _synthetic_obs(self, car: CarState, ball: BallState, phase: str, rng: np.random.Generator) -> np.ndarray:
        """
        Observation for a hand-built demonstration state, given the context every real state has:
        an opponent somewhere on the field, a ball already in play, and the jump and flip state of
        the demonstration phase ("ground", "jumped": airborne with the flip in hand, "dodging").
        """
        car.ball_touches = 1
        if phase == "ground":
            car.has_jump, car.has_flip = True, False
        elif phase == "jumped":
            car.has_jump, car.has_flip, car.air_timer = False, True, 0.15
        elif phase == "dodging":
            car.has_jump, car.has_flip, car.is_dodging = False, False, True
            car.flip_timer, car.air_timer = 0.25, 0.4
        car.is_supersonic = bool(np.linalg.norm(car.vel) >= 2200.0)

        yaw = float(rng.uniform(-math.pi, math.pi))
        speed = float(rng.uniform(0.0, 1800.0))
        opp = CarState(
            id=1, team=1,
            pos=np.array([rng.uniform(-3500.0, 3500.0), rng.uniform(-4500.0, 4500.0), 17.0], dtype=np.float32),
            vel=np.array([math.cos(yaw) * speed, math.sin(yaw) * speed, 0.0], dtype=np.float32),
            rot=np.array([0.0, yaw, 0.0], dtype=np.float32),
            rot_mat=rotation_to_rot_mat(0.0, yaw, 0.0),
            boost=float(rng.uniform(0.0, 100.0)),
            on_ground=True, has_jump=True, has_flip=False, ball_touches=1
        )
        return self.obs_builder.build_obs(car, MockArenaForObs(ball, [car, opp]))

    def generate_pretrain_dataset(
        self,
        parser: Optional[ReplayParser] = None,
        max_samples: int = 50000,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        pool_file = parser.pool_path if parser is not None else self.pool_path
        pool_stem = Path(pool_file).stem if pool_file else "replays_pool"
        cache_dir = Path("data/cache")
        cache_file = cache_dir / f"bc_dataset_{pool_stem}_{max_samples}_dim{self.obs_builder.obs_dim}.npz"

        if use_cache and cache_file.exists() and os.path.exists(pool_file):
            try:
                # Invalidate cache if the replay pool was modified after the cache was written
                if os.path.getmtime(pool_file) <= os.path.getmtime(cache_file):
                    with np.load(cache_file) as c:
                        obs = c["obs"]
                        act = c["act"]
                    if len(obs) > 0 and obs.shape[1] == self.obs_builder.obs_dim:
                        print(f"[BC Pretrainer] Fast-boot: Loaded {len(obs):,} cached frames from {cache_file}")
                        return obs, act
            except Exception as e:
                print(f"[BC Pretrainer] Cache read failed ({e}), regenerating dataset...")

        if parser is None:
            parser = ReplayParser(pool_path=self.pool_path)
        obs, act = self.generate_expert_dataset(parser, max_samples)

        if use_cache and len(obs) > 0:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_file, obs=obs, act=act)
                print(f"[BC Pretrainer] Saved {len(obs):,} BC dataset frames to cache: {cache_file}")
            except Exception as e:
                print(f"[BC Pretrainer] Warning: Failed to save BC cache: {e}")

        return obs, act

    def generate_expert_dataset(self, parser: ReplayParser, max_samples: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds (N, obs_dim) observation and (N, 8) expert action training pairs from replay pool.
        """
        if parser.states_buffer is None:
            parser.load_pool()
        if parser.states_buffer is None:
            return np.zeros((0, self.obs_builder.obs_dim), dtype=np.float32), np.zeros((0, 8), dtype=np.float32)

        data = parser.states_buffer
        total_frames = len(data["ball_pos"])
        if total_frames == 0:
            return np.zeros((0, self.obs_builder.obs_dim), dtype=np.float32), np.zeros((0, 8), dtype=np.float32)

        # Everything the replay does not record (opponent context aside): jump and flip state, timers,
        # touch-since-kickoff and pad state. Without it the cloned policy trains on states RocketSim never produces.
        recon = ReplayStateReconstructor(data)
        synth_rng = np.random.default_rng(0)

        # Frames whose second car is the parser's mirrored stand-in hold no real opponent
        usable = np.nonzero(~recon.synthetic_opponent)[0]
        indices = usable
        if len(usable) > max_samples:
            indices = np.random.choice(usable, size=max_samples, replace=False)

        obs_list = []
        act_list = []

        for idx in indices:
            b_pos = data["ball_pos"][idx]
            b_vel = data["ball_vel"][idx]
            c_pos = data["car_pos"][idx]
            c_vel = data["car_vel"][idx]
            c_rot = data["car_rot"][idx]
            c_bst = data["car_boost"][idx]
            c_angv = data["car_ang_vel"][idx] / REPLAY_ANG_VEL_SCALE if "car_ang_vel" in data else np.zeros_like(c_vel)

            ball = BallState(pos=b_pos, vel=b_vel)

            touches = 0 if recon.untouched[idx] else 1
            frame_cars = [
                CarState(
                    id=car_idx, team=car_idx, pos=c_pos[car_idx], vel=c_vel[car_idx], rot=c_rot[car_idx],
                    ang_vel=c_angv[car_idx], boost=float(c_bst[car_idx]), ball_touches=touches,
                    **recon.car_fields(idx, car_idx)
                )
                for car_idx in range(2)
            ]
            pad_active, pad_cooldown = recon.pads_at(idx)
            arena = MockArenaForObs(ball, frame_cars, pad_active, pad_cooldown)

            for car_idx in range(2):
                car_p = c_pos[car_idx]
                car_v = c_vel[car_idx]
                car_r = c_rot[car_idx]
                car_b = c_bst[car_idx]

                # Car state not replicated this frame: a carried-over duplicate carries no new label
                if not is_fresh_car_update(data, idx, car_idx):
                    continue

                car = frame_cars[car_idx]
                # Wheel contact from arena geometry; replays do not record it
                on_gnd_t = car.on_ground
                obs_vec = self.obs_builder.build_obs(car, arena)

                # ── True Human Action Extraction via Inverse Dynamics ───────────
                expert_act = None
                # Next genuine update of this car in the same recording, with its real time gap
                transition = find_car_transition(data, idx, car_idx)
                if transition is not None:
                    nxt, pair_dt = transition
                    c_pos_next = data["car_pos"][nxt]
                    c_vel_next = data["car_vel"][nxt]
                    c_rot_next = data["car_rot"][nxt]
                    c_bst_next = data["car_boost"][nxt]

                    c_p_next = c_pos_next[car_idx] if c_pos_next.ndim > 1 else c_pos_next
                    c_v_next = c_vel_next[car_idx] if c_vel_next.ndim > 1 else c_vel_next
                    c_r_next = c_rot_next[car_idx] if c_rot_next.ndim > 1 else c_rot_next
                    c_b_next = c_bst_next[car_idx] if c_bst_next.ndim > 0 else c_bst_next

                    expert_act = InverseDynamicsSolver.solve_car_action(
                        car_p, car_v, car_r, np.zeros(3, dtype=np.float32), float(car_b), on_gnd_t,
                        c_p_next, c_v_next, c_r_next, np.zeros(3, dtype=np.float32), float(c_b_next),
                        is_on_surface(c_p_next, InverseDynamicsSolver.basis(c_r_next)[:, 2]),
                        dt=pair_dt
                    )

                if expert_act is None:
                    # Analytical pursuit controller fallback for isolated boundary frames.
                    # Ball offset in the car's frame (obs 37..39, 1000 uu = 0.5); 34..36 are the
                    # 1.5 s predicted ball position in field coordinates, which this once read.
                    local_ball_x = float(obs_vec[37])
                    local_ball_y = float(obs_vec[38])
                    local_ball_z = float(obs_vec[39])

                    up_vec = car.get_up_vector()
                    is_on_wall = bool(car.pos[2] > 150.0 and abs(up_vec[2]) < 0.7)

                    if is_on_wall:
                        # On vertical wall: steer toward floor/ball (inverted lateral mapping on vertical surfaces)
                        steer = float(np.clip(local_ball_y * 3.0, -1.0, 1.0))
                        throttle = 1.0
                        handbrake = -1.0
                        roll = 0.0
                    elif local_ball_x < -0.3:
                        steer = 1.0 if local_ball_y > 0 else -1.0
                        throttle = 1.0
                        handbrake = 1.0 if (abs(local_ball_y) > 0.6 and car.on_ground) else -1.0
                        roll = 0.0
                    else:
                        steer = float(np.clip(local_ball_y * 3.0, -1.0, 1.0))
                        # Pacing throttle in close strike zone
                        throttle = min(1.0, max(0.3, local_ball_x * 2.0)) if (car.on_ground and abs(local_ball_y) < 0.3 and local_ball_x < 0.5) else 1.0
                        handbrake = -1.0
                        roll = 0.0

                    pitch, yaw, jump, boost = 0.0, 0.0, -1.0, -1.0
                    if is_on_wall and ball.pos[2] < 200.0:
                        jump = 1.0  # Wall-dodge jump recovery back down to pitch floor
                        roll = float(np.clip(-local_ball_y * 2.0, -1.0, 1.0))
                    elif car.on_ground:
                        if abs(local_ball_y) < 0.25 and local_ball_x > 0.3 and car.boost > 5.0:
                            boost = 1.0

                        fwd_vec = car.get_forward_vector()
                        fwd_speed = float(np.dot(car.vel, fwd_vec))
                        dist_to_ball = float(np.linalg.norm(ball.pos - car.pos))

                        # 1. Reverse Half-Flip: Facing away from ball while actively reversing backwards
                        if local_ball_x < -0.3 and fwd_speed < -150.0:
                            jump = 1.0
                            pitch = -1.0  # Nose-up backflip for half-flip turnaround

                        # 2. Kickoff Speed Rush Dodge:
                        elif (abs(ball.pos[0]) < 100.0 and abs(ball.pos[1]) < 100.0) and 0.3 < local_ball_x < 1.2:
                            jump = 1.0
                            if abs(local_ball_y) > 0.10:
                                flip_dir = float(np.sign(local_ball_y))
                                pitch = 0.85
                                yaw = flip_dir * 0.85
                                roll = flip_dir * 0.85
                            else:
                                pitch = 1.0  # Full front-flip speed dodge into kickoff ball

                        # 3. Open-Field Traversal Speed-Flip / Front-Flip:
                        elif local_ball_x > 0.3 and dist_to_ball > 500.0 and fwd_speed > 350.0:
                            jump = 1.0
                            if abs(local_ball_y) > 0.12:
                                flip_dir = float(np.sign(local_ball_y))
                                pitch = 0.85
                                yaw = flip_dir * 0.85
                                roll = flip_dir * 0.85
                            else:
                                pitch = 1.0  # Straight front-flip for downfield speed

                        # 4. Close-Quarters Strike Dodge (Challenging / 50-50 / Shot on Ball):
                        elif local_ball_x > 0.4 and dist_to_ball < 450.0 and ball.pos[2] < 180.0:
                            jump = 1.0
                            pitch = 0.80  # Power front-flip punch through the ball

                        # 5. Aerial Liftoff Jump (Elevated ball requiring aerial climb):
                        elif ball.pos[2] > 280.0 and dist_to_ball < 1500.0 and local_ball_x > 0.2:
                            jump = 1.0
                            pitch = -0.20  # Gentle nose-up pitch tilt for aerial climb (NEVER a -1.0 backflip dodge!)
                    else:
                        pitch = float(np.clip(-local_ball_z * 3.0, -1.0, 1.0))
                        yaw = steer
                        roll = float(np.clip(-local_ball_y * 1.5, -1.0, 1.0))
                        if local_ball_z > 0.15 and car.boost > 10.0:
                            boost = 1.0

                    expert_act = np.array([throttle, steer, pitch, yaw, roll, jump, boost, handbrake], dtype=np.float32)

                # Kickoff Sanitation: If frame is during active kickoff and player peeled away to corner boost,
                # sanitize expert action to enforce straight-ahead kickoff rush toward the ball.
                # obs 58 is the kickoff flag (ball resting at centre, untouched). This read obs 52, the
                # up component of the direction to the opponent goal, which is never above 0.5 on the ground.
                is_kickoff = bool(obs_vec[58] > 0.5)
                if is_kickoff:
                    expert_act[0] = 1.0  # Full forward throttle
                    # Ball to right (local lateral offset obs_vec[38] > 0) -> Steer right (expert_act[1] > 0)
                    expert_act[1] = float(np.clip(float(obs_vec[38]) * 2.5, -0.6, 0.6))
                    expert_act[6] = 1.0 if car.boost > 0 else -1.0  # Boost on kickoff

                # Add direct sample
                obs_list.append(obs_vec)
                act_list.append(expert_act)

        # ── Inject Synthetic Wall Recovery & Corner Exit Samples ──
        for side in [-1.0, 1.0]:  # Left (-1) and Right (+1) sidewalls
            wall_x = side * (ARENA_EXTENT_X - 96.0)
            for y_pos in np.linspace(-3000, 3000, 10):
                for z_pos in [300.0, 600.0, 1000.0]:
                    for heading_sign in [1.0, -1.0]:  # Driving North (+Y) or South (-Y)
                        yaw = math.pi / 2 if heading_sign > 0 else -math.pi / 2
                        roll = side * (math.pi / 2 if heading_sign > 0 else -math.pi / 2)
                        rot_mat = rotation_to_rot_mat(0.0, yaw, roll)
                        car_wall = CarState(
                            id=0, team=0,
                            pos=np.array([wall_x, y_pos, z_pos], dtype=np.float32),
                            vel=np.array([0.0, heading_sign * 800.0, 0.0], dtype=np.float32),
                            rot=np.array([0.0, yaw, roll], dtype=np.float32),
                            rot_mat=rot_mat,
                            boost=33.3,
                            on_ground=True
                        )
                        for bx in [-1000.0, 0.0, 1000.0]:
                            for by in [-1500.0, 0.0, 1500.0]:
                                ball_floor = BallState(pos=np.array([bx, by, 93.0], dtype=np.float32), vel=np.zeros(3, dtype=np.float32))
                                obs_w = self._synthetic_obs(car_wall, ball_floor, "ground", synth_rng)
                                steer_down = -float(side * heading_sign)
                                act_w = np.array([1.0, steer_down, 0.0, 0.0, float(side), 1.0, -1.0, -1.0], dtype=np.float32)

                                obs_list.append(obs_w)
                                act_list.append(act_w)
                                obs_list.append(obs_w * OBS_MIRROR_MASK_NP)
                                act_list.append(act_w * ACT_MIRROR_MASK_NP)

        # ── Inject Synthetic Front-Flip Demonstration Trajectories ──
        for heading_sign in [1.0, -1.0]:  # Facing North (+Y) or South (-Y)
            init_yaw = math.pi / 2 if heading_sign > 0 else -math.pi / 2
            for x_pos in [-1500.0, -500.0, 0.0, 500.0, 1500.0]:
                for y_start in [-2000.0, -500.0, 500.0, 2000.0]:
                    for ball_dist in [1200.0, 2400.0, 3500.0]:
                        ball_ff = BallState(
                            pos=np.array([x_pos, y_start + heading_sign * ball_dist, 93.15], dtype=np.float32),
                            vel=np.array([0.0, heading_sign * 500.0, 0.0], dtype=np.float32)
                        )

                        # Stage 1: Forward liftoff jump (car sprinting forward toward ball)
                        car_f1 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos, y_start, 22.0], dtype=np.float32),
                            vel=np.array([0.0, heading_sign * 1100.0, 190.0], dtype=np.float32),
                            rot=np.array([0.05, init_yaw, 0.0], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.05, init_yaw, 0.0),
                            boost=40.0,
                            on_ground=False
                        )
                        act_f1 = np.array([1.0, 0.0, 0.2, 0.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)

                        # Stage 2: Front-Flip Dodge Impulse (pitch = +1.0 forward dodge, jump = 1.0)
                        car_f2 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos, y_start + heading_sign * 80.0, 58.0], dtype=np.float32),
                            vel=np.array([0.0, heading_sign * 1600.0, 40.0], dtype=np.float32),
                            rot=np.array([0.35, init_yaw, 0.0], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.35, init_yaw, 0.0),
                            boost=35.0,
                            on_ground=False
                        )
                        act_f2 = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)

                        # Stage 3: Front-Flip In-Flight Rotation
                        car_f3 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos, y_start + heading_sign * 220.0, 55.0], dtype=np.float32),
                            vel=np.array([0.0, heading_sign * 1950.0, -30.0], dtype=np.float32),
                            rot=np.array([0.75, init_yaw, 0.0], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.75, init_yaw, 0.0),
                            boost=30.0,
                            on_ground=False
                        )
                        act_f3 = np.array([1.0, 0.0, 0.8, 0.0, 0.0, -1.0, 1.0, -1.0], dtype=np.float32)

                        # Stage 4: 4-Wheel Supersonic Touchdown
                        car_f4 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos, y_start + heading_sign * 420.0, 17.0], dtype=np.float32),
                            vel=np.array([0.0, heading_sign * 2200.0, 0.0], dtype=np.float32),
                            rot=np.array([0.0, init_yaw, 0.0], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.0, init_yaw, 0.0),
                            boost=25.0,
                            on_ground=True
                        )
                        act_f4 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)

                        for cs, act, phase in [(car_f1, act_f1, "jumped"), (car_f2, act_f2, "jumped"), (car_f3, act_f3, "dodging"), (car_f4, act_f4, "ground")]:
                            obs_val = self._synthetic_obs(cs, ball_ff, phase, synth_rng)
                            obs_list.append(obs_val)
                            act_list.append(act)
                            obs_list.append(obs_val * OBS_MIRROR_MASK_NP)
                            act_list.append(act * ACT_MIRROR_MASK_NP)

        # ── Inject Synthetic Diagonal Speed-Flip Demonstration Trajectories ──
        for heading_sign in [1.0, -1.0]:
            init_yaw = math.pi / 2 if heading_sign > 0 else -math.pi / 2
            for flip_side in [-1.0, 1.0]:  # Left vs Right diagonal speed-flip
                for x_pos in [-1200.0, 0.0, 1200.0]:
                    for y_start in [-1800.0, 0.0, 1800.0]:
                        ball_sf = BallState(
                            pos=np.array([x_pos + flip_side * 200.0, y_start + heading_sign * 2500.0, 93.15], dtype=np.float32),
                            vel=np.array([0.0, heading_sign * 600.0, 0.0], dtype=np.float32)
                        )

                        # Stage 1: Angled takeoff jump
                        car_sf1 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos, y_start, 22.0], dtype=np.float32),
                            vel=np.array([flip_side * 150.0, heading_sign * 1200.0, 190.0], dtype=np.float32),
                            rot=np.array([0.05, init_yaw + flip_side * 0.1, 0.0], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.05, init_yaw + flip_side * 0.1, 0.0),
                            boost=50.0,
                            on_ground=False
                        )
                        act_sf1 = np.array([1.0, float(flip_side * 0.2), 0.2, float(flip_side * 0.2), 0.0, 1.0, 1.0, -1.0], dtype=np.float32)

                        # Stage 2: Diagonal Speed-Flip Dodge (pitch = +0.85, yaw = +/-0.85, roll = +/-0.85)
                        car_sf2 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos + flip_side * 40.0, y_start + heading_sign * 90.0, 58.0], dtype=np.float32),
                            vel=np.array([flip_side * 300.0, heading_sign * 1700.0, 45.0], dtype=np.float32),
                            rot=np.array([0.3, init_yaw, float(flip_side * 0.4)], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.3, init_yaw, float(flip_side * 0.4)),
                            boost=45.0,
                            on_ground=False
                        )
                        act_sf2 = np.array([1.0, 0.0, 0.85, float(flip_side * 0.85), float(flip_side * 0.85), 1.0, 1.0, -1.0], dtype=np.float32)

                        # Stage 3: Flip Cancel + Counter Air-Roll
                        car_sf3 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos + flip_side * 80.0, y_start + heading_sign * 240.0, 50.0], dtype=np.float32),
                            vel=np.array([flip_side * 200.0, heading_sign * 2100.0, -25.0], dtype=np.float32),
                            rot=np.array([0.1, init_yaw, float(-flip_side * 0.3)], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.1, init_yaw, float(-flip_side * 0.3)),
                            boost=35.0,
                            on_ground=False
                        )
                        act_sf3 = np.array([1.0, 0.0, -0.6, float(-flip_side * 0.6), float(-flip_side * 0.6), -1.0, 1.0, -1.0], dtype=np.float32)

                        for cs, act, phase in [(car_sf1, act_sf1, "jumped"), (car_sf2, act_sf2, "jumped"), (car_sf3, act_sf3, "dodging")]:
                            obs_val = self._synthetic_obs(cs, ball_sf, phase, synth_rng)
                            obs_list.append(obs_val)
                            act_list.append(act)
                            obs_list.append(obs_val * OBS_MIRROR_MASK_NP)
                            act_list.append(act * ACT_MIRROR_MASK_NP)

        # ── Inject Synthetic Close-Quarters Strike Dodges (50/50 Hits into Ball) ──
        for heading_sign in [1.0, -1.0]:
            init_yaw = math.pi / 2 if heading_sign > 0 else -math.pi / 2
            for x_pos in [-1000.0, 0.0, 1000.0]:
                for y_start in [-2200.0, 0.0, 2200.0]:
                    for strike_dist in [150.0, 250.0, 380.0]:
                        ball_st = BallState(
                            pos=np.array([x_pos, y_start + heading_sign * strike_dist, 93.15], dtype=np.float32),
                            vel=np.array([0.0, heading_sign * 200.0, 0.0], dtype=np.float32)
                        )

                        # Car approaching ball in strike zone: MUST JUMP & FRONT-FLIP INTO BALL (NEVER BRAKE / BACKFLIP!)
                        car_st1 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos, y_start, 22.0], dtype=np.float32),
                            vel=np.array([0.0, heading_sign * 1400.0, 180.0], dtype=np.float32),
                            rot=np.array([0.1, init_yaw, 0.0], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.1, init_yaw, 0.0),
                            boost=30.0,
                            on_ground=False
                        )
                        act_st1 = np.array([1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)

                        car_st2 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos, y_start + heading_sign * 50.0, 50.0], dtype=np.float32),
                            vel=np.array([0.0, heading_sign * 1850.0, 40.0], dtype=np.float32),
                            rot=np.array([0.4, init_yaw, 0.0], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.4, init_yaw, 0.0),
                            boost=25.0,
                            on_ground=False
                        )
                        act_st2 = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)

                        for cs, act, phase in [(car_st1, act_st1, "jumped"), (car_st2, act_st2, "jumped")]:
                            obs_val = self._synthetic_obs(cs, ball_st, phase, synth_rng)
                            obs_list.append(obs_val)
                            act_list.append(act)
                            obs_list.append(obs_val * OBS_MIRROR_MASK_NP)
                            act_list.append(act * ACT_MIRROR_MASK_NP)

        # ── Inject Synthetic Half-Flip Demonstration Trajectories (From Genuine Reverse Only) ──
        for heading_sign in [1.0, -1.0]:  # Facing North (+Y) or South (-Y)
            init_yaw = math.pi / 2 if heading_sign > 0 else -math.pi / 2
            target_sign = -heading_sign    # Half-flip target is behind the car
            target_yaw = -init_yaw

            for roll_dir in [-1.0, 1.0]:   # Air-roll left vs right
                for x_pos in [-1500.0, 0.0, 1500.0]:
                    for y_start in [-1000.0, 1000.0]:
                        # Ball located downfield behind car
                        ball_hf = BallState(
                            pos=np.array([x_pos, y_start + target_sign * 3000.0, 93.15], dtype=np.float32),
                            vel=np.array([0.0, target_sign * 800.0, 0.0], dtype=np.float32)
                        )

                        # Stage 1: Reverse liftoff jump (car moving backward facing initial yaw)
                        car_s1 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos, y_start, 25.0], dtype=np.float32),
                            vel=np.array([0.0, target_sign * 350.0, 180.0], dtype=np.float32),
                            rot=np.array([0.0, init_yaw, 0.0], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.0, init_yaw, 0.0),
                            boost=50.0,
                            on_ground=False
                        )
                        act_s1 = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0], dtype=np.float32)

                        # Stage 2: Backflip Dodge Impulse (pitch = -1.0 nose-up backflip, jump = 1.0)
                        car_s2 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos, y_start + target_sign * 60.0, 75.0], dtype=np.float32),
                            vel=np.array([0.0, target_sign * 850.0, 120.0], dtype=np.float32),
                            rot=np.array([-0.75, init_yaw, 0.0], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(-0.75, init_yaw, 0.0),
                            boost=50.0,
                            on_ground=False
                        )
                        act_s2 = np.array([-1.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0, -1.0], dtype=np.float32)

                        # Stage 3: Flip Cancel + Air Roll (car inverted, pitch = +1.0 forward cancel, roll = +/-1.0, boost = 1.0)
                        car_s3 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos, y_start + target_sign * 180.0, 125.0], dtype=np.float32),
                            vel=np.array([0.0, target_sign * 1250.0, 60.0], dtype=np.float32),
                            rot=np.array([0.0, target_yaw, math.pi * 0.75 * roll_dir], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.0, target_yaw, math.pi * 0.75 * roll_dir),
                            boost=45.0,
                            on_ground=False
                        )
                        act_s3 = np.array([1.0, 0.0, 1.0, 0.0, float(roll_dir), -1.0, 1.0, -1.0], dtype=np.float32)

                        # Stage 4: 4-Wheel Landing Recovery + Supersonic Sprint
                        car_s4 = CarState(
                            id=0, team=0,
                            pos=np.array([x_pos, y_start + target_sign * 350.0, 17.0], dtype=np.float32),
                            vel=np.array([0.0, target_sign * 1650.0, 0.0], dtype=np.float32),
                            rot=np.array([0.0, target_yaw, 0.0], dtype=np.float32),
                            rot_mat=rotation_to_rot_mat(0.0, target_yaw, 0.0),
                            boost=35.0,
                            on_ground=True
                        )
                        act_s4 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0], dtype=np.float32)

                        for cs, act, phase in [(car_s1, act_s1, "jumped"), (car_s2, act_s2, "jumped"), (car_s3, act_s3, "dodging"), (car_s4, act_s4, "ground")]:
                            obs_val = self._synthetic_obs(cs, ball_hf, phase, synth_rng)
                            obs_list.append(obs_val)
                            act_list.append(act)
                            obs_list.append(obs_val * OBS_MIRROR_MASK_NP)
                            act_list.append(act * ACT_MIRROR_MASK_NP)

        return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 512,
        lr: float = 0.001,
        max_samples: int = 50000,
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

        obs_data, act_data = self.generate_expert_dataset(parser, max_samples=max_samples)
        if len(obs_data) == 0:
            self._is_running = False
            self.status["message"] = "Error: Replay pool is empty. Ingest .replay files first."
            if progress_cb:
                progress_cb(self.status)
            return self.status

        obs_tensor = torch.tensor(obs_data, dtype=torch.float32, device=self.device)
        act_tensor = torch.tensor(act_data, dtype=torch.float32, device=self.device)
        in_dim = obs_tensor.shape[1] if len(obs_tensor) > 0 else self.obs_builder.obs_dim

        # Initialize or load model
        model = ActorCritic(obs_dim=in_dim, act_dim=8, continuous_actions=True, use_layer_norm=True).to(self.device)
        orig_iteration = 0
        orig_global_step = 0
        if base_checkpoint and os.path.exists(base_checkpoint):
            try:
                ckpt = torch.load(base_checkpoint, map_location=self.device)
                saved_state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else (ckpt if isinstance(ckpt, dict) else ckpt)
                model_state = model.state_dict()
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
                model.load_state_dict(model_state, strict=False)
                if isinstance(ckpt, dict):
                    orig_iteration = ckpt.get("iteration", 0)
                    orig_global_step = ckpt.get("global_step", 0)
            except Exception as e:
                print(f"[Pretrainer] Warning: Could not load base checkpoint: {e}")

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

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
            epoch_steer_acc = 0.0
            epoch_jump_acc = 0.0
            epoch_boost_acc = 0.0

            for b in range(num_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                b_obs = obs_tensor[idx]
                b_act = act_tensor[idx]

                optimizer.zero_grad()
                
                # Dual Differentiable Loss (Continuous SmoothL1 + Binary BCEWithLogits)
                feat = model.actor_backbone(b_obs)
                pred_cont = torch.tanh(model.actor_mean(feat))
                pred_bin_logits = model.actor_binary(feat)

                target_cont = b_act[:, :5]
                target_bin = (b_act[:, 5:] > 0.0).float()

                loss_cont = nn.functional.smooth_l1_loss(pred_cont, target_cont)
                loss_bin = nn.functional.binary_cross_entropy_with_logits(pred_bin_logits, target_bin)
                loss = loss_cont + 0.5 * loss_bin

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                # Telemetry accuracy metrics
                steer_acc = (torch.sign(pred_cont[:, 1]) == torch.sign(target_cont[:, 1])).float().mean().item()
                jump_acc = ((pred_bin_logits[:, 0] > 0.0) == (target_bin[:, 0] > 0.5)).float().mean().item()
                boost_acc = ((pred_bin_logits[:, 1] > 0.0) == (target_bin[:, 1] > 0.5)).float().mean().item()
                
                epoch_steer_acc += steer_acc
                epoch_jump_acc += jump_acc
                epoch_boost_acc += boost_acc

            mean_loss = epoch_loss / num_batches
            mean_steer = (epoch_steer_acc / num_batches) * 100.0
            mean_jump = (epoch_jump_acc / num_batches) * 100.0
            mean_boost = (epoch_boost_acc / num_batches) * 100.0

            self.status["epoch"] = epoch
            self.status["loss"] = round(mean_loss, 4)
            self.status["action_accuracy"] = round(mean_steer, 1)
            self.status["jump_accuracy"] = round(mean_jump, 1)
            self.status["boost_accuracy"] = round(mean_boost, 1)
            self.status["progress_pct"] = round((epoch / epochs) * 100.0, 1)
            self.status["message"] = (
                f"Epoch {epoch}/{epochs} | Loss: {mean_loss:.4f} | "
                f"Steer: {mean_steer:.1f}% | Jump: {mean_jump:.1f}% | Boost: {mean_boost:.1f}%"
            )

            if progress_cb and (epoch % 5 == 0 or epoch == epochs):
                progress_cb(self.status)

        # Save pretrained model & baseline
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        payload = {
            "model_state_dict": model.state_dict(),
            "obs_dim": in_dim,
            "act_dim": 8,
            "continuous_actions": True,
            "continuous": True,
            "use_layer_norm": True,
            "activation": "leaky_relu",
            "pretrained": True,
            "pretrain_samples": dataset_size,
            "iteration": orig_iteration,
            "global_step": orig_global_step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(payload, self.checkpoint_path)
        baseline_path = os.path.join(os.path.dirname(self.checkpoint_path), "pretrained_baseline.pt")
        torch.save(payload, baseline_path)

        elapsed = round(time.time() - start_time, 1)
        self._is_running = False
        self.status["running"] = False
        self.status["message"] = f"[Pretrainer] Pretraining finished in {elapsed}s ({dataset_size:,} frames). Baseline model saved to {self.checkpoint_path}!"

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
