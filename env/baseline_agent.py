"""
Rule-Based & Neural Network Opponent Agents for Rocket League Mixup Training.
Provides support for Heuristic Chaser, SenseiBot Checkpoints (.pt), and TorchScript Models (Necto / Nexto).
"""

from __future__ import annotations
import os
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List, Dict, Any

from env.physics_engine import CarState, BallState, RocketSimArena, ARENA_EXTENT_Y
from env.observations import DefaultObservationBuilder
from env.actions import ContinuousActionParser, DiscreteActionParser


class BaseOpponent:
    """Abstract base class for all opponent bot types."""
    def get_action(self, car: CarState, arena_or_ball: Union[RocketSimArena, BallState]) -> np.ndarray:
        raise NotImplementedError


class BaselineChaser(BaseOpponent):
    """
    High-tempo heuristic opponent that challenges kickoffs and chases the ball directly.
    Incentivizes learning policies to execute disciplined kickoffs and 50/50 challenges.
    """
    def __init__(self, continuous_actions: bool = True):
        self.continuous_actions = continuous_actions

    def get_action(self, car: CarState, arena_or_ball: Union[RocketSimArena, BallState]) -> np.ndarray:
        ball = arena_or_ball.ball if isinstance(arena_or_ball, RocketSimArena) else arena_or_ball

        # Vector from car to ball
        diff = ball.pos - car.pos
        dist_2d = float(np.linalg.norm(diff[:2]))
        dist_3d = float(np.linalg.norm(diff))

        fwd = car.get_forward_vector()
        right = car.get_right_vector()

        fwd_dot = float(np.dot(diff[:2], fwd[:2]))
        right_dot = float(np.dot(diff[:2], right[:2]))

        # Proportional steering to face ball
        norm_diff = diff[:2] / max(1e-4, dist_2d)
        steer_target = float(norm_diff[0] * fwd[1] - norm_diff[1] * fwd[0])
        steer = float(np.clip(steer_target * 2.5, -1.0, 1.0))

        # Check kickoff state (stationary ball in center)
        is_kickoff = bool(abs(ball.pos[0]) < 50.0 and abs(ball.pos[1]) < 50.0 and float(np.linalg.norm(ball.vel)) < 100.0)

        throttle = 1.0
        pitch = 0.0
        yaw = 0.0
        roll = 0.0
        jump = 0.0
        boost = 0.0
        handbrake = 0.0

        if is_kickoff:
            # Kickoff Rusher Mode: Full throttle + boost straight at the ball
            throttle = 1.0
            boost = 1.0 if car.boost > 0 else 0.0
            # Flip / Dodge into the ball when close
            if dist_2d < 350.0 and car.on_ground:
                jump = 1.0
                pitch = -1.0  # Front flip
        else:
            # General Open-Field Pursuit
            if car.on_ground:
                # Accelerate forward when mostly aligned, or handbrake turn if facing away
                if fwd_dot > 0.0:
                    throttle = 1.0
                    # Boost on straightaways when well-aligned
                    if abs(steer) < 0.25 and fwd_dot > 300.0 and car.boost > 0:
                        boost = 1.0
                else:
                    throttle = 1.0
                    handbrake = 1.0 if abs(steer) > 0.5 else 0.0

                # Hop / Jump into aerial or bouncing balls
                if 120.0 < ball.pos[2] < 500.0 and dist_2d < 300.0:
                    jump = 1.0
                    pitch = -0.5 if fwd_dot > 0 else 0.0
            else:
                # Airborne orientation: simple pitch down / roll recovery
                if car.pos[2] > 200.0:
                    pitch = float(np.clip(-fwd[2] * 2.0, -1.0, 1.0))
                    roll = float(np.clip(-right[2] * 2.0, -1.0, 1.0))

        return np.array([throttle, steer, pitch, yaw, roll, jump, boost, handbrake], dtype=np.float32)


class CheckpointOpponentBot(BaseOpponent):
    """
    Opponent Bot powered by a trained SenseiBot Actor-Critic checkpoint (.pt).
    """
    def __init__(self, model_path: str, continuous_actions: bool = True, device: str = "cpu"):
        self.model_path = model_path
        self.device = torch.device(device)
        self.obs_builder = DefaultObservationBuilder(symmetric=True)
        self.discrete_parser = DiscreteActionParser()
        self.continuous_parser = ContinuousActionParser()
        self.model: Optional[nn.Module] = None
        self.continuous_actions = continuous_actions
        self._load_checkpoint()

    def _load_checkpoint(self):
        from agent.models import ActorCritic
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
                raise ValueError(f"Invalid ActorCritic checkpoint format in {self.model_path}")

            self.continuous_actions = ckpt.get("continuous_actions", True)
            obs_dim = self.obs_builder.obs_dim
            act_dim = 8 if self.continuous_actions else self.discrete_parser.action_dim

            self.model = ActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                continuous_actions=self.continuous_actions,
                use_layer_norm=ckpt.get("use_layer_norm", True)
            ).to(self.device)

            saved_state = ckpt["model_state_dict"]
            model_state = self.model.state_dict()

            # Flexible parameter migration
            migrated = False
            for k in list(saved_state.keys()):
                if k in model_state:
                    saved_p = saved_state[k]
                    curr_p = model_state[k]
                    if saved_p.shape != curr_p.shape:
                        migrated = True
                        slices = tuple(slice(0, min(s, c)) for s, c in zip(saved_p.shape, curr_p.shape))
                        curr_p[slices] = saved_p[slices]
                        model_state[k] = curr_p
                    else:
                        model_state[k] = saved_p

            if migrated:
                self.model.load_state_dict(model_state)
            else:
                self.model.load_state_dict(saved_state)

            self.model.eval()
            print(f"[Opponent Bot] Successfully loaded Sensei Checkpoint opponent: {os.path.basename(self.model_path)}")
        except Exception as e:
            print(f"[Opponent Bot] Error loading checkpoint {self.model_path}: {e}. Fallback to BaselineChaser.")
            self.model = None

    def get_action(self, car: CarState, arena_or_ball: Union[RocketSimArena, BallState]) -> np.ndarray:
        if self.model is None or not isinstance(arena_or_ball, RocketSimArena):
            return BaselineChaser().get_action(car, arena_or_ball)

        obs = self.obs_builder.build_obs(car, arena_or_ball)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            act_t, _, _, _ = self.model.get_action_and_value(obs_t, deterministic=True)
            if self.continuous_actions:
                return self.continuous_parser.parse_actions(act_t.squeeze(0).cpu().numpy())
            else:
                act_idx = int(act_t.squeeze().cpu().item())
                return self.discrete_parser.parse_actions(act_idx)


BOOST_LOCATIONS = np.array([
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (- 940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0),
    (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (- 940.0, 3310.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
], dtype=np.float32)

EARL_NORM = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4, dtype=np.float32)
EARL_INVERT = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4, dtype=np.float32)


class NectoNextoOpponentBot(BaseOpponent):
    """
    Opponent Bot powered by a TorchScript model (Necto, Nexto, or standard EARL Perceiver models).
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model: Optional[torch.jit.ScriptModule] = None
        self.is_nexto = False
        self.nexto_action_table: Optional[torch.Tensor] = None
        self.prev_action = np.zeros(8, dtype=np.float32)
        self._load_torchscript()

    def _load_torchscript(self):
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()

            # Check if Nexto (has embedded 90x8 lookup table in net.output constants)
            if hasattr(self.model, "net") and hasattr(self.model.net, "output"):
                out_mod = self.model.net.output
                if hasattr(out_mod, "code_with_constants"):
                    try:
                        _, consts = out_mod.code_with_constants
                        c0 = getattr(consts, "c0", None)
                        if c0 is not None and isinstance(c0, torch.Tensor) and c0.shape[-1] == 8:
                            self.is_nexto = True
                            self.nexto_action_table = c0.cpu().float()
                    except Exception:
                        self.is_nexto = False

            bot_name = "Nexto" if self.is_nexto else "Necto / TorchScript"
            print(f"[Opponent Bot] Successfully loaded {bot_name} opponent: {os.path.basename(self.model_path)}")
        except Exception as e:
            print(f"[Opponent Bot] Error loading TorchScript model {self.model_path}: {e}")
            self.model = None

    def _build_nexto_inputs(self, car: CarState, arena: RocketSimArena) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_players = len(arena.cars)
        n_boosts = len(BOOST_LOCATIONS)
        n_entities = n_players + 1 + n_boosts

        sel_ball = n_players
        sel_boosts = slice(sel_ball + 1, None)

        q = np.zeros((1, 1, 32), dtype=np.float32)
        kv = np.zeros((1, n_entities, 24), dtype=np.float32)
        m = np.zeros((1, n_entities), dtype=bool)

        # Ball
        kv[0, sel_ball, 3] = 1.0
        kv[0, sel_ball, 5:8] = arena.ball.pos
        kv[0, sel_ball, 8:11] = arena.ball.vel
        kv[0, sel_ball, 17:20] = arena.ball.ang_vel

        # Boosts
        kv[0, sel_boosts, 4] = 1.0
        kv[0, sel_boosts, 5:8] = BOOST_LOCATIONS
        kv[0, sel_boosts, 20] = 0.12 + 0.88 * (BOOST_LOCATIONS[:, 2] > 72)

        # Players
        main_idx = 0
        for idx, p in enumerate(arena.cars):
            if p is car:
                main_idx = idx
                kv[0, idx, 0] = 1.0  # is_self
            elif p.team == car.team:
                kv[0, idx, 1] = 1.0  # is_mate
            else:
                kv[0, idx, 2] = 1.0  # is_opp

            kv[0, idx, 5:8] = p.pos
            kv[0, idx, 8:11] = p.vel
            kv[0, idx, 11:14] = p.get_forward_vector()
            kv[0, idx, 14:17] = p.get_up_vector()
            kv[0, idx, 17:20] = p.ang_vel
            kv[0, idx, 20] = p.boost / 100.0
            kv[0, idx, 22] = 1.0 if p.on_ground else 0.0
            kv[0, idx, 23] = 1.0 if p.has_flip else 0.0

        if car.team == 1:
            kv[0, :, (1, 2)] = kv[0, :, (2, 1)]
            kv *= EARL_INVERT

        kv /= EARL_NORM

        q[0, 0, :24] = kv[0, main_idx, :].copy()
        q[0, 0, 24:] = self.prev_action

        # Convert to relative heading frame
        kv[..., 5:8] -= q[..., 5:8]
        forward = q[..., 11:14]
        theta = np.arctan2(forward[..., 0], forward[..., 1])
        theta = np.expand_dims(theta, axis=-1)
        ct = np.cos(theta)
        st = np.sin(theta)
        xs = kv[..., 5:20:3]
        ys = kv[..., 6:20:3]
        nx = ct * xs - st * ys
        ny = st * xs + ct * ys
        kv[..., 5:20:3] = nx
        kv[..., 6:20:3] = ny

        return (
            torch.from_numpy(q).to(self.device).float(),
            torch.from_numpy(kv).to(self.device).float(),
            torch.from_numpy(m).to(self.device).bool()
        )

    def _build_necto_inputs(self, car: CarState, arena: RocketSimArena) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_boosts = len(BOOST_LOCATIONS)
        n_players = len(arena.cars)
        qkv = np.zeros((1, 1 + n_players + n_boosts, 24), dtype=np.float32)

        # 1. Ball (Entity 0)
        qkv[0, 0, 3] = 1.0  # is_ball
        qkv[0, 0, 5:8] = arena.ball.pos
        qkv[0, 0, 8:11] = arena.ball.vel
        qkv[0, 0, 17:20] = arena.ball.ang_vel

        # 2. Players (Entities 1..N)
        main_idx = 1
        for idx, p in enumerate(arena.cars):
            n = 1 + idx
            if p is car:
                main_idx = n
                qkv[0, n, 0] = 1.0  # is_self
            elif p.team == car.team:
                qkv[0, n, 1] = 1.0  # is_teammate
            else:
                qkv[0, n, 2] = 1.0  # is_opponent

            qkv[0, n, 5:8] = p.pos
            qkv[0, n, 8:11] = p.vel
            qkv[0, n, 11:14] = p.get_forward_vector()
            qkv[0, n, 14:17] = p.get_up_vector()
            qkv[0, n, 17:20] = p.ang_vel
            qkv[0, n, 20] = p.boost / 100.0
            qkv[0, n, 21] = 0.0  # demo timer
            qkv[0, n, 22] = 1.0 if p.on_ground else 0.0
            qkv[0, n, 23] = 1.0 if p.has_flip else 0.0

        # 3. Boost pads
        n_boost_start = 1 + n_players
        qkv[0, n_boost_start:, 4] = 1.0  # is_boost
        qkv[0, n_boost_start:, 5:8] = BOOST_LOCATIONS
        qkv[0, n_boost_start:, 20] = 0.12 + 0.88 * (BOOST_LOCATIONS[:, 2] > 72)

        # Normalization
        qkv = qkv / EARL_NORM

        # Inversion for Orange team (Team 1)
        if car.team == 1:
            qkv[0, :, (1, 2)] = qkv[0, :, (2, 1)]
            qkv *= EARL_INVERT

        # Build query
        q = qkv[0, main_idx, :].copy()
        q = np.expand_dims(np.concatenate((q, self.prev_action), axis=0), axis=(0, 1))

        # Convert to relative coordinates
        kv = qkv.copy()
        kv[0, :, 5:11] -= q[0, 0, 5:11]

        mask = np.zeros((1, kv.shape[1]), dtype=bool)

        return (
            torch.from_numpy(q).to(self.device).float(),
            torch.from_numpy(kv).to(self.device).float(),
            torch.from_numpy(mask).to(self.device).bool()
        )

    def get_action(self, car: CarState, arena_or_ball: Union[RocketSimArena, BallState]) -> np.ndarray:
        if self.model is None or not isinstance(arena_or_ball, RocketSimArena):
            return BaselineChaser().get_action(car, arena_or_ball)

        try:
            if self.is_nexto:
                q_t, kv_t, mask_t = self._build_nexto_inputs(car, arena_or_ball)
                with torch.no_grad():
                    out = self.model((q_t, kv_t, mask_t))
                logits = out[0] if isinstance(out, (tuple, list)) else out
                best_idx = int(torch.argmax(logits, dim=-1).item())
                action = self.nexto_action_table[best_idx].numpy().copy()
            else:
                q_t, kv_t, mask_t = self._build_necto_inputs(car, arena_or_ball)
                with torch.no_grad():
                    out, _ = self.model((q_t, kv_t, mask_t))

                max_shape = max(o.shape[-1] for o in out)
                logits = torch.stack(
                    [
                        l if l.shape[-1] == max_shape
                        else torch.nn.functional.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                        for l in out
                    ]
                ).swapdims(0, 1).squeeze()

                actions = torch.argmax(logits, dim=-1).cpu().numpy().reshape((-1, 5))
                actions[:, 0] = actions[:, 0] - 1
                actions[:, 1] = actions[:, 1] - 1

                parsed = np.zeros((actions.shape[0], 8), dtype=np.float32)
                parsed[:, 0] = actions[:, 0]  # throttle
                parsed[:, 1] = actions[:, 1]  # steer
                parsed[:, 2] = actions[:, 0]  # pitch
                parsed[:, 3] = actions[:, 1] * (1 - actions[:, 4])  # yaw
                parsed[:, 4] = actions[:, 1] * actions[:, 4]  # roll
                parsed[:, 5] = actions[:, 2]  # jump
                parsed[:, 6] = actions[:, 3]  # boost
                parsed[:, 7] = actions[:, 4]  # handbrake
                action = parsed[0]

            self.prev_action = action.copy()
            return action
        except Exception as e:
            return BaselineChaser().get_action(car, arena_or_ball)


def create_opponent_bot(
    bot_type_or_path: Optional[str] = None,
    continuous_actions: bool = True,
    device: str = "cpu"
) -> BaseOpponent:
    """
    Factory function to instantiate opponent bots based on selection name or file path.
    Supports:
    - 'heuristic' / None / 'BaselineChaser': Rule-based HeuristicChaser
    - TorchScript models (.pt) such as necto-model.pt and nexto-model.pt
    - Checkpoint models (.pt) containing SenseiBot ActorCritic state dicts
    """
    if not bot_type_or_path or bot_type_or_path.lower() in ("heuristic", "baseline", "baselinechaser", "none"):
        return BaselineChaser(continuous_actions=continuous_actions)

    # Normalize file path
    clean_path = bot_type_or_path.strip().strip('"').strip("'")
    if not os.path.exists(clean_path):
        # Try checking in checkpoints/
        alt_path = os.path.join("checkpoints", os.path.basename(clean_path))
        if os.path.exists(alt_path):
            clean_path = alt_path
        else:
            print(f"[Opponent Bot] Path '{bot_type_or_path}' not found on disk. Falling back to BaselineChaser.")
            return BaselineChaser(continuous_actions=continuous_actions)

    # Check whether file is a TorchScript model or ActorCritic dict
    try:
        # First test if TorchScript
        try:
            ts_mod = torch.jit.load(clean_path, map_location="cpu")
            return NectoNextoOpponentBot(model_path=clean_path, device=device)
        except Exception:
            pass

        # Otherwise treat as Sensei ActorCritic checkpoint
        return CheckpointOpponentBot(model_path=clean_path, continuous_actions=continuous_actions, device=device)
    except Exception as e:
        print(f"[Opponent Bot] Failed to initialize opponent from '{clean_path}': {e}. Falling back to BaselineChaser.")
        return BaselineChaser(continuous_actions=continuous_actions)
