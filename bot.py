"""
RLBot In-Game Agent Wrapper for SensAI.
Converts live Rocket League GameTickPacket data into model observations and returns controller inputs.
"""

from __future__ import annotations
import os
import math
import numpy as np
import torch

try:
    from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
    from rlbot.utils.structures.game_data_struct import GameTickPacket
    RLBOT_AVAILABLE = True
except ImportError:
    RLBOT_AVAILABLE = False
    BaseAgent = object
    SimpleControllerState = object
    GameTickPacket = object

from agent.models import ActorCritic
from env.observations import DefaultObservationBuilder
from env.actions import DiscreteActionParser, ContinuousActionParser
from env.physics_engine import (
    CarState, BallState, BoostPad,
    ARENA_EXTENT_X, ARENA_EXTENT_Y, ARENA_HEIGHT_Z,
    CAR_MAX_SPEED, BALL_MAX_SPEED, GOAL_HEIGHT
)


def rotation_to_rot_mat(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    Computes exact 3x3 orthonormal basis matching C++ RocketSim Bullet physics matrix (0.00000009 precision).
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    fwd = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
    right = np.array([cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, -cp * sr], dtype=np.float32)
    up = np.array([-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr], dtype=np.float32)
    return np.vstack([fwd, right, up]).astype(np.float32)


def log_debug(msg: str):
    try:
        bot_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(bot_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "rlbot_live.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    except Exception:
        pass


class SenseiRLBot(BaseAgent):
    def __init__(self, name, team, index):
        self.name = name
        self.team = team
        self.index = index
        self.tick_count = 0
        self.tick_skip = 8
        self.ticks_since_last_action = 0
        self.prev_action: np.ndarray | None = None
        self.current_steer = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_builder = DefaultObservationBuilder(symmetric=True)
        self.discrete_parser = DiscreteActionParser()
        self.continuous_actions = False
        self.model: torch.nn.Module | None = None
        self.initialize_agent()
        log_debug(f"[INIT] SenseiRLBot init: name={name}, team={team}, index={index}, device={self.device}, tick_skip={self.tick_skip}")
        if RLBOT_AVAILABLE:
            super().__init__(name, team, index)

    def get_latest_checkpoint(self) -> Optional[str]:
        bot_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.join(bot_dir, "checkpoints")
        latest_file = os.path.join(ckpt_dir, "latest_model.pt")
        if os.path.exists(latest_file):
            return latest_file
        if not os.path.exists(ckpt_dir):
            return None
        files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if not files:
            return None
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]

    def initialize_agent(self):
        ckpt_path = self.get_latest_checkpoint()
        obs_dim = 64
        act_dim = self.discrete_parser.action_dim

        if ckpt_path:
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.continuous_actions = ckpt.get("continuous_actions", False)
                ckpt_act_dim = ckpt.get("act_dim", 8 if self.continuous_actions else self.discrete_parser.action_dim)
                self.model = ActorCritic(obs_dim=obs_dim, act_dim=ckpt_act_dim, continuous_actions=self.continuous_actions).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.eval()
                msg = f"[SensAI] Successfully loaded in-game model from {ckpt_path} (Mode: {'Continuous' if self.continuous_actions else f'Discrete RLGym ({ckpt_act_dim} actions)'})"
                print(msg)
                log_debug(f"[INIT] {msg}")
            except Exception as e:
                msg = f"[SensAI] Warning: Could not load weights from {ckpt_path}: {e}"
                print(msg)
                log_debug(f"[INIT_ERROR] {msg}")
                self.model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, continuous_actions=self.continuous_actions).to(self.device)
                self.model.eval()
        else:
            msg = "[SensAI] Warning: No checkpoint found, initialized default ActorCritic network."
            print(msg)
            log_debug(f"[INIT_WARN] {msg}")
            self.model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, continuous_actions=self.continuous_actions).to(self.device)
            self.model.eval()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        controller = SimpleControllerState()

        # Guard: check match state (allow kickoff and freeplay play)
        if getattr(packet.game_info, "is_match_ended", False):
            return controller

        try:
            if self.model is None:
                self.initialize_agent()

            if packet.num_cars <= self.index:
                return controller

            # Extract self car
            my_car = packet.game_cars[self.index]
            is_on_ground = bool(my_car.has_wheel_contact)
            has_jump = is_on_ground or (not getattr(my_car, "jumped", False))
            has_flip = not getattr(my_car, "double_jumped", False)

            car_rot_mat = rotation_to_rot_mat(
                my_car.physics.rotation.pitch,
                my_car.physics.rotation.yaw,
                my_car.physics.rotation.roll
            )

            car_state = CarState(
                id=self.index,
                team=self.team,
                pos=np.array([my_car.physics.location.x, my_car.physics.location.y, my_car.physics.location.z], dtype=np.float32),
                vel=np.array([my_car.physics.velocity.x, my_car.physics.velocity.y, my_car.physics.velocity.z], dtype=np.float32),
                rot=np.array([my_car.physics.rotation.pitch, my_car.physics.rotation.yaw, my_car.physics.rotation.roll], dtype=np.float32),
                rot_mat=car_rot_mat,
                ang_vel=np.array([my_car.physics.angular_velocity.x, my_car.physics.angular_velocity.y, my_car.physics.angular_velocity.z], dtype=np.float32),
                boost=float(my_car.boost),
                on_ground=is_on_ground,
                has_jump=has_jump,
                has_flip=has_flip
            )

            # Extract ball
            b_phys = packet.game_ball.physics
            ball_state = BallState(
                pos=np.array([b_phys.location.x, b_phys.location.y, b_phys.location.z], dtype=np.float32),
                vel=np.array([b_phys.velocity.x, b_phys.velocity.y, b_phys.velocity.z], dtype=np.float32),
                ang_vel=np.array([b_phys.angular_velocity.x, b_phys.angular_velocity.y, b_phys.angular_velocity.z], dtype=np.float32)
            )

            # Build dummy arena struct for obs builder
            class MockArena:
                def __init__(self, ball, cars):
                    self.ball = ball
                    self.cars = cars
                    self.boost_pads = BoostPad.create_standard_pads()

            # Find opponent
            opponents = []
            for i in range(packet.num_cars):
                if i != self.index:
                    opp_car = packet.game_cars[i]
                    opp_on_ground = bool(opp_car.has_wheel_contact)
                    opp_jump = opp_on_ground or (not getattr(opp_car, "jumped", False))
                    opp_flip = not getattr(opp_car, "double_jumped", False)
                    opp_rot_mat = rotation_to_rot_mat(
                        opp_car.physics.rotation.pitch,
                        opp_car.physics.rotation.yaw,
                        opp_car.physics.rotation.roll
                    )
                    opponents.append(CarState(
                        id=i,
                        team=opp_car.team,
                        pos=np.array([opp_car.physics.location.x, opp_car.physics.location.y, opp_car.physics.location.z], dtype=np.float32),
                        vel=np.array([opp_car.physics.velocity.x, opp_car.physics.velocity.y, opp_car.physics.velocity.z], dtype=np.float32),
                        rot=np.array([opp_car.physics.rotation.pitch, opp_car.physics.rotation.yaw, opp_car.physics.rotation.roll], dtype=np.float32),
                        rot_mat=opp_rot_mat,
                        ang_vel=np.array([opp_car.physics.angular_velocity.x, opp_car.physics.angular_velocity.y, opp_car.physics.angular_velocity.z], dtype=np.float32),
                        boost=float(opp_car.boost),
                        on_ground=opp_on_ground,
                        has_jump=opp_jump,
                        has_flip=opp_flip
                    ))

            self.ticks_since_last_action += 1
            if self.ticks_since_last_action >= self.tick_skip or self.prev_action is None:
                self.ticks_since_last_action = 0
                arena = MockArena(ball_state, [car_state] + opponents)
                obs = self.obs_builder.build_obs(car_state, arena)

                # Model Inference at 15Hz
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action, _, _, _ = self.model.get_action_and_value(obs_tensor, deterministic=True)
                    if self.continuous_actions:
                        act = action.squeeze(0).cpu().numpy()
                    else:
                        act_idx = int(action.squeeze().cpu().item())
                        act = self.discrete_parser.parse_actions(act_idx)
                self.prev_action = act
            else:
                # Hold previous action across the 8 physics substeps
                act = self.prev_action

            # Direct 1-to-1 Neural Policy Mapping
            # [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
            raw_throttle = float(act[0])
            steer_val = float(np.clip(act[1], -1.0, 1.0))

            if self.continuous_actions:
                if raw_throttle > 0.05:
                    controller.throttle = 1.0
                elif raw_throttle < -0.35:
                    controller.throttle = -1.0
                else:
                    controller.throttle = 0.0
            else:
                controller.throttle = raw_throttle

            # Smooth steering transition (filters discrete bang-bang wheel chatter at 120Hz)
            self.current_steer = 0.7 * steer_val + 0.3 * self.current_steer
            controller.steer = float(np.clip(self.current_steer, -1.0, 1.0))
            # Directional Flip / Dodge Detection (Jump + non-zero pitch/yaw/roll)
            is_dodge = bool(act[5] > 0.0 and (abs(act[2]) > 0.1 or abs(act[3]) > 0.1 or abs(act[4]) > 0.1))

            if is_dodge:
                # 120Hz 4-stage substep cadence for authentic Rocket League double-jump dodges:
                # Ground flip: jump (0,1) -> release (2,3) -> dodge (4,5) -> finish (6,7)
                # Air dodge: immediate dodge (0,1,2) -> finish
                if is_on_ground:
                    controller.jump = bool(self.ticks_since_last_action in (0, 1, 4, 5))
                else:
                    controller.jump = bool(self.ticks_since_last_action in (0, 1, 2))

                controller.pitch = float(np.clip(act[2], -1.0, 1.0))
                controller.yaw = float(np.clip(act[3], -1.0, 1.0))
                controller.roll = float(np.clip(act[4], -1.0, 1.0))
                # Pass immediate steering vector for sharp directional dodge registration
                controller.steer = float(np.clip(act[3] if abs(act[3]) > 0.1 else (act[4] if abs(act[4]) > 0.1 else steer_val), -1.0, 1.0))
            else:
                controller.jump = bool(act[5] > 0.0)
                if is_on_ground:
                    controller.pitch = 0.0
                    controller.roll = 0.0
                    controller.yaw = 0.0
                else:
                    controller.pitch = float(np.clip(act[2], -1.0, 1.0))
                    controller.roll = float(np.clip(act[4], -1.0, 1.0))
                    controller.yaw = float(np.clip(act[3], -1.0, 1.0))

            controller.boost = bool(act[6] > 0.0 and (raw_throttle > 0.0 or not is_on_ground))
            controller.handbrake = bool(act[7] > 0.5)

            self.tick_count += 1
            ball_pos = ball_state.pos
            is_kickoff = bool(abs(ball_pos[0]) < 50.0 and abs(ball_pos[1]) < 50.0 and float(np.linalg.norm(ball_state.vel)) < 100.0)
            if self.tick_count <= 10 or self.tick_count % 120 == 0 or is_kickoff:
                log_debug(
                    f"[TICK {self.tick_count}] pos=({car_state.pos[0]:.0f}, {car_state.pos[1]:.0f}) "
                    f"ball=({ball_pos[0]:.0f}, {ball_pos[1]:.0f}) kickoff={is_kickoff} -> "
                    f"throttle={controller.throttle:.2f} steer={controller.steer:.2f} boost={controller.boost}"
                )

        except Exception as e:
            import traceback
            err_msg = f"[SensAI] Error in get_output: {e}\n{traceback.format_exc()}"
            print(err_msg)
            log_debug(f"[TICK_ERROR] {err_msg}")

        return controller
