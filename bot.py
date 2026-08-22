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
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    fwd = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
    right = np.array([cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, -cp * sr], dtype=np.float32)
    up = np.array([-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr], dtype=np.float32)
    return np.array([fwd, right, up], dtype=np.float32)


class SenseiRLBot(BaseAgent):
    def __init__(self, name, team, index):
        if RLBOT_AVAILABLE:
            super().__init__(name, team, index)
        self.name = name
        self.team = team
        self.index = index
        self.model: torch.nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_builder = DefaultObservationBuilder(symmetric=True)
        self.discrete_parser = DiscreteActionParser()
        self.continuous_actions = False

    def get_latest_checkpoint(self) -> Optional[str]:
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
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
        act_dim = 8

        if ckpt_path:
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.continuous_actions = ckpt.get("continuous_actions", False)
                act_dim = ckpt.get("act_dim", 8 if self.continuous_actions else self.discrete_parser.action_dim)
                self.model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, continuous_actions=self.continuous_actions).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.eval()
                print(f"[SensAI] Loaded in-game model from {ckpt_path} (Mode: {'Continuous' if self.continuous_actions else 'Discrete RLGym (19 actions)'})")
            except Exception as e:
                print(f"[SensAI] Warning: Could not load weights from {ckpt_path}: {e}")
                self.model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, continuous_actions=self.continuous_actions).to(self.device)
                self.model.eval()
        else:
            print("[SensAI] Warning: No checkpoint found, initialized untrained network.")
            self.model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, continuous_actions=self.continuous_actions).to(self.device)
            self.model.eval()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        controller = SimpleControllerState()

        if not packet.game_info.is_round_active:
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

        arena = MockArena(ball_state, [car_state] + opponents)
        obs = self.obs_builder.build_obs(car_state, arena)

        # Model Inference
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _, _, _ = self.model.get_action_and_value(obs_tensor, deterministic=True)
            if self.continuous_actions:
                act = action.squeeze(0).cpu().numpy()
            else:
                act_idx = int(action.squeeze().cpu().item())
                act = self.discrete_parser.parse_actions(act_idx)

        # Action Mapping
        # [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        raw_throttle = float(act[0])
        steer_val = float(np.clip(act[1], -1.0, 1.0))

        if self.continuous_actions:
            # Continuous deadband
            if raw_throttle > 0.05:
                controller.throttle = 1.0
            elif raw_throttle < -0.35:
                controller.throttle = -1.0
            else:
                controller.throttle = 0.0
        else:
            # Discrete lookup table already contains exact discrete values
            controller.throttle = raw_throttle

        controller.steer = steer_val
        controller.pitch = float(np.clip(act[2], -1.0, 1.0))
        controller.yaw = float(np.clip(act[3], -1.0, 1.0))
        controller.roll = float(np.clip(act[4], -1.0, 1.0))
        controller.boost = bool(act[6] > 0.0 and (raw_throttle > 0.0 or not is_on_ground))

        # Kickoff strike-through commitment (ensures bots strike through the kickoff ball)
        ball_pos = ball_state.pos
        ball_speed = float(np.linalg.norm(ball_state.vel))
        is_kickoff = (abs(ball_pos[0]) < 50.0 and abs(ball_pos[1]) < 50.0 and ball_speed < 100.0)
        
        car_to_ball = ball_pos - car_state.pos
        dist_to_ball = float(np.linalg.norm(car_to_ball))
        if dist_to_ball > 1e-4:
            unit_to_ball = car_to_ball / dist_to_ball
            fwd_align = float(np.dot(car_state.get_forward_vector(), unit_to_ball))
            if is_kickoff and fwd_align > 0.3:
                controller.throttle = 1.0
                controller.boost = bool(car_state.boost > 0)

        # Handbrake: only engage for sharp low-to-medium speed turns to prevent involuntary high-speed spinouts
        car_speed = float(np.linalg.norm(car_state.vel))
        controller.handbrake = bool(act[7] > 0.6 and abs(steer_val) > 0.6 and car_speed < 1400.0 and not is_kickoff)

        # Direct 1-to-1 Jump mapping (eliminates involuntary spastic auto-flips)
        controller.jump = bool(act[5] > 0.0)

        # In-air vs on-ground orientation stabilization
        if is_on_ground:
            # On ground: steering only (prevent residual pitch/roll from causing awkward aerial twitches on minor bumps)
            controller.pitch = 0.0
            controller.roll = 0.0
        else:
            controller.pitch = float(np.clip(act[2], -1.0, 1.0))
            controller.roll = float(np.clip(act[4], -1.0, 1.0))

        return controller
