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
    class SimpleControllerState:
        def __init__(self):
            self.steer = 0.0
            self.throttle = 0.0
            self.pitch = 0.0
            self.yaw = 0.0
            self.roll = 0.0
            self.jump = False
            self.boost = False
            self.handbrake = False
            self.use_item = False
    BaseAgent = object
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
    Computes exact 3x3 orthonormal basis (Row 0: Forward, Row 1: Right, Row 2: Up).
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    fwd = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
    right = np.array([sy * cr - cy * sp * sr, -cy * cr - sy * sp * sr, cp * sr], dtype=np.float32)
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
        self.ground_dodge_active = False
        self.fast_aerial_active = False
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
        obs_dim = self.obs_builder.obs_dim
        act_dim = self.discrete_parser.action_dim

        if ckpt_path:
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.continuous_actions = ckpt.get("continuous_actions", False)
                ckpt_act_dim = 8 if self.continuous_actions else self.discrete_parser.action_dim
                self.model = ActorCritic(obs_dim=obs_dim, act_dim=ckpt_act_dim, continuous_actions=self.continuous_actions).to(self.device)
                
                saved_state = ckpt["model_state_dict"]
                model_state = self.model.state_dict()
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
                    self.model.load_state_dict(model_state)
                else:
                    self.model.load_state_dict(saved_state)
                self.model.debias_symmetric_actions()
                self.model.eval()
                self.loaded_ckpt_mtime = os.path.getmtime(ckpt_path) if os.path.exists(ckpt_path) else 0.0
                msg = f"[SensAI] Successfully loaded in-game model from {ckpt_path} (Mode: {'Continuous' if self.continuous_actions else f'Discrete RLGym ({ckpt_act_dim} actions)'}, ObsDim: {obs_dim})"
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
            # Periodic live check for newer training checkpoints (every 120 ticks = 1 second)
            if self.tick_count % 120 == 0:
                latest_ckpt = self.get_latest_checkpoint()
                if latest_ckpt and os.path.exists(latest_ckpt):
                    mtime = os.path.getmtime(latest_ckpt)
                    if mtime > getattr(self, "loaded_ckpt_mtime", 0.0):
                        self.initialize_agent()

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

            # Extract future ball trajectory from RLBot
            ball_prediction_slice = None
            pred_struct = None
            if RLBOT_AVAILABLE and hasattr(self, "get_ball_prediction_struct"):
                try:
                    pred_struct = self.get_ball_prediction_struct()
                    if pred_struct is not None and getattr(pred_struct, "num_slices", 0) > 30:
                        slice_idx = min(30, pred_struct.num_slices - 1)
                        loc = pred_struct.slices[slice_idx].physics.location
                        ball_prediction_slice = np.array([loc.x, loc.y, loc.z], dtype=np.float32)
                except Exception:
                    pass

            # Build dummy arena struct for obs builder
            class MockArena:
                def __init__(self, ball, cars, ball_pred=None, raw_pred_struct=None):
                    self.ball = ball
                    self.cars = cars
                    self.ball_prediction_slice = ball_pred
                    self._pred_struct = raw_pred_struct
                    self.boost_pads = BoostPad.create_standard_pads()

                def get_shot_threat(self, team: int):
                    defending_goal_y = -ARENA_EXTENT_Y if team == 0 else ARENA_EXTENT_Y
                    ball_vy = self.ball.vel[1]
                    is_moving_to_net = (ball_vy < -100.0) if team == 0 else (ball_vy > 100.0)
                    if not is_moving_to_net:
                        return False, 0.0, 0.0

                    if self._pred_struct is not None and getattr(self._pred_struct, "num_slices", 0) > 0:
                        num = min(self._pred_struct.num_slices, 120)
                        for i in range(num):
                            loc = self._pred_struct.slices[i].physics.location
                            if (team == 0 and loc.y <= -5120.0) or (team == 1 and loc.y >= 5120.0):
                                if abs(loc.x) < 950.0 and 0.0 < loc.z < 680.0:
                                    threat_intensity = max(0.1, 1.0 - (i / 120.0))
                                    entry_z_norm = min(1.0, max(0.0, loc.z / GOAL_HEIGHT))
                                    return True, threat_intensity, entry_z_norm

                    dy = defending_goal_y - self.ball.pos[1]
                    if abs(ball_vy) > 1e-4:
                        dt = dy / ball_vy
                        if 0.05 < dt < 3.0:
                            pred_x = self.ball.pos[0] + self.ball.vel[0] * dt
                            pred_z = self.ball.pos[2] + self.ball.vel[2] * dt + 0.5 * (-650.0) * (dt ** 2)
                            if abs(pred_x) < 950.0 and 0.0 < pred_z < 680.0:
                                threat_intensity = max(0.1, 1.0 - (dt / 3.0))
                                entry_z_norm = min(1.0, max(0.0, pred_z / GOAL_HEIGHT))
                                return True, threat_intensity, entry_z_norm

                    return False, 0.0, 0.0

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
                arena = MockArena(ball_state, [car_state] + opponents, ball_pred=ball_prediction_slice, raw_pred_struct=pred_struct)
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

                # Determine if a ground flip or fast aerial is initiated on tick 0
                jump_threshold = 0.5 if self.continuous_actions else 0.0
                jump_req = bool(act[5] > jump_threshold)
                self.fast_aerial_active = bool(jump_req and is_on_ground and act[2] > 0.1 and act[6] > 0.3)
                self.ground_dodge_active = bool(jump_req and is_on_ground and not self.fast_aerial_active)
            else:
                # Hold previous action across the 8 physics substeps
                act = self.prev_action

            # 1-to-1 Neural Policy Mapping (Matched with training physics engine)
            # Continuous Action vector: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
            raw_steer = float(np.clip(act[1], -1.0, 1.0))
            # 120Hz Smooth Steering Filter (Eliminates high-frequency wheel chatter while retaining instant response)
            self.current_steer = 0.65 * raw_steer + 0.35 * self.current_steer
            controller.throttle = float(np.clip(act[0], -1.0, 1.0))
            # RLBot Gamepad Steer (Inverted to match RocketSim training action mapping)
            controller.steer = -float(np.clip(self.current_steer, -1.0, 1.0))

            jump_threshold = 0.5 if self.continuous_actions else 0.0
            jump_requested = bool(act[5] > jump_threshold)

            # Aerial controls: active when airborne or when deliberately jumping/dodging
            if is_on_ground and not jump_requested:
                # Ground stability: keep air pitch and roll neutral to prevent death-rolls over wall curves and bumps
                controller.pitch = 0.0
                controller.yaw = -float(np.clip(act[3], -1.0, 1.0))
                controller.roll = 0.0
            else:
                controller.pitch = float(np.clip(act[2], -1.0, 1.0))
                controller.yaw = -float(np.clip(act[3], -1.0, 1.0))
                controller.roll = float(np.clip(act[4], -1.0, 1.0))

            # Double-Jump & Dodge 120Hz Substep Cadence (Allows natural speed-flips, wave-dashes, and aerials)
            if jump_requested:
                if self.ground_dodge_active:
                    # Ground flip: jump (ticks 0,1) -> release (ticks 2,3) -> dodge click (ticks 4,5)
                    controller.jump = bool(self.ticks_since_last_action in (0, 1, 4, 5))
                elif self.fast_aerial_active:
                    # Fast aerial climb: jump (ticks 0,1) -> release (ticks 2,3) -> double-jump click (ticks 4,5)
                    controller.jump = bool(self.ticks_since_last_action in (0, 1, 4, 5))
                    if self.ticks_since_last_action in (4, 5):
                        controller.pitch = 0.0
                        controller.steer = 0.0
                        controller.yaw = 0.0
                        controller.roll = 0.0
                else:
                    # Aerial dodge / jump
                    controller.jump = bool(self.ticks_since_last_action in (0, 1, 2))
            else:
                controller.jump = False

            boost_threshold = 0.3 if self.continuous_actions else 0.0
            controller.boost = bool(act[6] > boost_threshold)
            controller.handbrake = bool(act[7] > 0.5)

            self.tick_count += 1
            ball_pos = ball_state.pos
            is_kickoff = bool(abs(ball_pos[0]) < 50.0 and abs(ball_pos[1]) < 50.0 and float(np.linalg.norm(ball_state.vel)) < 100.0)
            if self.tick_count <= 10 or self.tick_count % 120 == 0 or is_kickoff:
                log_debug(
                    f"[TICK {self.tick_count}] pos=({car_state.pos[0]:.0f}, {car_state.pos[1]:.0f}) "
                    f"ball=({ball_pos[0]:.0f}, {ball_pos[1]:.0f}) kickoff={is_kickoff} -> "
                    f"thr={controller.throttle:.2f} str={controller.steer:+.2f} pit={controller.pitch:+.2f} "
                    f"yaw={controller.yaw:+.2f} rol={controller.roll:+.2f} jmp={controller.jump} bst={controller.boost}"
                )

        except Exception as e:
            import traceback
            err_msg = f"[SensAI] Error in get_output: {e}\n{traceback.format_exc()}"
            print(err_msg)
            log_debug(f"[TICK_ERROR] {err_msg}")

        return controller
