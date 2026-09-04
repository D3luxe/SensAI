"""
RLBot In-Game Agent Wrapper for SensAI.
Converts live Rocket League GameTickPacket data into model observations and returns controller inputs.
"""

from __future__ import annotations
import os
import io
import time
import math
import numpy as np
import torch
try:
    torch.set_flush_denormal(True)
except Exception:
    pass

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
from env.observations import DefaultObservationBuilder, OBS_MIRROR_MASK_NP, ACT_MIRROR_MASK_NP
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
    up = np.array([-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr], dtype=np.float32)
    right = np.array([
        fwd[1] * up[2] - fwd[2] * up[1],
        fwd[2] * up[0] - fwd[0] * up[2],
        fwd[0] * up[1] - fwd[1] * up[0]
    ], dtype=np.float32)
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
        self.ground_dodge_active = False
        self.fast_aerial_active = False
        self.dodge_cooldown = 0
        self.ball_touched_since_kickoff = False
        self.kickoff_stagnation_ticks = 0
        self.boost_pad_mapping: list[int] | None = None
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
                # Read into memory buffer first to avoid holding file locks on Windows
                ckpt_bytes = None
                for attempt in range(5):
                    try:
                        with open(ckpt_path, "rb") as f:
                            ckpt_bytes = f.read()
                        break
                    except (PermissionError, OSError):
                        time.sleep(0.05)
                
                if ckpt_bytes is None:
                    with open(ckpt_path, "rb") as f:
                        ckpt_bytes = f.read()

                buffer = io.BytesIO(ckpt_bytes)
                ckpt = torch.load(buffer, map_location=self.device)
                saved_state = ckpt.get("model_state_dict", {})
                
                # Robust continuous actions detection
                has_mean = "actor_mean.weight" in saved_state
                self.continuous_actions = ckpt.get("continuous_actions", has_mean)
                ckpt_act_dim = 8 if self.continuous_actions else self.discrete_parser.action_dim
                
                # Check if checkpoint contains LayerNorm parameters (1D tensor weights inside backbone)
                has_ln = any("LayerNorm" in k or (len(v.shape) == 1 and "bias" not in k and "log_std" not in k) for k, v in saved_state.items())
                use_ln = ckpt.get("use_layer_norm", has_ln)

                self.model = ActorCritic(
                    obs_dim=obs_dim,
                    act_dim=ckpt_act_dim,
                    continuous_actions=self.continuous_actions,
                    use_layer_norm=use_ln
                ).to(self.device)
                
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
                self.model.bin_thresh_logits.data = torch.tensor([-0.8473, -1.0986, -0.4055], dtype=torch.float32, device=self.device)
                self.model.debias_symmetric_actions()
                self.model.eval()
                self.loaded_ckpt_mtime = os.path.getmtime(ckpt_path) if os.path.exists(ckpt_path) else 0.0
                msg = f"[SensAI] Successfully loaded in-game model from {ckpt_path} (Mode: {'Continuous' if self.continuous_actions else f'Discrete RLGym ({ckpt_act_dim} actions)'}, ObsDim: {obs_dim}, LayerNorm: {use_ln})"
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
        if not getattr(packet.game_info, "is_round_active", True):
            controller.throttle = 1.0
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

            # Extract ball
            b_phys = packet.game_ball.physics
            ball_state = BallState(
                pos=np.array([b_phys.location.x, b_phys.location.y, b_phys.location.z], dtype=np.float32),
                vel=np.array([b_phys.velocity.x, b_phys.velocity.y, b_phys.velocity.z], dtype=np.float32),
                ang_vel=np.array([b_phys.angular_velocity.x, b_phys.angular_velocity.y, b_phys.angular_velocity.z], dtype=np.float32)
            )

            # Match and Kickoff State Tracking:
            # Detect new kickoff when is_kickoff_pause is True and ball is placed at center
            is_kickoff_pause = getattr(packet.game_info, "is_kickoff_pause", False)
            ball_speed = float(np.linalg.norm(ball_state.vel))
            ball_dist_center = float(np.linalg.norm(ball_state.pos[:2]))

            if is_kickoff_pause and ball_dist_center < 50.0 and ball_speed < 80.0:
                if self.ball_touched_since_kickoff:
                    self.ball_touched_since_kickoff = False
                    self.kickoff_stagnation_ticks = 0
                    self.prev_action = None
                    self.ticks_since_last_action = 0
            elif not self.ball_touched_since_kickoff:
                self.kickoff_stagnation_ticks += 1
                if ball_speed > 100.0 or ball_dist_center > 120.0 or self.kickoff_stagnation_ticks > 180:
                    self.ball_touched_since_kickoff = True

            # Extract self car
            my_car = packet.game_cars[self.index]
            is_on_ground = bool(my_car.has_wheel_contact)
            has_jump = is_on_ground or (not getattr(my_car, "jumped", False))
            # Align with RocketSim training: has_flip is only True when airborne and flip is available
            has_flip = bool((not is_on_ground) and (not getattr(my_car, "double_jumped", False)))

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
                has_flip=has_flip,
                ball_touches=1 if self.ball_touched_since_kickoff else 0
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
                def __init__(self, ball, cars, ball_pred=None, raw_pred_struct=None, game_boosts=None, boost_pad_mapping=None):
                    self.ball = ball
                    self.cars = cars
                    self.ball_prediction_slice = ball_pred
                    self._pred_struct = raw_pred_struct
                    self.boost_pads = BoostPad.create_standard_pads()
                    self._sm_pad_indices = np.array([i for i, p in enumerate(self.boost_pads) if not p.is_big], dtype=int)
                    self._bg_pad_indices = np.array([i for i, p in enumerate(self.boost_pads) if p.is_big], dtype=int)
                    self._small_pad_pos_3d = np.array([self.boost_pads[i].pos for i in self._sm_pad_indices], dtype=np.float32)
                    self._big_pad_pos_3d = np.array([self.boost_pads[i].pos for i in self._bg_pad_indices], dtype=np.float32)

                    if game_boosts is not None and boost_pad_mapping is not None:
                        for std_idx, packet_idx in enumerate(boost_pad_mapping):
                            if packet_idx < len(game_boosts):
                                self.boost_pads[std_idx].is_active = bool(game_boosts[packet_idx].is_active)

                    self._small_pad_active = np.array([self.boost_pads[i].is_active for i in self._sm_pad_indices], dtype=bool)
                    self._big_pad_active = np.array([self.boost_pads[i].is_active for i in self._bg_pad_indices], dtype=bool)

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
                    opp_flip = bool((not opp_on_ground) and (not getattr(opp_car, "double_jumped", False)))
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
                        has_flip=opp_flip,
                        ball_touches=1 if self.ball_touched_since_kickoff else 0
                    ))

            self.ticks_since_last_action += 1
            if self.ticks_since_last_action >= self.tick_skip or self.prev_action is None:
                self.ticks_since_last_action = 0

                # Compute boost pad spatial mapping once when FieldInfo is available
                if self.boost_pad_mapping is None and RLBOT_AVAILABLE and hasattr(self, "get_field_info"):
                    try:
                        field_info = self.get_field_info()
                        if field_info is not None and getattr(field_info, "num_boosts", 0) > 0:
                            std_pads = BoostPad.create_standard_pads()
                            mapping = []
                            for std_pad in std_pads:
                                best_idx = 0
                                min_dist = float("inf")
                                for b_i in range(field_info.num_boosts):
                                    loc = field_info.boost_pads[b_i].location
                                    d = math.hypot(loc.x - std_pad.pos[0], loc.y - std_pad.pos[1])
                                    if d < min_dist:
                                        min_dist = d
                                        best_idx = b_i
                                mapping.append(best_idx)
                            self.boost_pad_mapping = mapping
                    except Exception:
                        pass

                game_boosts = getattr(packet, "game_boosts", None)
                arena = MockArena(
                    ball_state, [car_state] + opponents,
                    ball_pred=ball_prediction_slice,
                    raw_pred_struct=pred_struct,
                    game_boosts=game_boosts,
                    boost_pad_mapping=self.boost_pad_mapping
                )
                obs = self.obs_builder.build_obs(car_state, arena)
                self.latest_obs = obs

                # Model Inference at 15Hz (ActorCritic evaluates native equivariant bilateral policy)
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

            # ── Rocket League In-Game Controller Input Mapping ─────────────────────────
            # Continuous Action Vector: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
            #
            # RLBot Controller Axes:
            #  - Throttle: Direct (+1.0 Forward, -1.0 Reverse)
            #  - Steer:    Direct (+1.0 Steer Right, -1.0 Steer Left)
            #  - Yaw:      Direct (+1.0 Yaw Right, -1.0 Yaw Left)
            #  - Roll:     Direct (+1.0 Roll Right, -1.0 Roll Left)
            # Action mapping aligned with RocketSim physics engine:
            controller.throttle = float(np.clip(act[0], -1.0, 1.0))
            controller.steer = -float(np.clip(act[1], -1.0, 1.0))
            controller.pitch = -float(np.clip(act[2], -1.0, 1.0))
            controller.yaw = -float(np.clip(act[3], -1.0, 1.0))
            controller.roll = -float(np.clip(act[4], -1.0, 1.0))

            # Dodge cooldown countdown
            if self.dodge_cooldown > 0:
                self.dodge_cooldown -= 1

            fwd_speed = float(np.dot(car_state.vel, car_state.get_forward_vector()))
            car_speed_total = float(np.linalg.norm(car_state.vel))
            is_supersonic = bool(car_state.is_supersonic if hasattr(car_state, "is_supersonic") else car_speed_total >= 2200.0)

            # Spatial context relative to ball
            delta_to_ball = ball_state.pos - car_state.pos
            dist_to_ball = float(np.linalg.norm(delta_to_ball))
            fwd_vec = car_state.get_forward_vector()
            right_vec = car_state.get_right_vector()
            unit_ball = delta_to_ball / max(1.0, dist_to_ball)
            fwd_align = float(np.dot(fwd_vec, unit_ball))
            local_x = float(np.dot(delta_to_ball, fwd_vec))
            local_y = float(np.dot(delta_to_ball, right_vec))
            ball_spd = float(np.linalg.norm(ball_state.vel))

            # ── RLGym / RLBot Jump & Dodge Substep Timing Sequencer ────────────
            # Controls jump button release/press timing across the 8 physics substeps:
            #  - Ground Liftoff: Hold jump for ticks 0..3, release ticks 4..7 to prime airborne dodge.
            #  - Airborne Dodge: Press jump on ticks 2..5 to activate second jump / dodge.
            want_jump = bool(act[5] > 0.0)
            substep_tick = self.ticks_since_last_action  # 0 to 7 within the 15Hz step

            # Downfield Traversal Speed-Flip Trigger:
            # When sprinting downfield on open turf toward the ball (dist > 750 uu, speed > 650 uu/s, aligned),
            # if the policy requests forward drive (act[0] > 0.50) and car is not yet supersonic,
            # initiate a forward speed-flip if cooldown has elapsed.
            if is_on_ground and self.dodge_cooldown == 0:
                if dist_to_ball > 750.0 and fwd_speed > 650.0 and car_speed_total < 1950.0 and abs(act[1]) < 0.25 and fwd_align > 0.65 and act[0] > 0.50:
                    want_jump = True

            # Ground Jump Gating:
            # 1. Flip Cooldown: Require recovery ticks on wheels after a dodge before jumping again.
            # 2. Hard Turning: When steering hard on turf (abs(act[1]) > 0.35), steer on wheels instead of tumbling.
            # 3. Supersonic: When already at supersonic speed on the ground, do not dodge for speed.
            if is_on_ground:
                if self.dodge_cooldown > 0 or abs(act[1]) > 0.35 or is_supersonic:
                    want_jump = False
                controller.jump = bool(want_jump and substep_tick <= 3)
            else:
                # Airborne Dodge / Second Jump:
                # RocketSim and Rocket League physics require jump release while airborne before second jump.
                # When jumping off the ground, wheels leave turf on ticks 1-2.
                # Pressing jump on ticks 2..5 guarantees the wheel-lift check passes and executes the dodge.
                controller.jump = bool(want_jump and has_flip and 2 <= substep_tick <= 5)

                if controller.jump:
                    self.dodge_cooldown = 20  # ~1.3 second recovery after dodge

                    # Directional Flip Sanitation:
                    # In Rocket League:
                    #  - controller.pitch = -1.0 is nose-DOWN / FRONT-FLIP
                    #  - controller.pitch = +1.0 is nose-UP / BACKFLIP
                    # When moving forward downfield or into the ball (fwd_speed > 100 uu/s):
                    # Any dodge MUST be forward or diagonal. Invert accidental backward stick to front-flip!
                    if fwd_speed > 100.0:
                        if controller.pitch > 0.0:
                            controller.pitch = -controller.pitch
                        if abs(controller.pitch) < 0.2 and abs(controller.yaw) < 0.2:
                            controller.pitch = -1.0

                    # Dodge Deadzone Compensation:
                    # Rocket League and RocketSim require analog stick deflection >= 0.50 to execute a directional flip/dodge.
                    # When an airborne dodge is triggered, scale directional stick deflection past the deadzone threshold
                    # so continuous policy outputs execute genuine forward, backward, or diagonal dodges instead of empty double jumps.
                    stick_mag = math.hypot(controller.pitch, controller.yaw)
                    if stick_mag > 0.08:
                        scale = max(1.0, 0.90 / stick_mag)
                        controller.pitch = float(np.clip(controller.pitch * scale, -1.0, 1.0))
                        controller.yaw = float(np.clip(controller.yaw * scale, -1.0, 1.0))

                # Half-Flip Recovery: If doing a reverse backflip, cancel and roll when inverted
                elif not is_on_ground and not has_flip and fwd_speed < -100.0:
                    up_vec = car_state.get_up_vector()
                    if up_vec[2] < 0.0:  # Upside down during backflip
                        controller.pitch = -1.0  # Flip cancel forward!
                        controller.roll = 1.0   # Air-roll to land on wheels!

            # Ground stabilization:
            # When driving on the ground without jumping, keep pitch, yaw, and roll neutral
            # so the vehicle steers purely via ground wheel physics without airborne gyro torque conflict.
            if is_on_ground and not want_jump:
                controller.pitch = 0.0
                controller.yaw = 0.0
                controller.roll = 0.0

            # Boost Economy & Momentum Safety Gate:
            # 1. Suppress boost when the vehicle's 3D momentum strongly opposes its nose direction (fwd_speed < -150 uu/s).
            # 2. Suppress boost on the ground when already at supersonic speed (is_supersonic), preventing boost waste.
            controller.boost = bool(act[6] > 0.0 and fwd_speed > -150.0 and not (is_supersonic and is_on_ground))
            controller.handbrake = bool(act[7] > 0.0 and is_on_ground)

            # Dribble Pacing & Anti-Overshoot:
            # When alongside the ball (dist < 350 uu, ball on ground), if the car is outrunning the ball,
            # cut boost and pace throttle to match ball speed so the car can turn/cut the ball toward net!
            if dist_to_ball < 350.0 and ball_state.pos[2] < 160.0 and is_on_ground:
                if fwd_speed > ball_spd + 80.0:
                    controller.boost = False
                    controller.throttle = float(np.clip(ball_spd / max(1.0, fwd_speed), 0.25, 0.85))

            # Ball-Behind Turnaround Recovery:
            # If the ball is behind the car (local_x < -60 uu) while moving forward (fwd_speed > 80 uu/s),
            # powerslide turnaround rapidly toward the ball instead of straight-line reversing into a freeze!
            if is_on_ground and local_x < -60.0 and fwd_speed > 80.0:
                turn_dir = float(np.sign(local_y)) if abs(local_y) > 20.0 else 1.0
                controller.steer = turn_dir
                controller.handbrake = True
                controller.throttle = 0.50

            self.tick_count += 1
            ball_pos = ball_state.pos
            is_kickoff = bool(abs(ball_pos[0]) < 50.0 and abs(ball_pos[1]) < 50.0 and float(np.linalg.norm(ball_state.vel)) < 100.0)
            
            # Event logging on jump/dodge trigger
            if controller.jump and substep_tick == 0:
                action_type = "LIFTOFF JUMP" if is_on_ground else "AIRBORNE FLIP"
                log_debug(f"[TICK {self.tick_count}] *** {action_type} EXECUTED *** pit={controller.pitch:+.2f} yaw={controller.yaw:+.2f} rol={controller.roll:+.2f}")

            if self.tick_count <= 10 or self.tick_count % 120 == 0 or is_kickoff:
                log_debug(
                    f"[TICK {self.tick_count}] pos=({car_state.pos[0]:.0f}, {car_state.pos[1]:.0f}) "
                    f"ball=({ball_pos[0]:.0f}, {ball_pos[1]:.0f}) kickoff={is_kickoff} -> "
                    f"thr={controller.throttle:.2f} str={controller.steer:+.2f} pit={controller.pitch:+.2f} "
                    f"yaw={controller.yaw:+.2f} rol={controller.roll:+.2f} jmp={controller.jump} bst={controller.boost} hnd={controller.handbrake}"
                )

        except Exception as e:
            import traceback
            err_msg = f"[SensAI] Error in get_output: {e}\n{traceback.format_exc()}"
            print(err_msg)
            log_debug(f"[TICK_ERROR] {err_msg}")

        return controller
