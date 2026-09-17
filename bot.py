"""
RLBot In-Game Agent Wrapper for SensAI.
Converts live Rocket League GamePacket data into model observations and returns controller inputs.

Targets the RLBot v5 python-interface. The v5 ball prediction runs at 120 Hz, the same rate as
the RocketSim arena used in training, so a prediction index means the same amount of future time
on both sides. Indices are still mapped through the slice spacing reported by the packet rather
than assumed, so a future change of rate degrades accuracy instead of silently halving horizons.
"""

from __future__ import annotations
import os
import io
import json
import time
import math
from typing import Optional, Sequence
import numpy as np
import torch
try:
    torch.set_flush_denormal(True)
except Exception:
    pass

try:
    from rlbot.flat import AirState, ControllerState, GamePacket, MatchPhase
    from rlbot.managers import Bot
    RLBOT_AVAILABLE = True
except ImportError:
    RLBOT_AVAILABLE = False

    class ControllerState:
        """Offline stand-in for rlbot.flat.ControllerState, used by tests and replay scripts."""
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

    class AirState:
        OnGround = 0
        Jumping = 1
        DoubleJumping = 2
        Dodging = 3
        InAir = 4

    class MatchPhase:
        Inactive = 0
        Countdown = 1
        Kickoff = 2
        Active = 3
        GoalScored = 4
        Replay = 5
        Paused = 6
        Ended = 7

    Bot = object
    GamePacket = object

from agent.models import ActorCritic
from agent.checkpoint import load_policy
from env.observations import DefaultObservationBuilder, OBS_MIRROR_MASK_NP, ACT_MIRROR_MASK_NP
from env.actions import DiscreteActionParser, ContinuousActionParser
from env.physics_engine import (
    CarState, BallState, BoostPad,
    ARENA_EXTENT_X, ARENA_EXTENT_Y, ARENA_HEIGHT_Z,
    CAR_MAX_SPEED, BALL_MAX_SPEED, GOAL_HALF_WIDTH, GOAL_HEIGHT,
    EFFECTIVE_GOAL_HALF_WIDTH, EFFECTIVE_GOAL_HEIGHT, BALL_RADIUS,
    PREDICTION_TICK_RATE, SHOT_THREAT_HORIZON_TICKS, SHOT_THREAT_HORIZON_S,
)

# Identifies this bot to the RLBot server when it is started manually rather than by the
# framework, which would otherwise supply RLBOT_AGENT_ID.
AGENT_ID = "antigravity/sensai"


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


class PredictionIndexer:
    """
    Maps a training-side slice index, in 120 Hz ticks, onto an index into an RLBot prediction.

    The rate is measured from the slices themselves rather than assumed. RLBot v4 published the
    prediction at 60 Hz and v5 publishes it at 120 Hz over the same 6 second span, so a hardcoded
    factor silently halves or doubles every horizon the moment the framework changes. Reading the
    spacing off `game_seconds` keeps the mapping correct under either rate, and keeps the live
    horizons equal to the ones the policy was trained against.
    """

    __slots__ = ("slices", "t0", "dt", "count")

    def __init__(self, slices: Optional[Sequence] = None):
        self.slices = slices if slices else None
        self.count = len(self.slices) if self.slices else 0
        self.t0 = float(self.slices[0].game_seconds) if self.count else 0.0
        # Two slices are enough to measure the spacing; fall back to the training rate when the
        # prediction is too short to measure, or reports a non-increasing time.
        dt = 1.0 / PREDICTION_TICK_RATE
        if self.count >= 2:
            measured = float(self.slices[1].game_seconds) - self.t0
            if measured > 1e-6:
                dt = measured
        self.dt = dt

    def __bool__(self) -> bool:
        return self.count > 0

    def index_at_tick(self, tick: int) -> int:
        """Index of the slice closest to `tick` ticks of 120 Hz time ahead of the prediction start."""
        idx = int(round((tick / PREDICTION_TICK_RATE) / self.dt))
        if idx < 0:
            return 0
        return min(idx, self.count - 1)

    def pos_at_tick(self, tick: int) -> Optional[np.ndarray]:
        if not self.count:
            return None
        loc = self.slices[self.index_at_tick(tick)].physics.location
        return np.array([loc.x, loc.y, loc.z], dtype=np.float32)

    def horizon_ticks(self) -> int:
        """How far the prediction actually reaches, in 120 Hz ticks."""
        if self.count < 1:
            return 0
        return int(round((self.count - 1) * self.dt * PREDICTION_TICK_RATE))




def _air_state_name(state) -> str:
    """Human-readable AirState, for the flip diagnostics.

    A dodge needs the car to have FINISHED its jump. A press that arrives while the state is
    still Jumping is a continuation of the held jump, not a new one, and the game drops it.
    The press log could not show that because it printed height and stance but never the
    state machine, so five ignored presses and five real dodges looked identical.
    """
    for name in ("OnGround", "Jumping", "DoubleJumping", "Dodging", "InAir"):
        if state == getattr(AirState, name, object()):
            return name
    return "state=%s" % (state,)


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


# Ticks after the bot's own ground jump press during which RLBot's lagging suspension raycast
# may still report OnGround while the car is already rising.
GROUND_JUMP_LAG_TICKS = 12
# Jump-probability logging for threshold calibration. On by default while calibrating;
# set SENSAI_LOG_JUMP_PROB=0 to silence.
LOG_JUMP_PROB = os.environ.get("SENSAI_LOG_JUMP_PROB", "1") == "1"


def is_car_grounded(raw_on_ground: bool, recently_jumped: bool, z: float, vz: float) -> bool:
    """RLBot grounded state, corrected for suspension raycast lag after the bot's own jump.

    Only overrides OnGround within a few ticks of a ground jump press, so driving onto the
    ball / another car or suspension rebound after a landing still counts as grounded.
    """
    return bool(raw_on_ground and not (recently_jumped and z >= 22.0 and vz > 50.0))


def handbrake_allowed(act_hb: float, on_ground: bool, z: float) -> bool:
    """Handbrake gate matching ActorCritic's can_slide mask (grounded or within 120 uu of the floor)."""
    return bool(act_hb > 0.0 and (on_ground or z < 120.0))


class MockArena:
    """
    Arena-shaped view of one live packet, for the observation builder and reward helpers.

    The builder was written against the RocketSim arena used in training, so live play has to
    present the same surface. Prediction lookups are answered from the RLBot prediction through a
    PredictionIndexer, in the same 120 Hz tick units the training arena uses.
    """
    def __init__(self, ball, cars, predictor=None, game_boosts=None, boost_pad_mapping=None):
        self.ball = ball
        self.cars = cars
        self._predictor = predictor
        self.boost_pads = BoostPad.create_standard_pads()
        self._sm_pad_indices = np.array([i for i, p in enumerate(self.boost_pads) if not p.is_big], dtype=int)
        self._bg_pad_indices = np.array([i for i, p in enumerate(self.boost_pads) if p.is_big], dtype=int)
        self._small_pad_pos_3d = np.array([self.boost_pads[i].pos for i in self._sm_pad_indices], dtype=np.float32)
        self._big_pad_pos_3d = np.array([self.boost_pads[i].pos for i in self._bg_pad_indices], dtype=np.float32)
        self._all_pad_pos_2d = np.array([p.pos[:2] for p in self.boost_pads], dtype=np.float32)

        if game_boosts is not None and boost_pad_mapping is not None:
            for std_idx, packet_idx in enumerate(boost_pad_mapping):
                if packet_idx < len(game_boosts):
                    self.boost_pads[std_idx].is_active = bool(game_boosts[packet_idx].is_active)
                    self.boost_pads[std_idx].cooldown_timer = float(getattr(game_boosts[packet_idx], "timer", 0.0))

        self._small_pad_active = np.array([self.boost_pads[i].is_active for i in self._sm_pad_indices], dtype=bool)
        self._big_pad_active = np.array([self.boost_pads[i].is_active for i in self._bg_pad_indices], dtype=bool)
        self._all_pad_active = np.array([p.is_active for p in self.boost_pads], dtype=bool)

    def get_predicted_ball_pos(self, slice_idx: int = 60) -> Optional[np.ndarray]:
        """
        Position of the ball `slice_idx` ticks of 120 Hz time from now.

        Returns None when no prediction is available, which lets the observation
        builder fall back to its own ballistic extrapolation. Returning the live ball
        position instead would tell the policy the ball is about to stand still.
        """
        if self._predictor is None:
            return None
        return self._predictor.pos_at_tick(slice_idx)

    def get_shot_threat(self, team: int):
        defending_goal_y = -ARENA_EXTENT_Y if team == 0 else ARENA_EXTENT_Y
        ball_vy = self.ball.vel[1]
        is_moving_to_net = (ball_vy < -100.0) if team == 0 else (ball_vy > 100.0)
        if not is_moving_to_net:
            return False, 0.0, 0.0

        if self._predictor:
            # Scan the same amount of future time the training scan covers, and decay
            # intensity over the same ramp. Both are expressed in seconds here, so the
            # publishing rate of the prediction cannot change what an intensity means.
            pred = self._predictor
            last = pred.index_at_tick(SHOT_THREAT_HORIZON_TICKS)
            for i in range(last + 1):
                loc = pred.slices[i].physics.location
                if (team == 0 and loc.y <= -5120.0) or (team == 1 and loc.y >= 5120.0):
                    is_clean_entry = bool(abs(loc.x) <= EFFECTIVE_GOAL_HALF_WIDTH and BALL_RADIUS <= loc.z <= EFFECTIVE_GOAL_HEIGHT)
                    is_grazing_entry = bool(not is_clean_entry and abs(loc.x) <= GOAL_HALF_WIDTH and BALL_RADIUS <= loc.z <= GOAL_HEIGHT)
                    if is_clean_entry or is_grazing_entry:
                        t_ahead = i * pred.dt
                        raw_intensity = max(0.1, 1.0 - (t_ahead / SHOT_THREAT_HORIZON_S))
                        threat_intensity = raw_intensity if is_clean_entry else (raw_intensity * 0.45)
                        entry_z_norm = min(1.0, max(0.0, loc.z / GOAL_HEIGHT))
                        return True, threat_intensity, entry_z_norm

        dy = defending_goal_y - self.ball.pos[1]
        if abs(ball_vy) > 1e-4:
            dt = dy / ball_vy
            if 0.05 < dt < SHOT_THREAT_HORIZON_S:
                pred_x = self.ball.pos[0] + self.ball.vel[0] * dt
                pred_z = self.ball.pos[2] + self.ball.vel[2] * dt + 0.5 * (-650.0) * (dt ** 2)
                is_clean_entry = bool(abs(pred_x) <= EFFECTIVE_GOAL_HALF_WIDTH and BALL_RADIUS <= pred_z <= EFFECTIVE_GOAL_HEIGHT)
                is_grazing_entry = bool(not is_clean_entry and abs(pred_x) <= GOAL_HALF_WIDTH and BALL_RADIUS <= pred_z <= GOAL_HEIGHT)
                if is_clean_entry or is_grazing_entry:
                    raw_intensity = max(0.1, 1.0 - (dt / SHOT_THREAT_HORIZON_S))
                    threat_intensity = raw_intensity if is_clean_entry else (raw_intensity * 0.45)
                    entry_z_norm = min(1.0, max(0.0, pred_z / GOAL_HEIGHT))
                    return True, threat_intensity, entry_z_norm

        return False, 0.0, 0.0


class SenseiRLBot(Bot):
    def __init__(self, name: str = "SensAI", team: int = 0, index: int = 0, agent_id: str = AGENT_ID):
        # The v5 base class only wires up sockets and handlers here; name, team and index arrive
        # later over the connection and overwrite these defaults. They are still accepted as
        # arguments so tests and the replay scripts can build a bot without a running match.
        if RLBOT_AVAILABLE:
            super().__init__(agent_id)
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
        # RocketSim publishes air_time and flip_time on its car state; the RLBot packet has
        # no equivalent, so they are integrated here from air_state transitions. Both reset
        # on ground contact, matching the training-side semantics exactly (verified against
        # RocketSim: neither is ever non-zero while on_ground).
        self._air_timers: dict[int, float] = {}
        self._flip_timers: dict[int, float] = {}
        self._flip_expired: dict[int, bool] = {}
        # Was the car on the ground when the current policy step began? See the dodge gate in
        # the substep sequencer: a dodge fired inside the liftoff step reads its direction from
        # an action chosen while grounded, where those channels are masked to zero.
        #
        # Defaults False because before the first step there is no step to guard. The boundary
        # below latches the real value on the very first tick (prev_action is None forces it),
        # so this value is unreachable in normal operation; it only shows up in harnesses that
        # set prev_action directly, and there the conservative choice would wrongly veto a
        # legitimate airborne dodge.
        self._step_began_grounded = False

        # Flip-press outcome tracking. The press log records an intent, not a result: the game
        # is free to drop a jump press, and when it does the line looks exactly like a press
        # that worked. These carry the press forward so the NEXT few ticks can say which it was.
        self._ground_jump_tick = 0         # tick of the last ground jump press, 0 = none
        # Liftoff outcome tracking, mirroring the flip verdicts below: did the ground press
        # actually launch the car (has_jumped / air state left OnGround), and from which surface?
        self._liftoff_press_tick = 0       # press awaiting a verdict, 0 = none
        self._liftoff_press_z = 0.0
        self._liftoff_press_up_z = 1.0     # car up-vector z at the press: ~1 floor, ~0 wall
        self._liftoffs_pressed = 0
        self._liftoffs_launched = 0
        self._liftoffs_ignored = 0
        # Touch logging: RLBot's per-player latest_touch timestamp when the packet carries it,
        # otherwise a ball velocity kick within contact range of the car.
        self._last_touch_seconds = None
        self._prev_ball_vel_log = None
        self._last_own_touch_tick = 0
        self._touches_logged = 0
        self._flip_press_tick = 0          # tick of the press awaiting a verdict, 0 = none
        self._flip_press_state = ""        # air state at the moment of that press
        self._flip_press_z = 0.0
        self._flips_pressed = 0
        self._flips_ignored = 0            # pressed while flip available, flip still available
        self._flips_consumed = 0
        self._diag_air_state = "OnGround"

        # Training-time observation envelope, for the on-screen out-of-distribution check.
        # Two mismatches between training and the live game have already cost real training
        # time here (the appended observation fields defaulting to zero, and the dodge firing
        # inside the liftoff step). This makes a third visible the moment it appears rather
        # than after a week of wondering why the bot plays worse in-game than in the sim.
        self._obs_stats = None
        self._obs_limit = None
        try:
            _stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "data", "obs_stats_train.json")
            with open(_stats_path, encoding="utf-8") as _fh:
                _st = json.load(_fh)
            _mn = np.asarray(_st["min"], dtype=np.float32)
            _mx = np.asarray(_st["max"], dtype=np.float32)
            self._obs_stats = True
            # Generous: twice the largest magnitude training ever produced, plus a constant.
            # Saturation can never reach it; a units error or a shifted feature order will.
            self._obs_limit = np.maximum(np.abs(_mn), np.abs(_mx)) * 2.0 + 0.5
        except Exception:
            self._obs_stats = None
            self._obs_limit = None
        self._diag_value = 0.0
        self._diag_ood = ""
        self._diag_resolution_set = False
        self._diag_render_failed = False
        self.boost_pad_mapping: list[int] | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_builder = DefaultObservationBuilder(symmetric=True)
        self.discrete_parser = DiscreteActionParser()
        self.continuous_actions = False
        self.model: torch.nn.Module | None = None
        self.initialize_agent()
        log_debug(f"[INIT] SenseiRLBot init: name={name}, team={team}, index={index}, device={self.device}, tick_skip={self.tick_skip}")

    def initialize(self):
        """Called by the v5 framework once name, team, index and field info are available."""
        self.build_boost_pad_mapping()
        log_debug(
            f"[INIT] connected: name={self.name}, team={self.team}, index={self.index}, "
            f"pads_mapped={self.boost_pad_mapping is not None}"
        )

    def build_boost_pad_mapping(self):
        """
        Match the field's boost pads onto this project's canonical pad order by position.

        The packet orders pads by y then x, which is not the canonical order the observation
        builder expects, and the order can differ per map.
        """
        if self.boost_pad_mapping is not None:
            return
        field_pads = getattr(getattr(self, "field_info", None), "boost_pads", None)
        if not field_pads:
            return
        mapping = []
        for std_pad in BoostPad.create_standard_pads():
            best_idx = 0
            min_dist = float("inf")
            for b_i, pad in enumerate(field_pads):
                loc = pad.location
                d = math.hypot(loc.x - std_pad.pos[0], loc.y - std_pad.pos[1])
                if d < min_dist:
                    min_dist = d
                    best_idx = b_i
            mapping.append(best_idx)
        self.boost_pad_mapping = mapping

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
        if not ckpt_path:
            # Playing an untrained network looks like a badly behaved bot rather than a missing
            # file, so refuse outright.
            msg = "[SensAI] No checkpoint found in checkpoints/; refusing to play with an untrained network."
            log_debug(f"[INIT_ERROR] {msg}")
            raise RuntimeError(msg)
        try:
            model, ckpt = load_policy(ckpt_path, device=str(self.device))
        except Exception as e:
            if self.model is not None:
                # Mid-game hot reload racing a trainer save: keep playing the policy already loaded.
                log_debug(f"[RELOAD_ERROR] [SensAI] Keeping current model; could not reload {ckpt_path}: {e}")
                return
            msg = f"[SensAI] Could not load {ckpt_path}: {e}"
            log_debug(f"[INIT_ERROR] {msg}")
            raise RuntimeError(msg) from e

        self.model = model
        self.continuous_actions = bool(model.continuous_actions)
        self.loaded_ckpt_mtime = os.path.getmtime(ckpt_path) if os.path.exists(ckpt_path) else 0.0
        msg = (f"[SensAI] Successfully loaded in-game model from {ckpt_path} (iteration {ckpt.get('iteration', '?')}, "
               f"Mode: {'Continuous' if self.continuous_actions else f'Discrete ({model.act_dim} actions)'}, "
               f"ObsDim: {model.obs_dim}, LayerNorm: {model.use_layer_norm})")
        print(msg)
        log_debug(f"[INIT] {msg}")

    def _update_air_flip_timers(self, packet) -> None:
        """Integrate per-car air and flip timers, which the RLBot packet does not carry.

        Called once per tick for every player. A car on the ground has both at zero, so the
        timers self-correct after any tick this bot skipped (goal replays, pauses) as soon
        as the cars respawn on the floor.
        """
        dt = 1.0 / 120.0
        for i, p in enumerate(packet.players):
            state = getattr(p, "air_state", AirState.OnGround)
            if state == AirState.OnGround:
                self._air_timers[i] = 0.0
                self._flip_timers[i] = 0.0
                self._flip_expired[i] = False
                continue
            self._air_timers[i] = self._air_timers.get(i, 0.0) + dt
            # flip_time starts when the dodge starts and keeps running until touchdown,
            # so it outlives the dodge animation itself.
            running = self._flip_timers.get(i, 0.0)
            if state == AirState.Dodging or running > 0.0:
                self._flip_timers[i] = running + dt
            else:
                self._flip_timers[i] = 0.0

            # Flip expiration:
            # 1. Spent flip via double jump or dodge
            # 2. Used first jump to liftoff and airborne duration has exceeded RocketSim / RL 1.25s timeout
            if getattr(p, "has_double_jumped", False) or getattr(p, "has_dodged", False):
                self._flip_expired[i] = True
            elif getattr(p, "has_jumped", False) and self._air_timers[i] >= 1.25:
                self._flip_expired[i] = True

    def _demo_state(self, player) -> tuple:
        """(demoed, seconds until respawn) from whichever field this RLBot build exposes."""
        timeout = float(getattr(player, "demolished_timeout", -1.0) or -1.0)
        if timeout > 0.0:
            return True, timeout
        return bool(getattr(player, "is_demolished", False)), 0.0

    def _diagnose_obs(self, obs) -> str:
        """Flag an observation the policy could not have been trained on.

        Deliberately narrow. The first version compared each feature against the 0.1/99.9
        training percentiles, which sounds reasonable and is not: scored against its own
        training data that rule fires on 19.7% of steps, because bounded quantities sit at
        their limits constantly. Angular velocity pinned at Rocket League's 5.5 rad/s cap and
        a unit-vector component at 1.0 are saturation, not anomalies, and it flashed red at
        both. A monitor that cries wolf one step in five is worse than none.

        Two replacements were measured and rejected:
          - rolling-mean drift against the training mean. A 20-second window is one continuous
            trajectory dominated by game phase, so its mean legitimately sits up to 6 sigma
            from the global mean. Unusable.
          - frozen-feature detection. Fires on 100% of steps, because features 80 and 81 (the
            nearest-pad cooldown timers) are dead in training too, and binary flags such as
            is_supersonic can legitimately hold one value for minutes.

        What survives are invariants that cannot false-positive: a non-finite value is always
        a bug, and a magnitude far beyond anything training produced means the units or the
        feature order are wrong. Neither of the two real train/deploy mismatches found in this
        project would have been caught here -- both were found by reading the code and running
        offline probes -- so treat a clean line as "nothing is corrupt", not "nothing is wrong".
        """
        if obs is None:
            return ""
        o = np.asarray(obs, dtype=np.float32)
        bad = ~np.isfinite(o)
        if bad.any():
            idx = np.nonzero(bad)[0][:3]
            return "NaN/Inf at " + " ".join(str(int(i)) for i in idx)
        if self._obs_limit is None:
            return ""
        n = min(len(o), len(self._obs_limit))
        over = np.abs(o[:n]) > self._obs_limit[:n]
        if not over.any():
            return ""
        idx = np.nonzero(over)[0]
        worst = idx[np.argsort(-np.abs(o[idx]))][:3]
        return "SCALE %d: " % len(idx) + " ".join(
            "%d=%.1f(max%.1f)" % (i, float(o[i]), float(self._obs_limit[i])) for i in worst)

    def _render_diagnostics(self, car_state, ball_state, act) -> None:
        """Draw a compact policy readout in-game.

        Two details of the RLBot v5 renderer matter and the first cost a whole test session:

        1. draw_string_2d takes SCREEN FRACTIONS, not pixels -- 0.1 means a tenth of the
           screen. The first version passed x=12, y=30, which is 1200% across and 3000% down,
           so every line was drawn far off-screen and nothing appeared. set_resolution below
           rescales the axes so the pixel coordinates used here mean what they look like.
        2. The colour argument is a flat.Color, exposed as class attributes on the renderer.
           Passing anything else raises inside the draw call.

        Failures are logged rather than swallowed. The previous version caught everything
        silently, which is why a mistake this basic survived a live test: the overlay simply
        did not appear and left nothing behind to explain why.
        """
        r = getattr(self, "renderer", None)
        if r is None:
            return
        try:
            if not self._diag_resolution_set:
                self._diag_resolution_set = True
                try:
                    r.set_resolution(1920, 1080)
                except Exception as e:
                    log_debug(f"[DIAG] set_resolution failed: {e!r}")
                try:
                    if not r.can_render():
                        log_debug("[DIAG] renderer reports can_render() False -- enable "
                                  "rendering in the match settings or nothing will draw.")
                except Exception:
                    pass

            obs = self.latest_obs
            dist = float(np.linalg.norm(ball_state.pos - car_state.pos))
            local_fwd = float(obs[37]) if obs is not None and len(obs) > 38 else 0.0
            local_right = float(obs[38]) if obs is not None and len(obs) > 38 else 0.0
            lines = [
                "SensAI   V=%+.2f   ball %.0fuu" % (self._diag_value, dist),
                "ball local  fwd %+.2f  right %+.2f" % (local_fwd, local_right),
                "thr %+.2f str %+.2f pit %+.2f yaw %+.2f rol %+.2f"
                % (act[0], act[1], act[2], act[3], act[4]),
                "jmp %d bst %d hnd %d | gnd %d flip %d dodge %d"
                % (act[5] > 0, act[6] > 0, act[7] > 0,
                   bool(car_state.on_ground), bool(car_state.has_flip),
                   bool(getattr(car_state, "is_dodging", False))),
                "boost %3.0f  air %.2fs  flip %.2fs"
                % (car_state.boost, getattr(car_state, "air_timer", 0.0),
                   getattr(car_state, "flip_timer", 0.0)),
                "air state %-13s  flips %d ok / %d ignored / %d pressed"
                % (self._diag_air_state, self._flips_consumed,
                   self._flips_ignored, self._flips_pressed),
            ]
            white = getattr(r, "white", None)
            red = getattr(r, "red", None)

            r.begin_rendering("sensai_diag")
            try:
                y = 40
                for text in lines:
                    r.draw_string_2d(text, 30, y, 1.0, white)
                    y += 24
                if self._diag_ood:
                    r.draw_string_2d(self._diag_ood, 30, y, 1.0, red)
            finally:
                r.end_rendering()
        except Exception as e:
            # Never take the match down, but never hide the reason either.
            if not self._diag_render_failed:
                self._diag_render_failed = True
                log_debug(f"[DIAG] overlay disabled after error: {e!r}")

    def _log_touch(self, my_car, car_state, ball_state, packet) -> None:
        """Log each of our ball touches with the inputs TouchBallReward scores it on."""
        ball_vel = np.asarray(ball_state.vel, dtype=np.float32)
        prev_ball_vel = self._prev_ball_vel_log
        self._prev_ball_vel_log = ball_vel.copy()
        if prev_ball_vel is None:
            return

        touched = False
        touch = getattr(my_car, "latest_touch", None)
        if touch is not None and hasattr(touch, "game_seconds"):
            secs = float(touch.game_seconds)
            if self._last_touch_seconds is not None and secs > self._last_touch_seconds + 1e-4:
                touched = True
            self._last_touch_seconds = secs
        else:
            dist = float(np.linalg.norm(ball_state.pos - car_state.pos))
            touched = bool(dist < 260.0 and float(np.linalg.norm(ball_vel - prev_ball_vel)) > 150.0)
            if touched and self.tick_count - self._last_own_touch_tick <= 4:
                touched = False  # same contact still resolving
        if not touched:
            return

        dt_since = (self.tick_count - self._last_own_touch_tick) / 120.0 if self._last_own_touch_tick else 99.0
        self._last_own_touch_tick = self.tick_count
        self._touches_logged += 1

        target_y = 5120.0 if car_state.team == 0 else -5120.0
        target = np.array([float(np.clip(ball_state.pos[0], -670.0, 670.0)), target_y, 225.0], dtype=np.float32)
        to_goal = target - ball_state.pos
        to_goal /= max(1e-4, float(np.linalg.norm(to_goal)))
        speed = float(np.linalg.norm(ball_vel))
        align = float(np.dot(ball_vel / speed, to_goal)) if speed > 1e-4 else 0.0
        added_goal = float(np.dot(ball_vel - prev_ball_vel, to_goal))
        rel_speed = float(np.linalg.norm(car_state.vel - ball_vel))
        car_to_ball = ball_state.pos - car_state.pos
        car_to_ball_2d = car_to_ball[:2] / max(1e-4, float(np.linalg.norm(car_to_ball[:2])))
        ball_to_goal_2d = to_goal[:2] / max(1e-4, float(np.linalg.norm(to_goal[:2])))
        shot_line = float(np.dot(car_to_ball_2d, ball_to_goal_2d))
        foe = None
        for i, p in enumerate(packet.players):
            if i != self.index and p.team != car_state.team:
                foe = p
                break
        foe_txt = (f"({foe.physics.location.x:.0f}, {foe.physics.location.y:.0f})" if foe is not None else "none")
        log_debug(
            f"[TICK {self.tick_count}] ### TOUCH ### n={self._touches_logged} since_prev={dt_since:.2f}s "
            f"car_z={float(car_state.pos[2]):.0f} air={not bool(car_state.on_ground)} "
            f"car_spd={float(np.linalg.norm(car_state.vel)):.0f} shot_line={shot_line:+.2f} "
            f"ball=({float(ball_state.pos[0]):.0f}, {float(ball_state.pos[1]):.0f}, {float(ball_state.pos[2]):.0f}) "
            f"ball_spd={float(np.linalg.norm(prev_ball_vel)):.0f}->{speed:.0f} goal_align={align:+.2f} "
            f"added_toward_goal={added_goal:+.0f} rel_spd={rel_speed:.0f} opp={foe_txt}"
        )

    def get_output(self, packet: GamePacket) -> ControllerState:
        controller = ControllerState()

        # Guard: check match state (allow kickoff and freeplay play)
        match_phase = getattr(packet.match_info, "match_phase", MatchPhase.Active)
        if match_phase == MatchPhase.Ended:
            return controller
        if match_phase in (MatchPhase.Inactive, MatchPhase.GoalScored, MatchPhase.Replay, MatchPhase.Paused):
            controller.throttle = 1.0
            return controller
        # Replays and some transitions publish no ball at all; nothing to observe.
        if not packet.balls:
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

            if len(packet.players) <= self.index:
                return controller

            # Extract ball
            b_phys = packet.balls[0].physics
            ball_state = BallState(
                pos=np.array([b_phys.location.x, b_phys.location.y, b_phys.location.z], dtype=np.float32),
                vel=np.array([b_phys.velocity.x, b_phys.velocity.y, b_phys.velocity.z], dtype=np.float32),
                ang_vel=np.array([b_phys.angular_velocity.x, b_phys.angular_velocity.y, b_phys.angular_velocity.z], dtype=np.float32)
            )

            # Match and Kickoff State Tracking:
            # Detect a new kickoff from the match phase plus the ball sitting at center
            is_kickoff_pause = match_phase in (MatchPhase.Countdown, MatchPhase.Kickoff)
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
            self._update_air_flip_timers(packet)
            my_car = packet.players[self.index]
            recently_jumped = bool(self._ground_jump_tick
                                   and self.tick_count - self._ground_jump_tick <= GROUND_JUMP_LAG_TICKS)
            is_on_ground = is_car_grounded(my_car.air_state == AirState.OnGround, recently_jumped,
                                           float(my_car.physics.location.z), float(my_car.physics.velocity.z))
            has_jump = is_on_ground or (not my_car.has_jumped)
            # Align with RocketSim training: has_flip is only True when airborne and flip is available
            has_flip = bool((not is_on_ground) and not self._flip_expired.get(self.index, False))
            air_state_name = _air_state_name(my_car.air_state)

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
            # Observation indices 94..98 read these. Left unset they default to zero, which
            # would tell the deployed policy it is never flipping and never supersonic while
            # training saw those true for 23% and 4% of steps respectively.
            my_demoed, my_demo_timer = self._demo_state(my_car)
            car_state.is_dodging = bool(my_car.air_state == AirState.Dodging)
            car_state.flip_timer = float(self._flip_timers.get(self.index, 0.0))
            car_state.air_timer = float(self._air_timers.get(self.index, 0.0))
            car_state.has_double_jumped = bool(my_car.has_double_jumped)
            car_state.is_supersonic = bool(getattr(my_car, "is_supersonic", False))
            car_state.demoed = my_demoed
            car_state.demo_timer = my_demo_timer

            # Extract future ball trajectory from RLBot. v5 publishes 720 slices at 120 Hz covering
            # 6 seconds; the indexer measures the spacing rather than assuming it.
            pred_slices = getattr(getattr(self, "ball_prediction", None), "slices", None)
            predictor = PredictionIndexer(pred_slices)

            # Find opponent
            opponents = []
            for i in range(len(packet.players)):
                if i != self.index:
                    opp_car = packet.players[i]
                    opp_on_ground = opp_car.air_state == AirState.OnGround
                    opp_jump = opp_on_ground or (not opp_car.has_jumped)
                    opp_flip = bool((not opp_on_ground) and not self._flip_expired.get(i, False))
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
                    # Observation indices 103..107 read these off the opponent.
                    opp_demoed, opp_demo_timer = self._demo_state(opp_car)
                    opponents[-1].is_dodging = bool(opp_car.air_state == AirState.Dodging)
                    opponents[-1].flip_timer = float(self._flip_timers.get(i, 0.0))
                    opponents[-1].air_timer = float(self._air_timers.get(i, 0.0))
                    opponents[-1].has_double_jumped = bool(opp_car.has_double_jumped)
                    opponents[-1].is_supersonic = bool(getattr(opp_car, "is_supersonic", False))
                    opponents[-1].demoed = opp_demoed
                    opponents[-1].demo_timer = opp_demo_timer

            self.ticks_since_last_action += 1
            if self.ticks_since_last_action >= self.tick_skip or self.prev_action is None:
                self.ticks_since_last_action = 0
                # Latch the stance the step starts from. The action about to be chosen has its
                # pitch/yaw/roll masked to exactly zero if the car is grounded, so that action
                # carries no dodge direction and must never be allowed to fire one.
                self._step_began_grounded = bool(is_on_ground)

                # Normally built in initialize(); retried here in case field info arrived late.
                if self.boost_pad_mapping is None:
                    try:
                        self.build_boost_pad_mapping()
                    except Exception:
                        pass

                arena = MockArena(
                    ball_state, [car_state] + opponents,
                    predictor=predictor if predictor else None,
                    game_boosts=packet.boost_pads,
                    boost_pad_mapping=self.boost_pad_mapping
                )
                obs = self.obs_builder.build_obs(car_state, arena)
                self.latest_obs = obs

                # Model Inference at 15Hz (ActorCritic evaluates native equivariant bilateral policy)
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action, _, _, value = self.model.get_action_and_value(obs_tensor, deterministic=True)
                    self._diag_value = float(value.reshape(-1)[0])
                    if LOG_JUMP_PROB and self.continuous_actions and getattr(self.model, "last_bin_logits", None) is not None:
                        jump_p = float(torch.sigmoid(self.model.last_bin_logits.reshape(-1)[0]))
                        if jump_p > 0.05:
                            log_debug(
                                f"[JUMP_PROB] tick={self.tick_count} p={jump_p:.3f} gnd={is_on_ground} "
                                f"z={float(car_state.pos[2]):.0f} vz={float(car_state.vel[2]):.0f} "
                                f"flip_avail={has_flip} last_ground_jump={self._ground_jump_tick}"
                            )
                    if self.continuous_actions:
                        act = action.squeeze(0).cpu().numpy()
                    else:
                        act_idx = int(action.squeeze().cpu().item())
                        act = self.discrete_parser.parse_actions(act_idx)
                self.prev_action = act
                self._diag_ood = self._diagnose_obs(obs)
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

            # Ground Jump Gating:
            # 1. Flip Cooldown: Require recovery ticks on wheels after a dodge before jumping again.
            # 2. Low-Speed Hard Turning: Suppress jump when sharply steering at low speeds (<350 uu/s) to prevent turf tumbling.
            is_low_speed_hard_steer = bool(car_speed_total < 350.0 and abs(act[1]) > 0.60)
            if is_on_ground:
                if self.dodge_cooldown > 0 or is_low_speed_hard_steer:
                    want_jump = False
                controller.jump = bool(want_jump and substep_tick <= 3)
            else:
                # Airborne Dodge / Second Jump:
                # RocketSim and Rocket League physics require jump release while airborne before
                # a second jump. Press jump on ticks 2..5 so the wheel-lift check passes.
                #
                # But NEVER inside the step that left the ground. Real Rocket League lifts the
                # wheels about two ticks after the press, so by substep 2 the car is already
                # airborne and this branch would fire -- using the action chosen at substep 0,
                # while the car was still grounded. The policy's pitch/yaw/roll are masked to
                # exactly zero in that state, so the dodge came out with no direction at all:
                # an empty stall that burned the flip. The log signature was a liftoff and a
                # flip two ticks apart, both reading pit=-0.00 yaw=-0.00 rol=-0.00, followed by
                # a genuine directional press on the next step that had no flip left to spend.
                #
                # Training never hit this. RocketSim holds is_on_ground true for six ticks after
                # the press (measured), so its window 2..5 closes before the car is airborne and
                # the flip survives to the next step, where the rotational channels are unmasked.
                # That is why 98.9% of dodges in training are directional (median stick 0.78)
                # while the deployed bot was stalling nearly every ground jump. Deferring the
                # dodge by one policy step reproduces the training behaviour exactly.
                controller.jump = bool(
                    want_jump and has_flip and 2 <= substep_tick <= 5
                    and not self._step_began_grounded
                )

                if controller.jump:
                    # Dodge Deadzone Compensation:
                    # Rocket League and RocketSim require analog stick deflection >= 0.50 to execute a directional flip/dodge.
                    # When an airborne dodge is triggered, scale directional stick deflection past the deadzone threshold
                    # so continuous policy outputs execute genuine forward, backward, or diagonal dodges instead of empty double jumps.
                    # Sub-0.50 stick deflection is preserved as a neutral double jump, allowing pitch-up + vertical climb.
                    stick_mag = math.hypot(controller.pitch, controller.yaw)
                    if stick_mag >= 0.50:
                        scale = max(1.0, 0.90 / stick_mag)
                        controller.pitch = float(np.clip(controller.pitch * scale, -1.0, 1.0))
                        controller.yaw = float(np.clip(controller.yaw * scale, -1.0, 1.0))
                        self.dodge_cooldown = 20  # Recovery after actual dodge/flip only

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
            # 1. Suppress boost when momentum strongly opposes nose direction (fwd_speed < -150 uu/s).
            # 2. Suppress boost on the ground when already at supersonic speed (is_supersonic), preventing boost waste.
            controller.boost = bool(act[6] > 0.0 and fwd_speed > -150.0 and not (is_supersonic and is_on_ground))
            controller.handbrake = handbrake_allowed(float(act[7]), is_on_ground, float(car_state.pos[2]))

            # Draw the readout once per policy step (15 Hz), not once per physics tick.
            if substep_tick == 0:
                self._diag_air_state = air_state_name
                self._render_diagnostics(car_state, ball_state, act)

            self.tick_count += 1

            # Did the last airborne press actually spend the flip? has_flip going False while a
            # press is pending is the only positive evidence that the game accepted it.
            if self._flip_press_tick:
                age = self.tick_count - self._flip_press_tick
                if not has_flip:
                    self._flips_consumed += 1
                    log_debug(
                        f"[TICK {self.tick_count}] +++ FLIP CONSUMED +++ press at tick "
                        f"{self._flip_press_tick} (air={self._flip_press_state}) took effect "
                        f"after {age} ticks. now air={air_state_name}. "
                        f"consumed={self._flips_consumed}/{self._flips_pressed}"
                    )
                    self._flip_press_tick = 0
                elif age > 24 or is_on_ground:
                    self._flips_ignored += 1
                    log_debug(
                        f"[TICK {self.tick_count}] --- FLIP IGNORED --- press at tick "
                        f"{self._flip_press_tick} (air={self._flip_press_state}, "
                        f"z={self._flip_press_z:.0f}) expired after {age} ticks with the flip "
                        f"still available (now air={air_state_name}, gnd={is_on_ground}). "
                        f"ignored={self._flips_ignored}/{self._flips_pressed}"
                    )
                    self._flip_press_tick = 0
            # Liftoff verdict. RLBot's OnGround lags the jump by several ticks, so wait for
            # has_jumped or a non-OnGround air state rather than trusting the stance right away.
            if self._liftoff_press_tick:
                age = self.tick_count - self._liftoff_press_tick
                launched = bool(getattr(my_car, "has_jumped", False)
                                or my_car.air_state != AirState.OnGround)
                if launched or age >= 8:
                    if launched:
                        self._liftoffs_launched += 1
                        tag, tally = "+++ LIFTOFF LAUNCHED +++", f"launched={self._liftoffs_launched}"
                    else:
                        self._liftoffs_ignored += 1
                        tag, tally = "--- LIFTOFF IGNORED ---", f"ignored={self._liftoffs_ignored}"
                    log_debug(
                        f"[TICK {self.tick_count}] {tag} press at tick {self._liftoff_press_tick} "
                        f"(z={self._liftoff_press_z:.0f}, up_z={self._liftoff_press_up_z:+.2f}) after {age} ticks: "
                        f"now air={air_state_name} jumped={bool(getattr(my_car, 'has_jumped', False))} "
                        f"z={float(car_state.pos[2]):.0f} vz={float(car_state.vel[2]):.0f} "
                        f"pos=({float(car_state.pos[0]):.0f}, {float(car_state.pos[1]):.0f}). "
                        f"{tally}/{self._liftoffs_pressed}"
                    )
                    self._liftoff_press_tick = 0
            self._log_touch(my_car, car_state, ball_state, packet)
            ball_pos = ball_state.pos
            is_kickoff = bool(abs(ball_pos[0]) < 50.0 and abs(ball_pos[1]) < 50.0 and float(np.linalg.norm(ball_state.vel)) < 100.0)
            
            # Event logging on jump/dodge trigger.
            #
            # This logs the PRESS, not the outcome. The game can ignore a press -- most often
            # because the flip is already spent and has_dodged has not yet come back through the
            # packet -- so a repeated line is not proof of a repeated flip. It also could not
            # distinguish a wavedash (dodge into the ground, land on wheels, flip refreshes)
            # from genuine spam, because it printed neither height nor stance. Both are now
            # included: a wavedash shows z near the floor with the flip refreshing after a
            # landing, while spam shows presses with has_flip already False.
            if controller.jump and (substep_tick == 0 or (not is_on_ground and substep_tick == 2)):
                action_type = "LIFTOFF JUMP" if is_on_ground else "AIRBORNE FLIP"
                if action_type == "LIFTOFF JUMP":
                    self._ground_jump_tick = self.tick_count
                    self._liftoffs_pressed += 1
                    self._liftoff_press_tick = self.tick_count
                    self._liftoff_press_z = float(car_state.pos[2])
                    self._liftoff_press_up_z = float(car_state.get_up_vector()[2])
                consumed = bool(getattr(my_car, "has_dodged", False) or
                                getattr(my_car, "has_double_jumped", False))
                ball_rel = ball_state.pos - car_state.pos
                log_debug(
                    f"[TICK {self.tick_count}] *** {action_type} PRESSED *** "
                    f"ball_dist={float(np.linalg.norm(ball_rel)):.0f} ball_z={float(ball_state.pos[2]):.0f} "
                    f"ball_rel_z={float(ball_rel[2]):+.0f} "
                    f"pit={controller.pitch:+.2f} yaw={controller.yaw:+.2f} rol={controller.roll:+.2f} "
                    f"z={float(car_state.pos[2]):.0f} gnd={is_on_ground} flip_avail={has_flip} "
                    f"already_dodged={consumed} spd={car_speed_total:.0f} "
                    f"air={air_state_name} sub={substep_tick} jumped={bool(getattr(my_car, 'has_jumped', False))}"
                )
                if action_type == "AIRBORNE FLIP":
                    # A press still pending a verdict when the next one arrives was ignored:
                    # the flip was available before and is available again, so nothing spent it.
                    if self._flip_press_tick and has_flip:
                        self._flips_ignored += 1
                        log_debug(
                            f"[TICK {self.tick_count}] --- FLIP IGNORED --- press at tick "
                            f"{self._flip_press_tick} (air={self._flip_press_state}, "
                            f"z={self._flip_press_z:.0f}) never consumed the flip after "
                            f"{self.tick_count - self._flip_press_tick} ticks. "
                            f"ignored={self._flips_ignored}/{self._flips_pressed}"
                        )
                    self._flips_pressed += 1
                    self._flip_press_tick = self.tick_count
                    self._flip_press_state = air_state_name
                    self._flip_press_z = float(car_state.pos[2])

            if self.tick_count <= 10 or self.tick_count % 120 == 0 or is_kickoff:
                foe = next((c for c in opponents if c.team != car_state.team), None)
                foe_txt = f"({foe.pos[0]:.0f}, {foe.pos[1]:.0f})" if foe is not None else "none"
                log_debug(
                    f"[TICK {self.tick_count}] pos=({car_state.pos[0]:.0f}, {car_state.pos[1]:.0f}) "
                    f"boost={float(car_state.boost):.0f} spd={car_speed_total:.0f} opp={foe_txt} "
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


if __name__ == "__main__":
    if not RLBOT_AVAILABLE:
        import sys
        print(
            f"[SensAI Error] RLBot v5 python interface ('rlbot>=2.0.0') is not available in the current Python runtime:\n"
            f"  Executable: {sys.executable}\n"
            f"  Version: {sys.version}\n"
            f"Please run the bot using the Python environment with rlbot installed, for example:\n"
            f"  C:/Users/coryf/AppData/Local/RLBotGUIX/Python311/python.exe bot.py\n",
            file=sys.stderr
        )
        sys.exit(1)

    # Passing the agent id here lets the bot be started by hand for development; when the
    # framework launches it, RLBOT_AGENT_ID takes precedence.
    SenseiRLBot(agent_id=AGENT_ID).run()
