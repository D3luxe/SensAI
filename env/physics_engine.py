"""
Headless 3D Physics Engine for Rocket League simulation.
Simulates car kinematics, ball aerodynamics, arena boundaries, collisions, boost pads, and scoring.
"""

from __future__ import annotations
import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

try:
    import RocketSim as rsim
    # Initialize RocketSim once
    try:
        rsim.init()
        ROCKETSIM_AVAILABLE = True
    except Exception as e:
        print(f"[RocketSim] Init note: {e}")
        ROCKETSIM_AVAILABLE = False
except ImportError:
    rsim = None
    ROCKETSIM_AVAILABLE = False

# Arena Constants (Unreal Units) - RLGym Standard Game Values
ARENA_EXTENT_X = 4096.0          # SIDE_WALL_X
ARENA_EXTENT_Y = 5120.0          # BACK_WALL_Y
ARENA_HEIGHT_Z = 2044.0          # CEILING_Z
BACK_NET_Y = 6000.0              # BACK_NET_Y
CORNER_OFFSET = 1152.0
CORNER_LIMIT = ARENA_EXTENT_X + ARENA_EXTENT_Y - CORNER_OFFSET  # 8064.0 uu

GOAL_WIDTH = 1785.51
GOAL_HALF_WIDTH = 892.755        # GOAL_CENTER_TO_POST
GOAL_HEIGHT = 642.775            # GOAL_HEIGHT
GOAL_DEPTH = 880.0               # BACK_NET_Y - BACK_WALL_Y

BALL_RADIUS = 91.25              # BALL_RADIUS
BALL_MAX_SPEED = 6000.0          # BALL_MAX_SPEED
BALL_RESTITUTION = 0.6
BALL_DRAG = 0.03
GRAVITY = -650.0                 # GRAVITY (uu/s^2)

CAR_MAX_SPEED = 2300.0           # CAR_MAX_SPEED
CAR_SUPERSONIC_SPEED = 2200.0    # SUPERSONIC_THRESHOLD
CAR_MAX_ANG_VEL = 5.5            # CAR_MAX_ANG_VEL (rad/s)
CAR_BOOST_ACCEL = 991.666
CAR_BOOST_CONSUMPTION = 33.3     # % per second
CAR_DRIVE_ACCEL = 1600.0
CAR_BRAKE_ACCEL = 3500.0
CAR_COAST_ACCEL = 525.0
CAR_JUMP_INITIAL_VEL = 292.0
CAR_JUMP_ACCEL = 1458.33
CAR_MAX_JUMP_TIME = 0.2
CAR_DODGE_IMPULSE = 500.0
CAR_AIR_PITCH_TORQUE = 12.46
CAR_AIR_YAW_TORQUE = 9.11
CAR_AIR_ROLL_TORQUE = 38.34
CAR_AIR_DAMPING = 2.0

# Octane Hitbox Dimensions (RLGym Standard)
CAR_LENGTH = 118.0074
CAR_WIDTH = 84.19941
CAR_HEIGHT = 36.15907


@dataclass
class BoostPad:
    pos: np.ndarray
    is_big: bool
    is_active: bool = True
    cooldown_timer: float = 0.0
    boost_amount: float = 100.0
    respawn_time: float = 10.0

    def __post_init__(self):
        self.boost_amount = 100.0 if self.is_big else 12.0
        self.respawn_time = 10.0 if self.is_big else 4.0

    @classmethod
    def create_standard_pads(cls) -> List[BoostPad]:
        pads = []
        # Big pads (corners & mid-sides)
        big_coords = [
            (3072.0, 4096.0),
            (-3072.0, 4096.0),
            (3584.0, 0.0),
            (-3584.0, 0.0),
            (3072.0, -4096.0),
            (-3072.0, -4096.0),
        ]
        for x, y in big_coords:
            pads.append(cls(pos=np.array([x, y, 73.0], dtype=np.float32), is_big=True))

        # Standard small pads distributed across field
        small_coords = [
            (0.0, -4240.0), (-1792.0, -4184.0), (1792.0, -4184.0),
            (-940.0, -3308.0), (940.0, -3308.0), (0.0, -2816.0),
            (-1792.0, -2484.0), (1792.0, -2484.0), (-3584.0, -2484.0), (3584.0, -2484.0),
            (0.0, -1024.0), (-1024.0, 0.0), (1024.0, 0.0), (0.0, 1024.0),
            (-3584.0, 2484.0), (3584.0, 2484.0), (-1792.0, 2484.0), (1792.0, 2484.0),
            (0.0, 2816.0), (-940.0, 3308.0), (940.0, 3308.0),
            (0.0, 4240.0), (-1792.0, 4184.0), (1792.0, 4184.0),
        ]
        for x, y in small_coords:
            pads.append(cls(pos=np.array([x, y, 73.0], dtype=np.float32), is_big=False))
        return pads


@dataclass
class BallState:
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, BALL_RADIUS], dtype=np.float32))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def reset(self, x: float = 0.0, y: float = 0.0, z: float = BALL_RADIUS):
        self.pos = np.array([x, y, z], dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)


@dataclass
class CarState:
    id: int
    team: int  # 0: Blue (starts Y < 0, defends -Y goal), 1: Orange (starts Y > 0, defends +Y goal)
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, -2000.0, 17.0], dtype=np.float32))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rot: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))  # Pitch, Yaw, Roll (radians)
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    boost: float = 33.3
    on_ground: bool = True
    has_jump: bool = True
    has_flip: bool = True
    is_jumping: bool = False
    jump_timer: float = 0.0
    air_timer: float = 0.0
    is_supersonic: bool = False
    ball_touches: int = 0
    demoed: bool = False
    demo_timer: float = 0.0
    prev_jump: bool = False
    just_dodged: bool = False
    rot_mat: Optional[np.ndarray] = None  # 3x3 orthonormal basis: [forward, right, up]

    def get_forward_vector(self) -> np.ndarray:
        if self.rot_mat is not None:
            return self.rot_mat[0].copy()
        p, y, _ = self.rot
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        return np.array([cp * cy, cp * sy, sp], dtype=np.float32)

    def get_right_vector(self) -> np.ndarray:
        fwd = self.get_forward_vector()
        up = self.get_up_vector()
        return np.cross(fwd, up).astype(np.float32)

    def get_up_vector(self) -> np.ndarray:
        if self.rot_mat is not None:
            return self.rot_mat[2].copy()
        p, y, r = self.rot
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        cr, sr = math.cos(r), math.sin(r)
        return np.array([
            -cy * sp * cr - sy * sr,
            -sy * sp * cr + cy * sr,
            cp * cr
        ], dtype=np.float32)


class RocketSimArena:
    """
    Simulates a full Rocket League arena step by step with car controls and ball physics.
    Utilizes native C++ Bullet Physics via RocketSim when available, with pure-Python fallback.
    """
    def __init__(self, num_players: int = 2, game_mode: str = "1v1"):
        self.num_players = num_players
        self.game_mode = game_mode
        self.ball = BallState()
        self.cars: List[CarState] = []
        self.boost_pads = BoostPad.create_standard_pads()
        self.scored_team: Optional[int] = None
        self.step_count = 0

        self._use_rsim = False
        self._rsim_arena = None
        self._rsim_cars = []

        if ROCKETSIM_AVAILABLE:
            try:
                self._rsim_arena = rsim.Arena(rsim.GameMode.SOCCAR)
                half = self.num_players // 2
                for _ in range(half):
                    self._rsim_cars.append(self._rsim_arena.add_car(rsim.Team.BLUE, rsim.CarConfig(rsim.CarConfig.OCTANE)))
                for _ in range(half):
                    self._rsim_cars.append(self._rsim_arena.add_car(rsim.Team.ORANGE, rsim.CarConfig(rsim.CarConfig.OCTANE)))

                def on_goal_cb(**kwargs):
                    st = kwargs.get("team", kwargs.get("scoring_team"))
                    if st is not None:
                        self.scored_team = 0 if (st == rsim.Team.BLUE or st == 0) else 1
                    else:
                        # Position-based fallback
                        b_pos_y = self._rsim_arena.ball.get_state().pos.y
                        self.scored_team = 0 if b_pos_y > 0 else 1

                def on_touch_cb(**kwargs):
                    car_obj = kwargs.get("car")
                    if car_obj:
                        for c in self.cars:
                            if c.id == car_obj.id - 1:
                                c.ball_touches += 1

                self._rsim_arena.set_goal_score_callback(on_goal_cb)
                self._rsim_arena.set_ball_touch_callback(on_touch_cb)
                self._use_rsim = True
            except Exception:
                self._use_rsim = False

        # Pre-allocated vectorized boost pad coordinates for fast SIMD distance lookups
        self._sm_pad_indices = np.array([i for i, p in enumerate(self.boost_pads) if not p.is_big], dtype=int)
        self._bg_pad_indices = np.array([i for i, p in enumerate(self.boost_pads) if p.is_big], dtype=int)
        self._small_pad_pos_3d = np.array([self.boost_pads[i].pos for i in self._sm_pad_indices], dtype=np.float32)
        self._big_pad_pos_3d = np.array([self.boost_pads[i].pos for i in self._bg_pad_indices], dtype=np.float32)
        self._small_pad_active = np.array([self.boost_pads[i].is_active for i in self._sm_pad_indices], dtype=bool)
        self._big_pad_active = np.array([self.boost_pads[i].is_active for i in self._bg_pad_indices], dtype=bool)

        self._init_cars()

    def _init_cars(self):
        self.cars.clear()
        half_players = self.num_players // 2
        # Team 0 (Blue)
        for i in range(half_players):
            offset = (i - (half_players - 1) / 2) * 500.0
            self.cars.append(CarState(
                id=len(self.cars),
                team=0,
                pos=np.array([offset, -2500.0, 17.0], dtype=np.float32),
                rot=np.array([0.0, math.pi / 2, 0.0], dtype=np.float32),  # Facing +Y
                boost=33.3
            ))
        # Team 1 (Orange)
        for i in range(half_players):
            offset = (i - (half_players - 1) / 2) * 500.0
            self.cars.append(CarState(
                id=len(self.cars),
                team=1,
                pos=np.array([offset, 2500.0, 17.0], dtype=np.float32),
                rot=np.array([0.0, -math.pi / 2, 0.0], dtype=np.float32),  # Facing -Y
                boost=33.3
            ))

    def reset(self, random_kickoff: bool = True):
        self.step_count = 0
        self.scored_team = None

        if self._use_rsim and self._rsim_arena:
            # Native C++ RocketSim Reset
            half_players = self.num_players // 2
            spawn_mode = "kickoff"
            if random_kickoff:
                roll = np.random.rand()
                if roll < 0.75:
                    spawn_mode = "kickoff"
                elif roll < 0.85:
                    spawn_mode = "striking"
                elif roll < 0.95:
                    spawn_mode = "contested"
                else:
                    spawn_mode = "wall_play"

            if spawn_mode == "kickoff":
                self._rsim_arena.reset_kickoff()
            elif spawn_mode == "striking":
                # Ball setup
                bx = float(np.random.uniform(-1500.0, 1500.0))
                by = float(np.random.uniform(-1000.0, 1000.0))
                bz = float(np.random.uniform(BALL_RADIUS, 350.0))
                b_vel_y = float(np.random.uniform(-200.0, 200.0))
                bs = rsim.BallState()
                bs.pos = rsim.Vec(bx, by, bz)
                bs.vel = rsim.Vec(float(np.random.uniform(-200.0, 200.0)), b_vel_y, float(np.random.uniform(0.0, 250.0)))
                self._rsim_arena.ball.set_state(bs)

                # Blue car (attacker)
                for i in range(half_players):
                    offset_dist = float(np.random.uniform(700.0, 1400.0))
                    cs = rsim.CarState()
                    cs.pos = rsim.Vec(bx + float(np.random.uniform(-200.0, 200.0)), max(-4800.0, by - offset_dist), 17.0)
                    cs.vel = rsim.Vec(0.0, float(np.random.uniform(0.0, 300.0)), 0.0)
                    cs.rot_mat = rsim.Angle(yaw=math.pi / 2, pitch=0.0, roll=0.0).as_rot_mat()
                    cs.boost = float(np.random.uniform(33.0, 75.0))
                    self._rsim_cars[i].set_state(cs)

                # Orange car (defender)
                for i in range(half_players):
                    cs = rsim.CarState()
                    cs.pos = rsim.Vec(float(np.random.uniform(-800.0, 800.0)), float(np.random.uniform(3000.0, 4500.0)), 17.0)
                    cs.vel = rsim.Vec(0.0, 0.0, 0.0)
                    cs.rot_mat = rsim.Angle(yaw=-math.pi / 2, pitch=0.0, roll=0.0).as_rot_mat()
                    cs.boost = float(np.random.uniform(33.0, 60.0))
                    self._rsim_cars[half_players + i].set_state(cs)
            elif spawn_mode == "contested":
                # Contested setup
                bx = float(np.random.uniform(-1000.0, 1000.0))
                by = float(np.random.uniform(-600.0, 600.0))
                bs = rsim.BallState()
                bs.pos = rsim.Vec(bx, by, BALL_RADIUS)
                bs.vel = rsim.Vec(0.0, 0.0, 0.0)
                self._rsim_arena.ball.set_state(bs)

                for i in range(half_players):
                    dist = float(np.random.uniform(800.0, 1300.0))
                    cs = rsim.CarState()
                    cs.pos = rsim.Vec(bx, by - dist, 17.0)
                    cs.vel = rsim.Vec(0.0, float(np.random.uniform(0.0, 200.0)), 0.0)
                    cs.rot_mat = rsim.Angle(yaw=math.pi / 2, pitch=0.0, roll=0.0).as_rot_mat()
                    cs.boost = float(np.random.uniform(33.0, 50.0))
                    self._rsim_cars[i].set_state(cs)

                for i in range(half_players):
                    dist = float(np.random.uniform(800.0, 1300.0))
                    cs = rsim.CarState()
                    cs.pos = rsim.Vec(bx, by + dist, 17.0)
                    cs.vel = rsim.Vec(0.0, -float(np.random.uniform(0.0, 200.0)), 0.0)
                    cs.rot_mat = rsim.Angle(yaw=-math.pi / 2, pitch=0.0, roll=0.0).as_rot_mat()
                    cs.boost = float(np.random.uniform(33.0, 50.0))
                    self._rsim_cars[half_players + i].set_state(cs)
            else:
                # Wall & Aerial Play setup (sidewall driving and wall clears)
                side_sign = 1.0 if np.random.rand() > 0.5 else -1.0
                wall_x = side_sign * 3800.0
                ball_y = float(np.random.uniform(-800.0, 800.0))
                ball_z = float(np.random.uniform(350.0, 750.0))

                bs = rsim.BallState()
                bs.pos = rsim.Vec(wall_x, ball_y, ball_z)
                bs.vel = rsim.Vec(0.0, float(np.random.uniform(100.0, 400.0)), float(np.random.uniform(-50.0, 150.0)))
                self._rsim_arena.ball.set_state(bs)

                # Blue car attacking wall
                for i in range(half_players):
                    cs = rsim.CarState()
                    cs.pos = rsim.Vec(side_sign * 4070.0, ball_y - float(np.random.uniform(400.0, 700.0)), max(100.0, ball_z - 150.0))
                    cs.vel = rsim.Vec(0.0, float(np.random.uniform(200.0, 500.0)), 0.0)
                    wall_roll = math.pi / 2 if side_sign > 0 else -math.pi / 2
                    cs.rot_mat = rsim.Angle(yaw=math.pi / 2, pitch=0.0, roll=wall_roll).as_rot_mat()
                    cs.boost = float(np.random.uniform(40.0, 80.0))
                    self._rsim_cars[i].set_state(cs)

                # Orange car defending midfield/net
                for i in range(half_players):
                    cs = rsim.CarState()
                    cs.pos = rsim.Vec(float(np.random.uniform(-600.0, 600.0)), float(np.random.uniform(2500.0, 4000.0)), 17.0)
                    cs.vel = rsim.Vec(0.0, 0.0, 0.0)
                    cs.rot_mat = rsim.Angle(yaw=-math.pi / 2, pitch=0.0, roll=0.0).as_rot_mat()
                    cs.boost = float(np.random.uniform(33.0, 60.0))
                    self._rsim_cars[half_players + i].set_state(cs)

            self._sync_from_rsim()
            return

        # Pure-Python Fallback Reset
        for pad in self.boost_pads:
            pad.is_active = True
            pad.cooldown_timer = 0.0

        kickoff_spawns_team0 = [
            (0.0, -4608.0, math.pi / 2),
            (-256.0, -3840.0, math.pi / 2),
            (256.0, -3840.0, math.pi / 2),
            (-2048.0, -2560.0, math.pi / 4),
            (2048.0, -2560.0, 3 * math.pi / 4),
        ]
        kickoff_spawns_team1 = [
            (0.0, 4608.0, -math.pi / 2),
            (256.0, 3840.0, -math.pi / 2),
            (-256.0, 3840.0, -math.pi / 2),
            (2048.0, 2560.0, -3 * math.pi / 4),
            (-2048.0, 2560.0, -math.pi / 4),
        ]

        half_players = self.num_players // 2
        self.ball.reset()
        # 20% chance for Goalkeeper Aerial Defense Scenario during training
        if random_kickoff and np.random.random() < 0.20 and half_players == 1:
            defending_team = np.random.randint(2)
            attacking_team = 1 - defending_team
            def_y = -4800.0 if defending_team == 0 else 4800.0
            att_y = 500.0 if defending_team == 0 else -500.0
            def_yaw = math.pi / 2 if defending_team == 0 else -math.pi / 2
            att_yaw = -math.pi / 2 if defending_team == 0 else math.pi / 2
            
            # Defending car in goalmouth
            def_car = self.cars[defending_team]
            def_car.pos = np.array([np.random.uniform(-300.0, 300.0), def_y, 17.0], dtype=np.float32)
            def_car.vel = np.zeros(3, dtype=np.float32)
            def_car.rot = np.array([0.0, def_yaw, 0.0], dtype=np.float32)
            def_car.boost = float(np.random.uniform(40.0, 80.0))
            def_car.on_ground = True
            def_car.has_jump = True
            def_car.has_flip = True
            def_car.ball_touches = 0

            # Attacking car trailing the shot
            att_car = self.cars[attacking_team]
            att_car.pos = np.array([np.random.uniform(-600.0, 600.0), att_y, 17.0], dtype=np.float32)
            att_car.vel = np.array([0.0, (-600.0 if defending_team == 0 else 600.0), 0.0], dtype=np.float32)
            att_car.rot = np.array([0.0, att_yaw, 0.0], dtype=np.float32)
            att_car.boost = float(np.random.uniform(33.0, 60.0))
            att_car.on_ground = True
            att_car.has_jump = True
            att_car.has_flip = True
            att_car.ball_touches = 0

            # Ball floating / arcing towards defending goal
            ball_y = np.random.uniform(-2500.0, -1500.0) if defending_team == 0 else np.random.uniform(1500.0, 2500.0)
            ball_vy = np.random.uniform(-1400.0, -900.0) if defending_team == 0 else np.random.uniform(900.0, 1400.0)
            self.ball.pos = np.array([np.random.uniform(-400.0, 400.0), ball_y, np.random.uniform(250.0, 500.0)], dtype=np.float32)
            self.ball.vel = np.array([np.random.uniform(-200.0, 200.0), ball_vy, np.random.uniform(300.0, 650.0)], dtype=np.float32)
            self.ball.ang_vel = np.zeros(3, dtype=np.float32)

            if self._use_rsim and self._rsim_arena:
                b_s = self._rsim_arena.ball.get_state()
                b_s.pos = rsim.Vec(float(self.ball.pos[0]), float(self.ball.pos[1]), float(self.ball.pos[2]))
                b_s.vel = rsim.Vec(float(self.ball.vel[0]), float(self.ball.vel[1]), float(self.ball.vel[2]))
                self._rsim_arena.ball.set_state(b_s)

                for c_i, c_obj in enumerate(self.cars):
                    rc = self._rsim_cars[c_i]
                    rc_s = rc.get_state()
                    rc_s.pos = rsim.Vec(float(c_obj.pos[0]), float(c_obj.pos[1]), float(c_obj.pos[2]))
                    rc_s.vel = rsim.Vec(float(c_obj.vel[0]), float(c_obj.vel[1]), float(c_obj.vel[2]))
                    rc_s.rot_mat = rsim.RotMat.from_angles(float(c_obj.rot[0]), float(c_obj.rot[1]), float(c_obj.rot[2]))
                    rc_s.boost = float(c_obj.boost)
                    rc.set_state(rc_s)
            return

        for i in range(half_players):
            idx = (spawn_idx + i) % len(kickoff_spawns_team0)
            x, y, yaw = kickoff_spawns_team0[idx]
            self.cars[i].pos = np.array([x, y, 17.0], dtype=np.float32)
            self.cars[i].vel = np.zeros(3, dtype=np.float32)
            self.cars[i].rot = np.array([0.0, yaw, 0.0], dtype=np.float32)
            self.cars[i].ang_vel = np.zeros(3, dtype=np.float32)
            self.cars[i].boost = 33.3
            self.cars[i].on_ground = True
            self.cars[i].has_jump = True
            self.cars[i].has_flip = True
            self.cars[i].ball_touches = 0

        for i in range(half_players):
            car_idx = half_players + i
            idx = (spawn_idx + i) % len(kickoff_spawns_team1)
            x, y, yaw = kickoff_spawns_team1[idx]
            self.cars[car_idx].pos = np.array([x, y, 17.0], dtype=np.float32)
            self.cars[car_idx].vel = np.zeros(3, dtype=np.float32)
            self.cars[car_idx].rot = np.array([0.0, yaw, 0.0], dtype=np.float32)
            self.cars[car_idx].ang_vel = np.zeros(3, dtype=np.float32)
            self.cars[car_idx].boost = 33.3
            self.cars[car_idx].on_ground = True
            self.cars[car_idx].has_jump = True
            self.cars[car_idx].has_flip = True
            self.cars[car_idx].ball_touches = 0

    def _sync_from_rsim(self):
        """Synchronizes Python CarState and BallState from C++ RocketSim."""
        b_state = self._rsim_arena.ball.get_state()
        self.ball.pos = b_state.pos.as_numpy().astype(np.float32)
        self.ball.vel = b_state.vel.as_numpy().astype(np.float32)
        self.ball.ang_vel = b_state.ang_vel.as_numpy().astype(np.float32)

        for i, r_car in enumerate(self._rsim_cars):
            c_state = r_car.get_state()
            ang = c_state.rot_mat.as_angle()
            car = self.cars[i]
            car.pos = c_state.pos.as_numpy().astype(np.float32)
            car.vel = c_state.vel.as_numpy().astype(np.float32)
            car.rot = np.array([ang.pitch, ang.yaw, ang.roll], dtype=np.float32)
            car.rot_mat = c_state.rot_mat.as_numpy().astype(np.float32)
            car.ang_vel = c_state.ang_vel.as_numpy().astype(np.float32)
            car.boost = float(c_state.boost)
            car.on_ground = bool(c_state.is_on_ground)
            car.has_jump = bool(not c_state.has_jumped or c_state.is_on_ground)
            car.has_flip = bool(not c_state.has_flipped and not c_state.is_on_ground)
            car.just_dodged = bool(c_state.is_flipping or c_state.has_flipped)
            car.is_supersonic = bool(c_state.is_supersonic)
            car.demoed = bool(c_state.is_demoed)

        # Synchronize boost pads
        r_pads = self._rsim_arena.get_boost_pads()
        for i, pad in enumerate(self.boost_pads):
            if i < len(r_pads):
                p_state = r_pads[i].get_state()
                pad.is_active = bool(p_state.is_active)
                pad.cooldown_timer = float(p_state.cooldown)

        if hasattr(self, "_sm_pad_indices"):
            for idx, pad_i in enumerate(self._sm_pad_indices):
                self._small_pad_active[idx] = self.boost_pads[pad_i].is_active
            for idx, pad_i in enumerate(self._bg_pad_indices):
                self._big_pad_active[idx] = self.boost_pads[pad_i].is_active

    def step(self, actions: List[np.ndarray], dt: float = 1.0 / 15.0) -> Tuple[bool, Optional[int]]:
        """
        Step simulation by dt seconds.
        Uses C++ RocketSim native Bullet Physics when available.
        """
        self.step_count += 1

        if self._use_rsim and self._rsim_arena:
            self.scored_team = None
            total_ticks = max(1, int(round(dt * 120.0)))

            dodge_flags = []
            for i, r_car in enumerate(self._rsim_cars):
                act = actions[i] if i < len(actions) else np.zeros(8, dtype=np.float32)
                is_on_ground = bool(r_car.get_state().is_on_ground)
                is_dodge = bool(act[5] > 0.0 and (abs(act[2]) > 0.1 or abs(act[3]) > 0.1 or abs(act[4]) > 0.1) and is_on_ground)
                dodge_flags.append(is_dodge)

            if any(dodge_flags) and total_ticks >= 8:
                # 3-phase 8-tick substep execution for authentic RocketSim C++ dodge physics:
                # Phase 1: Jump initiation (4 ticks to clear suspension)
                for i, r_car in enumerate(self._rsim_cars):
                    act = actions[i] if i < len(actions) else np.zeros(8, dtype=np.float32)
                    r_car.set_controls(rsim.CarControls(
                        throttle=float(act[0]), steer=-float(act[1]), pitch=-float(act[2]),
                        yaw=-float(act[3]), roll=float(act[4]), jump=bool(act[5] > 0.5),
                        boost=bool(act[6] > 0.3), handbrake=bool(act[7] > 0.5)
                    ))
                self._rsim_arena.step(4)

                # Phase 2: Jump release gate (2 ticks)
                for i, r_car in enumerate(self._rsim_cars):
                    act = actions[i] if i < len(actions) else np.zeros(8, dtype=np.float32)
                    r_car.set_controls(rsim.CarControls(
                        throttle=float(act[0]), steer=-float(act[1]), pitch=-float(act[2]),
                        yaw=-float(act[3]), roll=float(act[4]), jump=False if dodge_flags[i] else bool(act[5] > 0.5),
                        boost=bool(act[6] > 0.3), handbrake=bool(act[7] > 0.5)
                    ))
                self._rsim_arena.step(2)

                # Phase 3: Dodge flip trigger (remaining ticks)
                for i, r_car in enumerate(self._rsim_cars):
                    act = actions[i] if i < len(actions) else np.zeros(8, dtype=np.float32)
                    r_car.set_controls(rsim.CarControls(
                        throttle=float(act[0]), steer=-float(act[1]), pitch=-float(act[2]),
                        yaw=-float(act[3]), roll=float(act[4]), jump=bool(act[5] > 0.5),
                        boost=bool(act[6] > 0.3), handbrake=bool(act[7] > 0.5)
                    ))
                self._rsim_arena.step(total_ticks - 6)
            else:
                for i, r_car in enumerate(self._rsim_cars):
                    act = actions[i] if i < len(actions) else np.zeros(8, dtype=np.float32)
                    r_car.set_controls(rsim.CarControls(
                        throttle=float(act[0]), steer=-float(act[1]), pitch=-float(act[2]),
                        yaw=-float(act[3]), roll=float(act[4]), jump=bool(act[5] > 0.5),
                        boost=bool(act[6] > 0.3), handbrake=bool(act[7] > 0.5)
                    ))
                self._rsim_arena.step(total_ticks)

            self._sync_from_rsim()
            self._cached_pred_step = -1  # Invalidate cache on new physics step
            return (self.scored_team is not None), self.scored_team

    def get_predicted_ball_pos(self, slice_idx: int = 60) -> np.ndarray:
        """Returns cached 0.5s future ball position (calculated once per step per arena)."""
        if getattr(self, "_cached_pred_step", -1) == self.step_count and getattr(self, "_cached_pred_slice", None) is not None:
            return self._cached_pred_slice

        pred_pos = None
        if self._use_rsim and self._rsim_arena is not None:
            try:
                preds = self._rsim_arena.get_ball_prediction()
                if preds and len(preds) > slice_idx:
                    pred_pos = preds[slice_idx].pos.as_numpy().astype(np.float32)
            except Exception:
                pass
        elif hasattr(self, "ball_prediction_slice") and self.ball_prediction_slice is not None:
            pred_pos = self.ball_prediction_slice

        if pred_pos is None:
            dt = slice_idx / 120.0
            px = self.ball.pos[0] + self.ball.vel[0] * dt
            py = self.ball.pos[1] + self.ball.vel[1] * dt
            pz = max(93.0, self.ball.pos[2] + self.ball.vel[2] * dt + 0.5 * (-650.0) * (dt ** 2))
            if abs(px) > 4000.0:
                px = np.sign(px) * (4000.0 - (abs(px) - 4000.0) * 0.6)
            if abs(py) > 5000.0:
                py = np.sign(py) * (5000.0 - (abs(py) - 5000.0) * 0.6)
            pred_pos = np.array([px, py, pz], dtype=np.float32)

        self._cached_pred_step = self.step_count
        self._cached_pred_slice = pred_pos
        return pred_pos

    def get_shot_threat(self, team: int) -> Tuple[bool, float, float]:
        """
        Returns (is_threat, threat_intensity [0.0-1.0], entry_z_norm [0.0-1.0]).
        Calculates exact goal threat on the defending net from RocketSim C++ prediction or raycast.
        """
        defending_goal_y = -ARENA_EXTENT_Y if team == 0 else ARENA_EXTENT_Y
        ball_vy = self.ball.vel[1]
        is_moving_to_net = (ball_vy < -100.0) if team == 0 else (ball_vy > 100.0)
        if not is_moving_to_net:
            return False, 0.0, 0.0

        # Check RocketSim native prediction
        if self._use_rsim and self._rsim_arena is not None:
            try:
                preds = self._rsim_arena.get_ball_prediction()
                if preds:
                    for i, s in enumerate(preds):
                        pos = s.pos
                        if (team == 0 and pos.y <= -5120.0) or (team == 1 and pos.y >= 5120.0):
                            if abs(pos.x) < 950.0 and 0.0 < pos.z < 680.0:
                                threat_intensity = max(0.1, 1.0 - (i / 120.0))
                                entry_z_norm = min(1.0, max(0.0, pos.z / GOAL_HEIGHT))
                                return True, threat_intensity, entry_z_norm
            except Exception:
                pass

        # Ballistic raycast fallback for long-range floating lobs
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

        # Pure Python Fallback Step
        substeps = max(1, int(round(dt * 120.0)))
        sub_dt = dt / substeps

        for _ in range(substeps):
            goal, team = self._step_internal(actions, sub_dt)
            if goal:
                return True, team
        return False, None

    def _step_internal(self, actions: List[np.ndarray], dt: float) -> Tuple[bool, Optional[int]]:
        self.scored_team = None

        # 1. Update boost pad cooldowns
        for pad in self.boost_pads:
            if not pad.is_active:
                pad.cooldown_timer -= dt
                if pad.cooldown_timer <= 0.0:
                    pad.is_active = True
                    pad.cooldown_timer = 0.0

        # 2. Update each car
        for i, car in enumerate(self.cars):
            if car.demoed:
                car.demo_timer -= dt
                if car.demo_timer <= 0:
                    car.demoed = False
                    car.pos = np.array([-2000.0 if car.team == 0 else 2000.0, -4500.0 if car.team == 0 else 4500.0, 17.0], dtype=np.float32)
                    car.vel = np.zeros(3, dtype=np.float32)
                continue

            act = actions[i] if i < len(actions) else np.zeros(8, dtype=np.float32)
            throttle = float(np.clip(act[0], -1.0, 1.0))
            steer = float(np.clip(act[1], -1.0, 1.0))
            pitch = float(np.clip(act[2], -1.0, 1.0))
            yaw = float(np.clip(act[3], -1.0, 1.0))
            roll = float(np.clip(act[4], -1.0, 1.0))
            jump = bool(act[5] > 0.0)
            boost = bool(act[6] > 0.0 and car.boost > 0.0)
            handbrake = bool(act[7] > 0.5)

            # Boost consumption & acceleration
            if boost and car.boost > 0:
                car.boost = max(0.0, car.boost - CAR_BOOST_CONSUMPTION * dt)
                boost_fwd = car.get_forward_vector()
                car.vel += boost_fwd * (CAR_BOOST_ACCEL * dt)

            # Check supersonic
            speed = float(np.linalg.norm(car.vel))
            car.is_supersonic = (speed >= CAR_SUPERSONIC_SPEED)

            # Reset per-step action events
            car.just_dodged = False

            if car.on_ground:
                car.has_jump = True
                car.has_flip = True
                car.air_timer = 0.0

                # Ground driving
                fwd = car.get_forward_vector()
                fwd_speed = float(np.dot(car.vel, fwd))

                if throttle > 0:
                    drive_force = CAR_DRIVE_ACCEL * (1.0 - speed / CAR_MAX_SPEED) * throttle
                    car.vel += fwd * (drive_force * dt)
                elif throttle < 0:
                    if fwd_speed > 100.0:
                        car.vel -= fwd * (CAR_BRAKE_ACCEL * dt)
                    else:
                        car.vel += fwd * (CAR_DRIVE_ACCEL * 0.5 * throttle * dt)
                else:
                    car.vel *= max(0.0, 1.0 - 0.5 * dt)

                turn_rate = (3.5 - 2.0 * (speed / CAR_MAX_SPEED)) * steer
                if handbrake:
                    turn_rate *= 1.8
                    right = car.get_right_vector()
                    side_vel = np.dot(car.vel, right)
                    car.vel -= right * (side_vel * 0.4)
                car.rot[1] -= turn_rate * dt

                if jump and car.has_jump:
                    car.on_ground = False
                    car.is_jumping = True
                    car.jump_timer = CAR_MAX_JUMP_TIME
                    car.vel[2] += CAR_JUMP_INITIAL_VEL
                    car.has_jump = False
            else:
                car.air_timer += dt
                car.vel[2] += GRAVITY * dt

                if jump and car.is_jumping and car.jump_timer > 0:
                    car.vel[2] += CAR_JUMP_ACCEL * dt
                    car.jump_timer -= dt
                else:
                    car.is_jumping = False

                car.rot[0] += pitch * CAR_AIR_PITCH_TORQUE * dt
                car.rot[1] -= yaw * CAR_AIR_YAW_TORQUE * dt
                car.rot[2] += roll * CAR_AIR_ROLL_TORQUE * dt

                jump_edge = jump and not car.prev_jump
                if jump_edge and car.has_flip and not car.is_jumping and car.air_timer < 1.25:
                    fwd_input = throttle if abs(throttle) >= abs(pitch) else -pitch
                    side_input = steer if abs(steer) >= abs(yaw) else yaw
                    if abs(fwd_input) > 0.2 or abs(side_input) > 0.2:
                        car.has_flip = False
                        car.just_dodged = True
                        fwd = car.get_forward_vector()
                        right = car.get_right_vector()
                        dodge_dir = fwd * fwd_input + right * side_input
                        dodge_norm = np.linalg.norm(dodge_dir)
                        if dodge_norm > 1e-3:
                            dodge_dir /= dodge_norm
                            car.vel += dodge_dir * CAR_DODGE_IMPULSE
                            car.rot[0] += fwd_input * 1.5
                            car.rot[2] += side_input * 1.5

            car.prev_jump = jump

            curr_speed = float(np.linalg.norm(car.vel))
            if curr_speed > CAR_MAX_SPEED:
                car.vel = (car.vel / curr_speed) * CAR_MAX_SPEED

            car.pos += car.vel * dt

            if car.pos[2] <= 17.0:
                car.pos[2] = 17.0
                car.vel[2] = max(0.0, car.vel[2])
                car.on_ground = True
                car.rot[0] = 0.0
                car.rot[2] = 0.0

            if car.pos[2] >= ARENA_HEIGHT_Z - 17.0:
                car.pos[2] = ARENA_HEIGHT_Z - 17.0
                car.vel[2] = min(0.0, car.vel[2])

            car_corner = abs(car.pos[0]) + abs(car.pos[1])
            if car_corner > CORNER_LIMIT - 60.0:
                sign_x = math.copysign(1.0, car.pos[0])
                sign_y = math.copysign(1.0, car.pos[1])
                c_norm = np.array([-sign_x * 0.70710678, -sign_y * 0.70710678, 0.0], dtype=np.float32)
                pen = car_corner - (CORNER_LIMIT - 60.0)
                car.pos += c_norm * pen
                vel_in_norm = float(np.dot(car.vel, c_norm))
                if vel_in_norm < 0:
                    car.vel -= 1.2 * vel_in_norm * c_norm
            elif abs(car.pos[0]) > ARENA_EXTENT_X - 60.0:
                car.pos[0] = math.copysign(ARENA_EXTENT_X - 60.0, car.pos[0])
                car.vel[0] *= -0.2
            elif abs(car.pos[1]) > ARENA_EXTENT_Y - 60.0:
                if not (abs(car.pos[0]) < GOAL_HALF_WIDTH and car.pos[2] < GOAL_HEIGHT):
                    car.pos[1] = math.copysign(ARENA_EXTENT_Y - 60.0, car.pos[1])
                    car.vel[1] *= -0.2

            for pad in self.boost_pads:
                if pad.is_active and np.linalg.norm(car.pos - pad.pos) < 160.0:
                    if car.boost < 100.0:
                        car.boost = min(100.0, car.boost + pad.boost_amount)
                        pad.is_active = False
                        pad.cooldown_timer = pad.respawn_time

        # 3. Update Ball Physics
        self.ball.vel[2] += GRAVITY * dt
        self.ball.vel *= (1.0 - BALL_DRAG * dt)
        ball_speed = float(np.linalg.norm(self.ball.vel))
        if ball_speed > BALL_MAX_SPEED:
            self.ball.vel = (self.ball.vel / ball_speed) * BALL_MAX_SPEED

        self.ball.pos += self.ball.vel * dt

        if self.ball.pos[2] <= BALL_RADIUS:
            self.ball.pos[2] = BALL_RADIUS
            self.ball.vel[2] = -self.ball.vel[2] * BALL_RESTITUTION
        elif self.ball.pos[2] >= ARENA_HEIGHT_Z - BALL_RADIUS:
            self.ball.pos[2] = ARENA_HEIGHT_Z - BALL_RADIUS
            self.ball.vel[2] = -self.ball.vel[2] * BALL_RESTITUTION

        corner_val = abs(self.ball.pos[0]) + abs(self.ball.pos[1])
        if corner_val >= CORNER_LIMIT - BALL_RADIUS:
            sign_x = math.copysign(1.0, self.ball.pos[0])
            sign_y = math.copysign(1.0, self.ball.pos[1])
            c_norm = np.array([-sign_x * 0.70710678, -sign_y * 0.70710678, 0.0], dtype=np.float32)
            pen = corner_val - (CORNER_LIMIT - BALL_RADIUS)
            self.ball.pos += c_norm * pen
            vel_in_norm = float(np.dot(self.ball.vel, c_norm))
            if vel_in_norm < 0:
                self.ball.vel -= (1.0 + BALL_RESTITUTION) * vel_in_norm * c_norm

        elif abs(self.ball.pos[0]) >= ARENA_EXTENT_X - BALL_RADIUS:
            self.ball.pos[0] = math.copysign(ARENA_EXTENT_X - BALL_RADIUS, self.ball.pos[0])
            self.ball.vel[0] = -self.ball.vel[0] * BALL_RESTITUTION

        elif abs(self.ball.pos[1]) >= ARENA_EXTENT_Y - BALL_RADIUS:
            in_goal_x = abs(self.ball.pos[0]) < (GOAL_HALF_WIDTH - BALL_RADIUS * 0.5)
            in_goal_z = self.ball.pos[2] < (GOAL_HEIGHT - BALL_RADIUS * 0.5)

            if in_goal_x and in_goal_z:
                if self.ball.pos[1] > 0:
                    self.scored_team = 0
                    return True, 0
                else:
                    self.scored_team = 1
                    return True, 1
            else:
                self.ball.pos[1] = math.copysign(ARENA_EXTENT_Y - BALL_RADIUS, self.ball.pos[1])
                self.ball.vel[1] = -self.ball.vel[1] * BALL_RESTITUTION

        for car in self.cars:
            if car.demoed:
                continue
            delta = self.ball.pos - car.pos
            dist = float(np.linalg.norm(delta))
            min_dist = BALL_RADIUS + 75.0
            if dist < min_dist and dist > 1e-4:
                car.ball_touches += 1
                normal = delta / dist
                self.ball.pos = car.pos + normal * min_dist
                relative_vel = self.ball.vel - car.vel
                vel_along_normal = float(np.dot(relative_vel, normal))
                if vel_along_normal < 0:
                    restitution_impulse = -(1.0 + BALL_RESTITUTION) * vel_along_normal
                    car_hit_power = 200.0 + float(np.linalg.norm(car.vel)) * 0.4
                    self.ball.vel += normal * (restitution_impulse + car_hit_power)

        return False, None
