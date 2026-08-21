"""
Headless 3D Physics Engine for Rocket League simulation.
Simulates car kinematics, ball aerodynamics, arena boundaries, collisions, boost pads, and scoring.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

# Arena Constants (Unreal Units)
ARENA_EXTENT_X = 4096.0
ARENA_EXTENT_Y = 5120.0
ARENA_HEIGHT_Z = 2048.0
CORNER_OFFSET = 1152.0
CORNER_LIMIT = ARENA_EXTENT_X + ARENA_EXTENT_Y - CORNER_OFFSET  # 8064.0 uu

GOAL_WIDTH = 1785.6
GOAL_HALF_WIDTH = GOAL_WIDTH / 2.0  # 892.8
GOAL_HEIGHT = 642.775
GOAL_DEPTH = 880.0

BALL_RADIUS = 92.75
BALL_MAX_SPEED = 4000.0
BALL_RESTITUTION = 0.6
BALL_DRAG = 0.03
GRAVITY = -650.0  # uu/s^2

CAR_MAX_SPEED = 2300.0
CAR_SUPERSONIC_SPEED = 2200.0
CAR_BOOST_ACCEL = 991.666
CAR_BOOST_CONSUMPTION = 33.3  # % per second
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

CAR_LENGTH = 118.0
CAR_WIDTH = 84.2
CAR_HEIGHT = 36.16


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

    def get_forward_vector(self) -> np.ndarray:
        p, y, r = self.rot
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        return np.array([cy * cp, sy * cp, sp], dtype=np.float32)

    def get_right_vector(self) -> np.ndarray:
        p, y, r = self.rot
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        cr, sr = math.cos(r), math.sin(r)
        return np.array([
            sy * cr + cy * sp * sr,
            -cy * cr + sy * sp * sr,
            -cp * sr
        ], dtype=np.float32)

    def get_up_vector(self) -> np.ndarray:
        p, y, r = self.rot
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        cr, sr = math.cos(r), math.sin(r)
        return np.array([
            -sy * sr + cy * sp * cr,
            cy * sr + sy * sp * cr,
            cp * cr
        ], dtype=np.float32)


class RocketSimArena:
    """
    Simulates a full Rocket League arena step by step with car controls and ball physics.
    """
    def __init__(self, num_players: int = 2, game_mode: str = "1v1"):
        self.num_players = num_players
        self.game_mode = game_mode
        self.ball = BallState()
        self.cars: List[CarState] = []
        self.boost_pads = BoostPad.create_standard_pads()
        self.scored_team: Optional[int] = None
        self.step_count = 0
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
        for pad in self.boost_pads:
            pad.is_active = True
            pad.cooldown_timer = 0.0

        # Kickoff spawns
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

        # Choose State Initialization Mode
        # If random_kickoff is True: 35% Kickoff, 40% Shooting/Striking, 25% Contested 50-50
        spawn_mode = "kickoff"
        if random_kickoff:
            roll = np.random.rand()
            if roll < 0.35:
                spawn_mode = "kickoff"
            elif roll < 0.75:
                spawn_mode = "striking"
            else:
                spawn_mode = "contested"

        if spawn_mode == "striking":
            # Ball in midfield with forward trajectory
            bx = float(np.random.uniform(-1800.0, 1800.0))
            by = float(np.random.uniform(-1500.0, 1500.0))
            bz = float(np.random.uniform(BALL_RADIUS, 350.0))
            b_vel_y = float(np.random.uniform(-400.0, 400.0))
            self.ball.pos = np.array([bx, by, bz], dtype=np.float32)
            self.ball.vel = np.array([float(np.random.uniform(-300.0, 300.0)), b_vel_y, float(np.random.uniform(0.0, 300.0))], dtype=np.float32)
            self.ball.ang_vel = np.zeros(3, dtype=np.float32)

            # Team 0 car behind ball pressing forward
            for i in range(half_players):
                offset_dist = float(np.random.uniform(600.0, 1200.0))
                self.cars[i].pos = np.array([bx + np.random.uniform(-300.0, 300.0), max(-4800.0, by - offset_dist), 17.0], dtype=np.float32)
                self.cars[i].vel = np.array([0.0, float(np.random.uniform(400.0, 1000.0)), 0.0], dtype=np.float32)
                self.cars[i].rot = np.array([0.0, math.pi / 2, 0.0], dtype=np.float32)
                self.cars[i].boost = float(np.random.uniform(50.0, 100.0))
                self.cars[i].on_ground = True
                self.cars[i].has_jump = True
                self.cars[i].has_flip = True
                self.cars[i].is_jumping = False
                self.cars[i].jump_timer = 0.0
                self.cars[i].air_timer = 0.0
                self.cars[i].ball_touches = 0
                self.cars[i].demoed = False

            # Team 1 car in defending position
            for i in range(half_players):
                car_idx = half_players + i
                self.cars[car_idx].pos = np.array([float(np.random.uniform(-1000.0, 1000.0)), float(np.random.uniform(3000.0, 4500.0)), 17.0], dtype=np.float32)
                self.cars[car_idx].vel = np.zeros(3, dtype=np.float32)
                self.cars[car_idx].rot = np.array([0.0, -math.pi / 2, 0.0], dtype=np.float32)
                self.cars[car_idx].boost = float(np.random.uniform(33.0, 75.0))
                self.cars[car_idx].on_ground = True
                self.cars[car_idx].has_jump = True
                self.cars[car_idx].has_flip = True
                self.cars[car_idx].is_jumping = False
                self.cars[car_idx].jump_timer = 0.0
                self.cars[car_idx].air_timer = 0.0
                self.cars[car_idx].ball_touches = 0
                self.cars[car_idx].demoed = False

        elif spawn_mode == "contested":
            # Midfield 50-50 challenge
            bx = float(np.random.uniform(-1200.0, 1200.0))
            by = float(np.random.uniform(-800.0, 800.0))
            self.ball.pos = np.array([bx, by, BALL_RADIUS], dtype=np.float32)
            self.ball.vel = np.zeros(3, dtype=np.float32)
            self.ball.ang_vel = np.zeros(3, dtype=np.float32)

            for i in range(half_players):
                dist = float(np.random.uniform(700.0, 1200.0))
                self.cars[i].pos = np.array([bx, by - dist, 17.0], dtype=np.float32)
                self.cars[i].vel = np.array([0.0, float(np.random.uniform(400.0, 900.0)), 0.0], dtype=np.float32)
                self.cars[i].rot = np.array([0.0, math.pi / 2, 0.0], dtype=np.float32)
                self.cars[i].boost = float(np.random.uniform(40.0, 80.0))
                self.cars[i].on_ground = True
                self.cars[i].has_jump = True
                self.cars[i].has_flip = True
                self.cars[i].is_jumping = False
                self.cars[i].jump_timer = 0.0
                self.cars[i].air_timer = 0.0
                self.cars[i].ball_touches = 0
                self.cars[i].demoed = False

            for i in range(half_players):
                car_idx = half_players + i
                dist = float(np.random.uniform(700.0, 1200.0))
                self.cars[car_idx].pos = np.array([bx, by + dist, 17.0], dtype=np.float32)
                self.cars[car_idx].vel = np.array([0.0, -float(np.random.uniform(400.0, 900.0)), 0.0], dtype=np.float32)
                self.cars[car_idx].rot = np.array([0.0, -math.pi / 2, 0.0], dtype=np.float32)
                self.cars[car_idx].boost = float(np.random.uniform(40.0, 80.0))
                self.cars[car_idx].on_ground = True
                self.cars[car_idx].has_jump = True
                self.cars[car_idx].has_flip = True
                self.cars[car_idx].is_jumping = False
                self.cars[car_idx].jump_timer = 0.0
                self.cars[car_idx].air_timer = 0.0
                self.cars[car_idx].ball_touches = 0
                self.cars[car_idx].demoed = False

        else:
            # Standard Kickoff
            self.ball.reset()
            spawn_idx = np.random.randint(len(kickoff_spawns_team0)) if random_kickoff else 0

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
                self.cars[i].is_jumping = False
                self.cars[i].jump_timer = 0.0
                self.cars[i].air_timer = 0.0
                self.cars[i].ball_touches = 0
                self.cars[i].demoed = False

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
                self.cars[car_idx].is_jumping = False
                self.cars[car_idx].jump_timer = 0.0
                self.cars[car_idx].air_timer = 0.0
                self.cars[car_idx].ball_touches = 0
                self.cars[car_idx].demoed = False

    def step(self, actions: List[np.ndarray], dt: float = 1.0 / 15.0) -> Tuple[bool, Optional[int]]:
        """
        Step simulation by dt seconds with high-precision 120 Hz sub-stepping.
        Guarantees zero tunneling on ball collisions and smooth continuous physics.
        """
        self.step_count += 1
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
                    # Coasting friction
                    car.vel *= max(0.0, 1.0 - 0.5 * dt)

                # Steering (positive steer turns right / clockwise)
                turn_rate = (3.5 - 2.0 * (speed / CAR_MAX_SPEED)) * steer
                if handbrake:
                    turn_rate *= 1.8
                    # Side friction reduction for drifting
                    right = car.get_right_vector()
                    side_vel = np.dot(car.vel, right)
                    car.vel -= right * (side_vel * 0.4)
                car.rot[1] -= turn_rate * dt

                # Jump initiation
                if jump and car.has_jump:
                    car.on_ground = False
                    car.is_jumping = True
                    car.jump_timer = CAR_MAX_JUMP_TIME
                    car.vel[2] += CAR_JUMP_INITIAL_VEL
                    car.has_jump = False
            else:
                # In air
                car.air_timer += dt
                # Gravity
                car.vel[2] += GRAVITY * dt

                # Jump hold acceleration
                if jump and car.is_jumping and car.jump_timer > 0:
                    car.vel[2] += CAR_JUMP_ACCEL * dt
                    car.jump_timer -= dt
                else:
                    car.is_jumping = False

                # Aerial rotation (pitch up > 0, yaw right > 0, roll right > 0)
                car.rot[0] += pitch * CAR_AIR_PITCH_TORQUE * dt
                car.rot[1] -= yaw * CAR_AIR_YAW_TORQUE * dt
                car.rot[2] += roll * CAR_AIR_ROLL_TORQUE * dt

                # Dodge / Flip on jump re-press (rising edge) or first air jump
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
                            # Flip angular velocity boost
                            car.rot[0] += fwd_input * 1.5
                            car.rot[2] += side_input * 1.5

            car.prev_jump = jump

            # Clamp car speed
            curr_speed = float(np.linalg.norm(car.vel))
            if curr_speed > CAR_MAX_SPEED:
                car.vel = (car.vel / curr_speed) * CAR_MAX_SPEED

            # Position integration
            car.pos += car.vel * dt

            # Floor collision
            if car.pos[2] <= 17.0:
                car.pos[2] = 17.0
                car.vel[2] = max(0.0, car.vel[2])
                car.on_ground = True
                car.rot[0] = 0.0
                car.rot[2] = 0.0

            # Ceiling collision
            if car.pos[2] >= ARENA_HEIGHT_Z - 17.0:
                car.pos[2] = ARENA_HEIGHT_Z - 17.0
                car.vel[2] = min(0.0, car.vel[2])

            # Arena wall collisions for car
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
                # Check goal net opening
                if not (abs(car.pos[0]) < GOAL_HALF_WIDTH and car.pos[2] < GOAL_HEIGHT):
                    car.pos[1] = math.copysign(ARENA_EXTENT_Y - 60.0, car.pos[1])
                    car.vel[1] *= -0.2

            # Boost pad collection
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

        # Ball Floor & Ceiling Bounce
        if self.ball.pos[2] <= BALL_RADIUS:
            self.ball.pos[2] = BALL_RADIUS
            self.ball.vel[2] = -self.ball.vel[2] * BALL_RESTITUTION
        elif self.ball.pos[2] >= ARENA_HEIGHT_Z - BALL_RADIUS:
            self.ball.pos[2] = ARENA_HEIGHT_Z - BALL_RADIUS
            self.ball.vel[2] = -self.ball.vel[2] * BALL_RESTITUTION

        # Ball 45-Degree Slanted Corner Wall Bounce
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

        # Ball Side Wall Bounce
        elif abs(self.ball.pos[0]) >= ARENA_EXTENT_X - BALL_RADIUS:
            self.ball.pos[0] = math.copysign(ARENA_EXTENT_X - BALL_RADIUS, self.ball.pos[0])
            self.ball.vel[0] = -self.ball.vel[0] * BALL_RESTITUTION

        # Ball Back Wall & Goal Scoring Check
        elif abs(self.ball.pos[1]) >= ARENA_EXTENT_Y - BALL_RADIUS:
            in_goal_x = abs(self.ball.pos[0]) < (GOAL_HALF_WIDTH - BALL_RADIUS * 0.5)
            in_goal_z = self.ball.pos[2] < (GOAL_HEIGHT - BALL_RADIUS * 0.5)

            if in_goal_x and in_goal_z:
                # Goal scored!
                if self.ball.pos[1] > 0:
                    # Scored in Orange Goal -> Team 0 (Blue) scores!
                    self.scored_team = 0
                    return True, 0
                else:
                    # Scored in Blue Goal -> Team 1 (Orange) scores!
                    self.scored_team = 1
                    return True, 1
            else:
                # Rebound from back wall
                self.ball.pos[1] = math.copysign(ARENA_EXTENT_Y - BALL_RADIUS, self.ball.pos[1])
                self.ball.vel[1] = -self.ball.vel[1] * BALL_RESTITUTION

        # 4. Ball-Car Collisions
        for car in self.cars:
            if car.demoed:
                continue
            delta = self.ball.pos - car.pos
            dist = float(np.linalg.norm(delta))
            min_dist = BALL_RADIUS + 75.0  # Octane hitbox collision radius (92.75 + 75.0 = 167.75 uu)
            if dist < min_dist and dist > 1e-4:
                car.ball_touches += 1
                normal = delta / dist
                # Push ball outside car
                self.ball.pos = car.pos + normal * min_dist
                # Standard Rocket League elastic collision impulse
                relative_vel = self.ball.vel - car.vel
                vel_along_normal = float(np.dot(relative_vel, normal))
                if vel_along_normal < 0:
                    # Normal elastic collision impulse + Rocket League car impact bonus
                    restitution_impulse = -(1.0 + BALL_RESTITUTION) * vel_along_normal
                    car_hit_power = 200.0 + float(np.linalg.norm(car.vel)) * 0.4
                    self.ball.vel += normal * (restitution_impulse + car_hit_power)

        return False, None
