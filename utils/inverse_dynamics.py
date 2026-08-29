"""
Inverse Dynamics Solver for Rocket League / RocketSim.
Reconstructs exact continuous controller actions:
  [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
from consecutive state transitions (S_t -> S_t+1) based on Rocket League Bullet physics equations of motion.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any, Tuple, Optional


# Physical Constants matching RocketSim / Rocket League 120Hz physics
CAR_MAX_SPEED = 2300.0
CAR_SUPERSONIC_SPEED = 2200.0
GRAVITY_Z = -650.0

# Acceleration curves
DRIVE_ACCEL_ZERO = 3500.0   # uu/s^2 at 0 speed
BOOST_ACCEL = 991.667       # uu/s^2 forward boost force
BRAKE_ACCEL = 3500.0        # uu/s^2 braking deceleration
COAST_ACCEL = 525.0         # uu/s^2 friction coasting

# Airborne Torque Rates (rad/s^2)
PITCH_TORQUE = 12.46
YAW_TORQUE = 9.11
ROLL_TORQUE = 38.34


class InverseDynamicsSolver:
    """
    Solves for the most probable continuous action vector a_t in [-1, 1]^8
    that caused the transition from CarState(t) to CarState(t+1) given dt.
    """

    @staticmethod
    def solve_car_action(
        pos_t: np.ndarray,
        vel_t: np.ndarray,
        rot_t: np.ndarray,          # [pitch, yaw, roll] in radians
        ang_vel_t: np.ndarray,      # [wx, wy, wz] in rad/s (or zeros if unavailable)
        boost_t: float,
        on_ground_t: bool,
        pos_next: np.ndarray,
        vel_next: np.ndarray,
        rot_next: np.ndarray,
        ang_vel_next: np.ndarray,
        boost_next: float,
        on_ground_next: bool,
        dt: float = 1.0 / 30.0      # Replay frame interval (~30Hz or 15Hz)
    ) -> np.ndarray:
        """
        Solves action vector: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        """
        dt = max(1e-4, float(dt))

        # 1. Orientation Basis Matrices at time t
        cp, sp = math.cos(rot_t[0]), math.sin(rot_t[0])
        cy, sy = math.cos(rot_t[1]), math.sin(rot_t[1])
        cr, sr = math.cos(rot_t[2]), math.sin(rot_t[2])

        fwd = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
        right = np.array([-sy * cr + cy * sp * sr, cy * cr + sy * sp * sr, -cp * sr], dtype=np.float32)
        up = np.array([-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr], dtype=np.float32)

        # 2. Linear Accelerations
        measured_accel = (vel_next - vel_t) / dt
        # Remove gravity component
        non_grav_accel = measured_accel - np.array([0.0, 0.0, GRAVITY_Z], dtype=np.float32)

        a_fwd = float(np.dot(non_grav_accel, fwd))
        a_right = float(np.dot(non_grav_accel, right))
        a_up = float(np.dot(non_grav_accel, up))

        speed_fwd = float(np.dot(vel_t, fwd))
        current_speed = float(np.linalg.norm(vel_t))

        # 3. Boost & Throttle Recovery
        boost_delta = boost_t - boost_next
        boost_consumed = bool(boost_delta > 0.05 or (boost_t > 0 and a_fwd > 1200.0))

        if boost_consumed:
            boost_act = 1.0
            throttle_act = 1.0
        else:
            boost_act = -1.0
            # Solve throttle from drive curve: a = drive_accel(v) * throttle
            max_drive_accel = max(100.0, DRIVE_ACCEL_ZERO * (1.0 - min(current_speed, 1400.0) / 1400.0))
            if on_ground_t:
                if a_fwd > 50.0:
                    throttle_act = float(np.clip(a_fwd / max_drive_accel, 0.0, 1.0))
                elif a_fwd < -200.0:
                    # Braking or reverse
                    throttle_act = float(np.clip(a_fwd / BRAKE_ACCEL, -1.0, 0.0))
                else:
                    throttle_act = 0.0 if abs(speed_fwd) < 50.0 else 0.5
            else:
                throttle_act = 1.0 if a_fwd > 0 else 0.0

        # 4. Angular Rotations (Pitch, Yaw, Roll, Steer)
        # Angular change delta
        delta_rot = rot_next - rot_t
        # Wrap to [-pi, pi]
        delta_rot = (delta_rot + np.pi) % (2 * np.pi) - np.pi
        measured_omega = delta_rot / dt

        if on_ground_t and on_ground_next:
            # Ground Steering
            yaw_rate = float(measured_omega[1])
            if abs(speed_fwd) > 100.0:
                # Steer is proportional to yaw_rate * radius / speed
                steer_val = (yaw_rate * 500.0) / max(200.0, abs(speed_fwd))
                steer_act = float(np.clip(steer_val, -1.0, 1.0))
            else:
                steer_act = float(np.clip(yaw_rate * 0.5, -1.0, 1.0))

            # Handbrake / Powerslide Detection
            # High lateral slip with low yaw curvature or rapid direction snap
            lateral_slip = abs(float(np.dot(vel_t, right)))
            if lateral_slip > 350.0 and abs(steer_act) > 0.2:
                handbrake_act = 1.0
            else:
                handbrake_act = -1.0

            pitch_act = 0.0
            yaw_act = 0.0
            roll_act = 0.0
        else:
            # Airborne 3D Attitude Control
            steer_act = 0.0
            handbrake_act = -1.0

            # Pitch (in RocketSim/RLGym: -1.0 is nose down, +1.0 is nose up)
            pitch_rate = float(measured_omega[0])
            pitch_act = float(np.clip(pitch_rate / PITCH_TORQUE, -1.0, 1.0))

            # Yaw
            yaw_rate = float(measured_omega[1])
            yaw_act = float(np.clip(yaw_rate / YAW_TORQUE, -1.0, 1.0))

            # Roll
            roll_rate = float(measured_omega[2])
            roll_act = float(np.clip(roll_rate / ROLL_TORQUE, -1.0, 1.0))

        # 5. Jump & Dodge Detection
        # Initial jump: ground -> air with upward vertical impulse (vz >= 250 uu/s)
        jump_act = -1.0
        if on_ground_t and not on_ground_next and vel_next[2] > 200.0:
            jump_act = 1.0
        elif not on_ground_t and not on_ground_next:
            # Airborne dodge / double jump impulse (sudden velocity spike > 400 uu/s)
            delta_v_mag = float(np.linalg.norm(vel_next - vel_t))
            if delta_v_mag > 450.0 and a_fwd > 600.0:
                jump_act = 1.0
                if abs(pitch_act) < 0.2:
                    pitch_act = -1.0  # Front-flip dodge default

        return np.array([
            float(np.clip(throttle_act, -1.0, 1.0)),
            float(np.clip(steer_act, -1.0, 1.0)),
            float(np.clip(pitch_act, -1.0, 1.0)),
            float(np.clip(yaw_act, -1.0, 1.0)),
            float(np.clip(roll_act, -1.0, 1.0)),
            float(np.clip(jump_act, -1.0, 1.0)),
            float(np.clip(boost_act, -1.0, 1.0)),
            float(np.clip(handbrake_act, -1.0, 1.0))
        ], dtype=np.float32)

    @classmethod
    def batch_extract_actions(
        cls,
        car_pos: np.ndarray,        # (N, num_cars, 3) or (N, 3)
        car_vel: np.ndarray,        # (N, num_cars, 3)
        car_rot: np.ndarray,        # (N, num_cars, 3)
        car_boost: np.ndarray,      # (N, num_cars)
        dt: float = 1.0 / 30.0
    ) -> np.ndarray:
        """
        Vectorized/Batch action extraction across consecutive frames.
        Returns: (N-1, num_cars, 8) or (N-1, 8) action array.
        """
        n_frames = car_pos.shape[0]
        if n_frames < 2:
            return np.zeros((0, 8), dtype=np.float32)

        is_multi_car = (car_pos.ndim == 3)
        num_cars = car_pos.shape[1] if is_multi_car else 1

        actions = []
        for t in range(n_frames - 1):
            if is_multi_car:
                frame_acts = []
                for c in range(num_cars):
                    p_t, p_next = car_pos[t, c], car_pos[t + 1, c]
                    v_t, v_next = car_vel[t, c], car_vel[t + 1, c]
                    r_t, r_next = car_rot[t, c], car_rot[t + 1, c]
                    b_t, b_next = car_boost[t, c], car_boost[t + 1, c]

                    on_gnd_t = bool(p_t[2] < 25.0)
                    on_gnd_next = bool(p_next[2] < 25.0)

                    act = cls.solve_car_action(
                        p_t, v_t, r_t, np.zeros(3, dtype=np.float32), b_t, on_gnd_t,
                        p_next, v_next, r_next, np.zeros(3, dtype=np.float32), b_next, on_gnd_next,
                        dt=dt
                    )
                    frame_acts.append(act)
                actions.append(frame_acts)
            else:
                p_t, p_next = car_pos[t], car_pos[t + 1]
                v_t, v_next = car_vel[t], car_vel[t + 1]
                r_t, r_next = car_rot[t], car_rot[t + 1]
                b_t, b_next = car_boost[t], car_boost[t + 1]

                on_gnd_t = bool(p_t[2] < 25.0)
                on_gnd_next = bool(p_next[2] < 25.0)

                act = cls.solve_car_action(
                    p_t, v_t, r_t, np.zeros(3, dtype=np.float32), b_t, on_gnd_t,
                    p_next, v_next, r_next, np.zeros(3, dtype=np.float32), b_next, on_gnd_next,
                    dt=dt
                )
                actions.append(act)

        return np.array(actions, dtype=np.float32)
