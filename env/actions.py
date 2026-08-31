"""
Action Space Parsers for Rocket League Agents.
Supports continuous 8-axis control and discrete action tables.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Union


class ContinuousActionParser:
    """
    Standard continuous action parser: 8 continuous float outputs in [-1, 1].
    Indices:
    0: Throttle (-1 to 1)
    1: Steer (-1 to 1)
    2: Pitch (-1 to 1)
    3: Yaw (-1 to 1)
    4: Roll (-1 to 1)
    5: Jump (0 or 1, threshold > 0.0)
    6: Boost (0 or 1, threshold > 0.0)
    7: Handbrake (0 or 1, threshold > 0.0)
    """
    def __init__(self):
        self.action_dim = 8

    def parse_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        actions = np.array(raw_actions, dtype=np.float32).copy()

        # True continuous analog throttle with deadband around zero to eliminate micro-jitter
        thr = actions[..., 0]
        thr_deadband = (np.abs(thr) < 0.05)
        thr[thr_deadband] = 0.0
        actions[..., 0] = np.clip(thr, -1.0, 1.0)

        # Steering & aerial rotations: pure continuous in [-1.0, 1.0]
        actions[..., 1] = np.clip(actions[..., 1], -1.0, 1.0)
        actions[..., 2] = np.clip(actions[..., 2], -1.0, 1.0)
        actions[..., 3] = np.clip(actions[..., 3], -1.0, 1.0)
        actions[..., 4] = np.clip(actions[..., 4], -1.0, 1.0)

        # Binary button thresholds for Jump, Boost, and Handbrake (Hybrid Bernoulli actions in {-1.0, +1.0})
        actions[..., 5] = (actions[..., 5] > 0.0).astype(np.float32)  # Jump (> 0.0 Bernoulli threshold)
        actions[..., 6] = (actions[..., 6] > 0.0).astype(np.float32)  # Boost
        actions[..., 7] = (actions[..., 7] > 0.0).astype(np.float32)  # Handbrake
        return actions


class DiscreteActionParser:
    """
    Standard RLGym 24-Action Discrete Lookup Table (Ground Locomotion, Dodges, 3D Flight & Air-Roll).
    Equips the policy with complete 6-DOF aerospace controls, air-roll recoveries, and directional aerial carries.
    """
    def __init__(self):
        # [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        self.lookup_table = np.array([
            # ── GROUND LOCOMOTION & POWERSLIDES ──
            [ 0.0,  0.0,  0.0,  0.0,  0.0, 0, 0, 0],  # 0: Coast / Idle
            [ 1.0,  0.0,  0.0,  0.0,  0.0, 0, 0, 0],  # 1: Forward Drive
            [ 1.0, -1.0,  0.0,  0.0,  0.0, 0, 0, 0],  # 2: Forward + Steer Left
            [ 1.0,  1.0,  0.0,  0.0,  0.0, 0, 0, 0],  # 3: Forward + Steer Right
            [ 1.0,  0.0,  0.0,  0.0,  0.0, 0, 1, 0],  # 4: Forward Boost (Straight Line Rush)
            [ 1.0, -1.0,  0.0,  0.0,  0.0, 0, 1, 0],  # 5: Forward Boost + Steer Left
            [ 1.0,  1.0,  0.0,  0.0,  0.0, 0, 1, 0],  # 6: Forward Boost + Steer Right
            [-1.0,  0.0,  0.0,  0.0,  0.0, 0, 0, 0],  # 7: Brake / Reverse
            [-1.0, -1.0,  0.0,  0.0,  0.0, 0, 0, 0],  # 8: Reverse + Steer Left
            [-1.0,  1.0,  0.0,  0.0,  0.0, 0, 0, 0],  # 9: Reverse + Steer Right
            [ 1.0, -1.0,  0.0,  0.0,  0.0, 0, 0, 1],  # 10: Powerslide / Drift Left
            [ 1.0,  1.0,  0.0,  0.0,  0.0, 0, 0, 1],  # 11: Powerslide / Drift Right

            # ── JUMPS & FLIPS ──
            [ 1.0,  0.0,  0.0,  0.0,  0.0, 1, 0, 0],  # 12: Jump / Hop Forward
            [ 1.0,  0.0, -1.0,  0.0,  0.0, 1, 0, 0],  # 13: Front Flip / Speed Dodge
            [ 1.0, -1.0, -1.0, -1.0, -1.0, 1, 0, 0],  # 14: Left Diagonal Flip (Pitch -1, Yaw -1, Roll -1)
            [ 1.0,  1.0, -1.0,  1.0,  1.0, 1, 0, 0],  # 15: Right Diagonal Flip (Pitch -1, Yaw +1, Roll +1)
            [-1.0,  0.0,  1.0,  0.0,  0.0, 1, 0, 0],  # 16: Back Flip
            [ 0.0, -1.0,  0.0, -1.0, -1.0, 1, 0, 0],  # 17: Side Dodge Left (Yaw -1, Roll -1)
            [ 0.0,  1.0,  0.0,  1.0,  1.0, 1, 0, 0],  # 18: Side Dodge Right (Yaw +1, Roll +1)

            # ── 3D AERIAL FLIGHT & AIR-ROLL ──
            [ 1.0,  0.0,  1.0,  0.0,  0.0, 0, 1, 0],  # 19: Fast Aerial Climb (Nose UP + Boost)
            [ 1.0,  0.0,  1.0, -1.0,  0.0, 0, 1, 0],  # 20: Aerial Climb + Yaw Left + Boost
            [ 1.0,  0.0,  1.0,  1.0,  0.0, 0, 1, 0],  # 21: Aerial Climb + Yaw Right + Boost
            [ 1.0,  0.0,  1.0,  0.0, -1.0, 0, 1, 0],  # 22: Directional Air-Roll Left + Pitch UP + Boost
            [ 1.0,  0.0,  1.0,  0.0,  1.0, 0, 1, 0],  # 23: Directional Air-Roll Right + Pitch UP + Boost
        ], dtype=np.float32)
        self.action_dim = len(self.lookup_table)

    def parse_actions(self, discrete_indices: Union[int, np.ndarray]) -> np.ndarray:
        return self.lookup_table[discrete_indices]
