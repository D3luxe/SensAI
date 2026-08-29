"""
Rule-Based Baseline Agent & Heuristic Opponents for Rocket League Training.
Provides aggressive kickoff challenging and pursuit dynamics to prevent self-play collusion.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional
from env.physics_engine import CarState, BallState, ARENA_EXTENT_Y


class BaselineChaser:
    """
    High-tempo heuristic opponent that challenges kickoffs and chases the ball directly.
    Incentivizes learning policies to execute disciplined kickoffs and 50/50 challenges.
    """
    def __init__(self, continuous_actions: bool = True):
        self.continuous_actions = continuous_actions

    def get_action(self, car: CarState, ball: BallState) -> np.ndarray:
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
