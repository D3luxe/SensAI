"""
In-Game Scenario Replicator & Diagnostic Inspector.
Replicates live in-game scenarios from RLBot in RocketSim to inspect observations, neural policy outputs, and trajectories side-by-side.
"""

from __future__ import annotations
import os
import sys
import math
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import RocketSim as rsim
from env.physics_engine import CarState, BallState, BoostPad, RocketSimArena
from env.observations import DefaultObservationBuilder
from agent.models import ActorCritic
from bot import SenseiRLBot, rotation_to_rot_mat


def run_scenario_simulation(
    car_pos=(0.0, -3000.0, 17.0),
    car_vel=(0.0, 800.0, 0.0),
    car_yaw_deg=90.0,
    ball_pos=(400.0, -1500.0, 93.0),
    ball_vel=(0.0, 0.0, 0.0),
    num_steps=30,
    model_path="checkpoints/latest_model.pt"
):
    print("=" * 80)
    print(" SENSEIBOT IN-GAME SCENARIO REPLICATOR & DIAGNOSTIC INSPECTOR")
    print("=" * 80)

    if not os.path.exists(model_path):
        print(f"[Error] Model checkpoint not found at {model_path}")
        return

    ckpt = torch.load(model_path, map_location="cpu")
    model = ActorCritic(obs_dim=74, act_dim=8, continuous_actions=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.debias_symmetric_actions()
    model.eval()

    obs_builder = DefaultObservationBuilder(symmetric=True)
    arena = RocketSimArena(num_players=2, game_mode="1v1")
    arena.reset(random_kickoff=False)

    car_yaw_rad = math.radians(car_yaw_deg)
    cs = arena._rsim_cars[0].get_state()
    cs.pos = rsim.Vec(car_pos[0], car_pos[1], car_pos[2])
    cs.vel = rsim.Vec(car_vel[0], car_vel[1], car_vel[2])
    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=car_yaw_rad, roll=0.0).as_rot_mat()
    arena._rsim_cars[0].set_state(cs)

    bs = arena._rsim_arena.ball.get_state()
    bs.pos = rsim.Vec(ball_pos[0], ball_pos[1], ball_pos[2])
    bs.vel = rsim.Vec(ball_vel[0], ball_vel[1], ball_vel[2])
    arena._rsim_arena.ball.set_state(bs)
    arena._sync_from_rsim()

    print(f"Initial State:")
    print(f"  Car:  Pos=({car_pos[0]:.1f}, {car_pos[1]:.1f}, {car_pos[2]:.1f}) | Facing={car_yaw_deg:.1f} deg")
    print(f"  Ball: Pos=({ball_pos[0]:.1f}, {ball_pos[1]:.1f}, {ball_pos[2]:.1f})")
    print("-" * 80)

    trajectory = []
    for step in range(num_steps):
        car = arena.cars[0]
        obs = obs_builder.build_obs(car, arena)
        with torch.no_grad():
            act, _, _, val = model.get_action_and_value(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True)
            act_np = act[0].cpu().numpy()

        steer = float(act_np[1])
        throttle = float(act_np[0])
        boost = bool(act_np[6] > 0.3)
        handbrake = bool(act_np[7] > 0.5)
        loc_fwd = obs[34]
        loc_right = obs[35]

        car_pos_str = f"({car.pos[0]:.0f}, {car.pos[1]:.0f})"
        ball_pos_str = f"({arena.ball.pos[0]:.0f}, {arena.ball.pos[1]:.0f})"
        print(f"Step {step:2d} | Car {car_pos_str:>14} | Ball {ball_pos_str:>14} | Local Right: {loc_right:+.3f} | Steer: {steer:+.3f} | Thr: {throttle:+.2f} | Bst: {boost} | Hnd: {handbrake}")
        trajectory.append((car.pos[0], car.pos[1], steer))
        arena.step([act_np, np.zeros(8, dtype=np.float32)], dt=8.0 / 120.0)

        if arena.cars[0].ball_touches > 0:
            print("-" * 80)
            print(f">>> BALL TOUCH at step {step} ({step * 0.0667:.2f}s)!")
            break
    print("=" * 80)
    return trajectory


def replay_log_line(log_line: str, model_path="checkpoints/latest_model.pt"):
    """
    Parses a single line from rlbot_live.log and replicates the exact scenario.
    Example: '[TICK 600] pos=(-795, 470) ball=(-1365, 533) kickoff=False -> ...'
    """
    import re
    pos_match = re.search(r"pos=\(([-\d]+),\s*([-\d]+)\)", log_line)
    ball_match = re.search(r"ball=\(([-\d]+),\s*([-\d]+)\)", log_line)
    if not pos_match or not ball_match:
        print("[Error] Could not parse car or ball coordinates from log line.")
        return

    cx, cy = float(pos_match.group(1)), float(pos_match.group(2))
    bx, by = float(ball_match.group(1)), float(ball_match.group(2))
    print(f"[Replaying Log Line] Replicating Car=({cx}, {cy}) chasing Ball=({bx}, {by})")
    run_scenario_simulation(
        car_pos=(cx, cy, 17.0),
        car_vel=(0.0, 800.0, 0.0),
        car_yaw_deg=90.0,
        ball_pos=(bx, by, 93.0),
        ball_vel=(0.0, 0.0, 0.0),
        num_steps=25,
        model_path=model_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate In-Game Scenarios in RocketSim")
    parser.add_argument("--car_x", type=float, default=0.0)
    parser.add_argument("--car_y", type=float, default=-3000.0)
    parser.add_argument("--car_yaw", type=float, default=90.0)
    parser.add_argument("--ball_x", type=float, default=300.0)
    parser.add_argument("--ball_y", type=float, default=-1500.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--model", type=str, default="checkpoints/latest_model.pt")
    parser.add_argument("--log_line", type=str, default=None, help="Replay directly from a line of rlbot_live.log")

    args = parser.parse_args()

    if args.log_line:
        replay_log_line(args.log_line, model_path=args.model)
    else:
        run_scenario_simulation(
            car_pos=(args.car_x, args.car_y, 17.0),
            car_vel=(0.0, 600.0, 0.0),
            car_yaw_deg=args.car_yaw,
            ball_pos=(args.ball_x, args.ball_y, 93.0),
            ball_vel=(0.0, 0.0, 0.0),
            num_steps=args.steps,
            model_path=args.model
        )
