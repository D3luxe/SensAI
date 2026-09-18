"""
Dropped-ball scenario: how SensAI reacts to a ball falling at midfield, and what it is paid for it.

    orange goal (+Y)
    +------------------+
    |  O->             |   orange shadowing near its own goal line, rolling across (+X)
    |                  |
    |------------------|   centre line
    |  o               |   ball dropped from 1000 uu, no velocity, just short of the line
    |  ^               |   SensAI (blue) behind it, stopped or rolling slowly upfield
    |                  |
    +------------------+
    blue goal (-Y)

Places the cars and ball, then lets the checkpoint drive blue (deterministic, like the live bot)
and logs every step: SensAI's state, the policy's mean actions, the critic's value, the point
player_to_ball is pulling it toward, and every reward term that paid or charged it.

Presets:
    drop  (default) the layout above
    wall  the ball rolls into the side wall at a shallow angle, rides up the curve and drops back;
          SensAI follows it at dribbling pace, orange waits near its own goal

Usage:
    python scripts/drop_ball_scenario.py
    python scripts/drop_ball_scenario.py --preset wall
    python scripts/drop_ball_scenario.py --car-speed 250 --opponent necto --seconds 8
    python scripts/drop_ball_scenario.py --checkpoint checkpoints/checkpoint_iter_142400.pt

--opponent: self (the same checkpoint drives orange, default), idle (orange coasts with no input),
necto, nexto, heuristic, or a path to any checkpoint.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
os.environ.setdefault("RS_COLLISION_MESHES", os.path.join(ROOT, "collision_meshes"))

import RocketSim as rsim  # noqa: E402

from policy_health import load_agent, load_weights  # noqa: E402
from env.rocket_env import RocketLeagueEnv  # noqa: E402
from env.rewards import defensive_recovery_point  # noqa: E402

DT = 8.0 / 120.0
ACTION_NAMES = ("thr", "str", "pit", "yaw", "rol", "jmp", "bst", "hbk")
TERM_SHORT = {
    "ball_to_goal": "b2g", "own_goal_threat": "own_thr", "player_to_ball": "p2b", "jump_bridge": "jbridge",
    "touch": "touch", "boost": "boost", "powerslide": "pslide", "air_roll_recovery": "airrec",
    "retreat_flip": "rflip", "jump_cost": "jcost", "time_cost": "time", "spin_cost": "spin",
    "overshoot": "overshoot", "lateral_slip": "lslip", "slide_waste": "swaste", "goal": "GOAL",
}


# ball_vel z of -1 rather than 0: RocketSim leaves a ball placed with exactly zero velocity asleep
PRESETS = {
    "drop": dict(ball=(-2750.0, -220.0, 1000.0), ball_vel=(0.0, 0.0, -1.0), car=(-2750.0, -690.0),
                 car_yaw=90.0, car_speed=0.0, opp=(-2150.0, 3750.0), opp_speed=300.0),
    "wall": dict(ball=(2900.0, -600.0, 93.0), ball_vel=(1000.0, 650.0, 0.0), car=(2250.0, -1250.0),
                 car_yaw=40.0, car_speed=900.0, opp=(-800.0, 4300.0), opp_speed=0.0),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/latest_model.pt")
    p.add_argument("--opponent", default="self")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--preset", choices=sorted(PRESETS), default="drop")
    p.add_argument("--car-speed", type=float, default=None, help="SensAI's starting speed along its heading, uu/s")
    p.add_argument("--car-yaw", type=float, default=None, help="SensAI's heading in degrees (90 = upfield)")
    p.add_argument("--boost", type=float, default=33.0, help="starting boost for both cars")
    p.add_argument("--ball", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--ball-vel", type=float, nargs=3, default=None, metavar=("VX", "VY", "VZ"))
    p.add_argument("--car", type=float, nargs=2, default=None, metavar=("X", "Y"))
    p.add_argument("--opp", type=float, nargs=2, default=None, metavar=("X", "Y"))
    p.add_argument("--opp-speed", type=float, default=None, help="orange's starting speed across the field (+X)")
    p.add_argument("--min-term", type=float, default=0.003, help="hide reward terms smaller than this")
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def resolve_opponent(name):
    key = name.lower()
    if key == "necto":
        return "checkpoints/necto-model.pt"
    if key == "nexto":
        return "checkpoints/nexto-model.pt"
    return key if key in ("self", "idle", "heuristic") else name


def set_car(r_car, pos, yaw, speed, boost):
    cs = r_car.get_state()
    cs.pos = rsim.Vec(float(pos[0]), float(pos[1]), 17.0)
    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
    cs.vel = rsim.Vec(speed * math.cos(yaw), speed * math.sin(yaw), 0.0)
    cs.ang_vel = rsim.Vec(0.0, 0.0, 0.0)
    cs.boost = float(boost)
    r_car.set_state(cs)


def place(env, args):
    env.reset(random_kickoff=False)
    arena = env.arena
    bs = arena._rsim_arena.ball.get_state()
    bs.pos = rsim.Vec(*[float(v) for v in args.ball])
    bs.vel = rsim.Vec(*[float(v) for v in args.ball_vel])
    bs.ang_vel = rsim.Vec(0.0, 0.0, 0.0)
    arena._rsim_arena.ball.set_state(bs)
    set_car(arena._rsim_cars[0], args.car, math.radians(args.car_yaw), args.car_speed, args.boost)
    set_car(arena._rsim_cars[1], args.opp, 0.0, args.opp_speed, args.boost)          # orange faces +X
    arena._sync_from_rsim()
    # Every reward term re-reads its baseline (potentials, touch counts) from the placed state
    env.reward_manager.reset(arena)
    env.current_step = 0
    env.current_scenario = "drop_ball"
    env.scenario_timeout = 10 ** 9
    return np.array([env.obs_builder.build_obs(c, arena) for c in arena.cars], dtype=np.float32)


def fmt_terms(breakdown, min_term):
    items = sorted(((k, v) for k, v in breakdown.items() if abs(v) >= min_term), key=lambda kv: -abs(kv[1]))
    return " ".join(f"{TERM_SHORT.get(k, k)}{v:+.3f}" for k, v in items)


def car_events(car, prev):
    ev = []
    if prev["on_ground"] and not car.on_ground:
        ev.append("LIFTOFF")
    if car.just_dodged:
        ev.append("DODGE")
    elif prev["has_jump"] and not car.has_jump and not car.on_ground:
        ev.append("JUMP")
    if not prev["on_ground"] and car.on_ground:
        ev.append("LAND")
    return ev


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent, ckpt = load_agent(args.checkpoint)
    weights = load_weights(ckpt)
    opponent = resolve_opponent(args.opponent)
    selfplay = opponent == "self"
    idle = opponent == "idle"
    env = RocketLeagueEnv(
        game_mode="1v1", max_episode_steps=10 ** 9, reward_weights=weights,
        is_baseline_env=not selfplay, baseline_opponent_type="heuristic" if (selfplay or idle) else opponent,
    )
    obs = place(env, args)
    arena = env.arena
    p2b = env.reward_manager.combined.rewards["player_to_ball"]
    gamma = float(weights.get("gamma", 0.995))

    print(f"checkpoint {args.checkpoint} (iteration {ckpt.get('iteration', -1)})  opponent {args.opponent}")
    b = arena.ball.pos
    bv = args.ball_vel
    print(f"preset {args.preset}  |  ball ({b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}) vel ({bv[0]:.0f}, {bv[1]:.0f}, {bv[2]:.0f})  |  "
          f"SensAI ({args.car[0]:.0f}, {args.car[1]:.0f}) heading {args.car_yaw:.0f} deg at {args.car_speed:.0f} uu/s  |  orange ({args.opp[0]:.0f}, {args.opp[1]:.0f}) facing +X at "
          f"{args.opp_speed:.0f} uu/s  |  boost {args.boost:.0f}")
    print("columns: t | SensAI pos, speed, speed toward ball | dist to ball | ball z, vz | "
          "p2b target x,y,z (d = its ground distance from the ball; 0 = chasing the ball itself) | mean actions | V(s) | reward, terms")
    print("-" * 150)

    totals, ret, discounted = {}, 0.0, 0.0
    c0 = arena.cars[0]
    prev = dict(on_ground=c0.on_ground, has_jump=c0.has_jump)
    prev_touch = [c.ball_touches for c in arena.cars]
    steps = int(round(args.seconds / DT))

    for t in range(steps):
        with torch.no_grad():
            out = agent.get_action_and_value(torch.from_numpy(obs).float(), deterministic=True)
        act = out[0].numpy()
        value = float(out[-1].reshape(-1)[0])
        opp_action = np.zeros(8, dtype=np.float32) if idle else None
        obs, _, done, info = env.step(act, include_breakdown=True, opponent_action=opp_action)
        bd = info.get("reward_breakdown") or {}
        r = sum(bd.values())
        ret += r
        discounted += gamma ** t * r
        for k, v in bd.items():
            totals[k] = totals.get(k, 0.0) + v

        car, opp, ball = arena.cars[0], arena.cars[1], arena.ball
        to_ball = ball.pos - car.pos
        dist = float(np.linalg.norm(to_ball))
        speed = float(np.linalg.norm(car.vel))
        closing = float(np.dot(car.vel, to_ball)) / max(1.0, dist)
        target = p2b._prev_target.get(car.id)
        tgt = (f"({target[0]:6.0f},{target[1]:6.0f},{target[2]:4.0f}) d{np.linalg.norm(target[:2] - ball.pos[:2]):5.0f}"
               if target is not None else "-")
        a = act[0]
        acts = " ".join(f"{n}{a[i]:+.1f}" for i, n in enumerate(ACTION_NAMES))
        ev = car_events(car, prev)
        prev = dict(on_ground=car.on_ground, has_jump=car.has_jump)
        if car.ball_touches > prev_touch[0]:
            ev.append("** TOUCH **")
        if opp.ball_touches > prev_touch[1]:
            ev.append("** ORANGE TOUCH **")
        prev_touch = [car.ball_touches, opp.ball_touches]

        print(f"{t * DT:4.2f}s | ({car.pos[0]:6.0f},{car.pos[1]:6.0f},{car.pos[2]:4.0f}) {speed:4.0f} {closing:+5.0f} | "
              f"d{dist:5.0f} | z{ball.pos[2]:4.0f} vz{ball.vel[2]:+5.0f} | {tgt} | {acts} | V{value:+6.2f} | "
              f"{r:+.3f} {fmt_terms(bd, args.min_term)}" + (f"  <- {' '.join(ev)}" if ev else ""))
        if np.any(done):
            g = info.get("episode_goals", [0, 0])
            print(f"-- episode ended: {'GOAL FOR' if g[0] else 'GOAL AGAINST' if g[1] else 'reset'}")
            break

    print("-" * 150)
    rec = defensive_recovery_point(arena.ball.pos, car_team=0)
    print(f"final: SensAI ({arena.cars[0].pos[0]:.0f}, {arena.cars[0].pos[1]:.0f})  ball ({arena.ball.pos[0]:.0f}, "
          f"{arena.ball.pos[1]:.0f}, {arena.ball.pos[2]:.0f})  orange ({arena.cars[1].pos[0]:.0f}, "
          f"{arena.cars[1].pos[1]:.0f})  recovery point ({rec[0]:.0f}, {rec[1]:.0f})")
    print(f"return {ret:+.3f}  discounted (gamma {gamma}) {discounted:+.3f}")
    print("reward by term over the run:")
    for k, v in sorted(totals.items(), key=lambda kv: -abs(kv[1])):
        if abs(v) >= 1e-4:
            print(f"  {k:<18} {v:+.3f}")


if __name__ == "__main__":
    main()
