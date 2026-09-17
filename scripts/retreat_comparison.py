"""
Midfield retreat comparison: does dropping back from a free side-wall ball actually pay?

Spawns free-ball setups (a slow ball near a side wall in midfield, SensAI 400-900 uu from it
and roughly goalside) and plays each one twice from the identical start:

  natural      the policy plays from the first step
  chase-first  BaselineChaser drives SensAI straight at the ball for the first --force-steps
               steps, then the policy takes over

A run counts as a "retreat" when the natural policy drops more than 800 uu toward its own goal
within the first 3 s without touching the ball in the first 2 s.

It reports the retreat rate, the discounted return of both variants on the retreat cases, goals
for/against, the retreat rate by starting boost, and which reward terms account for the
difference. If the retreat is being learned out, the retreat rate falls and chase-first starts
winning the comparison.

Usage:
    python scripts/retreat_comparison.py
    python scripts/retreat_comparison.py --opponent nexto
    python scripts/retreat_comparison.py --opponent self --trials 100
    python scripts/retreat_comparison.py --checkpoint checkpoints/checkpoint_iter_107000.pt --opponent checkpoints/checkpoint_iter_104000.pt

--opponent: idle (parked far upfield, no inputs), nexto, necto, heuristic, self (the same
checkpoint), or a path to any checkpoint.
"""
import argparse
import collections
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

import RocketSim as rs  # noqa: E402

from policy_health import load_agent, load_weights  # noqa: E402
from env.rocket_env import RocketLeagueEnv  # noqa: E402
from env.baseline_agent import BaselineChaser, create_opponent_bot  # noqa: E402

RETREAT_DROP_UU = 800.0
RETREAT_WINDOW_STEPS = 45      # 3 s at 15 Hz
RETREAT_TOUCH_GRACE_STEPS = 30  # a touch within 2 s means it went for the ball


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/latest_model.pt")
    p.add_argument("--opponent", default="idle")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--steps", type=int, default=150, help="policy steps per rollout (15 Hz)")
    p.add_argument("--force-steps", type=int, default=20, help="chase-first steps before the policy takes over")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def resolve_opponent(name, checkpoint):
    key = name.lower()
    if key == "idle":
        return None
    if key == "nexto":
        return "checkpoints/nexto-model.pt"
    if key == "necto":
        return "checkpoints/necto-model.pt"
    if key == "heuristic":
        return "heuristic"
    if key == "self":
        return checkpoint
    return name


def sample_setup(rng, opponent_active):
    side = rng.choice([-1, 1])
    bx = side * rng.uniform(2200, 3700)
    by = rng.uniform(-300, 2000)
    ang = rng.uniform(0, 2 * math.pi)
    d = rng.uniform(400, 900)
    cx = float(np.clip(bx + d * math.cos(ang), -3900, 3900))
    cy = by - abs(d * math.sin(ang)) * rng.uniform(0.2, 1.0)
    if opponent_active:
        ox, oy = rng.uniform(-1500, 1500), rng.uniform(2500, 4000)
    else:
        ox, oy = rng.uniform(-2000, 2000), rng.uniform(3800, 5000)
    return dict(
        bx=bx, by=by, bvx=rng.uniform(-300, 300), bvy=rng.uniform(-300, 300),
        cx=cx, cy=cy, yaw=rng.uniform(-math.pi, math.pi), spd=rng.uniform(600, 1400),
        boost=rng.uniform(0, 100), ox=ox, oy=oy,
    )


class Harness:
    def __init__(self, agent, weights, opponent_path):
        self.agent = agent
        self.gamma = float(weights["gamma"])  # the trainer's discount, from config
        self.opponent_path = opponent_path
        self.env = RocketLeagueEnv(
            game_mode="1v1", max_episode_steps=100000, reward_weights=weights,
            is_baseline_env=opponent_path is not None,
        )
        self.chaser = BaselineChaser()

    def _fresh_opponent(self):
        # A new bot per rollout, so opponent-side action memory never leaks between variants
        if self.opponent_path is None:
            self.env.baseline_bot = None
        elif self.opponent_path == "heuristic":
            self.env.baseline_bot = BaselineChaser()
        else:
            self.env.baseline_bot = create_opponent_bot(self.opponent_path)

    def setup(self, P):
        env = self.env
        env.reset(random_kickoff=False)
        self._fresh_opponent()
        a = env.arena
        bs = a._rsim_arena.ball.get_state()
        bs.pos = rs.Vec(P["bx"], P["by"], 93)
        bs.vel = rs.Vec(P["bvx"], P["bvy"], 0)
        bs.ang_vel = rs.Vec(0, 0, 0)
        a._rsim_arena.ball.set_state(bs)
        cars = a._rsim_arena.get_cars()
        cs = cars[0].get_state()
        cs.pos = rs.Vec(P["cx"], P["cy"], 17)
        cs.vel = rs.Vec(P["spd"] * math.cos(P["yaw"]), P["spd"] * math.sin(P["yaw"]), 0)
        cs.rot_mat = rs.Angle(yaw=P["yaw"]).as_rot_mat()
        cs.ang_vel = rs.Vec(0, 0, 0)
        cs.boost = P["boost"]
        cars[0].set_state(cs)
        cs = cars[1].get_state()
        cs.pos = rs.Vec(P["ox"], P["oy"], 17)
        cs.vel = rs.Vec(0, 0, 0)
        cs.rot_mat = rs.Angle(yaw=-math.pi / 2).as_rot_mat()
        cs.boost = 33
        cars[1].set_state(cs)
        a._sync_from_rsim()
        env.reward_manager.reset(a)
        # env.reset() rolled a random training scenario, and its timeout / "resolved" rules (kickoff
        # stall, aerial whiff, save resolved...) would end these rollouts early. None apply here.
        env.current_scenario = "retreat_comparison"
        env.scenario_timeout = 10 ** 9
        env.current_step = 0
        env.episode_goals = [0, 0]
        for c in a.cars:
            c.ball_touches = 0
        return np.array([env.obs_builder.build_obs(c, a) for c in a.cars], dtype=np.float32)

    def rollout(self, P, steps, force_steps):
        obs = self.setup(P)
        env, a = self.env, self.env.arena
        ret, g = 0.0, 1.0
        min_y = P["cy"]
        first_touch = None
        terms = collections.Counter()
        for t in range(steps):
            with torch.no_grad():
                act = self.agent.get_action_and_value(torch.from_numpy(obs).float(), deterministic=True)[0].numpy()
            if t < force_steps:
                act[0] = self.chaser.get_action(a.cars[0], a)
            if self.opponent_path is None:
                act[1] = 0.0
            obs, r, done, info = env.step(act, include_breakdown=True)
            for k, v in (info.get("reward_breakdown") or {}).items():
                terms[k] += g * v
            ret += g * float(r[0])
            g *= self.gamma
            if t < RETREAT_WINDOW_STEPS:
                min_y = min(min_y, float(a.cars[0].pos[1]))
            if first_touch is None and a.cars[0].ball_touches > 0:
                first_touch = t
            if np.any(done):
                # The env auto-resets on done, clearing its counters, so read the goals from info
                goals = info.get("episode_goals", env.episode_goals)
                break
        else:
            goals = env.episode_goals
        return dict(
            ret=ret, drop=P["cy"] - min_y, first_touch=first_touch, terms=terms,
            goals_for=goals[0], goals_against=goals[1], steps=t + 1,
        )


def main():
    args = parse_args()
    agent, ckpt = load_agent(args.checkpoint)
    it = ckpt.get("iteration", -1)
    weights = load_weights(ckpt)
    opponent_path = resolve_opponent(args.opponent, args.checkpoint)
    harness = Harness(agent, weights, opponent_path)
    rng = np.random.default_rng(args.seed)

    print(f"checkpoint {args.checkpoint} (iteration {it})  opponent {args.opponent}  trials {args.trials}")

    rows = []
    for _ in range(args.trials):
        P = sample_setup(rng, opponent_path is not None)
        nat = harness.rollout(P, args.steps, 0)
        cha = harness.rollout(P, args.steps, args.force_steps)
        retreat = nat["drop"] > RETREAT_DROP_UU and (
            nat["first_touch"] is None or nat["first_touch"] > RETREAT_TOUCH_GRACE_STEPS
        )
        rows.append((retreat, P, nat, cha))

    R = [r for r in rows if r[0]]
    N = [r for r in rows if not r[0]]

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    def touch_step(res):
        return res["first_touch"] if res["first_touch"] is not None else args.steps

    def goals(subset, key):
        return (sum(r[2 if key == "nat" else 3]["goals_for"] for r in subset),
                sum(r[2 if key == "nat" else 3]["goals_against"] for r in subset))

    print(f"\nretreat rate: {len(R)}/{len(rows)} ({100.0 * len(R) / max(1, len(rows)):.0f}%)")
    if R:
        print(f"retreat cases:  natural return {mean([r[2]['ret'] for r in R]):+.2f}  |  "
              f"chase-first {mean([r[3]['ret'] for r in R]):+.2f}  |  "
              f"chase-first better in {sum(r[3]['ret'] > r[2]['ret'] for r in R)}/{len(R)}")
        print(f"                first touch step natural {mean([touch_step(r[2]) for r in R]):.0f}  vs  "
              f"chase-first {mean([touch_step(r[3]) for r in R]):.0f}")
        print(f"                goals for/against natural {goals(R, 'nat')}  chase-first {goals(R, 'cha')}")
    if N:
        print(f"other cases:    natural return {mean([r[2]['ret'] for r in N]):+.2f}  |  "
              f"chase-first {mean([r[3]['ret'] for r in N]):+.2f}")
    print(f"all cases:      goals for/against natural {goals(rows, 'nat')}  chase-first {goals(rows, 'cha')}")

    bands = ((0, 25), (25, 50), (50, 75), (75, 101))
    print("retreat rate by starting boost: " + "  ".join(
        f"{lo}-{min(hi, 100)}: {sum(r[0] for r in rows if lo <= r[1]['boost'] < hi)}/"
        f"{sum(1 for r in rows if lo <= r[1]['boost'] < hi)}"
        for lo, hi in bands))

    if R:
        nat_terms, cha_terms = collections.Counter(), collections.Counter()
        for r in R:
            nat_terms.update(r[2]["terms"])
            cha_terms.update(r[3]["terms"])
        n = len(R)
        print("\ndiscounted reward terms on retreat cases, per case (natural - chase-first):")
        for k in sorted(set(nat_terms) | set(cha_terms), key=lambda k: -abs(nat_terms[k] - cha_terms[k])):
            diff = (nat_terms[k] - cha_terms[k]) / n
            if abs(diff) > 0.005:
                print(f"  {k:20s} natural {nat_terms[k] / n:+.3f}  chase-first {cha_terms[k] / n:+.3f}  diff {diff:+.3f}")


if __name__ == "__main__":
    main()
