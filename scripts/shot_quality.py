"""
Shot quality and whiff diagnostic: what happens when SensAI hits the ball in the attacking half,
what each kind of touch is paid, and how often it commits to the ball and misses.

Plays kickoff-to-goal games of a checkpoint (blue, deterministic like the live bot) against an
opponent and reports:

  touches      every SensAI touch in the attacking half, split into
                 kickoff         the first touch off the centre spot
                 on_target       ball heading goalward with the trajectory inside the goal mouth
                 wide_backboard  goalward and deep (ball y > 3000) but off target
                 forward_miss    goalward and off target from further out
                 centering       from a wide ball (|x| > 1200) back toward the middle
                 other
               with the open angle to the goal mouth, placement, ball speed, the touch reward and
               discounted ball_to_goal reward it earned, and the outcome over the next 4 s
  distance     on-target shots by distance to goal, and how many of them went in
  whiffs       jumps/flips within 600 uu of the ball with no touch by either car in the next
               10 steps; "aimed" whiffs were driving at the ball and passed within 300 uu

--debias applies ActorCritic.debias_symmetric_actions() after loading, the way bot.py used to on
every load before it switched to sanitize_log_std(), for comparing against that older live policy.

Usage:
    python scripts/shot_quality.py
    python scripts/shot_quality.py --steps 27000 --seed 11
    python scripts/shot_quality.py --debias
    python scripts/shot_quality.py --opponent self
    python scripts/shot_quality.py --checkpoint checkpoints/checkpoint_iter_120000.pt --opponent nexto

--opponent: necto, nexto, self (the same checkpoint as orange), heuristic, or a path to any checkpoint.
Goals and touch counts come from one continuous run; at 15 Hz, 27000 steps is 30 minutes of play.
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

from policy_health import load_agent, load_weights  # noqa: E402
from env.rocket_env import RocketLeagueEnv  # noqa: E402
from env.physics_engine import ARENA_EXTENT_Y  # noqa: E402
from env.rewards import goal_mouth_open_angle, on_target_factor  # noqa: E402

GOAL_Y = ARENA_EXTENT_Y            # blue attacks +Y
OUTCOME_WINDOW_STEPS = 60          # 4 s
WHIFF_RADIUS = 600.0
WHIFF_WINDOW_STEPS = 10
WHIFF_CLOSE_UU = 300.0
CATEGORIES = ("kickoff", "on_target", "wide_backboard", "forward_miss", "centering", "other")
DIST_BANDS = ((0, 1500), (1500, 3000), (3000, 4500), (4500, 6000))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/latest_model.pt")
    p.add_argument("--opponent", default="necto")
    p.add_argument("--steps", type=int, default=20000, help="policy steps (15 Hz)")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--debias", action="store_true", help="apply the destructive debias_symmetric_actions() bot.py used to run on load")
    return p.parse_args()


def resolve_opponent(name):
    key = name.lower()
    if key == "necto":
        return "checkpoints/necto-model.pt"
    if key == "nexto":
        return "checkpoints/nexto-model.pt"
    return key if key in ("self", "heuristic") else name


def classify(pre_ball, ball_pos, ball_vel, placement):
    if abs(pre_ball[0]) < 200.0 and abs(pre_ball[1]) < 200.0 and pre_ball[2] < 150.0:
        return "kickoff"
    if ball_vel[1] > 300.0:
        if placement >= 0.5:
            return "on_target"
        return "wide_backboard" if ball_pos[1] > 3000.0 else "forward_miss"
    if abs(ball_pos[0]) > 1200.0 and ball_vel[0] * np.sign(ball_pos[0]) < -300.0:
        return "centering"
    return "other"


class Game:
    def __init__(self, agent, weights, opponent):
        self.agent = agent
        self.gamma = float(weights["gamma"])  # the trainer's discount, from config
        self.selfplay = opponent == "self"
        self.env = RocketLeagueEnv(
            game_mode="1v1", max_episode_steps=10 ** 9, reward_weights=weights,
            is_baseline_env=not self.selfplay,
            baseline_opponent_type="heuristic" if self.selfplay else opponent,
        )

    def kickoff(self):
        obs = self.env.reset(random_kickoff=False)
        # No scenario end rules: games run until a goal
        self.env.current_scenario = "shot_quality"
        self.env.scenario_timeout = 10 ** 9
        return obs

    def run(self, steps):
        env = self.env
        obs = self.kickoff()
        arena = env.arena
        events, whiffs, pending, watch = [], [], [], []
        goals = [0, 0]
        touches_total = 0
        prev_touch = [c.ball_touches for c in arena.cars]
        prev_has_flip, prev_on_ground = arena.cars[0].has_flip, arena.cars[0].on_ground

        for t in range(steps):
            with torch.no_grad():
                act = self.agent.get_action_and_value(torch.from_numpy(obs).float(), deterministic=True)[0].numpy()
            pre_ball = arena.ball.pos.copy()
            obs, _, done, info = env.step(act, include_breakdown=True)
            breakdown = info.get("reward_breakdown") or {}
            car, opp = arena.cars[0], arena.cars[1]

            if np.any(done):
                # The env auto-resets on done, clearing its counters, so read the goals from info
                g = info.get("episode_goals", [0, 0])
                goals[0] += g[0]
                goals[1] += g[1]
                outcome = "goal_for" if g[0] > 0 else ("goal_against" if g[1] > 0 else "reset")
                for ev in pending:
                    ev["outcome"] = outcome
                    events.append(ev)
                pending, watch = [], []
                obs = self.kickoff()
                prev_touch = [c.ball_touches for c in arena.cars]
                prev_has_flip, prev_on_ground = arena.cars[0].has_flip, arena.cars[0].on_ground
                continue

            sensai_touched = car.ball_touches > prev_touch[0]
            opp_touched = opp.ball_touches > prev_touch[1]
            prev_touch = [car.ball_touches, opp.ball_touches]
            ball_pos, ball_vel = arena.ball.pos, arena.ball.vel
            dist = float(np.linalg.norm(ball_pos - car.pos))

            still = []
            for ev in pending:
                ev["b2g"] += self.gamma ** (t - ev["t"]) * breakdown.get("ball_to_goal", 0.0)
                if opp_touched:
                    ev["outcome"] = "opp_touch"
                elif sensai_touched:
                    ev["outcome"] = "own_followup"
                elif t - ev["t"] >= OUTCOME_WINDOW_STEPS:
                    ev["outcome"] = "timeout"
                if ev["outcome"] is None:
                    still.append(ev)
                else:
                    events.append(ev)
            pending = still

            if sensai_touched:
                touches_total += 1
                for w in watch:
                    w["touched"] = True
                if ball_pos[1] > 0.0:
                    placement = on_target_factor(arena, GOAL_Y)
                    pending.append(dict(
                        t=t, x=float(pre_ball[0]), y=float(pre_ball[1]),
                        goal_dist=float(math.hypot(pre_ball[0], GOAL_Y - pre_ball[1])),
                        angle=goal_mouth_open_angle(pre_ball, GOAL_Y),
                        placement=placement, speed=float(np.linalg.norm(ball_vel)),
                        cat=classify(pre_ball, ball_pos, ball_vel, placement),
                        touch=breakdown.get("touch", 0.0), b2g=breakdown.get("ball_to_goal", 0.0),
                        outcome=None,
                    ))

            flipped = prev_has_flip and not car.has_flip and not car.on_ground
            jumped = prev_on_ground and not car.on_ground
            prev_has_flip, prev_on_ground = car.has_flip, car.on_ground
            if (flipped or jumped) and dist < WHIFF_RADIUS:
                speed = max(1.0, float(np.linalg.norm(car.vel)))
                watch.append(dict(
                    t=t, kind="flip" if flipped else "jump", y=float(ball_pos[1]), z=float(ball_pos[2]),
                    dist=dist, min_dist=dist, touched=sensai_touched, opp_touched=False,
                    aim=float(np.dot(car.vel / speed, (ball_pos - car.pos) / max(1.0, dist))),
                ))
            keep = []
            for w in watch:
                w["opp_touched"] = w["opp_touched"] or (opp_touched and not w["touched"])
                w["min_dist"] = min(w["min_dist"], dist)
                if t - w["t"] < WHIFF_WINDOW_STEPS:
                    keep.append(w)
                elif not w["touched"] and not w["opp_touched"]:
                    whiffs.append(w)
            watch = keep

        for ev in pending:
            ev["outcome"] = "unresolved"
            events.append(ev)
        return goals, touches_total, events, whiffs


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def report(goals, touches_total, events, whiffs):
    print(f"\ngoals for/against {tuple(goals)}  |  SensAI touches {touches_total}  |  attacking-half touches {len(events)}")

    print("\nattacking-half touches by category:")
    print(f"  {'category':16s} {'n':>4s} {'angle':>6s} {'place':>6s} {'speed':>6s} {'touch_r':>8s} {'b2g_r':>7s}  outcomes")
    for cat in CATEGORIES:
        rows = [e for e in events if e["cat"] == cat]
        if rows:
            print(f"  {cat:16s} {len(rows):4d} {mean([e['angle'] for e in rows]):6.1f} "
                  f"{mean([e['placement'] for e in rows]):6.2f} {mean([e['speed'] for e in rows]):6.0f} "
                  f"{mean([e['touch'] for e in rows]):8.3f} {mean([e['b2g'] for e in rows]):7.3f}  "
                  f"{dict(collections.Counter(e['outcome'] for e in rows))}")

    tight = [e for e in events if e["y"] > 3500 and e["angle"] < 20]
    if tight:
        print(f"\ntight-angle touches (ball y > 3500, < 20 deg of goal mouth visible): {len(tight)}")
        for cat in CATEGORIES:
            rows = [e for e in tight if e["cat"] == cat]
            if rows:
                print(f"  {cat:16s} {len(rows):4d}  touch_r {mean([e['touch'] for e in rows]):.3f}  "
                      f"b2g_r {mean([e['b2g'] for e in rows]):.3f}  "
                      f"outcomes {dict(collections.Counter(e['outcome'] for e in rows))}")

    shots = [e for e in events if e["cat"] == "on_target" and e["outcome"] in ("goal_for", "opp_touch")]
    if shots:
        print("\non-target shots by distance to goal centre (goal or opponent touch):")
        for lo, hi in DIST_BANDS:
            rows = [e for e in shots if lo <= e["goal_dist"] < hi]
            if rows:
                scored = sum(e["outcome"] == "goal_for" for e in rows)
                print(f"  {lo:4d}-{hi:4d} uu: {len(rows):3d} shots  goals {scored:3d} ({100.0 * scored / len(rows):.0f}%)  "
                      f"mean speed {mean([e['speed'] for e in rows]):.0f}  touch_r {mean([e['touch'] for e in rows]):.2f}")
    misses = [e for e in events if e["cat"] in ("forward_miss", "wide_backboard")]
    if misses:
        print("off-target goalward touches by distance: " + "  ".join(
            f"{lo}-{hi}: {sum(lo <= e['goal_dist'] < hi for e in misses)}" for lo, hi in DIST_BANDS))

    print(f"\nwhiffs (jump/flip within {WHIFF_RADIUS:.0f} uu of the ball, no touch by either car in "
          f"{WHIFF_WINDOW_STEPS} steps): {len(whiffs)}  per 100 SensAI touches {100.0 * len(whiffs) / max(1, touches_total):.1f}")
    for kind in ("flip", "jump"):
        rows = [w for w in whiffs if w["kind"] == kind]
        if rows:
            print(f"  {kind}: {len(rows)}  attacking half {sum(w['y'] > 0 for w in rows)}  "
                  f"mean dist {mean([w['dist'] for w in rows]):.0f}  ball z<200: {sum(w['z'] < 200 for w in rows)}  "
                  f"z 200-600: {sum(200 <= w['z'] < 600 for w in rows)}  z>=600: {sum(w['z'] >= 600 for w in rows)}")
    if whiffs:
        aimed = [w for w in whiffs if w["aim"] > 0.7 and w["min_dist"] < WHIFF_CLOSE_UU]
        print(f"  aimed at the ball and passed within {WHIFF_CLOSE_UU:.0f} uu without touching: {len(aimed)}  "
              f"(flip {sum(w['kind'] == 'flip' for w in aimed)}, jump {sum(w['kind'] == 'jump' for w in aimed)}, "
              f"ball z<200 {sum(w['z'] < 200 for w in aimed)})")
        print(f"  never got within {WHIFF_CLOSE_UU:.0f} uu: {sum(w['min_dist'] >= WHIFF_CLOSE_UU for w in whiffs)}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent, ckpt = load_agent(args.checkpoint)
    it = ckpt.get("iteration", -1)
    if args.debias:
        agent.debias_symmetric_actions()
    weights = load_weights(ckpt)
    opponent = resolve_opponent(args.opponent)

    print(f"checkpoint {args.checkpoint} (iteration {it})  opponent {args.opponent}  steps {args.steps}"
          f"{'  [debiased, as bot.py used to load it]' if args.debias else ''}")
    report(*Game(agent, weights, opponent).run(args.steps))


if __name__ == "__main__":
    main()
