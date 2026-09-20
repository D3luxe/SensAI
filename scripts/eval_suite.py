"""
Evaluation suite: a fixed, reward-independent scorecard for a checkpoint.

Every number here is measured on the pitch -- goals, touches, positions, speeds, boost, landings --
and none reads a reward term, so checkpoints trained on different reward versions are compared on
equal terms. The suite itself is versioned (SUITE_VERSION); results from different suite versions
are not compared.

  matches     SensAI (blue, deterministic, as the live bot plays) against the real Necto -- flip
              flag and scripted kickoff as it runs in RLBot -- over several fixed seeds, plus
              optionally Nexto and a pinned reference checkpoint. Reports the result, kickoffs,
              shooting, whiffs and overshoots, defending (goals against by cause, retreat dodges,
              back-wall climbs), boost use and landing quality.
  scenarios   Seeded, randomised starts against Necto that make specific situations common:
                drop     ball falling from 800-1200 uu near SensAI
                bounce   ball bouncing toward SensAI
                wall     ball running up a side wall ahead of SensAI
                retreat  ball rolling toward our net with SensAI upfield of it; the opponent sits idle
                         in its own goal, so the outcome is SensAI's retreat alone -- mostly saveable
                         with boost, borderline by flip-chaining at ~1500 uu/s
              Each reports outcomes only (first touch, ran under the ball, touch quality, saves,
              boost used, dodges, landings).

Usage:
    python scripts/eval_suite.py                                  # latest_model.pt -> evals/iter<N>.json
    python scripts/eval_suite.py --checkpoint checkpoints/checkpoint_iter_170000.pt --name v2_170k
    python scripts/eval_suite.py --quick                          # smoke test, ~1/4 size
    python scripts/eval_suite.py --reference checkpoints/checkpoint_iter_142400.pt
    python scripts/eval_suite.py --compare evals/v2_170k.json evals/v3_20k.json

Seed-to-seed spread is reported with every match metric; a difference smaller than that spread
is not a difference.
"""
import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
os.environ.setdefault("RS_COLLISION_MESHES", os.path.join(ROOT, "collision_meshes"))

import numpy as np  # noqa: E402

SUITE_VERSION = 1
STEPS_PER_MIN = 15 * 60
MATCH_SEEDS = (11, 12, 13)
MATCH_STEPS = 12000                      # 13.3 minutes of play per seed
SCENARIO_TRIALS = 24
SCENARIO_SEED = 1000
NECTO = "checkpoints/necto-model.pt"
NEXTO = "checkpoints/nexto-model.pt"

# Landing and retreat definitions shared by matches and scenarios
FLIGHT_MIN_STEPS = 5                     # airborne >= 0.33 s counts as a flight (not a bump)
WHEELS_DOWN_UP_Z = 0.7                   # up-vector z at touchdown; below this: door or roof
SETTLE_STEPS = 5                         # speed retained is read 0.33 s after touchdown
SUPERSONIC = 2200.0


# ---------------------------------------------------------------------------------------------
# Per-step motion statistics (boost, air time, landings, retreat boosting)
# ---------------------------------------------------------------------------------------------
class MotionStats:
    """
    Tracks blue each step:
      boost       mean held, share of steps empty, spent per minute, share of steps supersonic
      retreat     steps upfield of a ball that is in our half or heading for it; of those with
                  boost to spend (>= 12), the share on which boost was actually burnt
      landings    every flight of >= FLIGHT_MIN_STEPS: wheels-down at touchdown, and horizontal
                  speed SETTLE_STEPS after touchdown relative to speed at takeoff
    """

    def __init__(self):
        self.steps = 0
        self.air_steps = 0
        self.boost_sum = 0.0
        self.empty = 0
        self.supersonic = 0
        self.spent = 0.0
        self.retreat_steps = 0
        self.retreat_can_boost = 0
        self.retreat_boosting = 0
        self.flights = []                # (wheels_down, speed_kept)
        self._prev_boost = None
        self._flight = None
        self._settling = []
        self._last_ground_speed = None

    def episode_end(self):
        self._prev_boost = None
        self._flight = None
        self._settling = []
        self._last_ground_speed = None

    def __call__(self, t, arena, sensai_touched=False, opp_touched=False):
        car, ball = arena.cars[0], arena.ball
        self.steps += 1
        speed_h = float(np.hypot(car.vel[0], car.vel[1]))
        boost = float(car.boost)
        self.boost_sum += boost
        self.empty += boost < 1.0
        self.supersonic += float(np.linalg.norm(car.vel)) >= SUPERSONIC
        burnt = self._prev_boost is not None and boost < self._prev_boost - 0.05
        if burnt:
            self.spent += self._prev_boost - boost

        ball_y, ball_vy = float(ball.pos[1]), float(ball.vel[1])
        threatened = ball_y < 0.0 or ball_vy < -300.0
        if threatened and float(car.pos[1]) > ball_y + 300.0:
            self.retreat_steps += 1
            if self._prev_boost is not None and self._prev_boost >= 12.0:
                self.retreat_can_boost += 1
                self.retreat_boosting += burnt
        self._prev_boost = boost

        if not car.on_ground:
            self.air_steps += 1
            if self._flight is None:
                self._flight = {"steps": 0, "takeoff": self._last_ground_speed if self._last_ground_speed is not None else speed_h}
            self._flight["steps"] += 1
        else:
            if self._flight is not None and self._flight["steps"] >= FLIGHT_MIN_STEPS:
                up_z = float(car.get_up_vector()[2])
                self._settling.append({"left": SETTLE_STEPS, "takeoff": self._flight["takeoff"],
                                       "wheels_down": up_z >= WHEELS_DOWN_UP_Z})
            self._flight = None
            self._last_ground_speed = speed_h
        keep = []
        for s in self._settling:
            s["left"] -= 1
            if s["left"] <= 0:
                self.flights.append((s["wheels_down"], speed_h / max(300.0, s["takeoff"])))
            else:
                keep.append(s)
        self._settling = keep

    def summary(self):
        n = max(1, self.steps)
        minutes = self.steps / STEPS_PER_MIN
        wd = [f[0] for f in self.flights]
        kept = [f[1] for f in self.flights]
        return {
            "air_time_pct": 100.0 * self.air_steps / n,
            "boost_mean": self.boost_sum / n,
            "boost_empty_pct": 100.0 * self.empty / n,
            "boost_spent_per_min": self.spent / max(1e-6, minutes),
            "supersonic_pct": 100.0 * self.supersonic / n,
            "retreat_time_pct": 100.0 * self.retreat_steps / n,
            "retreat_boosting_pct": 100.0 * self.retreat_boosting / max(1, self.retreat_can_boost),
            "flights_per_min": len(self.flights) / max(1e-6, minutes),
            "landing_not_wheels_down_pct": 100.0 * (1.0 - float(np.mean(wd))) if wd else float("nan"),
            "landing_speed_kept_median": float(np.median(kept)) if kept else float("nan"),
        }


# ---------------------------------------------------------------------------------------------
# Jobs (run in worker processes)
# ---------------------------------------------------------------------------------------------
def _load(checkpoint):
    import torch
    torch.set_num_threads(1)
    from policy_health import load_agent, load_weights
    agent, ckpt = load_agent(checkpoint)
    return agent, ckpt, load_weights(ckpt)


def run_match(job):
    import torch
    from shot_quality import Game
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    agent, ckpt, weights = _load(job["checkpoint"])
    motion = MotionStats()
    goals, touches, events, whiffs, pos, kicks, gas = Game(agent, weights, job["opponent"]).run(job["steps"], observer=motion)
    minutes = job["steps"] / STEPS_PER_MIN
    per100 = 100.0 / max(1, touches)

    shots = [e for e in events if e["cat"] == "on_target"]
    shots_resolved = [e for e in shots if e["outcome"] in ("goal_for", "opp_touch")]
    aimed = [w for w in whiffs if w["aim"] > 0.7 and w["min_dist"] < 300.0]
    overs_aimed = [o for o in pos.overshoots if o["aim"] > 0.7]
    kicks = [k for k in kicks if k["first"] is not None]
    landed = [d for d in pos.dodges if d["landing_wall"] is not None]
    causes = collections.Counter(g["cause"] for g in gas)
    out = {
        "goals_for": goals[0], "goals_against": goals[1],
        "goal_diff_per_10min": (goals[0] - goals[1]) / minutes * 10.0,
        "goals_for_per_10min": goals[0] / minutes * 10.0,
        "goals_against_per_10min": goals[1] / minutes * 10.0,
        "touches_per_min": touches / minutes,
        "on_target_per_100_touches": len(shots) * per100,
        "shot_conversion_pct": 100.0 * sum(e["outcome"] == "goal_for" for e in shots_resolved) / max(1, len(shots_resolved)),
        "whiffs_per_100_touches": len(whiffs) * per100,
        "aimed_whiffs_per_100_touches": len(aimed) * per100,
        "overshoots_aimed_per_100_touches": len(overs_aimed) * per100,
        "kickoffs": len(kicks),
        "kickoff_first_touch_pct": 100.0 * sum(k["first"] in ("sensai", "both") for k in kicks) / max(1, len(kicks)),
        "kickoff_goals_for": sum(k.get("goal") == "goal_for" for k in kicks),
        "kickoff_goals_against": sum(k.get("goal") == "goal_against" for k in kicks),
        "retreat_dodges_per_100_touches": len(pos.dodges) * per100,
        "retreat_dodge_not_wheels_down_pct": 100.0 * sum(d["landing_up"] < WHEELS_DOWN_UP_Z for d in landed) / max(1, len(landed)),
        "backwall_climbs_per_100_touches": len(pos.climbs) * per100,
        "wall_time_pct": 100.0 * sum(pos.wall_steps.values()) / max(1, pos.steps),
    }
    for cause in ("caught upfield", "back-wall climb", "goal-side, far", "beaten goal-side", "own touch last", "off the kickoff"):
        out["ga_cause_" + cause.replace(" ", "_").replace(",", "").replace("-", "_") + "_pct"] = \
            100.0 * causes.get(cause, 0) / max(1, len(gas))
    out.update(motion.summary())
    return {"job": job, "iteration": ckpt.get("iteration"), "metrics": out}


# Scenario generators: every placement is drawn from a seeded RNG, so the same seed gives the same starts
def _clip_y(y):
    return float(np.clip(y, -4700.0, 4700.0))


def gen_drop(rng):
    cx, cy = rng.uniform(-2500, 2500), rng.uniform(-2500, 1500)
    return dict(ball=(cx + rng.uniform(-300, 300), cy + rng.uniform(300, 700), rng.uniform(800, 1200)),
                ball_vel=(0.0, 0.0, -rng.uniform(1, 300)), car=(cx, cy), car_yaw=90 + rng.uniform(-30, 30),
                car_speed=rng.uniform(0, 600), boost=rng.uniform(20, 60),
                opp=(rng.uniform(-1500, 1500), rng.uniform(3800, 4400)), opp_yaw=-90, opp_speed=rng.uniform(0, 300),
                seconds=4.0)


def gen_bounce(rng):
    cx, cy = rng.uniform(-2500, 2500), rng.uniform(-3000, 0)
    by = _clip_y(cy + rng.uniform(1800, 2600))
    bx = cx + rng.uniform(-500, 500)
    return dict(ball=(bx, by, rng.uniform(300, 600)),
                ball_vel=(rng.uniform(-200, 200), -rng.uniform(400, 1000), rng.uniform(200, 500)),
                car=(cx, cy), car_yaw=90 + rng.uniform(-20, 20), car_speed=rng.uniform(300, 900), boost=rng.uniform(20, 60),
                opp=(bx + rng.uniform(-600, 600), _clip_y(by + rng.uniform(1200, 1800))), opp_yaw=-90,
                opp_speed=rng.uniform(300, 800), seconds=4.0)


def gen_wall(rng):
    side = 1.0 if rng.uniform() < 0.5 else -1.0
    by = rng.uniform(-1500, 500)
    bx = side * rng.uniform(2600, 3000)
    cx, cy = side * rng.uniform(1800, 2400), by - rng.uniform(500, 900)
    return dict(ball=(bx, by, 93.0), ball_vel=(side * rng.uniform(700, 1100), rng.uniform(300, 800), 0.0),
                car=(cx, cy), car_yaw=math.degrees(math.atan2(by - cy, bx - cx)), car_speed=rng.uniform(600, 1100),
                boost=rng.uniform(20, 60), opp=(-side * rng.uniform(0, 1500), rng.uniform(3500, 4400)), opp_yaw=-90,
                opp_speed=0.0, seconds=5.0)


def gen_retreat(rng):
    bx, by = rng.uniform(-1200, 1200), rng.uniform(-800, 800)
    gx = rng.uniform(-600, 600)
    d = np.array([gx - bx, -5120.0 - by])
    v = d / np.linalg.norm(d) * rng.uniform(1200, 1600)
    facing_home = rng.uniform() < 0.5
    return dict(ball=(bx, by, 93.0), ball_vel=(float(v[0]), float(v[1]), 0.0),
                car=(bx + rng.uniform(-800, 800), _clip_y(by + rng.uniform(1800, 2800))),
                car_yaw=-90.0 if facing_home else 90.0, car_speed=rng.uniform(400, 1000) if facing_home else rng.uniform(400, 900),
                boost=rng.uniform(30, 50), opp=(rng.uniform(-1500, 1500), 4400.0), opp_yaw=-90, opp_speed=0.0,
                opp_idle=True, seconds=7.0, facing_home=facing_home)


SCENARIOS = {"drop": gen_drop, "bounce": gen_bounce, "wall": gen_wall, "retreat": gen_retreat}


def _place(env, p):
    import RocketSim as rsim
    from drop_ball_scenario import set_car
    env.reset(random_kickoff=False)
    arena = env.arena
    bs = arena._rsim_arena.ball.get_state()
    bs.pos = rsim.Vec(*[float(v) for v in p["ball"]])
    bs.vel = rsim.Vec(*[float(v) for v in p["ball_vel"]])
    bs.ang_vel = rsim.Vec(0.0, 0.0, 0.0)
    arena._rsim_arena.ball.set_state(bs)
    set_car(arena._rsim_cars[0], p["car"], math.radians(p["car_yaw"]), p["car_speed"], p["boost"])
    set_car(arena._rsim_cars[1], p["opp"], math.radians(p["opp_yaw"]), p["opp_speed"], p.get("opp_boost", 50.0))
    arena._sync_from_rsim()
    env.reward_manager.reset(arena)
    env.current_step = 0
    env.current_scenario = "eval"
    env.scenario_timeout = 10 ** 9
    return np.array([env.obs_builder.build_obs(c, arena) for c in arena.cars], dtype=np.float32)


def run_scenario(job):
    import torch
    from env.rocket_env import RocketLeagueEnv
    torch.manual_seed(job["seed"])
    agent, ckpt, weights = _load(job["checkpoint"])
    env = RocketLeagueEnv(game_mode="1v1", max_episode_steps=10 ** 9, reward_weights=weights,
                          is_baseline_env=True, baseline_opponent_type=NECTO)
    rng = np.random.default_rng(job["seed"])
    gen = SCENARIOS[job["scenario"]]
    trials = []
    for _ in range(job["trials"]):
        p = gen(rng)
        obs = _place(env, p)
        arena = env.arena
        motion = MotionStats()
        r = dict(first=None, t_first=None, ran_under=False, touch_speed=None, goalward=None, opp_next=False,
                 goal=None, boost_start=float(arena.cars[0].boost), dodges=0, goalside_t=None,
                 facing_home=p.get("facing_home"))
        touches = [0, 0]
        waiting_opp_next = None
        for t in range(int(p["seconds"] * 15)):
            with torch.no_grad():
                act = agent.get_action_and_value(torch.from_numpy(obs).float(), deterministic=True)[0].numpy()
            obs, _, done, info = env.step(act, opponent_action=np.zeros(8, np.float32) if p.get("opp_idle") else None)
            if np.any(done):
                g = info.get("episode_goals", [0, 0])
                r["goal"] = "for" if g[0] > 0 else ("against" if g[1] > 0 else None)
                break
            car, opp, ball = arena.cars[0], arena.cars[1], arena.ball
            s_t, o_t = car.ball_touches > touches[0], opp.ball_touches > touches[1]
            touches = [car.ball_touches, opp.ball_touches]
            motion(t, arena)
            r["dodges"] += bool(car.just_dodged)
            if r["goalside_t"] is None and float(car.pos[1]) < float(ball.pos[1]) - 100.0:
                r["goalside_t"] = t / 15.0
            if r["first"] is None:
                if s_t or o_t:
                    r["first"] = "sensai" if s_t else "opp"
                    r["t_first"] = t / 15.0
                    if s_t:
                        r["touch_speed"] = float(np.linalg.norm(ball.vel))
                        r["goalward"] = bool(float(ball.vel[1]) > 300.0)
                        waiting_opp_next = t
                else:
                    horiz = float(np.hypot(*(ball.pos[:2] - car.pos[:2])))
                    if horiz < 160.0 and float(ball.pos[2]) - float(car.pos[2]) >= 160.0:
                        r["ran_under"] = True
            elif waiting_opp_next is not None:
                if o_t:
                    r["opp_next"] = True
                    waiting_opp_next = None
                elif s_t or t - waiting_opp_next > 45:
                    waiting_opp_next = None
        r["boost_used"] = max(0.0, r["boost_start"] - float(arena.cars[0].boost))
        m = motion.summary()
        r["retreat_boosting_pct"] = m["retreat_boosting_pct"]
        r["flights"] = len(motion.flights)
        r["bad_landings"] = sum(not f[0] for f in motion.flights)
        trials.append(r)
    return {"job": job, "iteration": ckpt.get("iteration"), "metrics": summarise_scenario(job["scenario"], trials),
            "trials": trials}


def _rate(xs):
    return 100.0 * float(np.mean(xs)) if len(xs) else float("nan")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def summarise_scenario(name, trials):
    s_first = [t for t in trials if t["first"] == "sensai"]
    flights = sum(t["flights"] for t in trials)
    out = {
        "n": len(trials),
        "sensai_first_touch_pct": _rate([t["first"] == "sensai" for t in trials]),
        "opp_first_touch_pct": _rate([t["first"] == "opp" for t in trials]),
        "time_to_first_touch_s": _mean([t["t_first"] for t in s_first]),
        "ran_under_ball_pct": _rate([t["ran_under"] for t in trials]),
        "touch_ball_speed": _mean([t["touch_speed"] for t in s_first]),
        "touch_goalward_pct": _rate([t["goalward"] for t in s_first]),
        "opp_next_touch_pct": _rate([t["opp_next"] for t in s_first]),
        "conceded_pct": _rate([t["goal"] == "against" for t in trials]),
        "scored_pct": _rate([t["goal"] == "for" for t in trials]),
        "boost_used": _mean([t["boost_used"] for t in trials]),
        "dodges_per_trial": _mean([t["dodges"] for t in trials]),
        "bad_landing_pct": 100.0 * sum(t["bad_landings"] for t in trials) / max(1, flights),
    }
    if name == "retreat":
        out["retreat_boosting_pct"] = _mean([t["retreat_boosting_pct"] for t in trials])
        out["reached_goalside_pct"] = _rate([t["goalside_t"] is not None for t in trials])
        out["time_to_goalside_s"] = _mean([t["goalside_t"] for t in trials])
        for label, flag in (("facing_home", True), ("facing_away", False)):
            sub = [t for t in trials if t["facing_home"] == flag]
            out[f"conceded_pct_{label}"] = _rate([t["goal"] == "against" for t in sub])
    return out


def run_job(job):
    t0 = time.time()
    res = run_match(job) if job["kind"] == "match" else run_scenario(job)
    res["seconds"] = time.time() - t0
    return res


# ---------------------------------------------------------------------------------------------
# Driver, report, compare
# ---------------------------------------------------------------------------------------------
def build_jobs(args):
    steps = MATCH_STEPS // 4 if args.quick else MATCH_STEPS
    trials = SCENARIO_TRIALS // 4 if args.quick else SCENARIO_TRIALS
    seeds = MATCH_SEEDS[:1] if args.quick else MATCH_SEEDS
    jobs = [dict(kind="match", group="necto", opponent=NECTO, seed=s, steps=steps, checkpoint=args.checkpoint) for s in seeds]
    if not args.no_nexto:
        jobs.append(dict(kind="match", group="nexto", opponent=NEXTO, seed=seeds[0], steps=steps, checkpoint=args.checkpoint))
    if args.reference:
        jobs += [dict(kind="match", group="reference", opponent=args.reference, seed=s, steps=steps,
                      checkpoint=args.checkpoint) for s in seeds]
    for i, name in enumerate(SCENARIOS):
        jobs.append(dict(kind="scenario", group="scenario_" + name, scenario=name, seed=SCENARIO_SEED + i,
                         trials=trials, checkpoint=args.checkpoint))
    return jobs, dict(match_steps=steps, scenario_trials=trials, seeds=list(seeds))


def aggregate(results):
    groups = collections.defaultdict(list)
    for r in results:
        groups[r["job"]["group"]].append(r)
    out = {}
    for g, rs in groups.items():
        keys = rs[0]["metrics"].keys()
        out[g] = {}
        for k in keys:
            vals = [r["metrics"][k] for r in rs]
            finite = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
            out[g][k] = {"mean": float(np.mean(finite)) if finite else float("nan"),
                         "min": float(min(finite)) if finite else float("nan"),
                         "max": float(max(finite)) if finite else float("nan"),
                         "per_seed": vals}
    return out


def _fmt(v):
    if isinstance(v, float) and math.isnan(v):
        return "   -  "
    return f"{v:7.2f}" if abs(v) < 100 else f"{v:7.0f}"


def print_report(summary):
    for g, metrics in summary["results"].items():
        print(f"\n== {g} ==")
        for k, s in metrics.items():
            spread = f"  [{_fmt(s['min']).strip()} .. {_fmt(s['max']).strip()}]" if len(s["per_seed"]) > 1 else ""
            print(f"  {k:42s} {_fmt(s['mean'])}{spread}")


def compare(path_a, path_b):
    a, b = (json.load(open(p, encoding="utf-8")) for p in (path_a, path_b))
    if a.get("suite_version") != b.get("suite_version"):
        print(f"WARNING: suite versions differ ({a.get('suite_version')} vs {b.get('suite_version')}); not comparable")
    for x, p in ((a, path_a), (b, path_b)):
        print(f"{p}: iteration {x.get('iteration')}  reward {x.get('reward_identity')}")
    print("\nA difference is marked * when the ranges over seeds do not overlap (matches), or when it exceeds")
    print("two binomial standard errors (scenario rates). Everything else is inside the noise.")
    from utils.eval_results import compare_results   # one noise rule, shared with the UI
    group = None
    for r in compare_results(a, b):
        if r["group"] != group:
            group = r["group"]
            print(f"\n== {group} ==")
        d = r["delta"]
        print(f"  {r['metric']:42s} {_fmt(r['a'])} -> {_fmt(r['b'])}   {'+' if d >= 0 else ''}{_fmt(d).strip():>7s} {'*' if r['clear'] else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/latest_model.pt")
    p.add_argument("--name", default=None,
                   help="output name (default <version>_<M>M: the reward version and millions of steps into its run)")
    p.add_argument("--reference", default=None, help="also play this checkpoint head to head")
    p.add_argument("--no-nexto", action="store_true")
    p.add_argument("--quick", action="store_true", help="one seed, quarter size: a smoke test, not a result")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
        return

    import torch
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = {
        "suite_version": SUITE_VERSION,
        "checkpoint": args.checkpoint,
        "iteration": ckpt.get("iteration"),
        "global_step": ckpt.get("global_step"),
        "reward_identity": ckpt.get("reward_identity", "unstamped"),
        "reward_run_start_step": ckpt.get("reward_run_start_step"),
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "quick": bool(args.quick),
        # the head-to-head opponent, so a result can name who the "reference" group was played against
        "reference": args.reference,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    del ckpt
    jobs, sizes = build_jobs(args)
    meta.update(sizes)
    print(f"eval suite v{SUITE_VERSION}: {args.checkpoint} (iteration {meta['iteration']}), {len(jobs)} jobs on {args.workers} workers")
    t0 = time.time()
    with mp.get_context("spawn").Pool(processes=max(1, args.workers)) as pool:
        results = []
        for r in pool.imap_unordered(run_job, jobs):
            results.append(r)
            print(f"  done {r['job']['group']:<18s} seed {r['job']['seed']:<5d} {r['seconds']:6.0f}s")
    summary = {**meta, "wall_seconds": time.time() - t0, "results": aggregate(results),
               "raw": [{"job": r["job"], "metrics": r["metrics"]} for r in results]}
    os.makedirs("evals", exist_ok=True)
    stamp = meta["reward_identity"] if isinstance(meta["reward_identity"], dict) else {}
    if args.name:
        name = args.name
    elif stamp.get("version") and meta["reward_run_start_step"] is not None:
        # steps into this reward version's run, the clock the spec's decision points use
        name = f"{stamp['version']}_{(meta['global_step'] - meta['reward_run_start_step']) // 1_000_000}M"
    else:
        name = f"iter{meta['iteration']}"
    if args.quick:
        name += "_quick"
    out = os.path.join("evals", name + ".json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print_report(summary)
    print(f"\nwrote {out}  ({summary['wall_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
