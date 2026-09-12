"""
Rollout health check for the current policy: engagement, air control, reward mix.

These are the numbers that told us what was actually wrong, and they are all dense, so a
single run of a few minutes gives a usable reading. Run it every thousand iterations or so
and watch the trend rather than any single value.

    python scripts/policy_health.py                   # latest_model.pt, ~4 min
    python scripts/policy_health.py --quick           # fewer steps, noisier, ~1 min
    python scripts/policy_health.py --steps 2500      # tighter numbers, slower
    python scripts/policy_health.py checkpoints/checkpoint_iter_9000.pt
    python scripts/policy_health.py --history         # print the log and exit

Every run appends a row to logs/policy_health.csv so the trend survives across sessions.

WHAT TO LOOK AT

  touches/ep and zero-touch %   The engagement metrics. Dense, low variance, and the ones
                                that actually track whether the bot plays the game.
  airborne % and ang vel        Air control. Read them TOGETHER. Angular velocity is measured
                                only over airborne steps, so a policy that avoids flying can
                                show a falling airborne share while its actual air control is
                                unchanged. Falling ang vel with a steady airborne share is
                                improvement; falling airborne share with flat ang vel is
                                avoidance.
  pitch>0.9                     Fraction of airborne steps with saturated pitch. This is the
                                channel that drove the tumble.
  reward breakdown              Which terms are paying. A cost term shrinking can mean the
                                behaviour improved OR that the state it charges stopped
                                occurring; the airborne share tells you which.

WHAT NOT TO READ

  goal /ep                      Goals land at plus or minus 30 on roughly 0.2 goals per
                                episode, so at ~100 episodes the standard error is around 1.4.
                                Swings of a few points between runs are noise. It is printed
                                with its standard error for exactly that reason. Use touches.

TWO TRAPS THIS SCRIPT AVOIDS, both of which produced wrong answers before:

  1. Assigning to car.pos or ball.pos does nothing. RocketSim owns the state and overwrites
     it on the next step, so scripted scenarios are impossible this way and every number here
     comes from unmodified rollouts.
  2. car.ball_touches is CUMULATIVE. Testing it for greater than 0 counts every step after
     the first touch as a touch; only its increase means a new one.
"""
import argparse
import collections
import io
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HZ = 15.0
CSV = os.path.join(ROOT, "logs", "policy_health.csv")
COLUMNS = [
    "timestamp", "iteration", "episodes", "ep_len_steps", "touches_per_ep",
    "zero_touch_pct", "airborne_pct", "ang_vel", "spin_gt2_pct", "pitch_sat_pct",
    "upright_pct", "tumble_pct", "reverse_pct", "turnarounds_per_1k",
    "goal_per_ep", "goal_se",
]


def load_weights(global_step=None):
    """Reward weights exactly as the trainer applies them.

    Three layers, and missing the third made every earlier reading of this wrong. The yaml is
    the base, live_config.json overrides it at runtime, and then reward_annealing decays the
    named targets toward their goals over decay_steps. Reading only the first two reports the
    weight a term STARTED with. At global_step 392M of a 400M schedule that put powerslide at
    its config 0.2 when training was actually applying 0.0039, a factor of 51.
    """
    cfg = yaml.safe_load(io.open(os.path.join(ROOT, "config/default_config.yaml"),
                                 encoding="utf-8").read())
    rw = dict(cfg.get("rewards", {}))
    try:
        live = json.load(io.open(os.path.join(ROOT, "config/live_config.json"),
                                 encoding="utf-8"))
        rw.update(live.get("rewards", {}))
    except Exception as e:
        print("warning: could not read live_config.json (%s); using yaml weights only" % e)

    anneal = cfg.get("reward_annealing", {}) or {}
    notes = []
    if anneal.get("enabled") and global_step is not None:
        decay = max(1, int(anneal.get("decay_steps", 300000000)))
        progress = min(1.0, max(0.0, float(global_step) / decay))
        for key, target in (anneal.get("targets", {}) or {}).items():
            start = float(rw.get(key, 0.0))
            rw[key] = start + (float(target) - start) * progress
            notes.append("%s %.4f (from %.3f, %.0f%% annealed)"
                         % (key, rw[key], start, 100.0 * progress))
    if notes:
        print("annealed weights: " + " | ".join(notes))
    return rw


def load_agent(path):
    from agent.models import ActorCritic
    ck = torch.load(path, map_location="cpu", weights_only=False)
    agent = ActorCritic(obs_dim=ck["obs_dim"], act_dim=ck["act_dim"],
                        continuous_actions=ck["continuous_actions"],
                        use_layer_norm=ck["use_layer_norm"], activation=ck["activation"])
    agent.load_state_dict(ck["model_state_dict"])
    agent.eval()
    return agent, ck.get("iteration", -1), ck.get("global_step", None)


def run(agent, rw, n_envs, n_steps):
    from env.rocket_env import RocketLeagueEnv

    envs = [RocketLeagueEnv(game_mode="1v1", max_episode_steps=600, reward_weights=rw)
            for _ in range(n_envs)]
    obs = [e.reset() for e in envs]

    terms = collections.Counter()
    ep_goal = [0.0] * n_envs
    goal_totals = []
    ep_len = [0] * n_envs
    ep_touch = [0] * n_envs
    prev_t = [0] * n_envs
    lengths, touches = [], []

    air = upright = spin_gt2 = pitch_sat = 0
    air_steps = 0
    av_sum = 0.0
    b2g = {"ours": 0.0, "theirs": 0.0, "none": 0.0}
    owner = [None] * n_envs
    touch_seen = [dict() for _ in range(n_envs)]
    total_steps = 0
    grounded_steps = rev_grounded = turnarounds = 0
    was_dodging = [False] * n_envs

    for _ in range(n_steps):
        batch = np.concatenate(obs, axis=0)
        with torch.no_grad():
            a, _, _, _ = agent.get_action_and_value(torch.from_numpy(batch).float())
        a = a.numpy().reshape(n_envs, envs[0].num_players, -1)
        for i, e in enumerate(envs):
            c = e.arena.cars[0]
            airborne = not c.on_ground
            if airborne:
                air += 1
                av = float(np.linalg.norm(c.ang_vel))
                av_sum += av
                air_steps += 1
                if av > 2.0:
                    spin_gt2 += 1
                if abs(float(a[i][0][2])) > 0.9:
                    pitch_sat += 1
            if float(c.get_up_vector()[2]) > 0.7:
                upright += 1

            # Reverse pursuit and turnarounds. With the half-flip incentives removed, driving
            # backwards became the cheap way to chase a ball behind the car.
            fwd_speed = float(np.dot(c.vel, c.get_forward_vector()))
            if not airborne:
                grounded_steps += 1
                if fwd_speed < -100.0:
                    rev_grounded += 1
            dodging = bool(getattr(c, "is_dodging", False))
            if dodging and not was_dodging[i] and fwd_speed < -100.0:
                turnarounds += 1
            was_dodging[i] = dodging

            o, r, done, info = e.step(a[i], include_breakdown=True)
            bd = info.get("reward_breakdown") or {}
            for k, v in bd.items():
                terms[k] += float(v)
            ep_goal[i] += float(bd.get("goal", 0.0))

            # Ownership must be read AFTER the step, the same way the reward reads it. The
            # reward refreshes authorship at the top of its own call, so attributing a step to
            # the owner from before it disagrees on exactly the steps where possession changed
            # -- which are the only interesting ones.
            for cc in e.arena.cars:
                if int(getattr(cc, "ball_touches", 0)) > touch_seen[i].get(cc.id, 0):
                    owner[i] = cc.team
                touch_seen[i][cc.id] = int(getattr(cc, "ball_touches", 0))
            v = float(bd.get("ball_to_goal", 0.0))
            if owner[i] is None:
                b2g["none"] += v
            elif owner[i] == 0:
                b2g["ours"] += v
            else:
                b2g["theirs"] += v

            # ball_touches is cumulative: only the increase is a new touch.
            n = int(getattr(c, "ball_touches", 0))
            if n > prev_t[i]:
                ep_touch[i] += 1
            prev_t[i] = n

            ep_len[i] += 1
            total_steps += 1
            obs[i] = o
            if np.any(done):
                lengths.append(ep_len[i])
                touches.append(ep_touch[i])
                goal_totals.append(ep_goal[i])
                ep_len[i] = 0
                ep_touch[i] = 0
                prev_t[i] = 0
                ep_goal[i] = 0.0
                owner[i] = None
                touch_seen[i] = dict()
                obs[i] = e.reset()

    L = np.asarray(lengths) if lengths else np.asarray([0])
    T = np.asarray(touches) if touches else np.asarray([0])
    G = np.asarray(goal_totals) if goal_totals else np.asarray([0.0])
    return {
        "episodes": len(lengths),
        "ep_len_steps": float(L.mean()),
        "touches_per_ep": float(T.mean()),
        "zero_touch_pct": 100.0 * float((T == 0).mean()),
        "airborne_pct": 100.0 * air / max(1, total_steps),
        "ang_vel": av_sum / max(1, air_steps),
        "spin_gt2_pct": 100.0 * spin_gt2 / max(1, air_steps),
        "pitch_sat_pct": 100.0 * pitch_sat / max(1, air_steps),
        "upright_pct": 100.0 * upright / max(1, total_steps),
        # pitch_sat_pct is a share of AIRBORNE steps, so it climbs when the bot merely flies
        # less. This is the same quantity against all play, which is what actually changed.
        "tumble_pct": 100.0 * pitch_sat / max(1, total_steps),
        "reverse_pct": 100.0 * rev_grounded / max(1, grounded_steps),
        "turnarounds_per_1k": 1000.0 * turnarounds / max(1, total_steps),
        "goal_per_ep": float(G.mean()),
        "goal_se": float(G.std() / max(1.0, np.sqrt(len(G)))),
        "_terms": terms,
        "_steps": total_steps,
        "_b2g": b2g,
    }


def append_csv(row):
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    new = not os.path.exists(CSV)
    with io.open(CSV, "a", encoding="utf-8") as f:
        if new:
            f.write(",".join(COLUMNS) + "\n")
        f.write(",".join(
            ("%.4f" % row[c]) if isinstance(row[c], float) else str(row[c])
            for c in COLUMNS) + "\n")


def print_history(limit=12):
    if not os.path.exists(CSV):
        print("no history yet at %s" % CSV)
        return
    text = io.open(CSV, encoding="utf-8").read().strip()
    rows = [l.split(",") for l in text.split("\n")]
    head = rows[0]
    body = rows[1:][-limit:]
    show = ["iteration", "touches_per_ep", "zero_touch_pct", "airborne_pct",
            "ang_vel", "tumble_pct", "reverse_pct"]
    # Rows written before a column existed simply lack it. Keep them readable rather than
    # forcing the file to be deleted -- the whole point of the log is the long trend.
    idx = [head.index(c) if c in head else None for c in show]

    def cell(row, i, col):
        if i is not None and i < len(row):
            return row[i]
        # tumble_pct is recoverable from the two columns every old row already has.
        if col == "tumble_pct" and "airborne_pct" in head and "pitch_sat_pct" in head:
            try:
                a = float(row[head.index("airborne_pct")])
                pv = float(row[head.index("pitch_sat_pct")])
                return "%.4f" % (a * pv / 100.0)
            except (ValueError, IndexError):
                return "-"
        return "-"
    print("%-10s %12s %13s %11s %9s %10s %10s"
          % ("iteration", "touches/ep", "zero-touch %", "airborne %", "ang vel",
             "tumble %", "reverse %"))
    print("-" * 82)
    for r in body:
        print("%-10s %12s %13s %11s %9s %10s %10s"
              % tuple(cell(r, i, c) for i, c in zip(idx, show)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", default="checkpoints/latest_model.pt")
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1500,
                    help="policy steps per env (default 1500, about 4 minutes)")
    ap.add_argument("--quick", action="store_true", help="shorthand for --steps 500")
    ap.add_argument("--no-log", action="store_true", help="do not append to the csv")
    ap.add_argument("--history", action="store_true", help="print the csv and exit")
    args = ap.parse_args()

    if args.history:
        print_history()
        return

    steps = 500 if args.quick else args.steps
    path = args.checkpoint
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if not os.path.exists(path):
        print("no checkpoint at %s" % path)
        sys.exit(1)

    agent, iteration, global_step = load_agent(path)
    rw = load_weights(global_step)
    print("policy iteration %s   %d envs x %d steps" % (iteration, args.envs, steps))
    if args.quick:
        print("QUICK mode: fewer episodes, so treat every number as indicative only.")
    t0 = time.time()
    m = run(agent, rw, args.envs, steps)
    print("%d episodes in %.0fs" % (m["episodes"], time.time() - t0))
    print()

    print("ENGAGEMENT")
    print("  touches per episode      %8.2f" % m["touches_per_ep"])
    print("  episodes with no touch   %7.0f%%" % m["zero_touch_pct"])
    print("  episode length           %8.0f steps (%.1f s)"
          % (m["ep_len_steps"], m["ep_len_steps"] / HZ))
    print()
    print("AIR CONTROL   (read airborne share and ang vel together, see the header notes)")
    print("  airborne share           %7.0f%%" % m["airborne_pct"])
    print("  angular velocity         %8.2f rad/s   (cap 5.5)" % m["ang_vel"])
    print("  spinning above 2 rad/s   %7.0f%% of airborne steps" % m["spin_gt2_pct"])
    print("  pitch above 0.9          %7.0f%% of airborne steps" % m["pitch_sat_pct"])
    print("  ... as a share of ALL     %7.1f%%   <- compare THIS across runs; the line above"
          % m["tumble_pct"])
    print("                                      rises whenever the bot simply flies less")
    print("  upright                  %7.0f%% of all steps" % m["upright_pct"])
    print()
    print("GROUND MOVEMENT")
    print("  driving backwards        %7.0f%% of grounded steps" % m["reverse_pct"])
    print("  turnarounds (half-flips) %8.2f per 1000 steps" % m["turnarounds_per_1k"])
    print("  (a high reverse share with few turnarounds means the bot chases the ball")
    print("   backwards instead of turning around)")
    print()
    print("REWARD PER EPISODE")
    eps = max(1, m["episodes"])
    for k, v in sorted(m["_terms"].items(), key=lambda kv: -abs(kv[1])):
        if abs(v / eps) > 1e-3 and k != "goal":
            print("  %-22s %+9.4f" % (k, v / eps))
    print("  %-22s %+9.4f  +- %.2f   noise at this sample size, do not read"
          % ("goal", m["goal_per_ep"], m["goal_se"]))
    print()
    tot = sum(m["_b2g"].values())
    if abs(tot) > 1e-9:
        # This is an assertion, not a trend. The gate returns exactly 0.0 on steps the
        # opponent owns, so anything other than 100% / 0% / 0% means the gate is broken --
        # not that possession changed.
        leak = abs(m["_b2g"]["theirs"]) + abs(m["_b2g"]["none"])
        if leak > 1e-6:
            print("BALL_TO_GOAL AUTHORSHIP   *** GATE LEAKING ***")
            for label, key in (("our team", "ours"), ("opponent", "theirs"),
                               ("nobody yet", "none")):
                print("  %-12s %+9.4f per episode  %5.1f%%"
                      % (label, m["_b2g"][key] / eps, 100.0 * m["_b2g"][key] / tot))
            print("  ball_to_goal must pay only the team that last touched the ball.")
        else:
            print("ball_to_goal authorship gate: OK (100%% of the term earned on our own "
                  "possession, %+.4f per episode)" % (m["_b2g"]["ours"] / eps))
        print()

    if not args.no_log:
        row = {c: m[c] for c in COLUMNS if c in m}
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        row["iteration"] = iteration
        append_csv(row)
        print("appended to %s" % os.path.relpath(CSV, ROOT))
        print()
        print_history()


if __name__ == "__main__":
    main()
