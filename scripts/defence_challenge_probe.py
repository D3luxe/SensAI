"""
Defence and challenge probe: own-net entries on retreats, and how 50/50 challenges are won and lost.

Plays kickoff-to-goal matches of each checkpoint (blue, deterministic, as the live bot plays)
against an opponent through shot_quality.Game, and measures two things watched in-game on
2026-09-23 with v10 at iter ~228k:

  own net    SensAI rotating back and driving into its own net, often with the ball in a corner,
             then getting turned around. An entry is the car crossing our goal line between the
             posts. Each entry records where the ball was (corner, shot threat), how fast and
             nose-first the car went in, how long it stayed, whether it went off its wheels, how
             long it took after leaving to face the ball again, and what happened in the next 4 s.
             "Unforced" entries had no shot on target coming (ball not projected to cross our line
             between the posts within 2 s) -- those are the ones a target in the net would explain.
             Also reported: of the steps with the ball in one of our corners, the share SensAI
             spent in the net.

  challenges A contested touch: the first touch by either car after a lull of QUIET_STEPS with no
             touch, while the other car is within CHALLENGE_UU of the ball and was closing on it
             0.5 s earlier. The lull keeps a dribble from counting every touch, and the closing
             test keeps a car that is merely nearby from counting as a challenger. Each records
             who touched first and, 1 s before the touch, each car's naive
             time to the ball (horizontal distance / closing speed), SensAI's closing speed at 1 s,
             0.5 s and at the touch, its heading alignment, and an arc index (path driven over the
             last 2 s / straight-line displacement; 1.0 is a straight line). For challenges the
             opponent won: whether the ball ended up goal-side of SensAI within 1.5 s ("beaten"),
             passed over it ("chipped"), and whether a goal against followed within 5 s.
             The first touch after each kickoff is skipped (kickoffs are eval_suite's), and a touch
             by both cars on the same step is a tie, reported separately and left out of the
             won/lost split. Counting ties as wins once read 80% for blue in self-play, where
             mirrored policies meet head-on and tie often; decided challenges were 20 to 23.
             "Ahead but lost" is the hesitation signature: SensAI's time to the ball 1 s out was
             shorter than the opponent's, and the opponent still got there first.

Seed-to-seed spread (standard deviation across seeds) is printed with every number. Matches
are continuous, so events within a seed are correlated; a difference smaller than the spread is
not a difference.

    python scripts/defence_challenge_probe.py
    python scripts/defence_challenge_probe.py checkpoints/latest_model.pt checkpoints/baselines/v5_iter222000.pt
    python scripts/defence_challenge_probe.py --opponent nexto --seeds 11 12 13 14 --steps 9000
    python scripts/defence_challenge_probe.py --opponent self --json logs/defence_probe.json
"""
import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
os.environ.setdefault("RS_COLLISION_MESHES", os.path.join(ROOT, "collision_meshes"))

import numpy as np  # noqa: E402

from env.physics_engine import ARENA_EXTENT_Y, GOAL_HALF_WIDTH  # noqa: E402

STEPS_PER_MIN = 15 * 60
DEFAULT_CHECKPOINTS = ["checkpoints/latest_model.pt"]
OPPONENTS = {"necto": "checkpoints/necto-model.pt", "nexto": "checkpoints/nexto-model.pt"}

# Own net (blue defends -Y)
NET_ENTER_Y = -ARENA_EXTENT_Y            # past our goal line
NET_EXIT_Y = -ARENA_EXTENT_Y + 100.0     # hysteresis, so bouncing on the line is one entry
CORNER_X = GOAL_HALF_WIDTH + 600.0       # ball wider than this...
CORNER_Y = -ARENA_EXTENT_Y + 2000.0      # ...and deeper than this is in one of our corners
THREAT_S = 2.0                           # ball projected across our line between the posts this soon
FACING = 0.7                             # forward . unit(ball) above this is facing the ball
REORIENT_CAP_STEPS = 45                  # stop waiting to face the ball after 3 s
OFF_WHEELS_UP_Z = 0.5                    # up-vector z below this inside the net: turned over
NET_OUTCOME_STEPS = 60                   # 4 s

# Challenges
CHALLENGE_UU = 800.0
QUIET_STEPS = 15                         # 1 s with no touch before a touch can open a challenge
CHALLENGER_CLOSING = 200.0               # uu/s the other car must be closing at, 0.5 s out
LOOKBACK_STEPS = 30                      # 2 s of history for the arc index
ETA_STEP = 15                            # time-to-ball compared 1 s before the touch
ETA_CAP_S = 10.0
AFTER_STEPS = 22                         # 1.5 s to judge beaten / chipped
GOAL_AFTER_STEPS = 75                    # 5 s
CHIP_HORIZ_UU = 250.0
CHIP_ABOVE_UU = 150.0


def _unit2(v):
    n = math.hypot(float(v[0]), float(v[1]))
    return (float(v[0]) / n, float(v[1]) / n) if n > 1e-6 else (0.0, 0.0)


def _closing(pos, vel, ball_pos, ball_vel):
    """Rate the horizontal gap to the ball is shrinking, uu/s (positive = closing)."""
    u = _unit2(np.asarray(ball_pos[:2]) - np.asarray(pos[:2]))
    rel = np.asarray(vel[:2]) - np.asarray(ball_vel[:2])
    return rel[0] * u[0] + rel[1] * u[1]


def _eta(pos, vel, ball_pos, ball_vel):
    d = float(np.hypot(*(np.asarray(ball_pos[:2]) - np.asarray(pos[:2]))))
    c = _closing(pos, vel, ball_pos, ball_vel)
    return ETA_CAP_S if c < 50.0 else min(ETA_CAP_S, d / c)


class Probe:
    """shot_quality.Game observer: called every live step, and episode_end() after each goal."""

    def __init__(self):
        self.steps = 0
        self.corner_steps = 0
        self.corner_net_steps = 0
        self.entries = []
        self.challenges = []
        self._in_net = None
        self._open = []           # entries / challenges still collecting their aftermath
        self._hist = collections.deque(maxlen=LOOKBACK_STEPS + 1)
        self._last_touch_t = -10 ** 9
        self._kickoff_pending = True
        self._last_ball_y = 0.0

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _threat(ball):
        vy = float(ball.vel[1])
        if vy >= -1.0:
            return False
        t = (NET_ENTER_Y - float(ball.pos[1])) / vy
        if t < 0 or t > THREAT_S:
            return False
        x = float(ball.pos[0]) + float(ball.vel[0]) * t
        return abs(x) < GOAL_HALF_WIDTH

    @staticmethod
    def _facing_ball(car, ball):
        f = _unit2(car.get_forward_vector())
        u = _unit2(np.asarray(ball.pos[:2]) - np.asarray(car.pos[:2]))
        return f[0] * u[0] + f[1] * u[1] > FACING

    # -- per step --------------------------------------------------------------------------
    def __call__(self, t, arena, sensai_touched=False, opp_touched=False):
        car, opp, ball = arena.cars[0], arena.cars[1], arena.ball
        self.steps += 1
        self._last_ball_y = float(ball.pos[1])
        snap = dict(t=t, car_pos=car.pos[:2].copy(), car_vel=car.vel[:2].copy(),
                    opp_pos=opp.pos[:2].copy(), opp_vel=opp.vel[:2].copy(),
                    ball_pos=ball.pos[:2].copy(), ball_vel=ball.vel[:2].copy())
        self._hist.append(snap)

        in_corner = abs(float(ball.pos[0])) > CORNER_X and float(ball.pos[1]) < CORNER_Y
        in_net_now = float(car.pos[1]) < NET_ENTER_Y and abs(float(car.pos[0])) < GOAL_HALF_WIDTH
        if in_corner:
            self.corner_steps += 1
            self.corner_net_steps += in_net_now

        self._track_net(t, car, ball, in_corner)
        if sensai_touched or opp_touched:
            self._maybe_challenge(t, car, opp, ball, sensai_touched, opp_touched)
        self._update_open(t, car, ball, sensai_touched, opp_touched)

    def _track_net(self, t, car, ball, in_corner):
        y = float(car.pos[1])
        if self._in_net is None:
            if y < NET_ENTER_Y and abs(float(car.pos[0])) < GOAL_HALF_WIDTH:
                fwd = car.get_forward_vector()
                self._in_net = dict(
                    kind="net", t=t, ball_corner=in_corner, threat=self._threat(ball),
                    ball_our_half=float(ball.pos[1]) < 0.0,
                    speed=float(np.hypot(*car.vel[:2])), nose_in=float(fwd[1]) < -0.5,
                    off_wheels=False, steps_in=0, reorient_steps=None, exit_t=None, outcome=None)
                self.entries.append(self._in_net)
                self._open.append(self._in_net)
        else:
            e = self._in_net
            e["steps_in"] = t - e["t"]
            e["off_wheels"] |= float(car.get_up_vector()[2]) < OFF_WHEELS_UP_Z
            if y > NET_EXIT_Y:
                e["exit_t"] = t
                self._in_net = None

    def _maybe_challenge(self, t, car, opp, ball, s_t, o_t):
        quiet = t - self._last_touch_t >= QUIET_STEPS
        self._last_touch_t = t
        if self._kickoff_pending:
            self._kickoff_pending = False
            return
        if not quiet:
            return
        d_s = float(np.hypot(*(ball.pos[:2] - car.pos[:2])))
        d_o = float(np.hypot(*(ball.pos[:2] - opp.pos[:2])))
        h = list(self._hist)
        back = h[-1 - ETA_STEP] if len(h) > ETA_STEP else h[0]
        half = h[-1 - ETA_STEP // 2] if len(h) > ETA_STEP // 2 else h[0]
        pre = h[-2] if len(h) > 1 else h[-1]
        if not (s_t and o_t):
            who = "opp" if s_t else "car"
            if (d_o if s_t else d_s) > CHALLENGE_UU:
                return
            if _closing(half[who + "_pos"], half[who + "_vel"], half["ball_pos"], half["ball_vel"]) < CHALLENGER_CLOSING:
                return
        first = "both" if s_t and o_t else ("sensai" if s_t else "opp")

        eta_s = _eta(back["car_pos"], back["car_vel"], back["ball_pos"], back["ball_vel"])
        eta_o = _eta(back["opp_pos"], back["opp_vel"], back["ball_pos"], back["ball_vel"])
        path = sum(float(np.hypot(*(b["car_pos"] - a["car_pos"]))) for a, b in zip(h, h[1:]))
        disp = float(np.hypot(*(h[-1]["car_pos"] - h[0]["car_pos"])))
        f = _unit2(car.get_forward_vector())
        u = _unit2(np.asarray(back["ball_pos"]) - np.asarray(back["car_pos"]))
        self.challenges.append(dict(
            kind="challenge", t=t, first=first, d_sensai=d_s, d_opp=d_o,
            ball_z=float(ball.pos[2]), our_half=float(ball.pos[1]) < 0.0,
            eta_sensai=eta_s, eta_opp=eta_o, ahead=eta_s < eta_o,
            close_1s=_closing(back["car_pos"], back["car_vel"], back["ball_pos"], back["ball_vel"]),
            close_half=_closing(half["car_pos"], half["car_vel"], half["ball_pos"], half["ball_vel"]),
            # the step before the touch: this one already carries the ball's rebound
            close_now=_closing(pre["car_pos"], pre["car_vel"], pre["ball_pos"], pre["ball_vel"]),
            align_1s=f[0] * u[0] + f[1] * u[1],
            arc=path / disp if disp > 200.0 else None,
            beaten=False, chipped=False, goal_against=False, sensai_retouch=False))
        if first == "opp":
            self._open.append(self.challenges[-1])

    def _update_open(self, t, car, ball, s_t, o_t):
        still = []
        for e in self._open:
            age = t - e["t"]
            if e["kind"] == "net":
                if e["exit_t"] is not None and e["reorient_steps"] is None:
                    if self._facing_ball(car, ball):
                        e["reorient_steps"] = t - e["exit_t"]
                    elif t - e["exit_t"] >= REORIENT_CAP_STEPS:
                        e["reorient_steps"] = REORIENT_CAP_STEPS
                if e["outcome"] is None and age > 0:
                    if s_t:
                        e["outcome"] = "own touch"
                    elif o_t:
                        e["outcome"] = "opp touch"
                    elif age >= NET_OUTCOME_STEPS:
                        e["outcome"] = "no touch"
                if e["outcome"] is None or (e["exit_t"] is None or e["reorient_steps"] is None) and age < 600:
                    still.append(e)
            else:
                if age <= AFTER_STEPS and not e["sensai_retouch"]:
                    if s_t and age > 0:
                        e["sensai_retouch"] = True
                    elif float(ball.pos[1]) < float(car.pos[1]) - 100.0:
                        e["beaten"] = True
                    horiz = float(np.hypot(*(ball.pos[:2] - car.pos[:2])))
                    if horiz < CHIP_HORIZ_UU and float(ball.pos[2]) - float(car.pos[2]) > CHIP_ABOVE_UU:
                        e["chipped"] = True
                if age < GOAL_AFTER_STEPS:
                    still.append(e)
        self._open = still

    def episode_end(self):
        # Game reads the goal from the env and only tells us the episode ended; the ball's last
        # position says which net. Anything still open closes here.
        against = self._last_ball_y < 0.0
        for e in self._open:
            if e["kind"] == "net":
                if e["outcome"] is None:
                    e["outcome"] = "goal against" if against else "goal for"
            else:
                e["goal_against"] = against
        self._open = []
        self._in_net = None
        self._hist.clear()
        self._last_touch_t = -10 ** 9
        self._kickoff_pending = True


# ---------------------------------------------------------------------------------------------
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _pct(xs):
    xs = list(xs)
    return 100.0 * sum(bool(x) for x in xs) / len(xs) if xs else None


def summarise(p, steps):
    minutes = steps / STEPS_PER_MIN
    E = p.entries
    unforced = [e for e in E if not e["threat"]]
    C = p.challenges
    lost = [c for c in C if c["first"] == "opp"]
    won = [c for c in C if c["first"] == "sensai"]
    decided = won + lost
    return {
        "net_entries_per_10min": len(E) / minutes * 10.0,
        "net_unforced_pct": _pct(not e["threat"] for e in E),
        "net_unforced_ball_corner_pct": _pct(e["ball_corner"] for e in unforced),
        "net_unforced_ball_our_half_pct": _pct(e["ball_our_half"] for e in unforced),
        "net_nose_in_pct": _pct(e["nose_in"] for e in E),
        "net_entry_speed": _mean(e["speed"] for e in E),
        "net_time_in_s": _mean(e["steps_in"] / 15.0 for e in E),
        "net_off_wheels_pct": _pct(e["off_wheels"] for e in E),
        "net_reorient_s": _mean(e["reorient_steps"] / 15.0 if e["reorient_steps"] is not None else None for e in E),
        "net_next_opp_touch_pct": _pct(e["outcome"] == "opp touch" for e in E),
        "net_goal_against_pct": _pct(e["outcome"] == "goal against" for e in E),
        "corner_time_in_net_pct": 100.0 * p.corner_net_steps / max(1, p.corner_steps),
        "challenges_per_10min": len(C) / minutes * 10.0,
        "challenge_tie_pct": _pct(c["first"] == "both" for c in C),
        "challenge_won_pct": _pct(c["first"] == "sensai" for c in decided),
        "lost_beaten_pct": _pct(c["beaten"] for c in lost),
        "lost_chipped_pct": _pct(c["chipped"] for c in lost),
        "lost_goal_against_5s_pct": _pct(c["goal_against"] for c in lost),
        "lost_ahead_at_1s_pct": _pct(c["ahead"] for c in lost),
        "won_ahead_at_1s_pct": _pct(c["ahead"] for c in won),
        "lost_close_1s": _mean(c["close_1s"] for c in lost),
        "won_close_1s": _mean(c["close_1s"] for c in won),
        "lost_close_half": _mean(c["close_half"] for c in lost),
        "won_close_half": _mean(c["close_half"] for c in won),
        "lost_close_now": _mean(c["close_now"] for c in lost),
        "won_close_now": _mean(c["close_now"] for c in won),
        "lost_align_1s": _mean(c["align_1s"] for c in lost),
        "won_align_1s": _mean(c["align_1s"] for c in won),
        "lost_arc": _mean(c["arc"] for c in lost),
        "won_arc": _mean(c["arc"] for c in won),
        "n_entries": len(E),
        "n_challenges": len(C),
    }


def run_job(job):
    import torch
    torch.set_num_threads(1)
    from policy_health import load_agent, load_weights
    from shot_quality import Game
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    import random
    random.seed(job["seed"])
    agent, ckpt = load_agent(job["checkpoint"])
    weights = load_weights(ckpt)
    probe = Probe()
    Game(agent, weights, job["opponent"]).run(job["steps"], observer=probe)
    return {"job": job, "iteration": ckpt.get("iteration"), "metrics": summarise(probe, job["steps"])}


SECTIONS = [
    ("Own net", [
        ("net_entries_per_10min", "entries per 10 min", "{:.1f}"),
        ("net_unforced_pct", "unforced (no shot coming) %", "{:.0f}"),
        ("net_unforced_ball_corner_pct", "  of unforced, ball in our corner %", "{:.0f}"),
        ("net_unforced_ball_our_half_pct", "  of unforced, ball in our half %", "{:.0f}"),
        ("net_nose_in_pct", "went in nose-first %", "{:.0f}"),
        ("net_entry_speed", "entry speed uu/s", "{:.0f}"),
        ("net_time_in_s", "time in net s", "{:.2f}"),
        ("net_off_wheels_pct", "turned over inside %", "{:.0f}"),
        ("net_reorient_s", "exit -> facing ball s", "{:.2f}"),
        ("net_next_opp_touch_pct", "next touch was the opponent's %", "{:.0f}"),
        ("net_goal_against_pct", "goal against before any touch %", "{:.0f}"),
        ("corner_time_in_net_pct", "ball in our corner: time SensAI in net %", "{:.1f}"),
    ]),
    ("Challenges", [
        ("challenges_per_10min", "challenges per 10 min", "{:.1f}"),
        ("challenge_tie_pct", "tie (both touched the same step) %", "{:.0f}"),
        ("challenge_won_pct", "SensAI touched first, of decided %", "{:.0f}"),
        ("lost_beaten_pct", "lost: ball got goal-side of SensAI %", "{:.0f}"),
        ("lost_chipped_pct", "lost: ball went over SensAI %", "{:.0f}"),
        ("lost_goal_against_5s_pct", "lost: goal against within 5 s %", "{:.0f}"),
        ("lost_ahead_at_1s_pct", "lost although ahead on time-to-ball 1 s out %", "{:.0f}"),
        ("won_ahead_at_1s_pct", "won, and was ahead 1 s out %", "{:.0f}"),
        ("lost_close_1s", "lost: closing speed 1 s out", "{:.0f}"),
        ("won_close_1s", "won:  closing speed 1 s out", "{:.0f}"),
        ("lost_close_half", "lost: closing speed 0.5 s out", "{:.0f}"),
        ("won_close_half", "won:  closing speed 0.5 s out", "{:.0f}"),
        ("lost_close_now", "lost: closing speed at touch", "{:.0f}"),
        ("won_close_now", "won:  closing speed at touch", "{:.0f}"),
        ("lost_align_1s", "lost: heading at ball 1 s out", "{:.2f}"),
        ("won_align_1s", "won:  heading at ball 1 s out", "{:.2f}"),
        ("lost_arc", "lost: arc index (1 = straight)", "{:.2f}"),
        ("won_arc", "won:  arc index", "{:.2f}"),
    ]),
]


def aggregate(results):
    by_ck = collections.defaultdict(list)
    for r in results:
        by_ck[r["job"]["checkpoint"]].append(r)
    out = {}
    for ck, rs in by_ck.items():
        keys = rs[0]["metrics"].keys()
        agg = {}
        for k in keys:
            vals = [r["metrics"][k] for r in rs if r["metrics"][k] is not None]
            agg[k] = (float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0) if vals else (None, None)
        out[ck] = {"iteration": rs[0]["iteration"], "seeds": len(rs), "metrics": agg,
                   "totals": {"entries": sum(r["metrics"]["n_entries"] for r in rs),
                              "challenges": sum(r["metrics"]["n_challenges"] for r in rs)}}
    return out


def print_report(summary, opponent, steps):
    cks = list(summary)
    names = [f"{os.path.basename(c)} @{summary[c]['iteration']}" for c in cks]
    w = max(22, *(len(n) for n in names)) + 3
    print(f"\nvs {opponent}, {summary[cks[0]]['seeds']} seed(s) x {steps / STEPS_PER_MIN:.1f} min;"
          f"  mean +/- seed spread")
    print(" " * 46 + "".join(f"{n:>{w}}" for n in names))
    print(" " * 46 + "".join(f"{'(%d entries, %d challenges)' % (summary[c]['totals']['entries'], summary[c]['totals']['challenges']):>{w}}" for c in cks))
    for title, rows in SECTIONS:
        print(f"\n{title}")
        for key, label, fmt in rows:
            cells = []
            for c in cks:
                m, s = summary[c]["metrics"][key]
                cells.append("-" if m is None else f"{fmt.format(m)} +/- {fmt.format(s)}")
            print(f"  {label:44}" + "".join(f"{x:>{w}}" for x in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", nargs="*", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--opponent", default="necto", help="necto, nexto, self, heuristic, or a checkpoint path")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--steps", type=int, default=9000, help="15 Hz steps per seed (9000 = 10 min)")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--json", help="also write the per-seed and aggregate results here")
    args = ap.parse_args()

    opponent = OPPONENTS.get(args.opponent, args.opponent)
    cks = [c for c in args.checkpoints if os.path.exists(c) or print(f"skipping {c}: not found")]
    jobs = [dict(checkpoint=c, seed=s, steps=args.steps, opponent=opponent) for c in cks for s in args.seeds]
    if args.workers > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            results = pool.map(run_job, jobs)
    else:
        results = [run_job(j) for j in jobs]
    summary = aggregate(results)
    print_report(summary, args.opponent, args.steps)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"opponent": opponent, "steps": args.steps, "seeds": args.seeds,
                       "summary": summary, "per_seed": [{"checkpoint": r["job"]["checkpoint"],
                                                         "seed": r["job"]["seed"], "metrics": r["metrics"]}
                                                        for r in results]}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
