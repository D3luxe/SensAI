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
  positioning  time on walls and on the floor-to-wall curve, back-wall climbs while defending, dodges taken while
               retreating (speed gained toward home, and whether they landed wheels-down), and
               overshoots: drove inside 300 uu of the ball and gave the distance back untouched

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
from env.physics_engine import ARENA_EXTENT_X, ARENA_EXTENT_Y  # noqa: E402
from env.rewards import (  # noqa: E402
    defensive_recovery_point, goal_mouth_open_angle, is_car_on_wall, on_target_factor,
)

GOAL_Y = ARENA_EXTENT_Y            # blue attacks +Y
OUTCOME_WINDOW_STEPS = 60          # 4 s
WHIFF_RADIUS = 600.0
WHIFF_WINDOW_STEPS = 10
WHIFF_CLOSE_UU = 300.0
CATEGORIES = ("kickoff", "on_target", "wide_backboard", "forward_miss", "centering", "other")
DIST_BANDS = ((0, 1500), (1500, 3000), (3000, 4500), (4500, 6000))

# Positioning & recovery section
WALL_SIDE_X = 3400.0               # |x| beyond which a wall contact is the side wall
WALL_BACK_Y = 4400.0               # |y| beyond which it is a backboard
BACKWALL_CLIMB_Z = 200.0           # height that counts as climbing the backboard, not driving at it
BACKWALL_EXIT_Z = 100.0
OVERSHOOT_NEAR_UU = 400.0          # an approach starts once this close to the ball
OVERSHOOT_CLOSE_UU = 300.0         # ...and only counts as an overshoot if it got this close
OVERSHOOT_RECEDE_UU = 150.0        # ...then gave this much distance back without a touch
OVERSHOOT_ABANDON_UU = 800.0
CURVE_GAP_UU = 900.0               # this close to a SIDE wall is play on the floor-to-wall curve
#                                    (back-wall proximity is ordinary goalkeeping, measured by climbs)
OVERSHOOT_AIMED = 0.7              # heading alignment above which a miss was a genuine attempt, not a peel-off
LANDING_WHEELS_DOWN = 0.7          # up-vector z at touchdown; below this it landed on a door or roof


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


def aim_at_ball(car, ball_pos, dist):
    """Alignment of the car's heading with the direction to the ball, in [-1, 1]; 0 below 200 uu/s."""
    speed = float(np.linalg.norm(car.vel))
    if speed < 200.0:
        return 0.0
    return float(np.dot(car.vel, ball_pos - car.pos)) / (speed * max(1.0, dist))


def ball_surface(ball_pos, car_on_wall):
    """
    Where the play is, as the reward terms see it:
      wall   the car is on a wall or transition ramp (is_car_on_wall)
      curve  the ball is within CURVE_GAP_UU of a SIDE wall, i.e. the floor-to-wall curve, where a
             carry runs along the ramp without the car ever registering as on a wall
      floor  open pitch
    """
    if car_on_wall:
        return "wall"
    return "curve" if ARENA_EXTENT_X - abs(float(ball_pos[0])) < CURVE_GAP_UU else "floor"


class Positioning:
    """
    Where SensAI spends its time and what its recoveries cost it, tracked per step for blue.

      wall time        share of steps on a wall, on the side-wall curve, or in a corner, by half
                       of the pitch -- a low carry along the curve never registers as a wall
      wall touches     touches taken there, and how often the opponent got the next one
      back-wall climbs stretches spent climbing our own backboard while the ball is in our half:
                       how long, how high, how far the ball was, and what happened next
      retreat dodges   dodges while upfield of a ball in our half, scored by the speed they added
                       toward the defensive recovery point and whether they landed wheels-down
      overshoots       drove inside OVERSHOOT_CLOSE_UU of the ball and gave the distance back
                       without either car touching it, split by cause: the car carrying itself
                       past the ball (the fumble OvershootCost charges) versus the ball outrunning
                       a car that was still driving at it, which nothing charges
    """

    def __init__(self):
        self.steps = 0
        self.wall_steps = collections.Counter()      # (surface, half) -> steps
        self.wall_touches = 0
        self.wall_touch_lost = collections.Counter()
        self.climbs = []
        self.dodges = []
        self.overshoots = []
        self.charges = []                            # what OvershootCost actually charged, per event
        self._climb = None
        self._dodge = None
        self._approach = None
        self._prev_vel = None
        self._pending_climb = []
        self._pending_wall_touch = []

    def reset(self):
        """A goal or episode reset: nothing in flight carries across it."""
        self._climb = self._dodge = self._approach = None
        self._prev_vel = None
        self._pending_climb, self._pending_wall_touch = [], []

    def step(self, t, car, ball_pos, sensai_touched, opp_touched, overshoot_charge=0.0):
        if overshoot_charge < 0.0:
            self.charges.append(overshoot_charge)
        prev_vel = self._prev_vel
        self._prev_vel = car.vel.copy()
        self.steps += 1

        car_y, car_z = float(car.pos[1]), float(car.pos[2])
        dist = float(np.linalg.norm(ball_pos - car.pos))
        our_half = car_y < 0.0                       # blue defends -Y
        ball_our_half = float(ball_pos[1]) < 0.0
        on_wall = is_car_on_wall(car)
        up_z = float(car.get_up_vector()[2])

        # wall occupancy
        on_curve = ARENA_EXTENT_X - abs(float(car.pos[0])) < CURVE_GAP_UU
        if on_wall or on_curve:
            if on_wall:
                surface = "back wall" if abs(car_y) > WALL_BACK_Y else (
                    "side wall" if abs(float(car.pos[0])) > WALL_SIDE_X else "corner")
            else:
                surface = "curve"
            self.wall_steps[(surface, "defending" if our_half else "attacking")] += 1
            if sensai_touched:
                self.wall_touches += 1
                self._pending_wall_touch.append((t, surface))

        for pending in list(self._pending_wall_touch):
            start, surface = pending
            if opp_touched:
                self.wall_touch_lost[surface] += 1
                self._pending_wall_touch.remove(pending)
            elif (sensai_touched and t > start) or t - start >= OUTCOME_WINDOW_STEPS:
                self._pending_wall_touch.remove(pending)

        # back-wall climbs while the ball is in our half
        climbing = bool(our_half and ball_our_half and abs(car_y) > WALL_BACK_Y and car_z > BACKWALL_CLIMB_Z)
        if climbing and self._climb is None:
            self._climb = dict(t=t, peak_z=car_z, ball_dist=dist, steps=0)
        elif self._climb is not None:
            self._climb["peak_z"] = max(self._climb["peak_z"], car_z)
            if car_z < BACKWALL_EXIT_Z or abs(car_y) < WALL_BACK_Y - 100.0:
                self._climb["steps"] = t - self._climb["t"]
                self._pending_climb.append(self._climb)
                self.climbs.append(self._climb)
                self._climb = None

        for climb in list(self._pending_climb):
            if opp_touched:
                climb["next"] = "opp touch"
            elif sensai_touched:
                climb["next"] = "own touch"
            elif t - climb["t"] >= OUTCOME_WINDOW_STEPS:
                climb["next"] = "no touch"
            if "next" in climb:
                self._pending_climb.remove(climb)

        # dodges taken while retreating
        if car.just_dodged and ball_our_half and car_y > float(ball_pos[1]) and prev_vel is not None:
            home = defensive_recovery_point(ball_pos, 0)
            to_home = home[:2] - car.pos[:2]
            norm = max(1e-4, float(np.linalg.norm(to_home)))
            unit = to_home / norm
            self._dodge = dict(
                t=t, gain=float(np.dot(car.vel[:2] - prev_vel[:2], unit)),
                home_dist=norm, on_wall=on_wall, landed=None, landing_up=0.0, landing_wall=None,
            )
            self.dodges.append(self._dodge)
        if self._dodge is not None and self._dodge["landed"] is None:
            if car.on_ground and t > self._dodge["t"]:
                self._dodge["landed"] = t - self._dodge["t"]
                self._dodge["landing_up"] = up_z
                self._dodge["landing_wall"] = on_wall
            elif t - self._dodge["t"] >= OUTCOME_WINDOW_STEPS:
                self._dodge["landed"] = OUTCOME_WINDOW_STEPS

        # overshoots
        if sensai_touched or opp_touched:
            self._approach = None   # somebody played it; nothing was missed
        elif self._approach is None:
            aim = aim_at_ball(car, ball_pos, dist)
            if dist < OVERSHOOT_NEAR_UU and aim > 0.0:
                self._approach = dict(t=t, min_d=dist, aim=aim, ball_z=float(ball_pos[2]), recede=0.0,
                                      where=ball_surface(ball_pos, on_wall),
                                      half="defending" if our_half else "attacking")
        else:
            self._approach["min_d"] = min(self._approach["min_d"], dist)
            self._approach["aim"] = max(self._approach["aim"], aim_at_ball(car, ball_pos, dist))
            unit = (ball_pos - car.pos) / max(1.0, dist)
            self._approach["recede"] = max(self._approach["recede"], -float(np.dot(car.vel, unit)))
            if dist > self._approach["min_d"] + OVERSHOOT_RECEDE_UU:
                if self._approach["min_d"] < OVERSHOOT_CLOSE_UU:
                    self.overshoots.append(self._approach)
                self._approach = None
            elif dist > OVERSHOOT_ABANDON_UU:
                self._approach = None


KICKOFF_GOAL_S = 10.0              # a goal this soon after the kickoff is scored off it
GA_HISTORY_STEPS = 150             # 10 s of per-step snapshots kept for the goals-against breakdown
GA_WINDOW_STEPS = 45               # 3 s: the lead-up examined before the shot that scored
GA_CLOSE_UU = 1200.0               # goal-side and this close at the shot: it was there and got beaten


class Kickoffs:
    """
    Who wins each kickoff and what it is worth.

    For every kickoff: which car touched first (or both on the same step, a true 50/50 whose
    ball-speed change is shared), the ball's speed and heading off that contact, the touch reward
    SensAI was paid for it, how far the opponent was from the ball at the time, who touched next,
    and whether a goal followed within KICKOFF_GOAL_S.
    """

    def __init__(self):
        self.done = []
        self.cur = None

    def start(self, t):
        self.cur = dict(start=t, first=None)

    def step(self, t, arena, sensai_touched, opp_touched, touch_r):
        k = self.cur
        if k is None:
            return
        if k["first"] is None:
            if not (sensai_touched or opp_touched):
                return
            car, opp, ball = arena.cars[0], arena.cars[1], arena.ball
            k.update(
                first="both" if sensai_touched and opp_touched else ("sensai" if sensai_touched else "opp"),
                t_first=t, secs=(t - k["start"]) / 15.0, touch_r=touch_r,
                ball_speed=float(np.linalg.norm(ball.vel)), ball_vy=float(ball.vel[1]),
                opp_dist=float(np.linalg.norm(opp.pos - ball.pos)),
                car_dist=float(np.linalg.norm(car.pos - ball.pos)), next=None,
            )
            return
        if k["next"] is None:
            if opp_touched:
                k["next"] = "opp"
            elif sensai_touched:
                k["next"] = "own"
            elif t - k["t_first"] >= OUTCOME_WINDOW_STEPS:
                k["next"] = "none"
        if t - k["t_first"] == 45:
            k["ball_y_3s"] = float(arena.ball.pos[1])

    def end(self, t, outcome):
        k = self.cur
        if k is not None:
            k["goal"] = outcome if outcome in ("goal_for", "goal_against") and (t - k["start"]) / 15.0 <= KICKOFF_GOAL_S else None
            self.done.append(k)
        self.cur = None


def analyse_goal_against(hist, t_goal, ko_start, pos, whiffs):
    """
    What SensAI was doing when the opponent took the shot that beat it, from the per-step history.

    The shot is the opponent's last touch before the goal. The primary cause is the first of:
      own touch last    SensAI touched it last (deflection or own goal)
      off the kickoff   scored within KICKOFF_GOAL_S of the kickoff
      back-wall climb   climbing our backboard at any point in the 3 s before the shot or after it
      beaten goal-side  goal-side of the ball and within GA_CLOSE_UU at the shot: it was there and lost
      goal-side, far    goal-side but further out: late rotation / slow to close
      caught upfield    the wrong side of the ball when it was shot
    with independent flags for what else happened in the 3 s before the shot.
    """
    steps = list(hist)
    opp_idx = [i for i, h in enumerate(steps) if h["o_touch"]]
    own_idx = [i for i, h in enumerate(steps) if h["s_touch"]]
    shot_i = opp_idx[-1] if opp_idx else len(steps) - 1
    shot = steps[shot_i]
    t_shot = shot["t"]
    w0 = t_shot - GA_WINDOW_STEPS
    car, ball = shot["car_pos"], shot["ball_pos"]
    dist = float(np.linalg.norm(ball - car))
    goal_side = float(car[1]) < float(ball[1]) - 100.0

    def overlaps(start, end):
        return start <= t_goal and end >= w0

    climbs = [(c["t"], c["t"] + c["steps"]) for c in pos.climbs]
    if pos._climb is not None:
        climbs.append((pos._climb["t"], t_goal))
    climbed = any(overlaps(a, b) for a, b in climbs)
    flags = dict(
        retreat_dodge=any(w0 <= d["t"] <= t_shot for d in pos.dodges),
        overshoot=any(w0 <= o.get("t", -1) <= t_shot for o in pos.overshoots),
        whiff=any(w0 <= w["t"] <= t_shot for w in whiffs),
        touched=any(steps[i]["t"] >= w0 for i in own_idx if i <= shot_i),
        airborne=not shot["on_ground"],
        on_wall=shot["on_wall"],
        climb=climbed,
    )
    if own_idx and (not opp_idx or own_idx[-1] > shot_i):
        cause = "own touch last"
    elif (t_goal - ko_start) / 15.0 <= KICKOFF_GOAL_S:
        cause = "off the kickoff"
    elif climbed:
        cause = "back-wall climb"
    elif goal_side and dist < GA_CLOSE_UU:
        cause = "beaten goal-side"
    elif goal_side:
        cause = "goal-side, far"
    else:
        cause = "caught upfield"
    return dict(
        cause=cause, flags=flags, dist=dist, goal_side=goal_side, boost=shot["boost"],
        shot_to_goal=(t_goal - t_shot) / 15.0, since_ko=(t_goal - ko_start) / 15.0,
        shot_goal_dist=float(math.hypot(ball[0], -GOAL_Y - ball[1])),
        car_goal_dist=float(math.hypot(car[0], -GOAL_Y - car[1])),
        ball_speed=float(np.linalg.norm(shot["ball_vel"])),
    )


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
        pos = Positioning()
        kicks = Kickoffs()
        kicks.start(0)
        ko_start = 0
        hist = collections.deque(maxlen=GA_HISTORY_STEPS)
        goals_against = []
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
                if outcome == "goal_against" and hist:
                    goals_against.append(analyse_goal_against(hist, t, ko_start, pos, whiffs))
                kicks.end(t, outcome)
                hist.clear()
                for ev in pending:
                    ev["outcome"] = outcome
                    events.append(ev)
                pending, watch = [], []
                pos.reset()
                obs = self.kickoff()
                kicks.start(t + 1)
                ko_start = t + 1
                prev_touch = [c.ball_touches for c in arena.cars]
                prev_has_flip, prev_on_ground = arena.cars[0].has_flip, arena.cars[0].on_ground
                continue

            sensai_touched = car.ball_touches > prev_touch[0]
            opp_touched = opp.ball_touches > prev_touch[1]
            prev_touch = [car.ball_touches, opp.ball_touches]
            ball_pos, ball_vel = arena.ball.pos, arena.ball.vel
            dist = float(np.linalg.norm(ball_pos - car.pos))
            pos.step(t, car, ball_pos, sensai_touched, opp_touched,
                     overshoot_charge=breakdown.get("overshoot", 0.0))
            kicks.step(t, arena, sensai_touched, opp_touched, breakdown.get("touch", 0.0))
            hist.append(dict(t=t, car_pos=car.pos.copy(), on_ground=bool(car.on_ground),
                             on_wall=bool(is_car_on_wall(car)), boost=float(car.boost),
                             ball_pos=ball_pos.copy(), ball_vel=ball_vel.copy(),
                             s_touch=sensai_touched, o_touch=opp_touched))

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
        return goals, touches_total, events, whiffs, pos, kicks.done, goals_against


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def report_positioning(pos, touches_total):
    per100 = 100.0 / max(1, touches_total)
    print("\npositioning & recovery:")
    if pos.wall_steps:
        total_wall = sum(pos.wall_steps.values())
        detail = "  ".join(f"{surf}/{half} {n * 100.0 / max(1, pos.steps):.1f}%"
                           for (surf, half), n in sorted(pos.wall_steps.items()))
        print(f"  on a wall or side curve: {total_wall * 100.0 / max(1, pos.steps):.1f}% of steps   {detail}")
    if pos.wall_touches:
        lost = sum(pos.wall_touch_lost.values())
        print(f"  touches taken on a wall or curve: {pos.wall_touches}  opponent got the next touch after "
              f"{lost} ({100.0 * lost / pos.wall_touches:.0f}%)  by surface {dict(pos.wall_touch_lost)}")

    if pos.climbs:
        resolved = [c for c in pos.climbs if "next" in c]
        print(f"  back-wall climbs with the ball in our half: {len(pos.climbs)}  "
              f"per 100 touches {len(pos.climbs) * per100:.1f}")
        print(f"    mean {mean([c['steps'] for c in pos.climbs]) / 15.0:.2f} s  "
              f"peak z {mean([c['peak_z'] for c in pos.climbs]):.0f}  "
              f"ball {mean([c['ball_dist'] for c in pos.climbs]):.0f} uu away at entry  "
              f"next touch {dict(collections.Counter(c['next'] for c in resolved))}")

    if pos.dodges:
        landed = [d for d in pos.dodges if d["landing_wall"] is not None]
        spun = [d for d in landed if d["landing_up"] < LANDING_WHEELS_DOWN]
        print(f"  dodges while upfield of a ball in our half: {len(pos.dodges)}  "
              f"per 100 touches {len(pos.dodges) * per100:.1f}")
        print(f"    mean speed gained toward home {mean([d['gain'] for d in pos.dodges]):+.0f} uu/s  "
              f"({sum(d['gain'] < 0 for d in pos.dodges)} lost ground)  "
              f"started on a wall {sum(d['on_wall'] for d in pos.dodges)}")
        if landed:
            print(f"    landed {len(landed)}: not wheels-down {len(spun)} "
                  f"({100.0 * len(spun) / len(landed):.0f}%)  on a wall {sum(d['landing_wall'] for d in landed)}  "
                  f"mean airtime {mean([d['landed'] for d in landed]) / 15.0:.2f} s")

    if pos.overshoots:
        by = collections.Counter((o["where"], o["half"]) for o in pos.overshoots)
        print(f"  overshoots (inside {OVERSHOOT_CLOSE_UU:.0f} uu of the ball, gave it back untouched): "
              f"{len(pos.overshoots)}  per 100 touches {len(pos.overshoots) * per100:.1f}")
        print("    " + "  ".join(f"{surf}/{half} {n}" for (surf, half), n in sorted(by.items()))
              + f"  mean closest {mean([o['min_d'] for o in pos.overshoots]):.0f} uu  "
              f"ball z<200 {sum(o['ball_z'] < 200 for o in pos.overshoots)}")
        if pos.charges:
            print(f"    charged by OvershootCost: {len(pos.charges)}  mean {mean(pos.charges):.3f}  "
                  f"total {sum(pos.charges):.1f} over the run")
        carried = [o for o in pos.overshoots if o["recede"] > 100.0]
        print(f"    the car carried itself past: {len(carried)}  per 100 touches {len(carried) * per100:.1f}"
              f"   the ball outran it: {len(pos.overshoots) - len(carried)}")
        aimed = [o for o in pos.overshoots if o["aim"] > OVERSHOOT_AIMED]
        if aimed:
            aimed_by = collections.Counter((o["where"], o["half"]) for o in aimed)
            print(f"    driving straight at it (aim > {OVERSHOOT_AIMED}), so a genuine attempt rather than a "
                  f"peel-off: {len(aimed)}  per 100 touches {len(aimed) * per100:.1f}")
            print("      " + "  ".join(f"{surf}/{half} {n}" for (surf, half), n in sorted(aimed_by.items()))
                  + f"  mean closest {mean([o['min_d'] for o in aimed]):.0f} uu  "
                  f"ball z<200 {sum(o['ball_z'] < 200 for o in aimed)}")


def report_kickoffs(kicks):
    kicks = [k for k in kicks if k["first"] is not None]
    if not kicks:
        return
    by = collections.Counter(k["first"] for k in kicks)
    print(f"\nkickoffs: {len(kicks)}  first touch sensai {by['sensai']}  opponent {by['opp']}  "
          f"both on the same step {by['both']}")
    for who in ("sensai", "both", "opp"):
        rows = [k for k in kicks if k["first"] == who]
        if not rows:
            continue
        print(f"  {who:6s} {len(rows):4d}  contact at {mean([k['secs'] for k in rows]):.2f} s  "
              f"ball speed {mean([k['ball_speed'] for k in rows]):5.0f}  "
              f"toward opp goal {sum(k['ball_vy'] > 300 for k in rows):3d}  toward ours {sum(k['ball_vy'] < -300 for k in rows):3d}  "
              f"touch_r {mean([k['touch_r'] for k in rows]):.3f}  opp dist {mean([k['opp_dist'] for k in rows]):4.0f}  "
              f"our dist {mean([k['car_dist'] for k in rows]):4.0f}")
        print(f"         next touch {dict(collections.Counter(k['next'] for k in rows))}  "
              f"ball in our half 3 s later {sum(k.get('ball_y_3s', 1.0) < 0 for k in rows)}  "
              f"goals within {KICKOFF_GOAL_S:.0f} s {dict(collections.Counter(k['goal'] for k in rows if k['goal']))}")


def report_goals_against(gas):
    if not gas:
        return
    n = len(gas)
    print(f"\ngoals against: {n}  (the shot = the opponent's last touch; lead-up = the {GA_WINDOW_STEPS / 15.0:.0f} s before it)")
    print(f"  mean shot-to-goal {mean([g['shot_to_goal'] for g in gas]):.2f} s  "
          f"shot from {mean([g['shot_goal_dist'] for g in gas]):.0f} uu out  "
          f"ball speed {mean([g['ball_speed'] for g in gas]):.0f}  "
          f"SensAI {mean([g['car_goal_dist'] for g in gas]):.0f} uu from its goal, "
          f"{mean([g['dist'] for g in gas]):.0f} uu from the ball, boost {mean([g['boost'] for g in gas]):.0f}")
    print("  primary cause:")
    for cause, cnt in collections.Counter(g["cause"] for g in gas).most_common():
        rows = [g for g in gas if g["cause"] == cause]
        flags = collections.Counter(f for g in rows for f, v in g["flags"].items() if v)
        print(f"    {cause:17s} {cnt:3d} ({100.0 * cnt / n:3.0f}%)  ball dist {mean([g['dist'] for g in rows]):5.0f}  "
              f"since kickoff {mean([g['since_ko'] for g in rows]):5.1f} s  flags {dict(flags.most_common())}")
    flags = collections.Counter(f for g in gas for f, v in g["flags"].items() if v)
    print("  in the lead-up (any cause): " + "  ".join(f"{f} {c} ({100.0 * c / n:.0f}%)" for f, c in flags.most_common()))


def report(goals, touches_total, events, whiffs, pos=None, kicks=None, goals_against=None):
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

    if pos is not None:
        report_positioning(pos, touches_total)
    if kicks:
        report_kickoffs(kicks)
    if goals_against:
        report_goals_against(goals_against)


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
