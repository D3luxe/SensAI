"""
Training starts added with reward v3 (docs/reward_v3_spec.md section 4). State setting, not reward
terms, is how v3 makes hard situations common enough to learn.

  BounceDropSetter  a ball falling onto, or bouncing toward, one car      (bounce_drop_prob)
  RetreatSetter     the ball heading for one team's net, its car upfield  (retreat_prob)

Both pick the team in the situation at random and draw from wider ranges than the eval suite's
drop/bounce/retreat generators (scripts/eval_suite.py), so training never memorises the eval starts.

Frozen with v3: this file is hashed into v3's reward identity.
"""
from __future__ import annotations

import math
import random
from typing import Any, List, Tuple

import RocketSim as rsim

from env.physics_engine import ARENA_EXTENT_X, ARENA_EXTENT_Y
from env.state_setters import BaseStateSetter

CAR_Z = 17.0
BALL_REST_Z = 93.0
FIELD_X = ARENA_EXTENT_X - 400.0   # keep starts off the side walls
FIELD_Y = ARENA_EXTENT_Y - 420.0   # and the back walls


def _clip(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def _teams(rsim_arena: Any) -> List[Tuple[Any, int]]:
    return [(car, 0 if car.team == rsim.Team.BLUE else 1) for car in rsim_arena.get_cars()]


def _set_ball(rsim_arena: Any, pos, vel) -> None:
    bs = rsim_arena.ball.get_state()
    bs.pos = rsim.Vec(float(pos[0]), float(pos[1]), float(pos[2]))
    bs.vel = rsim.Vec(float(vel[0]), float(vel[1]), float(vel[2]))
    bs.ang_vel = rsim.Vec(0.0, 0.0, 0.0)
    rsim_arena.ball.set_state(bs)


def _set_car(car: Any, x: float, y: float, yaw: float, speed: float, boost: float) -> None:
    cs = rsim.CarState()
    cs.pos = rsim.Vec(_clip(x, FIELD_X), _clip(y, FIELD_Y), CAR_Z)
    cs.rot_mat = rsim.Angle(pitch=0.0, yaw=yaw, roll=0.0).as_rot_mat()
    cs.vel = rsim.Vec(math.cos(yaw) * speed, math.sin(yaw) * speed, 0.0)
    cs.ang_vel = rsim.Vec(0.0, 0.0, 0.0)
    cs.boost = boost
    car.set_state(cs)


def _yaw_to(x: float, y: float, tx: float, ty: float, spread: float) -> float:
    return math.atan2(ty - y, tx - x) + random.uniform(-spread, spread)


def _place_others(cars, protagonist, ball_xy, near_team: int, s: float) -> None:
    """Everyone but the protagonist: opponents 1500-4000 uu from the ball, teammates spread upfield of it."""
    bx, by = ball_xy
    for car, team in cars:
        if car is protagonist:
            continue
        if team == near_team:
            x, y = random.uniform(-FIELD_X, FIELD_X), by - s * random.uniform(500.0, 2500.0)
        else:
            ang, d = random.uniform(0.0, 2.0 * math.pi), random.uniform(1500.0, 4000.0)
            x, y = bx + d * math.cos(ang), by + d * math.sin(ang)
        _set_car(car, x, y, _yaw_to(x, y, bx, by, 1.0), random.uniform(0.0, 1200.0), random.uniform(0.0, 100.0))


class BounceDropSetter(BaseStateSetter):
    """
    A ball dropping near one car (half the time) or bouncing toward it from ahead (the other half).
    Targets the whiff problem: running under a falling ball, mistiming a bounce.
    """

    def reset(self, rsim_arena: Any, num_players: int) -> None:
        cars = _teams(rsim_arena)
        protagonist, team = random.choice(cars)
        s = 1.0 if team == 0 else -1.0   # the direction this team attacks

        drop = random.random() < 0.5
        ahead = random.uniform(150.0, 1000.0) if drop else random.uniform(1200.0, 3000.0)
        # the car far enough from the wall it faces that the ball starts inside the field
        cx, cy = random.uniform(-3000.0, 3000.0), s * random.uniform(-3800.0, FIELD_Y - 200.0 - ahead)
        heading = math.pi / 2 * s + random.uniform(-0.8, 0.8)
        fx, fy = math.cos(heading), math.sin(heading)
        if drop:
            # drop: ball 150-1000 uu ahead of the car, 400-1500 uu up, falling
            side = random.uniform(-500.0, 500.0)
            bx, by = cx + fx * ahead - fy * side, cy + fy * ahead + fx * side
            ball = (bx, by, random.uniform(400.0, 1500.0))
            vel = (random.uniform(-300.0, 300.0), random.uniform(-300.0, 300.0), -random.uniform(0.0, 400.0))
            speed = random.uniform(0.0, 1000.0)
        else:
            # bounce: ball 1200-3000 uu ahead, coming back toward the car and rising
            side = random.uniform(-700.0, 700.0)
            bx, by = cx + fx * ahead - fy * side, cy + fy * ahead + fx * side
            toward = random.uniform(300.0, 1200.0)
            ball = (bx, by, random.uniform(200.0, 800.0))
            vel = (-fx * toward + random.uniform(-300.0, 300.0), -fy * toward + random.uniform(-300.0, 300.0),
                   random.uniform(0.0, 700.0))
            speed = random.uniform(200.0, 1200.0)

        bx, by = _clip(ball[0], FIELD_X), _clip(ball[1], FIELD_Y)
        _set_ball(rsim_arena, (bx, by, ball[2]), vel)
        _set_car(protagonist, cx, cy, _yaw_to(cx, cy, bx, by, 0.6), speed, random.uniform(0.0, 100.0))
        _place_others(cars, protagonist, (bx, by), team, s)


class RetreatSetter(BaseStateSetter):
    """
    The ball rolling or bouncing toward one team's net with that team's car upfield of it, facing
    home or away. Targets the retreat problem: getting back goal-side in time, boost versus flips.
    """

    def reset(self, rsim_arena: Any, num_players: int) -> None:
        cars = _teams(rsim_arena)
        protagonist, team = random.choice(cars)
        s = 1.0 if team == 0 else -1.0
        own_goal_y = -ARENA_EXTENT_Y * s

        bx, by = random.uniform(-2000.0, 2000.0), random.uniform(-2000.0, 1500.0) * s
        gx = random.uniform(-900.0, 900.0)
        dx, dy = gx - bx, own_goal_y - by
        n = math.hypot(dx, dy)
        v = random.uniform(800.0, 2000.0)
        if random.random() < 0.7:
            ball, vz = (bx, by, BALL_REST_Z), 0.0
        else:
            ball, vz = (bx, by, random.uniform(150.0, 600.0)), random.uniform(-300.0, 300.0)
        _set_ball(rsim_arena, ball, (dx / n * v, dy / n * v, vz))

        # the car 1000-3500 uu upfield of the ball, facing home or upfield
        cx, cy = bx + random.uniform(-1200.0, 1200.0), by + s * random.uniform(1000.0, 3500.0)
        facing_home = random.random() < 0.5
        yaw = (-math.pi / 2 * s if facing_home else math.pi / 2 * s) + random.uniform(-0.7, 0.7)
        _set_car(protagonist, cx, cy, yaw, random.uniform(0.0, 1400.0), random.uniform(0.0, 100.0))

        for car, t in cars:
            if car is protagonist:
                continue
            if t != team and random.random() < 0.6:
                # an opponent racing the ball in from behind it
                x, y = bx + random.uniform(-1500.0, 1500.0), by + s * random.uniform(500.0, 2500.0)
                _set_car(car, x, y, -math.pi / 2 * s + random.uniform(-0.5, 0.5),
                         random.uniform(500.0, 1500.0), random.uniform(0.0, 100.0))
            elif t != team:
                # or somewhere in its own half, out of the race
                x, y = random.uniform(-3000.0, 3000.0), s * random.uniform(1500.0, 4400.0)
                _set_car(car, x, y, _yaw_to(x, y, bx, by, 1.0), random.uniform(0.0, 800.0), random.uniform(0.0, 100.0))
            else:
                x, y = random.uniform(-FIELD_X, FIELD_X), cy + s * random.uniform(-1000.0, 1000.0)
                _set_car(car, x, y, random.uniform(-math.pi, math.pi), random.uniform(0.0, 1200.0), random.uniform(0.0, 100.0))
