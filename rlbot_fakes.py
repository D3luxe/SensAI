"""
Minimal stand-ins for the RLBot v5 flat types, for tests and offline scripts.

bot.py reads live match state straight off a `GamePacket`, so exercising it without a running
game needs objects of the same shape. These helpers build them, and keep the field names in one
place so a schema change is a single edit rather than one per test.
"""

from __future__ import annotations

from bot import AirState, MatchPhase

# Spacing of the slices RLBot v5 publishes: 720 of them, 6 seconds at 120 Hz.
V5_PREDICTION_DT = 1.0 / 120.0


class Struct:
    """Attribute bag standing in for a flatbuffers table or struct."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def vec(x=0.0, y=0.0, z=0.0) -> Struct:
    return Struct(x=float(x), y=float(y), z=float(z))


def physics(location=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0)) -> Struct:
    return Struct(
        location=vec(*location),
        velocity=vec(*velocity),
        rotation=Struct(pitch=float(rotation[0]), yaw=float(rotation[1]), roll=float(rotation[2])),
        angular_velocity=vec(*angular_velocity),
    )


def car(team=0, boost=33.3, air_state=AirState.OnGround, has_jumped=False,
        has_double_jumped=False, has_dodged=False, is_supersonic=False, **phys) -> Struct:
    return Struct(
        team=int(team),
        boost=float(boost),
        air_state=air_state,
        has_jumped=bool(has_jumped),
        has_double_jumped=bool(has_double_jumped),
        has_dodged=bool(has_dodged),
        is_supersonic=bool(is_supersonic),
        physics=physics(**phys),
    )


def ball(**phys) -> Struct:
    return Struct(physics=physics(**phys))


def packet(players, balls, match_phase=MatchPhase.Active, boost_pads=None,
           seconds_elapsed=0.0) -> Struct:
    return Struct(
        players=list(players),
        balls=list(balls),
        boost_pads=list(boost_pads) if boost_pads is not None else [],
        match_info=Struct(match_phase=match_phase, seconds_elapsed=float(seconds_elapsed)),
    )


def ball_prediction(locations, start_seconds=0.0, dt=V5_PREDICTION_DT) -> Struct:
    """
    Build a prediction from an iterable of (x, y, z), spaced `dt` apart.

    Pass `dt` explicitly to simulate a framework publishing at some other rate; bot.py measures
    the spacing off `game_seconds` rather than assuming it, so the mapped horizons should come out
    the same either way.
    """
    slices = [
        Struct(game_seconds=start_seconds + i * dt, physics=physics(location=loc))
        for i, loc in enumerate(locations)
    ]
    return Struct(slices=slices)
