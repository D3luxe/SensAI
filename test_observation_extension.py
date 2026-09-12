"""
Regression tests for the 14 features appended to the observation at indices 94..107.

The block covers three things the policy previously could not see:
  - its own flip and air state, which it is in for roughly a quarter of every second
  - scalar magnitudes a linear layer approximates poorly (speeds, opponent distance)
  - whether the opponent can still challenge, or currently exists at all

What these tests protect:
  1. Every appended feature is mirror-invariant, verified against a physical X reflection
     rather than against the mask that is supposed to describe it.
  2. None of them is a constant. Four of the underlying CarState fields were never synced
     from RocketSim and read as identical zeros until that was fixed; a feature wired to a
     dead field is dead weight that no metric would ever surface.
  3. The indices the action mask reads by hard-coded position are unchanged. Appending is
     safe; inserting anywhere below 22 silently breaks masking.
"""

import math
import unittest

import numpy as np

from env.physics_engine import CarState, BallState, CAR_MAX_SPEED, BALL_MAX_SPEED
from env.observations import (
    OBS_DIM,
    OBS_MIRROR_MASK_NP,
    DefaultObservationBuilder,
)

APPENDED = slice(94, 108)


class MockArena:
    def __init__(self, ball, cars):
        self.ball = ball
        self.cars = cars
        self.boost_pads = []


def _rot(pitch, yaw, roll):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array([
        [cp * cy, cp * sy, sp],
        [cy * sp * sr - cr * sy, cr * cy + sp * sr * sy, -cp * sr],
        [-cr * cy * sp - sr * sy, cy * sr - cr * sp * sy, cp * cr],
    ], dtype=np.float32)


class TestAppendedFeatureSymmetry(unittest.TestCase):
    def test_appended_features_survive_physical_reflection(self):
        """Reflect the world across X and confirm the appended block is unchanged.

        The existing parity test in test_symmetry_and_rewards covers the full vector, but
        it leaves the new fields at their defaults, so the appended indices pass there
        trivially. Here every one of them is given a distinctive non-zero value first.
        """
        builder = DefaultObservationBuilder(symmetric=True)
        pitch, yaw, roll = 0.35, 1.2, -0.45
        rm = _rot(pitch, yaw, roll)

        def make(sign):
            car = CarState(
                id=0, team=0,
                pos=np.array([sign * 1200.0, -800.0, 450.0], dtype=np.float32),
                vel=np.array([sign * -400.0, 600.0, 200.0], dtype=np.float32),
                ang_vel=np.array([1.5, sign * -2.0, sign * 0.8], dtype=np.float32),
                boost=65.0, on_ground=False, has_jump=False, has_flip=True,
            )
            r = rm.copy()
            if sign < 0:
                r[0, 0] = -rm[0, 0]
                r[2, 0] = -rm[2, 0]
                r[1, 1] = -rm[1, 1]
                r[1, 2] = -rm[1, 2]
            car.rot_mat = r
            # The appended self state, all distinctive.
            car.is_dodging = True
            car.flip_timer = 0.8
            car.air_timer = 1.3
            car.has_double_jumped = True
            car.is_supersonic = True

            opp = CarState(
                id=1, team=1,
                pos=np.array([sign * -600.0, 1500.0, 17.0], dtype=np.float32),
                vel=np.array([sign * 300.0, -200.0, 0.0], dtype=np.float32),
                ang_vel=np.array([0.0, 0.0, sign * -0.5], dtype=np.float32),
                boost=33.0, on_ground=True, has_flip=True,
            )
            opp.is_dodging = True
            opp.is_supersonic = True
            opp.demoed = True
            opp.demo_timer = 1.8

            ball = BallState(
                pos=np.array([sign * 300.0, 1000.0, 300.0], dtype=np.float32),
                vel=np.array([sign * 500.0, -700.0, 150.0], dtype=np.float32),
                ang_vel=np.array([-1.2, sign * 0.9, sign * -2.5], dtype=np.float32),
            )
            return builder.build_obs(car, MockArena(ball, [car, opp]))

        obs = make(+1.0)
        obs_reflected = make(-1.0)

        # The mask claims every appended feature is invariant. Check the physics agrees.
        self.assertTrue(
            np.all(OBS_MIRROR_MASK_NP[APPENDED] == 1.0),
            "the appended block is declared mirror-invariant",
        )
        np.testing.assert_allclose(
            obs_reflected[APPENDED], obs[APPENDED], atol=1e-5,
            err_msg="appended features must be unchanged by a physical X reflection",
        )
        # And they must not be trivially zero, or the assertion above proves nothing.
        self.assertGreater(
            int(np.count_nonzero(obs[APPENDED])), 10,
            "fixture must actually exercise the appended block",
        )

    def test_action_mask_indices_are_unchanged(self):
        """The model reads obs[18..21] by hard-coded index to build the action mask."""
        builder = DefaultObservationBuilder(symmetric=True)
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            boost=42.0, on_ground=True, has_jump=False, has_flip=True,
        )
        opp = CarState(id=1, team=1, pos=np.array([0.0, 2000.0, 17.0], dtype=np.float32))
        ball = BallState(pos=np.array([0.0, 500.0, 93.0], dtype=np.float32))
        obs = builder.build_obs(car, MockArena(ball, [car, opp]))

        self.assertAlmostEqual(float(obs[18]), 0.42, places=5, msg="index 18 must stay boost")
        self.assertEqual(float(obs[19]), 1.0, "index 19 must stay on_ground")
        self.assertEqual(float(obs[20]), 0.0, "index 20 must stay has_jump")
        self.assertEqual(float(obs[21]), 1.0, "index 21 must stay has_flip")


class TestAppendedFeatureSemantics(unittest.TestCase):
    def _obs(self, car, opp, ball=None):
        builder = DefaultObservationBuilder(symmetric=True)
        if ball is None:
            ball = BallState(pos=np.array([0.0, 500.0, 93.0], dtype=np.float32))
        return builder.build_obs(car, MockArena(ball, [car, opp]))

    def _pair(self, opp_vel):
        car = CarState(
            id=0, team=0,
            pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
            vel=np.zeros(3, dtype=np.float32), on_ground=True,
        )
        opp = CarState(
            id=1, team=1,
            pos=np.array([0.0, 2000.0, 17.0], dtype=np.float32),
            vel=np.array(opp_vel, dtype=np.float32), on_ground=True,
        )
        return car, opp

    def test_closing_speed_sign(self):
        """Positive when the gap is shrinking. A sign error here would invert the meaning
        of the feature without changing its distribution, which no variance check catches."""
        car, opp = self._pair([0.0, -1000.0, 0.0])       # opponent driving at us
        self.assertGreater(self._obs(car, opp)[102], 0.0)

        car, opp = self._pair([0.0, 1000.0, 0.0])        # opponent retreating
        self.assertLess(self._obs(car, opp)[102], 0.0)

        car, opp = self._pair([1000.0, 0.0, 0.0])        # purely lateral
        self.assertAlmostEqual(float(self._obs(car, opp)[102]), 0.0, places=5)

    def test_opponent_distance_scale(self):
        car, opp = self._pair([0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(self._obs(car, opp)[101]), 2000.0 / 6000.0, places=4)

    def test_speed_magnitudes(self):
        car, opp = self._pair([0.0, 0.0, 0.0])
        car.vel = np.array([0.0, 1150.0, 0.0], dtype=np.float32)
        ball = BallState(pos=np.array([0.0, 500.0, 93.0], dtype=np.float32),
                         vel=np.array([3000.0, 0.0, 0.0], dtype=np.float32))
        obs = self._obs(car, opp, ball)
        self.assertAlmostEqual(float(obs[99]), 1150.0 / CAR_MAX_SPEED, places=4)
        self.assertAlmostEqual(float(obs[100]), 3000.0 / BALL_MAX_SPEED, places=4)

    def test_demoed_opponent_reads_as_absent(self):
        """The whole point of the demo features: an open net must be visible as one."""
        car, opp = self._pair([0.0, 0.0, 0.0])
        alive = self._obs(car, opp)
        self.assertEqual(float(alive[106]), 1.0)
        self.assertEqual(float(alive[107]), 0.0)

        opp.demoed = True
        opp.demo_timer = 1.5
        dead = self._obs(car, opp)
        self.assertEqual(float(dead[106]), 0.0, "a demoed opponent must read as not alive")
        self.assertAlmostEqual(float(dead[107]), 0.5, places=4, msg="timer normalized by 3 s")

    def test_missing_opponent_zeroes_the_dependent_features(self):
        car = CarState(id=0, team=0, pos=np.array([0.0, 0.0, 17.0], dtype=np.float32))
        ball = BallState(pos=np.array([0.0, 500.0, 93.0], dtype=np.float32))
        builder = DefaultObservationBuilder(symmetric=True)
        obs = builder.build_obs(car, MockArena(ball, [car]))
        np.testing.assert_array_equal(obs[101:108], np.zeros(7, dtype=np.float32))
        self.assertTrue(np.isfinite(obs).all())


class TestNoAppendedFeatureIsDead(unittest.TestCase):
    """Four of the underlying CarState fields were never synced from RocketSim and read as
    identical zeros. A feature wired to one of those is invisible dead weight, so the
    self-state block is checked against the live backend rather than a mock."""

    def test_self_state_features_vary_in_a_real_rollout(self):
        from env.rocket_env import VectorizedRocketEnv

        vec = VectorizedRocketEnv(num_envs=8, max_episode_steps=600, baseline_opponent_ratio=0.0)
        obs = vec.reset()
        P = vec.num_players_per_env
        seen = [obs.reshape(-1, vec.obs_dim).copy()]
        rng = np.random.default_rng(0)
        for _ in range(240):
            a = rng.uniform(-1.0, 1.0, size=(8, P, vec.act_dim)).astype(np.float32)
            obs, _, _, _ = vec.step(a)
            seen.append(obs.reshape(-1, vec.obs_dim).copy())
        stacked = np.concatenate(seen, axis=0)

        self.assertEqual(stacked.shape[1], OBS_DIM)
        self.assertTrue(np.isfinite(stacked).all(), "observation must never go non-finite")

        # Demo state is genuinely rare (about 0.4% of car-steps), so it is verified
        # deterministically in TestAppendedFeatureSemantics instead of by sampling.
        must_vary = {
            94: "is_dodging", 95: "flip_timer", 96: "air_timer", 98: "is_supersonic",
            99: "|v_car|", 100: "|v_ball|", 101: "opponent distance", 102: "closing speed",
        }
        for idx, name in must_vary.items():
            col = stacked[:, idx]
            self.assertGreater(
                float(col.std()), 1e-6,
                "feature {} ({}) is constant across the rollout, so it is wired to a field "
                "the backend never populates".format(idx, name),
            )


class TestDeployedBotPopulatesAppendedFields(unittest.TestCase):
    """The in-game path builds its own CarState from the RLBot packet.

    Any appended field it forgets to set defaults to zero, so the deployed policy would see
    a car that is never flipping, never supersonic and facing an opponent that never dies,
    while training saw those states 23%, 4% and 0.5% of the time. That mismatch is invisible
    in every training metric, which is exactly why it is asserted here.
    """

    def _packet(self, air_state, is_supersonic=False, opp_demo_timeout=-1.0):
        import rlbot_fakes
        from bot import AirState, MatchPhase

        class S:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

        def mkcar(y, team, air, sup, demo):
            return S(
                team=team, boost=30.0, air_state=air, has_jumped=True,
                has_double_jumped=False, has_dodged=False, is_supersonic=sup,
                demolished_timeout=demo,
                physics=S(
                    location=S(x=0.0, y=y, z=300.0),
                    velocity=S(x=0.0, y=0.0, z=0.0),
                    rotation=S(pitch=0.0, yaw=math.pi / 2, roll=0.0),
                    angular_velocity=S(x=0.0, y=0.0, z=0.0),
                ),
            )

        me = mkcar(0.0, 0, air_state, is_supersonic, -1.0)
        opp = mkcar(2000.0, 1, AirState.OnGround, False, opp_demo_timeout)
        ball = S(physics=S(location=S(x=0.0, y=500.0, z=93.0),
                           velocity=S(x=0.0, y=0.0, z=0.0),
                           angular_velocity=S(x=0.0, y=0.0, z=0.0)))
        return S(players=[me, opp], balls=[ball], boost_pads=[],
                 match_info=S(match_phase=MatchPhase.Active)), rlbot_fakes

    def _run(self, air_state, **kw):
        from bot import SenseiRLBot
        packet, rlbot_fakes = self._packet(air_state, **kw)
        bot = SenseiRLBot("TestBot", 0, 0)
        bot.ball_prediction = rlbot_fakes.ball_prediction(
            [(0.0, 0.0, 93.0) for _ in range(720)]
        )
        bot.ticks_since_last_action = 8
        bot.get_output(packet)
        self.assertIsNotNone(bot.latest_obs)
        self.assertEqual(len(bot.latest_obs), OBS_DIM)
        return bot, packet

    def test_dodging_reaches_the_observation(self):
        from bot import AirState
        bot, _ = self._run(AirState.Dodging)
        self.assertEqual(float(bot.latest_obs[94]), 1.0, "is_dodging must reach index 94")

    def test_supersonic_reaches_the_observation(self):
        from bot import AirState
        bot, _ = self._run(AirState.InAir, is_supersonic=True)
        self.assertEqual(float(bot.latest_obs[98]), 1.0, "is_supersonic must reach index 98")

    def test_air_timer_accumulates_and_resets_on_landing(self):
        from bot import AirState, SenseiRLBot
        import rlbot_fakes
        packet, _ = self._packet(AirState.InAir)
        bot = SenseiRLBot("TestBot", 0, 0)
        bot.ball_prediction = rlbot_fakes.ball_prediction([(0.0, 0.0, 93.0) for _ in range(720)])
        for _ in range(60):
            bot.ticks_since_last_action = 8
            bot.get_output(packet)
        airborne = float(bot.latest_obs[96])
        self.assertGreater(airborne, 0.0, "air_timer must accumulate while airborne")

        packet.players[0].air_state = AirState.OnGround
        bot.ticks_since_last_action = 8
        bot.get_output(packet)
        self.assertEqual(float(bot.latest_obs[96]), 0.0, "air_timer must reset on landing")

    def test_demoed_opponent_reaches_the_observation(self):
        from bot import AirState
        bot, _ = self._run(AirState.OnGround, opp_demo_timeout=1.5)
        self.assertEqual(float(bot.latest_obs[106]), 0.0, "a demoed opponent must read as not alive")
        self.assertAlmostEqual(float(bot.latest_obs[107]), 0.5, places=3)


class TestDeployedJumpSequencer(unittest.TestCase):
    """A dodge must never fire inside the policy step that left the ground.

    Real Rocket League lifts the wheels about two ticks after the jump press, so the
    airborne branch of the substep sequencer would fire at substep 2 using the action chosen
    at substep 0 -- while the car was still grounded, where the policy's pitch/yaw/roll are
    masked to exactly zero. The dodge came out directionless, an empty stall that burned the
    flip for nothing.

    RocketSim holds is_on_ground true for six ticks after the press, so training's window
    closes before the car is airborne and the flip survives to the next step where the
    rotational channels carry a real direction. This test pins the deployed bot to that same
    behaviour.
    """

    def _bot_and_packet(self, air_state, has_jumped=False):
        import rlbot_fakes
        from bot import SenseiRLBot, AirState, MatchPhase

        class S:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

        def mkcar(y, team, air, jumped):
            return S(team=team, boost=50.0, air_state=air, has_jumped=jumped,
                     has_double_jumped=False, has_dodged=False, is_supersonic=False,
                     demolished_timeout=-1.0,
                     physics=S(location=S(x=0.0, y=y, z=17.0 if air == AirState.OnGround else 60.0),
                               velocity=S(x=0.0, y=900.0, z=0.0),
                               rotation=S(pitch=0.0, yaw=math.pi / 2, roll=0.0),
                               angular_velocity=S(x=0.0, y=0.0, z=0.0)))

        me = mkcar(0.0, 0, air_state, has_jumped)
        opp = mkcar(2500.0, 1, AirState.OnGround, False)
        ball = S(physics=S(location=S(x=0.0, y=800.0, z=93.0),
                           velocity=S(x=0.0, y=0.0, z=0.0),
                           angular_velocity=S(x=0.0, y=0.0, z=0.0)))
        packet = S(players=[me, opp], balls=[ball], boost_pads=[],
                   match_info=S(match_phase=MatchPhase.Active))
        bot = SenseiRLBot("TestBot", 0, 0)
        bot.ball_prediction = rlbot_fakes.ball_prediction(
            [(0.0, 0.0, 93.0) for _ in range(720)])
        return bot, packet

    def test_no_dodge_inside_the_step_that_left_the_ground(self):
        from bot import AirState
        bot, packet = self._bot_and_packet(AirState.OnGround)
        bot.dodge_cooldown = 0

        # Substep 0 on the ground: this is where the action, and its zeroed rotation, is chosen.
        bot.ticks_since_last_action = bot.tick_skip
        bot.get_output(packet)
        self.assertTrue(bot._step_began_grounded,
                        "the step must be latched as having begun on the ground")

        # The wheels leave the turf mid-step, as they do in the real game.
        packet.players[0].air_state = AirState.InAir
        packet.players[0].has_jumped = True
        fired = False
        for _ in range(6):
            c = bot.get_output(packet)
            fired = fired or bool(c.jump)
        self.assertFalse(
            fired,
            "a dodge fired inside the liftoff step; its direction comes from a grounded "
            "action whose pitch/yaw/roll are masked to zero, so it is an empty stall")

    def test_dodge_is_available_on_a_step_that_began_airborne(self):
        from bot import AirState
        bot, packet = self._bot_and_packet(AirState.InAir, has_jumped=True)
        bot.dodge_cooldown = 0
        bot.ticks_since_last_action = bot.tick_skip
        bot.get_output(packet)
        self.assertFalse(bot._step_began_grounded)
        # The gate must not block a genuine airborne dodge; only the liftoff step is excluded.
        self.assertFalse(bot._step_began_grounded,
                         "an airborne step must stay eligible to dodge")


class TestDeployDiagnostics(unittest.TestCase):
    """The in-game overlay exists to catch a third train/deploy mismatch.

    Two have already cost real training time: the appended observation fields defaulting to
    zero in the deployed bot, and the dodge firing inside the liftoff step. Both were silent
    in every training metric. The out-of-distribution check makes the next one visible while
    it is happening.
    """

    def _bot(self):
        from bot import SenseiRLBot
        return SenseiRLBot("TestBot", 0, 0)

    def test_training_envelope_is_available(self):
        b = self._bot()
        self.assertIsNotNone(
            b._obs_limit,
            "data/obs_stats_train.json must ship with the bot or the check is inert")
        self.assertEqual(len(b._obs_limit), OBS_DIM)

    def test_no_false_positives_on_real_training_observations(self):
        """The check this replaced fired on 19.7% of training steps.

        It compared each feature against the 0.1/99.9 training percentiles, so bounded
        quantities tripped it constantly: angular velocity pinned at Rocket League's 5.5 rad/s
        cap, and unit-vector components at 1.0. Both are saturation, not anomalies, and both
        flashed red in a live match. A monitor that cries wolf one step in five gets ignored,
        which is worse than not having one. Any replacement must be silent on real data.
        """
        import json
        b = self._bot()
        stats = json.load(open("data/obs_stats_train.json"))
        mean = np.asarray(stats["mean"], dtype=np.float32)
        mn = np.asarray(stats["min"], dtype=np.float32)
        mx = np.asarray(stats["max"], dtype=np.float32)

        for label, sample in (("mean", mean), ("min", mn), ("max", mx)):
            self.assertEqual(
                b._diagnose_obs(sample), "",
                "the %s of the training distribution must never be flagged" % label)

        # And every per-feature extreme in isolation, which is where saturation lives.
        for i in range(OBS_DIM):
            for extreme in (mn[i], mx[i]):
                probe = mean.copy()
                probe[i] = extreme
                self.assertEqual(
                    b._diagnose_obs(probe), "",
                    "feature %d at its training extreme %.3f was flagged" % (i, extreme))

    def test_non_finite_values_are_always_reported(self):
        import json
        b = self._bot()
        obs = np.asarray(json.load(open("data/obs_stats_train.json"))["mean"], dtype=np.float32)
        obs[37] = np.nan
        self.assertIn("NaN", b._diagnose_obs(obs))

    def test_scale_errors_are_reported(self):
        """A units mistake or a shifted feature order produces magnitudes training never saw."""
        import json
        b = self._bot()
        obs = np.asarray(json.load(open("data/obs_stats_train.json"))["mean"], dtype=np.float32)
        obs[3] = 40.0
        msg = b._diagnose_obs(obs)
        self.assertIn("SCALE", msg)
        self.assertIn("3=", msg)

    def test_overlay_draws_on_screen(self):
        """The first overlay drew nothing in a live match and left no trace of why.

        draw_string_2d takes SCREEN FRACTIONS, not pixels: 0.1 means a tenth of the screen.
        The original passed x=12, y=30, which is 1200% across and 3000% down, so every line
        landed far outside the viewport. The old test only asserted "does not raise", which
        that bug satisfies perfectly. This one mirrors the real renderer -- fractional
        coordinates, a Color foreground, begin/end group lifecycle -- and checks the lines
        actually land somewhere a person can see.
        """
        import json
        from env.physics_engine import CarState, BallState

        class FakeColor:
            def __init__(self, *a, **k):
                pass

        class FakeRenderer:
            white = FakeColor(255, 255, 255)
            red = FakeColor(255, 0, 0)

            def __init__(self):
                self._w = 1.0
                self._h = 1.0
                self.group = None
                self.draws = []

            def set_resolution(self, w, h):
                self._w, self._h = 1.0 / w, 1.0 / h

            def can_render(self):
                return True

            def begin_rendering(self, gid="default"):
                assert self.group is None, "begin_rendering called twice without end"
                self.group = gid

            def end_rendering(self):
                assert self.group is not None, "end_rendering without begin"
                self.group = None

            def draw_string_2d(self, text, x, y, scale, foreground=None,
                               background=None, h_align=None, v_align=None):
                assert self.group is not None, "draw outside a render group"
                self.draws.append((text, x * self._w, y * self._h))

        b = self._bot()
        r = FakeRenderer()
        b.renderer = r
        b.latest_obs = np.asarray(
            json.load(open("data/obs_stats_train.json"))["mean"], dtype=np.float32)
        car = CarState(id=0, team=0, pos=np.array([0.0, 0.0, 17.0], dtype=np.float32),
                       boost=42.0, on_ground=True, has_flip=True)
        ball = BallState(pos=np.array([0.0, 1500.0, 93.0], dtype=np.float32))
        b._render_diagnostics(car, ball, np.zeros(8, dtype=np.float32))

        self.assertGreaterEqual(len(r.draws), 5, "the overlay must actually draw something")
        self.assertIsNone(r.group, "the render group must be closed")
        for text, x, y in r.draws:
            self.assertTrue(
                0.0 <= x <= 1.0 and 0.0 <= y <= 1.0,
                "line %r drawn at (%.3f, %.3f), outside the screen" % (text, x, y))
        self.assertFalse(b._diag_render_failed, "the overlay reported an internal error")

    def test_overlay_failure_is_logged_not_swallowed(self):
        """A silent diagnostic is worse than none: it hides its own breakage."""
        from env.physics_engine import CarState, BallState

        class Broken:
            def set_resolution(self, *a, **k):
                pass

            def can_render(self):
                return True

            def begin_rendering(self, *a, **k):
                raise RuntimeError("signature mismatch")

            def end_rendering(self, *a, **k):
                pass

        b = self._bot()
        b.renderer = Broken()
        b.latest_obs = np.zeros(OBS_DIM, dtype=np.float32)
        car = CarState(id=0, team=0, pos=np.array([0.0, 0.0, 17.0], dtype=np.float32))
        ball = BallState(pos=np.array([0.0, 500.0, 93.0], dtype=np.float32))
        b._render_diagnostics(car, ball, np.zeros(8, dtype=np.float32))
        self.assertTrue(b._diag_render_failed,
                        "an overlay error must be recorded so it can be logged, not hidden")

    def test_a_broken_renderer_cannot_take_the_bot_down(self):
        """RLBot's rendering signatures vary across builds. A diagnostic that crashes the
        match is worse than no diagnostic."""
        from env.physics_engine import CarState, BallState

        class Broken:
            def begin_rendering(self, *a, **k):
                raise RuntimeError("signature mismatch")

            def end_rendering(self, *a, **k):
                pass

        b = self._bot()
        b.renderer = Broken()
        b.latest_obs = np.zeros(OBS_DIM, dtype=np.float32)
        car = CarState(id=0, team=0, pos=np.array([0.0, 0.0, 17.0], dtype=np.float32))
        ball = BallState(pos=np.array([0.0, 500.0, 93.0], dtype=np.float32))
        b._render_diagnostics(car, ball, np.zeros(8, dtype=np.float32))  # must not raise

    def test_no_renderer_attached_is_handled(self):
        from env.physics_engine import CarState, BallState
        b = self._bot()
        b.renderer = None
        b.latest_obs = np.zeros(OBS_DIM, dtype=np.float32)
        car = CarState(id=0, team=0, pos=np.array([0.0, 0.0, 17.0], dtype=np.float32))
        ball = BallState(pos=np.array([0.0, 500.0, 93.0], dtype=np.float32))
        b._render_diagnostics(car, ball, np.zeros(8, dtype=np.float32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
