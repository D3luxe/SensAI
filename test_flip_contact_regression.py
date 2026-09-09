"""
Regression tests for the flip-reluctance reward audit.

Guarantees:
  1. A forward flip that CONNECTS with the ball is not misclassified as a backflip.
     (delta_v is measured across a full env step; the collision impulse reverses it.)
  2. A connecting flip scores strictly higher than the identical flip that whiffs.
  3. Horizontal basis projections are normalized, so alignments and local-frame
     distances do not silently shrink when the car is pitched.
  4. just_dodged is an edge, not a level that latches until touchdown.
  5. BoostReward situational penalties scale monotonically with boost_lose_weight.
  6. The goal-line save requires an intervening opponent touch (anti-farming).
  7. The touchdown crash penalty is exempt during a close ball engagement.
  8. Boost transit shaping has no discontinuity at the moment of liftoff.

These construct CarState directly rather than stepping the simulator, so they run
without a RocketSim backend.
"""

import unittest
import numpy as np

from env.physics_engine import RocketSimArena
from env.rewards import JumpBridgeReward, AirRollRecoveryReward, BoostReward, GoalReward
try:
    from env.rewards import unit_horiz
except ImportError:  # pre-fix tree
    unit_horiz = None

NOSE_PLUS_Y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)


def flip_scenario(arena, gap=400.0, ball_z=93.0, ball_y=1000.0,
                  vel_before=1400.0, opp_y=5000.0):
    """Car airborne, nose +Y, flip available, closing on the ball at `gap` uu."""
    arena.ball.pos = np.array([0.0, ball_y, ball_z], dtype=np.float32)
    arena.ball.vel = np.zeros(3, dtype=np.float32)
    car = arena.cars[0]
    car.team = 0
    car.pos = np.array([0.0, ball_y - gap, 60.0], dtype=np.float32)
    car.vel = np.array([0.0, vel_before, 0.0], dtype=np.float32)
    car.rot_mat = NOSE_PLUS_Y.copy()
    car.on_ground = False
    car.has_flip = True
    opp = arena.cars[1]
    opp.team = 1
    opp.pos = np.array([0.0, opp_y, 17.0], dtype=np.float32)
    opp.vel = np.zeros(3, dtype=np.float32)
    return car


def execute_flip(car, arena, connects: bool):
    """Advance one env step's worth of state: the flip fires, and optionally strikes the ball."""
    car.has_flip = False
    car.just_dodged = True
    if connects:
        # Contact bleeds forward speed off the car and dumps it into the ball.
        car.vel = np.array([0.0, 900.0, 0.0], dtype=np.float32)
        car.ball_touches += 1
        arena.ball.vel = np.array([0.0, 1800.0, 200.0], dtype=np.float32)
    else:
        # Clean dodge impulse, no contact.
        car.vel = np.array([0.0, 1900.0, 0.0], dtype=np.float32)
    act = np.zeros(8, dtype=np.float32)
    act[0] = 1.0   # throttle
    act[2] = 1.0   # pitch +1 == front flip (see env/actions.py row 13)
    act[5] = 1.0   # jump
    return act


class TestFlipContactRegression(unittest.TestCase):
    def setUp(self):
        self.arena = RocketSimArena()
        self.arena.reset()

    # ── 1 & 2. The headline defect ────────────────────────────────────────────
    def test_connecting_forward_flip_is_rewarded(self):
        for gap in (300.0, 400.0, 600.0):
            with self.subTest(gap=gap):
                arena = RocketSimArena(); arena.reset()
                car = flip_scenario(arena, gap=gap)
                jb = JumpBridgeReward(weight=0.8); jb.reset(arena)
                act = execute_flip(car, arena, connects=True)
                r = jb.get_reward(car, arena, act, False, None)
                self.assertGreater(
                    r, 0.0,
                    f"A front flip into a ground ball at gap={gap} must be rewarded; "
                    f"got {r:.3f}. A negative value means the contact-contaminated "
                    f"delta_v classifier has flagged it as a backflip again.")

    def test_contact_does_not_change_flip_classification(self):
        """
        The same front flip, evaluated twice: once where contact bled 500 uu/s off the car
        (so the measured delta_v points BACKWARD) and once where contact barely changed the
        car's velocity (so delta_v is below the 80 uu/s gate and the stick input is used).
        Both are front flips into the ball and must score the same. Before the fix the
        contaminated one was charged an extra -0.64 as a misclassified backflip.
        """
        def flip_reward(vel_after):
            arena = RocketSimArena(); arena.reset()
            car = flip_scenario(arena, gap=400.0)
            jb = JumpBridgeReward(weight=0.8); jb.reset(arena)
            car.has_flip = False
            car.just_dodged = True
            car.vel = np.array([0.0, vel_after, 0.0], dtype=np.float32)
            car.ball_touches += 1
            arena.ball.vel = np.array([0.0, 1800.0, 200.0], dtype=np.float32)
            act = np.zeros(8, dtype=np.float32)
            act[0] = 1.0; act[2] = 1.0; act[5] = 1.0
            return jb.get_reward(car, arena, act, False, None)

        # Both retain nearly the same speed, so speed-scaled bonuses are matched; only the
        # measured impulse direction differs (one above the 80 uu/s gate, one below).
        heavy_contact = flip_reward(1300.0)   # delta_v = -100 uu/s: reads as a backflip
        light_contact = flip_reward(1390.0)   # delta_v = -10 uu/s: below the gate
        self.assertAlmostEqual(
            heavy_contact, light_contact, delta=0.06,
            msg=f"Flip classification must not depend on how hard the ball was struck: "
                f"heavy contact scored {heavy_contact:.3f} vs {light_contact:.3f} for a light "
                f"touch. The delta_v classifier is reading the collision impulse.")

    def test_connecting_flip_outscores_identical_whiff(self):
        arena_hit = RocketSimArena(); arena_hit.reset()
        car_hit = flip_scenario(arena_hit, gap=400.0)
        jb_hit = JumpBridgeReward(weight=0.8); jb_hit.reset(arena_hit)
        r_hit = jb_hit.get_reward(car_hit, arena_hit, execute_flip(car_hit, arena_hit, True), False, None)

        arena_miss = RocketSimArena(); arena_miss.reset()
        car_miss = flip_scenario(arena_miss, gap=400.0)
        jb_miss = JumpBridgeReward(weight=0.8); jb_miss.reset(arena_miss)
        r_miss = jb_miss.get_reward(car_miss, arena_miss, execute_flip(car_miss, arena_miss, False), False, None)

        self.assertGreater(
            r_hit, r_miss,
            f"Connecting flip ({r_hit:.3f}) must outscore the identical whiff "
            f"({r_miss:.3f}); otherwise the gradient rewards missing the ball.")

    def test_bouncing_ball_below_250uu_is_a_valid_aerial_attempt(self):
        """A pitch-up attempt at a ball at 180 uu must not be charged as a backflip."""
        arena = RocketSimArena(); arena.reset()
        car = flip_scenario(arena, gap=350.0, ball_z=180.0)
        car.pos[2] = 40.0
        jb = JumpBridgeReward(weight=0.8); jb.reset(arena)
        car.has_flip = False
        car.just_dodged = True
        car.vel = np.array([0.0, 1200.0, 400.0], dtype=np.float32)
        act = np.zeros(8, dtype=np.float32)
        act[0] = 1.0
        act[2] = -1.0  # nose up: fast-aerial attempt
        act[5] = 1.0
        r = jb.get_reward(car, arena, act, False, None)
        self.assertGreaterEqual(
            r, 0.0,
            f"Pitch-up toward a ball at z=180 is an aerial attempt, not a bad "
            f"backflip; got {r:.3f}.")

    # ── 3. Unnormalized horizontal basis vectors ──────────────────────────────
    def test_unit_horiz_is_a_unit_vector_for_pitched_basis(self):
        self.assertIsNotNone(unit_horiz, "env.rewards must expose the unit_horiz helper.")
        # Car pitched 60 degrees nose-up: |fwd[:2]| == 0.5
        pitched = np.array([0.0, 0.5, 0.866], dtype=np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(pitched[:2])), 0.5, places=3)
        u = unit_horiz(pitched)
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=5,
                               msg="unit_horiz must return a unit XY direction.")
        self.assertTrue(np.allclose(unit_horiz(np.array([0.0, 0.0, 1.0], dtype=np.float32)),
                                    np.zeros(2)), "Vertical nose must degrade to zero, not NaN.")

    def test_flick_window_is_invariant_to_car_pitch(self):
        """local_x/local_y are distances in uu; a pitched car must not shrink its own window."""
        self.assertIsNotNone(unit_horiz, "env.rewards must expose the unit_horiz helper.")
        arena = RocketSimArena(); arena.reset()
        car = arena.cars[0]
        car.pos = np.array([0.0, 0.0, 40.0], dtype=np.float32)
        arena.ball.pos = np.array([0.0, 100.0, 150.0], dtype=np.float32)
        car_to_ball = arena.ball.pos - car.pos

        flat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
        pitched = np.array([[0, 0.5, 0.866], [-1, 0, 0], [0, -0.866, 0.5]], dtype=np.float32)

        lx_flat = float(np.dot(car_to_ball[:2], unit_horiz(flat[0])))
        lx_pitched = float(np.dot(car_to_ball[:2], unit_horiz(pitched[0])))
        self.assertAlmostEqual(lx_flat, lx_pitched, places=3,
                               msg="Pitching the car must not halve local_x.")

    # ── 4. just_dodged latching ───────────────────────────────────────────────
    def test_just_dodged_is_edge_and_is_dodging_is_level(self):
        car = self.arena.cars[0]
        self.assertTrue(hasattr(car, "is_dodging"),
                        "CarState must expose is_dodging as the flip-duration level flag, "
                        "so just_dodged can stay a single-step edge.")
        self.assertFalse(car.just_dodged)
        self.assertFalse(car.is_dodging)

    # ── 5. Boost knob monotonicity ────────────────────────────────────────────
    def test_boost_penalties_scale_with_lose_weight(self):
        def supersonic_waste(lose_weight):
            arena = RocketSimArena(); arena.reset()
            car = arena.cars[0]
            car.rot_mat = NOSE_PLUS_Y.copy()
            car.pos = np.array([0.0, 0.0, 17.0], dtype=np.float32)
            car.vel = np.array([0.0, 2300.0, 0.0], dtype=np.float32)
            car.on_ground = True
            car.boost = 50.0
            arena.ball.pos = np.array([0.0, 3000.0, 93.0], dtype=np.float32)
            arena.ball.vel = np.array([0.0, 500.0, 0.0], dtype=np.float32)
            br = BoostReward(gain_weight=0.6, lose_weight=lose_weight)
            br.reset(arena)
            car.boost = 45.0  # burned boost this step
            act = np.zeros(8, dtype=np.float32); act[0] = 1.0; act[6] = 1.0
            return br.get_reward(car, arena, act, False, None)

        r_low = supersonic_waste(0.3)
        r_high = supersonic_waste(1.2)
        self.assertLess(r_high, r_low,
                        "Raising boost_lose_weight must deepen the usage penalty; if it does "
                        "not, the flat constants have escaped the weight again.")
        self.assertAlmostEqual(r_high / r_low, 4.0, delta=0.35,
                               msg="Penalty should scale roughly linearly with lose_weight.")

    # ── 6. Save farming ───────────────────────────────────────────────────────
    def test_goal_line_save_requires_intervening_opponent_touch(self):
        arena = RocketSimArena(); arena.reset()
        car = arena.cars[0]; car.team = 0
        opp = arena.cars[1]; opp.team = 1
        car.pos = np.array([0.0, -4600.0, 17.0], dtype=np.float32)
        car.rot_mat = NOSE_PLUS_Y.copy()
        opp.pos = np.array([0.0, 2000.0, 17.0], dtype=np.float32)
        arena.ball.pos = np.array([0.0, -4400.0, 93.0], dtype=np.float32)
        arena.ball.vel = np.array([0.0, 900.0, 0.0], dtype=np.float32)

        gr = GoalReward(goal_weight=30.0, concede_weight=-30.0, save_weight=12.0)
        gr.reset(arena)
        act = np.zeros(8, dtype=np.float32)

        # First clear off a loose ball the bot did not author: this is a real save.
        car.ball_touches += 1
        r_first = gr.get_reward(car, arena, act, False, None)
        self.assertGreater(r_first, 0.0,
                           "A clear off a ball the bot did not put in motion is a genuine save.")

        # Second clear with no opponent touch in between: the bot is now the author of the
        # ball's trajectory, so this is the push-then-clear farming loop. No credit.
        car.ball_touches += 1
        r_farm = gr.get_reward(car, arena, act, False, None)
        self.assertEqual(r_farm, 0.0,
                         "Re-clearing a ball the bot itself last touched must not pay "
                         "save_weight; that is the push-then-clear farming loop.")

        # Genuine save: opponent shoots, bot clears.
        opp.ball_touches += 1
        gr.get_reward(car, arena, act, False, None)
        car.ball_touches += 1
        r_real = gr.get_reward(car, arena, act, False, None)
        self.assertGreater(r_real, 0.0, "A clear following an opponent touch is a real save.")

    # ── 7. One-sided landing accounting ───────────────────────────────────────
    def test_touchdown_crash_penalty_exempt_during_close_engagement(self):
        def landing_reward(ball_gap):
            arena = RocketSimArena(); arena.reset()
            car = arena.cars[0]
            car.pos = np.array([0.0, 0.0, 40.0], dtype=np.float32)
            car.vel = np.array([0.0, 200.0, -300.0], dtype=np.float32)
            # Landing on the door: up vector horizontal -> best_landing_align == 0.0
            car.rot_mat = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)
            car.on_ground = False
            arena.ball.pos = np.array([0.0, ball_gap, 93.0], dtype=np.float32)
            arena.ball.vel = np.zeros(3, dtype=np.float32)
            rec = AirRollRecoveryReward(weight=1.0)
            rec.reset(arena)
            rec._airborne_ticks[car.id] = 15
            rec._was_disoriented[car.id] = True
            rec._disoriented_this_flight[car.id] = True
            rec._prev_on_ground[car.id] = False
            rec._prev_up_z[car.id] = 0.0
            return rec.get_reward(car, arena, np.zeros(8, dtype=np.float32), False, None)

        near = landing_reward(200.0)    # inside a ball engagement
        far = landing_reward(2000.0)    # open-field recovery
        self.assertGreaterEqual(
            near, far,
            "A messy landing that concludes a close ball challenge must not be charged the "
            "full crash penalty while its in-flight recovery reward is suppressed.")

    # ── 8. Ground-only shaping cliff ──────────────────────────────────────────
    def test_boost_transit_shaping_has_no_liftoff_cliff(self):
        def shaping(car_z, on_ground):
            arena = RocketSimArena(); arena.reset()
            car = arena.cars[0]
            car.rot_mat = NOSE_PLUS_Y.copy()
            # Heading +X toward the big pad at (3584, 0), ~980 uu away.
            car.rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            car.pos = np.array([2600.0, 0.0, car_z], dtype=np.float32)
            car.vel = np.array([1000.0, 0.0, 0.0], dtype=np.float32)
            car.on_ground = on_ground
            car.boost = 10.0
            arena.ball.pos = np.array([0.0, 3000.0, 93.0], dtype=np.float32)
            br = BoostReward(gain_weight=1.4, lose_weight=0.6)
            br.reset(arena)
            return br.get_reward(car, arena, np.zeros(8, dtype=np.float32), False, None)

        grounded = shaping(17.0, True)
        just_airborne = shaping(30.0, False)
        self.assertGreater(grounded, 0.0, "Fixture must actually produce pad-transit shaping.")
        self.assertGreater(
            just_airborne, 0.5 * grounded,
            "Leaving the turf must not instantly delete the pad-transit shaping stream; "
            "that is a standing opportunity cost for jumping.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
