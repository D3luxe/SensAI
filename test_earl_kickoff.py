"""
Necto/Nexto opponents as they run live (RLBot v5 ports, VirxEC/NectoFamily).

Covers:
  1. has_flip: a grounded car is reported as having its flip, as RLGym and rlgym_compat do.
  2. The scripted kickoff: the bot returns per-tick blocks of EARL_KICKOFF_TICKS from the moment
     the kickoff starts, the arena plays them one row per tick, the script stops for good when the
     ball moves, and the model sees the scripted controls as its previous action.
  3. The kickoff is contested: from every spawn Necto reaches the ball within a few hundred uu of
     a car driving straight at it, instead of trailing by 400-1500 uu as the model alone did.
"""
import random
import unittest

import numpy as np

from env.baseline_agent import (
    EARL_KICKOFF_TICKS, EARL_TICK_SKIP, NectoNextoOpponentBot, _earl_has_flip, create_opponent_bot,
)
from env.rocket_env import RocketLeagueEnv

NECTO = "checkpoints/necto-model.pt"


def kickoff_env(spawn_index):
    env = RocketLeagueEnv(game_mode="1v1", max_episode_steps=10 ** 9, is_baseline_env=True,
                          baseline_opponent_type=NECTO)
    orig = random.randint
    random.randint = lambda a, b: spawn_index
    try:
        env.reset(random_kickoff=False)
    finally:
        random.randint = orig
    env.current_scenario = "test"
    env.scenario_timeout = 10 ** 9
    return env


class TestEarlObservation(unittest.TestCase):
    def test_grounded_car_has_its_flip(self):
        env = kickoff_env(0)
        car = env.arena.cars[1]
        self.assertTrue(car.on_ground)
        self.assertFalse(car.has_flip, "the sim's own convention, which SensAI trains on, is unchanged")
        self.assertEqual(_earl_has_flip(car), 1.0)


class TestScriptedKickoff(unittest.TestCase):
    def test_bot_returns_the_script_tick_by_tick(self):
        env = kickoff_env(0)
        bot = env.baseline_bot
        self.assertIsInstance(bot, NectoNextoOpponentBot)
        a = env.arena
        for step in range(3):
            block = bot.get_action(a.cars[1], a)
            self.assertEqual(block.shape, (EARL_TICK_SKIP, 8))
            np.testing.assert_array_equal(block, EARL_KICKOFF_TICKS[step * 8:(step + 1) * 8])
            np.testing.assert_array_equal(bot.prev_action, block[-1])
            # advance the physics with the block, blue parked
            a.step([np.zeros(8, np.float32), block], dt=8 / 120.0, bot_mask=[False, True])

    def test_the_arena_plays_the_block_per_tick(self):
        # Boost pressed on every tick of the first block: exactly 8 ticks of boost are spent
        env = kickoff_env(4)
        a = env.arena
        before = float(a.cars[1].boost)
        block = env.baseline_bot.get_action(a.cars[1], a)
        a.step([np.zeros(8, np.float32), block], dt=8 / 120.0, bot_mask=[False, True])
        spent = before - float(a.cars[1].boost)
        self.assertAlmostEqual(spent, 8 * 33.3 / 120.0, delta=0.3)

    def test_script_stops_when_the_ball_moves(self):
        env = kickoff_env(0)
        bot = env.baseline_bot
        a = env.arena
        self.assertIsNotNone(bot.kickoff_controls(a.cars[1], a))
        a.ball.vel = np.array([0.0, 500.0, 0.0], dtype=np.float32)
        self.assertIsNone(bot.kickoff_controls(a.cars[1], a))
        a.ball.vel = np.zeros(3, dtype=np.float32)
        a.cars[1].vel = np.array([0.0, -1500.0, 0.0], dtype=np.float32)
        self.assertIsNone(bot.kickoff_controls(a.cars[1], a), "a finished kickoff never restarts mid-play")

    def test_batched_path_scripts_each_arena(self):
        envs = [kickoff_env(i) for i in range(3)]
        bot = create_opponent_bot(NECTO)
        acts = bot.batch_get_actions([e.arena.cars[1] for e in envs], [e.arena for e in envs],
                                     prev_actions=[np.zeros(8, np.float32)] * 3)
        for act in acts:
            np.testing.assert_array_equal(act, EARL_KICKOFF_TICKS[:8])


class TestKickoffIsContested(unittest.TestCase):
    def test_necto_meets_a_straight_charge_from_every_spawn(self):
        for spawn in range(5):
            env = kickoff_env(spawn)
            a = env.arena
            for _ in range(60):
                # Blue: throttle + boost straight at the ball (every spawn faces it)
                blue = np.zeros((2, 8), np.float32)
                blue[0, 0], blue[0, 6] = 1.0, 1.0
                env.step(blue)
                if a.cars[0].ball_touches or a.cars[1].ball_touches:
                    break
            gap = float(np.linalg.norm(a.cars[1].pos[:2] - a.ball.pos[:2]))
            self.assertLess(gap, 450.0, f"spawn {spawn}: Necto {gap:.0f} uu from the ball at first contact")


if __name__ == "__main__":
    unittest.main()
