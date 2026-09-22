"""
Reward v7's replay starts (env/replay_sampling_v7.py, docs/reward_v7_replay_spec.md).

  - each pruning rule removes the frames it names, and only those
  - each tag on a hand-built frame, and exclusivity in priority order
  - sampled tag frequencies follow the weights; pruned frames are never drawn
  - each mirror is an exact symmetry: RocketSim plays the mirrored state as the mirror of the original
  - the uniform (v2-v6) path draws the same frames as before for a fixed seed, mapped or not
  - parsers in one process share one mapped buffer; the store round-trips the .npz
  - the index is cached beside the store and reloads identical
  - a version with a frozen pool refuses any other pool
  - team replays are rejected at ingest
  - v7's reward and synthetic scenarios are v6's; its scenario payload carries the sampling settings
"""
import gc
import json
import math
import os
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

from env import replay_sampling_v7 as rs
from utils import replay_parser as rp

TAG = {t: i for i, t in enumerate(rs.TAGS)}


def frame(ball=(0, 0, 93), ball_vel=(300, 0, 0), cars=((-2000, -3000, 17), (2000, 3000, 17)),
          car_vel=((0, 0, 0), (0, 0, 0)), t=10.0, car_t=(10.0, 10.0)):
    return dict(ball_pos=ball, ball_vel=ball_vel, car_pos=cars, car_vel=car_vel,
                car_rot=((0.0, 0.3, 0.0), (0.0, -2.0, 0.1)), car_boost=(40.0, 60.0),
                frame_time=t, car_time=car_t)


# One frame per situation, plus one per pruning rule
CASES = {
    "open_play": frame(),
    "challenge": frame(cars=((0, -500, 17), (0, 500, 17))),
    "goal_threat": frame(ball=(0, 3000, 93), ball_vel=(0, 1000, 0), cars=((0, -3000, 17), (2000, 4500, 17))),
    "aerial": frame(ball=(0, 0, 800), cars=((0, -300, 700), (0, 4000, 17))),
    "carry": frame(ball=(500, -1000, 150), ball_vel=(0, 1000, 0), cars=((500, -1000, 17), (0, 3000, 17)),
                   car_vel=((0, 1000, 0), (0, 0, 0))),
    "carry_contested": frame(ball=(500, -1000, 150), ball_vel=(0, 1000, 0), cars=((500, -1000, 17), (500, -300, 17)),
                             car_vel=((0, 1000, 0), (0, 0, 0))),
    "prune_countdown": frame(ball_vel=(0, 0, 0), cars=((-2048, -2560, 17), (2048, 2560, 17))),
    "prune_in_net": frame(ball=(0, 5200, 300)),
    "prune_stale": frame(car_t=(10.0, 9.0)),
    "prune_spawn": frame(cars=((0, 0, 17), (2000, 3000, 17))),
}


def pool_of(names, repeat=1):
    rows = [CASES[n] for n in names for _ in range(repeat)]
    pool = {k: np.array([r[k] for r in rows], dtype=np.float64 if k.endswith("time") else np.float32)
            for k in rows[0]}
    pool["car_ang_vel"] = np.zeros_like(pool["car_pos"])
    pool["segment_id"] = np.zeros(len(rows), dtype=np.int32)
    return pool


SITUATIONS = ["open_play", "challenge", "goal_threat", "aerial", "carry", "carry_contested"]
PRUNES = ["prune_countdown", "prune_in_net", "prune_stale", "prune_spawn"]
SETTINGS = json.load(open("config/reward_versions/v7.json", encoding="utf-8"))["settings"]["replay_sampling"]


class TestPruneAndTag(unittest.TestCase):
    def test_each_prune_rule_removes_its_frame_and_nothing_else(self):
        keep = rs.prune_mask(pool_of(SITUATIONS + PRUNES))
        self.assertTrue(keep[:len(SITUATIONS)].all())
        self.assertFalse(keep[len(SITUATIONS):].any())

    def test_timing_rules_are_skipped_for_pools_without_timing(self):
        pool = pool_of(["prune_stale"])
        del pool["frame_time"], pool["car_time"]
        self.assertTrue(rs.prune_mask(pool).all())

    def test_each_situation_gets_its_tag(self):
        tags = rs.tag_frames(pool_of(SITUATIONS))
        self.assertEqual([rs.TAGS[t] for t in tags], SITUATIONS)

    def test_tags_are_exclusive_in_priority_order(self):
        # carry_contested also meets challenge's rule and carry's; the first in TAGS wins
        self.assertEqual(rs.TAGS.index("carry_contested"), 0)
        tags = rs.tag_frames(pool_of(["carry_contested", "challenge"]))
        self.assertEqual(list(tags), [TAG["carry_contested"], TAG["challenge"]])

    def test_index_groups_kept_frames_by_tag(self):
        pool = pool_of(PRUNES + SITUATIONS[::-1])
        order, counts = rs.build_index(pool)
        self.assertEqual(int(counts.sum()), len(SITUATIONS))
        self.assertTrue(all(c == 1 for c in counts))
        tags = rs.tag_frames(pool)
        self.assertEqual([rs.TAGS[tags[i]] for i in order], list(rs.TAGS))


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.pool = pool_of(SITUATIONS + PRUNES, repeat=5)
        self.order, self.counts = rs.build_index(self.pool)
        self.tags = rs.tag_frames(self.pool)
        self.keep = rs.prune_mask(self.pool)

    def draws(self, settings, n):
        s = rs.ReplaySampler(self.pool, self.order, self.counts, settings)
        random.seed(7)
        return [s.draw_index() for _ in range(n)]

    def test_tag_frequencies_follow_the_weights(self):
        n = 40_000
        idx = self.draws(SETTINGS, n)
        self.assertTrue(self.keep[idx].all(), "a pruned frame was drawn")
        seen = np.bincount(self.tags[idx], minlength=len(rs.TAGS)) / n
        for i, t in enumerate(rs.TAGS):
            p = SETTINGS["tag_weights"][t]
            self.assertLess(abs(seen[i] - p), 5 * math.sqrt(p * (1 - p) / n), t)

    def test_a_tag_with_no_frames_is_renormalised_away(self):
        pool = pool_of(["open_play", "challenge"], repeat=3)
        order, counts = rs.build_index(pool)
        s = rs.ReplaySampler(pool, order, counts, {"tag_weights": {"carry": 0.9, "open_play": 0.05, "challenge": 0.05}})
        random.seed(1)
        tags = rs.tag_frames(pool)[[s.draw_index() for _ in range(2000)]]
        self.assertEqual(set(rs.TAGS[t] for t in tags), {"open_play", "challenge"})

    def test_unknown_tags_are_refused(self):
        with self.assertRaises(ValueError):
            rs.ReplaySampler(self.pool, self.order, self.counts, {"tag_weights": {"dribble": 1.0}})

    def test_mirrors_are_applied_at_their_rates(self):
        s = rs.ReplaySampler(self.pool, self.order, self.counts,
                             {"tag_weights": {"carry": 1.0}, "flip_x_prob": 0.5, "team_swap_prob": 0.5})
        random.seed(3)
        xs = [float(s.sample()["ball_pos"][0]) for _ in range(4000)]
        self.assertAlmostEqual(np.mean(np.array(xs) > 0), 0.5, delta=0.04)


class TestMirrors(unittest.TestCase):
    STATE = {k: np.array(v, dtype=np.float32) for k, v in {
        "ball_pos": (800.0, -1200.0, 400.0), "ball_vel": (300.0, 900.0, 250.0),
        "car_pos": ((600.0, -2000.0, 17.0), (-900.0, 1500.0, 300.0)),
        "car_vel": ((500.0, 800.0, 0.0), (-200.0, -700.0, 150.0)),
        "car_rot": ((0.0, 1.1, 0.0), (0.4, -2.3, 0.7)),
        "car_boost": (30.0, 70.0)}.items()}

    @classmethod
    def setUpClass(cls):
        import RocketSim as rsim
        cls.rsim = rsim

    def axes(self, rot):
        # Keywords, as ReplayStateSetter passes them: rsim.Angle's positional order is not (p, y, r)
        m = self.rsim.Angle(pitch=float(rot[0]), yaw=float(rot[1]), roll=float(rot[2])).as_rot_mat()
        return [np.array([v.x, v.y, v.z]) for v in (m.forward, m.right, m.up)]

    def test_mirror_x_reflects_every_vector_and_the_car_axes(self):
        m = rs.mirror_x(self.STATE)
        R = np.diag([-1.0, 1.0, 1.0])
        np.testing.assert_allclose(m["ball_pos"], R @ self.STATE["ball_pos"])
        np.testing.assert_allclose(m["car_vel"], self.STATE["car_vel"] @ R)
        for i in range(2):
            f0, r0, u0 = self.axes(self.STATE["car_rot"][i])
            f1, r1, u1 = self.axes(m["car_rot"][i])
            np.testing.assert_allclose(f1, R @ f0, atol=1e-5)
            np.testing.assert_allclose(u1, R @ u0, atol=1e-5)
            np.testing.assert_allclose(r1, -(R @ r0), atol=1e-5)  # a reflection flips handedness

    def test_swap_teams_turns_the_pitch_and_exchanges_the_cars(self):
        m = rs.swap_teams(self.STATE)
        R = np.diag([-1.0, -1.0, 1.0])
        np.testing.assert_allclose(m["ball_vel"], R @ self.STATE["ball_vel"])
        for i, j in ((0, 1), (1, 0)):
            np.testing.assert_allclose(m["car_pos"][i], R @ self.STATE["car_pos"][j])
            self.assertEqual(m["car_boost"][i], self.STATE["car_boost"][j])
            for a0, a1 in zip(self.axes(self.STATE["car_rot"][j]), self.axes(m["car_rot"][i])):
                np.testing.assert_allclose(a1, R @ a0, atol=1e-5)

    def play(self, state, ticks=90):
        """RocketSim from `state` with no inputs: (ball position, car positions) after `ticks`."""
        from env.physics_engine import RocketSimArena
        from env.state_setters import ReplayStateSetter
        arena = RocketSimArena(num_players=2, game_mode="1v1")
        stub = mock.Mock()
        stub.sample_state.return_value = state
        ReplayStateSetter(parser=stub).reset(arena._rsim_arena, 2)
        arena._rsim_arena.step(ticks)
        b = arena._rsim_arena.ball.get_state().pos
        cars = [c.get_state().pos for c in arena._rsim_arena.get_cars()]
        return np.array([b.x, b.y, b.z]), [np.array([p.x, p.y, p.z]) for p in cars]

    def test_rocketsim_plays_each_mirror_as_the_mirror_of_the_original(self):
        ball0, cars0 = self.play(self.STATE)
        for fn, R, perm in ((rs.mirror_x, np.diag([-1.0, 1.0, 1.0]), (0, 1)),
                            (rs.swap_teams, np.diag([-1.0, -1.0, 1.0]), (1, 0))):
            ball1, cars1 = self.play(fn(self.STATE))
            np.testing.assert_allclose(ball1, R @ ball0, atol=1.0, err_msg=fn.__name__)
            for i, j in enumerate(perm):
                np.testing.assert_allclose(cars1[i], R @ cars0[j], atol=1.0, err_msg=fn.__name__)


class TestPoolStore(unittest.TestCase):
    """The parser side: the uniform path, the shared mapped store, the index cache, the frozen pool."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = os.path.join(self.tmp.name, "pool.npz")
        self.pool = pool_of(SITUATIONS + PRUNES, repeat=7)
        np.savez_compressed(self.path, **self.pool)

    def tearDown(self):
        rp.release_shared_pools()
        gc.collect()
        self.tmp.cleanup()

    def mapped(self):
        return mock.patch.object(rp, "MMAP_MIN_BYTES", 0)

    def old_sample_state(self, n, seed):
        """ReplayParser.sample_state before v7, for the arrays read straight from the .npz."""
        data = np.load(self.path)
        random.seed(seed)
        out = []
        for _ in range(n):
            idx = random.randint(0, len(data["ball_pos"]) - 1)
            out.append((data["ball_pos"][idx].copy(), data["car_pos"][idx].copy(), data["car_rot"][idx].copy()))
        return out

    def test_the_uniform_path_draws_what_it_always_drew(self):
        expected = self.old_sample_state(60, seed=11)
        for mapped in (False, True):
            with mock.patch.object(rp, "MMAP_MIN_BYTES", 0 if mapped else 1 << 40):
                parser = rp.ReplayParser(pool_path=self.path)
                self.assertEqual(parser.store_dir is not None, mapped)
                random.seed(11)
                for b, c, r in expected:
                    s = parser.sample_state(2)
                    np.testing.assert_array_equal(s["ball_pos"], b)
                    np.testing.assert_array_equal(s["car_pos"], c)
                    np.testing.assert_array_equal(s["car_rot"], r)
            del parser
            rp.release_shared_pools()

    def test_parsers_in_a_process_share_one_mapped_buffer(self):
        with self.mapped():
            a, b = rp.ReplayParser(pool_path=self.path), rp.ReplayParser(pool_path=self.path)
        self.assertIs(a.states_buffer["ball_pos"], b.states_buffer["ball_pos"])
        self.assertIsInstance(a.states_buffer["ball_pos"], np.memmap)
        for k, v in self.pool.items():
            np.testing.assert_array_equal(a.states_buffer[k], v)
        self.assertEqual(os.path.dirname(a.store_dir), os.path.splitext(self.path)[0] + ".store")

    def test_the_index_is_cached_beside_the_store(self):
        with self.mapped():
            parser = rp.ReplayParser(pool_path=self.path)
        order, counts = rs.load_or_build_index(parser.states_buffer, parser.store_dir)
        cached = [f for f in os.listdir(parser.store_dir) if f.startswith("v7_")]
        self.assertEqual(len(cached), 2)
        order2, counts2 = rs.load_or_build_index(parser.states_buffer, parser.store_dir)
        self.assertIsInstance(order2, np.memmap)
        np.testing.assert_array_equal(order, order2)
        np.testing.assert_array_equal(counts, counts2)

    def test_the_configured_setter_never_starts_from_a_pruned_frame(self):
        from env.state_setters import ReplayStateSetter
        setter = ReplayStateSetter(parser=rp.ReplayParser(pool_path=self.path))
        setter.configure(SETTINGS)
        keep_balls = {tuple(b) for b in self.pool["ball_pos"][rs.prune_mask(self.pool)]}
        random.seed(5)
        for _ in range(300):
            s = setter.sampler.sample(2)
            ball = (abs(float(s["ball_pos"][0])), abs(float(s["ball_pos"][1])), float(s["ball_pos"][2]))
            self.assertIn(ball, {(abs(x), abs(y), z) for x, y, z in keep_balls})
        setter.configure(None)
        self.assertIsNone(setter.sampler)

    def test_a_frozen_pool_refuses_any_other(self):
        snap = {"version": "v7", "settings": {"replay_sampling": SETTINGS}, "replay_pool": None}
        with self.assertRaisesRegex(RuntimeError, "none is frozen"):
            rs.check_replay_pool(snap, self.path)
        snap["replay_pool"] = rs.current_fingerprint(SETTINGS, self.path)
        self.assertEqual(rs.check_replay_pool(snap, self.path)["sha"], snap["replay_pool"]["sha"])
        self.assertEqual(snap["replay_pool"]["kept_frames"], 7 * len(SITUATIONS))

        changed = pool_of(SITUATIONS + PRUNES, repeat=8)
        np.savez_compressed(self.path, **changed)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            rs.check_replay_pool(snap, self.path)

    def test_versions_without_replay_sampling_are_not_checked(self):
        self.assertIsNone(rs.check_replay_pool({"version": "v6", "settings": {}}, "nowhere.npz"))


class TestIngestGuard(unittest.TestCase):
    def test_team_replays_are_rejected(self):
        parser = rp.ReplayParser(pool_path=os.path.join(tempfile.gettempdir(), "no_pool_here.npz"))
        out = mock.Mock(returncode=0, stdout=json.dumps({"properties": {"TeamSize": 2},
                                                         "objects": ["x"], "network_frames": {"frames": [{}]}}))
        with mock.patch.object(rp, "_ensure_rrrocket", return_value="rrrocket"), \
                mock.patch.object(rp.subprocess, "run", return_value=out):
            self.assertIsNone(parser._extract_replay_binary("team.replay"))


class TestVersion(unittest.TestCase):
    def test_v7_is_v5s_reward_with_its_kickoff_share_held(self):
        # v6 was not adopted, so v7 changes the starts of the adopted version, v5
        from env.reward_registry import CODE_FILES, load_snapshot, make_reward_manager
        from env.rewards_v3 import RewardManagerV3
        v5, v7 = load_snapshot("v5"), load_snapshot("v7")
        self.assertEqual(v7["start_checkpoint"], "checkpoints/baselines/v5_iter222000.pt")
        self.assertTrue(set(CODE_FILES["v5"]) < set(CODE_FILES["v7"]))
        self.assertNotIn("env/rewards_v6.py", CODE_FILES["v7"])
        v5, v7 = v5["settings"], v7["settings"]
        self.assertEqual(v7["rewards"], v5["rewards"])
        self.assertEqual(v7["gamma"], v5["gamma"])
        self.assertEqual(v7["reward_annealing"], v5["reward_annealing"])
        self.assertIsInstance(make_reward_manager("v7"), RewardManagerV3)
        for k, v in v5["scenarios"].items():
            if k not in ("kickoff_prob", "replay_prob"):
                self.assertEqual(v7["scenarios"][k], v, k)
        self.assertEqual((v7["scenarios"]["kickoff_prob"], v7["scenarios"]["replay_prob"]), (0.24, 0.36))
        self.assertAlmostEqual(sum(v7["scenarios"].values()), 1.0)
        self.assertAlmostEqual(sum(v7["replay_sampling"]["tag_weights"].values()), 1.0)
        self.assertEqual(set(v7["replay_sampling"]["tag_weights"]), set(rs.TAGS))

    def test_scenario_payload_carries_replay_sampling_only_for_v7(self):
        from env.reward_registry import apply_reward_version, load_snapshot, scenario_payload
        for v, has in (("v6", False), ("v7", True)):
            cfg = apply_reward_version({"reward_version": v, "replay_sampling": {"stale": True},
                                        "hyperparameters": {"gamma": load_snapshot(v)["settings"]["gamma"]}})
            self.assertEqual("replay_sampling" in scenario_payload(cfg), has, v)
            if has:
                self.assertEqual(scenario_payload(cfg)["replay_sampling"], SETTINGS)

    def test_the_scenario_setter_configures_the_replay_sampler(self):
        from env.state_setters import WeightedScenarioSetter
        setter = WeightedScenarioSetter()
        with mock.patch.object(setter.replay_setter, "configure") as configure:
            setter.update_weights({"scenarios": {"replay_prob": 0.36, "replay_sampling": SETTINGS}})
            configure.assert_called_once_with(SETTINGS)
            setter.update_weights({"scenarios": {"replay_prob": 0.45}})
            configure.assert_called_once()
        self.assertEqual(setter.replay_prob, 0.45)


if __name__ == "__main__":
    unittest.main()
