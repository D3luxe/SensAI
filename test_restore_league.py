"""
Tests for scripts/restore_league.py: a snapshot's bare checkpoint paths land in the named archive,
the current league is backed up before it is replaced, and a snapshot naming a checkpoint that no
longer exists is refused rather than restored with a hole in it.
"""
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(ROOT, "scripts", "restore_league.py")
SNAP = "before_v5_run_202609211333"


def load_module(root_override):
    spec = importlib.util.spec_from_file_location("restore_league_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = root_override
    mod._training_is_live = lambda: (False, "test")
    return mod


def rating(name, path, anchor=False):
    return {"name": name, "path": path, "mu": 30.0, "sigma": 1.0, "is_anchor": anchor}


class TestRestoreLeague(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="restoreleague_")
        for d in ("logs", "checkpoints/archive/v5_run", "checkpoints/baselines"):
            os.makedirs(os.path.join(self.tmp, d))
        for rel in ("checkpoints/archive/v5_run/checkpoint_iter_222200.pt",
                    "checkpoints/archive/v5_run/checkpoint_iter_222000.pt",
                    "checkpoints/baselines/v3_iter198000.pt"):
            open(os.path.join(self.tmp, rel), "wb").close()
        self.mod = load_module(self.tmp)
        self.write_snapshot(["checkpoints/checkpoint_iter_222200.pt", "checkpoints\\checkpoint_iter_222000.pt"])
        self.write_live("live-league")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, rel, text):
        with io.open(os.path.join(self.tmp, rel), "w", encoding="utf-8") as f:
            f.write(text)

    def read(self, rel):
        with io.open(os.path.join(self.tmp, rel), encoding="utf-8") as f:
            return f.read()

    def write_snapshot(self, ckpts):
        ratings = {c: rating(os.path.basename(c.replace("\\", "/"))[:-3], c) for c in ckpts}
        ratings["checkpoints/baselines/v3_iter198000.pt"] = rating("v3_iter198000", "checkpoints/baselines/v3_iter198000.pt", True)
        lb = {"version": "1.0", "ratings": ratings, "history": [{"model_a": ckpts[0], "model_b": "heuristic"}]}
        self.write(f"logs/trueskill_leaderboard.json.{SNAP}", json.dumps(lb))
        self.write(f"logs/league_state.json.{SNAP}", json.dumps({"king_of_the_hill": ckpts[0], "elite_pool": ckpts}))
        self.write(f"logs/trueskill_leaderboard.results.jsonl.{SNAP}",
                   json.dumps({"a": ckpts[0], "b": ckpts[-1], "a_score": 5, "b_score": 3, "kind": "calibration"}) + "\n")

    def write_live(self, marker):
        self.write("logs/trueskill_leaderboard.json", json.dumps({"marker": marker, "ratings": {}}))
        self.write("logs/league_state.json", json.dumps({"marker": marker}))
        self.write("logs/trueskill_leaderboard.results.jsonl", marker + "\n")

    def run_main(self, *extra):
        argv = ["restore_league.py", "--snapshot", SNAP, "--run", "v5_run", *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
            return self.mod.main()

    def test_paths_are_repointed_into_the_archive(self):
        self.assertEqual(self.run_main(), 0)
        lb = json.loads(self.read("logs/trueskill_leaderboard.json"))
        self.assertIn("checkpoints/archive/v5_run/checkpoint_iter_222200.pt", lb["ratings"])
        self.assertIn("checkpoints/archive/v5_run/checkpoint_iter_222000.pt", lb["ratings"])
        r = lb["ratings"]["checkpoints/archive/v5_run/checkpoint_iter_222200.pt"]
        self.assertEqual(r["path"], "checkpoints/archive/v5_run/checkpoint_iter_222200.pt")
        self.assertEqual(r["name"], "v5/checkpoint_iter_222200")
        self.assertEqual(lb["history"][0]["model_a"], "checkpoints/archive/v5_run/checkpoint_iter_222200.pt")
        state = json.loads(self.read("logs/league_state.json"))
        self.assertEqual(state["king_of_the_hill"], "checkpoints/archive/v5_run/checkpoint_iter_222200.pt")
        self.assertNotIn("checkpoints/checkpoint_iter", self.read("logs/trueskill_leaderboard.results.jsonl"))

    def test_anchor_paths_are_left_alone(self):
        self.assertEqual(self.run_main(), 0)
        lb = json.loads(self.read("logs/trueskill_leaderboard.json"))
        self.assertEqual(lb["ratings"]["checkpoints/baselines/v3_iter198000.pt"]["name"], "v3_iter198000")

    def test_current_league_is_backed_up_first(self):
        self.assertEqual(self.run_main(), 0)
        backups = [f for f in os.listdir(os.path.join(self.tmp, "logs")) if ".before_restore_" in f]
        self.assertEqual(len(backups), 3)
        state_backup = next(f for f in backups if f.startswith("league_state.json"))
        self.assertEqual(json.loads(self.read(f"logs/{state_backup}"))["marker"], "live-league")

    def test_a_missing_checkpoint_is_refused_and_nothing_changes(self):
        self.write_snapshot(["checkpoints/checkpoint_iter_222200.pt", "checkpoints/checkpoint_iter_199000.pt"])
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(json.loads(self.read("logs/league_state.json"))["marker"], "live-league")

    def test_dry_run_changes_nothing(self):
        self.assertEqual(self.run_main("--dry-run"), 0)
        self.assertEqual(json.loads(self.read("logs/league_state.json"))["marker"], "live-league")
        self.assertFalse([f for f in os.listdir(os.path.join(self.tmp, "logs")) if ".before_restore_" in f])

    def test_refuses_while_training_is_live(self):
        self.mod._training_is_live = lambda: (True, "pid 1 is running")
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(json.loads(self.read("logs/league_state.json"))["marker"], "live-league")


if __name__ == "__main__":
    unittest.main()
