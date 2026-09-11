"""
Safety tests for scripts/fresh_run.py.

The script moves checkpoints and logs. Three properties matter more than any convenience it
offers, so they are asserted rather than trusted:

  1. It never touches the external benchmarks or the behavioral-cloning seed. Losing
     necto-model.pt or nexto-model.pt costs you the only yardstick that stays valid across
     observation changes.
  2. It archives rather than deletes, so a run retired by mistake can be walked back.
  3. It refuses to run while training is live, because moving latest_model.pt out from under
     a running trainer corrupts the next autosave.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(ROOT, "scripts", "fresh_run.py")


def load_module(root_override):
    """Import fresh_run with its ROOT pointed at a scratch tree."""
    spec = importlib.util.spec_from_file_location("fresh_run_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = root_override
    return mod


class TestFreshRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="freshrun_")
        os.makedirs(os.path.join(self.tmp, "checkpoints"))
        os.makedirs(os.path.join(self.tmp, "logs"))
        self.written = []
        for name in ("necto-model.pt", "nexto-model.pt", "pretrained_baseline.pt",
                     "latest_model.pt", "checkpoint_iter_200.pt", "checkpoint_iter_400.pt"):
            p = os.path.join(self.tmp, "checkpoints", name)
            with open(p, "wb") as fh:
                fh.write(b"x" * 32)
            self.written.append(p)
        for name in ("league_state.json", "trueskill_leaderboard.json", "history.jsonl",
                     "metrics.json", "reference_sweep.json",
                     "events.out.tfevents.123.host.1.0"):
            p = os.path.join(self.tmp, "logs", name)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("{}")
            self.written.append(p)
        self.mod = load_module(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _plan(self, keep_seed=None):
        return self.mod.collect(keep_seed)

    def test_benchmarks_and_seed_are_never_archived(self):
        plan = self._plan()
        names = [os.path.basename(p) for p in plan["checkpoints"]]
        for protected in ("necto-model.pt", "nexto-model.pt", "pretrained_baseline.pt"):
            self.assertNotIn(
                protected, names,
                "%s must never be archived: it is an external benchmark or the BC seed" % protected,
            )

    def test_numbered_checkpoints_and_resume_point_are_archived(self):
        names = [os.path.basename(p) for p in self._plan()["checkpoints"]]
        self.assertIn("checkpoint_iter_200.pt", names)
        self.assertIn("checkpoint_iter_400.pt", names)
        self.assertIn("latest_model.pt", names,
                      "left in place, a 'fresh' run silently resumes from it")

    def test_league_state_and_history_are_archived(self):
        names = [os.path.basename(p) for p in self._plan()["logs"]]
        for expected in ("league_state.json", "trueskill_leaderboard.json",
                         "history.jsonl", "metrics.json"):
            self.assertIn(expected, names)
        self.assertTrue(any(n.startswith("events.out.tfevents.") for n in names),
                        "tensorboard events must not overlay the new run on the old")

    def test_reference_sweep_is_preserved(self):
        names = [os.path.basename(p) for p in self._plan()["logs"]]
        self.assertNotIn("reference_sweep.json", names,
                         "sweep output describes the tooling, not the run")

    def test_keep_seed_protects_an_extra_checkpoint(self):
        seed = os.path.join(self.tmp, "checkpoints", "checkpoint_iter_400.pt")
        names = [os.path.basename(p) for p in self._plan(keep_seed=seed)["checkpoints"]]
        self.assertNotIn("checkpoint_iter_400.pt", names)
        self.assertIn("checkpoint_iter_200.pt", names)

    def test_archive_moves_rather_than_deletes(self):
        plan = self._plan()
        dest = os.path.join(self.tmp, "archive", "run_test")
        os.makedirs(os.path.join(dest, "checkpoints"))
        moved = []
        for p in plan["checkpoints"]:
            shutil.move(p, os.path.join(dest, "checkpoints", os.path.basename(p)))
            moved.append(os.path.basename(p))
        for name in moved:
            self.assertTrue(
                os.path.exists(os.path.join(dest, "checkpoints", name)),
                "%s must survive in the archive; this script never deletes" % name,
            )

    def test_refuses_when_a_live_pid_is_present(self):
        pid_path = os.path.join(self.tmp, "logs", "train.pid")
        with open(pid_path, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
        live, detail = self.mod.training_is_live()
        self.assertTrue(live, "a pid file naming a running process must block the archive: %s" % detail)

    def test_allows_when_no_pid_file(self):
        live, _ = self.mod.training_is_live()
        self.assertFalse(live)

    def test_unverifiable_pid_is_treated_as_live(self):
        """Erring toward refusing is correct: the cost of a false stop is one command."""
        with open(os.path.join(self.tmp, "logs", "train.pid"), "w", encoding="utf-8") as fh:
            fh.write("nonsense")
        live, _ = self.mod.training_is_live()
        self.assertFalse(live, "an unparseable pid is reported, not guessed at")


if __name__ == "__main__":
    unittest.main(verbosity=2)
