"""
Teardown behaviour for TrainingProcessManager.

Two failures motivated these. The reader thread reached through self.process on every
use while stop_training set it to None underneath, producing
"'NoneType' object has no attribute 'stdout'" in the console on every stop. And
Popen.terminate() on Windows is TerminateProcess, which skips atexit, so the training
run's 16 environment workers were orphaned rather than cleaned up.
"""

from __future__ import annotations
import os
import subprocess
import sys
import threading
import time
import unittest

from utils.process_manager import TrainingProcessManager


class TestReaderThreadTeardown(unittest.TestCase):
    def test_reader_survives_process_being_cleared_mid_read(self):
        """
        stop_training nulls self.process while the reader is parked in readline.

        The reader must hold its own stream reference and exit quietly. Previously it
        raised AttributeError, which killed the thread before it closed the pipe.
        """
        mgr = TrainingProcessManager()
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c",
             "import time,sys\n"
             "print('hello', flush=True)\n"
             "time.sleep(30)\n"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        mgr.process = proc
        errors = []

        def run():
            try:
                mgr._reader_thread(proc.stdout)
            except BaseException as exc:      # noqa: BLE001 - the point is to catch any
                errors.append(exc)

        t = threading.Thread(target=run, daemon=True)
        t.start()

        # Let the reader consume a line and block on the next readline.
        deadline = time.time() + 5
        while time.time() < deadline and not any("hello" in x for x in mgr.log_buffer):
            time.sleep(0.05)
        self.assertTrue(any("hello" in x for x in mgr.log_buffer), "reader never read a line")

        # Exactly what stop_training does, in the order it does it.
        mgr.process = None
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        t.join(timeout=5)

        self.assertFalse(t.is_alive(), "reader thread did not exit")
        self.assertEqual(errors, [], f"reader thread raised: {errors}")


class TestProcessTreeTeardown(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "tree-kill path under test is the Windows one")
    def test_kill_process_tree_reaps_grandchildren(self):
        """
        train.py is not a leaf: it spawns environment workers that outlive a plain
        terminate(). The whole tree must go.
        """
        parent = subprocess.Popen(
            [sys.executable, "-u", "-c",
             "import subprocess, sys, time\n"
             "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
             "print(kid.pid, flush=True)\n"
             "time.sleep(60)\n"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        try:
            child_pid = int(parent.stdout.readline().strip())
            self.assertTrue(self._alive(child_pid), "grandchild never started")

            TrainingProcessManager._kill_process_tree(parent.pid)

            deadline = time.time() + 10
            while time.time() < deadline and self._alive(child_pid):
                time.sleep(0.2)
            self.assertFalse(
                self._alive(child_pid),
                "grandchild survived the tree kill -- workers would orphan on Stop Training"
            )
        finally:
            for proc in (parent,):
                try:
                    proc.kill()
                except Exception:
                    pass

    @staticmethod
    def _alive(pid: int) -> bool:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True
        ).stdout
        return str(pid) in out


if __name__ == "__main__":
    unittest.main()
