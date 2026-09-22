"""
The eval automation: following a run (scripts/auto_eval.py, ui/auto_eval_panel.py) and pooling a
decision point (ui/eval_pooling.py, scripts/pool_evals.py, the Evaluation tab's table).

  - a checkpoint that already has a result is not evaluated twice, and --every N thins the series
  - the watcher starts, reports what it has produced, and stops
  - pooling averages over every seed of every result, with the standard error the specs quote
  - goals-against causes are pooled as goals per 10 min, not as shares
  - the table renders both states and marks a negative head to head
"""
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest

from ui import auto_eval_panel, eval_pooling

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


auto_eval = _load_script("auto_eval")


def result(iteration, steps_m, h2h_seeds, ga=30.0, upfield_pct=40.0):
    return {
        "reward_identity": {"version": "v9"}, "iteration": iteration, "quick": False,
        "global_step": int(steps_m * 1e6), "reward_run_start_step": 0,
        "results": {
            "necto": {"goals_against_per_10min": {"mean": ga, "per_seed": [ga]},
                      "ga_cause_caught_upfield_pct": {"mean": upfield_pct, "per_seed": [upfield_pct]}},
            "reference": {"goal_diff_per_10min": {"mean": sum(h2h_seeds) / len(h2h_seeds),
                                                  "per_seed": list(h2h_seeds),
                                                  "min": min(h2h_seeds), "max": max(h2h_seeds)}},
        },
    }


class TestAutoEvalWatcher(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        for it, steps in ((1000, 10), (1200, 13)):
            with open(os.path.join(self.dir, f"v9_{steps}M.json"), "w", encoding="utf-8") as f:
                json.dump(result(it, steps, [1.0]), f)

    def test_evaluated_iterations_come_from_the_results(self):
        self.assertEqual(auto_eval.evaluated_iterations(self.dir), {1000, 1200})

    def test_a_quick_result_does_not_count_as_evaluated(self):
        r = result(1400, 16, [1.0])
        r["quick"] = True
        with open(os.path.join(self.dir, "v9_16M_quick.json"), "w", encoding="utf-8") as f:
            json.dump(r, f)
        self.assertNotIn(1400, auto_eval.evaluated_iterations(self.dir))

    def test_iteration_is_read_from_the_checkpoint_name(self):
        self.assertEqual(auto_eval.iteration_of("checkpoints/checkpoint_iter_231400.pt"), 231400)

    def test_every_n_thins_by_checkpoint_interval(self):
        paths = [f"checkpoint_iter_{i}.pt" for i in (231200, 231400, 231600, 231800)]
        self.assertEqual([p for p in paths if (auto_eval.iteration_of(p) // 200) % 2 == 0],
                         ["checkpoint_iter_231200.pt", "checkpoint_iter_231600.pt"])

    def test_head_to_head_reads_a_finished_result(self):
        self.assertEqual(auto_eval.head_to_head(os.path.join(self.dir, "v9_10M.json")), (1.0, [1.0]))
        self.assertIsNone(auto_eval.head_to_head(os.path.join(self.dir, "missing.json")))


class TestRunnerLifecycle(unittest.TestCase):
    """The panel's process control, driven with a stand-in for scripts/auto_eval.py."""

    def setUp(self):
        self.runner = auto_eval_panel.AutoEvalRunner()

    def tearDown(self):
        self.runner.stop()

    def start_fake(self, body):
        self.runner.process = __import__("subprocess").Popen(
            [sys.executable, "-u", "-c", body], stdout=-1, stderr=-2, text=True, bufsize=1,
            encoding="utf-8", errors="replace")
        self.runner.started_at = time.time()
        self.runner.settings = {"every": 1, "workers": 4, "backfill": False}
        import threading
        t = threading.Thread(target=self.runner._drain, args=(self.runner.process.stdout,), daemon=True)
        t.start()
        return t

    def test_a_started_watcher_reports_what_it_produced_and_stops(self):
        thread = self.start_fake("print('  checkpoint_iter_1.pt -> v9_1M.json   40s   head to head  +1.50')\n"
                                 "print('  FAILED checkpoint_iter_2.pt (1)')\n"
                                 "import time; time.sleep(30)")
        deadline = time.time() + 10
        while self.runner.status()["evaluated"] == 0 and time.time() < deadline:
            time.sleep(0.05)
        st = self.runner.status()
        self.assertTrue(st["running"])
        self.assertEqual((st["evaluated"], st["failed"]), (1, 1))
        self.assertIn("v9_1M.json", st["last"])
        self.assertIn("head to head", self.runner.log_text())
        self.assertIn("Stopped", self.runner.stop())
        thread.join(timeout=5)
        self.assertFalse(self.runner.status()["running"])

    def test_stopping_when_idle_says_so(self):
        self.assertEqual(self.runner.stop(), "Not running.")
        self.assertFalse(self.runner.is_running())

    def test_the_panel_shows_both_states(self):
        off = auto_eval_panel.panel_html(self.runner.status(), "v7", "checkpoints/baselines/v5_iter222000.pt")
        self.assertIn("Not following", off)
        self.assertIn("v5_iter222000.pt", off)
        on = auto_eval_panel.panel_html({"running": True, "evaluated": 12, "failed": 0, "last": "x -> y",
                                         "settings": {"every": 1, "workers": 4}, "elapsed": 3700.0,
                                         "exit_code": None}, "v7", "")
        self.assertIn("Following the run", on)
        self.assertIn("<b>12</b>", on)
        self.assertIn("1h 01m", on)


class TestPooling(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        for it, steps, seeds in ((1000, 100, [-3.0, 1.0, 2.0]), (1200, 103, [4.0, 6.0, 5.0]),
                                 (1400, 200, [0.0, 1.0, -1.0])):
            with open(os.path.join(self.dir, f"v9_{steps}M.json"), "w", encoding="utf-8") as f:
                json.dump(result(it, steps, seeds), f)

    def pick(self, **kw):
        return eval_pooling.pick_results(version="v9", eval_dir=self.dir, **kw)

    def test_a_window_picks_the_results_around_a_decision_point(self):
        self.assertEqual(sorted(n for n, _ in self.pick(around=100)), ["v9_100M", "v9_103M"])
        self.assertEqual([n for n, _ in self.pick(around=200)], ["v9_200M"])

    def test_last_n_takes_the_newest_by_steps(self):
        self.assertEqual([n for n, _ in self.pick(last=2)], ["v9_103M", "v9_200M"])

    def test_causes_pool_as_goals_per_10_min(self):
        idx = [h for _, _, h, _ in eval_pooling.COLUMNS].index("upfld")
        res = self.pick(around=100)[0][1]
        self.assertAlmostEqual(eval_pooling.row_values(res, eval_pooling.COLUMNS)[idx], 12.0)  # 40% of 30

    def test_pooling_counts_every_seed_of_every_result(self):
        p = eval_pooling.pooled(self.pick(around=100), eval_pooling.COLUMNS)
        self.assertEqual((len(p["seeds"]), p["positive"]), (6, 5))
        self.assertAlmostEqual(p["h2h"], 2.5)
        import math
        import statistics as st
        seeds = [-3.0, 1.0, 2.0, 4.0, 6.0, 5.0]
        self.assertEqual(sorted(p["seeds"]), sorted(seeds))
        self.assertAlmostEqual(p["se"], st.stdev(seeds) / math.sqrt(len(seeds)), places=9)
        self.assertAlmostEqual(p["sigmas"], 2.5 / p["se"], places=9)

    def test_the_text_report_states_the_significance(self):
        out = eval_pooling.text_report("t", self.pick(around=100))
        self.assertIn("n=6 se=1.34 +ve=5", out)
        self.assertIn("standard errors above zero", out)

    def test_the_table_renders_a_row_per_result_plus_the_pooled_row(self):
        html = eval_pooling.table_html("t", self.pick(around=100), show_boost=True)
        self.assertEqual(html.count("<tr>"), 3)          # two results plus the header
        self.assertIn("pool-total", html)
        self.assertIn("6 seeds · 5 positive", html)
        self.assertIn("pool-h2h pos", html)               # v9_103M beat the reference on its own row
        self.assertIn("retLow", html)                     # the boost columns were asked for
        losing = eval_pooling.table_html("t", [("v9_9M", result(900, 9, [-4.0, -2.0, -3.0]))])
        self.assertIn("pool-h2h neg", losing)             # a losing checkpoint is marked as such
        self.assertIn("No results in this window yet", eval_pooling.table_html("t", []))


if __name__ == "__main__":
    unittest.main()
