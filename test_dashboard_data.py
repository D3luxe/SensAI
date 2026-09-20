"""
The data behind the dashboard: run boundaries in logs/history.jsonl, eval-result comparison, and
the unit-test output parser.
"""
import json
import os
import tempfile
import unittest

from utils import eval_results, run_history
from utils.test_runner import parse_unittest_output


def _rec(it, version=None, start=None, reward=1.0):
    r = {"iteration": it, "global_step": it * 100, "mean_reward": reward, "telemetry": {"jump_rate_pct": 5.0}}
    if version:
        r.update(reward_version=version, reward_run_start_step=start)
    return r


class TestRunHistory(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "history.jsonl")
        run_history._cache.pop(self.path, None)

    def _write(self, recs, mode="w"):
        with open(self.path, mode) as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    def test_current_run_stops_at_the_previous_version(self):
        self._write([_rec(i) for i in range(1, 50)] + [_rec(i, "v3", 4800) for i in range(49, 60)])
        run = run_history.current_run(self.path)
        self.assertEqual(run["key"], ("v3", 4800))
        self.assertEqual([r["iteration"] for r in run["records"]], list(range(49, 60)))

    def test_unstamped_history_is_one_legacy_v2_run(self):
        self._write([_rec(i) for i in range(1, 30)])
        run = run_history.current_run(self.path)
        self.assertEqual(run["key"], ("v2", 0))
        self.assertEqual(len(run["records"]), 29)

    def test_appends_are_read_incrementally_and_a_new_run_resets(self):
        self._write([_rec(i, "v3", 0) for i in range(5)])
        self.assertEqual(len(run_history.current_run(self.path)["records"]), 5)
        self._write([_rec(i, "v3", 0) for i in range(5, 8)], mode="a")
        self.assertEqual(len(run_history.current_run(self.path)["records"]), 8)
        self._write([_rec(i, "v4", 800) for i in range(8, 10)], mode="a")
        run = run_history.current_run(self.path)
        self.assertEqual(run["key"], ("v4", 800))
        self.assertEqual(len(run["records"]), 2)

    def test_charted_fields_survive_compaction(self):
        rec = _rec(1, "v4", 0)
        rec.update(approx_kl=0.02, clip_fraction=0.15, explained_variance=0.84, learning_rate=3e-4)
        self._write([rec])
        got = run_history.current_run(self.path)["records"][0]
        for k in ("approx_kl", "clip_fraction", "explained_variance", "learning_rate"):
            self.assertIn(k, got, f"{k} is charted but dropped by run_history.KEEP")

    def test_a_half_written_line_waits(self):
        self._write([_rec(0, "v3", 0)])
        run_history.current_run(self.path)
        with open(self.path, "a") as f:
            f.write('{"iteration": 1, "reward_ver')
        self.assertEqual(len(run_history.current_run(self.path)["records"]), 1)
        with open(self.path, "a") as f:
            f.write('sion": "v3", "reward_run_start_step": 0}\n')
        self.assertEqual(len(run_history.current_run(self.path)["records"]), 2)


def _result(values, seeds=True):
    def stat(v):
        per = [v - 1, v, v + 1] if seeds else [v]
        return {"mean": v, "min": min(per), "max": max(per), "per_seed": per}
    return {"suite_version": 1, "results": {g: {k: stat(v) for k, v in m.items()} for g, m in values.items()}}


class TestEvalResults(unittest.TestCase):
    def test_seed_ranges_decide_what_is_a_difference(self):
        a = _result({"necto": {"goal_diff_per_10min": -40.0, "touches_per_min": 5.0}})
        b = _result({"necto": {"goal_diff_per_10min": -30.0, "touches_per_min": 5.5}})
        rows = {r["metric"]: r for r in eval_results.compare_results(a, b)}
        self.assertEqual(rows["goal_diff_per_10min"]["verdict"], "better")
        self.assertEqual(rows["touches_per_min"]["verdict"], "noise")

    def test_direction_decides_better_or_worse(self):
        a = _result({"necto": {"ga_cause_caught_upfield_pct": 30.0}})
        b = _result({"necto": {"ga_cause_caught_upfield_pct": 50.0}})
        self.assertEqual(eval_results.compare_results(a, b)[0]["verdict"], "worse")

    def test_scenario_rates_use_binomial_error(self):
        a = _result({"scenario_retreat": {"n": 24, "conceded_pct": 20.0}}, seeds=False)
        b = _result({"scenario_retreat": {"n": 24, "conceded_pct": 70.0}}, seeds=False)
        c = _result({"scenario_retreat": {"n": 24, "conceded_pct": 30.0}}, seeds=False)
        self.assertEqual(eval_results.compare_results(a, b)[0]["verdict"], "worse")
        self.assertEqual(eval_results.compare_results(a, c)[0]["verdict"], "noise")

    def test_head_to_head_is_judged_on_the_sign_of_its_seeds(self):
        beats = _result({"reference": {"goal_diff_per_10min": 5.0}})       # seeds 4, 5, 6
        loses = _result({"reference": {"goal_diff_per_10min": -5.0}})      # seeds -6, -5, -4
        level = _result({"reference": {"goal_diff_per_10min": 0.5}})       # seeds -0.5, 0.5, 1.5
        for res, verdict in ((beats, "better"), (loses, "worse"), (level, "noise")):
            rows = eval_results.head_to_head_rows(res)
            self.assertEqual(rows[0]["verdict"], verdict)
        self.assertEqual(eval_results.head_to_head_rows(_result({"necto": {"goal_diff_per_10min": 5.0}})), [])

    def test_scorecard_leads_with_the_head_to_head_when_there_is_one(self):
        res = _result({"reference": {"goal_diff_per_10min": 5.0}, "necto": {"goal_diff_per_10min": -30.0}})
        res["reference"] = "checkpoints/baselines/v3_iter198000.pt"
        html = eval_results.scorecard_html(res, None, "b", None)
        self.assertIn("Head to head vs v3_iter198000.pt", html)
        self.assertLess(html.index("Head to head"), html.index("Primary"))

    def test_real_baselines_match_the_cli(self):
        if not all(os.path.exists(p) for p in ("evals/baselines/v2_iter172000.json", "evals/baselines/v2_iter173600.json")):
            self.skipTest("baseline evals not present")
        a, b = eval_results.load("evals/baselines/v2_iter172000.json"), eval_results.load("evals/baselines/v2_iter173600.json")
        self.assertEqual(sum(r["clear"] for r in eval_results.compare_results(a, b)), 17)
        html = eval_results.scorecard_html(b, a, "b", "a")
        for section in eval_results.HEADLINE:
            self.assertIn(section, html)


class TestUnittestParser(unittest.TestCase):
    def test_failed_run(self):
        out = ("FAIL: test_x (test_a.TestA.test_x)\nERROR: test_y (test_b.TestB.test_y)\n"
               "Ran 10 tests in 1.5s\n\nFAILED (failures=1, errors=1, skipped=2)\n")
        r = parse_unittest_output(out)
        self.assertEqual((r["total_tests"], r["passed"], r["skipped"]), (10, 6, 2))
        self.assertEqual([f["test"] for f in r["failing"]], ["test_a.TestA.test_x", "test_b.TestB.test_y"])
        self.assertFalse(r["all_passed"])

    def test_clean_run(self):
        r = parse_unittest_output("Ran 691 tests in 309.8s\n\nOK (skipped=1)\n")
        self.assertTrue(r["all_passed"])
        self.assertEqual(r["passed"], 690)


if __name__ == "__main__":
    unittest.main()
