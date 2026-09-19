"""
Runs the unit test suite for the dashboard and remembers the last result.

The suite runs in a separate Python process (`python -m unittest discover`), so a five-minute run
that loads RocketSim, torch models and replay pools never touches the UI process. The result is
cached in logs/test_results.json with the git commit and the newest source-file time it ran
against, so the dashboard can say when a result no longer describes the code.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

CACHE_FILE = "logs/test_results.json"
_cache: Optional[Dict[str, Any]] = None


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def _newest_source_mtime() -> float:
    newest = 0.0
    for pattern in ("*.py", "agent/*.py", "env/*.py", "utils/*.py", "ui/*.py", "scripts/*.py", "config/*.yaml",
                    "config/reward_versions/*.json"):
        for p in glob.glob(pattern):
            try:
                newest = max(newest, os.path.getmtime(p))
            except OSError:
                pass
    return newest


def _load_cache() -> Optional[Dict[str, Any]]:
    global _cache
    if _cache is None and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                _cache = json.load(f)
        except (OSError, ValueError):
            _cache = None
    return _cache


def _save_cache(data: Dict[str, Any]):
    global _cache
    _cache = data
    try:
        os.makedirs("logs", exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def parse_unittest_output(text: str) -> Dict[str, Any]:
    ran = re.search(r"^Ran (\d+) tests? in ([\d.]+)s", text, re.M)
    total = int(ran.group(1)) if ran else 0
    summary = re.search(r"^(OK|FAILED)(?: \((.*)\))?\s*$", text, re.M)
    counts = {"failures": 0, "errors": 0, "skipped": 0}
    if summary and summary.group(2):
        for part in summary.group(2).split(","):
            k, _, v = part.strip().partition("=")
            if k in counts and v.isdigit():
                counts[k] = int(v)
    failing: List[Dict[str, str]] = []
    for kind, name, where in re.findall(r"^(FAIL|ERROR): (\S+) \(([^)]+)\)", text, re.M):
        failing.append({"kind": kind, "test": f"{where}"})
    passed = total - counts["failures"] - counts["errors"] - counts["skipped"]
    return {"total_tests": total, "passed": passed, **counts, "failing": failing,
            "all_passed": bool(summary and summary.group(1) == "OK" and total > 0)}


def run_all_unit_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run the whole suite in a subprocess; returns (and caches) the parsed result."""
    start = time.time()
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-p", "test_*.py"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = proc.stderr + ("\n" + proc.stdout if proc.stdout.strip() else "")
    res = parse_unittest_output(out)
    res.update(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), ran_at=time.time(), duration_seconds=round(time.time() - start, 1),
               git_head=_git_head(), raw_output=out[-60000:])
    _save_cache(res)
    return res


def get_cached_or_run_tests(force_refresh: bool = False) -> Dict[str, Any]:
    """The last result without running anything (unless force_refresh), with a staleness note."""
    if force_refresh:
        return run_all_unit_tests()
    res = _load_cache()
    if not res or "ran_at" not in res:
        # Nothing cached, or a result from the old in-process runner, which did not record what it ran against
        return {"status": "NOT_RUN", "total_tests": 0, "passed": 0, "failing": [], "all_passed": False,
                "raw_output": "No test run recorded yet.", "stale": True}
    res = dict(res)
    head = _git_head()
    res["stale"] = bool((head and res.get("git_head") and head != res.get("git_head"))
                        or _newest_source_mtime() > float(res.get("ran_at", 0)))
    return res


def format_test_results_markdown(res: Dict[str, Any]) -> str:
    if res.get("status") == "NOT_RUN":
        return "No test run recorded yet. **Run all** runs the suite in the background (about 5 minutes)."
    total, passed = res.get("total_tests", 0), res.get("passed", 0)
    head = ("**All tests pass**" if res.get("all_passed") else f"**{len(res.get('failing', []))} failing**")
    lines = [f"{head}: {passed} of {total} passed"
             + (f", {res.get('skipped')} skipped" if res.get("skipped") else "")
             + f" · {res.get('duration_seconds', 0):.0f} s · run {res.get('timestamp', '?')}"
             + (f" on {res['git_head']}" if res.get("git_head") else "")]
    if res.get("stale"):
        lines.append("<span style='color:#fbbf24'>The code has changed since this run; run again for a current result.</span>")
    for f in res.get("failing", [])[:25]:
        lines.append(f"- {f['kind']}: `{f['test']}`")
    if len(res.get("failing", [])) > 25:
        lines.append(f"- … and {len(res['failing']) - 25} more (see the output)")
    return "\n\n".join(lines[:2]) + ("\n\n" + "\n".join(lines[2:]) if len(lines) > 2 else "")
