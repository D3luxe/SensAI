"""
Reading logs/history.jsonl without re-reading it.

The trainer appends one JSON record per iteration and the file never shrinks during a run (it is
hundreds of MB after a long one), so the UI cannot parse it on every refresh. This keeps a
per-process cache of the current reward run's records and reads only the bytes appended since the
last call.

A "run" is the stretch of records sharing one (reward_version, reward_run_start_step). Records
written before the trainer stamped those fields belong to the legacy v2 run. mean_reward and the
losses are only comparable within a run: a new reward version changes the scale of everything.

  current_run(path)     compact records of the run the newest record belongs to
  recent(path, n)       the last n records whatever run they belong to
  sampled(path, n)      n records spread over the whole file (for an all-history view)
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

HISTORY_FILE = "logs/history.jsonl"
# Fields the UI charts. A record keeps only these, so anything the trainer starts logging has to be
# added here before it can reach a curve.
KEEP = ("iteration", "global_step", "mean_reward", "policy_loss", "value_loss", "entropy", "sps",
        "ball_touches", "goals", "reward_version", "reward_run_start_step", "critic_warmup", "telemetry",
        "timestamp", "approx_kl", "clip_fraction", "explained_variance", "learning_rate")
_CHUNK = 4 * 1024 * 1024
# Records held per run. Past this the older half is thinned to every other record, so a run of any
# length costs bounded memory and its curve keeps its shape; the backward scan stops here too.
MAX_RECORDS = 20000

_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}


def run_key(rec: Dict[str, Any]) -> Tuple[str, int]:
    return (str(rec.get("reward_version") or "v2"), int(rec.get("reward_run_start_step") or 0))


def _compact(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: rec[k] for k in KEEP if k in rec}


def _parse(line: bytes) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line.decode("utf-8", errors="ignore"))
    except ValueError:
        return None


def _scan_back_for_run(fh, size: int) -> Tuple[List[Dict[str, Any]], Optional[Tuple[str, int]]]:
    """Walk backwards from the end in chunks until a record from an earlier run appears."""
    records: List[Dict[str, Any]] = []
    key = None
    end, carry = size, b""
    while end > 0:
        start = max(0, end - _CHUNK)
        fh.seek(start)
        buf = fh.read(end - start) + carry
        lines = buf.split(b"\n")
        carry = lines.pop(0) if start > 0 else b""
        if start == 0 and carry:
            lines.insert(0, carry)
            carry = b""
        batch = []
        stop = False
        for raw in reversed(lines):
            rec = _parse(raw)
            if rec is None:
                continue
            k = run_key(rec)
            if key is None:
                key = k
            if k != key:
                stop = True
                break
            batch.append(_compact(rec))
        records.extend(batch)
        if stop or len(records) >= MAX_RECORDS:
            break
        end = start
    records.reverse()
    return records, key


def current_run(path: str = HISTORY_FILE) -> Dict[str, Any]:
    """{"key": (version, start_step) or None, "records": [...]} for the newest run in the file."""
    if not os.path.exists(path):
        return {"key": None, "records": []}
    size = os.path.getsize(path)
    with _lock:
        c = _cache.get(path)
        if c is None or size < c["offset"]:
            with open(path, "rb") as fh:
                records, key = _scan_back_for_run(fh, size)
            c = {"offset": size, "key": key, "records": records}
            _cache[path] = c
        elif size > c["offset"]:
            with open(path, "rb") as fh:
                fh.seek(c["offset"])
                data = fh.read(size - c["offset"])
            # only complete lines; a half-written tail waits for the next call
            cut = data.rfind(b"\n") + 1
            for raw in data[:cut].split(b"\n"):
                rec = _parse(raw)
                if rec is None:
                    continue
                k = run_key(rec)
                if k != c["key"]:
                    c["key"], c["records"] = k, []
                c["records"].append(_compact(rec))
            if len(c["records"]) > MAX_RECORDS:
                half = len(c["records"]) // 2
                c["records"] = c["records"][:half][::2] + c["records"][half:]
            c["offset"] += cut
        return {"key": c["key"], "records": list(c["records"])}


def recent(path: str = HISTORY_FILE, n: int = 100) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    size = os.path.getsize(path)
    want = min(size, max(8192, n * 4096))
    with open(path, "rb") as fh:
        fh.seek(size - want)
        lines = fh.read(want).split(b"\n")
    if size > want and lines:
        lines.pop(0)
    out = [r for r in (_parse(x) for x in lines) if r is not None]
    return [_compact(r) for r in out[-n:]]


def sampled(path: str = HISTORY_FILE, n: int = 300) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    size = os.path.getsize(path)
    out = []
    with open(path, "rb") as fh:
        for i in range(n):
            fh.seek(int(i * size / n))
            if i:
                fh.readline()
            rec = _parse(fh.readline())
            if rec is not None:
                out.append(_compact(rec))
    return out
