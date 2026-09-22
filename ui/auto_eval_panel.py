"""
The Evaluation tab's "Follow the run" panel: scripts/auto_eval.py as a managed background process.

A full eval is ~45 s on 4 workers and a checkpoint lands every ~7 min, so the suite can keep up with
training and every decision point pools a real series instead of three checkpoints picked by hand.
The watcher runs below the trainer's scheduling priority, and is stopped when the UI asks or when
the UI exits.

  AutoEvalRunner.get_instance()   start / stop / status of the watcher
  panel_html(...)                 what the tab shows: state, what it is doing, what it has produced
"""
from __future__ import annotations

import html
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BELOW_NORMAL = 0x00004000 if os.name == "nt" else 0


class AutoEvalRunner:
    """One watcher per UI process. Its log is kept here so the panel can show what it last did."""

    _instance: Optional["AutoEvalRunner"] = None

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.log = deque(maxlen=400)
        self.started_at: Optional[float] = None
        self.settings: Dict[str, Any] = {}
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(cls) -> "AutoEvalRunner":
        if cls._instance is None:
            cls._instance = AutoEvalRunner()
        return cls._instance

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _drain(self, stream):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                self.log.append(line.rstrip())
        except (ValueError, OSError):
            pass
        finally:
            try:
                stream.close()
            except (ValueError, OSError):
                pass

    def start(self, every: int = 1, workers: int = 4, backfill: bool = False) -> str:
        if self.is_running():
            return "Already following the run."
        cmd = [sys.executable, "-u", os.path.join("scripts", "auto_eval.py"),
               "--every", str(max(1, int(every))), "--workers", str(max(1, int(workers)))]
        if backfill:
            cmd.append("--backfill")
        kwargs = {"creationflags": BELOW_NORMAL} if BELOW_NORMAL else {}
        try:
            self.process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, bufsize=1, encoding="utf-8", errors="replace", **kwargs)
        except OSError as e:
            return f"Could not start: {e}"
        self.log.clear()
        self.started_at = time.time()
        self.settings = {"every": int(every), "workers": int(workers), "backfill": bool(backfill)}
        self._thread = threading.Thread(target=self._drain, args=(self.process.stdout,), daemon=True)
        self._thread.start()
        return f"Following the run: every {every} checkpoint(s) on {workers} workers."

    def stop(self) -> str:
        if not self.is_running():
            self.process = None
            return "Not running."
        proc = self.process
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        except OSError as e:
            return f"Could not stop: {e}"
        finally:
            self.process = None
        return "Stopped. Any eval already in flight finishes on its own."

    def log_text(self, lines: int = 12) -> str:
        return "\n".join(list(self.log)[-lines:])

    def status(self) -> Dict[str, Any]:
        done = [l for l in self.log if " -> " in l]
        failed = [l for l in self.log if l.strip().startswith("FAILED")]
        return {"running": self.is_running(), "evaluated": len(done), "failed": len(failed),
                "last": done[-1].strip() if done else "", "settings": dict(self.settings),
                "elapsed": (time.time() - self.started_at) if self.started_at else 0.0,
                "exit_code": None if self.process is None or self.is_running() else self.process.poll()}


def _elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def panel_html(status: Dict[str, Any], version: str = "", reference: str = "") -> str:
    """State of the watcher, and what it is for. Rendered under the Follow-the-run controls."""
    running = status.get("running")
    dot = "ae-on" if running else "ae-off"
    label = "Following the run" if running else "Not following"
    s = status.get("settings") or {}
    bits: List[str] = []
    if running:
        bits.append(f"every {s.get('every', 1)} checkpoint · {s.get('workers', 4)} workers")
        bits.append(f"running {_elapsed(status.get('elapsed', 0))}")
    if status.get("evaluated"):
        bits.append(f"<b>{status['evaluated']}</b> evaluated")
    if status.get("failed"):
        bits.append(f"<span class='ae-bad'>{status['failed']} failed</span>")
    if not running and status.get("exit_code") not in (None, 0):
        bits.append(f"<span class='ae-bad'>exited {status['exit_code']}</span>")
    last = html.escape(status.get("last", ""))
    ref = os.path.basename(reference) if reference else "nothing"
    return (f"<div class='ae-card'>"
            f"<div class='ae-head'><span class='ae-dot {dot}'></span>{html.escape(label)}"
            f"<span class='ae-sub'>{' · '.join(bits)}</span></div>"
            f"<div class='ae-last'>{last or 'Each new checkpoint is evaluated as training writes it, then pooled at a decision point.'}</div>"
            f"<div class='ae-foot'>Reward {html.escape(version or '?')} · head to head vs {html.escape(ref)} · "
            f"~45 s per checkpoint, below the trainer's priority</div></div>")


AUTO_EVAL_CSS = """
.ae-card { background: #111c2e; border: 1px solid #1e293b; border-radius: 10px; padding: 9px 11px; }
.ae-head { color: #e2e8f0; font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 7px; }
.ae-sub { color: #94a3b8; font-weight: 400; font-size: 11.5px; margin-left: auto; text-align: right; }
.ae-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }
.ae-dot.ae-on { background: #34d399; box-shadow: 0 0 0 3px rgba(52,211,153,0.15); }
.ae-dot.ae-off { background: #475569; }
.ae-last { color: #cbd5e1; font-size: 11.5px; margin-top: 6px; font-family: ui-monospace, monospace; word-break: break-all; }
.ae-foot { color: #64748b; font-size: 11px; margin-top: 6px; }
.ae-bad { color: #f87171; }
"""
