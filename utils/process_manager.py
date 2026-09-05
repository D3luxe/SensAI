"""
Process and Background Task Manager for Training Runs and TensorBoard.
Ensures non-blocking GUI execution and thread-safe IPC controls.
"""

from __future__ import annotations
import os
import sys
import time
import json
import subprocess
import threading
from collections import deque
from typing import Optional, List, Dict, Any, Tuple


class TrainingProcessManager:
    """
    Manages background execution of the training loop, capturing stdout/stderr logs and enabling live IPC controls.
    """
    _instance: Optional[TrainingProcessManager] = None

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.tensorboard_proc: Optional[subprocess.Popen] = None
        self.log_buffer = deque(maxlen=2000)
        self.log_thread: Optional[threading.Thread] = None
        self.is_paused = False
        self.start_time: Optional[float] = None
        self.live_config_file = "config/live_config.json"

    @classmethod
    def get_instance(cls) -> TrainingProcessManager:
        if cls._instance is None:
            cls._instance = TrainingProcessManager()
        return cls._instance

    def _reader_thread(self):
        if self.process and self.process.stdout:
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                stripped = line.rstrip()
                self.log_buffer.append(stripped)
                # Also echo to console if needed
            self.process.stdout.close()

    def start_training(
        self,
        config_path: str = "config/default_config.yaml",
        checkpoint_path: Optional[str] = None
    ) -> Tuple[bool, str]:
        if self.is_running():
            return False, "Training is already running!"

        # Ensure live config is set to unpaused
        self.update_live_config({"paused": False, "save_checkpoint_requested": False})
        self.is_paused = False
        self.start_time = time.time()
        self.log_buffer.clear()
        self.log_buffer.append(f"[{time.strftime('%H:%M:%S')}] Launching Training Process...")

        cmd = [sys.executable, "-u", "train.py", "--config", config_path]
        if checkpoint_path and os.path.exists(checkpoint_path):
            cmd.extend(["--checkpoint", checkpoint_path])
            self.log_buffer.append(f"[{time.strftime('%H:%M:%S')}] Resuming from checkpoint: {checkpoint_path}")
        else:
            self.log_buffer.append(f"[{time.strftime('%H:%M:%S')}] Starting fresh run (no checkpoint loaded).")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )

            # Write PID and start time files
            os.makedirs("logs", exist_ok=True)
            with open(os.path.join("logs", "train.pid"), "w") as f:
                f.write(str(self.process.pid))
            with open(os.path.join("logs", "start_time.txt"), "w") as f:
                f.write(str(self.start_time))

            self.log_thread = threading.Thread(target=self._reader_thread, daemon=True)
            self.log_thread.start()
            return True, f"Training started successfully (PID: {self.process.pid})"
        except Exception as e:
            return False, f"Failed to start training: {e}"

    def stop_training(self) -> Tuple[bool, str]:
        if not self.is_running():
            return False, "No active training process to stop."

        try:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                self.process = None
            else:
                # Orphan process cleanup by PID
                pid_file = os.path.join("logs", "train.pid")
                if os.path.exists(pid_file):
                    try:
                        with open(pid_file, "r") as f:
                            pid = int(f.read().strip())
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/F", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            os.kill(pid, 9)
                    except Exception:
                        pass

            # Remove pidfile
            for fname in ["train.pid", "start_time.txt"]:
                p = os.path.join("logs", fname)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

            self.is_paused = False
            self.update_live_config({"paused": False})
            self.log_buffer.append(f"[{time.strftime('%H:%M:%S')}] Training process stopped.")
            return True, "Training stopped."
        except Exception as e:
            return False, f"Error stopping training: {e}"

    def toggle_pause(self) -> Tuple[bool, str]:
        if not self.is_running():
            return False, "No active training process."

        self.is_paused = not self.is_paused
        self.update_live_config({"paused": self.is_paused})
        status = "Paused" if self.is_paused else "Resumed"
        self.log_buffer.append(f"[{time.strftime('%H:%M:%S')}] Training {status.lower()}.")
        return True, f"Training {status}."

    def trigger_save_checkpoint(self) -> Tuple[bool, str]:
        if not self.is_running():
            return False, "Training is not running."

        self.update_live_config({"save_checkpoint_requested": True})
        self.log_buffer.append(f"[{time.strftime('%H:%M:%S')}] Requested manual checkpoint save.")
        return True, "Checkpoint save requested."

    def get_live_config(self) -> Dict[str, Any]:
        if os.path.exists(self.live_config_file):
            try:
                with open(self.live_config_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def update_live_config(self, updates: Dict[str, Any]):
        live = {}
        if os.path.exists(self.live_config_file):
            try:
                with open(self.live_config_file, "r") as f:
                    live = json.load(f)
            except Exception:
                live = {}

        live.update(updates)
        live["updated_at"] = time.time()

        os.makedirs(os.path.dirname(self.live_config_file), exist_ok=True)
        with open(self.live_config_file, "w") as f:
            json.dump(live, f, indent=2)

    def _is_pid_alive(self, pid: int) -> bool:
        if os.name == "nt":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                SYNCHRONIZE = 0x00100000
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            except Exception:
                return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def is_running(self) -> bool:
        if self.process is not None:
            if self.process.poll() is None:
                return True
            else:
                self.process = None

        # Reconnect check via pidfile and recent metrics mtime
        pid_file = os.path.join("logs", "train.pid")
        if os.path.exists(pid_file):
            try:
                with open(pid_file, "r") as f:
                    stored_pid = int(f.read().strip())
                if self._is_pid_alive(stored_pid):
                    # Check if metrics updated recently (within last 20s)
                    metrics_file = os.path.join("logs", "metrics.json")
                    if os.path.exists(metrics_file):
                        if (time.time() - os.path.getmtime(metrics_file)) < 25.0:
                            return True
            except Exception:
                pass
        return False

    def get_logs(self, max_lines: int = 150) -> str:
        lines = list(self.log_buffer)[-max_lines:]
        return "\n".join(lines)

    def get_status_info(self) -> Dict[str, Any]:
        running = self.is_running()
        
        # Read live config for accurate pause state
        paused = False
        if os.path.exists(self.live_config_file):
            try:
                with open(self.live_config_file, "r") as f:
                    live = json.load(f)
                    paused = bool(live.get("paused", False))
            except Exception:
                paused = self.is_paused
        self.is_paused = paused

        # Calculate accurate PID and elapsed time
        current_pid = None
        if self.process and self.process.poll() is None:
            current_pid = self.process.pid
        elif running:
            pid_file = os.path.join("logs", "train.pid")
            if os.path.exists(pid_file):
                try:
                    with open(pid_file, "r") as f:
                        current_pid = int(f.read().strip())
                except Exception:
                    pass

        elapsed = 0.0
        start_time_file = os.path.join("logs", "start_time.txt")
        if running:
            if self.start_time:
                elapsed = time.time() - self.start_time
            elif os.path.exists(start_time_file):
                try:
                    with open(start_time_file, "r") as f:
                        s_time = float(f.read().strip())
                        elapsed = time.time() - s_time
                except Exception:
                    pass

        # Read latest metrics JSON
        metrics = {}
        metrics_file = os.path.join("logs", "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
            except Exception:
                pass

        return {
            "running": running,
            "paused": paused,
            "pid": current_pid,
            "elapsed_seconds": int(elapsed),
            "metrics": metrics
        }

    def start_tensorboard(self, port: int = 6006) -> Tuple[bool, str]:
        if self.tensorboard_proc and self.tensorboard_proc.poll() is None:
            return True, f"TensorBoard is already running on http://localhost:{port}"

        try:
            cmd = [sys.executable, "-m", "tensorboard.main", "--logdir=logs", f"--port={port}"]
            self.tensorboard_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            return True, f"TensorBoard started on http://localhost:{port}"
        except Exception as e:
            return False, f"Failed to start TensorBoard: {e}"
