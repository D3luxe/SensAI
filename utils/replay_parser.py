"""
Rocket League Match Replay Ingestion & State Parsing Engine.
Extracts 3D car states, velocities, rotations, and ball dynamics from match replays
and builds indexed numpy datasets for ReplayStateSetter training.
"""

from __future__ import annotations
import os
import glob
import json
import random
import time
import math
import shutil
import subprocess
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


def _quat_to_euler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """
    Converts a replay quaternion (x, y, z, w) to (pitch, yaw, roll) in radians, in RocketSim's
    rsim.Angle convention. Read off the rotation matrix's forward, right and up columns so that
    rsim.Angle(pitch, yaw, roll).as_rot_mat() reproduces the replay orientation exactly.
    """
    fwd_z = 2.0 * (x * z - y * w)
    pitch = math.asin(max(-1.0, min(1.0, fwd_z)))
    yaw = math.atan2(2.0 * (x * y + z * w), 1.0 - 2.0 * (y * y + z * z))
    right_z = 2.0 * (y * z + x * w)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(-right_z, up_z)
    return float(pitch), float(yaw), float(roll)


def _find_rrrocket() -> Optional[str]:
    """Finds rrrocket binary in local project bin directory or system PATH."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, "bin", "rrrocket.exe"),
        os.path.join(project_root, "bin", "rrrocket"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    found = shutil.which("rrrocket")
    if found:
        return found
    return None


def _ensure_rrrocket() -> Optional[str]:
    """Ensures rrrocket executable is available, downloading prebuilt binary on Windows x86_64 if needed."""
    existing = _find_rrrocket()
    if existing:
        return existing

    import platform
    if platform.system() == "Windows" and platform.machine().lower() in ["amd64", "x86_64"]:
        try:
            import urllib.request
            import zipfile
            import io
            url = "https://github.com/nickbabcock/rrrocket/releases/download/v0.11.5/rrrocket-0.11.5-x86_64-pc-windows-msvc.zip"
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            bin_dir = os.path.join(project_root, "bin")
            os.makedirs(bin_dir, exist_ok=True)
            print(f"[ReplayParser] Auto-downloading rrrocket binary from GitHub releases...")
            req = urllib.request.Request(url, headers={"User-Agent": "SenseiBot"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_zip = resp.read()
            with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
                for member in z.namelist():
                    if member.endswith("rrrocket.exe"):
                        with z.open(member) as source, open(os.path.join(bin_dir, "rrrocket.exe"), "wb") as target:
                            target.write(source.read())
            return _find_rrrocket()
        except Exception as e:
            print(f"[ReplayParser] Warning: Could not auto-download rrrocket binary: {e}")
    return None



def get_default_demo_dir() -> str:
    """Auto-detects active Rocket League demo directory (OneDrive or standard Documents)."""
    candidates = [
        os.path.expandvars(r"%USERPROFILE%\OneDrive\Documents\RL_ML_Training\replays"),
        os.path.expandvars(r"%USERPROFILE%\OneDrive\Documents\My Games\Rocket League\TAGame\Demos"),
        os.path.expandvars(r"%USERPROFILE%\Documents\My Games\Rocket League\TAGame\Demos"),
        os.path.join("data", "replays")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[1]


DEFAULT_DEMO_DIR = get_default_demo_dir()
DEFAULT_POOL_PATH = os.path.join("data", "replays", "replays_pool.npz")

# Keep the most recent frames only. At 30 Hz one 1v1 replay is ~5-7k frames.
MAX_POOL_FRAMES = 2_000_000

REQUIRED_POOL_KEYS = ("ball_pos", "ball_vel", "car_pos", "car_vel", "car_rot", "car_boost")
# Written by full-rate parsing; absent from legacy pools.
#   frame_time  (N,)          replay clock of the network frame, seconds
#   car_time    (N, cars)     replay clock of that car's last RigidBody update, seconds
#   car_ang_vel (N, cars, 3)  angular velocity as replicated (world axes, raw network scale ~x80-100)
#   segment_id  (N,)          contiguous-recording id; frames from different replays never pair
TIMED_POOL_KEYS = ("frame_time", "car_time", "car_ang_vel", "segment_id")
# Legacy pools sampled every 10th frame of a 30 Hz replay.
LEGACY_FRAME_DT = 10.0 / 30.0
# Longest gap between two car updates still treated as one continuous transition.
MAX_TRANSITION_DT = 0.15


def _car_vec(arr: np.ndarray, i: int, car_idx: int) -> np.ndarray:
    a = arr[i]
    return a[car_idx] if a.ndim > 1 else a


def is_fresh_car_update(data: Dict[str, np.ndarray], idx: int, car_idx: int) -> bool:
    """True if the car's physics state was replicated in frame idx (not a carried-over duplicate)."""
    if "car_time" not in data:
        return True
    return abs(float(data["frame_time"][idx]) - float(data["car_time"][idx, car_idx])) < 1e-3


def find_car_transition(data: Dict[str, np.ndarray], idx: int, car_idx: int) -> Optional[Tuple[int, float]]:
    """
    Returns (next_idx, dt) for the next genuine state update of this car after frame idx, or None when
    there is no usable continuous transition (end of recording, gap too long, or a teleport such as a
    demolition respawn or kickoff reset).
    """
    n = len(data["car_pos"])
    if "car_time" not in data:
        if idx + 1 >= n:
            return None
        if float(np.linalg.norm(_car_vec(data["car_pos"], idx + 1, car_idx) - _car_vec(data["car_pos"], idx, car_idx))) >= 250.0:
            return None
        return idx + 1, LEGACY_FRAME_DT

    car_time, seg = data["car_time"], data["segment_id"]
    t0 = float(car_time[idx, car_idx])
    j = idx + 1
    while j < n and seg[j] == seg[idx]:
        dt = float(car_time[j, car_idx]) - t0
        if dt > 1e-4:
            if dt > MAX_TRANSITION_DT:
                return None
            p0, p1 = data["car_pos"][idx, car_idx], data["car_pos"][j, car_idx]
            v_max = max(float(np.linalg.norm(data["car_vel"][idx, car_idx])), float(np.linalg.norm(data["car_vel"][j, car_idx])))
            if float(np.linalg.norm(p1 - p0)) > v_max * dt * 1.5 + 100.0:
                return None
            return j, dt
        j += 1
    return None


def _merge_pool(buffer: Optional[Dict[str, np.ndarray]], chunks: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenates parsed chunks onto the pool, offsetting segment ids and enforcing MAX_POOL_FRAMES."""
    keys = list(REQUIRED_POOL_KEYS)
    chunks_timed = all(all(k in c for k in TIMED_POOL_KEYS) for c in chunks)
    buffer_timed = buffer is None or all(k in buffer for k in TIMED_POOL_KEYS)
    if chunks_timed and buffer_timed:
        keys += list(TIMED_POOL_KEYS)
    elif chunks_timed:
        print("[ReplayParser] Warning: appending timed frames to a legacy pool drops timing; clear the pool to keep it.")

    seg_base = 0
    if buffer is not None and "segment_id" in keys and len(buffer["segment_id"]):
        seg_base = int(buffer["segment_id"].max()) + 1
    parts = {k: ([] if buffer is None else [buffer[k]]) for k in keys}
    for c in chunks:
        for k in keys:
            v = np.asarray(c[k])
            if k == "segment_id":
                v = v.astype(np.int32) + seg_base
            parts[k].append(v)
        if "segment_id" in keys:
            seg_base = int(parts["segment_id"][-1].max()) + 1

    merged = {k: np.concatenate(parts[k], axis=0) for k in keys}
    if len(merged["ball_pos"]) > MAX_POOL_FRAMES:
        merged = {k: v[-MAX_POOL_FRAMES:] for k, v in merged.items()}
    return merged


class ReplayParser:
    """
    Parses Rocket League replay states and maintains a fast, memory-mapped replay buffer.
    """
    # Keep every Nth network frame of a .replay (replays record at 30 Hz).
    frame_stride: int = 1

    def __init__(self, pool_path: str = DEFAULT_POOL_PATH, demo_dir: Optional[str] = None):
        self.pool_path = pool_path
        self.demo_dir = demo_dir or DEFAULT_DEMO_DIR
        os.makedirs(os.path.dirname(self.pool_path), exist_ok=True)
        self.states_buffer: Optional[Dict[str, np.ndarray]] = None
        self.last_ingest_report: Dict[str, Any] = {
            "total_files": 0,
            "parsed_files": 0,
            "rejected_files": [],
            "total_frames": 0
        }
        self.load_pool()

    def load_pool(self) -> bool:
        """Loads cached replay frame states from disk if present."""
        if os.path.exists(self.pool_path):
            try:
                data = np.load(self.pool_path)
                self.states_buffer = {
                    "ball_pos": data["ball_pos"],       # (N, 3)
                    "ball_vel": data["ball_vel"],       # (N, 3)
                    "car_pos": data["car_pos"],         # (N, num_cars, 3)
                    "car_vel": data["car_vel"],         # (N, num_cars, 3)
                    "car_rot": data["car_rot"],         # (N, num_cars, 3) pitch, yaw, roll
                    "car_boost": data["car_boost"]      # (N, num_cars)
                }
                if all(k in data.files for k in TIMED_POOL_KEYS):
                    self.states_buffer.update({k: data[k] for k in TIMED_POOL_KEYS})
                return True
            except Exception as e:
                print(f"[ReplayParser] Warning: Could not load {self.pool_path}: {e}")
        return False

    def save_pool(self):
        """Saves current state buffer to compressed npz."""
        if self.states_buffer is not None:
            os.makedirs(os.path.dirname(self.pool_path), exist_ok=True)
            np.savez_compressed(self.pool_path, **self.states_buffer)

    def get_pool_stats(self) -> Dict[str, Any]:
        """Returns statistics on the current replay pool."""
        if not os.path.exists(self.pool_path):
            self.states_buffer = None
            return {"total_frames": 0, "num_matches": 0, "file_size_mb": 0.0}

        if self.states_buffer is None:
            self.load_pool()
        if self.states_buffer is None:
            return {"total_frames": 0, "num_matches": 0, "file_size_mb": 0.0}

        n_frames = len(self.states_buffer["ball_pos"])
        file_size = os.path.getsize(self.pool_path) / (1024 * 1024) if os.path.exists(self.pool_path) else 0.0
        return {
            "total_frames": n_frames,
            "num_matches": max(1, n_frames // 250),
            "file_size_mb": round(file_size, 2)
        }

    def clear_pool(self) -> bool:
        """Clears active memory buffer and removes saved pool from disk."""
        self.states_buffer = None
        if os.path.exists(self.pool_path):
            try:
                os.remove(self.pool_path)
            except Exception as e:
                print(f"[ReplayParser] Warning: Could not remove {self.pool_path}: {e}")
        return True

    def scan_demos(self, max_replays: int = 50, sort: str = "newest") -> List[str]:
        """Scans demo_dir for replay/dataset files, returns sorted file paths."""
        if not os.path.exists(self.demo_dir):
            return []
        files = (
            glob.glob(os.path.join(self.demo_dir, "*.replay")) +
            glob.glob(os.path.join(self.demo_dir, "*.npz")) +
            glob.glob(os.path.join(self.demo_dir, "*.json"))
        )
        if sort == "newest":
            files.sort(key=os.path.getmtime, reverse=True)
        elif sort == "oldest":
            files.sort(key=os.path.getmtime)
        elif sort == "random":
            random.shuffle(files)
        if max_replays > 0 and len(files) > max_replays:
            files = files[:max_replays]
        return files

    def ingest_zip(self, zip_path: str) -> Tuple[int, int]:
        """
        Extracts and ingests all .replay, .npz, and .json files contained inside a .zip archive.
        Uses multi-engine extraction (zipfile -> tar.exe -> powershell) to handle large/Zip64 archives.
        Handles nested subfolders and temporary cleanup automatically.
        """
        import zipfile
        import tempfile
        import shutil
        import subprocess

        if not os.path.exists(zip_path):
            return 0, 0

        temp_dir = tempfile.mkdtemp(prefix="rl_replays_zip_")
        try:
            extracted_ok = False

            # 1. Try standard Python zipfile
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                extracted_ok = True
            except Exception:
                pass

            # 2. Fallback to Windows built-in tar.exe
            if not extracted_ok or not os.listdir(temp_dir):
                try:
                    subprocess.run(['tar', '-xf', zip_path, '-C', temp_dir], capture_output=True)
                    if os.listdir(temp_dir):
                        extracted_ok = True
                except Exception:
                    pass

            # 3. Fallback to PowerShell Expand-Archive
            if not extracted_ok or not os.listdir(temp_dir):
                try:
                    ps_cmd = f"Expand-Archive -LiteralPath '{zip_path}' -DestinationPath '{temp_dir}' -Force"
                    subprocess.run(['powershell', '-Command', ps_cmd], capture_output=True)
                except Exception:
                    pass

            files = []
            for root, _, filenames in os.walk(temp_dir):
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in [".replay", ".npz", ".json"]:
                        files.append(os.path.join(root, fn))

            if not files:
                return 0, 0

            chunks = []
            processed_count = 0
            rejected_files = []
            for fpath in files:
                try:
                    frames = self._parse_file(fpath)
                    if frames and len(frames["ball_pos"]) > 0:
                        chunks.append(frames)
                        processed_count += 1
                    else:
                        rejected_files.append(os.path.basename(fpath))
                except Exception as e:
                    rejected_files.append(os.path.basename(fpath))
                    print(f"[ReplayParser] Error reading extracted {fpath}: {e}")

            self.last_ingest_report = {
                "total_files": len(files),
                "parsed_files": processed_count,
                "rejected_files": rejected_files,
                "total_frames": 0
            }

            if not chunks:
                return 0, 0

            n_new = sum(len(c["ball_pos"]) for c in chunks)
            self.states_buffer = _merge_pool(self.states_buffer, chunks)
            self.last_ingest_report["total_frames"] = n_new
            self.save_pool()
            return processed_count, n_new
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def sample_state(self, num_cars: int = 2) -> Optional[Dict[str, Any]]:
        """
        Samples a single random game state from the replay pool.
        """
        if self.states_buffer is None:
            if not self.load_pool():
                return None

        n_frames = len(self.states_buffer["ball_pos"])
        if n_frames == 0:
            return None

        idx = random.randint(0, n_frames - 1)
        b_pos = self.states_buffer["ball_pos"][idx].copy()
        b_vel = self.states_buffer["ball_vel"][idx].copy()
        c_pos = self.states_buffer["car_pos"][idx].copy()
        c_vel = self.states_buffer["car_vel"][idx].copy()
        c_rot = self.states_buffer["car_rot"][idx].copy()
        c_bst = self.states_buffer["car_boost"][idx].copy()

        # Handle car count dimension adaptation
        if len(c_pos) < num_cars:
            # Duplicate / mirror if needed
            pad_count = num_cars - len(c_pos)
            c_pos = np.vstack([c_pos, -c_pos[:pad_count]])
            c_vel = np.vstack([c_vel, -c_vel[:pad_count]])
            c_rot = np.vstack([c_rot, c_rot[:pad_count]])
            c_bst = np.concatenate([c_bst, c_bst[:pad_count]])
        elif len(c_pos) > num_cars:
            c_pos = c_pos[:num_cars]
            c_vel = c_vel[:num_cars]
            c_rot = c_rot[:num_cars]
            c_bst = c_bst[:num_cars]

        return {
            "ball_pos": b_pos,
            "ball_vel": b_vel,
            "car_pos": c_pos,
            "car_vel": c_vel,
            "car_rot": c_rot,
            "car_boost": c_bst
        }

    def ingest_directory(
        self,
        directory: Optional[str] = None,
        max_replays: int = 50,
        sort_mode: str = "newest",
        sort: Optional[str] = None,
        progress_cb: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Scans a local directory for .replay / .npz / .json files, respects max_replays limit,
        extracts frames, and appends them to the replay pool.
        Returns: (num_replays_processed, num_frames_ingested)
        """
        _t0 = time.time()
        if directory is None:
            directory = self.demo_dir
        if sort is not None:
            sort_mode = sort

        if not os.path.exists(directory):
            return {"parsed_files": 0, "total_frames": 0, "elapsed_seconds": 0.0}

        files = glob.glob(os.path.join(directory, "*.replay")) + glob.glob(os.path.join(directory, "*.npz")) + glob.glob(os.path.join(directory, "*.json"))
        if not files:
            return {"parsed_files": 0, "total_frames": 0, "elapsed_seconds": 0.0}

        # Sort files based on user preference
        if sort_mode == "newest":
            files.sort(key=os.path.getmtime, reverse=True)
        elif sort_mode == "oldest":
            files.sort(key=os.path.getmtime)
        elif sort_mode == "random":
            random.shuffle(files)

        # Enforce configurable ingestion limit
        if max_replays > 0 and len(files) > max_replays:
            files = files[:max_replays]

        chunks = []
        processed_count = 0
        total_files = len(files)
        rejected_files = []

        for i, file_path in enumerate(files):
            try:
                frames = self._parse_file(file_path)
                if frames and len(frames["ball_pos"]) > 0:
                    chunks.append(frames)
                    processed_count += 1
                else:
                    rejected_files.append(os.path.basename(file_path))
            except Exception as e:
                rejected_files.append(os.path.basename(file_path))
                print(f"[ReplayParser] Error reading {file_path}: {e}")

            if progress_cb:
                progress_cb(float(i + 1) / total_files, f"Ingested {i+1}/{total_files} replays...")

        self.last_ingest_report = {
            "total_files": total_files,
            "parsed_files": processed_count,
            "rejected_files": rejected_files,
            "total_frames": 0
        }

        if not chunks:
            return {"parsed_files": 0, "total_frames": 0, "elapsed_seconds": 0.0}

        n_new = sum(len(c["ball_pos"]) for c in chunks)
        self.states_buffer = _merge_pool(self.states_buffer, chunks)
        self.last_ingest_report["total_frames"] = n_new
        self.save_pool()
        return {"parsed_files": processed_count, "total_frames": n_new, "elapsed_seconds": round(time.time() - _t0, 2)}

    def _parse_file(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Parses an individual replay file or pre-formatted numpy/json dataset."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".npz":
            data = np.load(file_path)
            out = {k: data[k] for k in REQUIRED_POOL_KEYS}
            out.update({k: data[k] for k in TIMED_POOL_KEYS if k in data.files})
            return out

        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {
                "ball_pos": np.array(raw["ball_pos"], dtype=np.float32),
                "ball_vel": np.array(raw["ball_vel"], dtype=np.float32),
                "car_pos": np.array(raw["car_pos"], dtype=np.float32),
                "car_vel": np.array(raw["car_vel"], dtype=np.float32),
                "car_rot": np.array(raw["car_rot"], dtype=np.float32),
                "car_boost": np.array(raw["car_boost"], dtype=np.float32)
            }

        elif ext == ".replay":
            # Direct binary frame extractor for Rocket League .replay files
            return self._extract_replay_binary(file_path)

        return None

    def _extract_replay_binary(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extracts genuine in-game frame tuples from Rocket League .replay files using rrrocket.
        Decodes ball and car 3D positions, velocities, orientations, and boost amounts.
        """
        rrrocket_bin = _ensure_rrrocket()
        if not rrrocket_bin:
            print(f"[ReplayParser] rrrocket parser executable not found. Cannot parse '{os.path.basename(file_path)}'.")
            return None

        try:
            proc = subprocess.run(
                [rrrocket_bin, "-n", file_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=60
            )
            if proc.returncode != 0:
                print(f"[ReplayParser] Corrupt or unparseable replay '{os.path.basename(file_path)}' (code {proc.returncode}).")
                return None

            data = json.loads(proc.stdout)
        except Exception as e:
            print(f"[ReplayParser] Error parsing '{os.path.basename(file_path)}' with rrrocket: {e}")
            return None

        objects = data.get("objects", [])
        frames = data.get("network_frames", {}).get("frames", [])
        if not frames or not objects:
            return None

        actor_types: Dict[int, str] = {}
        car_teams: Dict[int, int] = {}
        boost_to_car: Dict[int, int] = {}
        active_ball_id: Optional[int] = None
        active_cars: Dict[int, Dict[str, Any]] = {}
        active_ball = {"pos": [0.0, 0.0, 93.0], "vel": [0.0, 0.0, 0.0]}

        extracted_b_pos = []
        extracted_b_vel = []
        extracted_c_pos = []
        extracted_c_vel = []
        extracted_c_rot = []
        extracted_c_bst = []
        extracted_frame_time = []
        extracted_car_time = []
        extracted_c_angv = []

        for f_idx, f in enumerate(frames):
            for da in f.get("deleted_actors", []):
                if da == active_ball_id:
                    active_ball_id = None
                if da in active_cars:
                    del active_cars[da]
                if da in boost_to_car:
                    del boost_to_car[da]
                if da in actor_types:
                    del actor_types[da]

            for na in f.get("new_actors", []):
                aid = na["actor_id"]
                obj_id = na.get("object_id", -1)
                obj_name = objects[obj_id] if 0 <= obj_id < len(objects) else ""
                if "Ball" in obj_name and "Ball_Default" in obj_name:
                    actor_types[aid] = "ball"
                    active_ball_id = aid
                elif "Car_Default" in obj_name or ("Car." in obj_name and "CarComponent" not in obj_name):
                    actor_types[aid] = "car"
                    active_cars[aid] = {
                        "pos": [0.0, 0.0, 17.0],
                        "vel": [0.0, 0.0, 0.0],
                        "rot": [0.0, 0.0, 0.0],
                        "ang_vel": [0.0, 0.0, 0.0],
                        "time": float(f.get("time", 0.0)),
                        "boost": 33.3
                    }
                elif "CarComponent_Boost" in obj_name:
                    actor_types[aid] = "boost"

            for ua in f.get("updated_actors", []):
                aid = ua["actor_id"]
                atype = actor_types.get(aid)
                attr = ua.get("attribute", {})
                obj_id = ua.get("object_id", -1)
                obj_name = objects[obj_id] if 0 <= obj_id < len(objects) else ""

                if atype == "car":
                    if "TeamPaint" in attr:
                        car_teams[aid] = attr["TeamPaint"].get("team", 0)
                    if "RigidBody" in attr:
                        rb = attr["RigidBody"]
                        loc = rb.get("location")
                        vel = rb.get("linear_velocity")
                        rot = rb.get("rotation")
                        if loc and aid in active_cars:
                            active_cars[aid]["pos"] = [loc["x"], loc["y"], loc["z"]]
                            active_cars[aid]["vel"] = [vel["x"], vel["y"], vel["z"]] if vel else [0.0, 0.0, 0.0]
                            if rot and "w" in rot:
                                active_cars[aid]["rot"] = list(_quat_to_euler(rot["x"], rot["y"], rot["z"], rot["w"]))
                            ang = rb.get("angular_velocity")
                            active_cars[aid]["ang_vel"] = [ang["x"], ang["y"], ang["z"]] if ang else [0.0, 0.0, 0.0]
                            active_cars[aid]["time"] = float(f.get("time", 0.0))

                elif atype == "ball":
                    if "RigidBody" in attr:
                        rb = attr["RigidBody"]
                        loc = rb.get("location")
                        vel = rb.get("linear_velocity")
                        if loc:
                            active_ball["pos"] = [loc["x"], loc["y"], loc["z"]]
                            active_ball["vel"] = [vel["x"], vel["y"], vel["z"]] if vel else [0.0, 0.0, 0.0]

                elif atype == "boost" or "CarComponent" in obj_name:
                    if "ActiveActor" in attr:
                        tgt = attr["ActiveActor"].get("actor")
                        if tgt and tgt != -1:
                            boost_to_car[aid] = tgt
                    if "ReplicatedBoost" in attr:
                        b_amt = attr["ReplicatedBoost"].get("boost_amount", 85)
                        cid = boost_to_car.get(aid)
                        if cid and cid in active_cars:
                            active_cars[cid]["boost"] = round((b_amt / 255.0) * 100.0, 1)

            # Keep every frame_stride-th network frame when ball and at least 1 car exist
            if (f_idx % self.frame_stride == 0) and active_ball_id is not None and len(active_cars) >= 1:
                sorted_cars = sorted(active_cars.items(), key=lambda item: car_teams.get(item[0], 0))
                c0 = sorted_cars[0][1]
                if len(sorted_cars) >= 2:
                    c1 = sorted_cars[1][1]
                else:
                    c1 = {
                        "pos": [-c0["pos"][0], -c0["pos"][1], c0["pos"][2]],
                        "vel": [-c0["vel"][0], -c0["vel"][1], c0["vel"][2]],
                        "rot": [c0["rot"][0], c0["rot"][1] + math.pi, c0["rot"][2]],
                        "ang_vel": [-c0["ang_vel"][0], -c0["ang_vel"][1], c0["ang_vel"][2]],
                        "time": c0["time"],
                        "boost": c0["boost"]
                    }

                extracted_b_pos.append(active_ball["pos"])
                extracted_b_vel.append(active_ball["vel"])
                extracted_c_pos.append([c0["pos"], c1["pos"]])
                extracted_c_vel.append([c0["vel"], c1["vel"]])
                extracted_c_rot.append([c0["rot"], c1["rot"]])
                extracted_c_bst.append([c0["boost"], c1["boost"]])
                extracted_frame_time.append(float(f.get("time", 0.0)))
                extracted_car_time.append([c0["time"], c1["time"]])
                extracted_c_angv.append([c0["ang_vel"], c1["ang_vel"]])

        if not extracted_b_pos:
            return None

        return {
            "ball_pos": np.array(extracted_b_pos, dtype=np.float32),
            "ball_vel": np.array(extracted_b_vel, dtype=np.float32),
            "car_pos": np.array(extracted_c_pos, dtype=np.float32),
            "car_vel": np.array(extracted_c_vel, dtype=np.float32),
            "car_rot": np.array(extracted_c_rot, dtype=np.float32),
            "car_boost": np.array(extracted_c_bst, dtype=np.float32),
            "frame_time": np.array(extracted_frame_time, dtype=np.float64),
            "car_time": np.array(extracted_car_time, dtype=np.float64),
            "car_ang_vel": np.array(extracted_c_angv, dtype=np.float32),
            "segment_id": np.zeros(len(extracted_frame_time), dtype=np.int32)
        }
