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
    """Converts a quaternion (x, y, z, w) to Euler angles (pitch, yaw, roll) in radians."""
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
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


class ReplayParser:
    """
    Parses Rocket League replay states and maintains a fast, memory-mapped replay buffer.
    """
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

            extracted_b_pos = []
            extracted_b_vel = []
            extracted_c_pos = []
            extracted_c_vel = []
            extracted_c_rot = []
            extracted_c_bst = []

            processed_count = 0
            rejected_files = []
            for fpath in files:
                try:
                    frames = self._parse_file(fpath)
                    if frames and len(frames["ball_pos"]) > 0:
                        extracted_b_pos.append(frames["ball_pos"])
                        extracted_b_vel.append(frames["ball_vel"])
                        extracted_c_pos.append(frames["car_pos"])
                        extracted_c_vel.append(frames["car_vel"])
                        extracted_c_rot.append(frames["car_rot"])
                        extracted_c_bst.append(frames["car_boost"])
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

            if not extracted_b_pos:
                return 0, 0

            new_b_pos = np.vstack(extracted_b_pos)
            new_b_vel = np.vstack(extracted_b_vel)
            new_c_pos = np.vstack(extracted_c_pos)
            new_c_vel = np.vstack(extracted_c_vel)
            new_c_rot = np.vstack(extracted_c_rot)
            new_c_bst = np.vstack(extracted_c_bst)

            if self.states_buffer is not None:
                self.states_buffer["ball_pos"] = np.vstack([self.states_buffer["ball_pos"], new_b_pos])
                self.states_buffer["ball_vel"] = np.vstack([self.states_buffer["ball_vel"], new_b_vel])
                self.states_buffer["car_pos"] = np.vstack([self.states_buffer["car_pos"], new_c_pos])
                self.states_buffer["car_vel"] = np.vstack([self.states_buffer["car_vel"], new_c_vel])
                self.states_buffer["car_rot"] = np.vstack([self.states_buffer["car_rot"], new_c_rot])
                self.states_buffer["car_boost"] = np.vstack([self.states_buffer["car_boost"], new_c_bst])
            else:
                self.states_buffer = {
                    "ball_pos": new_b_pos,
                    "ball_vel": new_b_vel,
                    "car_pos": new_c_pos,
                    "car_vel": new_c_vel,
                    "car_rot": new_c_rot,
                    "car_boost": new_c_bst
                }

            if len(self.states_buffer["ball_pos"]) > 100000:
                for k in self.states_buffer:
                    self.states_buffer[k] = self.states_buffer[k][-100000:]

            self.last_ingest_report["total_frames"] = len(new_b_pos)
            self.save_pool()
            return processed_count, len(new_b_pos)
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

        extracted_b_pos = []
        extracted_b_vel = []
        extracted_c_pos = []
        extracted_c_vel = []
        extracted_c_rot = []
        extracted_c_bst = []

        processed_count = 0
        total_files = len(files)
        rejected_files = []

        for i, file_path in enumerate(files):
            try:
                frames = self._parse_file(file_path)
                if frames and len(frames["ball_pos"]) > 0:
                    extracted_b_pos.append(frames["ball_pos"])
                    extracted_b_vel.append(frames["ball_vel"])
                    extracted_c_pos.append(frames["car_pos"])
                    extracted_c_vel.append(frames["car_vel"])
                    extracted_c_rot.append(frames["car_rot"])
                    extracted_c_bst.append(frames["car_boost"])
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

        if not extracted_b_pos:
            return {"parsed_files": 0, "total_frames": 0, "elapsed_seconds": 0.0}

        new_b_pos = np.vstack(extracted_b_pos)
        new_b_vel = np.vstack(extracted_b_vel)
        new_c_pos = np.vstack(extracted_c_pos)
        new_c_vel = np.vstack(extracted_c_vel)
        new_c_rot = np.vstack(extracted_c_rot)
        new_c_bst = np.vstack(extracted_c_bst)

        if self.states_buffer is not None:
            self.states_buffer["ball_pos"] = np.vstack([self.states_buffer["ball_pos"], new_b_pos])
            self.states_buffer["ball_vel"] = np.vstack([self.states_buffer["ball_vel"], new_b_vel])
            self.states_buffer["car_pos"] = np.vstack([self.states_buffer["car_pos"], new_c_pos])
            self.states_buffer["car_vel"] = np.vstack([self.states_buffer["car_vel"], new_c_vel])
            self.states_buffer["car_rot"] = np.vstack([self.states_buffer["car_rot"], new_c_rot])
            self.states_buffer["car_boost"] = np.vstack([self.states_buffer["car_boost"], new_c_bst])
        else:
            self.states_buffer = {
                "ball_pos": new_b_pos,
                "ball_vel": new_b_vel,
                "car_pos": new_c_pos,
                "car_vel": new_c_vel,
                "car_rot": new_c_rot,
                "car_boost": new_c_bst
            }

        # Keep max 100,000 frames to ensure fast training and memory efficiency
        if len(self.states_buffer["ball_pos"]) > 100000:
            for k in self.states_buffer:
                self.states_buffer[k] = self.states_buffer[k][-100000:]

        self.last_ingest_report["total_frames"] = len(new_b_pos)
        self.save_pool()
        return {"parsed_files": processed_count, "total_frames": len(new_b_pos), "elapsed_seconds": round(time.time() - _t0, 2)}

    def _parse_file(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Parses an individual replay file or pre-formatted numpy/json dataset."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".npz":
            data = np.load(file_path)
            return {
                "ball_pos": data["ball_pos"],
                "ball_vel": data["ball_vel"],
                "car_pos": data["car_pos"],
                "car_vel": data["car_vel"],
                "car_rot": data["car_rot"],
                "car_boost": data["car_boost"]
            }

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

            # Sample every 10 frames (~3-6 Hz) when ball and at least 1 car exist
            if (f_idx % 10 == 0) and active_ball_id is not None and len(active_cars) >= 1:
                sorted_cars = sorted(active_cars.items(), key=lambda item: car_teams.get(item[0], 0))
                c0 = sorted_cars[0][1]
                if len(sorted_cars) >= 2:
                    c1 = sorted_cars[1][1]
                else:
                    c1 = {
                        "pos": [-c0["pos"][0], -c0["pos"][1], c0["pos"][2]],
                        "vel": [-c0["vel"][0], -c0["vel"][1], c0["vel"][2]],
                        "rot": [c0["rot"][0], c0["rot"][1] + math.pi, c0["rot"][2]],
                        "boost": c0["boost"]
                    }

                extracted_b_pos.append(active_ball["pos"])
                extracted_b_vel.append(active_ball["vel"])
                extracted_c_pos.append([c0["pos"], c1["pos"]])
                extracted_c_vel.append([c0["vel"], c1["vel"]])
                extracted_c_rot.append([c0["rot"], c1["rot"]])
                extracted_c_bst.append([c0["boost"], c1["boost"]])

        if not extracted_b_pos:
            return None

        return {
            "ball_pos": np.array(extracted_b_pos, dtype=np.float32),
            "ball_vel": np.array(extracted_b_vel, dtype=np.float32),
            "car_pos": np.array(extracted_c_pos, dtype=np.float32),
            "car_vel": np.array(extracted_c_vel, dtype=np.float32),
            "car_rot": np.array(extracted_c_rot, dtype=np.float32),
            "car_boost": np.array(extracted_c_bst, dtype=np.float32)
        }
