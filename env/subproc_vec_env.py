"""
Subprocess-Parallel Vectorized Rocket League Environment.

The pure-Python reward and observation code that dominates rollout cost cannot be
parallelized with threads under CPython's GIL (measured: 12 worker threads run the
same rollout 2.2x SLOWER than one). This module distributes environments across
worker *processes* instead, which scales near-linearly with physical cores.

Observations, rewards, dones, actions and per-step touch counts live in shared
memory so no bulk array is ever pickled. Only terminal episode summaries (typically
zero to two environments per step) travel over the control pipes.
"""

from __future__ import annotations
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import List, Tuple, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Shared memory plumbing
# ---------------------------------------------------------------------------

class _SharedBlock:
    """Owns (or attaches to) one named shared-memory block viewed as a numpy array."""

    def __init__(self, name: Optional[str], shape: Tuple[int, ...], dtype, create: bool):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * self.dtype.itemsize
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self.array = np.ndarray(shape, dtype=self.dtype, buffer=self.shm.buf)
        self._owner = create

    @property
    def name(self) -> str:
        return self.shm.name

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass
        if self._owner:
            try:
                self.shm.unlink()
            except Exception:
                pass


def _spec(block: _SharedBlock) -> Tuple[str, Tuple[int, ...], str]:
    return (block.name, block.shape, block.dtype.str)


def _attach(spec) -> _SharedBlock:
    name, shape, dtype = spec
    return _SharedBlock(name, tuple(shape), np.dtype(dtype), create=False)


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker_main(remote, env_kwargs, specs, start, end, project_root):
    """
    Runs `end - start` RocketLeagueEnv instances and writes results directly into
    the shared buffers owned by the parent. Never returns until told to close.
    """
    import sys
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Each worker is single-threaded on purpose: the policy forward passes here are
    # tiny (batch 1-32 MLPs) and intra-op threading costs more than it saves, while
    # oversubscribing N workers x M threads would thrash the machine.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    from env.rocket_env import VectorizedRocketEnv

    n_local = end - start
    obs_b = _attach(specs["obs"])
    rew_b = _attach(specs["rew"])
    done_b = _attach(specs["done"])
    act_b = _attach(specs["act"])
    touch_b = _attach(specs["touch"])
    term_obs_b = _attach(specs["term_obs"])
    trunc_b = _attach(specs["trunc"])

    obs_v = obs_b.array[start:end]
    rew_v = rew_b.array[start:end]
    done_v = done_b.array[start:end]
    act_v = act_b.array[start:end]
    touch_v = touch_b.array[start:end]
    term_obs_v = term_obs_b.array[start:end]
    trunc_v = trunc_b.array[start:end]

    kwargs = dict(env_kwargs)
    kwargs["num_envs"] = n_local
    kwargs["num_workers"] = 1
    vec = VectorizedRocketEnv(**kwargs)

    # Point the child vec-env's own rollout buffers straight at shared memory so
    # env.step writes its results where the parent will read them, with no copy.
    vec._obs_buffer = obs_v
    vec._rew_buffer = rew_v
    vec._done_buffer = done_v
    vec._term_obs_buffer = term_obs_v
    vec._trunc_buffer = trunc_v

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == "step":
                _, _, _, infos = vec.step(act_v)
                # Terminal summaries only: everything else is already in shared memory.
                terminals = []
                for k, info in enumerate(infos):
                    touch_v[k] = int(info.get("step_touches", 0))
                    if info.get("done", False):
                        terminals.append((
                            start + k,
                            int(info.get("step", 0)),
                            bool(info.get("is_goal", False)),
                            list(info.get("episode_rewards", [])),
                            list(info.get("episode_touches", [])),
                            list(info.get("episode_goals", [0, 0])),
                        ))
                remote.send(terminals)

            elif cmd == "reset":
                vec.reset()  # writes into obs_v via the aliased buffer
                touch_v[:] = 0
                remote.send(None)

            elif cmd == "set_stratified_opponents":
                vec.set_stratified_opponents(data[start:end])
                remote.send(None)

            elif cmd == "update_reward_weights":
                vec.update_reward_weights(data)
                remote.send(None)

            elif cmd == "update_scenarios":
                vec.update_scenarios(data)
                remote.send(None)

            elif cmd == "update_baseline_opponent":
                vec.update_baseline_opponent(data[0], data[1])
                remote.send(None)

            elif cmd == "get_learner_mask":
                remote.send(vec.get_learner_mask())

            elif cmd == "close":
                remote.send(None)
                break

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        for b in (obs_b, rew_b, done_b, act_b, touch_b):
            b.close()


# ---------------------------------------------------------------------------
# Parent-side vector env
# ---------------------------------------------------------------------------

class SubprocVectorizedRocketEnv:
    """
    Drop-in replacement for VectorizedRocketEnv that runs environments across
    worker processes. Exposes the same surface the PPO trainer uses.

    The `infos` list returned by `step` carries the keys the trainer consumes
    (`step_touches`, `done`, `is_goal`, `step`, `episode_rewards`,
    `episode_touches`, `episode_goals`); the richer per-step diagnostic fields of
    the single-process env are not shipped across the process boundary.
    """

    def __init__(
        self,
        num_envs: int = 64,
        game_mode: str = "1v1",
        tick_skip: int = 8,
        max_episode_steps: int = 1500,
        reward_weights: Optional[Dict[str, float]] = None,
        continuous_actions: bool = True,
        self_play: bool = True,
        baseline_opponent_ratio: float = 0.25,
        baseline_opponent_type: str = "heuristic",
        num_workers: Optional[int] = None,
    ):
        self.num_envs = num_envs
        self.game_mode = game_mode
        self.tick_skip = tick_skip
        self.max_episode_steps = max_episode_steps
        self.continuous_actions = continuous_actions
        self.baseline_opponent_ratio = max(0.0, min(1.0, baseline_opponent_ratio))
        self.baseline_opponent_type = baseline_opponent_type

        if num_workers is None:
            num_workers = max(1, min(8, (os.cpu_count() or 4) // 2))
        self.num_workers = max(1, min(int(num_workers), num_envs))

        # Probe dimensions in-process (cheap: one arena, torn down immediately).
        from env.rocket_env import RocketLeagueEnv
        probe = RocketLeagueEnv(
            game_mode=game_mode,
            tick_skip=tick_skip,
            max_episode_steps=max_episode_steps,
            reward_weights=reward_weights,
            continuous_actions=continuous_actions,
            self_play=self_play,
        )
        self.num_players_per_env = probe.num_players
        self.obs_dim = probe.obs_dim
        self.act_dim = probe.act_dim
        del probe

        P = self.num_players_per_env
        # Discrete policies hand down one action index per player, not a vector.
        act_shape = (num_envs, P, self.act_dim) if continuous_actions else (num_envs, P)
        self._obs_b = _SharedBlock(None, (num_envs, P, self.obs_dim), np.float32, create=True)
        self._rew_b = _SharedBlock(None, (num_envs, P), np.float32, create=True)
        self._done_b = _SharedBlock(None, (num_envs, P), np.bool_, create=True)
        self._act_b = _SharedBlock(None, act_shape, np.float32, create=True)
        self._touch_b = _SharedBlock(None, (num_envs,), np.int32, create=True)
        # Time-limit bootstrapping channel. The terminal observation is bulk data, so it
        # rides shared memory like the rest rather than being pickled over the pipes.
        self._term_obs_b = _SharedBlock(None, (num_envs, P, self.obs_dim), np.float32, create=True)
        self._trunc_b = _SharedBlock(None, (num_envs, P), np.bool_, create=True)

        self._obs = self._obs_b.array
        self._rew = self._rew_b.array
        self._done = self._done_b.array
        self._act = self._act_b.array
        self._touch = self._touch_b.array
        self._term_obs = self._term_obs_b.array
        self._trunc = self._trunc_b.array

        specs = {
            "obs": _spec(self._obs_b),
            "rew": _spec(self._rew_b),
            "done": _spec(self._done_b),
            "act": _spec(self._act_b),
            "touch": _spec(self._touch_b),
            "term_obs": _spec(self._term_obs_b),
            "trunc": _spec(self._trunc_b),
        }

        env_kwargs = dict(
            game_mode=game_mode,
            tick_skip=tick_skip,
            max_episode_steps=max_episode_steps,
            reward_weights=reward_weights,
            continuous_actions=continuous_actions,
            self_play=self_play,
            baseline_opponent_ratio=self.baseline_opponent_ratio,
            baseline_opponent_type=baseline_opponent_type,
        )

        # Contiguous, balanced env ranges: worker w owns [starts[w], starts[w+1]).
        base, extra = divmod(num_envs, self.num_workers)
        self.ranges: List[Tuple[int, int]] = []
        cur = 0
        for w in range(self.num_workers):
            n = base + (1 if w < extra else 0)
            self.ranges.append((cur, cur + n))
            cur += n

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ctx = mp.get_context("spawn")
        self._remotes = []
        self._procs = []
        for (start, end) in self.ranges:
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_worker_main,
                args=(child, env_kwargs, specs, start, end, project_root),
                daemon=True,
            )
            p.start()
            child.close()
            self._remotes.append(parent)
            self._procs.append(p)

        # Parent-side mirror of which envs face a scripted/league opponent, so
        # get_learner_mask() needs no round trip during the rollout.
        num_baseline = int(round(num_envs * self.baseline_opponent_ratio)) if game_mode == "1v1" else 0
        self._is_baseline = np.array(
            [i >= num_envs - num_baseline for i in range(num_envs)], dtype=bool
        )
        self._closed = False
        print(f"[SubprocVecEnv] {num_envs} envs across {self.num_workers} worker processes "
              f"({[e - s for s, e in self.ranges]} envs each)")

    # -- collective helpers -------------------------------------------------

    def _broadcast(self, cmd: str, data: Any = None):
        for r in self._remotes:
            r.send((cmd, data))
        return [r.recv() for r in self._remotes]

    # -- gym-ish surface ----------------------------------------------------

    def reset(self) -> np.ndarray:
        self._broadcast("reset")
        return self._obs.copy()

    def step(self, actions: np.ndarray):
        # Publish actions through shared memory, then fan out and join.
        np.copyto(self._act, np.asarray(actions, dtype=np.float32).reshape(self._act.shape))
        for r in self._remotes:
            r.send(("step", None))

        infos: List[Dict[str, Any]] = [
            {"step_touches": 0, "done": False, "is_goal": False, "step": 0}
            for _ in range(self.num_envs)
        ]
        for r in self._remotes:
            for (idx, step, is_goal, ep_rew, ep_touch, ep_goals) in r.recv():
                infos[idx] = {
                    "step_touches": 0,
                    "done": True,
                    "is_goal": is_goal,
                    "step": step,
                    "episode_rewards": ep_rew,
                    "episode_touches": ep_touch,
                    "episode_goals": ep_goals,
                }

        touches = self._touch
        for i in range(self.num_envs):
            infos[i]["step_touches"] = int(touches[i])

        return self._obs, self._rew, self._done, infos

    def get_truncation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Time-limit bootstrapping channel for the most recent step, flattened to
        (num_envs * P,) and (num_envs * P, obs_dim). See VectorizedRocketEnv.get_truncation.
        """
        return self._trunc.reshape(-1), self._term_obs.reshape(-1, self.obs_dim)

    def get_learner_mask(self) -> np.ndarray:
        P = self.num_players_per_env
        mask = np.ones((self.num_envs, P), dtype=bool)
        # Blue (slot 0) always learns; in a baseline/league match Orange does not.
        mask[self._is_baseline, 1:] = False
        return mask.reshape(-1)

    # -- live reconfiguration ----------------------------------------------

    def set_stratified_opponents(self, opponent_assignments: List[Optional[str]]):
        assignments = list(opponent_assignments[: self.num_envs])
        if len(assignments) < self.num_envs:
            assignments += [None] * (self.num_envs - len(assignments))
        self._broadcast("set_stratified_opponents", assignments)
        self._is_baseline = np.array([a is not None for a in assignments], dtype=bool)

    def update_reward_weights(self, weights: Dict[str, float]):
        self._broadcast("update_reward_weights", weights)

    def update_scenarios(self, config_dict: Dict[str, Any]):
        self._broadcast("update_scenarios", config_dict)

    def update_baseline_opponent(self, ratio: float, opponent_type: Optional[str] = None):
        self.baseline_opponent_ratio = max(0.0, min(1.0, ratio))
        if opponent_type is not None:
            self.baseline_opponent_type = opponent_type
        self._broadcast("update_baseline_opponent", (self.baseline_opponent_ratio, self.baseline_opponent_type))
        num_baseline = int(round(self.num_envs * self.baseline_opponent_ratio)) if self.game_mode == "1v1" else 0
        self._is_baseline = np.array(
            [i >= self.num_envs - num_baseline for i in range(self.num_envs)], dtype=bool
        )

    def update_baseline_ratio(self, ratio: float):
        self.update_baseline_opponent(ratio, self.baseline_opponent_type)

    # -- teardown -----------------------------------------------------------

    def close(self):
        if self._closed:
            return
        self._closed = True
        for r in self._remotes:
            try:
                r.send(("close", None))
                r.recv()
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        for b in (self._obs_b, self._rew_b, self._done_b, self._act_b, self._touch_b,
                  self._term_obs_b, self._trunc_b):
            b.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
