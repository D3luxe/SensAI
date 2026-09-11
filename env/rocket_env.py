"""
Rocket League Gym Environment & Vectorized Parallel Environment Engine.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from env.physics_engine import RocketSimArena, CarState
from env.rewards import RewardManager
from env.observations import DefaultObservationBuilder
from env.actions import ContinuousActionParser, DiscreteActionParser
from env.baseline_agent import BaselineChaser, BaseOpponent, NectoNextoOpponentBot, CheckpointOpponentBot, create_opponent_bot


SCENARIO_TIMEOUTS: Dict[str, int] = {
    "kickoff":       225,   # 15s  (backed by 75-step stagnation check)
    "aerial":        120,   # 8s   (aerial contact window)
    "goalie_save":    90,   # 6s   (reaction & save window)
    "wall_play":     135,   # 9s   (wall climb & infield transition)
    "wall_rebound":  120,   # 8s   (backboard read & rebound)
    "turnaround":    180,   # 12s  (recovery & 180 cut)
    "dribble_flick": 375,   # 25s  (extended carry & flick)
    "replay":        450,   # 30s  (match state rollout)
    "custom":        300,   # 20s
}


class RocketLeagueEnv:
    """
    Standard single-instance Gymnasium-compatible Rocket League match environment.
    Supports symmetric self-play and asymmetric baseline challenger matches.
    """
    def __init__(
        self,
        game_mode: str = "1v1",
        tick_skip: int = 8,
        max_episode_steps: int = 1500,
        reward_weights: Optional[Dict[str, float]] = None,
        continuous_actions: bool = True,
        self_play: bool = True,
        is_baseline_env: bool = False,
        baseline_opponent_type: str = "heuristic"
    ):
        self.game_mode = game_mode
        self.num_players = 2 if game_mode == "1v1" else (4 if game_mode == "2v2" else 6)
        self.tick_skip = tick_skip
        self.max_episode_steps = max_episode_steps
        self.self_play = self_play
        self.is_baseline_env = is_baseline_env
        self.continuous_actions = continuous_actions
        self.baseline_opponent_type = baseline_opponent_type
        self.baseline_bot = create_opponent_bot(baseline_opponent_type, continuous_actions=continuous_actions) if is_baseline_env else None

        self.arena = RocketSimArena(num_players=self.num_players, game_mode=game_mode)
        self.obs_builder = DefaultObservationBuilder(symmetric=True)
        self.reward_manager = RewardManager(reward_weights=reward_weights)
        self.action_parser = ContinuousActionParser() if continuous_actions else DiscreteActionParser()

        self.obs_dim = self.obs_builder.obs_dim
        self.act_dim = self.action_parser.action_dim

        self.current_step = 0
        self.episode_rewards = [0.0] * self.num_players
        self.episode_touches = [0] * self.num_players
        self.episode_goals = [0] * 2
        self.current_scenario = "kickoff"
        self.scenario_timeout = self.max_episode_steps
        self._save_defend_sign = 0.0

    def update_reward_weights(self, weights: Dict[str, float]):
        self.reward_manager.update_weights(weights)

    def update_scenarios(self, config_dict: Dict[str, Any]):
        self.arena.set_scenario_weights(config_dict)

    def reset(self, random_kickoff: bool = True) -> np.ndarray:
        self.arena.reset(random_kickoff=random_kickoff)
        self.reward_manager.reset(self.arena)
        self.current_step = 0
        self.episode_rewards = [0.0] * self.num_players
        self.episode_touches = [0] * self.num_players
        self.episode_goals = [0] * 2

        self.current_scenario = self.arena.current_scenario
        self.scenario_timeout = min(self.max_episode_steps, SCENARIO_TIMEOUTS.get(self.current_scenario, self.max_episode_steps))
        if self.current_scenario == "goalie_save":
            bvy = float(self.arena.ball.vel[1])
            self._save_defend_sign = -1.0 if bvy < 0.0 else 1.0
        else:
            self._save_defend_sign = 0.0

        obs = []
        for car in self.arena.cars:
            obs.append(self.obs_builder.build_obs(car, self.arena))
        return np.array(obs, dtype=np.float32)

    def step(
        self,
        raw_actions: np.ndarray,
        out_obs: Optional[np.ndarray] = None,
        out_rews: Optional[np.ndarray] = None,
        include_breakdown: bool = False,
        opponent_action: Optional[np.ndarray] = None,
        out_term_obs: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Step simulation by tick_skip sub-ticks.
        raw_actions: shape (num_players, action_dim)
        Returns: (obs, rewards, dones, info)
        """
        self.current_step += 1
        actions_to_parse = raw_actions.copy()

        # If baseline environment in 1v1, override Orange bot action with BaselineChaser / Opponent Bot
        if self.is_baseline_env and len(self.arena.cars) > 1:
            if opponent_action is not None:
                actions_to_parse[1] = opponent_action
            elif self.baseline_bot is not None:
                actions_to_parse[1] = self.baseline_bot.get_action(self.arena.cars[1], self.arena)

        parsed_actions = self.action_parser.parse_actions(actions_to_parse)

        is_goal = False
        scoring_team = None

        # Construct bot_mask: True = external Necto/Nexto agent, bypass jump sequencer
        bot_mask = None
        if self.is_baseline_env and self.baseline_bot is not None and isinstance(self.baseline_bot, NectoNextoOpponentBot):
            bot_mask = [False] * len(self.arena.cars)
            bot_mask[1] = True  # Orange slot (index 1) is always the external bot in 1v1

        # Execute physics sub-ticks (full tick_skip interval)
        is_goal, scoring_team = self.arena.step(parsed_actions, dt=float(self.tick_skip) / 120.0, bot_mask=bot_mask)

        # Calculate rewards and observations
        if out_obs is None:
            out_obs = np.empty((self.num_players, self.obs_dim), dtype=np.float32)
        if out_rews is None:
            out_rews = np.empty(self.num_players, dtype=np.float32)

        info_rewards = {}
        prev_touches_sum = sum(self.episode_touches)

        for i, car in enumerate(self.arena.cars):
            # Only generate detailed reward breakdown dictionary if explicitly requested
            r, r_dict = self.reward_manager.get_reward(car, self.arena, parsed_actions[i], is_goal, scoring_team, include_breakdown=(include_breakdown and i == 0))
            out_rews[i] = r
            self.episode_rewards[i] += r
            self.episode_touches[i] = car.ball_touches
            self.obs_builder.build_obs(car, self.arena, out=out_obs[i])
            if include_breakdown and i == 0:
                info_rewards = r_dict

        if is_goal and scoring_team is not None:
            self.episode_goals[scoring_team] += 1

        step_touches = max(0, sum(self.episode_touches) - prev_touches_sum)

        # RLGym Kickoff Stagnation Rule: If ball is untouched on kickoff after 75 steps (5.0s), terminate episode!
        bx, by, bz = self.arena.ball.pos[0], self.arena.ball.pos[1], self.arena.ball.pos[2]
        bvx, bvy = self.arena.ball.vel[0], self.arena.ball.vel[1]
        total_episode_touches = sum(self.episode_touches)

        is_kickoff_stalled = (
            self.current_scenario == "kickoff"
            and self.current_step > 75
            and abs(bx) < 20.0
            and abs(by) < 20.0
            and (abs(bvx) + abs(bvy)) < 80.0
        )

        # 1. Hard scenario timeout
        is_scenario_timeout = (self.current_step >= self.scenario_timeout)

        # 2. Aerial resolved: Whiffed aerial, ball dropped to floor (Z < 250) without a single touch
        is_aerial_resolved = (
            self.current_scenario == "aerial"
            and self.current_step > 20
            and total_episode_touches == 0
            and bz < 250.0
        )

        # 3. Goalie save resolved: Ball moving away from defending net after shot flight window (> 30 steps)
        is_save_resolved = False
        if self.current_scenario == "goalie_save" and self.current_step > 30:
            ball_moving_away = (bvy * self._save_defend_sign) < -300.0
            is_save_resolved = ball_moving_away and (total_episode_touches > 0 or abs(by) < 3500.0)

        # 4. Wall play / rebound resolved: Whiffed off wall onto infield turf without a touch
        is_wall_resolved = (
            self.current_scenario in ("wall_play", "wall_rebound")
            and self.current_step > 30
            and total_episode_touches == 0
            and bz < 200.0
            and abs(bx) < 2500.0
        )

        done = (
            is_scenario_timeout
            or is_goal
            or is_kickoff_stalled
            or is_aerial_resolved
            or is_save_resolved
            or is_wall_resolved
        )
        dones = np.array([done] * self.num_players, dtype=bool)

        # A goal is the only exit that carries a genuine terminal value of zero. Every other
        # exit here is a TRUNCATION: the clock ran out, or a scenario was declared resolved.
        # Treating those as terminations teaches the critic that the future is worthless
        # whenever a timer the car cannot observe happens to expire -- and the observation
        # vector carries no clock at all, so that signal is pure noise to it. The trainer
        # bootstraps V(s) across these instead, which is why the pre-reset observation has
        # to survive the auto-reset below.
        is_truncated = bool(done and not is_goal)

        info = {
            "is_goal": is_goal,
            "truncated": is_truncated,
            "scoring_team": scoring_team,
            "step": self.current_step,
            "step_touches": step_touches,
            "episode_rewards": list(self.episode_rewards) if done else self.episode_rewards,
            "episode_touches": list(self.episode_touches) if done else self.episode_touches,
            "episode_goals": list(self.episode_goals) if done else self.episode_goals,
            "reward_breakdown": info_rewards,
            "is_baseline_env": self.is_baseline_env,
            "done": done,
            "scenario": self.current_scenario,
        }

        # Auto-reset on goal/max steps/stalled kickoff. The observation the agent actually
        # arrived at is overwritten by the reset, so capture it first: bootstrapping a
        # truncation against the post-reset observation would value a finished play by the
        # fresh scenario that replaced it.
        if done:
            if out_term_obs is not None:
                out_term_obs[:] = out_obs
            else:
                info["terminal_observation"] = out_obs.copy()
            reset_obs = self.reset()
            out_obs[:] = reset_obs[:]

        return out_obs, out_rews, dones, info


class VectorizedRocketEnv:
    """
    Vectorized parallel environment container running multiple RocketLeagueEnv instances simultaneously.
    Provides fast, pre-allocated zero-copy observation and reward tensor aggregation with
    persistent multi-threaded chunked parallel environment stepping.
    """
    def __init__(
        self,
        num_envs: int = 16,
        game_mode: str = "1v1",
        tick_skip: int = 8,
        max_episode_steps: int = 1500,
        reward_weights: Optional[Dict[str, float]] = None,
        continuous_actions: bool = True,
        self_play: bool = True,
        baseline_opponent_ratio: float = 0.25,
        baseline_opponent_type: str = "heuristic",
        num_workers: Optional[int] = None
    ):
        self.num_envs = num_envs
        self.game_mode = game_mode
        self.tick_skip = tick_skip
        self.max_episode_steps = max_episode_steps
        self.reward_weights = reward_weights
        self.continuous_actions = continuous_actions
        self.self_play = self_play
        self.baseline_opponent_ratio = max(0.0, min(1.0, baseline_opponent_ratio))
        self.baseline_opponent_type = baseline_opponent_type

        num_baseline = int(round(num_envs * self.baseline_opponent_ratio)) if game_mode == "1v1" else 0
        self.envs = [
            RocketLeagueEnv(
                game_mode=game_mode,
                tick_skip=tick_skip,
                max_episode_steps=max_episode_steps,
                reward_weights=reward_weights,
                continuous_actions=continuous_actions,
                self_play=self_play,
                is_baseline_env=(i >= num_envs - num_baseline),
                baseline_opponent_type=baseline_opponent_type
            )
            for i in range(num_envs)
        ]
        self.num_players_per_env = self.envs[0].num_players
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].act_dim

        # Pre-allocated zero-copy continuous rollout buffers
        self._obs_buffer = np.zeros((num_envs, self.num_players_per_env, self.obs_dim), dtype=np.float32)
        self._rew_buffer = np.zeros((num_envs, self.num_players_per_env), dtype=np.float32)
        self._done_buffer = np.zeros((num_envs, self.num_players_per_env), dtype=bool)
        # Time-limit bootstrapping channel: the observation each environment actually reached
        # before its auto-reset, plus a flag marking which of those exits were truncations
        # rather than goals. Only rows flagged in _trunc_buffer hold meaningful data.
        self._term_obs_buffer = np.zeros((num_envs, self.num_players_per_env, self.obs_dim), dtype=np.float32)
        self._trunc_buffer = np.zeros((num_envs, self.num_players_per_env), dtype=bool)

        # Opponent batched inference engine state
        self._opponent_prev_actions = np.zeros((num_envs, 8), dtype=np.float32)
        self._batched_opp_actions: List[Optional[np.ndarray]] = [None] * num_envs
        self._nexto_envs: List[int] = []
        self._necto_envs: List[int] = []
        self._checkpoint_groups: Dict[str, List[int]] = {}
        self._heuristic_envs: List[int] = []
        self._build_opponent_groups()

        # Environment stepping is single-threaded by design. The reward and observation
        # code is pure Python, so worker *threads* only add GIL contention: measured at
        # 64 envs, 12 threads ran the same rollout 2.2x slower than one. Process-level
        # parallelism lives in env/subproc_vec_env.py instead; `num_workers` is accepted
        # and ignored here so both vector envs share a constructor signature.
        self.num_workers = 1
        self.chunks = [(0, num_envs)]
        self.worker_threads = []

    def _build_opponent_groups(self):
        """Pre-groups environment indices by opponent model type for zero-overhead batched execution."""
        self._nexto_envs = []
        self._necto_envs = []
        self._checkpoint_groups = {}
        self._heuristic_envs = []

        for i, env in enumerate(self.envs):
            if not env.is_baseline_env or env.baseline_bot is None or len(env.arena.cars) <= 1:
                continue
            bot = env.baseline_bot
            if isinstance(bot, NectoNextoOpponentBot):
                if bot.is_nexto:
                    self._nexto_envs.append(i)
                else:
                    self._necto_envs.append(i)
            elif isinstance(bot, CheckpointOpponentBot):
                path = getattr(bot, "model_path", "default")
                if path not in self._checkpoint_groups:
                    self._checkpoint_groups[path] = []
                self._checkpoint_groups[path].append(i)
            else:
                self._heuristic_envs.append(i)

    def _batch_evaluate_opponents(self):
        """Vectorized batched forward pass across all baseline/league opponent bots."""
        # 1. Nexto (batched)
        if self._nexto_envs:
            bot = self.envs[self._nexto_envs[0]].baseline_bot
            cars = [self.envs[i].arena.cars[1] for i in self._nexto_envs]
            arenas = [self.envs[i].arena for i in self._nexto_envs]
            pas = [self._opponent_prev_actions[i] for i in self._nexto_envs]
            acts = bot.batch_get_actions(cars, arenas, prev_actions=pas)
            for k, i in enumerate(self._nexto_envs):
                act = acts[k]
                self._batched_opp_actions[i] = act
                self._opponent_prev_actions[i] = act.copy()

        # 2. Necto (batched)
        if self._necto_envs:
            bot = self.envs[self._necto_envs[0]].baseline_bot
            cars = [self.envs[i].arena.cars[1] for i in self._necto_envs]
            arenas = [self.envs[i].arena for i in self._necto_envs]
            pas = [self._opponent_prev_actions[i] for i in self._necto_envs]
            acts = bot.batch_get_actions(cars, arenas, prev_actions=pas)
            for k, i in enumerate(self._necto_envs):
                act = acts[k]
                self._batched_opp_actions[i] = act
                self._opponent_prev_actions[i] = act.copy()

        # 3. Checkpoint bots (grouped by checkpoint model)
        for path, indices in self._checkpoint_groups.items():
            bot = self.envs[indices[0]].baseline_bot
            cars = [self.envs[i].arena.cars[1] for i in indices]
            arenas = [self.envs[i].arena for i in indices]
            acts = bot.batch_get_actions(cars, arenas)
            for k, i in enumerate(indices):
                self._batched_opp_actions[i] = acts[k]

        # 4. Heuristic Chaser
        for i in self._heuristic_envs:
            bot = self.envs[i].baseline_bot
            self._batched_opp_actions[i] = bot.get_action(self.envs[i].arena.cars[1], self.envs[i].arena)

    def update_baseline_ratio(self, ratio: float):
        """Dynamically reconfigures the number of environments running against the baseline opponent."""
        self.update_baseline_opponent(ratio=ratio, opponent_type=self.baseline_opponent_type)

    def update_baseline_opponent(self, ratio: float, opponent_type: Optional[str] = None):
        """Dynamically reconfigures baseline ratio and/or opponent bot type across all vectorized environments."""
        self.baseline_opponent_ratio = max(0.0, min(1.0, ratio))
        if opponent_type is not None:
            self.baseline_opponent_type = opponent_type

        num_baseline = int(round(self.num_envs * self.baseline_opponent_ratio)) if self.game_mode == "1v1" else 0
        for i, env in enumerate(self.envs):
            is_baseline = (i >= self.num_envs - num_baseline)
            env.is_baseline_env = is_baseline
            env.baseline_opponent_type = self.baseline_opponent_type
            env.baseline_bot = create_opponent_bot(self.baseline_opponent_type, continuous_actions=self.continuous_actions) if is_baseline else None
        self._build_opponent_groups()

    def set_stratified_opponents(self, opponent_assignments: List[Optional[str]]):
        """
        Dynamically configures per-environment opponent assignments for stratified league self-play.
        opponent_assignments: list of length num_envs containing opponent bot path/identifier or None for pure self-play.
        """
        for i, opp_spec in enumerate(opponent_assignments[:self.num_envs]):
            env = self.envs[i]
            if opp_spec is None:
                # Pure Self-Play: both Blue and Orange are policy learners
                env.is_baseline_env = False
                env.baseline_opponent_type = "self_play"
                env.baseline_bot = None
            else:
                # League / Baseline Opponent: Blue is learner, Orange is opponent bot
                env.is_baseline_env = True
                if getattr(env, "baseline_opponent_type", None) != opp_spec or env.baseline_bot is None:
                    env.baseline_opponent_type = opp_spec
                    env.baseline_bot = create_opponent_bot(opp_spec, continuous_actions=self.continuous_actions)
        self._build_opponent_groups()

    def get_truncation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Time-limit bootstrapping channel for the most recent step.

        Returns (truncated, terminal_obs), flattened to (num_envs * num_players,) and
        (num_envs * num_players, obs_dim). `truncated` marks the actors whose episode ended
        on a clock or a resolved-scenario check rather than on a goal; `terminal_obs` holds
        the observation they reached before the auto-reset. Rows not flagged are stale.
        """
        return (
            self._trunc_buffer.reshape(-1),
            self._term_obs_buffer.reshape(-1, self.obs_dim),
        )

    def get_learner_mask(self) -> np.ndarray:
        """
        Returns boolean mask of shape (num_envs * num_players_per_env,)
        True for policy learner actors, False for hardcoded baseline opponent actors.
        """
        mask = []
        for env in self.envs:
            if env.is_baseline_env:
                mask.extend([True, False])  # Blue is learner, Orange is baseline bot
            else:
                mask.extend([True] * self.num_players_per_env)  # All are learners in self-play
        return np.array(mask, dtype=bool)

    def update_reward_weights(self, weights: Dict[str, float]):
        for env in self.envs:
            env.reward_manager.update_weights(weights)

    def update_scenarios(self, config_dict: Dict[str, Any]):
        for env in self.envs:
            env.update_scenarios(config_dict)

    def reset(self) -> np.ndarray:
        self._opponent_prev_actions.fill(0.0)
        self._trunc_buffer.fill(False)
        for i, env in enumerate(self.envs):
            obs = env.reset()
            self._obs_buffer[i] = obs
        return self._obs_buffer.copy()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        actions shape: (num_envs, num_players, act_dim)
        Steps every environment in turn, updating the pre-allocated numpy buffers in place.
        """
        self._batch_evaluate_opponents()

        self._trunc_buffer.fill(False)
        all_infos = []
        for i, env in enumerate(self.envs):
            _, _, dones, info = env.step(
                actions[i],
                out_obs=self._obs_buffer[i],
                out_rews=self._rew_buffer[i],
                opponent_action=self._batched_opp_actions[i],
                out_term_obs=self._term_obs_buffer[i]
            )
            self._done_buffer[i] = dones
            if info.get("truncated", False):
                self._trunc_buffer[i] = dones
            if dones[0]:
                self._opponent_prev_actions[i].fill(0.0)
            all_infos.append(info)

        return (
            self._obs_buffer,
            self._rew_buffer,
            self._done_buffer,
            all_infos
        )

