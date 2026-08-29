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
from env.baseline_agent import BaselineChaser


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
        is_baseline_env: bool = False
    ):
        self.game_mode = game_mode
        self.num_players = 2 if game_mode == "1v1" else (4 if game_mode == "2v2" else 6)
        self.tick_skip = tick_skip
        self.max_episode_steps = max_episode_steps
        self.self_play = self_play
        self.is_baseline_env = is_baseline_env
        self.continuous_actions = continuous_actions
        self.baseline_bot = BaselineChaser(continuous_actions=continuous_actions) if is_baseline_env else None

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

        obs = []
        for car in self.arena.cars:
            obs.append(self.obs_builder.build_obs(car, self.arena))
        return np.array(obs, dtype=np.float32)

    def step(self, raw_actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Step simulation by tick_skip sub-ticks.
        raw_actions: shape (num_players, action_dim)
        Returns: (obs, rewards, dones, info)
        """
        self.current_step += 1
        actions_to_parse = raw_actions.copy()

        # If baseline environment in 1v1, override Orange bot action with BaselineChaser
        if self.is_baseline_env and self.baseline_bot is not None and len(self.arena.cars) > 1:
            actions_to_parse[1] = self.baseline_bot.get_action(self.arena.cars[1], self.arena.ball)

        parsed_actions = self.action_parser.parse_actions(actions_to_parse)

        is_goal = False
        scoring_team = None

        # Execute physics sub-ticks (full tick_skip interval)
        is_goal, scoring_team = self.arena.step(parsed_actions, dt=float(self.tick_skip) / 120.0)

        # Calculate rewards and observations
        obs = []
        rewards = []
        info_rewards = {}
        prev_touches_sum = sum(self.episode_touches)

        for i, car in enumerate(self.arena.cars):
            r, r_dict = self.reward_manager.get_reward(car, self.arena, parsed_actions[i], is_goal, scoring_team)
            rewards.append(r)
            self.episode_rewards[i] += r
            self.episode_touches[i] = car.ball_touches
            obs.append(self.obs_builder.build_obs(car, self.arena))
            if i == 0:
                info_rewards = r_dict

        if is_goal and scoring_team is not None:
            self.episode_goals[scoring_team] += 1

        step_touches = max(0, sum(self.episode_touches) - prev_touches_sum)

        # RLGym Kickoff Stagnation Rule: If ball is untouched on kickoff after 75 steps (5.0s), terminate episode!
        is_kickoff_stalled = (self.current_step > 75 and abs(self.arena.ball.pos[0]) < 20.0 and abs(self.arena.ball.pos[1]) < 20.0 and np.linalg.norm(self.arena.ball.vel) < 80.0)
        done = (self.current_step >= self.max_episode_steps) or is_goal or is_kickoff_stalled
        dones = np.array([done] * self.num_players, dtype=bool)

        info = {
            "is_goal": is_goal,
            "scoring_team": scoring_team,
            "step": self.current_step,
            "step_touches": step_touches,
            "episode_rewards": list(self.episode_rewards),
            "episode_touches": list(self.episode_touches),
            "episode_goals": list(self.episode_goals),
            "reward_breakdown": info_rewards,
            "is_baseline_env": self.is_baseline_env
        }

        # Auto-reset on goal/max steps/stalled kickoff
        if done:
            self.reset()

        return np.array(obs, dtype=np.float32), np.array(rewards, dtype=np.float32), dones, info


class VectorizedRocketEnv:
    """
    Vectorized parallel environment container running multiple RocketLeagueEnv instances simultaneously.
    Supports dynamic partitioning between self-play environments and baseline bot opponents.
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
        baseline_opponent_ratio: float = 0.25
    ):
        self.num_envs = num_envs
        self.game_mode = game_mode
        self.tick_skip = tick_skip
        self.max_episode_steps = max_episode_steps
        self.reward_weights = reward_weights
        self.continuous_actions = continuous_actions
        self.self_play = self_play
        self.baseline_opponent_ratio = max(0.0, min(1.0, baseline_opponent_ratio))

        num_baseline = int(round(num_envs * self.baseline_opponent_ratio)) if game_mode == "1v1" else 0
        self.envs = [
            RocketLeagueEnv(
                game_mode=game_mode,
                tick_skip=tick_skip,
                max_episode_steps=max_episode_steps,
                reward_weights=reward_weights,
                continuous_actions=continuous_actions,
                self_play=self_play,
                is_baseline_env=(i >= num_envs - num_baseline)
            )
            for i in range(num_envs)
        ]
        self.num_players_per_env = self.envs[0].num_players
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].act_dim

    def update_baseline_ratio(self, ratio: float):
        """Dynamically reconfigures the number of environments running against the baseline opponent."""
        self.baseline_opponent_ratio = max(0.0, min(1.0, ratio))
        num_baseline = int(round(self.num_envs * self.baseline_opponent_ratio)) if self.game_mode == "1v1" else 0
        for i, env in enumerate(self.envs):
            is_baseline = (i >= self.num_envs - num_baseline)
            env.is_baseline_env = is_baseline
            env.baseline_bot = BaselineChaser(continuous_actions=self.continuous_actions) if is_baseline else None

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
        all_obs = []
        for env in self.envs:
            obs = env.reset()
            all_obs.append(obs)
        # Shape: (num_envs, num_players, obs_dim)
        return np.array(all_obs, dtype=np.float32)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        actions shape: (num_envs, num_players, act_dim)
        """
        all_obs = []
        all_rews = []
        all_dones = []
        all_infos = []

        for i, env in enumerate(self.envs):
            obs, rews, dones, info = env.step(actions[i])
            all_obs.append(obs)
            all_rews.append(rews)
            all_dones.append(dones)
            all_infos.append(info)

        return (
            np.array(all_obs, dtype=np.float32),
            np.array(all_rews, dtype=np.float32),
            np.array(all_dones, dtype=bool),
            all_infos
        )
