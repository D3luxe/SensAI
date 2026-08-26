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


class RocketLeagueEnv:
    """
    Standard single-instance Gymnasium-compatible Rocket League match environment.
    """
    def __init__(
        self,
        game_mode: str = "1v1",
        tick_skip: int = 8,
        max_episode_steps: int = 1500,
        reward_weights: Optional[Dict[str, float]] = None,
        continuous_actions: bool = True,
        self_play: bool = True
    ):
        self.game_mode = game_mode
        self.num_players = 2 if game_mode == "1v1" else (4 if game_mode == "2v2" else 6)
        self.tick_skip = tick_skip
        self.max_episode_steps = max_episode_steps
        self.self_play = self_play
        self.continuous_actions = continuous_actions

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
        parsed_actions = self.action_parser.parse_actions(raw_actions)

        is_goal = False
        scoring_team = None

        # Execute physics sub-ticks
        dt = (1.0 / 120.0)
        for _ in range(self.tick_skip):
            goal, team = self.arena.step(parsed_actions, dt=dt)
            if goal:
                is_goal = True
                scoring_team = team
                break

        # Calculate rewards and observations
        obs = []
        rewards = []
        info_rewards = {}

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

        # RLGym Kickoff Stagnation Rule: If ball is untouched on kickoff after 75 steps (5.0s), terminate episode!
        is_kickoff_stalled = (self.current_step > 75 and abs(self.arena.ball.pos[0]) < 20.0 and abs(self.arena.ball.pos[1]) < 20.0 and np.linalg.norm(self.arena.ball.vel) < 80.0)
        done = (self.current_step >= self.max_episode_steps) or is_goal or is_kickoff_stalled
        dones = np.array([done] * self.num_players, dtype=bool)

        info = {
            "is_goal": is_goal,
            "scoring_team": scoring_team,
            "step": self.current_step,
            "episode_rewards": list(self.episode_rewards),
            "episode_touches": list(self.episode_touches),
            "episode_goals": list(self.episode_goals),
            "reward_breakdown": info_rewards
        }

        # Auto-reset on goal/max steps/stalled kickoff
        if done:
            self.reset()

        return np.array(obs, dtype=np.float32), np.array(rewards, dtype=np.float32), dones, info


class VectorizedRocketEnv:
    """
    Vectorized parallel environment container running multiple RocketLeagueEnv instances simultaneously.
    """
    def __init__(
        self,
        num_envs: int = 16,
        game_mode: str = "1v1",
        tick_skip: int = 8,
        max_episode_steps: int = 1500,
        reward_weights: Optional[Dict[str, float]] = None,
        continuous_actions: bool = True,
        self_play: bool = True
    ):
        self.num_envs = num_envs
        self.envs = [
            RocketLeagueEnv(
                game_mode=game_mode,
                tick_skip=tick_skip,
                max_episode_steps=max_episode_steps,
                reward_weights=reward_weights,
                continuous_actions=continuous_actions,
                self_play=self_play
            )
            for _ in range(num_envs)
        ]
        self.num_players_per_env = self.envs[0].num_players
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].act_dim

    def update_reward_weights(self, weights: Dict[str, float]):
        for env in self.envs:
            env.reward_manager.update_weights(weights)

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
