"""
Agent Package for Rocket League Bot.
"""

from agent.models import ActorCritic
from agent.ppo import PPOTrainer

__all__ = ["ActorCritic", "PPOTrainer"]
