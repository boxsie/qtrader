"""
Q-Learning components for QTrader
"""

from .model_pytorch import DuelingDQN, DoubleDQNAgent
from .memory_prioritized import Memory, EpisodeMemory
from .trainer_pytorch import Trainer

__all__ = [
    'DuelingDQN',
    'DoubleDQNAgent',
    'Memory',
    'EpisodeMemory',
    'Trainer',
]
