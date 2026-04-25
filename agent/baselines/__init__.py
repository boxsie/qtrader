"""
Baseline trading strategies for comparison.

These simple strategies help validate whether the RL agent
is actually learning useful patterns or just overfitting.

If the RL agent can't consistently beat these baselines,
it's likely learning noise rather than signal.
"""

from .base import BaselineStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .random_agent import RandomStrategy
from .ma_crossover import MACrossoverStrategy

__all__ = [
    'BaselineStrategy',
    'BuyAndHoldStrategy',
    'RandomStrategy',
    'MACrossoverStrategy',
]
