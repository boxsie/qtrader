"""
Base reward calculator interface for modular reward system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class RewardCalculator(ABC):
    """
    Base class for all reward calculators.

    Each calculator computes an immediate reward based on current context.
    The Q-learning algorithm handles long-term consequences through Bellman updates.
    """

    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight: Multiplier for this reward component (default 1.0)
        """
        self.weight = weight
        self.name = self.__class__.__name__

    @abstractmethod
    def reset(self):
        """Reset internal state for new episode"""
        pass

    @abstractmethod
    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate immediate reward given current context.

        Context contains:
            - action: Action taken this step (0-8)
            - price: Current price
            - cash: Current cash balance
            - crypto_quantity: Current crypto holdings
            - portfolio_value: Current total portfolio value
            - starting_cash: Initial capital
            - average_buy_price: Average buy price for cost basis
            - fee_pct: Trading fee percentage
            - step: Current step number
            - trade_info: Dict with trade details if a trade executed, else None
                - type: 'BUY' or 'SELL'
                - quantity: Amount traded
                - fee: Fee paid
                - profit: Profit/loss (for sells)
                - profit_pct: Profit percentage (for sells)

        Returns:
            Reward value (will be multiplied by self.weight)
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Return diagnostic info for logging/debugging.

        Returns:
            Dictionary with any useful info (e.g., internal state, counters)
        """
        return {}
