"""
Portfolio growth reward calculator.

Rewards increases in total portfolio value.
"""

from typing import Dict, Any
from collections import deque
import numpy as np
from .base import RewardCalculator


class PortfolioGrowthReward(RewardCalculator):
    """
    Reward based on portfolio value growth.

    Tracks portfolio value and rewards:
    - New all-time highs
    - Growth above recent average
    - Punishes decline below starting capital
    """

    def __init__(self, weight: float = 1.0, peak_multiplier: float = 2.0, history_size: int = 100):
        """
        Args:
            weight: Overall weight for this reward
            peak_multiplier: Multiplier for new peak rewards (default 2x)
            history_size: How many steps of history to track
        """
        super().__init__(weight)
        self.peak_multiplier = peak_multiplier
        self.history_size = history_size
        self.peak_value = 0.0
        self.value_history = deque(maxlen=history_size)

    def reset(self):
        """Reset tracking for new episode"""
        self.peak_value = 0.0
        self.value_history.clear()

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate reward from portfolio growth.

        Args:
            context: Current context

        Returns:
            Reward based on portfolio value changes
        """
        portfolio_value = context['portfolio_value']
        starting_cash = context['starting_cash']

        # Initialize peak on first step
        if self.peak_value == 0.0:
            self.peak_value = starting_cash

        # Track history
        self.value_history.append(portfolio_value)

        # Only reward SIGNIFICANT new all-time highs (>0.5% gain)
        if portfolio_value > self.peak_value:
            growth_pct = (portfolio_value - self.peak_value) / self.peak_value
            if growth_pct > 0.005:  # Only reward >0.5% gains
                growth = (portfolio_value - self.peak_value) / starting_cash * 100
                self.peak_value = portfolio_value
                return growth * self.peak_multiplier
            else:
                # Update peak but don't reward tiny fluctuations
                self.peak_value = portfolio_value

        # Remove the "above average" reward - it was causing buy-and-hold behavior
        # The agent should focus on realized profits, not unrealized gains

        # Below starting capital - penalty
        if portfolio_value < starting_cash * 0.95:  # Only penalize if >5% down
            performance = (portfolio_value - starting_cash) / starting_cash
            return performance * 2.0  # Reduced from 3.0

        return 0.0

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic info"""
        return {
            'peak_value': self.peak_value,
            'current_value': self.value_history[-1] if self.value_history else 0,
            'value_history_size': len(self.value_history)
        }
