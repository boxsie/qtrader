"""
Trading Frequency Reward

Penalizes overtrading and encourages selective, strategic trades.
"""

from typing import Dict, Any
from .base import RewardCalculator


class TradingFrequencyReward(RewardCalculator):
    """
    Penalizes excessive trading, rewards strategic patience.

    Applies a small penalty on each trade to discourage overtrading.
    """

    def __init__(self, weight: float = 1.0, penalty_per_trade: float = 0.1):
        """
        Args:
            weight: Overall weight for this reward component
            penalty_per_trade: Small penalty applied to each trade
        """
        super().__init__(weight)
        self.penalty_per_trade = penalty_per_trade
        self.trade_count = 0

    def reset(self):
        """Reset for new episode"""
        self.trade_count = 0

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate trade frequency penalty.

        Applies small penalty on each trade to discourage overtrading.
        """
        trade_info = context.get('trade_info')

        # Penalize each trade to discourage overtrading
        if trade_info is not None:
            self.trade_count += 1
            return -self.penalty_per_trade

        return 0.0

    def get_info(self) -> Dict[str, Any]:
        """Get diagnostic info"""
        return {
            'trade_count': self.trade_count,
            'penalty_per_trade': self.penalty_per_trade
        }
