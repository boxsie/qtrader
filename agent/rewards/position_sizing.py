"""
Position Sizing Reward

Rewards optimal position sizes - not too concentrated, not too small.
Encourages proper risk management.
"""

from typing import Dict, Any
from .base import RewardCalculator


class PositionSizingReward(RewardCalculator):
    """
    Rewards maintaining optimal position size relative to portfolio.

    Penalizes being too concentrated (risky) or too small (inefficient).
    """

    def __init__(self, weight: float = 1.0,
                 optimal_position_pct: float = 0.7,
                 tolerance: float = 0.2,
                 max_penalty: float = 2.0):
        """
        Args:
            weight: Overall weight for this reward component
            optimal_position_pct: Target position size (0.7 = 70% invested)
            tolerance: Acceptable deviation (0.2 = ±20%)
            max_penalty: Maximum penalty for extreme positions
        """
        super().__init__(weight)
        self.optimal_position_pct = optimal_position_pct
        self.tolerance = tolerance
        self.max_penalty = max_penalty

    def reset(self):
        """Reset for new episode"""
        pass

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate position sizing reward.

        Rewards being near optimal position size, penalizes extremes.
        """
        portfolio_value = context['portfolio_value']
        crypto_quantity = context['crypto_quantity']
        price = context['price']

        if portfolio_value == 0:
            return 0.0

        # Calculate current position percentage
        position_value = crypto_quantity * price
        position_pct = position_value / portfolio_value

        # Calculate deviation from optimal
        deviation = abs(position_pct - self.optimal_position_pct)

        # Within tolerance? Small positive reward
        if deviation <= self.tolerance:
            reward = 0.5 * (1 - deviation / self.tolerance)
            return reward

        # Outside tolerance? Penalty proportional to deviation
        excess_deviation = deviation - self.tolerance
        penalty = min(excess_deviation * 5.0, self.max_penalty)
        return -penalty

    def get_info(self) -> Dict[str, Any]:
        """Get diagnostic info"""
        return {
            'optimal_position_pct': self.optimal_position_pct,
            'tolerance': self.tolerance,
            'max_penalty': self.max_penalty
        }
