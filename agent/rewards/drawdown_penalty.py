"""
Drawdown Penalty Reward

Penalizes portfolio value dropping below peak (drawdown).
Encourages risk management and capital preservation.
"""

from typing import Dict, Any
from .base import RewardCalculator


class DrawdownPenaltyReward(RewardCalculator):
    """
    Penalizes drawdowns from peak portfolio value.

    Encourages the agent to preserve capital and avoid losses.
    """

    def __init__(self, weight: float = 1.0, penalty_multiplier: float = 5.0):
        """
        Args:
            weight: Overall weight for this reward component
            penalty_multiplier: How much to penalize drawdowns (higher = more penalty)
        """
        super().__init__(weight)
        self.penalty_multiplier = penalty_multiplier
        self.peak_portfolio_value = 0.0

    def reset(self):
        """Reset for new episode"""
        self.peak_portfolio_value = 0.0

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate drawdown penalty.

        Returns negative reward when portfolio is below peak.
        """
        portfolio_value = context['portfolio_value']
        starting_cash = context['starting_cash']

        # Track peak
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            return 0.0  # At new peak, no penalty

        # Calculate drawdown as percentage
        if self.peak_portfolio_value > 0:
            drawdown_pct = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value * 100

            # Penalize drawdown (negative reward)
            penalty = -drawdown_pct * self.penalty_multiplier
            return penalty

        return 0.0

    def get_info(self) -> Dict[str, Any]:
        """Get diagnostic info"""
        return {
            'peak_portfolio_value': self.peak_portfolio_value,
            'penalty_multiplier': self.penalty_multiplier
        }
