"""
Sharpe Ratio Reward

Rewards risk-adjusted returns using rolling Sharpe ratio.
Encourages consistent gains with low volatility.
"""

from typing import Dict, Any
from collections import deque
import numpy as np
from .base import RewardCalculator


class SharpeRatioReward(RewardCalculator):
    """
    Rewards high Sharpe ratio (returns relative to volatility).

    Encourages consistent, stable returns over volatile swings.
    """

    def __init__(self, weight: float = 1.0, window: int = 50,
                 sharpe_multiplier: float = 0.5):
        """
        Args:
            weight: Overall weight for this reward component
            window: Rolling window for Sharpe calculation
            sharpe_multiplier: How much to reward Sharpe improvements
        """
        super().__init__(weight)
        self.window = window
        self.sharpe_multiplier = sharpe_multiplier
        self.returns_history = deque(maxlen=window)
        self.last_portfolio_value = None

    def reset(self):
        """Reset for new episode"""
        self.returns_history.clear()
        self.last_portfolio_value = None

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate Sharpe ratio reward.

        Returns positive reward for improving risk-adjusted returns.
        """
        portfolio_value = context['portfolio_value']
        starting_cash = context['starting_cash']

        # Calculate return for this step
        if self.last_portfolio_value is not None and self.last_portfolio_value > 0:
            step_return = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
            self.returns_history.append(step_return)

        self.last_portfolio_value = portfolio_value

        # Need enough history to calculate Sharpe
        if len(self.returns_history) < 10:
            return 0.0

        # Calculate rolling Sharpe ratio
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return > 0:
            sharpe = mean_return / std_return
            # Annualized approximation (assuming ~252 trading days)
            sharpe_annualized = sharpe * np.sqrt(252)

            # Reward positive Sharpe, penalize negative
            reward = sharpe_annualized * self.sharpe_multiplier
            return reward

        return 0.0

    def get_info(self) -> Dict[str, Any]:
        """Get diagnostic info"""
        if len(self.returns_history) > 0:
            returns = np.array(self.returns_history)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = mean_return / std_return if std_return > 0 else 0
        else:
            mean_return = 0
            std_return = 0
            sharpe = 0

        return {
            'window': self.window,
            'returns_count': len(self.returns_history),
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'current_sharpe': float(sharpe)
        }
