"""
Opportunity cost reward calculator.

Penalizes holding cash during price increases and holding crypto during price decreases.
"""

from typing import Dict, Any
from collections import deque
from .base import RewardCalculator


class OpportunityCostReward(RewardCalculator):
    """
    Reward/penalize based on opportunity cost.

    - Holding cash while price rises: penalty (missed gains)
    - Holding crypto while price falls: penalty (unrealized loss)
    - Holding cash while price stable: small reward (capital preservation)
    - Holding crypto while price rises: small reward (good position)
    """

    def __init__(self, weight: float = 1.0, lookback: int = 5):
        """
        Args:
            weight: Overall weight for this reward
            lookback: Number of steps to look back for price change
        """
        super().__init__(weight)
        self.lookback = lookback
        self.price_history = deque(maxlen=lookback + 1)

    def reset(self):
        """Reset tracking for new episode"""
        self.price_history.clear()

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate opportunity cost reward.

        Args:
            context: Current context

        Returns:
            Reward/penalty based on opportunity cost
        """
        price = context['price']
        cash = context['cash']
        crypto_quantity = context['crypto_quantity']
        portfolio_value = context['portfolio_value']

        # Track price history
        self.price_history.append(price)

        # Need enough history
        if len(self.price_history) < self.lookback + 1:
            return 0.0

        # Calculate price change over lookback period
        old_price = self.price_history[0]
        price_change_pct = (price - old_price) / old_price

        # Calculate position as % of portfolio
        cash_pct = cash / portfolio_value if portfolio_value > 0 else 0
        crypto_pct = 1.0 - cash_pct

        reward = 0.0

        # Penalize holding cash during uptrends (missed opportunity)
        if price_change_pct > 0.02 and cash_pct > 0.3:
            # Penalty scales with how much cash and how big the move
            reward -= abs(price_change_pct) * cash_pct * 30

        # Penalize holding crypto during downtrends (unrealized loss)
        elif price_change_pct < -0.02 and crypto_pct > 0.3:
            # Penalty scales with how much crypto and how big the drop
            reward -= abs(price_change_pct) * crypto_pct * 30

        # Small reward for being positioned correctly
        elif price_change_pct > 0.01 and crypto_pct > 0.5:
            # Holding crypto during uptrend
            reward += price_change_pct * crypto_pct * 10

        elif price_change_pct < -0.01 and cash_pct > 0.5:
            # Holding cash during downtrend
            reward += abs(price_change_pct) * cash_pct * 10

        return reward

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic info"""
        return {
            'price_history_size': len(self.price_history)
        }
