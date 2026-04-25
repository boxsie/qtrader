"""
Momentum Alignment Reward

Rewards trades that align with price momentum.
Penalizes counter-trend trades.
"""

from typing import Dict, Any
from collections import deque
from .base import RewardCalculator


class MomentumAlignmentReward(RewardCalculator):
    """
    Rewards buying when price is rising, selling when falling.
    Encourages trend-following behavior.
    """

    def __init__(self, weight: float = 1.0, lookback: int = 5,
                 alignment_reward: float = 2.0, misalignment_penalty: float = 1.0):
        """
        Args:
            weight: Overall weight for this reward component
            lookback: How many steps back to calculate momentum
            alignment_reward: Reward for aligned trades
            misalignment_penalty: Penalty for counter-trend trades
        """
        super().__init__(weight)
        self.lookback = lookback
        self.alignment_reward = alignment_reward
        self.misalignment_penalty = misalignment_penalty
        self.price_history = deque(maxlen=lookback + 1)

    def reset(self):
        """Reset for new episode"""
        self.price_history.clear()

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate momentum alignment reward.

        BUY during uptrend = positive
        SELL during downtrend = positive
        BUY during downtrend = negative
        SELL during uptrend = negative
        """
        price = context['price']
        action = context['action']
        trade_info = context.get('trade_info')

        self.price_history.append(price)

        # Need enough history
        if len(self.price_history) < self.lookback + 1:
            return 0.0

        # Calculate momentum (price change over lookback period)
        momentum = price - self.price_history[0]

        # Only reward/penalize actual trades
        if trade_info is None:
            return 0.0

        trade_type = trade_info['type']

        # BUY during uptrend or SELL during downtrend = aligned
        if (trade_type == 'BUY' and momentum > 0) or (trade_type == 'SELL' and momentum < 0):
            return self.alignment_reward

        # BUY during downtrend or SELL during uptrend = misaligned
        if (trade_type == 'BUY' and momentum < 0) or (trade_type == 'SELL' and momentum > 0):
            return -self.misalignment_penalty

        return 0.0

    def get_info(self) -> Dict[str, Any]:
        """Get diagnostic info"""
        return {
            'lookback': self.lookback,
            'price_history_size': len(self.price_history),
            'alignment_reward': self.alignment_reward,
            'misalignment_penalty': self.misalignment_penalty
        }
