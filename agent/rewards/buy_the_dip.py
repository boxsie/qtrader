"""
Buy The Dip Reward

Rewards buying during price drops and selling during price spikes.
Encourages contrarian, value-seeking behavior.
"""

from typing import Dict, Any
from collections import deque
from .base import RewardCalculator


class BuyTheDipReward(RewardCalculator):
    """
    Rewards buying when price recently dropped, selling when price recently spiked.

    Encourages value-seeking and profit-taking behavior.
    """

    def __init__(self, weight: float = 1.0, lookback: int = 10,
                 dip_threshold: float = 0.02, spike_threshold: float = 0.02,
                 dip_buy_reward: float = 5.0, spike_sell_reward: float = 5.0):
        """
        Args:
            weight: Overall weight for this reward component
            lookback: How many steps to look back for dips/spikes
            dip_threshold: Price drop % to qualify as a dip (0.02 = 2%)
            spike_threshold: Price rise % to qualify as a spike (0.02 = 2%)
            dip_buy_reward: Reward for buying the dip
            spike_sell_reward: Reward for selling the spike
        """
        super().__init__(weight)
        self.lookback = lookback
        self.dip_threshold = dip_threshold
        self.spike_threshold = spike_threshold
        self.dip_buy_reward = dip_buy_reward
        self.spike_sell_reward = spike_sell_reward
        self.price_history = deque(maxlen=lookback + 1)

    def reset(self):
        """Reset for new episode"""
        self.price_history.clear()

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate buy-the-dip / sell-the-spike reward.
        """
        price = context['price']
        trade_info = context.get('trade_info')

        self.price_history.append(price)

        # Need enough history
        if len(self.price_history) < self.lookback + 1:
            return 0.0

        # Only reward actual trades
        if trade_info is None:
            return 0.0

        trade_type = trade_info['type']

        # Calculate recent price change
        recent_high = max(self.price_history)
        recent_low = min(self.price_history)

        # Is this a dip? (price near recent low)
        if recent_high > 0:
            drop_from_high = (recent_high - price) / recent_high
            is_dip = drop_from_high >= self.dip_threshold

            if is_dip and trade_type == 'BUY':
                # Bought the dip! Good contrarian move
                return self.dip_buy_reward

        # Is this a spike? (price near recent high)
        if recent_low > 0:
            rise_from_low = (price - recent_low) / recent_low
            is_spike = rise_from_low >= self.spike_threshold

            if is_spike and trade_type == 'SELL':
                # Sold the spike! Taking profits
                return self.spike_sell_reward

        return 0.0

    def get_info(self) -> Dict[str, Any]:
        """Get diagnostic info"""
        return {
            'lookback': self.lookback,
            'dip_threshold': self.dip_threshold,
            'spike_threshold': self.spike_threshold,
            'price_history_size': len(self.price_history)
        }
