"""
Profit/Loss reward calculator.

Rewards realized profits from trades and penalizes fees.
"""

from typing import Dict, Any
from .base import RewardCalculator


class ProfitLossReward(RewardCalculator):
    """
    Reward based on immediate profit/loss from trades.

    - Sells: Reward profit, punish loss
    - Buys: Penalize fees
    - Holds: No reward
    """

    def __init__(self, weight: float = 1.0, profit_multiplier: float = 10.0, loss_multiplier: float = 2.0):
        """
        Args:
            weight: Overall weight for this reward
            profit_multiplier: Multiplier for profitable trades (default 10x)
            loss_multiplier: Multiplier for losing trades (default 2x)
        """
        super().__init__(weight)
        self.profit_multiplier = profit_multiplier
        self.loss_multiplier = loss_multiplier
        self.total_profit = 0.0
        self.total_fees = 0.0

    def reset(self):
        """Reset tracking for new episode"""
        self.total_profit = 0.0
        self.total_fees = 0.0

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        """
        Calculate reward from trade profit/loss.

        Args:
            context: Current context

        Returns:
            Reward based on profit/loss
        """
        trade_info = context.get('trade_info')
        if trade_info is None:
            return 0.0

        starting_cash = context['starting_cash']
        fee = trade_info.get('fee', 0)
        self.total_fees += fee

        # Normalize fee penalty
        fee_penalty = -fee / starting_cash * 100

        # SELL: reward profit aggressively, punish loss moderately
        if trade_info['type'] == 'SELL':
            profit = trade_info.get('profit', 0)
            self.total_profit += profit

            # Normalize profit/loss as percentage of starting capital
            profit_normalized = profit / starting_cash * 100

            if profit > 0:
                profit_reward = profit_normalized * self.profit_multiplier
            else:
                profit_reward = profit_normalized * self.loss_multiplier

            return profit_reward + fee_penalty

        # BUY: fee penalty with slight multiplier to discourage excessive buying
        elif trade_info['type'] == 'BUY':
            return fee_penalty * 1.5  # Slightly penalize buy fees more

        return 0.0

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic info"""
        return {
            'total_profit': self.total_profit,
            'total_fees': self.total_fees,
            'net_profit': self.total_profit - self.total_fees
        }
