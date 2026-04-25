"""
Modular reward system for RL trading agents.

Easy plug-and-play reward calculators.
"""

from .base import RewardCalculator
from .manager import RewardManager
from .profit_loss import ProfitLossReward
from .portfolio_growth import PortfolioGrowthReward
from .opportunity_cost import OpportunityCostReward
from .drawdown_penalty import DrawdownPenaltyReward
from .momentum_alignment import MomentumAlignmentReward
from .trade_frequency import TradingFrequencyReward
from .buy_the_dip import BuyTheDipReward
from .position_sizing import PositionSizingReward
from .sharpe_ratio import SharpeRatioReward
from .factory import create_reward_manager, get_available_rewards, register_reward

__all__ = [
    'RewardCalculator',
    'RewardManager',
    'ProfitLossReward',
    'PortfolioGrowthReward',
    'OpportunityCostReward',
    'DrawdownPenaltyReward',
    'MomentumAlignmentReward',
    'TradingFrequencyReward',
    'BuyTheDipReward',
    'PositionSizingReward',
    'SharpeRatioReward',
    'create_reward_manager',
    'get_available_rewards',
    'register_reward',
]
