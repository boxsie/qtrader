"""
Reward factory - builds RewardManager from YAML configuration.
"""

from typing import Dict, Any, List
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


# Registry of available reward calculators
REWARD_REGISTRY = {
    'profit_loss': ProfitLossReward,
    'portfolio_growth': PortfolioGrowthReward,
    'opportunity_cost': OpportunityCostReward,
    'drawdown_penalty': DrawdownPenaltyReward,
    'momentum_alignment': MomentumAlignmentReward,
    'trade_frequency': TradingFrequencyReward,
    'buy_the_dip': BuyTheDipReward,
    'position_sizing': PositionSizingReward,
    'sharpe_ratio': SharpeRatioReward,
}


def create_reward_manager(reward_config: List[Dict[str, Any]]) -> RewardManager:
    """
    Create RewardManager from YAML configuration.

    Args:
        reward_config: List of reward calculator configs, each containing:
            - type: Name of reward calculator (e.g., 'profit_loss')
            - weight: Overall weight/multiplier
            - params: Dict of calculator-specific parameters (optional)

    Returns:
        RewardManager instance with configured calculators

    Example config:
        reward_config:
          - type: profit_loss
            weight: 1.0
            params:
              profit_multiplier: 10.0
              loss_multiplier: 2.0

          - type: portfolio_growth
            weight: 2.0
            params:
              peak_multiplier: 2.0

          - type: opportunity_cost
            weight: 0.3
            params:
              lookback: 5
    """
    calculators = []

    for calc_config in reward_config:
        calc_type = calc_config['type']
        weight = calc_config.get('weight', 1.0)
        params = calc_config.get('params', {})

        # Get calculator class from registry
        if calc_type not in REWARD_REGISTRY:
            raise ValueError(f"Unknown reward type: {calc_type}. Available: {list(REWARD_REGISTRY.keys())}")

        calc_class = REWARD_REGISTRY[calc_type]

        # Create instance with weight and params
        calculator = calc_class(weight=weight, **params)
        calculators.append(calculator)

    return RewardManager(calculators)


def get_available_rewards() -> List[str]:
    """Get list of available reward calculator types"""
    return list(REWARD_REGISTRY.keys())


def register_reward(name: str, reward_class):
    """
    Register a custom reward calculator.

    Args:
        name: Name to use in YAML configs
        reward_class: RewardCalculator subclass
    """
    REWARD_REGISTRY[name] = reward_class
