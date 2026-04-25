"""
Example: How to configure the modular reward system.

This shows different reward configurations you can try.
"""

from agent.rewards import (
    RewardManager,
    ProfitLossReward,
    PortfolioGrowthReward,
    OpportunityCostReward
)


# Example 1: Profit-focused (aggressive trading)
def profit_focused_rewards():
    """Focus on realized profits from trades"""
    return RewardManager([
        ProfitLossReward(weight=2.0, profit_multiplier=10.0, loss_multiplier=2.0),
        PortfolioGrowthReward(weight=1.0, peak_multiplier=2.0),
    ])


# Example 2: Portfolio growth focused (conservative)
def growth_focused_rewards():
    """Focus on steady portfolio growth"""
    return RewardManager([
        ProfitLossReward(weight=0.5, profit_multiplier=5.0, loss_multiplier=3.0),
        PortfolioGrowthReward(weight=3.0, peak_multiplier=3.0),
    ])


# Example 3: Balanced (all components)
def balanced_rewards():
    """Balance between profits, growth, and opportunity cost"""
    return RewardManager([
        ProfitLossReward(weight=1.0, profit_multiplier=10.0, loss_multiplier=2.0),
        PortfolioGrowthReward(weight=2.0, peak_multiplier=2.0),
        OpportunityCostReward(weight=0.3, lookback=5)
    ])


# Example 4: Minimal (just profit/loss)
def minimal_rewards():
    """Only reward realized profits, nothing else"""
    return RewardManager([
        ProfitLossReward(weight=1.0, profit_multiplier=10.0, loss_multiplier=2.0)
    ])


# Example 5: Custom mix
def custom_rewards():
    """Create your own custom mix"""
    return RewardManager([
        # Heavy emphasis on portfolio growth
        PortfolioGrowthReward(weight=5.0, peak_multiplier=3.0),

        # Moderate profit/loss tracking
        ProfitLossReward(weight=1.0, profit_multiplier=8.0, loss_multiplier=3.0),

        # Light opportunity cost penalty
        OpportunityCostReward(weight=0.2, lookback=10)
    ])


# How to use in your training script:
if __name__ == "__main__":
    # 1. Create reward manager with your chosen config
    reward_manager = balanced_rewards()

    # 2. Pass it to your broker
    from agent.broker import Broker
    from agent.ticker import Ticker

    # ... setup ticker ...
    # broker = Broker(ticker, starting_cash=1000, reward_manager=reward_manager)

    # 3. The broker will automatically use the modular reward system

    # You can also inspect active rewards:
    print("Active reward calculators:")
    for calc_name in reward_manager.list_calculators():
        print(f"  - {calc_name}")

    # Add/remove rewards dynamically:
    # reward_manager.add_calculator(OpportunityCostReward(weight=0.5))
    # reward_manager.remove_calculator('OpportunityCostReward')
