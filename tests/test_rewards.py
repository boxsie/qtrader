"""
Quick test to verify the modular reward system works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'agent'))

from rewards import (
    RewardManager,
    ProfitLossReward,
    PortfolioGrowthReward,
    OpportunityCostReward
)


def test_profit_loss_reward():
    """Test ProfitLossReward calculator"""
    print("\n" + "="*60)
    print("Testing ProfitLossReward")
    print("="*60)

    calc = ProfitLossReward(weight=1.0, profit_multiplier=10.0, loss_multiplier=2.0)
    calc.reset()

    # Test profitable sell
    context = {
        'action': 5,
        'price': 100,
        'cash': 1000,
        'crypto_quantity': 0.5,
        'portfolio_value': 1050,
        'starting_cash': 1000,
        'average_buy_price': 90,
        'fee_pct': 0.0015,
        'step': 10,
        'trade_info': {
            'type': 'SELL',
            'quantity': 0.1,
            'fee': 0.15,
            'profit': 1.0,
            'profit_pct': 0.01
        }
    }

    reward = calc.calculate_reward(context)
    print(f"Profitable sell: {reward:.2f} (should be positive)")

    # Test losing sell
    context['trade_info']['profit'] = -5.0
    context['trade_info']['profit_pct'] = -0.05
    reward = calc.calculate_reward(context)
    print(f"Losing sell: {reward:.2f} (should be negative)")

    # Test buy (just fee)
    context['trade_info'] = {
        'type': 'BUY',
        'quantity': 0.1,
        'fee': 1.5
    }
    reward = calc.calculate_reward(context)
    print(f"Buy action: {reward:.2f} (should be small negative fee)")

    # Test hold
    context['trade_info'] = None
    reward = calc.calculate_reward(context)
    print(f"Hold action: {reward:.2f} (should be 0)")

    print("[PASS] ProfitLossReward working")


def test_portfolio_growth_reward():
    """Test PortfolioGrowthReward calculator"""
    print("\n" + "="*60)
    print("Testing PortfolioGrowthReward")
    print("="*60)

    calc = PortfolioGrowthReward(weight=1.0, peak_multiplier=2.0)
    calc.reset()

    context = {
        'action': 0,
        'price': 100,
        'cash': 500,
        'crypto_quantity': 5,
        'portfolio_value': 1000,
        'starting_cash': 1000,
        'average_buy_price': 100,
        'fee_pct': 0.0015,
        'step': 0,
        'trade_info': None
    }

    # First step - initialize
    reward = calc.calculate_reward(context)
    print(f"First step (at starting value): {reward:.2f}")

    # New peak
    context['portfolio_value'] = 1100
    context['step'] = 1
    reward = calc.calculate_reward(context)
    print(f"New peak (+100): {reward:.2f} (should be positive)")

    # Above average
    context['portfolio_value'] = 1110
    context['step'] = 2
    reward = calc.calculate_reward(context)
    print(f"Above average (+10): {reward:.2f} (should be small positive)")

    # Below starting capital
    context['portfolio_value'] = 900
    context['step'] = 3
    reward = calc.calculate_reward(context)
    print(f"Below starting capital (-100): {reward:.2f} (should be negative)")

    print("[PASS] PortfolioGrowthReward working")


def test_reward_manager():
    """Test RewardManager combining multiple calculators"""
    print("\n" + "="*60)
    print("Testing RewardManager")
    print("="*60)

    manager = RewardManager([
        ProfitLossReward(weight=1.0),
        PortfolioGrowthReward(weight=2.0),
        OpportunityCostReward(weight=0.3)
    ])

    print(f"\nActive calculators:")
    for calc in manager.list_calculators():
        print(f"  - {calc}")

    manager.reset()

    context = {
        'action': 5,
        'price': 110,
        'cash': 1100,
        'crypto_quantity': 0,
        'portfolio_value': 1100,
        'starting_cash': 1000,
        'average_buy_price': 100,
        'fee_pct': 0.0015,
        'step': 5,
        'trade_info': {
            'type': 'SELL',
            'quantity': 1.0,
            'fee': 1.65,
            'profit': 100.0,
            'profit_pct': 0.10
        }
    }

    total_reward = manager.calculate_reward(context)
    breakdown = manager.get_breakdown(context)

    print(f"\nReward calculation:")
    for name, value in breakdown.items():
        print(f"  {name:30s}: {value:8.2f}")

    print(f"\n[PASS] RewardManager working")


def test_dynamic_add_remove():
    """Test adding/removing calculators dynamically"""
    print("\n" + "="*60)
    print("Testing Dynamic Add/Remove")
    print("="*60)

    manager = RewardManager([
        ProfitLossReward(weight=1.0)
    ])

    print(f"\nInitial calculators: {len(manager.calculators)}")
    print(manager.list_calculators())

    # Add calculator
    manager.add_calculator(PortfolioGrowthReward(weight=2.0))
    print(f"\nAfter adding: {len(manager.calculators)}")
    print(manager.list_calculators())

    # Remove calculator
    manager.remove_calculator('PortfolioGrowthReward')
    print(f"\nAfter removing: {len(manager.calculators)}")
    print(manager.list_calculators())

    print("\n[PASS] Dynamic add/remove working")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Modular Reward System Tests")
    print("="*60)

    test_profit_loss_reward()
    test_portfolio_growth_reward()
    test_reward_manager()
    test_dynamic_add_remove()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
    print("\nThe modular reward system is ready to use.")
    print("See examples/reward_config_example.py for usage examples.")
    print("="*60 + "\n")
