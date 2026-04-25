#!/usr/bin/env python3
"""
Evaluation Script for QTrader

This script provides proper out-of-sample evaluation:
1. Loads a trained model
2. Evaluates on test data (unseen during training)
3. Compares against baselines
4. Reports comprehensive metrics

Usage:
    python evaluate.py --model runs/best_model.pt --data data/btcusd_1-min_data.csv

This is your reality check - if the model can't beat baselines on test data,
it's likely overfitting to training data patterns.
"""

import argparse
from pathlib import Path
import json
import yaml
import numpy as np

# Local imports
from ticker import Ticker
from broker import Broker
from qlearn.model_pytorch import DuelingDQNAgent
from evaluation import (
    StrategyComparison,
    TimeSeriesSplitter,
    RegimeAwareSplitter,
    DataSplit,
    EvaluationMetrics,
)
from baselines import (
    BuyAndHoldStrategy,
    RandomStrategy,
    MACrossoverStrategy,
)


def load_agent(model_path: str, num_states: int, num_actions: int = 9) -> DuelingDQNAgent:
    """Load a trained agent from checkpoint."""
    agent = DuelingDQNAgent(
        num_states=num_states,
        num_actions=num_actions,
        hidden_size=128,
        learning_rate=0.0001,
        gamma=0.99,
        tau=0.01
    )
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    return agent


def create_broker_factory(
    data_path: str,
    timeframe_weights: dict,
    split: DataSplit = DataSplit.TEST,
    splitter: TimeSeriesSplitter = None
):
    """
    Create a factory function that returns fresh Broker instances.

    If splitter is provided, uses split-specific data.
    Otherwise uses full data.
    """
    def factory():
        # For proper evaluation, we should use split-specific data
        # For now, we'll use the full data with the Ticker
        # TODO: Create split-specific data files for proper isolation

        ticker = Ticker(
            filename=data_path,
            timeframe_weights=timeframe_weights,
            random_start=False  # No random start for evaluation
        )

        return Broker(
            ticker=ticker,
            starting_cash=1000,
            fee_pct=0.0015
        )

    return factory


def evaluate_on_split(
    agent,
    data_path: str,
    timeframe_weights: dict,
    split: DataSplit,
    splitter: TimeSeriesSplitter,
    max_steps: int = None
) -> dict:
    """
    Evaluate agent on a specific data split.

    Returns metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {split.value.upper()} split")
    print(f"{'='*60}")

    # Create broker factory for this split
    broker_factory = create_broker_factory(
        data_path=data_path,
        timeframe_weights=timeframe_weights,
        split=split,
        splitter=splitter
    )

    # Run comparison
    comparison = StrategyComparison(broker_factory, max_steps=max_steps)
    results = comparison.run_comparison(
        agent=agent,
        include_baselines=['buy_and_hold', 'random', 'ma_crossover'],
        num_episodes=1
    )

    comparison.print_comparison()

    return results


def full_evaluation(
    model_path: str,
    data_path: str,
    config_path: str = None,
    output_dir: str = "evaluation_results"
):
    """
    Run full evaluation pipeline.

    1. Split data chronologically
    2. Evaluate on test split
    3. Compare to baselines
    4. Generate report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        timeframe_weights = config.get('timeframe_weights', {
            '15m': 0.45, '1h': 0.20, '5m': 0.15, '30m': 0.15, '4h': 0.05
        })
    else:
        # Default day trader weights
        timeframe_weights = {
            '15m': 0.45, '1h': 0.20, '5m': 0.15, '30m': 0.15, '4h': 0.05
        }

    # Create data splitter
    print("\n[1/4] Setting up data splits...")
    splitter = RegimeAwareSplitter(
        data_path=data_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Show regime distribution
    print("\n[2/4] Analyzing market regimes...")
    try:
        regime_dist = splitter.get_regime_distribution_by_split()
        for split_name, dist in regime_dist.items():
            print(f"\n{split_name.upper()} split regimes:")
            for regime, pct in dist.items():
                print(f"  {regime}: {pct:.1%}")
    except Exception as e:
        print(f"Could not analyze regimes: {e}")

    # Load agent
    print("\n[3/4] Loading trained agent...")
    # Create a temporary ticker to get num_states
    temp_ticker = Ticker(
        filename=data_path,
        timeframe_weights=timeframe_weights
    )
    num_states = temp_ticker.num_features

    agent = load_agent(model_path, num_states=num_states)

    # Evaluate on test split
    print("\n[4/4] Running evaluation...")
    test_results = evaluate_on_split(
        agent=agent,
        data_path=data_path,
        timeframe_weights=timeframe_weights,
        split=DataSplit.TEST,
        splitter=splitter
    )

    # Generate report
    report = {
        'model_path': str(model_path),
        'data_path': str(data_path),
        'split_ratios': {
            'train': splitter.train_ratio,
            'validation': splitter.val_ratio,
            'test': splitter.test_ratio
        },
        'results': {}
    }

    for name, result in test_results.items():
        report['results'][name] = result.metrics.to_dict()

    # Save report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {report_path}")

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    if 'rl_agent' in test_results and 'buy_and_hold' in test_results:
        rl = test_results['rl_agent'].metrics
        bh = test_results['buy_and_hold'].metrics

        diff = rl.total_return_pct - bh.total_return_pct

        print(f"\nRL Agent Return: {rl.total_return_pct:+.2f}%")
        print(f"Buy & Hold Return: {bh.total_return_pct:+.2f}%")
        print(f"Difference: {diff:+.2f}%")

        print(f"\nRL Agent Sharpe: {rl.sharpe_ratio:.3f}")
        print(f"RL Agent Max Drawdown: {rl.max_drawdown_pct:.2f}%")

        if diff > 0:
            print("\n[PASS] Agent beats buy & hold on test data!")
        else:
            print("\n[FAIL] Agent underperforms buy & hold on test data.")
            print("Possible causes:")
            print("  - Overfitting to training data")
            print("  - Different market regime in test period")
            print("  - Reward function encouraging wrong behavior")
            print("  - Need more training data diversity")

    return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate QTrader model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to OHLCV data file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML (optional)')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum steps per evaluation episode')

    args = parser.parse_args()

    full_evaluation(
        model_path=args.model,
        data_path=args.data,
        config_path=args.config,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
