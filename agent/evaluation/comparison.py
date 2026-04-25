"""
Strategy comparison tools for evaluating RL agent vs baselines.

This is the most important module - it tells you whether your
RL agent is actually learning or just overfitting.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from ..baselines import (
    BaselineStrategy,
    BuyAndHoldStrategy,
    RandomStrategy,
    MACrossoverStrategy,
)
from .metrics import EvaluationMetrics, compute_metrics


@dataclass
class ComparisonResult:
    """Results from comparing strategies."""
    strategy_name: str
    metrics: EvaluationMetrics
    portfolio_history: List[float]
    trade_history: List[dict]


class StrategyComparison:
    """
    Compare RL agent against baseline strategies.

    This is your reality check - if you can't beat baselines,
    the RL model isn't learning useful patterns.
    """

    def __init__(self, broker_factory, max_steps: int = None):
        """
        Initialize comparison.

        Args:
            broker_factory: Callable that returns a fresh Broker instance
            max_steps: Maximum steps per evaluation episode
        """
        self.broker_factory = broker_factory
        self.max_steps = max_steps
        self.results: Dict[str, ComparisonResult] = {}

    def evaluate_baseline(
        self,
        strategy: BaselineStrategy,
        num_episodes: int = 1
    ) -> ComparisonResult:
        """
        Evaluate a baseline strategy.

        Args:
            strategy: Baseline strategy to evaluate
            num_episodes: Number of episodes to average over

        Returns:
            ComparisonResult with metrics
        """
        all_metrics = []
        all_portfolio_histories = []
        all_trade_histories = []

        for _ in range(num_episodes):
            broker = self.broker_factory()
            result = strategy.evaluate(broker, max_steps=self.max_steps)

            # Get detailed data from broker
            price = broker._last_tick['Close'] if broker._last_tick is not None else 0
            portfolio_value = broker._cash + (broker._crypto_quantity * price)

            all_metrics.append(result)
            all_trade_histories.extend(broker._trade_history)

        # Average metrics across episodes
        avg_result = {
            'strategy': strategy.name,
            'total_reward': np.mean([m['total_reward'] for m in all_metrics]),
            'final_portfolio_value': np.mean([m['final_portfolio_value'] for m in all_metrics]),
            'roi': np.mean([m['roi'] for m in all_metrics]),
            'sharpe_ratio': np.mean([m['sharpe_ratio'] for m in all_metrics]),
            'max_drawdown': np.mean([m['max_drawdown'] for m in all_metrics]),
            'num_trades': np.mean([m['num_trades'] for m in all_metrics]),
            'win_rate': np.mean([m['win_rate'] for m in all_metrics]),
            'total_fees': np.mean([m['total_fees'] for m in all_metrics]),
        }

        # Create metrics object
        metrics = EvaluationMetrics(
            total_return_pct=avg_result['roi'],
            annualized_return_pct=avg_result['roi'],  # Simplified
            total_reward=avg_result['total_reward'],
            sharpe_ratio=avg_result['sharpe_ratio'],
            sortino_ratio=0,  # Not computed for baselines
            max_drawdown_pct=avg_result['max_drawdown'],
            max_drawdown_duration=0,
            calmar_ratio=0,
            num_trades=int(avg_result['num_trades']),
            win_rate=avg_result['win_rate'],
            profit_factor=0,
            avg_trade_profit=0,
            total_fees=avg_result['total_fees'],
        )

        return ComparisonResult(
            strategy_name=strategy.name,
            metrics=metrics,
            portfolio_history=[],  # Not tracked for baselines
            trade_history=all_trade_histories,
        )

    def evaluate_rl_agent(
        self,
        agent,
        num_episodes: int = 1
    ) -> ComparisonResult:
        """
        Evaluate RL agent.

        Args:
            agent: Trained RL agent with policy_net.get_action()
            num_episodes: Number of episodes

        Returns:
            ComparisonResult with metrics
        """
        all_portfolio_values = []
        all_rewards = []
        all_trades = []

        for _ in range(num_episodes):
            broker = self.broker_factory()
            broker.reset()
            state = broker.get_state()

            if state is None:
                continue

            episode_reward = 0
            episode_portfolio = [broker._starting_cash]
            hidden_state = None
            steps = 0

            while True:
                action, hidden_state = agent.policy_net.get_action(
                    state,
                    epsilon=0.0,  # No exploration during evaluation
                    hidden_state=hidden_state
                )

                done, next_state, reward, info = broker.update(action)
                episode_reward += reward
                steps += 1

                if info:
                    episode_portfolio.append(info['portfolio_value'])

                if done:
                    break

                if self.max_steps and steps >= self.max_steps:
                    break

                state = next_state

            all_portfolio_values.append(episode_portfolio)
            all_rewards.append(episode_reward)
            all_trades.extend(broker._trade_history)

        # Use last episode's portfolio for metrics
        if all_portfolio_values:
            portfolio = np.array(all_portfolio_values[-1])
            starting_value = broker._starting_cash

            metrics = compute_metrics(
                portfolio_values=portfolio,
                trades=all_trades,
                total_reward=np.mean(all_rewards),
                starting_value=starting_value,
            )
        else:
            metrics = EvaluationMetrics(
                total_return_pct=0, annualized_return_pct=0, total_reward=0,
                sharpe_ratio=0, sortino_ratio=0, max_drawdown_pct=0,
                max_drawdown_duration=0, calmar_ratio=0, num_trades=0,
                win_rate=0, profit_factor=0, avg_trade_profit=0, total_fees=0
            )

        return ComparisonResult(
            strategy_name="RL Agent",
            metrics=metrics,
            portfolio_history=all_portfolio_values[-1] if all_portfolio_values else [],
            trade_history=all_trades,
        )

    def run_comparison(
        self,
        agent=None,
        include_baselines: List[str] = None,
        num_episodes: int = 1
    ) -> Dict[str, ComparisonResult]:
        """
        Run full comparison of RL agent vs baselines.

        Args:
            agent: RL agent (optional, can run baselines only)
            include_baselines: List of baseline names to include
                Options: 'buy_and_hold', 'random', 'ma_crossover', 'rsi'
            num_episodes: Episodes per strategy

        Returns:
            Dictionary of strategy name -> ComparisonResult
        """
        if include_baselines is None:
            include_baselines = ['buy_and_hold', 'random', 'ma_crossover']

        results = {}

        # Run baselines
        baseline_map = {
            'buy_and_hold': BuyAndHoldStrategy(),
            'random': RandomStrategy(seed=42),
            'ma_crossover': MACrossoverStrategy(fast_period=10, slow_period=30),
        }

        for name in include_baselines:
            if name in baseline_map:
                print(f"Evaluating {name}...")
                results[name] = self.evaluate_baseline(
                    baseline_map[name],
                    num_episodes=num_episodes
                )

        # Run RL agent
        if agent is not None:
            print("Evaluating RL Agent...")
            results['rl_agent'] = self.evaluate_rl_agent(
                agent,
                num_episodes=num_episodes
            )

            # Calculate relative performance
            if 'buy_and_hold' in results:
                bh_return = results['buy_and_hold'].metrics.total_return_pct
                rl_return = results['rl_agent'].metrics.total_return_pct
                results['rl_agent'].metrics.vs_buy_hold_pct = rl_return - bh_return

            if 'random' in results:
                rand_return = results['random'].metrics.total_return_pct
                rl_return = results['rl_agent'].metrics.total_return_pct
                results['rl_agent'].metrics.vs_random_pct = rl_return - rand_return

        self.results = results
        return results

    def print_comparison(self):
        """Print comparison table."""
        if not self.results:
            print("No results to display. Run comparison first.")
            return

        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON")
        print("=" * 80)

        # Header
        headers = ["Strategy", "Return %", "Sharpe", "Max DD %", "Trades", "Win Rate", "Fees"]
        header_fmt = "{:<20} {:>10} {:>8} {:>10} {:>8} {:>10} {:>10}"
        print(header_fmt.format(*headers))
        print("-" * 80)

        # Sort by return
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].metrics.total_return_pct,
            reverse=True
        )

        for name, result in sorted_results:
            m = result.metrics
            row = [
                name[:20],
                f"{m.total_return_pct:+.2f}",
                f"{m.sharpe_ratio:.3f}",
                f"{m.max_drawdown_pct:.2f}",
                str(m.num_trades),
                f"{m.win_rate:.1%}",
                f"${m.total_fees:.2f}",
            ]
            print(header_fmt.format(*row))

        print("=" * 80)

        # Summary
        if 'rl_agent' in self.results and 'buy_and_hold' in self.results:
            rl = self.results['rl_agent'].metrics
            bh = self.results['buy_and_hold'].metrics

            diff = rl.total_return_pct - bh.total_return_pct
            status = "BEATS" if diff > 0 else "LOSES TO"

            print(f"\nRL Agent {status} Buy & Hold by {abs(diff):.2f}%")

            if diff > 0:
                print("The agent is learning useful patterns.")
            else:
                print("WARNING: Agent underperforms simple buy & hold!")
                print("Consider: more training, different rewards, or data issues.")

        print()

    def save_results(self, filepath: str):
        """Save comparison results to JSON."""
        output = {}
        for name, result in self.results.items():
            output[name] = {
                'metrics': result.metrics.to_dict(),
                'num_trades': len(result.trade_history),
            }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {filepath}")


def quick_compare(agent, broker_factory, max_steps: int = None) -> Dict[str, ComparisonResult]:
    """
    Quick comparison of RL agent vs all baselines.

    Args:
        agent: Trained RL agent
        broker_factory: Callable returning fresh Broker
        max_steps: Max steps per episode

    Returns:
        Comparison results dictionary
    """
    comparison = StrategyComparison(broker_factory, max_steps=max_steps)
    results = comparison.run_comparison(
        agent=agent,
        include_baselines=['buy_and_hold', 'random', 'ma_crossover'],
        num_episodes=1
    )
    comparison.print_comparison()
    return results
