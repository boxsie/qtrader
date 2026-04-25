"""
Performance metrics for trading strategy evaluation.

These metrics are standard in quantitative finance and help
assess risk-adjusted performance, not just raw returns.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


def calculate_returns(portfolio_values: np.ndarray) -> np.ndarray:
    """
    Calculate period returns from portfolio values.

    Args:
        portfolio_values: Array of portfolio values over time

    Returns:
        Array of period returns
    """
    portfolio_values = np.array(portfolio_values)
    if len(portfolio_values) < 2:
        return np.array([])

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return returns


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 525600  # Minutes per year (default for 1-min data)
) -> float:
    """
    Calculate annualized Sharpe ratio.

    The Sharpe ratio measures risk-adjusted return:
    - Higher is better
    - > 1.0 is good
    - > 2.0 is very good
    - > 3.0 is exceptional

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    returns = np.array(returns)

    if len(returns) < 2:
        return 0.0

    # Remove NaN/Inf
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0

    # Per-period risk-free rate
    rf_per_period = risk_free_rate / periods_per_year

    # Excess returns
    excess_returns = returns - rf_per_period

    # Sharpe ratio (annualized)
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    sharpe_annualized = sharpe * np.sqrt(periods_per_year)

    return sharpe_annualized


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 525600
) -> float:
    """
    Calculate annualized Sortino ratio.

    Like Sharpe but only penalizes downside volatility.
    Better for strategies with asymmetric returns.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    returns = np.array(returns)

    if len(returns) < 2:
        return 0.0

    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    # Downside returns only
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0

    downside_std = np.std(downside_returns)
    sortino = np.mean(excess_returns) / downside_std
    sortino_annualized = sortino * np.sqrt(periods_per_year)

    return sortino_annualized


def calculate_max_drawdown(portfolio_values: np.ndarray) -> tuple:
    """
    Calculate maximum drawdown and its duration.

    Max drawdown is the largest peak-to-trough decline.
    This is crucial for understanding worst-case scenarios.

    Args:
        portfolio_values: Array of portfolio values

    Returns:
        Tuple of (max_drawdown_pct, drawdown_duration_steps)
    """
    portfolio_values = np.array(portfolio_values)

    if len(portfolio_values) < 2:
        return 0.0, 0

    # Running maximum
    peak = np.maximum.accumulate(portfolio_values)

    # Drawdown at each point
    drawdown = (peak - portfolio_values) / peak

    # Maximum drawdown
    max_dd = np.max(drawdown)

    # Find drawdown duration (steps from peak to recovery)
    max_dd_idx = np.argmax(drawdown)

    # Find when the drawdown started (last peak before max_dd_idx)
    peaks_before = np.where(drawdown[:max_dd_idx] == 0)[0]
    start_idx = peaks_before[-1] if len(peaks_before) > 0 else 0

    # Find recovery point (first time we reach peak again after max_dd_idx)
    recovery_candidates = np.where(drawdown[max_dd_idx:] == 0)[0]
    if len(recovery_candidates) > 0:
        end_idx = max_dd_idx + recovery_candidates[0]
    else:
        end_idx = len(drawdown) - 1

    duration = end_idx - start_idx

    return max_dd * 100, duration  # Return as percentage


def calculate_calmar_ratio(
    portfolio_values: np.ndarray,
    periods_per_year: int = 525600
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Measures return per unit of drawdown risk.
    - > 1.0 is acceptable
    - > 3.0 is good
    - > 5.0 is excellent

    Args:
        portfolio_values: Array of portfolio values
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    portfolio_values = np.array(portfolio_values)

    if len(portfolio_values) < 2:
        return 0.0

    # Total return
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

    # Annualize
    num_periods = len(portfolio_values)
    years = num_periods / periods_per_year
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = total_return

    # Max drawdown
    max_dd, _ = calculate_max_drawdown(portfolio_values)
    max_dd_decimal = max_dd / 100

    if max_dd_decimal == 0:
        return np.inf if annualized_return > 0 else 0.0

    return annualized_return / max_dd_decimal


def calculate_win_rate(trades: List[dict]) -> float:
    """
    Calculate win rate from trade history.

    Args:
        trades: List of trade dictionaries with 'profit' key

    Returns:
        Win rate as decimal (0-1)
    """
    profits = [t.get('profit', 0) for t in trades if t.get('type') == 'SELL']

    if len(profits) == 0:
        return 0.0

    wins = sum(1 for p in profits if p > 0)
    return wins / len(profits)


def calculate_profit_factor(trades: List[dict]) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    - > 1.0 means profitable
    - > 1.5 is good
    - > 2.0 is very good

    Args:
        trades: List of trade dictionaries

    Returns:
        Profit factor
    """
    profits = [t.get('profit', 0) for t in trades if t.get('type') == 'SELL']

    if len(profits) == 0:
        return 0.0

    gross_profit = sum(p for p in profits if p > 0)
    gross_loss = abs(sum(p for p in profits if p < 0))

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""

    # Basic performance
    total_return_pct: float
    annualized_return_pct: float
    total_reward: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    calmar_ratio: float

    # Trading metrics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_profit: float
    total_fees: float

    # Comparison
    vs_buy_hold_pct: Optional[float] = None
    vs_random_pct: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_return_pct': self.total_return_pct,
            'annualized_return_pct': self.annualized_return_pct,
            'total_reward': self.total_reward,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_drawdown_duration': self.max_drawdown_duration,
            'calmar_ratio': self.calmar_ratio,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_profit': self.avg_trade_profit,
            'total_fees': self.total_fees,
            'vs_buy_hold_pct': self.vs_buy_hold_pct,
            'vs_random_pct': self.vs_random_pct,
        }

    def __str__(self) -> str:
        """Pretty print metrics."""
        lines = [
            "=" * 50,
            "EVALUATION METRICS",
            "=" * 50,
            "",
            "Performance:",
            f"  Total Return:      {self.total_return_pct:+.2f}%",
            f"  Annualized Return: {self.annualized_return_pct:+.2f}%",
            f"  Total Reward:      {self.total_reward:.2f}",
            "",
            "Risk-Adjusted:",
            f"  Sharpe Ratio:      {self.sharpe_ratio:.3f}",
            f"  Sortino Ratio:     {self.sortino_ratio:.3f}",
            f"  Calmar Ratio:      {self.calmar_ratio:.3f}",
            f"  Max Drawdown:      {self.max_drawdown_pct:.2f}%",
            f"  Drawdown Duration: {self.max_drawdown_duration} steps",
            "",
            "Trading:",
            f"  Num Trades:        {self.num_trades}",
            f"  Win Rate:          {self.win_rate:.1%}",
            f"  Profit Factor:     {self.profit_factor:.2f}",
            f"  Avg Trade Profit:  ${self.avg_trade_profit:.2f}",
            f"  Total Fees:        ${self.total_fees:.2f}",
        ]

        if self.vs_buy_hold_pct is not None:
            lines.extend([
                "",
                "vs Baselines:",
                f"  vs Buy&Hold:       {self.vs_buy_hold_pct:+.2f}%",
            ])

        if self.vs_random_pct is not None:
            lines.append(f"  vs Random:         {self.vs_random_pct:+.2f}%")

        lines.append("=" * 50)
        return "\n".join(lines)


def compute_metrics(
    portfolio_values: np.ndarray,
    trades: List[dict],
    total_reward: float,
    starting_value: float,
    periods_per_year: int = 525600
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics from episode data.

    Args:
        portfolio_values: Array of portfolio values over time
        trades: List of trade dictionaries
        total_reward: Total RL reward
        starting_value: Initial portfolio value
        periods_per_year: Periods per year for annualization

    Returns:
        EvaluationMetrics object
    """
    portfolio_values = np.array(portfolio_values)
    returns = calculate_returns(portfolio_values)

    # Basic performance
    total_return = (portfolio_values[-1] - starting_value) / starting_value * 100

    num_periods = len(portfolio_values)
    years = num_periods / periods_per_year
    if years > 0 and portfolio_values[-1] / starting_value > 0:
        annualized = ((portfolio_values[-1] / starting_value) ** (1 / years) - 1) * 100
    else:
        annualized = total_return

    # Risk metrics
    sharpe = calculate_sharpe_ratio(returns, periods_per_year=periods_per_year)
    sortino = calculate_sortino_ratio(returns, periods_per_year=periods_per_year)
    max_dd, dd_duration = calculate_max_drawdown(portfolio_values)
    calmar = calculate_calmar_ratio(portfolio_values, periods_per_year=periods_per_year)

    # Trading metrics
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)

    sell_trades = [t for t in trades if t.get('type') == 'SELL']
    avg_profit = np.mean([t.get('profit', 0) for t in sell_trades]) if sell_trades else 0
    total_fees = sum(t.get('fee', 0) for t in trades)

    return EvaluationMetrics(
        total_return_pct=total_return,
        annualized_return_pct=annualized,
        total_reward=total_reward,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_dd,
        max_drawdown_duration=dd_duration,
        calmar_ratio=calmar,
        num_trades=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_profit=avg_profit,
        total_fees=total_fees,
    )
