"""
Evaluation module for QTrader.

Provides metrics and comparison tools for evaluating
trading strategies against baselines.
"""

from .metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    EvaluationMetrics,
    compute_metrics,
)
from .comparison import StrategyComparison, quick_compare
from .data_split import (
    DataSplit,
    MarketRegime,
    TimeSeriesSplitter,
    RegimeAwareSplitter,
    label_market_regime,
)

__all__ = [
    # Metrics
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'EvaluationMetrics',
    'compute_metrics',
    # Comparison
    'StrategyComparison',
    'quick_compare',
    # Data splitting
    'DataSplit',
    'MarketRegime',
    'TimeSeriesSplitter',
    'RegimeAwareSplitter',
    'label_market_regime',
]
