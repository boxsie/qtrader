"""
Train/Validation/Test data splitting for proper ML evaluation.

CRITICAL: Without proper time-based splits, you cannot tell
if your model is learning real patterns or just memorizing history.

The splits MUST be chronological:
- Train: Oldest data (e.g., 2020-2022)
- Validation: Middle data (e.g., Jan-Jun 2023)
- Test: Newest data (e.g., Jul-Dec 2023) - NEVER TRAIN ON THIS

Using random splits for time series is WRONG because:
1. Future data leaks into training (look-ahead bias)
2. Model learns spurious correlations from shuffling
3. Results are unrealistically optimistic
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from enum import Enum


class DataSplit(Enum):
    """Data split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ALL = "all"  # For baselines that need full data


class MarketRegime(Enum):
    """Market regime labels for regime-aware evaluation."""
    BULL = "bull"       # >20% gain in 30 days
    BEAR = "bear"       # >20% loss in 30 days
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class TimeSeriesSplitter:
    """
    Manages chronological train/validation/test splits for time series data.

    This is the most important class for preventing overfitting.
    """

    def __init__(
        self,
        data_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """
        Initialize splitter.

        Args:
            data_path: Path to OHLCV CSV file
            train_ratio: Fraction of data for training (oldest)
            val_ratio: Fraction for validation (middle)
            test_ratio: Fraction for testing (newest)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"

        self.data_path = Path(data_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Load data to get split indices
        self._load_metadata()

    def _load_metadata(self):
        """Load data and compute split boundaries."""
        # Just get the length without loading all data
        self.total_rows = sum(1 for _ in open(self.data_path)) - 1  # -1 for header

        # Compute split indices
        self.train_end = int(self.total_rows * self.train_ratio)
        self.val_end = int(self.total_rows * (self.train_ratio + self.val_ratio))
        self.test_end = self.total_rows

        print(f"\n{'='*60}")
        print(f"Time Series Split Configuration")
        print(f"{'='*60}")
        print(f"Total rows: {self.total_rows:,}")
        print(f"Train: rows 0 - {self.train_end:,} ({self.train_ratio:.0%})")
        print(f"Val:   rows {self.train_end:,} - {self.val_end:,} ({self.val_ratio:.0%})")
        print(f"Test:  rows {self.val_end:,} - {self.test_end:,} ({self.test_ratio:.0%})")
        print(f"{'='*60}\n")

    def get_split_config(self, split: DataSplit) -> Dict:
        """
        Get configuration for loading a specific split.

        Returns dict with 'skip_rows' and 'n_rows' for pd.read_csv.
        """
        if split == DataSplit.ALL:
            return {
                'skip_rows': 0,
                'n_rows': None,  # All rows
                'start_row': 0,
                'end_row': self.total_rows
            }

        if split == DataSplit.TRAIN:
            return {
                'skip_rows': 0,
                'n_rows': self.train_end,
                'start_row': 0,
                'end_row': self.train_end
            }

        elif split == DataSplit.VALIDATION:
            return {
                'skip_rows': self.train_end,
                'n_rows': self.val_end - self.train_end,
                'start_row': self.train_end,
                'end_row': self.val_end
            }

        elif split == DataSplit.TEST:
            return {
                'skip_rows': self.val_end,
                'n_rows': self.test_end - self.val_end,
                'start_row': self.val_end,
                'end_row': self.test_end
            }

        raise ValueError(f"Unknown split: {split}")

    def create_split_file(self, split: DataSplit, output_dir: str = None) -> Path:
        """
        Create a separate CSV file for a specific split.

        This is useful for caching - each split can have its own cached indicators.

        Args:
            split: Which split to create
            output_dir: Output directory (default: same as source)

        Returns:
            Path to created file
        """
        if output_dir is None:
            output_dir = self.data_path.parent

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config = self.get_split_config(split)

        # Read the split
        df = pd.read_csv(
            self.data_path,
            skiprows=range(1, config['skip_rows'] + 1) if config['skip_rows'] > 0 else None,
            nrows=config['n_rows']
        )

        # Save to new file
        split_file = output_dir / f"{self.data_path.stem}_{split.value}.csv"
        df.to_csv(split_file, index=False)

        print(f"Created {split.value} split: {split_file} ({len(df):,} rows)")
        return split_file

    def load_split(self, split: DataSplit) -> pd.DataFrame:
        """
        Load a specific split as DataFrame.

        Args:
            split: Which split to load

        Returns:
            DataFrame with OHLCV data
        """
        config = self.get_split_config(split)

        if config['skip_rows'] > 0:
            df = pd.read_csv(
                self.data_path,
                skiprows=range(1, config['skip_rows'] + 1),
                nrows=config['n_rows']
            )
        else:
            df = pd.read_csv(
                self.data_path,
                nrows=config['n_rows']
            )

        return df

    def get_date_ranges(self) -> Dict[str, Tuple[str, str]]:
        """
        Get date ranges for each split (if data has timestamps).

        Returns:
            Dict of split_name -> (start_date, end_date)
        """
        try:
            # Load minimal data to get timestamps
            df = pd.read_csv(self.data_path, usecols=['Timestamp'])

            if 'Timestamp' in df.columns:
                # Convert if Unix timestamp
                if df['Timestamp'].dtype in ['int64', 'float64']:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
                else:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

                return {
                    'train': (
                        str(df['Timestamp'].iloc[0]),
                        str(df['Timestamp'].iloc[self.train_end-1])
                    ),
                    'validation': (
                        str(df['Timestamp'].iloc[self.train_end]),
                        str(df['Timestamp'].iloc[self.val_end-1])
                    ),
                    'test': (
                        str(df['Timestamp'].iloc[self.val_end]),
                        str(df['Timestamp'].iloc[-1])
                    )
                }
        except Exception as e:
            print(f"Could not get date ranges: {e}")

        return {}


def label_market_regime(
    prices: pd.Series,
    window: int = 30 * 24 * 60,  # 30 days in minutes
    bull_threshold: float = 0.20,
    bear_threshold: float = -0.20
) -> pd.Series:
    """
    Label market regimes based on rolling returns.

    Args:
        prices: Series of prices
        window: Lookback window in periods
        bull_threshold: Return threshold for bull market (20%)
        bear_threshold: Return threshold for bear market (-20%)

    Returns:
        Series of MarketRegime labels
    """
    # Calculate rolling return over window
    rolling_return = prices.pct_change(periods=window)

    # Label regimes
    def get_regime(ret):
        if pd.isna(ret):
            return MarketRegime.UNKNOWN.value
        if ret >= bull_threshold:
            return MarketRegime.BULL.value
        if ret <= bear_threshold:
            return MarketRegime.BEAR.value
        return MarketRegime.SIDEWAYS.value

    return rolling_return.apply(get_regime)


def analyze_regime_distribution(labels: pd.Series) -> Dict[str, float]:
    """
    Analyze distribution of market regimes.

    Args:
        labels: Series of regime labels

    Returns:
        Dict of regime -> percentage
    """
    counts = labels.value_counts(normalize=True)
    return counts.to_dict()


class RegimeAwareSplitter(TimeSeriesSplitter):
    """
    Splitter that also tracks market regimes for regime-aware evaluation.

    This helps you understand:
    - Does the model work in bull markets?
    - Does it work in bear markets?
    - Does it work in sideways markets?
    """

    def __init__(
        self,
        data_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        regime_window: int = 30 * 24 * 60  # 30 days in minutes
    ):
        super().__init__(data_path, train_ratio, val_ratio, test_ratio)
        self.regime_window = regime_window
        self._regimes = None

    def compute_regimes(self) -> pd.Series:
        """Compute market regimes for all data."""
        if self._regimes is not None:
            return self._regimes

        # Load prices
        df = pd.read_csv(self.data_path, usecols=['Close'])
        prices = df['Close']

        # Label regimes
        self._regimes = label_market_regime(
            prices,
            window=self.regime_window
        )

        # Print distribution
        dist = analyze_regime_distribution(self._regimes)
        print("\nMarket Regime Distribution:")
        for regime, pct in dist.items():
            print(f"  {regime}: {pct:.1%}")

        return self._regimes

    def get_regime_distribution_by_split(self) -> Dict[str, Dict[str, float]]:
        """Get regime distribution for each split."""
        regimes = self.compute_regimes()

        return {
            'train': analyze_regime_distribution(regimes.iloc[:self.train_end]),
            'validation': analyze_regime_distribution(
                regimes.iloc[self.train_end:self.val_end]
            ),
            'test': analyze_regime_distribution(regimes.iloc[self.val_end:])
        }
