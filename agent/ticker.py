"""
Market Data Ticker - Multi-Timeframe Trading System

Provides synchronized access to multiple timeframe data with configurable weights.
This mimics how real traders analyze markets:
- Scalpers: Focus on 1m/5m with context from 15m
- Day traders: Focus on 15m/1h with context from 4h
- Swing traders: Focus on 4h/1d with context from weekly

All timeframes are pre-aggregated and cached for maximum performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import pickle
import ta
from indicators import Indicators


class Ticker:
    """
    Manages multiple timeframe data with synchronized indices.

    Supports: 1m, 5m, 15m, 30m, 1h, 4h, 1d candles
    """

    TIMEFRAME_MINUTES = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }

    def __init__(self, filename, timeframe_weights, use_recent_data=False, recent_data_rows=None, random_start=False, episode_length=None):
        """
        Initialize multi-timeframe ticker.

        Args:
            filename: Path to 1-minute OHLCV data
            timeframe_weights: Dict of {timeframe: weight} (must sum to 1.0)
            use_recent_data: Only use recent N rows
            recent_data_rows: Number of recent rows to use
            random_start: Randomize starting position on reset_index()
            episode_length: Episode length for calculating valid random range
        """
        self._filename = Path(filename)
        self._weights = self._validate_weights(timeframe_weights)
        self._use_recent_data = use_recent_data
        self._recent_data_rows = recent_data_rows
        self._random_start = random_start
        self._episode_length = episode_length

        # Determine primary timeframe (highest weight)
        self._primary_tf = max(self._weights.items(), key=lambda x: x[1])[0]

        print(f"\n{'='*60}")
        print(f"Multi-Timeframe Ticker Initialization")
        print(f"{'='*60}")
        print(f"Primary Timeframe: {self._primary_tf} (weight: {self._weights[self._primary_tf]:.2f})")
        print(f"Active Timeframes: {[tf for tf, w in self._weights.items() if w > 0]}")

        # Load and process all data
        self.reset()

    def _validate_weights(self, weights):
        """Ensure weights are valid and sum to 1.0"""
        if not weights:
            raise ValueError("Timeframe weights cannot be empty")

        # Check all timeframes are valid
        for tf in weights.keys():
            if tf not in self.TIMEFRAME_MINUTES:
                raise ValueError(f"Invalid timeframe: {tf}. Must be one of {list(self.TIMEFRAME_MINUTES.keys())}")

        # Check weights sum to 1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")

        return weights

    def _get_cache_path(self):
        """Generate cache path with dataset hash folder and config hash filename"""
        # Hash the dataset (file mod time + size)
        stat = self._filename.stat()
        dataset_hash = hashlib.md5(f"{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()[:12]

        # Hash the config (recent data settings + weights)
        config_str = f"recent_{self._use_recent_data}_{self._recent_data_rows}_weights_{sorted(self._weights.items())}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

        # Create cache directory structure: data/.cache/{dataset_hash}/
        cache_dir = self._filename.parent / ".cache" / dataset_hash
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache filename includes config hash
        cache_file = cache_dir / f"mtf_{config_hash}.pkl"
        return cache_file

    def reset(self):
        """Load and prepare all timeframe data"""
        # Check for cache
        cache_file = self._get_cache_path()

        if cache_file.exists():
            print(f"\n[Cache] Loading from {cache_file.relative_to(self._filename.parent)}...")
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.timeframes = cache_data['timeframes']
                    self._start_idx = cache_data['start_idx']
                    self._idx = self._start_idx
                    self.candles_per_day = cache_data['candles_per_day']
                print(f"[Cache] Loaded successfully!")
                print(f"\n{'='*60}")
                print(f"Multi-Timeframe Ticker Ready!")
                print(f"{'='*60}\n")
                return
            except Exception as e:
                print(f"[Cache] Failed to load cache: {e}")
                print(f"[Cache] Regenerating...")

        # Load raw 1-minute data
        print(f"\n[1/4] Loading 1-minute data from {self._filename}...")
        raw_data = self._load_raw_data()

        # Pre-aggregate to all timeframes
        print(f"\n[2/4] Aggregating to all timeframes...")
        self._aggregate_all_timeframes(raw_data)

        # Pre-compute indicators for each timeframe
        print(f"\n[3/4] Computing indicators for each timeframe...")
        self._precompute_all_indicators()

        # Setup indices
        print(f"\n[4/4] Aligning timeframe indices...")
        self._setup_indices()

        # Save cache
        cache_file = self._get_cache_path()
        print(f"\n[Cache] Saving to {cache_file.relative_to(self._filename.parent)}...")
        try:
            cache_data = {
                'timeframes': self.timeframes,
                'start_idx': self._start_idx,
                'candles_per_day': self.candles_per_day
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"[Cache] Saved successfully!")
        except Exception as e:
            print(f"[Cache] Failed to save: {e}")

        print(f"\n{'='*60}")
        print(f"Multi-Timeframe Ticker Ready!")
        print(f"{'='*60}\n")

    def _load_raw_data(self):
        """Load 1-minute base data"""
        if not self._filename.exists():
            raise FileNotFoundError(f"Data file not found: {self._filename}")

        # Load data
        if self._use_recent_data and self._recent_data_rows:
            total_rows = sum(1 for _ in open(self._filename)) - 1
            skip_rows = max(0, total_rows - self._recent_data_rows)
            if skip_rows > 0:
                df = pd.read_csv(self._filename, skiprows=range(1, skip_rows + 1))
            else:
                df = pd.read_csv(self._filename)
            print(f"  Loaded {len(df):,} rows (skipped {skip_rows:,} old rows)")
        else:
            df = pd.read_csv(self._filename)
            print(f"  Loaded {len(df):,} rows")

        # Convert Unix timestamp to datetime if needed
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

        # Normalize columns
        if 'Volume' in df.columns and 'Volume_(Currency)' not in df.columns:
            df['Volume_(Currency)'] = df['Volume']

        if 'Open' not in df.columns:
            df['Open'] = df['Close']

        # Verify required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume_(Currency)']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        return df

    def _aggregate_candles(self, df, minutes):
        """Aggregate 1-min candles to target timeframe"""
        if minutes == 1:
            return df.copy()

        aggregated = []
        for i in range(0, len(df), minutes):
            chunk = df.iloc[i:i+minutes]
            if len(chunk) == 0:
                continue

            candle = {
                'Open': chunk['Open'].iloc[0],
                'High': chunk['High'].max(),
                'Low': chunk['Low'].min(),
                'Close': chunk['Close'].iloc[-1],
                'Volume_(Currency)': chunk['Volume_(Currency)'].sum()
            }

            # Preserve timestamp if available
            if 'Timestamp' in chunk.columns:
                candle['Timestamp'] = chunk['Timestamp'].iloc[0]

            aggregated.append(candle)

        return pd.DataFrame(aggregated)

    def _aggregate_all_timeframes(self, raw_data):
        """Pre-aggregate data to all timeframes"""
        self.timeframes = {}

        for tf, minutes in self.TIMEFRAME_MINUTES.items():
            print(f"  Aggregating {tf}...", end=' ')
            self.timeframes[tf] = self._aggregate_candles(raw_data, minutes)
            print(f"{len(self.timeframes[tf]):,} candles")

    def _precompute_indicators_for_timeframe(self, df, tf_name):
        """Compute all technical indicators for a timeframe"""
        # Price-based features
        df['return_1'] = df['Close'].pct_change(1)
        df['return_5'] = df['Close'].pct_change(5)
        df['return_10'] = df['Close'].pct_change(10)
        df['return_20'] = df['Close'].pct_change(20)

        # Volatility
        df['volatility_10'] = df['Close'].pct_change().rolling(10).std()
        df['volatility_30'] = df['Close'].pct_change().rolling(30).std()

        # Moving averages
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()

        # Technical indicators
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['Close'])
        df['bb_high'] = ta.volatility.bollinger_hband(df['Close'])
        df['bb_low'] = ta.volatility.bollinger_lband(df['Close'])
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # Volume
        df['volume_sma'] = df['Volume_(Currency)'].rolling(20).mean()

        return df

    def _precompute_all_indicators(self):
        """Compute indicators for all timeframes"""
        for tf in self.timeframes.keys():
            print(f"  Computing {tf} indicators...", end=' ')
            self.timeframes[tf] = self._precompute_indicators_for_timeframe(self.timeframes[tf], tf)
            print(f"Done")

    def _setup_indices(self):
        """Initialize index tracking for all timeframes"""
        # Calculate minimum starting index where ALL timeframes have enough data
        min_window_size = 100  # Minimum candles needed for indicators

        # For each active timeframe, calculate required primary index
        required_primary_indices = []

        for tf, weight in self._weights.items():
            if weight <= 0:
                continue

            # How many candles does this timeframe need?
            tf_min_candles = min_window_size

            # Calculate what primary index gives us enough data in this timeframe
            primary_minutes = self.TIMEFRAME_MINUTES[self._primary_tf]
            target_minutes = self.TIMEFRAME_MINUTES[tf]
            ratio = target_minutes / primary_minutes

            if ratio >= 1:
                # Target is coarser - need more primary candles
                required_primary_idx = int(tf_min_candles * ratio)
            else:
                # Target is finer - need fewer primary candles
                required_primary_idx = int(tf_min_candles / (1/ratio))

            required_primary_indices.append(required_primary_idx)
            print(f"  {tf}: needs primary_idx >= {required_primary_idx} (ratio={ratio:.2f})")

        # Start at the maximum required index to ensure ALL timeframes have data
        self._start_idx = max(required_primary_indices) if required_primary_indices else 100
        self._idx = self._start_idx

        # Calculate candles per day for each timeframe
        self.candles_per_day = {
            tf: 1440 // minutes for tf, minutes in self.TIMEFRAME_MINUTES.items()
        }

        print(f"\n  Primary TF: {self._primary_tf}")
        print(f"  Starting index: {self._start_idx}")
        print(f"  Candles available: {len(self.timeframes[self._primary_tf]) - self._start_idx}")

    def _get_aligned_index(self, target_tf):
        """Get corresponding index in target timeframe"""
        primary_minutes = self.TIMEFRAME_MINUTES[self._primary_tf]
        target_minutes = self.TIMEFRAME_MINUTES[target_tf]

        # How many target candles fit in one primary candle?
        ratio = target_minutes / primary_minutes

        # If ratio < 1: target has MORE candles (finer granularity)
        # If ratio > 1: target has FEWER candles (coarser granularity)

        if ratio >= 1:
            # Target is coarser (e.g., primary=15m, target=1h)
            # 1h candle covers 4 15m candles
            aligned_idx = int(self._idx / ratio)
        else:
            # Target is finer (e.g., primary=15m, target=5m)
            # One 15m candle = 3 5m candles, so multiply
            aligned_idx = int(self._idx * (1/ratio))

        # Ensure index is valid
        max_idx = len(self.timeframes[target_tf]) - 1
        return min(aligned_idx, max_idx)

    def get_current_state(self, position_info=None, portfolio_value=None):
        """
        Get current multi-timeframe state vector.

        Returns weighted combination of all timeframe indicators.
        """
        state_vectors = []

        for tf, weight in self._weights.items():
            if weight <= 0:
                continue

            # Get aligned index for this timeframe
            tf_idx = self._get_aligned_index(tf)

            # Create a dataframe window for this timeframe
            tf_data = self.timeframes[tf]
            start_idx = max(0, tf_idx - 100)
            window = tf_data.iloc[start_idx:tf_idx+1]

            # Calculate indicators
            indicator_calc = Indicators(normalize=True)
            indicators = indicator_calc.calculate_all_indicators(
                window,
                position_info,
                portfolio_value
            )

            if indicators is not None:
                # Weight the indicators
                weighted = indicators * weight
                state_vectors.append(weighted)
            else:
                # This should NEVER happen if _setup_indices worked correctly
                raise RuntimeError(
                    f"Insufficient data for {tf} at primary_idx={self._idx}, "
                    f"tf_idx={tf_idx}, window_size={len(window)}. "
                    f"This indicates a bug in _setup_indices or _get_aligned_index."
                )

        if not state_vectors:
            return None

        # Concatenate all weighted timeframe states
        return np.concatenate(state_vectors)

    def get_next(self):
        """
        Get next candle from primary timeframe.

        Returns:
            (candle, is_complete) tuple
        """
        if self._idx >= len(self.timeframes[self._primary_tf]):
            return pd.Series(), True

        candle = self.timeframes[self._primary_tf].iloc[self._idx]
        self._idx += 1

        return candle, False

    def get_bulk(self, count):
        """Get multiple candles from primary timeframe"""
        candles = []
        for _ in range(count):
            candle, done = self.get_next()
            if done:
                break
            candles.append((candle, done))
        return candles

    def reset_index(self):
        """Reset to start of data (or random position if random_start enabled)"""
        if self._random_start and self._episode_length:
            # Random start within valid range (leaving room for full episode)
            max_start = len(self.timeframes[self._primary_tf]) - self._episode_length
            if max_start > self._start_idx:
                old_idx = self._idx
                self._idx = np.random.randint(self._start_idx, max_start)
                # Only log on actual resets (not initial setup)
                if hasattr(self, '_reset_count'):
                    self._reset_count += 1
                    if self._reset_count % 10 == 0:  # Log every 10th episode
                        print(f"[Random Start] Episode {self._reset_count}: Starting at {self._primary_tf} candle {self._idx} (out of {len(self.timeframes[self._primary_tf])})")
                else:
                    self._reset_count = 0
            else:
                self._idx = self._start_idx
        else:
            self._idx = self._start_idx

    def __len__(self):
        """Length is based on primary timeframe"""
        return len(self.timeframes[self._primary_tf])

    @property
    def current_index(self):
        return self._idx

    @property
    def remaining_ticks(self):
        return len(self.timeframes[self._primary_tf]) - self._idx

    def get_current_timestamp(self):
        """Get timestamp of current candle if available"""
        if self._idx >= len(self.timeframes[self._primary_tf]):
            return None
        candle = self.timeframes[self._primary_tf].iloc[self._idx]
        return candle.get('Timestamp', None)

    @property
    def num_features(self):
        """Calculate total number of features in state vector"""
        # 59 features per timeframe × number of active timeframes
        # (58 base + 1 cash_ratio)
        active_tfs = sum(1 for w in self._weights.values() if w > 0)
        return 59 * active_tfs
