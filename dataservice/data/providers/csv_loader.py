"""CSV file data provider for local/fallback data"""

from typing import List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
import logging

from .base import DataProvider

logger = logging.getLogger(__name__)


class CSVProvider(DataProvider):
    """Load OHLCV data from local CSV files"""

    TIMEFRAME_MINUTES = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440
    }

    def __init__(self, data_path: str = "/app/data"):
        self.data_path = Path(data_path)
        self._cache: dict = {}
        self._symbol_files = {
            'BTCUSDT': 'btcusd_1-min_data.csv',
            'BTC': 'btcusd_1-min_data.csv',
        }

    def _get_csv_path(self, symbol: str) -> Optional[Path]:
        """Get CSV file path for symbol"""
        filename = self._symbol_files.get(symbol.upper())
        if filename:
            path = self.data_path / filename
            if path.exists():
                return path

        # Try direct filename match
        for ext in ['.csv', '_1-min_data.csv']:
            path = self.data_path / f"{symbol.lower()}{ext}"
            if path.exists():
                return path

        return None

    def _load_csv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load and cache CSV data"""
        if symbol in self._cache:
            return self._cache[symbol]

        path = self._get_csv_path(symbol)
        if not path:
            logger.warning(f"No CSV file found for {symbol}")
            return None

        logger.info(f"Loading CSV: {path}")

        try:
            df = pd.read_csv(path)

            # Normalize column names
            col_map = {
                'timestamp': 'Timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'Volume_(Currency)': 'Volume',
            }
            df.columns = [col_map.get(c.lower(), c) for c in df.columns]

            # Convert timestamp
            if df['Timestamp'].dtype in ['float64', 'int64']:
                # Unix timestamp
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
            else:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])

            # Ensure numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Sort by timestamp
            df = df.sort_values('Timestamp').reset_index(drop=True)

            self._cache[symbol] = df
            logger.info(f"Loaded {len(df)} candles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV for {symbol}: {e}")
            return None

    def _resample_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """Resample 1m data to higher timeframes"""
        if timeframe == '1m':
            return df

        minutes = self.TIMEFRAME_MINUTES.get(timeframe)
        if not minutes:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return df

        df = df.set_index('Timestamp')
        resampled = df.resample(f'{minutes}min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        return resampled.reset_index()

    async def get_historical(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get historical OHLCV data from CSV"""
        df = self._load_csv(symbol)
        if df is None:
            return pd.DataFrame()

        # Resample to requested timeframe
        df = self._resample_timeframe(df.copy(), timeframe)

        # Apply time filters
        if start:
            df = df[df['Timestamp'] >= start]
        if end:
            df = df[df['Timestamp'] <= end]

        # Apply limit (most recent)
        if limit and len(df) > limit:
            df = df.tail(limit)

        return df.reset_index(drop=True)

    async def get_latest(self, symbol: str, timeframe: str) -> Optional[dict]:
        """Get most recent candle"""
        df = await self.get_historical(symbol, timeframe, limit=1)
        if df.empty:
            return None

        row = df.iloc[-1]
        return {
            'Timestamp': row['Timestamp'].isoformat(),
            'Open': float(row['Open']),
            'High': float(row['High']),
            'Low': float(row['Low']),
            'Close': float(row['Close']),
            'Volume': float(row['Volume'])
        }

    def get_supported_timeframes(self) -> List[str]:
        """Return supported timeframes"""
        return list(self.TIMEFRAME_MINUTES.keys())

    def get_supported_symbols(self) -> List[str]:
        """Return symbols with available CSV files"""
        symbols = []
        for symbol, filename in self._symbol_files.items():
            if (self.data_path / filename).exists():
                symbols.append(symbol)
        return symbols
