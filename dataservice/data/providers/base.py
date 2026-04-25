"""Abstract base class for data providers"""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable
from datetime import datetime
import pandas as pd


class DataProvider(ABC):
    """Abstract base class for OHLCV data providers"""

    @abstractmethod
    async def get_historical(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Returns DataFrame with columns:
        - Timestamp (datetime)
        - Open (float)
        - High (float)
        - Low (float)
        - Close (float)
        - Volume (float)
        """
        pass

    @abstractmethod
    async def get_latest(self, symbol: str, timeframe: str) -> Optional[dict]:
        """Get the most recent candle"""
        pass

    @abstractmethod
    def get_supported_timeframes(self) -> List[str]:
        """Return list of supported timeframes"""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Return list of supported symbols"""
        pass

    async def subscribe_realtime(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable
    ) -> None:
        """Subscribe to real-time updates (optional)"""
        raise NotImplementedError("Real-time streaming not supported")

    async def unsubscribe_realtime(self, symbol: str) -> None:
        """Unsubscribe from real-time updates"""
        pass
