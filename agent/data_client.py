"""
Data Service Client for QTrader Agent

Provides data fetching with caching and fallback to local CSV.
Integrates sentiment features into state vector.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try async imports
try:
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

# Sync fallback
import requests


class DataServiceClient:
    """
    Client for QTrader Data Service.

    Features:
    - Async HTTP requests to data service
    - In-memory caching with TTL
    - Graceful fallback when service unavailable
    - Sentiment features for state vector
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8766",
        cache_ttl: int = 300,  # 5 minutes
        timeout: float = 10.0,
        enabled: bool = True
    ):
        self.service_url = service_url.rstrip('/')
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.enabled = enabled

        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._connected = False
        self._last_check = None

    def check_connection(self) -> bool:
        """Check if data service is available (sync)"""
        if not self.enabled:
            return False

        # Don't spam health checks
        if self._last_check and (datetime.now() - self._last_check).seconds < 30:
            return self._connected

        try:
            response = requests.get(
                f"{self.service_url}/health",
                timeout=5
            )
            self._connected = response.status_code == 200
        except Exception:
            self._connected = False

        self._last_check = datetime.now()
        return self._connected

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache:
            return False
        ts = self._cache_timestamps.get(key, datetime.min)
        return (datetime.now() - ts).total_seconds() < self.cache_ttl

    def _update_cache(self, key: str, data: Any):
        """Update cache with new data"""
        self._cache[key] = data
        self._cache_timestamps[key] = datetime.now()

    def get_ohlcv_sync(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> Optional[List[Dict]]:
        """
        Fetch OHLCV data synchronously.

        Returns list of candle dicts or None if unavailable.
        """
        if not self.enabled:
            return None

        cache_key = f"ohlcv_{symbol}_{timeframe}_{limit}"

        # Check cache
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            response = requests.get(
                f"{self.service_url}/api/data/ohlcv/{symbol}",
                params={"timeframe": timeframe, "limit": limit},
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                candles = data.get('candles', [])
                self._update_cache(cache_key, candles)
                return candles

        except Exception as e:
            logger.debug(f"Data service error: {e}")

        return None

    def get_current_price_sync(self, symbol: str) -> Optional[float]:
        """Get current price synchronously"""
        if not self.enabled:
            return None

        cache_key = f"price_{symbol}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            response = requests.get(
                f"{self.service_url}/api/data/ohlcv/{symbol}/price",
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                price = data.get('price')
                if price:
                    self._update_cache(cache_key, price)
                    return price

        except Exception as e:
            logger.debug(f"Price fetch error: {e}")

        return None

    def get_sentiment_sync(self, symbol: str) -> Dict[str, Any]:
        """
        Get sentiment data synchronously.

        Returns dict with sentiment features.
        """
        if not self.enabled:
            return self._default_sentiment()

        cache_key = f"sentiment_{symbol}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            response = requests.get(
                f"{self.service_url}/api/news/sentiment/{symbol}",
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                self._update_cache(cache_key, data)
                return data

        except Exception as e:
            logger.debug(f"Sentiment fetch error: {e}")

        return self._default_sentiment()

    def get_sentiment_features_sync(self, symbol: str) -> Dict[str, Any]:
        """
        Get sentiment features formatted for state vector.

        Returns dict with feature_vector array.
        """
        if not self.enabled:
            return self._default_sentiment_features()

        cache_key = f"sentiment_features_{symbol}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            response = requests.get(
                f"{self.service_url}/api/news/sentiment/{symbol}/features",
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                self._update_cache(cache_key, data)
                return data

        except Exception as e:
            logger.debug(f"Sentiment features error: {e}")

        return self._default_sentiment_features()

    def get_llm_bias_sync(self, symbol: str) -> Dict[str, Any]:
        """Get LLM market bias score"""
        if not self.enabled:
            return {'bias_score': 0.0, 'confidence': 0.5}

        cache_key = f"llm_bias_{symbol}"

        # LLM bias cached longer (15 min) since it's expensive
        if cache_key in self._cache:
            ts = self._cache_timestamps.get(cache_key, datetime.min)
            if (datetime.now() - ts).total_seconds() < 900:  # 15 min
                return self._cache[cache_key]

        try:
            response = requests.get(
                f"{self.service_url}/api/analysis/bias/{symbol}",
                timeout=30  # LLM can be slow
            )

            if response.status_code == 200:
                data = response.json()
                self._update_cache(cache_key, data)
                return data

        except Exception as e:
            logger.debug(f"LLM bias error: {e}")

        return {'bias_score': 0.0, 'confidence': 0.5}

    def get_state_features(self, symbol: str) -> np.ndarray:
        """
        Get sentiment features as numpy array for state vector.

        Returns array of 5 features:
        - news_sentiment_score (-1 to 1)
        - social_sentiment_score (-1 to 1)
        - sentiment_confidence (0 to 1)
        - news_volume_normalized (0 to 1)
        - llm_bias_score (-1 to 1)
        """
        features = self.get_sentiment_features_sync(symbol)
        feature_vector = features.get('feature_vector', [0.0, 0.0, 0.5, 0.0, 0.0])
        return np.array(feature_vector, dtype=np.float32)

    def _default_sentiment(self) -> Dict[str, Any]:
        """Default sentiment when service unavailable"""
        return {
            'symbol': '',
            'overall_sentiment': 'neutral',
            'sentiment_score': 0.0,
            'confidence': 0.5,
            'news_count': 0,
            'social_count': 0
        }

    def _default_sentiment_features(self) -> Dict[str, Any]:
        """Default sentiment features"""
        return {
            'features': {
                'news_sentiment_score': 0.0,
                'social_sentiment_score': 0.0,
                'sentiment_confidence': 0.5,
                'news_volume_24h': 0.0,
                'llm_bias_score': 0.0
            },
            'feature_vector': [0.0, 0.0, 0.5, 0.0, 0.0]
        }

    # Async methods (for future use)
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> Optional[List[Dict]]:
        """Async OHLCV fetch"""
        if not ASYNC_AVAILABLE or not self.enabled:
            return self.get_ohlcv_sync(symbol, timeframe, limit)

        cache_key = f"ohlcv_{symbol}_{timeframe}_{limit}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.service_url}/api/data/ohlcv/{symbol}",
                    params={"timeframe": timeframe, "limit": limit},
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        candles = data.get('candles', [])
                        self._update_cache(cache_key, candles)
                        return candles

        except Exception as e:
            logger.debug(f"Async data service error: {e}")

        return None

    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Async sentiment fetch"""
        if not ASYNC_AVAILABLE or not self.enabled:
            return self.get_sentiment_sync(symbol)

        cache_key = f"sentiment_{symbol}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.service_url}/api/news/sentiment/{symbol}",
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._update_cache(cache_key, data)
                        return data

        except Exception as e:
            logger.debug(f"Async sentiment error: {e}")

        return self._default_sentiment()
