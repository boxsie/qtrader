"""Binance API data provider"""

from typing import List, Optional, Callable
from datetime import datetime, timedelta
import asyncio
import logging

import aiohttp
import pandas as pd

from .base import DataProvider

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, max_requests: int = 1200, period: int = 60):
        self.max_requests = max_requests
        self.period = period
        self.requests: list = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request slot is available"""
        async with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.period)

            # Remove old requests
            self.requests = [r for r in self.requests if r > cutoff]

            # Wait if at capacity
            if len(self.requests) >= self.max_requests:
                wait_time = (self.requests[0] - cutoff).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            self.requests.append(now)


class BinanceProvider(DataProvider):
    """Binance API data provider"""

    BASE_URL = "https://api.binance.com/api/v3"
    WS_URL = "wss://stream.binance.com:9443/ws"

    TIMEFRAME_MAP = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
    }

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self._rate_limiter = RateLimiter(max_requests=1200, period=60)
        self._ws_connections: dict = {}

    async def get_historical(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch historical klines from Binance"""

        await self._rate_limiter.acquire()

        binance_tf = self.TIMEFRAME_MAP.get(timeframe, timeframe)

        params = {
            'symbol': symbol.upper().replace('/', ''),
            'interval': binance_tf,
            'limit': min(limit, 1000)  # Binance max is 1000
        }

        if start:
            params['startTime'] = int(start.timestamp() * 1000)
        if end:
            params['endTime'] = int(end.timestamp() * 1000)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/klines",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"Binance API error: {error}")
                        return pd.DataFrame()

                    data = await response.json()

            if not data:
                return pd.DataFrame()

            # Parse response
            df = pd.DataFrame(data, columns=[
                'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'CloseTime', 'QuoteVolume', 'Trades', 'TakerBuyBase',
                'TakerBuyQuote', 'Ignore'
            ])

            # Convert types
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)

            return df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

        except aiohttp.ClientError as e:
            logger.error(f"Binance connection error: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching from Binance: {e}")
            return pd.DataFrame()

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

    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current ticker price"""
        await self._rate_limiter.acquire()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/ticker/price",
                    params={'symbol': symbol.upper()},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data['price'])
        except Exception as e:
            logger.error(f"Error getting ticker price: {e}")
        return None

    def get_supported_timeframes(self) -> List[str]:
        """Return supported timeframes"""
        return list(self.TIMEFRAME_MAP.keys())

    def get_supported_symbols(self) -> List[str]:
        """Return commonly traded symbols"""
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
            'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT'
        ]

    async def subscribe_realtime(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable
    ) -> None:
        """Subscribe to WebSocket kline stream"""
        binance_tf = self.TIMEFRAME_MAP.get(timeframe, timeframe)
        stream_name = f"{symbol.lower()}@kline_{binance_tf}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"{self.WS_URL}/{stream_name}"
                ) as ws:
                    self._ws_connections[symbol] = ws
                    logger.info(f"Subscribed to {stream_name}")

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = msg.json()
                            candle = self._parse_ws_kline(data)
                            await callback(candle)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self._ws_connections.pop(symbol, None)

    def _parse_ws_kline(self, data: dict) -> dict:
        """Parse WebSocket kline message"""
        k = data.get('k', {})
        return {
            'Timestamp': datetime.fromtimestamp(k.get('t', 0) / 1000),
            'Open': float(k.get('o', 0)),
            'High': float(k.get('h', 0)),
            'Low': float(k.get('l', 0)),
            'Close': float(k.get('c', 0)),
            'Volume': float(k.get('v', 0)),
            'is_closed': k.get('x', False)
        }

    async def unsubscribe_realtime(self, symbol: str) -> None:
        """Close WebSocket connection"""
        ws = self._ws_connections.pop(symbol, None)
        if ws:
            await ws.close()
