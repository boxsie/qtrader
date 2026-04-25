"""OHLCV data API routes"""

from typing import Optional, List
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import logging

from data.providers.binance import BinanceProvider
from data.providers.csv_loader import CSVProvider

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize providers
binance_provider = BinanceProvider()
csv_provider = CSVProvider()


class CandleResponse(BaseModel):
    """Single candle response"""
    Timestamp: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float


class OHLCVResponse(BaseModel):
    """OHLCV data response"""
    symbol: str
    timeframe: str
    count: int
    candles: List[dict]
    source: str


@router.get("/ohlcv/{symbol}", response_model=OHLCVResponse)
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query(default="1h", description="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d"),
    limit: int = Query(default=100, ge=1, le=1000, description="Number of candles"),
    start: Optional[str] = Query(default=None, description="Start time ISO format"),
    end: Optional[str] = Query(default=None, description="End time ISO format"),
    source: str = Query(default="auto", description="Data source: binance, csv, auto")
):
    """
    Get OHLCV candlestick data for a symbol.

    - **symbol**: Trading pair (e.g., BTCUSDT)
    - **timeframe**: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
    - **limit**: Max candles to return (1-1000)
    - **source**: Data source preference (auto tries Binance first, falls back to CSV)
    """
    # Parse dates
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    df = None
    used_source = ""

    # Try Binance first if auto or binance
    if source in ["auto", "binance"]:
        try:
            df = await binance_provider.get_historical(
                symbol=symbol,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
                limit=limit
            )
            if not df.empty:
                used_source = "binance"
                logger.info(f"Got {len(df)} candles from Binance")
        except Exception as e:
            logger.warning(f"Binance failed: {e}")

    # Fallback to CSV
    if (df is None or df.empty) and source in ["auto", "csv"]:
        try:
            df = await csv_provider.get_historical(
                symbol=symbol,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
                limit=limit
            )
            if not df.empty:
                used_source = "csv"
                logger.info(f"Got {len(df)} candles from CSV")
        except Exception as e:
            logger.warning(f"CSV failed: {e}")

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for {symbol} {timeframe}"
        )

    # Convert to response format
    candles = []
    for _, row in df.iterrows():
        candles.append({
            'Timestamp': row['Timestamp'].isoformat() if hasattr(row['Timestamp'], 'isoformat') else str(row['Timestamp']),
            'Open': float(row['Open']),
            'High': float(row['High']),
            'Low': float(row['Low']),
            'Close': float(row['Close']),
            'Volume': float(row['Volume'])
        })

    return OHLCVResponse(
        symbol=symbol.upper(),
        timeframe=timeframe,
        count=len(candles),
        candles=candles,
        source=used_source
    )


@router.get("/ohlcv/{symbol}/latest")
async def get_latest_candle(
    symbol: str,
    timeframe: str = Query(default="1h")
):
    """Get the most recent candle for a symbol"""
    # Try Binance first
    result = await binance_provider.get_latest(symbol, timeframe)
    if result:
        return {"symbol": symbol, "timeframe": timeframe, "candle": result, "source": "binance"}

    # Fallback to CSV
    result = await csv_provider.get_latest(symbol, timeframe)
    if result:
        return {"symbol": symbol, "timeframe": timeframe, "candle": result, "source": "csv"}

    raise HTTPException(status_code=404, detail=f"No data for {symbol}")


@router.get("/ohlcv/{symbol}/price")
async def get_current_price(symbol: str):
    """Get current price from Binance ticker"""
    price = await binance_provider.get_ticker_price(symbol)
    if price:
        return {"symbol": symbol, "price": price, "source": "binance"}

    # Fallback to last CSV close
    latest = await csv_provider.get_latest(symbol, "1m")
    if latest:
        return {"symbol": symbol, "price": latest['Close'], "source": "csv"}

    raise HTTPException(status_code=404, detail=f"No price for {symbol}")


@router.get("/symbols")
async def list_symbols():
    """List available symbols"""
    return {
        "binance": binance_provider.get_supported_symbols(),
        "csv": csv_provider.get_supported_symbols()
    }


@router.get("/timeframes")
async def list_timeframes():
    """List supported timeframes"""
    return {
        "supported": binance_provider.get_supported_timeframes()
    }
