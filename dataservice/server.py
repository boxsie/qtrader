"""
QTrader Data Service - Standalone data fetching, news, and LLM analysis

Provides:
- OHLCV data from Binance/CSV
- News and sentiment aggregation
- Web search
- LLM-powered market analysis via Ollama

To run:
    python server.py --host 0.0.0.0 --port 8766
"""

import sys
from pathlib import Path

# Add parent for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import logging
from datetime import datetime
from typing import Set, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import yaml

from core.config import Settings
from storage.database import init_db, get_session
from api.routes import ohlcv, news, search, analysis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load settings
settings = Settings()

# WebSocket connections for streaming
active_connections: Set[WebSocket] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("=" * 60)
    logger.info("QTrader Data Service Starting")
    logger.info("=" * 60)

    # Initialize database
    await init_db(settings.database_url)
    logger.info(f"Database initialized: {settings.database_url}")

    yield

    logger.info("Data Service shutting down")


app = FastAPI(
    title="QTrader Data Service",
    description="Centralized data fetching, news/sentiment, and LLM analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(ohlcv.router, prefix="/api/data", tags=["OHLCV Data"])
app.include_router(news.router, prefix="/api/news", tags=["News & Sentiment"])
app.include_router(search.router, prefix="/api/search", tags=["Web Search"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["LLM Analysis"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "dataservice",
        "timestamp": datetime.now().isoformat(),
        "active_streams": len(active_connections)
    }


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "QTrader Data Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ohlcv": "/api/data/ohlcv/{symbol}",
            "indicators": "/api/data/indicators/{symbol}",
            "sentiment": "/api/news/sentiment/{symbol}",
            "search": "/api/search/web",
            "analysis": "/api/analysis/market",
            "stream": "/ws/stream"
        }
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket for real-time data streaming"""
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"Stream client connected. Active: {len(active_connections)}")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif data.get("type") == "subscribe":
                # Handle subscription requests
                symbol = data.get("symbol", "BTCUSDT")
                timeframe = data.get("timeframe", "1m")
                await websocket.send_json({
                    "type": "subscribed",
                    "symbol": symbol,
                    "timeframe": timeframe
                })
                logger.info(f"Client subscribed to {symbol} {timeframe}")

    except WebSocketDisconnect:
        active_connections.discard(websocket)
        logger.info(f"Stream client disconnected. Active: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        active_connections.discard(websocket)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QTrader Data Service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("QTrader Data Service")
    logger.info("=" * 60)
    logger.info(f"API:       http://{args.host}:{args.port}")
    logger.info(f"WebSocket: ws://{args.host}:{args.port}/ws/stream")
    logger.info(f"Health:    http://{args.host}:{args.port}/health")
    logger.info("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
