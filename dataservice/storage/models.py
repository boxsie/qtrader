"""Database models for data service"""

from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field
from sqlalchemy import Index, UniqueConstraint


class OHLCVCandle(SQLModel, table=True):
    """OHLCV candle data cache"""
    __tablename__ = "ohlcv_candles"
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uq_ohlcv'),
        Index('idx_ohlcv_lookup', 'symbol', 'timeframe', 'timestamp'),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timeframe: str = Field(index=True)
    timestamp: datetime = Field(index=True)
    open: float
    high: float
    low: float
    close: float
    volume: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class NewsArticle(SQLModel, table=True):
    """News article storage with sentiment"""
    __tablename__ = "news_articles"

    id: Optional[int] = Field(default=None, primary_key=True)
    external_id: str = Field(unique=True, index=True)
    title: str
    content: str
    source: str  # twitter, reddit, web, etc.
    url: str
    published_at: datetime = Field(index=True)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    # Sentiment analysis results
    sentiment_score: Optional[float] = None  # -1 to 1
    sentiment_label: Optional[str] = None  # bullish, bearish, neutral

    # Related symbols (stored as comma-separated)
    symbols: str = ""


class SentimentSnapshot(SQLModel, table=True):
    """Aggregated sentiment snapshots over time"""
    __tablename__ = "sentiment_snapshots"
    __table_args__ = (
        Index('idx_sentiment_lookup', 'symbol', 'timestamp'),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: datetime = Field(index=True)

    # Aggregated metrics
    overall_sentiment: str  # bullish, bearish, neutral
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1

    # Component counts
    news_count: int = 0
    twitter_count: int = 0
    reddit_count: int = 0

    # Component scores
    news_sentiment: float = 0.0
    social_sentiment: float = 0.0


class AnalysisCache(SQLModel, table=True):
    """Cache for LLM analysis results"""
    __tablename__ = "analysis_cache"

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    query_hash: str = Field(unique=True, index=True)
    analysis: str  # JSON response
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
