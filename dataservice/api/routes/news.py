"""News and sentiment API routes"""

from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class NewsArticle(BaseModel):
    """News article response"""
    id: str
    title: str
    content: str
    source: str
    url: str
    published_at: str
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None


class SentimentResponse(BaseModel):
    """Aggregated sentiment response"""
    symbol: str
    timestamp: str
    overall_sentiment: str  # bullish, bearish, neutral
    sentiment_score: float  # -1 to 1
    confidence: float
    news_count: int
    social_count: int
    components: dict


@router.get("/latest", response_model=List[NewsArticle])
async def get_latest_news(
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    limit: int = Query(default=20, ge=1, le=100),
    source: Optional[str] = Query(default=None, description="Filter by source")
):
    """
    Get latest news articles.

    Note: Requires API keys for Twitter/Reddit. Returns placeholder data otherwise.
    """
    # Placeholder - would fetch from news providers
    return [
        NewsArticle(
            id="placeholder-1",
            title="Bitcoin Market Analysis",
            content="Placeholder news content. Configure Twitter/Reddit API keys for real news.",
            source="placeholder",
            url="https://example.com",
            published_at=datetime.now().isoformat(),
            sentiment_score=0.0,
            sentiment_label="neutral"
        )
    ]


@router.get("/sentiment/{symbol}", response_model=SentimentResponse)
async def get_sentiment(
    symbol: str,
    hours: int = Query(default=24, ge=1, le=168, description="Lookback hours")
):
    """
    Get aggregated sentiment for a symbol.

    Combines:
    - News sentiment (from news articles)
    - Social sentiment (Twitter, Reddit)
    - Web search results

    Returns sentiment score from -1 (bearish) to 1 (bullish).
    """
    # Placeholder sentiment - would aggregate from providers
    return SentimentResponse(
        symbol=symbol.upper(),
        timestamp=datetime.now().isoformat(),
        overall_sentiment="neutral",
        sentiment_score=0.0,
        confidence=0.5,
        news_count=0,
        social_count=0,
        components={
            "news_sentiment": 0.0,
            "twitter_sentiment": 0.0,
            "reddit_sentiment": 0.0,
            "note": "Configure API keys for real sentiment data"
        }
    )


@router.get("/sentiment/{symbol}/features")
async def get_sentiment_features(symbol: str):
    """
    Get sentiment features formatted for RL agent state vector.

    Returns 5 features:
    - news_sentiment_score (-1 to 1)
    - social_sentiment_score (-1 to 1)
    - sentiment_confidence (0 to 1)
    - news_volume_normalized (0 to 1)
    - llm_bias_score (-1 to 1)
    """
    # Placeholder features - would compute from real data
    return {
        "symbol": symbol.upper(),
        "features": {
            "news_sentiment_score": 0.0,
            "social_sentiment_score": 0.0,
            "sentiment_confidence": 0.5,
            "news_volume_24h": 0.0,
            "llm_bias_score": 0.0
        },
        "feature_vector": [0.0, 0.0, 0.5, 0.0, 0.0],
        "timestamp": datetime.now().isoformat(),
        "note": "Configure API keys and Ollama for real features"
    }
