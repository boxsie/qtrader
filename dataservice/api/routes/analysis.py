"""LLM market analysis API routes"""

from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import logging
import json

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import Ollama client
try:
    from llm.ollama_client import OllamaClient
    from llm.analyst import MarketAnalyst
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM modules not available")


class AnalysisRequest(BaseModel):
    """Market analysis request"""
    symbol: str
    question: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Market analysis response"""
    symbol: str
    analysis: str
    bias: str  # bullish, bearish, neutral
    confidence: float
    tools_used: list
    timestamp: str


@router.post("/market", response_model=AnalysisResponse)
async def analyze_market(request: AnalysisRequest):
    """
    Get LLM-powered market analysis for a symbol.

    The LLM can autonomously:
    - Search the web for news
    - Fetch price data
    - Check technical indicators
    - Analyze social sentiment

    Requires Ollama running at configured URL.
    """
    symbol = request.symbol.upper()
    question = request.question or f"Analyze the current market conditions for {symbol}"

    if not LLM_AVAILABLE:
        # Return placeholder analysis
        return AnalysisResponse(
            symbol=symbol,
            analysis=f"LLM analysis not available. Configure Ollama to enable AI-powered analysis.\n\nYour question: {question}",
            bias="neutral",
            confidence=0.0,
            tools_used=[],
            timestamp=datetime.now().isoformat()
        )

    try:
        # Initialize LLM client and analyst
        client = OllamaClient()
        analyst = MarketAnalyst(client)

        # Run analysis
        result = await analyst.analyze_market(symbol, question)

        # Parse bias from analysis (simple heuristic)
        analysis_lower = result.get('analysis', '').lower()
        if 'bullish' in analysis_lower or 'buy' in analysis_lower:
            bias = 'bullish'
            confidence = 0.7
        elif 'bearish' in analysis_lower or 'sell' in analysis_lower:
            bias = 'bearish'
            confidence = 0.7
        else:
            bias = 'neutral'
            confidence = 0.5

        return AnalysisResponse(
            symbol=symbol,
            analysis=result.get('analysis', 'No analysis generated'),
            bias=bias,
            confidence=confidence,
            tools_used=result.get('tools_used', []),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/summary/{symbol}")
async def get_market_summary(symbol: str):
    """
    Get a quick market summary without full LLM analysis.

    Combines available data:
    - Current price
    - Recent price change
    - Sentiment (if available)
    """
    return {
        "symbol": symbol.upper(),
        "summary": {
            "price": None,
            "change_24h": None,
            "sentiment": "neutral",
            "volume_24h": None
        },
        "timestamp": datetime.now().isoformat(),
        "note": "Full data requires Binance connection"
    }


@router.get("/bias/{symbol}")
async def get_llm_bias(symbol: str):
    """
    Get the LLM's market bias as a single score.

    Returns a value from -1 (very bearish) to 1 (very bullish).
    This can be used directly as a state feature for the RL agent.
    """
    # Placeholder - would run quick LLM query
    return {
        "symbol": symbol.upper(),
        "bias_score": 0.0,  # -1 to 1
        "confidence": 0.5,
        "timestamp": datetime.now().isoformat(),
        "note": "Configure Ollama for real LLM bias"
    }
