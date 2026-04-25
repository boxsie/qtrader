"""Web search API routes"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import DuckDuckGo search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo_search not installed")


class SearchResult(BaseModel):
    """Search result item"""
    title: str
    url: str
    snippet: str


class SearchResponse(BaseModel):
    """Web search response"""
    query: str
    results: List[SearchResult]
    count: int


@router.post("/web", response_model=SearchResponse)
async def web_search(
    query: str,
    limit: int = Query(default=10, ge=1, le=20)
):
    """
    Search the web for crypto news and information.

    Uses DuckDuckGo (no API key required).
    """
    if not DDGS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Web search not available. Install duckduckgo-search."
        )

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=limit):
                results.append(SearchResult(
                    title=r.get('title', ''),
                    url=r.get('href', ''),
                    snippet=r.get('body', '')
                ))

        return SearchResponse(
            query=query,
            results=results,
            count=len(results)
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/crypto/{symbol}")
async def search_crypto_news(
    symbol: str,
    limit: int = Query(default=10, ge=1, le=20)
):
    """
    Search for recent crypto news about a symbol.

    Automatically constructs relevant search queries.
    """
    queries = [
        f"{symbol} cryptocurrency news today",
        f"{symbol} price analysis",
        f"{symbol} market update"
    ]

    all_results = []
    seen_urls = set()

    if DDGS_AVAILABLE:
        try:
            with DDGS() as ddgs:
                for q in queries:
                    for r in ddgs.text(q, max_results=limit // len(queries)):
                        url = r.get('href', '')
                        if url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(SearchResult(
                                title=r.get('title', ''),
                                url=url,
                                snippet=r.get('body', '')
                            ))
        except Exception as e:
            logger.error(f"Crypto search error: {e}")

    return {
        "symbol": symbol.upper(),
        "results": all_results[:limit],
        "count": len(all_results[:limit])
    }
