"""Tool definitions for LLM function calling"""

# Tool definitions in OpenAI-compatible format (supported by Ollama)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information about cryptocurrencies, market news, or any topic. Use this to find recent news and developments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_data",
            "description": "Get current and historical price data for a cryptocurrency including OHLCV candles",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The trading pair symbol (e.g., BTCUSDT, ETHUSDT)"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Candle timeframe: 1m, 5m, 15m, 1h, 4h, 1d",
                        "default": "1h"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of candles to fetch",
                        "default": 24
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_price",
            "description": "Get the current real-time price for a cryptocurrency",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The trading pair symbol (e.g., BTCUSDT)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_indicators",
            "description": "Calculate technical analysis indicators for a symbol (RSI, MACD, Bollinger Bands, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The trading pair symbol"
                    },
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of indicators to calculate: rsi, macd, bollinger, sma, ema, atr"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_sentiment",
            "description": "Get aggregated market sentiment from news and social media for a cryptocurrency",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The cryptocurrency symbol (e.g., BTC, ETH)"
                    }
                },
                "required": ["symbol"]
            }
        }
    }
]


# System prompts for different analysis types
MARKET_ANALYSIS_SYSTEM = """You are an expert cryptocurrency market analyst with deep knowledge of:
- Technical analysis (chart patterns, indicators, support/resistance)
- Fundamental analysis (on-chain metrics, adoption, development activity)
- Sentiment analysis (social media, news, market psychology)
- Risk management (position sizing, stop losses, risk/reward)

You have access to tools that can:
- Search the web for current news and information
- Fetch real-time price and volume data
- Calculate technical indicators
- Check market sentiment

When analyzing markets:
1. First gather relevant data using your tools
2. Look for confluence between different signals
3. Consider both bullish and bearish scenarios
4. Provide specific, actionable insights
5. Always mention key risk factors

Be concise but thorough. Focus on actionable information.
Format your analysis with clear sections."""


QUICK_ANALYSIS_SYSTEM = """You are a crypto market analyst. Provide a brief market assessment.
Be concise - 2-3 key points maximum. Focus on the most important factor right now."""
