"""Market analyst using LLM with tool calling"""

from typing import Dict, Any, Optional, List
import logging
import json

from .ollama_client import OllamaClient
from .tools import TOOLS, MARKET_ANALYSIS_SYSTEM

logger = logging.getLogger(__name__)


class MarketAnalyst:
    """LLM-powered market analysis with autonomous tool use"""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        max_iterations: int = 5
    ):
        self.llm = ollama_client or OllamaClient()
        self.max_iterations = max_iterations

        # Tool handlers - would connect to actual data services
        self.tool_handlers = {
            'search_web': self._handle_search_web,
            'get_price_data': self._handle_get_price_data,
            'get_current_price': self._handle_get_current_price,
            'calculate_indicators': self._handle_calculate_indicators,
            'get_market_sentiment': self._handle_get_market_sentiment,
        }

    async def analyze_market(
        self,
        symbol: str,
        question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis.

        The LLM will autonomously decide which tools to use.
        """
        prompt = question or f"""
Analyze the current market conditions for {symbol}.

Consider:
1. Recent price action and key levels
2. Technical indicator signals
3. Current news and sentiment
4. Risk factors

Provide a clear assessment with:
- Market bias (bullish/bearish/neutral)
- Key price levels to watch
- Main risks
- Recommended approach
"""

        # Check if Ollama is available
        if not await self.llm.check_health():
            return {
                'analysis': f"Ollama not available. Cannot analyze {symbol}.",
                'tools_used': [],
                'iterations': 0,
                'error': 'ollama_unavailable'
            }

        # Initial LLM call with tools
        response = await self.llm.generate_with_tools(
            prompt=prompt,
            tools=TOOLS,
            system=MARKET_ANALYSIS_SYSTEM
        )

        messages = [
            {"role": "system", "content": MARKET_ANALYSIS_SYSTEM},
            {"role": "user", "content": prompt}
        ]
        tools_used = []
        iterations = 0

        # Tool use loop
        while response.get('type') == 'tool_calls' and iterations < self.max_iterations:
            iterations += 1
            tool_calls = response.get('tool_calls', [])

            for tool_call in tool_calls:
                func = tool_call.get('function', {})
                tool_name = func.get('name', '')
                tool_args = func.get('arguments', {})

                # Parse arguments if string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except:
                        tool_args = {}

                logger.info(f"Executing tool: {tool_name}({tool_args})")

                # Execute tool
                if tool_name in self.tool_handlers:
                    result = await self.tool_handlers[tool_name](**tool_args)
                    tools_used.append({
                        'tool': tool_name,
                        'args': tool_args,
                        'result_summary': str(result)[:200]
                    })

                    # Add tool result to conversation
                    messages.append({
                        "role": "assistant",
                        "content": f"I'll use the {tool_name} tool."
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Tool result for {tool_name}: {json.dumps(result)}"
                    })

            # Continue conversation
            response = await self.llm.chat(
                messages=messages,
                tools=TOOLS
            )

        # Extract final analysis
        final_content = response.get('content', '')

        return {
            'analysis': final_content,
            'tools_used': tools_used,
            'iterations': iterations
        }

    async def get_quick_bias(self, symbol: str) -> Dict[str, Any]:
        """Get a quick market bias score without full analysis"""
        prompt = f"""In one word, what is your market bias for {symbol} right now: bullish, bearish, or neutral?
Also give a confidence score from 0 to 100.

Format: BIAS: [word] CONFIDENCE: [number]"""

        response = await self.llm.generate(prompt)

        # Parse response
        bias = 'neutral'
        confidence = 50

        response_lower = response.lower()
        if 'bullish' in response_lower:
            bias = 'bullish'
        elif 'bearish' in response_lower:
            bias = 'bearish'

        # Extract confidence number
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            confidence = min(100, max(0, int(numbers[-1])))

        # Convert to -1 to 1 scale
        bias_score = {
            'bullish': confidence / 100,
            'bearish': -confidence / 100,
            'neutral': 0
        }.get(bias, 0)

        return {
            'bias': bias,
            'bias_score': bias_score,
            'confidence': confidence / 100,
            'raw_response': response
        }

    # Tool handler implementations
    async def _handle_search_web(self, query: str) -> Dict:
        """Execute web search"""
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append({
                        'title': r.get('title', ''),
                        'snippet': r.get('body', '')[:200]
                    })
            return {'results': results, 'count': len(results)}
        except Exception as e:
            return {'error': str(e), 'results': []}

    async def _handle_get_price_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 24
    ) -> Dict:
        """Get price data"""
        try:
            from data.providers.binance import BinanceProvider
            provider = BinanceProvider()
            df = await provider.get_historical(symbol, timeframe, limit=limit)

            if df.empty:
                return {'error': 'No data', 'symbol': symbol}

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'latest_price': float(df['Close'].iloc[-1]),
                'price_change': float((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100),
                'high': float(df['High'].max()),
                'low': float(df['Low'].min()),
                'volume': float(df['Volume'].sum())
            }
        except Exception as e:
            return {'error': str(e), 'symbol': symbol}

    async def _handle_get_current_price(self, symbol: str) -> Dict:
        """Get current price"""
        try:
            from data.providers.binance import BinanceProvider
            provider = BinanceProvider()
            price = await provider.get_ticker_price(symbol)
            return {'symbol': symbol, 'price': price}
        except Exception as e:
            return {'error': str(e), 'symbol': symbol}

    async def _handle_calculate_indicators(
        self,
        symbol: str,
        indicators: Optional[List[str]] = None
    ) -> Dict:
        """Calculate technical indicators"""
        indicators = indicators or ['rsi', 'macd']

        try:
            from data.providers.binance import BinanceProvider
            import pandas as pd

            provider = BinanceProvider()
            df = await provider.get_historical(symbol, '1h', limit=100)

            if df.empty:
                return {'error': 'No data', 'symbol': symbol}

            results = {'symbol': symbol}

            # Calculate requested indicators
            if 'rsi' in indicators:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                results['rsi'] = float(100 - (100 / (1 + rs.iloc[-1])))

            if 'sma' in indicators or 'macd' in indicators:
                results['sma_20'] = float(df['Close'].rolling(20).mean().iloc[-1])
                results['sma_50'] = float(df['Close'].rolling(50).mean().iloc[-1])

            if 'macd' in indicators:
                ema12 = df['Close'].ewm(span=12).mean()
                ema26 = df['Close'].ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                results['macd'] = float(macd.iloc[-1])
                results['macd_signal'] = float(signal.iloc[-1])
                results['macd_histogram'] = float(macd.iloc[-1] - signal.iloc[-1])

            return results

        except Exception as e:
            return {'error': str(e), 'symbol': symbol}

    async def _handle_get_market_sentiment(self, symbol: str) -> Dict:
        """Get market sentiment"""
        # Placeholder - would aggregate from news/social providers
        return {
            'symbol': symbol,
            'sentiment': 'neutral',
            'score': 0.0,
            'note': 'Sentiment analysis requires API keys'
        }
