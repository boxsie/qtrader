"""Ollama API client with tool/function calling support"""

from typing import List, Dict, Any, Optional
import aiohttp
import logging
import json

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client for Ollama API with tool calling"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b"
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model

    async def check_health(self) -> bool:
        """Check if Ollama is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        return []

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Simple text generation"""

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"Ollama error: {error}")
                        return f"Error: {error}"

                    data = await response.json()
                    return data.get('message', {}).get('content', '')

        except aiohttp.ClientError as e:
            logger.error(f"Connection error: {e}")
            return f"Connection error: {e}"
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {e}"

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate with tool use (function calling).

        Returns dict with either:
        - {'type': 'content', 'content': '...'} for text response
        - {'type': 'tool_calls', 'tool_calls': [...]} for tool invocations
        """

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": tools,
                        "stream": False,
                        "options": {
                            "temperature": temperature
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        return {'type': 'error', 'error': error}

                    data = await response.json()
                    message = data.get('message', {})

                    if 'tool_calls' in message and message['tool_calls']:
                        return {
                            'type': 'tool_calls',
                            'tool_calls': message['tool_calls']
                        }
                    else:
                        return {
                            'type': 'content',
                            'content': message.get('content', '')
                        }

        except Exception as e:
            logger.error(f"Tool generation error: {e}")
            return {'type': 'error', 'error': str(e)}

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Multi-turn chat with optional tools.

        messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature}
            }

            if tools:
                payload["tools"] = tools

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        return {'type': 'error', 'error': error}

                    data = await response.json()
                    message = data.get('message', {})

                    if 'tool_calls' in message and message['tool_calls']:
                        return {
                            'type': 'tool_calls',
                            'tool_calls': message['tool_calls'],
                            'content': message.get('content', '')
                        }
                    else:
                        return {
                            'type': 'content',
                            'content': message.get('content', '')
                        }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {'type': 'error', 'error': str(e)}
