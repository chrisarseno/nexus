"""
LLM Backend - Configurable provider for chapter generation

Supports:
- Anthropic (Claude) - fast, API-based
- Ollama - local, free, slower
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import asyncio


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    tokens_used: int
    model: str
    provider: str
    duration_seconds: float = 0
    input_tokens: int = 0
    output_tokens: int = 0


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8000,
    ) -> LLMResponse:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        pass
    
    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return f"{self.__class__.__name__}"


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend."""
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
    
    @property
    def name(self) -> str:
        return f"Anthropic ({self.model})"
    
    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8000,
    ) -> LLMResponse:
        import time
        start = time.time()
        
        client = self._get_client()
        
        # Run sync client in thread pool
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        
        duration = time.time() - start
        
        return LLMResponse(
            content=message.content[0].text,
            tokens_used=message.usage.input_tokens + message.usage.output_tokens,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            model=message.model,
            provider="anthropic",
            duration_seconds=duration
        )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Claude Sonnet pricing (as of late 2024)
        # $3 per 1M input, $15 per 1M output
        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost


class OllamaBackend(LLMBackend):
    """Ollama local backend with Qwen3 thinking mode support."""
    
    def __init__(
        self,
        model: str = "qwen3:30b-a3b",
        host: str = "http://localhost:11434",
        thinking_mode: bool = True,  # Enable Qwen3 deep thinking by default
        strip_thinking: bool = True,  # Remove <think> tags from output
    ):
        self.model = model
        self.host = host
        self.thinking_mode = thinking_mode
        self.strip_thinking = strip_thinking
    
    @property
    def name(self) -> str:
        mode = " [thinking]" if self.thinking_mode and "qwen3" in self.model.lower() else ""
        return f"Ollama ({self.model}{mode})"
    
    def _is_qwen3(self) -> bool:
        """Check if this is a Qwen3 model."""
        return "qwen3" in self.model.lower()
    
    def _extract_content(self, raw_content: str) -> tuple[str, str]:
        """Extract thinking and final content from Qwen3 response.
        
        Returns:
            tuple: (thinking_content, final_content)
        """
        import re
        
        thinking = ""
        content = raw_content
        
        # Extract <think>...</think> blocks
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, raw_content, re.DOTALL)
        
        if think_matches:
            thinking = "\n".join(think_matches)
            if self.strip_thinking:
                content = re.sub(think_pattern, '', raw_content, flags=re.DOTALL).strip()
        
        return thinking, content
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8000,
    ) -> LLMResponse:
        import time
        import aiohttp
        
        start = time.time()
        
        # Use chat API with system message for better results
        url = f"{self.host}/api/chat"
        messages = []
        
        # For Qwen3: Add thinking mode instruction to system prompt
        effective_system = system_prompt
        if self._is_qwen3() and self.thinking_mode:
            thinking_instruction = "You are a thoughtful assistant. Use /think to reason deeply before responding."
            if system_prompt:
                effective_system = f"{thinking_instruction}\n\n{system_prompt}"
            else:
                effective_system = thinking_instruction
        
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        
        # For Qwen3: Prepend /think to enable thinking mode
        effective_prompt = prompt
        if self._is_qwen3() and self.thinking_mode:
            effective_prompt = f"/think\n{prompt}"
        
        messages.append({"role": "user", "content": effective_prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": 32768,  # Qwen3 supports up to 32K context
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Ollama error: {error}")
                
                data = await resp.json()
        
        duration = time.time() - start
        
        # Extract response from chat format
        raw_content = data.get("message", {}).get("content", "")
        
        # Process Qwen3 thinking output
        thinking, content = self._extract_content(raw_content)
        
        # Ollama returns eval_count for output tokens, prompt_eval_count for input
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        
        return LLMResponse(
            content=content,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider="ollama",
            duration_seconds=duration
        )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Local = free (just electricity)
        return 0.0
    
    async def check_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m["name"] for m in data.get("models", [])]
                        # Check if our model (or base name) is available
                        model_base = self.model.split(":")[0]
                        return any(model_base in m for m in models)
            return False
        except Exception:
            return False
    
    async def list_models(self) -> list[str]:
        """List available models."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [m["name"] for m in data.get("models", [])]
            return []
        except Exception:
            return []
    
    async def pull_model(self, progress_callback=None) -> bool:
        """Pull model if not available."""
        import aiohttp
        url = f"{self.host}/api/pull"
        payload = {"name": self.model, "stream": True}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=7200)) as resp:
                    if resp.status != 200:
                        return False
                    
                    # Stream progress
                    async for line in resp.content:
                        if line and progress_callback:
                            try:
                                import json
                                data = json.loads(line)
                                status = data.get("status", "")
                                if "pulling" in status or "downloading" in status:
                                    completed = data.get("completed", 0)
                                    total = data.get("total", 0)
                                    if total > 0:
                                        pct = (completed / total) * 100
                                        progress_callback(f"{status}: {pct:.1f}%")
                                else:
                                    progress_callback(status)
                            except:
                                pass
                    
                    return True
        except Exception as e:
            print(f"Failed to pull model: {e}")
            return False


def get_backend(
    provider: str = "anthropic",
    model: Optional[str] = None,
    **kwargs
) -> LLMBackend:
    """Factory function to get LLM backend.
    
    Args:
        provider: "anthropic", "openai", or "ollama"
        model: Model name (provider-specific)
        **kwargs: Additional provider-specific args
        
    Returns:
        LLMBackend instance
    """
    if provider == "anthropic":
        return AnthropicBackend(
            model=model or "claude-sonnet-4-20250514",
            **kwargs
        )
    elif provider == "openai":
        return OpenAIBackend(
            model=model or "gpt-4o",
            **kwargs
        )
    elif provider == "ollama":
        return OllamaBackend(
            model=model or "qwen3:30b-a3b",
            thinking_mode=kwargs.pop("thinking_mode", True),
            strip_thinking=kwargs.pop("strip_thinking", True),
            **kwargs
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend - ideal for creative content like blueprint generation."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8000,
    ) -> LLMResponse:
        import time
        start = time.time()
        
        client = self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Run sync client in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages
            )
        )
        
        duration = time.time() - start
        
        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=response.usage.prompt_tokens + response.usage.completion_tokens,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
            provider="openai",
            duration_seconds=duration
        )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # GPT-4o pricing (as of late 2024)
        # $2.50 per 1M input, $10 per 1M output
        input_cost = (input_tokens / 1_000_000) * 2.5
        output_cost = (output_tokens / 1_000_000) * 10.0
        return input_cost + output_cost


# Convenience presets
BACKENDS = {
    # Cloud providers
    "anthropic-sonnet": lambda: AnthropicBackend(model="claude-sonnet-4-20250514"),
    "anthropic-haiku": lambda: AnthropicBackend(model="claude-haiku-4-20250514"),
    "openai-gpt4o": lambda: OpenAIBackend(model="gpt-4o"),
    "openai-gpt4o-mini": lambda: OpenAIBackend(model="gpt-4o-mini"),
    
    # Qwen3 with thinking mode (recommended for creative writing)
    "ollama-qwen3-30b": lambda: OllamaBackend(model="qwen3:30b-a3b", thinking_mode=True),  # MoE - best efficiency
    "ollama-qwen3-8b": lambda: OllamaBackend(model="qwen3:8b", thinking_mode=True),        # Fast, laptop-friendly
    "ollama-qwen3-14b": lambda: OllamaBackend(model="qwen3:14b", thinking_mode=True),      # Middle ground
    "ollama-qwen3-32b": lambda: OllamaBackend(model="qwen3:32b", thinking_mode=True),      # Dense, high quality
    
    # Qwen3 without thinking mode (faster, less reasoning)
    "ollama-qwen3-30b-fast": lambda: OllamaBackend(model="qwen3:30b-a3b", thinking_mode=False),
    "ollama-qwen3-8b-fast": lambda: OllamaBackend(model="qwen3:8b", thinking_mode=False),
    
    # Qwen2.5 (legacy)
    "ollama-qwen-32b": lambda: OllamaBackend(model="qwen2.5:32b"),
    "ollama-qwen-14b": lambda: OllamaBackend(model="qwen2.5:14b"),
    "ollama-qwen-7b": lambda: OllamaBackend(model="qwen2.5:7b"),
    
    # Other models
    "ollama-llama-8b": lambda: OllamaBackend(model="llama3.1:8b"),
    "ollama-llama-70b": lambda: OllamaBackend(model="llama3.1:70b"),
}


def get_preset_backend(preset: str) -> LLMBackend:
    """Get a backend from preset name."""
    if preset not in BACKENDS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(BACKENDS.keys())}")
    return BACKENDS[preset]()


def list_presets() -> list[str]:
    """List available backend presets."""
    return list(BACKENDS.keys())
