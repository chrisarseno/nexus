"""
LLM Client Manager

Centralized LLM client management with automatic provider selection based on
configuration and API key availability.
"""

import logging
from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    TOGETHER = "together"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"


class LLMClient:
    """
    Unified LLM client that abstracts different providers.

    Automatically selects the best available provider based on config and API keys.
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider to use (openai, anthropic, google, etc.)
            model: Model name to use
        """
        try:
            from ..config import config
        except ImportError:
            # Fallback for when module is imported from external scripts
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from config import config

        self.config = config
        self.provider = provider or config.llm.default_provider
        self.model = model or config.llm.default_model

        # Initialize provider client
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate provider client."""
        try:
            if self.provider == "openai":
                self._init_openai()
            elif self.provider == "anthropic":
                self._init_anthropic()
            elif self.provider == "google":
                self._init_google()
            elif self.provider == "cohere":
                self._init_cohere()
            else:
                logger.warning(f"Provider {self.provider} not yet implemented, falling back to simulation")

        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            logger.info("Falling back to simulated responses")

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI

            if not self.config.llm.openai_api_key:
                logger.warning("OPENAI_API_KEY not set")
                return

            self.client = AsyncOpenAI(
                api_key=self.config.llm.openai_api_key,
                organization=os.getenv('OPENAI_ORG_ID')
            )
            logger.info(f"✅ OpenAI client initialized with model: {self.model}")

        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import AsyncAnthropic

            if not self.config.llm.anthropic_api_key:
                logger.warning("ANTHROPIC_API_KEY not set")
                return

            self.client = AsyncAnthropic(
                api_key=self.config.llm.anthropic_api_key
            )
            logger.info(f"✅ Anthropic client initialized with model: {self.model}")

        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")

    def _init_google(self):
        """Initialize Google Generative AI client."""
        try:
            import google.generativeai as genai

            if not self.config.llm.google_api_key:
                logger.warning("GOOGLE_API_KEY not set")
                return

            genai.configure(api_key=self.config.llm.google_api_key)
            self.client = genai
            logger.info(f"✅ Google Generative AI client initialized with model: {self.model}")

        except ImportError:
            logger.error("google-generativeai package not installed. Run: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Google client: {e}")

    def _init_cohere(self):
        """Initialize Cohere client."""
        try:
            import cohere

            if not self.config.llm.cohere_api_key:
                logger.warning("COHERE_API_KEY not set")
                return

            self.client = cohere.AsyncClient(api_key=self.config.llm.cohere_api_key)
            logger.info(f"✅ Cohere client initialized with model: {self.model}")

        except ImportError:
            logger.error("cohere package not installed. Run: pip install cohere")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion from the LLM.

        Args:
            prompt: The prompt to send
            system: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            Dict with 'content', 'model', 'tokens', 'cost'
        """
        if not self.client:
            logger.warning("No LLM client initialized, using simulated response")
            return await self._simulate_completion(prompt)

        try:
            if self.provider == "openai":
                return await self._complete_openai(prompt, system, temperature, max_tokens, **kwargs)
            elif self.provider == "anthropic":
                return await self._complete_anthropic(prompt, system, temperature, max_tokens, **kwargs)
            elif self.provider == "google":
                return await self._complete_google(prompt, system, temperature, max_tokens, **kwargs)
            elif self.provider == "cohere":
                return await self._complete_cohere(prompt, system, temperature, max_tokens, **kwargs)
            else:
                return await self._simulate_completion(prompt)

        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            return await self._simulate_completion(prompt)

    async def _complete_openai(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete using OpenAI."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return {
            'content': response.choices[0].message.content,
            'model': response.model,
            'tokens': {
                'prompt': response.usage.prompt_tokens,
                'completion': response.usage.completion_tokens,
                'total': response.usage.total_tokens
            },
            'cost': self._calculate_cost_openai(response.usage, response.model)
        }

    async def _complete_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete using Anthropic."""
        response = await self.client.messages.create(
            model=self.model,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return {
            'content': response.content[0].text,
            'model': response.model,
            'tokens': {
                'prompt': response.usage.input_tokens,
                'completion': response.usage.output_tokens,
                'total': response.usage.input_tokens + response.usage.output_tokens
            },
            'cost': self._calculate_cost_anthropic(response.usage, response.model)
        }

    async def _complete_google(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete using Google Generative AI."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        model = self.client.GenerativeModel(self.model)
        response = await model.generate_content_async(
            full_prompt,
            generation_config={
                'temperature': temperature,
                'max_output_tokens': max_tokens,
                **kwargs
            }
        )

        return {
            'content': response.text,
            'model': self.model,
            'tokens': {
                'prompt': 0,  # Google doesn't always provide token counts
                'completion': 0,
                'total': 0
            },
            'cost': 0.0  # Estimate or use API to get cost
        }

    async def _complete_cohere(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete using Cohere."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        response = await self.client.generate(
            model=self.model,
            prompt=full_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return {
            'content': response.generations[0].text,
            'model': self.model,
            'tokens': {
                'prompt': 0,
                'completion': 0,
                'total': 0
            },
            'cost': 0.0
        }

    async def _simulate_completion(self, prompt: str) -> Dict[str, Any]:
        """Simulate an LLM completion (for testing without API keys)."""
        await asyncio.sleep(0.1)  # Simulate network delay

        return {
            'content': f"[SIMULATED] Response to: {prompt[:100]}...",
            'model': 'simulated',
            'tokens': {
                'prompt': len(prompt.split()),
                'completion': 50,
                'total': len(prompt.split()) + 50
            },
            'cost': 0.0
        }

    def _calculate_cost_openai(self, usage, model: str) -> float:
        """Calculate cost for OpenAI models."""
        # Pricing as of 2024 (update as needed)
        pricing = {
            'gpt-4': {'prompt': 0.03, 'completion': 0.06},  # per 1K tokens
            'gpt-4-turbo': {'prompt': 0.01, 'completion': 0.03},
            'gpt-3.5-turbo': {'prompt': 0.0005, 'completion': 0.0015}
        }

        # Get base model name
        base_model = model.split('-')[0] + '-' + model.split('-')[1] if '-' in model else model

        prices = pricing.get(base_model, pricing['gpt-4'])

        prompt_cost = (usage.prompt_tokens / 1000) * prices['prompt']
        completion_cost = (usage.completion_tokens / 1000) * prices['completion']

        return prompt_cost + completion_cost

    def _calculate_cost_anthropic(self, usage, model: str) -> float:
        """Calculate cost for Anthropic models."""
        # Pricing as of 2024
        pricing = {
            'claude-3-opus': {'prompt': 0.015, 'completion': 0.075},
            'claude-3-sonnet': {'prompt': 0.003, 'completion': 0.015},
            'claude-3-haiku': {'prompt': 0.00025, 'completion': 0.00125}
        }

        # Default to opus if unknown
        prices = pricing.get(model, pricing['claude-3-opus'])

        prompt_cost = (usage.input_tokens / 1000) * prices['prompt']
        completion_cost = (usage.output_tokens / 1000) * prices['completion']

        return prompt_cost + completion_cost


# Convenience function to get a client
def get_llm_client(provider: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    """Get an LLM client instance."""
    return LLMClient(provider=provider, model=model)


# Import os for OpenAI org_id
import os

__all__ = ['LLMClient', 'LLMProvider', 'get_llm_client']
