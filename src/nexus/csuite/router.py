"""
LLM Router for C-Suite Agents.

Provides intelligent routing of LLM requests to appropriate models
based on task complexity tier:

- Strategic (Agent level): Complex reasoning, goal understanding
- Planning (Manager level): Task decomposition, coordination
- Execution (Specialist level): Atomic tasks, structured output

Routes to local models by default, escalating to cloud only when needed.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model complexity tiers."""
    EXECUTION = "execution"    # Small, fast, atomic tasks (1-3B)
    PLANNING = "planning"      # Medium, task decomposition (3-7B)
    STRATEGIC = "strategic"    # Larger, goal understanding (7-14B)
    CLOUD = "cloud"            # Cloud fallback for novel/high-stakes


@dataclass
class LLMResult:
    """Result from an LLM completion."""
    text: str
    model: str
    tier: ModelTier
    tokens_used: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    provider: str  # "ollama", "openai", "anthropic"
    tier: ModelTier
    context_window: int = 4096
    cost_per_1k_tokens: float = 0.0  # USD, 0 for local
    priority: int = 0  # Higher = preferred
    enabled: bool = True


class LLMRouter:
    """
    Routes LLM requests to appropriate models based on tier.

    Supports:
    - Ollama (local models)
    - OpenAI (cloud fallback)
    - Anthropic (cloud fallback)

    Default routing:
    - execution → qwen2.5:1.5b or phi3:mini (local)
    - planning → phi3:medium or llama3.2:3b (local)
    - strategic → mistral:7b or qwen2.5:14b (local)
    - cloud → claude-3-sonnet or gpt-4o (when needed)
    """

    # Default model configurations
    DEFAULT_MODELS: Dict[str, ModelConfig] = {
        # Execution tier (small, fast)
        "qwen2.5:1.5b": ModelConfig(
            name="qwen2.5:1.5b",
            provider="ollama",
            tier=ModelTier.EXECUTION,
            context_window=4096,
            priority=10,
        ),
        "phi3:mini": ModelConfig(
            name="phi3:mini",
            provider="ollama",
            tier=ModelTier.EXECUTION,
            context_window=4096,
            priority=5,
        ),

        # Planning tier (medium)
        "phi3:medium": ModelConfig(
            name="phi3:medium",
            provider="ollama",
            tier=ModelTier.PLANNING,
            context_window=8192,
            priority=10,
        ),
        "llama3.2:3b": ModelConfig(
            name="llama3.2:3b",
            provider="ollama",
            tier=ModelTier.PLANNING,
            context_window=8192,
            priority=5,
        ),
        "qwen2.5:3b": ModelConfig(
            name="qwen2.5:3b",
            provider="ollama",
            tier=ModelTier.PLANNING,
            context_window=8192,
            priority=8,
        ),

        # Strategic tier (larger)
        "mistral:7b": ModelConfig(
            name="mistral:7b",
            provider="ollama",
            tier=ModelTier.STRATEGIC,
            context_window=8192,
            priority=10,
        ),
        "qwen2.5:7b": ModelConfig(
            name="qwen2.5:7b",
            provider="ollama",
            tier=ModelTier.STRATEGIC,
            context_window=32768,
            priority=8,
        ),
        "qwen2.5:14b": ModelConfig(
            name="qwen2.5:14b",
            provider="ollama",
            tier=ModelTier.STRATEGIC,
            context_window=32768,
            priority=5,  # Lower priority due to slower speed
        ),

        # Cloud tier (fallback)
        "claude-3-5-sonnet": ModelConfig(
            name="claude-3-5-sonnet-20241022",
            provider="anthropic",
            tier=ModelTier.CLOUD,
            context_window=200000,
            cost_per_1k_tokens=0.003,
            priority=10,
        ),
        "gpt-4o-mini": ModelConfig(
            name="gpt-4o-mini",
            provider="openai",
            tier=ModelTier.CLOUD,
            context_window=128000,
            cost_per_1k_tokens=0.00015,
            priority=8,
        ),
    }

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        custom_models: Optional[Dict[str, ModelConfig]] = None,
    ):
        self.ollama_base_url = ollama_base_url
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key

        # Merge default and custom models
        self._models = dict(self.DEFAULT_MODELS)
        if custom_models:
            self._models.update(custom_models)

        # Track available models (checked on first use)
        self._available_models: Optional[set] = None
        self._model_check_time: Optional[datetime] = None

        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._requests_by_tier: Dict[str, int] = {t.value: 0 for t in ModelTier}

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tier: str = "execution",
        context: Optional[Dict[str, Any]] = None,
        force_model: Optional[str] = None,
    ) -> LLMResult:
        """
        Complete a prompt using the appropriate model for the tier.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            tier: Model tier (execution, planning, strategic, cloud)
            context: Additional context for routing decisions
            force_model: Force a specific model (bypasses tier routing)

        Returns:
            LLMResult with response text and metadata
        """
        start_time = datetime.now()
        context = context or {}

        # Determine which model to use
        if force_model:
            model_name = force_model
        else:
            model_name = await self._select_model(tier, context)

        if not model_name:
            raise RuntimeError(f"No available model for tier: {tier}")

        model_config = self._models.get(model_name)
        if not model_config:
            raise RuntimeError(f"Unknown model: {model_name}")

        # Route to appropriate provider
        if model_config.provider == "ollama":
            result = await self._complete_ollama(model_name, prompt, system_prompt)
        elif model_config.provider == "openai":
            result = await self._complete_openai(model_name, prompt, system_prompt)
        elif model_config.provider == "anthropic":
            result = await self._complete_anthropic(model_name, prompt, system_prompt)
        else:
            raise RuntimeError(f"Unknown provider: {model_config.provider}")

        # Update result metadata
        result.tier = ModelTier(tier) if isinstance(tier, str) else tier
        result.latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        result.cost_usd = (result.tokens_used / 1000) * model_config.cost_per_1k_tokens

        # Update statistics
        self._total_requests += 1
        self._total_tokens += result.tokens_used
        self._total_cost += result.cost_usd
        self._requests_by_tier[tier] = self._requests_by_tier.get(tier, 0) + 1

        return result

    async def _select_model(self, tier: str, context: Dict[str, Any]) -> Optional[str]:
        """Select the best available model for the tier."""
        # Ensure we have an up-to-date list of available models
        await self._check_available_models()

        # Get models for this tier
        tier_enum = ModelTier(tier) if isinstance(tier, str) else tier
        candidates = [
            (name, config)
            for name, config in self._models.items()
            if config.tier == tier_enum and config.enabled
        ]

        if not candidates:
            # Fall back to next tier up
            if tier_enum == ModelTier.EXECUTION:
                return await self._select_model(ModelTier.PLANNING.value, context)
            elif tier_enum == ModelTier.PLANNING:
                return await self._select_model(ModelTier.STRATEGIC.value, context)
            elif tier_enum == ModelTier.STRATEGIC:
                return await self._select_model(ModelTier.CLOUD.value, context)
            return None

        # Filter to available models
        available_candidates = [
            (name, config)
            for name, config in candidates
            if name in self._available_models or config.provider != "ollama"
        ]

        if not available_candidates:
            # Try next tier
            if tier_enum != ModelTier.CLOUD:
                return await self._select_model(ModelTier.CLOUD.value, context)
            return None

        # Sort by priority (higher first)
        available_candidates.sort(key=lambda x: x[1].priority, reverse=True)

        return available_candidates[0][0]

    async def _check_available_models(self, force_refresh: bool = False) -> None:
        """Check which Ollama models are available."""
        # Cache for 5 minutes
        if (
            not force_refresh
            and self._available_models is not None
            and self._model_check_time is not None
        ):
            elapsed = (datetime.now() - self._model_check_time).total_seconds()
            if elapsed < 300:
                return

        self._available_models = set()

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.ollama_base_url}/api/tags",
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        self._available_models.add(model.get("name", ""))
        except Exception as e:
            logger.warning(f"Failed to check Ollama models: {e}")

        # Add cloud models if API keys are present
        if self.openai_api_key:
            self._available_models.add("gpt-4o-mini")
            self._available_models.add("gpt-4o")
        if self.anthropic_api_key:
            self._available_models.add("claude-3-5-sonnet")
            self._available_models.add("claude-3-5-sonnet-20241022")

        self._model_check_time = datetime.now()
        logger.info(f"Available models: {self._available_models}")

    async def _complete_ollama(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str]
    ) -> LLMResult:
        """Complete using Ollama."""
        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

        return LLMResult(
            text=data.get("message", {}).get("content", ""),
            model=model,
            tier=ModelTier.EXECUTION,  # Will be updated by caller
            tokens_used=data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
            metadata={
                "provider": "ollama",
                "eval_count": data.get("eval_count", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
            }
        )

    async def _complete_openai(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str]
    ) -> LLMResult:
        """Complete using OpenAI."""
        if not self.openai_api_key:
            raise RuntimeError("OpenAI API key not configured")

        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                },
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        return LLMResult(
            text=data["choices"][0]["message"]["content"],
            model=model,
            tier=ModelTier.CLOUD,
            tokens_used=usage.get("total_tokens", 0),
            metadata={
                "provider": "openai",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        )

    async def _complete_anthropic(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str]
    ) -> LLMResult:
        """Complete using Anthropic."""
        if not self.anthropic_api_key:
            raise RuntimeError("Anthropic API key not configured")

        import httpx

        request_body = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            request_body["system"] = system_prompt

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        return LLMResult(
            text=data["content"][0]["text"],
            model=model,
            tier=ModelTier.CLOUD,
            tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            metadata={
                "provider": "anthropic",
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "requests_by_tier": dict(self._requests_by_tier),
            "available_models": list(self._available_models or []),
        }

    def get_models_for_tier(self, tier: str) -> List[str]:
        """Get all configured models for a tier."""
        tier_enum = ModelTier(tier) if isinstance(tier, str) else tier
        return [
            name for name, config in self._models.items()
            if config.tier == tier_enum and config.enabled
        ]
