"""
Strategic ensemble orchestrator that integrates selection strategies with model inference.

This module bridges the gap between the ensemble inference system and the
advanced selection strategies, providing a unified interface for strategy-based
model selection.
"""

import logging
import asyncio
from typing import List, Tuple, Optional
from enum import Enum

from nexus.core.models.base import ModelResponse
from nexus.core.scoring import ResponseScorer
from nexus.core.strategies import (
    WeightedVotingStrategy,
    CascadingStrategy,
    DynamicWeightStrategy,
    MajorityVotingStrategy,
    CostOptimizedStrategy,
    EnsembleResult,
    EnsembleStrategy
)

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available ensemble strategies."""
    WEIGHTED_VOTING = "weighted_voting"
    CASCADING = "cascading"
    DYNAMIC_WEIGHT = "dynamic_weight"
    MAJORITY_VOTING = "majority_voting"
    COST_OPTIMIZED = "cost_optimized"
    SIMPLE_BEST = "simple_best"  # Default: just pick highest score


class StrategicEnsemble:
    """
    Orchestrates ensemble inference with pluggable selection strategies.

    This class manages the execution of multiple models and applies
    sophisticated selection strategies to choose the optimal response.
    """

    def __init__(self):
        """Initialize strategic ensemble orchestrator."""
        self.scorer = ResponseScorer()

        # Initialize strategies with sensible defaults
        self.strategies = {
            StrategyType.WEIGHTED_VOTING: WeightedVotingStrategy(),
            StrategyType.CASCADING: CascadingStrategy(confidence_threshold=0.75, max_cascades=3),
            StrategyType.DYNAMIC_WEIGHT: DynamicWeightStrategy(),
            StrategyType.MAJORITY_VOTING: MajorityVotingStrategy(),
            StrategyType.COST_OPTIMIZED: CostOptimizedStrategy(min_quality_threshold=0.7),
        }

        logger.info("StrategicEnsemble initialized with 5 strategies")

    def set_strategy_config(self, strategy_type: StrategyType, **kwargs):
        """
        Configure a specific strategy with custom parameters.

        Args:
            strategy_type: Type of strategy to configure
            **kwargs: Strategy-specific configuration parameters
        """
        if strategy_type == StrategyType.WEIGHTED_VOTING:
            self.strategies[strategy_type] = WeightedVotingStrategy(**kwargs)
        elif strategy_type == StrategyType.CASCADING:
            self.strategies[strategy_type] = CascadingStrategy(**kwargs)
        elif strategy_type == StrategyType.DYNAMIC_WEIGHT:
            self.strategies[strategy_type] = DynamicWeightStrategy(**kwargs)
        elif strategy_type == StrategyType.MAJORITY_VOTING:
            self.strategies[strategy_type] = MajorityVotingStrategy(**kwargs)
        elif strategy_type == StrategyType.COST_OPTIMIZED:
            self.strategies[strategy_type] = CostOptimizedStrategy(**kwargs)

        logger.info(f"Configured {strategy_type.value} strategy with {kwargs}")

    async def execute_with_strategy(
        self,
        model_ensemble: List,
        prompt: str,
        strategy_type: StrategyType = StrategyType.SIMPLE_BEST,
        **strategy_kwargs
    ) -> EnsembleResult:
        """
        Execute ensemble inference with a specific strategy.

        Args:
            model_ensemble: List of model instances
            prompt: Input prompt
            strategy_type: Selection strategy to use
            **strategy_kwargs: Additional arguments for the strategy

        Returns:
            EnsembleResult with selected response and metadata

        Raises:
            ValueError: If model ensemble is empty or prompt is invalid
            RuntimeError: If all models fail
        """
        if not model_ensemble:
            raise ValueError("Model ensemble is empty")

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        logger.info(
            f"Executing strategic ensemble with {len(model_ensemble)} models, "
            f"strategy={strategy_type.value}"
        )

        # Generate responses from all models concurrently
        tasks = [model.generate(prompt) for model in model_ensemble]

        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during concurrent model generation: {e}")
            raise RuntimeError("Failed to generate responses from models")

        # Score all responses
        scored_responses = []
        total_cost = 0.0
        total_latency = 0.0

        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Model {model_ensemble[i].name} failed: {response}")
                continue

            if not response.success:
                logger.warning(
                    f"Model {model_ensemble[i].name} returned error: {response.error}"
                )
                continue

            try:
                score = self.scorer.score_response(response, prompt)
                scored_responses.append((score, response))
                total_cost += response.cost
                total_latency += response.latency_ms
                logger.debug(
                    f"{response.model_name}: score={score:.3f}, "
                    f"latency={response.latency_ms:.0f}ms, cost=${response.cost:.4f}"
                )
            except Exception as e:
                logger.error(f"Error scoring response from {model_ensemble[i].name}: {e}")
                continue

        if not scored_responses:
            raise RuntimeError("All models failed to generate valid responses")

        # Apply strategy
        if strategy_type == StrategyType.SIMPLE_BEST:
            # Simple best: just pick highest score
            scored_responses.sort(reverse=True, key=lambda x: x[0])
            best_score, best_response = scored_responses[0]

            result = EnsembleResult(
                content=best_response.content,
                model_name=best_response.model_name,
                provider=best_response.provider,
                score=best_score,
                confidence=best_score,
                strategy_used="simple_best",
                models_queried=len(scored_responses),
                total_cost=total_cost,
                total_latency_ms=total_latency,
                metadata={"all_scores": {r[1].model_name: r[0] for r in scored_responses}}
            )
        else:
            # Use sophisticated strategy
            strategy = self.strategies[strategy_type]
            result = await strategy.select_response(scored_responses, prompt)

        logger.info(
            f"Selected {result.model_name} using {result.strategy_used} strategy "
            f"(score={result.score:.3f}, confidence={result.confidence:.3f}, "
            f"models_queried={result.models_queried}/{len(model_ensemble)})"
        )

        return result

    def get_strategy(self, strategy_type: StrategyType) -> Optional[EnsembleStrategy]:
        """Get a strategy instance by type."""
        return self.strategies.get(strategy_type)

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return [s.value for s in StrategyType]


# Global instance for the API to use
strategic_ensemble = StrategicEnsemble()
