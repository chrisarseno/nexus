"""
Fantastic Palm Tree Integration

Seamless integration between Cog-Eng and the Fantastic Palm Tree unified intelligence system.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio

from .api.client import CognitiveEngine, CogEngConfig, CogEngResponse
from .config import config

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for FPT integration."""
    # Cog-Eng settings
    enable_consciousness: bool = True
    enable_learning: bool = True
    enable_agents: bool = True

    # Integration mode
    mode: str = "enhance"  # standalone, pass_through, enhance

    # Feature flags
    consciousness_monitoring: bool = True
    knowledge_accumulation: bool = True
    multi_agent_verification: bool = True


class CogEngAdapter:
    """
    Adapter for integrating Cog-Eng with Fantastic Palm Tree.

    Supports three integration modes:
    1. Standalone: Uses only Cog-Eng
    2. Pass-Through: FPT -> Cog-Eng sequential processing
    3. Enhance: FPT + Cog-Eng parallel processing with result combination

    Usage:
        adapter = CogEngAdapter()
        await adapter.initialize()

        result = await adapter.process_with_consciousness(
            prompt="Your task here",
            enable_multi_agent=True
        )
    """

    def __init__(
        self,
        fpt_system: Optional[Any] = None,
        integration_config: Optional[IntegrationConfig] = None
    ):
        """
        Initialize the adapter.

        Args:
            fpt_system: Optional FPT UnifiedIntelligenceSystem instance
            integration_config: Optional integration configuration
        """
        self.fpt_system = fpt_system
        self.integration_config = integration_config or IntegrationConfig()

        # Apply config from .env
        if config.fpt.path:
            logger.info(f"FPT path configured: {config.fpt.path}")

        if config.fpt.integration_mode:
            self.integration_config.mode = config.fpt.integration_mode

        # Initialize Cog-Eng
        self.cog_eng = None
        cog_config = CogEngConfig(
            enable_consciousness=self.integration_config.enable_consciousness,
            enable_learning=self.integration_config.enable_learning,
            enable_agents=self.integration_config.enable_agents
        )
        self.cog_eng = CognitiveEngine(config=cog_config)

        # Integration state
        self.initialized = False
        self.stats = {
            'total_requests': 0,
            'cog_eng_only': 0,
            'fpt_only': 0,
            'combined': 0
        }

        logger.info(f"CogEng-FPT Adapter created in {self.integration_config.mode} mode")

    async def initialize(self):
        """Initialize the adapter."""
        if self.initialized:
            return

        logger.info("Initializing CogEng-FPT Adapter...")

        # Initialize Cog-Eng
        if self.cog_eng:
            await self.cog_eng.initialize()

        # Initialize FPT if available
        if self.fpt_system and hasattr(self.fpt_system, 'initialize'):
            await self.fpt_system.initialize()

        self.initialized = True
        logger.info("[SUCCESS] CogEng-FPT Adapter initialized")

    async def process_with_consciousness(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        enable_multi_agent: bool = True,
        use_fpt_pipeline: bool = True
    ) -> Dict[str, Any]:
        """
        Process a prompt with consciousness enhancement.

        Args:
            prompt: The input prompt
            context: Additional context
            enable_multi_agent: Whether to use multi-agent orchestration
            use_fpt_pipeline: Whether to use FPT pipeline

        Returns:
            Dict with results including consciousness insights
        """
        if not self.initialized:
            await self.initialize()

        self.stats['total_requests'] += 1

        # Standalone mode: Only Cog-Eng
        if self.integration_config.mode == 'standalone':
            self.stats['cog_eng_only'] += 1
            return await self._process_cog_eng_only(prompt, context, enable_multi_agent)

        # Pass-through mode: FPT -> Cog-Eng
        if self.integration_config.mode == 'pass_through' and use_fpt_pipeline:
            self.stats['combined'] += 1
            return await self._process_pass_through(prompt, context, enable_multi_agent)

        # Enhance mode: FPT + Cog-Eng parallel
        if self.integration_config.mode == 'enhance' and use_fpt_pipeline:
            self.stats['combined'] += 1
            return await self._process_enhance_mode(prompt, context, enable_multi_agent)

        # Default: FPT only
        self.stats['fpt_only'] += 1
        return await self._process_fpt_only(prompt, context)

    async def _process_cog_eng_only(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        enable_multi_agent: bool
    ) -> Dict[str, Any]:
        """Process using only Cog-Eng."""
        logger.info("Processing with Cog-Eng only")

        result = await self.cog_eng.process(
            task=prompt,
            context=context or {},
            priority="normal",
            require_verification=enable_multi_agent
        )

        return {
            'response': result.response,
            'confidence': result.confidence,
            'mode': 'cog_eng_only',
            'consciousness_state': result.consciousness_state,
            'learning_insights': result.learning_insights,
            'safety_evaluation': result.safety_evaluation,
            'agents_involved': result.agents_involved,
            'processing_time': result.processing_time,
            'metadata': result.metadata
        }

    async def _process_pass_through(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        enable_multi_agent: bool
    ) -> Dict[str, Any]:
        """Process through FPT first, then Cog-Eng."""
        logger.info("Processing in pass-through mode (FPT -> Cog-Eng)")

        # First: Process through FPT
        fpt_result = await self._process_fpt_only(prompt, context)

        # Second: Process through Cog-Eng with FPT results as context
        enhanced_context = {
            **(context or {}),
            'fpt_result': fpt_result
        }

        cog_eng_result = await self.cog_eng.process(
            task=f"Enhance and verify: {prompt}",
            context=enhanced_context,
            priority="normal"
        )

        return {
            'response': cog_eng_result.response,
            'confidence': cog_eng_result.confidence,
            'mode': 'pass_through',
            'fpt_result': fpt_result,
            'consciousness_state': cog_eng_result.consciousness_state,
            'learning_insights': cog_eng_result.learning_insights,
            'safety_evaluation': cog_eng_result.safety_evaluation,
            'agents_involved': cog_eng_result.agents_involved,
            'processing_time': cog_eng_result.processing_time,
            'metadata': cog_eng_result.metadata
        }

    async def _process_enhance_mode(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        enable_multi_agent: bool
    ) -> Dict[str, Any]:
        """Process FPT and Cog-Eng in parallel, combine results."""
        logger.info("Processing in enhance mode (FPT + Cog-Eng parallel)")

        # Run both in parallel
        fpt_task = self._process_fpt_only(prompt, context)
        cog_eng_task = self.cog_eng.process(
            task=prompt,
            context=context or {},
            priority="normal"
        )

        fpt_result, cog_eng_result = await asyncio.gather(fpt_task, cog_eng_task)

        # Combine results
        combined_response = self._combine_results(fpt_result, cog_eng_result)

        return {
            'response': combined_response,
            'confidence': (fpt_result.get('confidence', 0.5) + cog_eng_result.confidence) / 2,
            'mode': 'enhance',
            'fpt_result': fpt_result,
            'cog_eng_result': {
                'response': cog_eng_result.response,
                'confidence': cog_eng_result.confidence,
                'agents_involved': cog_eng_result.agents_involved
            },
            'consciousness_state': cog_eng_result.consciousness_state,
            'learning_insights': cog_eng_result.learning_insights,
            'safety_evaluation': cog_eng_result.safety_evaluation,
            'processing_time': cog_eng_result.processing_time
        }

    async def _process_fpt_only(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process using only FPT pipeline."""
        logger.info("Processing with FPT only")

        # If FPT system is available, use it
        if self.fpt_system and hasattr(self.fpt_system, 'process'):
            try:
                result = await self.fpt_system.process(prompt, context)
                return result
            except Exception as e:
                logger.error(f"Error processing with FPT: {e}")

        # Placeholder for FPT pipeline
        return {
            'response': f"FPT processed: {prompt}",
            'confidence': 0.85,
            'mode': 'fpt_only',
            'metadata': {
                'prompt': prompt,
                'context': context,
                'note': 'FPT system not connected'
            }
        }

    def _combine_results(
        self,
        fpt_result: Dict[str, Any],
        cog_eng_result: CogEngResponse
    ) -> str:
        """Combine results from FPT and Cog-Eng."""
        return f"""Combined Intelligence Analysis:

Fantastic Palm Tree Result:
{fpt_result.get('response', '')}

Cog-Eng Enhanced Insights:
{cog_eng_result.response}

Confidence: {(fpt_result.get('confidence', 0.5) + cog_eng_result.confidence) / 2:.2f}
Agents Involved: {', '.join(cog_eng_result.agents_involved)}
        """.strip()

    def get_consciousness_state(self) -> Optional[Dict[str, Any]]:
        """Get current consciousness state."""
        if self.cog_eng and self.cog_eng.consciousness_core:
            return self.cog_eng.consciousness_core.get_system_state()
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self.stats,
            'cog_eng_initialized': self.cog_eng is not None and self.cog_eng.initialized,
            'fpt_connected': self.fpt_system is not None,
            'mode': self.integration_config.mode
        }

    async def shutdown(self):
        """Shutdown the adapter."""
        if self.cog_eng:
            await self.cog_eng.shutdown()

        if self.fpt_system and hasattr(self.fpt_system, 'shutdown'):
            await self.fpt_system.shutdown()

        logger.info("CogEng-FPT Adapter shutdown complete")


# Convenience function for quick integration
async def process_with_cog_eng(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    fpt_system: Optional[Any] = None,
    mode: str = "enhance"
) -> Dict[str, Any]:
    """
    Quick helper function to process with Cog-Eng integration.

    Args:
        prompt: Input prompt
        context: Optional context
        fpt_system: Optional FPT system instance
        mode: Integration mode (standalone, pass_through, enhance)

    Returns:
        Processing results with consciousness insights

    Example:
        result = await process_with_cog_eng(
            "Analyze market trends",
            mode="enhance"
        )
    """
    config = IntegrationConfig(mode=mode)
    adapter = CogEngAdapter(fpt_system=fpt_system, integration_config=config)
    await adapter.initialize()

    try:
        return await adapter.process_with_consciousness(prompt, context)
    finally:
        await adapter.shutdown()
