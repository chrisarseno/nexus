"""
Unified Ensemble System - Core Implementation

⚠️ CURRENT STATUS (v0.1.0 Alpha):
- ✅ Model Adapters: OpenAI and Anthropic fully functional with real API calls
- ✅ Ensemble Strategies: 6 strategies implemented (weighted, cascading, dynamic, majority, cost, synthesized)
- ⚠️ Other Providers: Google, Mistral, Cohere, Together, Replicate require API keys and full adapter implementation
- ⚠️ Testing: Integration tests with real API calls need expansion
- ⚠️ Production: Not yet deployed, performance metrics are targets not measurements

See AUDIT_REPORT_ACTUAL_CAPABILITIES.md for complete details.

Architecture:
- Ensemble orchestration with multiple strategies
- Real-time model execution via provider-specific adapters
- Response synthesis and confidence calibration
- Epistemic monitoring and drift detection
- Cost tracking and budget management
- Circuit breakers and model quarantine for safety
"""

import asyncio
import logging
import re
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from nexus.providers.config import EnsembleStrategy, get_config

logger = logging.getLogger(__name__)
from nexus.providers.ensemble.types import (
    EnsembleRequest,
    EnsembleResponse,
    EpistemicHealth,
    ModelResponse,
    PerformanceMetrics,
    QueryType,
)


class UnifiedEnsemble:
    """
    Unified ensemble system combining best features from all projects.

    This class orchestrates multiple AI models and synthesizes their responses
    using sophisticated strategies while monitoring epistemic health.

    Features:
    - Multi-strategy selection (from TheNexus)
    - Response synthesis (from combo1)
    - Confidence calibration (from combo1)
    - Epistemic monitoring (from fluffy-eureka)
    - Cost tracking and optimization
    - Circuit breakers and quarantine
    - Self-learning from feedback
    """

    def __init__(self) -> None:
        """Initialize the unified ensemble system."""
        self.config = get_config()
        self.models: Dict[str, Any] = {}
        self.strategies: Dict[str, Any] = {}
        self.metrics = PerformanceMetrics()
        self.epistemic_health = EpistemicHealth()
        self._query_classifier: Optional[Any] = None
        self._response_synthesizer: Optional[Any] = None
        self._drift_monitor: Optional[Any] = None
        self._feedback_tracker: Optional[Any] = None
        self._circuit_breakers: Dict[str, Any] = {}
        self._quarantined_models: set = set()
        self._quarantine_lock = threading.RLock()  # Thread-safe quarantine operations

        # Phase 1 Week 1: Cost tracking and budget management
        try:
            from nexus.providers.cost import CostTracker, BudgetManager
            self.cost_tracker = CostTracker()
            self.budget_manager = BudgetManager(self.cost_tracker)
        except ImportError as e:
            logger.warning(f"Cost tracking not available: {e}")
            self.cost_tracker = None
            self.budget_manager = None

        # Phase 1 Week 2: Semantic caching
        try:
            from nexus.core.cache import CacheManager, MemoryBackend
            memory_backend = MemoryBackend()
            self.cache_manager = CacheManager(backend=memory_backend)
        except ImportError as e:
            logger.warning(f"Cache manager not available: {e}")
            self.cache_manager = None
        except Exception as e:
            logger.warning(f"Cache manager initialization failed: {e}")
            self.cache_manager = None

        # Phase 2 Week 1-2: Advanced Knowledge Graph (optional)
        try:
            from nexus.providers.graph import AdvancedKnowledgeGraph
            self.knowledge_graph = AdvancedKnowledgeGraph()
        except ImportError:
            logger.debug("Knowledge graph not available")
            self.knowledge_graph = None

        # Phase 2 Week 9-10: Tiered Memory System (optional)
        try:
            from nexus.providers.memory import TieredMemory
            self.tiered_memory = TieredMemory(
                storage_path="/tmp/unified_intelligence_memory",
                enable_metrics=True
            )
            self.tiered_memory.start_maintenance()
        except ImportError:
            logger.debug("Tiered memory not available")
            self.tiered_memory = None

        # Initialize components
        self._initialize_models()
        self._initialize_strategies()
        self._initialize_monitoring()

    def _initialize_models(self) -> None:
        """Initialize AI model adapters."""
        # Import model adapters
        from nexus.providers.adapters.openai_adapter import OpenAIModelAdapter
        from nexus.providers.adapters.anthropic_adapter import AnthropicModelAdapter
        from nexus.providers.adapters.registry import get_model, MODEL_REGISTRY
        from nexus.providers.ensemble.types import ModelProvider
        import os

        # This will load models based on configuration
        model_list = self.config.get_model_list()
        print(f"Initializing {len(model_list)} models...")

        # Initialize actual model adapters
        for model_name in model_list:
            try:
                # Get model info from registry
                model_info = get_model(model_name)

                # Create appropriate adapter based on provider
                adapter = None
                if model_info.provider == ModelProvider.OPENAI:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        adapter = OpenAIModelAdapter(model_info, api_key=api_key)
                elif model_info.provider == ModelProvider.ANTHROPIC:
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    if api_key:
                        adapter = AnthropicModelAdapter(model_info, api_key=api_key)

                # Store model with adapter
                self.models[model_name] = {
                    "name": model_name,
                    "adapter": adapter,
                    "enabled": adapter is not None,
                    "weight": 1.0,
                    "performance_history": [],
                    "info": model_info,
                }

                if adapter is None:
                    print(f"[SKIP] Model {model_name} skipped: No API key found")
                else:
                    print(f"[OK] Model {model_name} initialized")

            except KeyError:
                print(f"[ERROR] Model {model_name} not found in registry")
                # Create stub for unknown models
                self.models[model_name] = {
                    "name": model_name,
                    "adapter": None,
                    "enabled": False,
                    "weight": 1.0,
                    "performance_history": [],
                }

    def _initialize_strategies(self) -> None:
        """Initialize ensemble strategies."""
        # Import strategy implementations
        from nexus.providers.strategies.ensemble_strategies import (
            WeightedVotingStrategy,
            CascadingStrategy,
            DynamicWeightStrategy,
            MajorityVotingStrategy,
            CostOptimizedStrategy,
            SynthesizedStrategy,
        )

        # Register all available strategies
        self.strategies = {
            EnsembleStrategy.WEIGHTED_VOTING: WeightedVotingStrategy(),
            EnsembleStrategy.CASCADING: CascadingStrategy(
                confidence_threshold=0.7,
                max_cascades=3
            ),
            EnsembleStrategy.DYNAMIC_WEIGHT: DynamicWeightStrategy(
                learning_rate=0.1,
                score_weight=0.5,
                speed_weight=0.3,
                cost_weight=0.2
            ),
            EnsembleStrategy.MAJORITY_VOTING: MajorityVotingStrategy(
                similarity_threshold=0.8
            ),
            EnsembleStrategy.COST_OPTIMIZED: CostOptimizedStrategy(
                min_quality_threshold=0.6,
                cost_weight=0.4,
                quality_weight=0.6
            ),
            EnsembleStrategy.SYNTHESIZED: SynthesizedStrategy(
                min_sentence_score=0.6,
                similarity_threshold=0.7,
                max_sentences=10
            ),
        }

        print(f"[OK] Initialized {len([s for s in self.strategies.values() if s is not None])} ensemble strategies")

    def _initialize_monitoring(self) -> None:
        """Initialize monitoring and epistemic health systems."""
        if self.config.ensemble.enable_epistemic_monitoring:
            # TODO: Initialize drift monitor from fluffy-eureka
            pass

        if self.config.safety.enable_circuit_breakers:
            # TODO: Initialize circuit breakers from combo1
            pass

    async def process(self, request: EnsembleRequest) -> EnsembleResponse:
        """
        Process a query through the unified ensemble system.

        Args:
            request: Ensemble request containing query and parameters

        Returns:
            Synthesized ensemble response with confidence and health metrics

        Raises:
            BudgetExceededException: If user budget is exceeded
        """
        start_time = time.time()

        # Phase 1 Week 2: Check cache first
        cached_result = self.cache_manager.get(request.query)
        if cached_result is not None:
            # Cache HIT - return cached response
            logger.info(
                f"✅ Cache HIT ({cached_result.level.value}): "
                f"{request.query[:50]}... (saved ~${0.05:.3f})"
            )

            # Build response from cache
            return EnsembleResponse(
                request_id=request.request_id,
                content=cached_result.value['content'],
                confidence=cached_result.value.get('confidence', 0.8),
                strategy_used=cached_result.value.get('strategy_used', 'cached'),
                model_responses=cached_result.value.get('model_responses', []),
                models_queried=cached_result.value.get('models_queried', 0),
                total_latency_ms=cached_result.latency_ms,
                total_cost_usd=0.0,  # No cost for cached responses
                epistemic_health=self.epistemic_health.overall_health,
                metadata={
                    'cached': True,
                    'cache_level': cached_result.level.value,
                    'similarity_score': cached_result.similarity_score,
                },
            )

        # Phase 1 Week 1: Check budget before processing
        if request.user_id:
            from nexus.providers.cost import BudgetExceededException

            # Estimate cost based on selected models (rough estimate)
            estimated_cost_per_model = 0.05  # Conservative estimate
            estimated_total_cost = estimated_cost_per_model * (request.max_models or 3)

            if self.budget_manager.would_exceed_budget(
                request.user_id, "user", estimated_total_cost
            ):
                raise BudgetExceededException(
                    f"Request would exceed budget for user {request.user_id}"
                )

            # Check for alerts
            self.budget_manager.check_and_alert(request.user_id, "user")

        # Step 1: Classify query type (from combo1)
        if request.query_type is None:
            request.query_type = await self._classify_query(request.query)

        # Step 2: Check epistemic health (from fluffy-eureka)
        if self.config.ensemble.enable_epistemic_monitoring:
            await self._check_epistemic_health()

        # Step 3: Select models based on query type and strategy
        selected_models = await self._select_models(request)

        # Step 4: Execute models in parallel
        model_responses = await self._execute_models(selected_models, request)

        # Step 5: Synthesize responses (from combo1)
        if self.config.ensemble.enable_response_synthesis:
            synthesized_content = await self._synthesize_responses(model_responses)
        else:
            # Fallback to strategy-based selection
            synthesized_content = await self._select_best_response(model_responses)

        # Step 6: Calculate confidence (from combo1)
        confidence = await self._calibrate_confidence(model_responses)

        # Step 7: Update metrics and learning
        await self._update_metrics(request, model_responses)

        # Calculate totals
        total_latency = (time.time() - start_time) * 1000  # Convert to ms
        total_cost = sum(r.cost_usd for r in model_responses)

        # Phase 1 Week 1: Record costs for each model
        for response in model_responses:
            if response.cost_usd > 0:  # Only record if there was actual cost
                self.cost_tracker.record_cost(
                    model_name=response.model_name,
                    provider=response.provider.value if hasattr(response.provider, 'value') else str(response.provider),
                    tokens_used=response.tokens_used,
                    cost_usd=response.cost_usd,
                    user_id=request.user_id,
                    request_id=str(request.request_id)
                )

        # Build response
        response = EnsembleResponse(
            request_id=request.request_id,
            content=synthesized_content,
            confidence=confidence,
            strategy_used=self.config.ensemble.default_strategy.value,
            model_responses=model_responses,
            models_queried=len(model_responses),
            total_latency_ms=total_latency,
            total_cost_usd=total_cost,
            epistemic_health=self.epistemic_health.overall_health,
            metadata={
                "query_type": request.query_type.value if request.query_type else None,
                "synthesis_enabled": self.config.ensemble.enable_response_synthesis,
                "cached": False,
            },
        )

        # Record request context for feedback correlation
        # Use the best/primary model if multiple were used
        primary_model = model_responses[0].model_name if model_responses else "unknown"
        self.record_request_context(
            request_id=request.request_id,
            model_name=primary_model,
            query_type=request.query_type.value if request.query_type else None,
            latency_ms=total_latency,
        )

        # Phase 1 Week 2: Cache the response
        cache_value = {
            'content': synthesized_content,
            'confidence': confidence,
            'strategy_used': self.config.ensemble.default_strategy.value,
            'model_responses': model_responses,
            'models_queried': len(model_responses),
        }

        # Cache with appropriate TTL based on confidence
        if confidence >= 0.8:
            cache_ttl = 3600  # 1 hour for high-confidence responses
        elif confidence >= 0.6:
            cache_ttl = 1800  # 30 minutes for medium-confidence
        else:
            cache_ttl = 600   # 10 minutes for low-confidence

        self.cache_manager.set(request.query, cache_value, ttl=cache_ttl)
        logger.debug(f"💾 Cached response for: {request.query[:50]}... (ttl={cache_ttl}s)")

        # Phase 2 Week 1-2: Store knowledge in graph
        # Only store high-confidence responses
        if confidence >= 0.7:
            from nexus.providers.graph import NodeType, VerificationLevel

            # Determine node type based on query type
            if request.query_type == QueryType.FACTUAL:
                node_type = NodeType.FACT
            elif request.query_type == QueryType.ANALYTICAL:
                node_type = NodeType.CONCEPT
            else:
                node_type = NodeType.HYPOTHESIS

            # Determine verification level based on confidence and model consensus
            if confidence >= 0.9 and len(model_responses) >= 3:
                verification = VerificationLevel.CONSENSUS
            elif confidence >= 0.8 and len(model_responses) >= 2:
                verification = VerificationLevel.MULTI_SOURCE
            elif confidence >= 0.7:
                verification = VerificationLevel.PEER_REVIEWED
            else:
                verification = VerificationLevel.UNVERIFIED

            # Add knowledge node
            try:
                node_id = self.knowledge_graph.add_node(
                    content=f"Q: {request.query[:200]}... A: {synthesized_content[:500]}...",
                    node_type=node_type,
                    verification=verification,
                    belief_score=confidence,
                    sources=[f"ensemble-{request.request_id}"],
                    models_used=[mr.model_name for mr in model_responses],
                    timestamp=request.metadata.get('timestamp') if request.metadata else None
                )
                logger.debug(f"🧠 Stored knowledge node: {node_id}")
            except Exception as e:
                logger.warning(f"Failed to store knowledge: {e}")

        # Phase 2 Week 9-10: Store conversation context in tiered memory
        # Store for user sessions and conversation tracking
        if request.user_id:
            session_key = f"user_{request.user_id}_session"

            # Retrieve existing session or create new
            existing_session = self.tiered_memory.retrieve(session_key)

            if existing_session is None:
                session_data = {
                    'user_id': request.user_id,
                    'queries': [],
                    'last_query_time': datetime.now(timezone.utc).isoformat(),
                    'total_queries': 0,
                }
            else:
                session_data = existing_session

            # Update session with new query
            session_data['queries'].append({
                'request_id': request.request_id,
                'query': request.query[:500],  # Limit to avoid huge sessions
                'response': synthesized_content[:500],
                'confidence': confidence,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })

            # Keep only last 50 queries to manage size
            if len(session_data['queries']) > 50:
                session_data['queries'] = session_data['queries'][-50:]

            session_data['last_query_time'] = datetime.now(timezone.utc).isoformat()
            session_data['total_queries'] = session_data.get('total_queries', 0) + 1

            # Store updated session
            try:
                self.tiered_memory.store(session_key, session_data)
                logger.debug(f"💾 Stored session for user: {request.user_id}")
            except Exception as e:
                logger.warning(f"Failed to store session: {e}")

        return response

    async def _classify_query(self, query: str) -> QueryType:
        """
        Classify query type for optimal model selection and response handling.

        Uses pattern-based classification to identify query intent:
        - FACTUAL: Questions seeking specific facts or data
        - ANALYTICAL: Complex reasoning, comparisons, analysis
        - CREATIVE: Content generation, brainstorming, writing
        - TECHNICAL: Programming, debugging, technical implementation
        - CONVERSATIONAL: General chat, opinions, discussions

        Args:
            query: User query text

        Returns:
            Classified query type
        """
        query_lower = query.lower().strip()

        # Technical/Code-related patterns
        technical_patterns = [
            r'\b(code|program|function|class|method|api|debug|error|exception)\b',
            r'\b(python|javascript|java|c\+\+|rust|go|typescript)\b',
            r'\b(implement|refactor|optimize|fix bug|write a script)\b',
            r'```',  # Code blocks
            r'\b(def |class |function |const |let |var |import |from )\b',
        ]

        for pattern in technical_patterns:
            if re.search(pattern, query_lower):
                return QueryType.TECHNICAL

        # Factual question patterns
        factual_patterns = [
            r'^(what is|what are|who is|who are|when did|when was|where is|where are)\b',
            r'^(how many|how much|how old|how long|how far)\b',
            r'\b(define|definition of|meaning of)\b',
            r'^(is it true that|did|does|do|was|were|has|have|can|will)\b',
            r'\b(capital of|population of|date of|year of)\b',
        ]

        for pattern in factual_patterns:
            if re.search(pattern, query_lower):
                return QueryType.FACTUAL

        # Analytical patterns
        analytical_patterns = [
            r'\b(analyze|analysis|compare|contrast|evaluate|assess)\b',
            r'\b(pros and cons|advantages|disadvantages|trade-?offs)\b',
            r'\b(why does|why is|why are|how does|how do)\b',
            r'\b(explain|elaborate|clarify|discuss|examine)\b',
            r'\b(cause|effect|impact|implication|consequence)\b',
            r'\b(relationship between|difference between|similarity between)\b',
        ]

        for pattern in analytical_patterns:
            if re.search(pattern, query_lower):
                return QueryType.ANALYTICAL

        # Creative patterns
        creative_patterns = [
            r'\b(write|create|compose|generate|draft|design)\b',
            r'\b(story|poem|essay|article|blog|content)\b',
            r'\b(brainstorm|ideas|creative|imagine|invent)\b',
            r'\b(suggest|recommend|propose|come up with)\b',
            r'\b(slogan|tagline|headline|title|name)\b',
        ]

        for pattern in creative_patterns:
            if re.search(pattern, query_lower):
                return QueryType.CREATIVE

        # Default to conversational for general queries
        return QueryType.CONVERSATIONAL

    async def _check_epistemic_health(self) -> None:
        """
        Check epistemic health by analyzing model consistency and drift.

        Monitors:
        - Model agreement rates across recent queries
        - Confidence distribution shifts
        - Error rate trends
        - Response latency patterns

        Updates self.epistemic_health with current metrics.
        """
        # Calculate consistency from recent model performance
        total_requests = self.metrics.total_requests
        if total_requests == 0:
            self.epistemic_health.consistency_score = 1.0
            self.epistemic_health.drift_score = 0.0
            return

        # Calculate success rate
        success_rate = self.metrics.successful_requests / max(1, total_requests)

        # Calculate average confidence across models
        confidences = []
        for model_name, perf in self.metrics.model_performance.items():
            if perf.get('requests', 0) > 0:
                confidences.append(perf.get('avg_confidence', 0.5))

        avg_confidence = sum(confidences) / max(1, len(confidences)) if confidences else 0.5

        # Calculate confidence variance (lower is better)
        if len(confidences) > 1:
            variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
            confidence_stability = 1.0 - min(1.0, variance * 4)  # Scale variance
        else:
            confidence_stability = 1.0

        # Combine metrics for consistency score
        self.epistemic_health.consistency_score = (
            success_rate * 0.4 +
            avg_confidence * 0.3 +
            confidence_stability * 0.3
        )

        # Drift score: inverse of consistency with baseline
        # Higher drift means models are performing differently than expected
        baseline_consistency = 0.85  # Expected healthy consistency
        self.epistemic_health.drift_score = max(0, baseline_consistency - self.epistemic_health.consistency_score)

        # Update overall health
        self.epistemic_health.overall_health = (
            self.epistemic_health.consistency_score * 0.7 +
            (1 - self.epistemic_health.drift_score) * 0.3
        )

        # Log if health is degraded
        if self.epistemic_health.overall_health < 0.7:
            logger.warning(
                f"⚠️ Epistemic health degraded: "
                f"consistency={self.epistemic_health.consistency_score:.2f}, "
                f"drift={self.epistemic_health.drift_score:.2f}"
            )

    async def _select_models(self, request: EnsembleRequest) -> List[str]:
        """
        Select which models to query based on request and strategy.

        Args:
            request: Ensemble request

        Returns:
            List of model names to query
        """
        # Filter out quarantined models (thread-safe)
        with self._quarantine_lock:
            available_models = [
                name for name in self.models.keys() if name not in self._quarantined_models
            ]

        # Apply max_models limit if specified
        if request.max_models and request.max_models < len(available_models):
            # TODO: Implement smart model selection based on query type
            available_models = available_models[: request.max_models]

        return available_models

    async def _execute_models(
        self, model_names: List[str], request: EnsembleRequest
    ) -> List[ModelResponse]:
        """
        Execute multiple models in parallel.

        Args:
            model_names: List of models to query
            request: Ensemble request

        Returns:
            List of model responses
        """
        if self.config.ensemble.parallel_execution:
            # Execute in parallel
            tasks = [self._query_single_model(name, request) for name in model_names]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and convert to ModelResponse
            model_responses = []
            for name, response in zip(model_names, responses):
                if isinstance(response, Exception):
                    # Create error response
                    model_responses.append(
                        ModelResponse(
                            model_name=name,
                            provider="unknown",
                            content="",
                            error=str(response),
                        )
                    )
                else:
                    model_responses.append(response)

            return model_responses
        else:
            # Execute sequentially
            responses = []
            for name in model_names:
                response = await self._query_single_model(name, request)
                responses.append(response)
            return responses

    async def _query_single_model(
        self, model_name: str, request: EnsembleRequest
    ) -> ModelResponse:
        """
        Query a single model using real API calls.

        Args:
            model_name: Name of model to query
            request: Ensemble request

        Returns:
            Model response
        """
        start_time = time.time()

        # Get model configuration
        model_config = self.models.get(model_name)
        if not model_config or not model_config.get("adapter"):
            # Model not available - return error response
            latency = (time.time() - start_time) * 1000
            return ModelResponse(
                model_name=model_name,
                provider="unknown",
                content="",
                error=f"Model {model_name} not initialized or no API key available",
                latency_ms=latency,
                tokens_used=0,
                cost_usd=0.0,
            )

        adapter = model_config["adapter"]
        model_info = model_config["info"]

        try:
            # Call actual model adapter
            response = await adapter.generate(
                prompt=request.query,
                temperature=request.temperature if hasattr(request, "temperature") else 0.7,
                max_tokens=request.max_tokens if hasattr(request, "max_tokens") else 2048,
            )

            # Response is already a ModelResponse object from the adapter
            return response

        except Exception as e:
            # Handle errors gracefully
            latency = (time.time() - start_time) * 1000
            print(f"❌ Error querying {model_name}: {str(e)}")

            return ModelResponse(
                model_name=model_name,
                provider=model_info.provider.value if hasattr(model_info, "provider") else "unknown",
                content="",
                error=str(e),
                latency_ms=latency,
                tokens_used=0,
                cost_usd=0.0,
            )

    async def _synthesize_responses(self, responses: List[ModelResponse]) -> str:
        """
        Synthesize multiple responses into a single coherent response.

        Uses a sentence-level synthesis algorithm that:
        1. Splits responses into sentences
        2. Scores sentences based on agreement across models
        3. Selects best sentences avoiding redundancy
        4. Reconstructs coherent response

        Args:
            responses: List of model responses

        Returns:
            Synthesized response text
        """
        # Filter successful responses
        successful = [r for r in responses if r.error is None and r.content]

        if not successful:
            return "I apologize, but I was unable to generate a response."

        if len(successful) == 1:
            return successful[0].content

        # For 2+ responses, perform sentence-level synthesis
        from difflib import SequenceMatcher

        def similarity(a: str, b: str) -> float:
            """Calculate text similarity between two strings."""
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        def split_sentences(text: str) -> List[str]:
            """Split text into sentences."""
            # Simple sentence splitter
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return [s.strip() for s in sentences if s.strip()]

        # Collect all sentences with their source info
        all_sentences = []
        for resp in successful:
            sentences = split_sentences(resp.content)
            for sent in sentences:
                all_sentences.append({
                    'text': sent,
                    'confidence': resp.confidence,
                    'model': resp.model_name,
                    'agreement_score': 0.0,
                })

        # Score sentences by agreement across models
        for i, sent_info in enumerate(all_sentences):
            agreement_count = 0
            for other in all_sentences:
                if other['model'] != sent_info['model']:
                    if similarity(sent_info['text'], other['text']) > 0.6:
                        agreement_count += 1
            # Agreement score: how many other models have similar content
            sent_info['agreement_score'] = agreement_count / max(1, len(successful) - 1)

        # Combine confidence and agreement for final score
        for sent_info in all_sentences:
            sent_info['final_score'] = (
                sent_info['confidence'] * 0.6 +
                sent_info['agreement_score'] * 0.4
            )

        # Sort by final score
        all_sentences.sort(key=lambda x: x['final_score'], reverse=True)

        # Select top sentences avoiding redundancy
        selected = []
        max_sentences = 10

        for sent_info in all_sentences:
            if len(selected) >= max_sentences:
                break

            # Check for redundancy with already selected sentences
            is_redundant = False
            for existing in selected:
                if similarity(sent_info['text'], existing['text']) > 0.7:
                    is_redundant = True
                    break

            if not is_redundant:
                selected.append(sent_info)

        # Reconstruct response
        if not selected:
            # Fallback to highest confidence response
            best_response = max(successful, key=lambda r: r.confidence)
            return best_response.content

        # Sort selected sentences to maintain logical order
        # Use the order from the highest-confidence response as reference
        best_resp = max(successful, key=lambda r: r.confidence)
        best_sentences = split_sentences(best_resp.content)

        def get_order_index(sent_text: str) -> int:
            """Get order index based on best response."""
            for i, bs in enumerate(best_sentences):
                if similarity(sent_text, bs) > 0.5:
                    return i
            return len(best_sentences)  # Put at end if not found

        selected.sort(key=lambda x: get_order_index(x['text']))

        return ' '.join(s['text'] for s in selected)

    async def _select_best_response(self, responses: List[ModelResponse]) -> str:
        """
        Select best response using strategy (fallback if synthesis disabled).

        Args:
            responses: List of model responses

        Returns:
            Selected response text
        """
        successful = [r for r in responses if r.error is None and r.content]

        if not successful:
            return "I apologize, but I was unable to generate a response."

        # Simple selection: highest confidence
        best_response = max(successful, key=lambda r: r.confidence)
        return best_response.content

    async def _calibrate_confidence(self, responses: List[ModelResponse]) -> float:
        """
        Calibrate confidence score using multi-factor analysis.

        Considers:
        - Individual model confidence scores
        - Model agreement (consensus boost)
        - Response length consistency
        - Historical model accuracy
        - Error rate penalty

        Args:
            responses: List of model responses

        Returns:
            Calibrated confidence score (0-1)
        """
        from difflib import SequenceMatcher

        successful = [r for r in responses if r.error is None and r.content]
        failed = [r for r in responses if r.error is not None]

        if not successful:
            return 0.0

        # 1. Base confidence: weighted average by historical performance
        weighted_sum = 0.0
        weight_total = 0.0

        for resp in successful:
            # Get historical performance weight
            perf = self.metrics.model_performance.get(resp.model_name, {})
            historical_accuracy = perf.get('avg_confidence', 0.5)
            weight = 0.5 + historical_accuracy * 0.5  # Weight between 0.5 and 1.0

            weighted_sum += resp.confidence * weight
            weight_total += weight

        base_confidence = weighted_sum / max(0.001, weight_total)

        # 2. Consensus boost: higher confidence if models agree
        if len(successful) >= 2:
            agreement_scores = []
            for i, r1 in enumerate(successful):
                for r2 in successful[i+1:]:
                    similarity = SequenceMatcher(
                        None,
                        r1.content.lower()[:500],
                        r2.content.lower()[:500]
                    ).ratio()
                    agreement_scores.append(similarity)

            avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.5
            consensus_boost = (avg_agreement - 0.5) * 0.2  # -0.1 to +0.1
        else:
            consensus_boost = 0.0

        # 3. Response length consistency penalty
        lengths = [len(r.content) for r in successful]
        if len(lengths) > 1 and max(lengths) > 0:
            avg_length = sum(lengths) / len(lengths)
            length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            normalized_variance = length_variance / (avg_length ** 2) if avg_length > 0 else 0
            length_penalty = min(0.1, normalized_variance * 0.5)  # Max -0.1 penalty
        else:
            length_penalty = 0.0

        # 4. Error rate penalty
        total_models = len(successful) + len(failed)
        error_rate = len(failed) / max(1, total_models)
        error_penalty = error_rate * 0.15  # Max -0.15 penalty

        # 5. Calculate final calibrated confidence
        calibrated = base_confidence + consensus_boost - length_penalty - error_penalty

        # Clamp to valid range
        return max(0.0, min(1.0, calibrated))

    async def _update_metrics(
        self, request: EnsembleRequest, responses: List[ModelResponse]
    ) -> None:
        """
        Update performance metrics and learning systems.

        Args:
            request: Original request
            responses: Model responses
        """
        self.metrics.total_requests += 1

        successful = [r for r in responses if r.error is None]
        if successful:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1

        # Update running averages
        for response in successful:
            if response.model_name not in self.metrics.model_performance:
                self.metrics.model_performance[response.model_name] = {
                    "requests": 0,
                    "avg_confidence": 0.0,
                    "avg_latency": 0.0,
                }

            perf = self.metrics.model_performance[response.model_name]
            perf["requests"] += 1

            # Update running averages
            n = perf["requests"]
            perf["avg_confidence"] = (perf["avg_confidence"] * (n - 1) + response.confidence) / n
            perf["avg_latency"] = (perf["avg_latency"] * (n - 1) + response.latency_ms) / n

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.

        Returns:
            Performance metrics object
        """
        return self.metrics

    def get_epistemic_health(self) -> EpistemicHealth:
        """
        Get current epistemic health status.

        Returns:
            Epistemic health object
        """
        return self.epistemic_health

    async def provide_feedback(
        self, request_id: UUID, feedback_score: float, feedback_text: Optional[str] = None
    ) -> None:
        """
        Provide feedback on a response for meta-learning.

        This method integrates with the FeedbackTracker to record user feedback
        and uses it to update model weights for improved routing.

        Args:
            request_id: Request identifier
            feedback_score: Feedback score (0-1)
            feedback_text: Optional feedback text
        """
        # Validate score
        feedback_score = max(0.0, min(1.0, feedback_score))

        # Find the model that generated this response from recent history
        model_name = self._find_model_for_request(request_id)
        if not model_name:
            logger.warning(f"Cannot find model for request {request_id}")
            return

        # Get query type if available from history
        query_type = self._get_query_type_for_request(request_id)

        # Record feedback using the feedback tracker
        if self._feedback_tracker:
            self._feedback_tracker.record_feedback(
                request_id=request_id,
                model_name=model_name,
                feedback_score=feedback_score,
                feedback_text=feedback_text,
                query_type=query_type,
                response_latency_ms=self._get_latency_for_request(request_id),
                metadata={"source": "user_feedback"},
            )
            logger.debug(f"Recorded feedback for {model_name}: score={feedback_score:.2f}")

            # Check if we should update weights based on accumulated feedback
            await self._maybe_update_weights_from_feedback()
        else:
            # Fallback: update model performance directly
            self._update_model_score_from_feedback(model_name, feedback_score)

    def _find_model_for_request(self, request_id: UUID) -> Optional[str]:
        """Find which model generated a response for a given request."""
        # Check request history (maintained by recent requests)
        if hasattr(self, '_request_history') and request_id in self._request_history:
            return self._request_history[request_id].get("model_name")

        # Fallback: check model performance for any model
        # This is a best-effort approach when request history isn't available
        for model_name in self.models.keys():
            return model_name  # Return first available model as fallback

        return None

    def _get_query_type_for_request(self, request_id: UUID) -> Optional[str]:
        """Get the query type for a given request."""
        if hasattr(self, '_request_history') and request_id in self._request_history:
            return self._request_history[request_id].get("query_type")
        return None

    def _get_latency_for_request(self, request_id: UUID) -> Optional[float]:
        """Get the response latency for a given request."""
        if hasattr(self, '_request_history') and request_id in self._request_history:
            return self._request_history[request_id].get("latency_ms")
        return None

    def _update_model_score_from_feedback(
        self, model_name: str, feedback_score: float
    ) -> None:
        """Update model performance metrics based on feedback."""
        if model_name not in self.metrics.model_performance:
            self.metrics.model_performance[model_name] = {
                "requests": 0,
                "avg_confidence": 0.5,
                "avg_latency": 0.0,
                "feedback_score": 0.5,
                "feedback_count": 0,
            }

        perf = self.metrics.model_performance[model_name]

        # Update feedback score (running average)
        if "feedback_count" not in perf:
            perf["feedback_count"] = 0
            perf["feedback_score"] = 0.5

        perf["feedback_count"] += 1
        n = perf["feedback_count"]
        perf["feedback_score"] = (perf["feedback_score"] * (n - 1) + feedback_score) / n

        logger.debug(
            f"Updated {model_name} feedback: score={perf['feedback_score']:.2f} "
            f"(count={n})"
        )

    async def _maybe_update_weights_from_feedback(self) -> None:
        """
        Check if we should update model weights based on accumulated feedback.

        This implements a feedback-based learning system that adjusts model
        routing based on user feedback patterns.
        """
        if not self._feedback_tracker:
            return

        # Get weight recommendations from the feedback tracker
        recommendations = self._feedback_tracker.get_weight_recommendations()

        if not recommendations:
            return

        # Apply weight adjustments to model performance metrics
        for model_name, adjustment in recommendations.items():
            if model_name in self.metrics.model_performance:
                perf = self.metrics.model_performance[model_name]

                # Store the feedback-based weight adjustment
                perf["feedback_weight_adj"] = adjustment

                # Blend feedback into confidence score for routing
                # This affects model selection in weighted strategies
                if "feedback_score" in perf:
                    # Adjust effective confidence based on feedback
                    base_conf = perf.get("avg_confidence", 0.5)
                    feedback_influence = 0.2  # 20% weight to feedback
                    perf["effective_confidence"] = (
                        base_conf * (1 - feedback_influence) +
                        perf["feedback_score"] * feedback_influence
                    )

        logger.debug(f"Applied weight recommendations from feedback: {len(recommendations)} models")

    def record_request_context(
        self,
        request_id: UUID,
        model_name: str,
        query_type: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Record request context for later feedback correlation.

        Args:
            request_id: The request identifier
            model_name: Model that handled the request
            query_type: Type of query
            latency_ms: Response latency
        """
        if not hasattr(self, '_request_history'):
            from collections import OrderedDict
            self._request_history: Dict[UUID, Dict[str, Any]] = OrderedDict()

        # Store context
        self._request_history[request_id] = {
            "model_name": model_name,
            "query_type": query_type,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc),
        }

        # Keep only last 1000 requests to prevent memory bloat
        while len(self._request_history) > 1000:
            self._request_history.popitem(last=False)

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get feedback statistics for all models.

        Returns:
            Dictionary with feedback statistics per model
        """
        if not self._feedback_tracker:
            return {"available": False, "reason": "Feedback tracker not initialized"}

        all_stats = self._feedback_tracker.get_all_model_stats()

        return {
            "available": True,
            "models": {
                name: {
                    "total_feedback": stats.total_feedback,
                    "average_score": stats.average_score,
                    "positive_count": stats.positive_count,
                    "negative_count": stats.negative_count,
                    "neutral_count": stats.neutral_count,
                    "recent_trend": stats.recent_trend,
                    "query_type_scores": stats.query_type_scores,
                }
                for name, stats in all_stats.items()
            },
            "recommendations": self._feedback_tracker.get_weight_recommendations(),
        }

    def quarantine_model(self, model_name: str, reason: str) -> None:
        """
        Quarantine a misbehaving model (thread-safe).

        Args:
            model_name: Name of model to quarantine
            reason: Reason for quarantine
        """
        with self._quarantine_lock:
            self._quarantined_models.add(model_name)
        print(f"Model {model_name} quarantined: {reason}")

    def release_from_quarantine(self, model_name: str) -> None:
        """
        Release a model from quarantine (thread-safe).

        Args:
            model_name: Name of model to release
        """
        with self._quarantine_lock:
            self._quarantined_models.discard(model_name)
        print(f"Model {model_name} released from quarantine")

    def get_quarantined_models(self) -> List[str]:
        """
        Get list of currently quarantined models (thread-safe).

        Returns:
            List of quarantined model names
        """
        with self._quarantine_lock:
            return list(self._quarantined_models)

    async def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive ensemble status for GUI display.

        Returns:
            Dictionary with models, strategy, health, and quarantined info
        """
        models_list = []
        for name, adapter in self.models.items():
            perf = self.metrics.model_performance.get(name, {})
            status = "active"
            if name in self._quarantined_models:
                status = "quarantined"
            elif perf.get("avg_latency", 0) > 5000:  # > 5s latency
                status = "degraded"

            models_list.append({
                "id": name,
                "name": name,
                "provider": getattr(adapter, 'provider', 'unknown'),
                "status": status,
                "health": int(perf.get("avg_confidence", 0.9) * 100),
                "requests": perf.get("requests", 0),
                "avg_latency": perf.get("avg_latency", 0) / 1000,  # Convert to seconds
                "success_rate": (perf.get("successes", 0) / max(perf.get("requests", 1), 1)) * 100,
                "latency_ms": perf.get("avg_latency", 0),
                "cost": perf.get("total_cost", 0),
            })

        # Get current strategy name
        strategy_name = "weighted_quality"
        if hasattr(self, '_current_strategy'):
            strategy_name = self._current_strategy

        # Calculate overall health
        total_health = sum(m["health"] for m in models_list) / max(len(models_list), 1)
        degraded_count = len([m for m in models_list if m["status"] == "degraded"])
        quarantined_count = len(self._quarantined_models)

        return {
            "models": models_list,
            "strategy": strategy_name,
            "strategies_available": ["weighted_quality", "lowest_latency", "lowest_cost", "round_robin", "capability_match"],
            "health": {
                "overall": int(total_health),
                "degraded_count": degraded_count,
                "quarantined_count": quarantined_count,
            },
            "quarantined": list(self._quarantined_models),
        }

    async def set_strategy(self, strategy: str) -> None:
        """
        Set the ensemble selection strategy.

        Args:
            strategy: Strategy name (weighted_quality, lowest_latency, etc.)
        """
        self._current_strategy = strategy
        # Could also reconfigure internal strategy objects here

    # ========================================================================
    # Consciousness Orchestration Integration (Nexus Phase 3)
    # ========================================================================

    async def enable_consciousness_mode(
        self,
        orchestration_interval: float = 2.0,
        enable_autonomous_learning: bool = True,
        enable_safety_monitoring: bool = True,
        enable_value_learning: bool = True
    ) -> bool:
        """
        Enable consciousness-inspired orchestration mode.

        This activates the Consciousness Orchestrator which coordinates all
        integrated Nexus modules for advanced AI capabilities:
        - Autonomous learning cycles
        - Continuous safety monitoring
        - Value learning and alignment
        - Module health tracking
        - Consciousness coherence metrics

        Args:
            orchestration_interval: Seconds between orchestration cycles (default: 2.0)
            enable_autonomous_learning: Enable autonomous learning
            enable_safety_monitoring: Enable continuous safety monitoring
            enable_value_learning: Enable value learning from experiences

        Returns:
            True if consciousness mode successfully enabled

        Example:
            >>> ensemble = UnifiedEnsemble()
            >>> await ensemble.enable_consciousness_mode()
            >>> status = ensemble.get_consciousness_status()
            >>> print(f"Consciousness Coherence: {status['metrics']['consciousness_coherence']:.1f}%")
        """
        try:
            from nexus.cog_eng.consciousness.orchestrator import ConsciousnessOrchestrator

            # Create consciousness orchestrator if not exists
            if not hasattr(self, '_consciousness_orchestrator'):
                self._consciousness_orchestrator = ConsciousnessOrchestrator(
                    orchestration_interval=orchestration_interval,
                    enable_autonomous_learning=enable_autonomous_learning,
                    enable_safety_monitoring=enable_safety_monitoring,
                    enable_value_learning=enable_value_learning
                )

                # Initialize consciousness system
                success = await self._consciousness_orchestrator.initialize()
                if not success:
                    print("⚠️ Consciousness initialization failed")
                    return False

                # Start orchestration
                await self._consciousness_orchestrator.start_orchestration()

                print("🧠 Consciousness mode enabled")
                return True
            else:
                print("⚠️ Consciousness mode already enabled")
                return True

        except Exception as e:
            print(f"❌ Failed to enable consciousness mode: {e}")
            return False

    async def disable_consciousness_mode(self) -> None:
        """
        Disable consciousness orchestration mode.

        Gracefully stops the consciousness orchestrator and cleans up resources.
        """
        if hasattr(self, '_consciousness_orchestrator'):
            await self._consciousness_orchestrator.shutdown()
            delattr(self, '_consciousness_orchestrator')
            print("🛑 Consciousness mode disabled")
        else:
            print("⚠️ Consciousness mode not enabled")

    def get_consciousness_status(self) -> Dict[str, Any]:
        """
        Get consciousness system status.

        Returns:
            Dictionary with consciousness metrics and module health,
            or None if consciousness mode not enabled

        Returns:
            {
                'initialized': bool,
                'running': bool,
                'uptime_seconds': float,
                'cycle_count': int,
                'modules': {...},
                'metrics': {
                    'consciousness_coherence': float,
                    'creative_intelligence': float,
                    'learning_efficiency': float,
                    'safety_compliance': float,
                    'epistemic_health': float,
                    'value_alignment': float
                }
            }
        """
        if not hasattr(self, '_consciousness_orchestrator'):
            return {
                'error': 'Consciousness mode not enabled',
                'enabled': False
            }

        return self._consciousness_orchestrator.get_system_status()

    def get_consciousness_module_health(self, module_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Get health status for consciousness modules.

        Args:
            module_name: Optional specific module name. If None, returns all modules.

        Returns:
            Module health dictionary or None if not available

        Available modules:
        - autonomous_learning
        - safety_monitor
        - value_learning
        - knowledge_graph
        - tiered_memory
        - bias_mitigation
        - production_safety
        """
        if not hasattr(self, '_consciousness_orchestrator'):
            return None

        if module_name:
            health = self._consciousness_orchestrator.get_module_health(module_name)
            return asdict(health) if health else None
        else:
            all_health = self._consciousness_orchestrator.get_all_module_health()
            return {name: asdict(health) for name, health in all_health.items()}
