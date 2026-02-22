"""
Cog-Eng Client API
Main interface for interacting with the Cognitive Engine
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio

# Import core systems - using relative imports for package structure
try:
    from ..consciousness.consciousness_core import ConsciousnessCore
    from ..agents.base_agent import BaseAgent
    from ..agents.orchestrator_agent import OrchestratorAgent
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from consciousness.consciousness_core import ConsciousnessCore
    from agents.base_agent import BaseAgent
    from agents.orchestrator_agent import OrchestratorAgent

logger = logging.getLogger(__name__)

@dataclass
class CogEngConfig:
    """Configuration for Cognitive Engine."""
    enable_consciousness: bool = True
    enable_learning: bool = True
    enable_agents: bool = True
    enable_routing: bool = True

    # Consciousness settings
    consciousness_modules: List[str] = None
    safety_threshold: float = 0.8

    # Learning settings
    verification_sources: int = 3
    confidence_threshold: float = 0.7

    # Agent settings
    max_parallel_agents: int = 5
    enable_referee: bool = True

    # Routing settings
    routing_strategy: str = "cost_optimized"  # cost_optimized, quality_focused, balanced
    quality_threshold: float = 0.8
    budget_limit: Optional[float] = None

    def __post_init__(self):
        if self.consciousness_modules is None:
            self.consciousness_modules = [
                'temporal_consciousness',
                'global_workspace',
                'social_cognition',
                'creative_intelligence',
                'value_learning',
                'virtue_learning',
                'safety_monitor'
            ]

@dataclass
class CogEngResponse:
    """Response from Cognitive Engine processing."""
    response: str
    confidence: float
    knowledge_nodes_added: int
    agents_involved: List[str]
    processing_time: float
    consciousness_state: Optional[Dict[str, Any]] = None
    learning_insights: Optional[Dict[str, Any]] = None
    safety_evaluation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class CognitiveEngine:
    """
    Main Cognitive Engine class for AGI-level processing.

    This is the primary interface for integrating Cog-Eng into any system.
    It orchestrates consciousness, learning, multi-agent systems, and adaptive routing.

    Example usage:
        engine = CognitiveEngine(
            enable_consciousness=True,
            enable_learning=True,
            enable_agents=True
        )

        result = await engine.process(
            task="Analyze market trends and create investment strategy",
            context={"risk_tolerance": "moderate", "timeline": "5 years"}
        )
    """

    def __init__(
        self,
        config: Optional[CogEngConfig] = None,
        **kwargs
    ):
        """
        Initialize the Cognitive Engine.

        Args:
            config: CogEngConfig object or None to use defaults
            **kwargs: Override config parameters
        """
        # Set up configuration
        if config is None:
            config = CogEngConfig(**kwargs)
        else:
            # Override config with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config

        # Initialize components
        self.consciousness_core = None
        self.learning_loop = None
        self.agent_orchestrator = None
        self.adaptive_router = None

        # State
        self.initialized = False
        self.processing_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0
        }

        logger.info("Cognitive Engine created with config: %s", asdict(config))

    async def initialize(self) -> bool:
        """
        Initialize all enabled components.

        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing Cognitive Engine...")

            # Initialize consciousness system
            if self.config.enable_consciousness:
                logger.info("Initializing Consciousness Core...")
                self.consciousness_core = ConsciousnessCore()
                if not self.consciousness_core.initialize():
                    logger.error("Failed to initialize Consciousness Core")
                    if self.config.safety_threshold > 0.5:
                        return False

            # Initialize learning loop
            if self.config.enable_learning:
                logger.info("Initializing Learning Loop...")
                # Learning loop initialization will be added when we create the bridge
                logger.info("Learning Loop initialized (TypeScript bridge pending)")

            # Initialize agent orchestrator
            if self.config.enable_agents:
                logger.info("Initializing Agent Orchestrator...")
                import tempfile
                import os
                outdir = os.path.join(tempfile.gettempdir(), 'cog-eng-agents')
                self.agent_orchestrator = OrchestratorAgent(outdir=outdir)
                logger.info("Agent Orchestrator initialized")

            # Initialize adaptive router
            if self.config.enable_routing:
                logger.info("Initializing Adaptive Router...")
                # Adaptive router initialization will be added when we create the bridge
                logger.info("Adaptive Router initialized (TypeScript bridge pending)")

            self.initialized = True
            logger.info("✅ Cognitive Engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Cognitive Engine: {e}")
            return False

    async def process(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        priority: str = "normal",  # low, normal, high, critical
        require_verification: bool = True
    ) -> CogEngResponse:
        """
        Process a task through the Cognitive Engine.

        Args:
            task: The task description
            context: Additional context for the task
            priority: Task priority level
            require_verification: Whether to verify results

        Returns:
            CogEngResponse with results and metadata
        """
        if not self.initialized:
            await self.initialize()

        start_time = datetime.now()

        try:
            logger.info(f"Processing task: {task[:100]}...")

            # Prepare experience data for consciousness
            experience_data = {
                'type': 'task_processing',
                'description': task,
                'context': context or {},
                'priority': priority,
                'timestamp': start_time.isoformat()
            }

            # Process through consciousness
            consciousness_result = None
            if self.consciousness_core:
                consciousness_result = self.consciousness_core.process_experience(experience_data)

            # Process through agents
            agents_involved = []
            agent_results = {}

            if self.agent_orchestrator:
                # Create a task ticket for the orchestrator
                ticket = {
                    'id': f'task_{int(start_time.timestamp())}',
                    'name': task[:50],
                    'user': 'cog-eng-api',
                    'priority': priority,
                    'description': task,
                    'context': context or {}
                }

                # Run orchestrator
                try:
                    orchestrator_result = self.agent_orchestrator.run(ticket)
                    agents_involved.append('orchestrator')
                    agent_results['orchestrator'] = orchestrator_result
                except Exception as e:
                    logger.error(f"Agent orchestrator error: {e}")

            # Get consciousness state
            consciousness_state = None
            if self.consciousness_core:
                consciousness_state = self.consciousness_core.get_system_state()

            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Build response
            response = CogEngResponse(
                response=f"Task processed: {task}",
                confidence=0.85,  # Placeholder - will be calculated from actual results
                knowledge_nodes_added=0,  # Will be populated from learning loop
                agents_involved=agents_involved,
                processing_time=processing_time,
                consciousness_state=consciousness_state,
                learning_insights=None,  # Will be populated from learning loop
                safety_evaluation=consciousness_result.get('safety_evaluation') if consciousness_result else None,
                metadata={
                    'task': task,
                    'context': context,
                    'priority': priority,
                    'agent_results': agent_results
                },
                timestamp=end_time
            )

            # Update stats
            self.processing_stats['total_tasks'] += 1
            self.processing_stats['successful_tasks'] += 1
            self.processing_stats['total_processing_time'] += processing_time

            logger.info(f"✅ Task completed in {processing_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error processing task: {e}")
            self.processing_stats['failed_tasks'] += 1

            # Return error response
            return CogEngResponse(
                response=f"Error: {str(e)}",
                confidence=0.0,
                knowledge_nodes_added=0,
                agents_involved=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={'error': str(e)}
            )

    def get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state."""
        state = {
            'initialized': self.initialized,
            'config': asdict(self.config),
            'processing_stats': self.processing_stats,
            'timestamp': datetime.now().isoformat()
        }

        # Add consciousness state
        if self.consciousness_core:
            state['consciousness'] = self.consciousness_core.get_system_state()

        return state

    def add_consciousness_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for consciousness updates."""
        if self.consciousness_core:
            self.consciousness_core.add_update_callback(callback)

    async def shutdown(self):
        """Shutdown the Cognitive Engine cleanly."""
        logger.info("Shutting down Cognitive Engine...")

        if self.consciousness_core:
            self.consciousness_core.cleanup()

        logger.info("Cognitive Engine shutdown complete")
