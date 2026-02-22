"""Intelligence layer for Nexus Platform."""

from nexus.intelligence.models import (
    Message, MessageRole, Conversation, MemoryChunk, SearchHit, TopicMention
)
from nexus.intelligence.memory import NexusMemory, ChunkingConfig
from nexus.intelligence.knowledge import (
    KnowledgeGraph, Entity, EntityType, Relation, RelationType, Fact
)
from nexus.intelligence.truth import TruthVerifier, VerificationResult, ConfidenceLevel
from nexus.intelligence.goals import GoalManager, Goal, GoalStatus, Milestone, MilestoneStatus
from nexus.intelligence.tasks import (
    TaskManager, Task, TaskStatus, TaskPriority, Blocker, BlockerType, TaskNote
)
from nexus.intelligence.decisions import DecisionLog, Decision
from nexus.intelligence.corrections import CorrectionLog, Correction
from nexus.intelligence.preferences import PreferenceStore, Preference, PreferenceCategory
from nexus.intelligence.continuity import ContinuityManager, Session, FocusContext, HandoffReport


class NexusIntelligence:
    """Facade for all intelligence modules."""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self._initialized = False

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration."""
        from pathlib import Path

        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)

        return {
            "storage": {
                "chroma_path": "data/chroma",
                "sqlite_path": "data/sqlite/nexus.db"
            },
            "embedding": {
                "ollama_url": "http://localhost:11434",
                "primary_model": "nomic-embed-text",
                "fallback_model": "mxbai-embed-large"
            }
        }

    async def initialize(self):
        """Initialize all modules."""
        from nexus.storage import LocalEmbedder, VectorStore, SQLiteStore, EmbeddingConfig

        # Build embedding config from settings
        embed_cfg = self.config.get("embedding", {})
        self.embedder = LocalEmbedder(EmbeddingConfig(
            backend=embed_cfg.get("backend", "auto"),
            builtin_model=embed_cfg.get("builtin_model", "all-MiniLM-L6-v2"),
            ollama_base_url=embed_cfg.get("ollama_url", "http://localhost:11434"),
            ollama_model=embed_cfg.get("ollama_model", embed_cfg.get("primary_model", "nomic-embed-text")),
            ollama_fallback_model=embed_cfg.get("ollama_fallback_model", embed_cfg.get("fallback_model", "mxbai-embed-large")),
            auto_start_ollama=embed_cfg.get("auto_start_ollama", True),
            batch_size=embed_cfg.get("batch_size", 32),
            timeout_seconds=embed_cfg.get("timeout_seconds", 30.0),
            cache_enabled=embed_cfg.get("cache_enabled", True)
        ))
        await self.embedder.__aenter__()

        # Vector store - supports Redis, ChromaDB, or SQLite fallback
        storage_cfg = self.config.get("storage", {})
        redis_cfg = storage_cfg.get("redis", {})
        redis_config = None
        if redis_cfg.get("enabled") or redis_cfg.get("cloud_url"):
            redis_config = {
                "host": redis_cfg.get("host", "localhost"),
                "port": redis_cfg.get("port", 6379),
                "password": redis_cfg.get("password"),
                "cloud_url": redis_cfg.get("cloud_url")
            }

        self.vector_store = VectorStore(
            persist_directory=storage_cfg.get("chroma_path", "data/chroma"),
            collection_name="nexus_memory",
            backend=storage_cfg.get("vector_backend", "auto"),
            redis_config=redis_config
        )
        self.vector_store.initialize()

        self.sqlite = SQLiteStore(self.config["storage"]["sqlite_path"])
        await self.sqlite.initialize()

        # Intelligence modules
        self.memory = NexusMemory(
            self.vector_store, self.sqlite, self.embedder
        )

        self.knowledge = KnowledgeGraph(
            self.sqlite, self.vector_store, self.embedder
        )

        self.truth = TruthVerifier(
            self.knowledge, self.sqlite, self.vector_store, self.embedder
        )

        self.goals = GoalManager(self.sqlite)
        self.tasks = TaskManager(self.sqlite)
        self.decisions = DecisionLog(self.sqlite)
        self.corrections = CorrectionLog(self.sqlite)
        self.preferences = PreferenceStore(self.sqlite)

        self.continuity = ContinuityManager(
            self.sqlite, self.tasks, self.goals, self.decisions
        )

        # Initialize provider systems for GUI visibility
        await self._initialize_provider_systems()

        # Initialize Autonomous COO
        await self._initialize_coo()

        self._initialized = True

    async def _initialize_provider_systems(self):
        """Initialize ensemble, monitoring, and discovery systems."""
        import os

        # Model Registry / Discovery
        try:
            from nexus.providers.adapters.dynamic_registry import DynamicModelRegistry

            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            self._model_registry = DynamicModelRegistry(
                openrouter_api_key=openrouter_key,
                auto_discover=True,
            )
            await self._model_registry.initialize()
        except Exception as e:
            print(f"Warning: Could not initialize model registry: {e}")
            self._model_registry = None

        # Drift Monitor
        try:
            from nexus.providers.monitoring.drift_monitor import DriftMonitor

            self._drift_monitor = DriftMonitor()
        except Exception as e:
            print(f"Warning: Could not initialize drift monitor: {e}")
            self._drift_monitor = None

        # Feedback Tracker
        try:
            from nexus.providers.monitoring.feedback_tracker import FeedbackTracker
            from pathlib import Path

            feedback_path = Path(self.config.get("storage", {}).get("sqlite_path", "data/sqlite/nexus.db")).parent / "feedback.json"
            self._feedback_tracker = FeedbackTracker(persistence_path=str(feedback_path))
        except Exception as e:
            print(f"Warning: Could not initialize feedback tracker: {e}")
            self._feedback_tracker = None

        # Ensemble Orchestrator
        try:
            from nexus.providers.ensemble.core import UnifiedEnsemble

            self._ensemble = UnifiedEnsemble()
            # UnifiedEnsemble initializes in __init__, no async init needed

            # Wire up drift monitor and feedback tracker if available
            if self._drift_monitor:
                self._ensemble._drift_monitor = self._drift_monitor
            if self._feedback_tracker:
                self._ensemble._feedback_tracker = self._feedback_tracker
        except Exception as e:
            print(f"Warning: Could not initialize ensemble: {e}")
            self._ensemble = None

    async def _initialize_coo(self):
        """Initialize the COO (Chief Operating Officer).

        Tries the new C-suite COO architecture first, falling back to the
        legacy AutonomousCOO if not available.
        """
        # Try new C-suite COO first
        try:
            await self._initialize_csuite_coo()
            if self._csuite_coo:
                print("C-suite COO initialized (new architecture)")
                return
        except Exception as e:
            print(f"C-suite COO not available: {e}")
            self._csuite_coo = None

        # Fall back to legacy COO
        try:
            from nexus.coo import AutonomousCOO, COOConfig, ExecutionMode
            from pathlib import Path

            # Configure COO
            coo_config = self.config.get("coo", {})
            learning_db = Path(self.config.get("storage", {}).get("sqlite_path", "data/sqlite/nexus.db")).parent / "coo_learning.db"

            config = COOConfig(
                mode=ExecutionMode(coo_config.get("mode", "supervised")),
                observation_interval_seconds=coo_config.get("observation_interval", 30),
                priority_refresh_seconds=coo_config.get("priority_refresh", 60),
                max_concurrent_executions=coo_config.get("max_concurrent", 3),
                auto_execute_confidence=coo_config.get("auto_execute_confidence", 0.9),
                learning_enabled=coo_config.get("learning_enabled", True),
                learning_db_path=str(learning_db),
                daily_budget_usd=coo_config.get("daily_budget", 50.0),
            )

            self._coo = AutonomousCOO(intelligence=self, config=config)
            await self._coo.initialize()

            # Don't auto-start - let the GUI or caller decide
            print("Legacy COO initialized (not started)")

        except Exception as e:
            print(f"Warning: Could not initialize COO: {e}")
            import traceback
            traceback.print_exc()
            self._coo = None

    async def _initialize_csuite_coo(self):
        """Initialize the new C-suite COO architecture."""
        from nexus.csuite.coo import COOAgent
        from nexus.csuite.router import LLMRouter
        import os

        # Create LLM router for the COO
        llm_router = LLMRouter(
            ollama_base_url=self.config.get("embedding", {}).get("ollama_url", "http://localhost:11434"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

        # Create and initialize the new C-suite COO
        self._csuite_coo = COOAgent(llm_router=llm_router)
        await self._csuite_coo.initialize()

        # Register other C-suite agents if available
        await self._register_csuite_agents()

    async def _register_csuite_agents(self):
        """Register other C-suite agents with the COO."""
        if not self._csuite_coo:
            return

        # Try to import and register CIO (Sentinel)
        try:
            # Sentinel agents would be registered here when available
            # from sentinel.agents import SentinelAgent
            # self._csuite_coo.register_agent(SentinelAgent())
            pass
        except ImportError:
            pass

        # Try to import and register CTO (Forge)
        try:
            # Forge agents would be registered here when available
            # from forge.agents import ForgeAgent
            # self._csuite_coo.register_agent(ForgeAgent())
            pass
        except ImportError:
            pass

        # Additional C-suite agents would be registered similarly:
        # - CSO (Content Strategy Officer)
        # - CKO (Chief Knowledge Officer)
        # - CRO (Chief Research Officer)
        # - CFO (Chief Financial Officer)

    async def start_coo(self) -> bool:
        """Start the COO. Returns True on success."""
        print(f"[NexusIntelligence.start_coo] Called")
        print(f"  _csuite_coo: {getattr(self, '_csuite_coo', None)}")
        print(f"  _coo: {getattr(self, '_coo', None)}")

        # Try new C-suite COO first
        if hasattr(self, '_csuite_coo') and self._csuite_coo:
            print(f"  Attempting C-suite COO start...")
            try:
                await self._csuite_coo.start()
                print(f"  C-suite COO started successfully")
                return True
            except Exception as e:
                print(f"  Error starting C-suite COO: {e}")
                import traceback
                traceback.print_exc()
                return False

        # Fall back to legacy COO
        if self._coo:
            print(f"  Attempting legacy COO start...")
            try:
                result = await self._coo.start()
                print(f"  Legacy COO start returned: {result}")
                return result if result is not None else True
            except Exception as e:
                print(f"  Error starting legacy COO: {e}")
                import traceback
                traceback.print_exc()
                return False

        print("[NexusIntelligence.start_coo] COO not initialized - cannot start")
        return False

    async def stop_coo(self):
        """Stop the COO."""
        # Stop C-suite COO if available
        if hasattr(self, '_csuite_coo') and self._csuite_coo:
            try:
                await self._csuite_coo.stop()
            except Exception as e:
                print(f"Error stopping C-suite COO: {e}")

        # Stop legacy COO if available
        if self._coo:
            await self._coo.stop()

    def get_coo(self):
        """Get the COO instance.

        Returns the C-suite COO if available, otherwise the legacy COO.
        """
        if hasattr(self, '_csuite_coo') and self._csuite_coo:
            return self._csuite_coo
        return self._coo

    def get_csuite_coo(self):
        """Get the new C-suite COO specifically."""
        return getattr(self, '_csuite_coo', None)

    def get_legacy_coo(self):
        """Get the legacy COO specifically."""
        return self._coo

    def is_coo_ready(self) -> bool:
        """Check if COO is initialized and ready."""
        if hasattr(self, '_csuite_coo') and self._csuite_coo:
            return True
        return self._coo is not None

    async def shutdown(self):
        """Shutdown all modules."""
        if self._initialized:
            # Shutdown C-suite COO first
            if hasattr(self, '_csuite_coo') and self._csuite_coo:
                try:
                    await self._csuite_coo.stop()
                except Exception:
                    pass

            # Shutdown legacy COO
            if hasattr(self, '_coo') and self._coo:
                try:
                    await self._coo.stop()
                except Exception:
                    pass

            # Shutdown provider systems
            if hasattr(self, '_ensemble') and self._ensemble:
                try:
                    await self._ensemble.shutdown()
                except Exception:
                    pass

            if hasattr(self, '_model_registry') and self._model_registry:
                try:
                    await self._model_registry.cleanup()
                except Exception:
                    pass

            if hasattr(self, '_feedback_tracker') and self._feedback_tracker:
                try:
                    self._feedback_tracker.save()
                except Exception:
                    pass

            # Shutdown core systems
            await self.embedder.__aexit__(None, None, None)
            await self.sqlite.close()
            self._initialized = False


__all__ = [
    # Facade
    "NexusIntelligence",
    # Models
    "Message", "MessageRole", "Conversation", "MemoryChunk", "SearchHit", "TopicMention",
    # Memory
    "NexusMemory", "ChunkingConfig",
    # Knowledge
    "KnowledgeGraph", "Entity", "EntityType", "Relation", "RelationType", "Fact",
    # Truth
    "TruthVerifier", "VerificationResult", "ConfidenceLevel",
    # Goals
    "GoalManager", "Goal", "GoalStatus", "Milestone", "MilestoneStatus",
    # Tasks
    "TaskManager", "Task", "TaskStatus", "TaskPriority", "Blocker", "BlockerType", "TaskNote",
    # Decisions
    "DecisionLog", "Decision",
    # Corrections
    "CorrectionLog", "Correction",
    # Preferences
    "PreferenceStore", "Preference", "PreferenceCategory",
    # Continuity
    "ContinuityManager", "Session", "FocusContext", "HandoffReport",
]
