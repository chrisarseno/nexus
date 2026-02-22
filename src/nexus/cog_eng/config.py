"""
Cog-Eng Configuration Management

Centralized configuration using environment variables with validation.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv, dotenv_values
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Load .env file - try multiple locations
env_paths = [
    Path(__file__).parent.parent.parent.parent / '.env',  # FPT root (fantastic-palm-tree-main/.env)
    Path(__file__).parent.parent.parent / '.env',  # Module root (cog-eng/.env)
    Path(__file__).parent.parent / '.env',  # src/.env
    Path(__file__).parent / '.env',  # src/cog_eng/.env
]

env_values: Dict[str, str] = {}
env_loaded = False
for env_path in env_paths:
    resolved_path = env_path.resolve()
    if resolved_path.exists():
        # Load into dictionary - this is the reliable method
        env_values = dotenv_values(resolved_path)
        # Also try to load into os.environ
        load_dotenv(resolved_path, override=True)
        logger.info(f"Loaded environment from {resolved_path}")
        env_loaded = True
        break

if not env_loaded:
    logger.warning(f"No .env file found in standard locations, using system environment variables")


# Helper function to get environment variable with fallback to env_values dict
def _getenv(key: str, default: str = None) -> Optional[str]:
    """Get environment variable, checking both os.environ and env_values dict."""
    # First try os.environ
    value = os.getenv(key)
    if value:
        return value
    # Fallback to env_values dict
    return env_values.get(key, default)


@dataclass
class LLMConfig:
    """LLM API configuration."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    together_api_key: Optional[str] = None
    replicate_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None

    default_provider: str = "openai"
    default_model: str = "gpt-4"
    research_model: str = "gpt-4"
    codegen_model: str = "gpt-4"
    simple_model: str = "gpt-3.5-turbo"


@dataclass
class BudgetConfig:
    """Budget and cost management configuration."""
    hourly_limit: float = 10.0
    daily_limit: float = 100.0
    monthly_limit: float = 1000.0
    enable_optimization: bool = True
    target_cost_per_request: float = 0.10


@dataclass
class ComponentConfig:
    """Component enable/disable configuration."""
    enable_consciousness: bool = True
    enable_learning: bool = True
    enable_agents: bool = True
    enable_routing: bool = True


@dataclass
class ConsciousnessConfig:
    """Consciousness system configuration."""
    safety_threshold: float = 0.8
    modules: List[str] = None

    def __post_init__(self):
        if self.modules is None:
            self.modules = [
                'temporal_consciousness',
                'global_workspace',
                'social_cognition',
                'creative_intelligence',
                'value_learning',
                'virtue_learning',
                'safety_monitor'
            ]


@dataclass
class LearningConfig:
    """Learning system configuration."""
    verification_sources: int = 3
    confidence_threshold: float = 0.7


@dataclass
class AgentConfig:
    """Agent system configuration."""
    max_parallel: int = 5
    enable_referee: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///./data/cogeng.db"
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class RedisConfig:
    """Redis configuration."""
    url: str = "redis://localhost:6379/0"
    enabled: bool = False
    ttl_seconds: int = 3600


@dataclass
class GRPCConfig:
    """gRPC bridge configuration."""
    host: str = "localhost"
    port: int = 50051
    max_workers: int = 10


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 3737
    workers: int = 4
    enable_docs: bool = True
    enable_cors: bool = True
    api_key_enabled: bool = False
    api_keys: List[str] = None

    def __post_init__(self):
        if self.api_keys is None:
            self.api_keys = []


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    log_level: str = "INFO"
    log_format: str = "json"
    enable_prometheus: bool = True
    prometheus_port: int = 9090


@dataclass
class FantasticPalmTreeConfig:
    """Fantastic Palm Tree integration configuration."""
    path: Optional[str] = None
    integration_mode: str = "enhance"  # standalone, pass_through, enhance
    enable_prompt_optimizer: bool = True
    enable_ensemble: bool = True
    enable_memory: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    signing_key: str = "changeme-secret"
    enable_input_validation: bool = True
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 60


@dataclass
class AdvancedConfig:
    """Advanced settings configuration."""
    routing_strategy: str = "cost_optimized"
    quality_threshold: float = 0.8
    research_max_iterations: int = 10
    research_default_depth: str = "moderate"
    codegen_max_iterations: int = 5
    codegen_target_quality: float = 0.85
    codegen_default_language: str = "python"
    enable_contradiction_detection: bool = True
    contradiction_tolerance: float = 0.3
    min_agreement_ratio: float = 0.6


class Config:
    """
    Main configuration class that loads all settings from environment variables.

    Usage:
        from config import config

        # Access LLM config
        api_key = config.llm.openai_api_key

        # Access budget config
        limit = config.budget.hourly_limit
    """

    def __init__(self):
        """Load configuration from environment variables."""
        self.environment = _getenv('ENVIRONMENT', 'development')

        # LLM Configuration
        self.llm = LLMConfig(
            openai_api_key=_getenv('OPENAI_API_KEY'),
            anthropic_api_key=_getenv('ANTHROPIC_API_KEY'),
            google_api_key=_getenv('GOOGLE_API_KEY'),
            cohere_api_key=_getenv('COHERE_API_KEY'),
            together_api_key=_getenv('TOGETHER_API_KEY'),
            replicate_api_key=_getenv('REPLICATE_API_KEY'),
            huggingface_api_key=_getenv('HUGGINGFACE_API_KEY'),
            default_provider=_getenv('DEFAULT_LLM_PROVIDER', 'openai'),
            default_model=_getenv('DEFAULT_LLM_MODEL', 'gpt-4'),
            research_model=_getenv('RESEARCH_MODEL', 'gpt-4'),
            codegen_model=_getenv('CODEGEN_MODEL', 'gpt-4'),
            simple_model=_getenv('SIMPLE_MODEL', 'gpt-3.5-turbo')
        )

        # Budget Configuration
        self.budget = BudgetConfig(
            hourly_limit=float(_getenv('HOURLY_BUDGET_LIMIT', '10.0')),
            daily_limit=float(_getenv('DAILY_BUDGET_LIMIT', '100.0')),
            monthly_limit=float(_getenv('MONTHLY_BUDGET_LIMIT', '1000.0')),
            enable_optimization=_getenv('ENABLE_COST_OPTIMIZATION', 'true').lower() == 'true',
            target_cost_per_request=float(_getenv('TARGET_COST_PER_REQUEST', '0.10'))
        )

        # Component Configuration
        self.components = ComponentConfig(
            enable_consciousness=_getenv('ENABLE_CONSCIOUSNESS', 'true').lower() == 'true',
            enable_learning=_getenv('ENABLE_LEARNING', 'true').lower() == 'true',
            enable_agents=_getenv('ENABLE_AGENTS', 'true').lower() == 'true',
            enable_routing=_getenv('ENABLE_ROUTING', 'true').lower() == 'true'
        )

        # Consciousness Configuration
        modules_str = _getenv('CONSCIOUSNESS_MODULES', '')
        modules = None
        if modules_str:
            modules = [m.strip() for m in modules_str.split(',')]

        self.consciousness = ConsciousnessConfig(
            safety_threshold=float(_getenv('CONSCIOUSNESS_SAFETY_THRESHOLD', '0.8')),
            modules=modules
        )

        # Learning Configuration
        self.learning = LearningConfig(
            verification_sources=int(_getenv('LEARNING_VERIFICATION_SOURCES', '3')),
            confidence_threshold=float(_getenv('LEARNING_CONFIDENCE_THRESHOLD', '0.7'))
        )

        # Agent Configuration
        self.agents = AgentConfig(
            max_parallel=int(_getenv('MAX_PARALLEL_AGENTS', '5')),
            enable_referee=_getenv('ENABLE_AGENT_REFEREE', 'true').lower() == 'true'
        )

        # Database Configuration
        self.database = DatabaseConfig(
            url=_getenv('DATABASE_URL', 'sqlite:///./data/cogeng.db')
        )

        # Redis Configuration
        self.redis = RedisConfig(
            url=_getenv('REDIS_URL', 'redis://localhost:6379/0'),
            enabled=_getenv('ENABLE_REDIS_CACHE', 'false').lower() == 'true',
            ttl_seconds=int(_getenv('CACHE_TTL_SECONDS', '3600'))
        )

        # gRPC Configuration
        self.grpc = GRPCConfig(
            host=_getenv('GRPC_SERVER_HOST', 'localhost'),
            port=int(_getenv('GRPC_SERVER_PORT', '50051')),
            max_workers=int(_getenv('GRPC_MAX_WORKERS', '10'))
        )

        # API Configuration
        api_keys_str = _getenv('API_KEYS', '')
        api_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]

        self.api = APIConfig(
            host=_getenv('API_HOST', '0.0.0.0'),
            port=int(_getenv('API_PORT', '3737')),
            workers=int(_getenv('API_WORKERS', '4')),
            enable_docs=_getenv('ENABLE_API_DOCS', 'true').lower() == 'true',
            enable_cors=_getenv('ENABLE_CORS', 'true').lower() == 'true',
            api_key_enabled=_getenv('API_KEY_ENABLED', 'false').lower() == 'true',
            api_keys=api_keys
        )

        # Monitoring Configuration
        self.monitoring = MonitoringConfig(
            log_level=_getenv('LOG_LEVEL', 'INFO'),
            log_format=_getenv('LOG_FORMAT', 'json'),
            enable_prometheus=_getenv('ENABLE_PROMETHEUS', 'true').lower() == 'true',
            prometheus_port=int(_getenv('PROMETHEUS_PORT', '9090'))
        )

        # Fantastic Palm Tree Configuration
        self.fpt = FantasticPalmTreeConfig(
            path=_getenv('FANTASTIC_PALM_TREE_PATH'),
            integration_mode=_getenv('FPT_INTEGRATION_MODE', 'enhance'),
            enable_prompt_optimizer=_getenv('FPT_ENABLE_PROMPT_OPTIMIZER', 'true').lower() == 'true',
            enable_ensemble=_getenv('FPT_ENABLE_ENSEMBLE', 'true').lower() == 'true',
            enable_memory=_getenv('FPT_ENABLE_MEMORY', 'true').lower() == 'true'
        )

        # Security Configuration
        self.security = SecurityConfig(
            signing_key=_getenv('AGENT_SIGNING_KEY', 'changeme-secret'),
            enable_input_validation=_getenv('ENABLE_INPUT_VALIDATION', 'true').lower() == 'true',
            enable_rate_limiting=_getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true',
            rate_limit_per_minute=int(_getenv('RATE_LIMIT_PER_MINUTE', '60'))
        )

        # Advanced Configuration
        self.advanced = AdvancedConfig(
            routing_strategy=_getenv('ROUTING_STRATEGY', 'cost_optimized'),
            quality_threshold=float(_getenv('QUALITY_THRESHOLD', '0.8')),
            research_max_iterations=int(_getenv('RESEARCH_MAX_ITERATIONS', '10')),
            research_default_depth=_getenv('RESEARCH_DEFAULT_DEPTH', 'moderate'),
            codegen_max_iterations=int(_getenv('CODEGEN_MAX_ITERATIONS', '5')),
            codegen_target_quality=float(_getenv('CODEGEN_TARGET_QUALITY', '0.85')),
            codegen_default_language=_getenv('CODEGEN_DEFAULT_LANGUAGE', 'python'),
            enable_contradiction_detection=_getenv('ENABLE_CONTRADICTION_DETECTION', 'true').lower() == 'true',
            contradiction_tolerance=float(_getenv('CONTRADICTION_TOLERANCE', '0.3')),
            min_agreement_ratio=float(_getenv('MIN_AGREEMENT_RATIO', '0.6'))
        )

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate configuration and warn about missing required values."""
        warnings = []

        # Check for LLM API keys
        if not any([
            self.llm.openai_api_key,
            self.llm.anthropic_api_key,
            self.llm.google_api_key
        ]):
            warnings.append("No LLM API keys configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")

        # Check security
        if self.security.signing_key == 'changeme-secret':
            warnings.append("Using default AGENT_SIGNING_KEY. Change this in production!")

        # Log warnings
        for warning in warnings:
            logger.warning(f"Config Warning: {warning}")

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == 'production'

    def has_llm_key(self, provider: str = None) -> bool:
        """Check if an LLM API key is configured."""
        if provider is None:
            provider = self.llm.default_provider

        key_map = {
            'openai': self.llm.openai_api_key,
            'anthropic': self.llm.anthropic_api_key,
            'google': self.llm.google_api_key,
            'cohere': self.llm.cohere_api_key,
            'together': self.llm.together_api_key,
            'replicate': self.llm.replicate_api_key,
            'huggingface': self.llm.huggingface_api_key
        }

        return bool(key_map.get(provider.lower()))


# Global configuration instance
config = Config()

# Convenience exports
__all__ = [
    'config',
    'Config',
    'LLMConfig',
    'BudgetConfig',
    'ComponentConfig',
    'ConsciousnessConfig',
    'LearningConfig',
    'AgentConfig',
    'DatabaseConfig',
    'RedisConfig',
    'GRPCConfig',
    'APIConfig',
    'MonitoringConfig',
    'FantasticPalmTreeConfig',
    'SecurityConfig',
    'AdvancedConfig'
]
