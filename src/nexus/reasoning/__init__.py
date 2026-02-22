"""
Nexus Reasoning System

Advanced reasoning engines for self-improvement and meta-learning.

This system provides:
- Meta-reasoning for self-improvement
- Chain-of-thought reasoning with explanations
- Pattern-based reasoning and inference
- Dynamic adaptive learning
- Reasoning analytics and performance tracking
"""

from .meta_reasoner import MetaReasoner
from .chain_of_thought import ChainOfThoughtEngine
from .pattern_reasoner import PatternReasoner
from .dynamic_learner import DynamicLearner
from .analytics import AnalyticsModule
from .echo import EchoModule

# Aliases for backward compatibility
ChainOfThought = ChainOfThoughtEngine
ReasoningAnalytics = AnalyticsModule
Echo = EchoModule

__all__ = [
    "MetaReasoner",
    "ChainOfThought",
    "ChainOfThoughtEngine",
    "PatternReasoner",
    "DynamicLearner",
    "ReasoningAnalytics",
    "AnalyticsModule",
    "Echo",
    "EchoModule",
]

__version__ = "1.0.0"
