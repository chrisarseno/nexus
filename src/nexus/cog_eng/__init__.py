"""
Cognitive Engine - AGI Core for Nexus Platform

Imported from standalone cog-eng project.
Provides consciousness, autonomous research, and code generation.
"""

__version__ = "0.1.0"
__author__ = "Cog-Eng Contributors"
__license__ = "MIT"

# Core imports
from .consciousness.consciousness_core import ConsciousnessCore
from .capabilities.autonomous_research_agent import AutonomousResearchAgent
from .capabilities.self_improving_codegen import SelfImprovingCodeGenerator

__all__ = [
    "ConsciousnessCore",
    "AutonomousResearchAgent",
    "SelfImprovingCodeGenerator",
    "__version__"
]
