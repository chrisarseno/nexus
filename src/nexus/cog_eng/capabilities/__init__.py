"""AGI Capabilities Module"""

from .autonomous_research_agent import AutonomousResearchAgent, ResearchGoal
from .self_improving_codegen import SelfImprovingCodeGenerator, CodeGenerationRequest, CodeQuality

__all__ = [
    "AutonomousResearchAgent",
    "ResearchGoal",
    "SelfImprovingCodeGenerator",
    "CodeGenerationRequest",
    "CodeQuality"
]
