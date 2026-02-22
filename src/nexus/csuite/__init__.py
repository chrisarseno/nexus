"""
Nexus C-Suite Agent Framework.

This module provides the base architecture for all C-suite mega-agents:
- COO (Chief Operating Officer) - Operations, goals, health, reporting
- CIO (Chief Information Officer) - Infrastructure (Sentinel)
- CTO (Chief Technology Officer) - Product development (Forge)
- CSO (Chief Strategy Officer) - Content strategy
- CKO (Chief Knowledge Officer) - Knowledge, memory, learning
- CRO (Chief Research Officer) - Research, discovery, analysis
- CFO (Chief Financial Officer) - Budget, cost tracking

All agents follow the three-tier hierarchical pattern:
    Agent (Executive) → Manager (Coordinator) → Specialist (Executor)
"""

from nexus.csuite.base import (
    # Enums
    TaskStatus,
    TaskPriority,
    GoalStatus,
    ObjectiveStatus,

    # Data classes
    Task,
    TaskResult,
    Goal,
    Objective,
    SpecialistCapability,

    # Base classes
    Specialist,
    Manager,
    CSuiteAgent,
)

from nexus.csuite.router import LLMRouter, ModelTier, LLMResult

__all__ = [
    # Enums
    "TaskStatus",
    "TaskPriority",
    "GoalStatus",
    "ObjectiveStatus",

    # Data classes
    "Task",
    "TaskResult",
    "Goal",
    "Objective",
    "SpecialistCapability",

    # Base classes
    "Specialist",
    "Manager",
    "CSuiteAgent",

    # LLM Router
    "LLMRouter",
    "ModelTier",
    "LLMResult",
]
