"""
Expert Base Classes - Core framework for the Panel of Experts system
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod


class TaskType(Enum):
    """Types of tasks experts can handle."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    CODING = "coding"
    REVIEW = "review"
    STRATEGY = "strategy"
    GENERAL = "general"


class Confidence(Enum):
    """Confidence levels for expert opinions."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class Task:
    """A task to be processed by experts."""
    id: str
    description: str
    task_type: TaskType = TaskType.GENERAL
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10, higher = more important
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_prompt(self) -> str:
        """Convert task to a prompt string."""
        parts = [f"Task: {self.description}"]
        if self.constraints:
            parts.append(f"Constraints: {', '.join(self.constraints)}")
        if self.context:
            parts.append(f"Context: {self.context}")
        return "\n".join(parts)


@dataclass
class ExpertOpinion:
    """An expert's opinion on a task."""
    expert_name: str
    expert_role: str
    analysis: str
    recommendation: str
    confidence: float  # 0.0 - 1.0
    reasoning: str
    concerns: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertResult:
    """Result of an expert executing a task."""
    expert_name: str
    task_id: str
    success: bool
    output: Any
    confidence: float
    execution_time: float  # seconds
    tokens_used: int = 0
    cost_usd: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertPersona:
    """Definition of an expert's personality and capabilities."""
    name: str
    role: str
    description: str
    system_prompt: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    task_types: List[TaskType] = field(default_factory=list)
    preferred_models: List[str] = field(default_factory=list)
    temperature: float = 0.7
    weight: float = 1.0  # Voting weight in consensus
    
    def matches_task(self, task: Task) -> float:
        """Return match score (0-1) for how well this expert fits the task."""
        if task.task_type in self.task_types:
            return 1.0
        if TaskType.GENERAL in self.task_types:
            return 0.5
        return 0.2


class BaseExpert(ABC):
    """Abstract base class for all experts."""
    
    def __init__(self, persona: ExpertPersona, platform=None):
        self.persona = persona
        self.platform = platform
        self._history: List[ExpertOpinion] = []
    
    @property
    def name(self) -> str:
        return self.persona.name
    
    @property
    def role(self) -> str:
        return self.persona.role
    
    @abstractmethod
    async def analyze(self, task: Task) -> ExpertOpinion:
        """Analyze a task and provide an opinion."""
        pass
    
    @abstractmethod
    async def execute(self, task: Task) -> ExpertResult:
        """Execute a task and return results."""
        pass
    
    async def critique(self, result: ExpertResult) -> ExpertOpinion:
        """Critique another expert's result."""
        critique_task = Task(
            id=f"critique_{result.task_id}",
            description=f"Review and critique this result: {result.output}",
            task_type=TaskType.REVIEW
        )
        return await self.analyze(critique_task)
    
    def get_history(self, limit: int = 10) -> List[ExpertOpinion]:
        """Get recent opinion history."""
        return self._history[-limit:]
    
    def _record_opinion(self, opinion: ExpertOpinion):
        """Record an opinion in history."""
        self._history.append(opinion)
        # Keep history bounded
        if len(self._history) > 100:
            self._history = self._history[-100:]
