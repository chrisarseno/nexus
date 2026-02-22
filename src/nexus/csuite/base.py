"""
C-Suite Agent Base Classes.

This module defines the three-tier architecture for all C-suite agents:

    CSuiteAgent (Executive)
        └── Manager (Coordinator)
                └── Specialist (Task Executor)

Architecture Overview:
- **Agents** are executives that own a domain (COO, CIO, CTO, etc.)
- **Managers** coordinate groups of specialists for sub-domains
- **Specialists** execute individual atomic tasks with precision

This architecture enables:
- Layered LLM usage (complex→simple as you go down)
- Parallel execution across specialists
- Cross-validation for accuracy
- Specialization for precision
- Easy scaling by adding specialists
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from nexus.csuite.router import LLMRouter

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class TaskStatus(str, Enum):
    """Status of a task in the hierarchy."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GoalStatus(str, Enum):
    """Status of a strategic goal."""
    DRAFT = "draft"
    ACTIVE = "active"
    ACHIEVED = "achieved"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"


class ObjectiveStatus(str, Enum):
    """Status of an objective under a goal."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TaskResult:
    """Result from executing a task."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    confidence: float = 1.0
    duration_ms: float = 0.0
    specialist_id: Optional[str] = None
    tokens_used: int = 0
    model_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "specialist_id": self.specialist_id,
            "tokens_used": self.tokens_used,
            "model_used": self.model_used,
            "metadata": self.metadata,
        }


@dataclass
class Task:
    """A task to be executed by the hierarchy."""
    id: str = field(default_factory=lambda: str(uuid4()))
    task_type: str = ""
    title: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[TaskResult] = None
    parent_task_id: Optional[str] = None
    objective_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None  # Agent ID (CIO, CTO, etc.)
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "title": self.title,
            "description": self.description,
            "parameters": self.parameters,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result.to_dict() if self.result else None,
            "parent_task_id": self.parent_task_id,
            "objective_id": self.objective_id,
            "subtask_ids": self.subtask_ids,
            "assigned_to": self.assigned_to,
            "dependencies": self.dependencies,
        }


@dataclass
class Objective:
    """A measurable objective under a goal."""
    id: str = field(default_factory=lambda: str(uuid4()))
    goal_id: str = ""
    title: str = ""
    description: str = ""
    success_criteria: str = ""
    target_metric: Optional[float] = None
    current_metric: Optional[float] = None
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    tasks: List[str] = field(default_factory=list)  # Task IDs
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def progress(self) -> float:
        """Calculate progress as percentage."""
        if self.target_metric and self.current_metric:
            return min(100.0, (self.current_metric / self.target_metric) * 100)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "title": self.title,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "target_metric": self.target_metric,
            "current_metric": self.current_metric,
            "progress": self.progress,
            "status": self.status.value,
            "tasks": self.tasks,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class Goal:
    """A strategic goal from the CEO."""
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    description: str = ""
    owner: str = "CEO"
    status: GoalStatus = GoalStatus.DRAFT
    target_date: Optional[datetime] = None
    objectives: List[str] = field(default_factory=list)  # Objective IDs
    created_at: datetime = field(default_factory=datetime.now)
    achieved_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "owner": self.owner,
            "status": self.status.value,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "objectives": self.objectives,
            "created_at": self.created_at.isoformat(),
            "achieved_at": self.achieved_at.isoformat() if self.achieved_at else None,
            "tags": self.tags,
        }


@dataclass
class SpecialistCapability:
    """Describes what a specialist can do."""
    name: str
    task_types: List[str]
    description: str = ""
    confidence: float = 0.9
    max_concurrent: int = 5
    requires_llm: bool = False
    preferred_model_tier: str = "execution"  # strategic, planning, execution


# ============================================================================
# SPECIALIST - ATOMIC TASK EXECUTOR
# ============================================================================

class Specialist(ABC):
    """
    Base class for Specialists - atomic task executors.

    Specialists are the lowest tier in the hierarchy. They:
    - Execute single, focused tasks
    - Have deep expertise in one narrow area
    - Return results with confidence scores
    - Use small/specialized LLMs when needed
    - Can be parallelized for throughput
    """

    def __init__(
        self,
        specialist_id: Optional[str] = None,
        llm_router: Optional["LLMRouter"] = None
    ):
        self.id = specialist_id or f"{self.__class__.__name__}_{uuid4().hex[:8]}"
        self._llm_router = llm_router
        self._running = False
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._current_tasks: Dict[str, Task] = {}

    def set_llm_router(self, router: "LLMRouter") -> None:
        """Set or update the LLM router for this specialist."""
        self._llm_router = router

    async def llm_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tier: str = "execution",
        context: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Request LLM completion using the appropriate model tier.

        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            tier: Model tier (execution, planning, strategic)
            context: Additional context for routing

        Returns:
            LLM response text or None if no router
        """
        if not self._llm_router:
            logger.debug(f"Specialist {self.id} has no LLM router")
            return None

        try:
            result = await self._llm_router.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                tier=tier,
                context=context or {}
            )
            return result.text
        except Exception as e:
            logger.warning(f"LLM completion failed for {self.id}: {e}")
            return None

    @property
    @abstractmethod
    def capability(self) -> SpecialistCapability:
        """Define what this specialist can do."""
        pass

    @property
    def name(self) -> str:
        """Human-readable name."""
        return self.capability.name

    @property
    def is_available(self) -> bool:
        """Check if specialist can accept more work."""
        return len(self._current_tasks) < self.capability.max_concurrent

    @property
    def stats(self) -> Dict[str, Any]:
        """Get specialist statistics."""
        total = self._tasks_completed + self._tasks_failed
        return {
            "id": self.id,
            "name": self.name,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "current_tasks": len(self._current_tasks),
            "available": self.is_available,
            "success_rate": self._tasks_completed / total if total > 0 else 1.0,
        }

    def can_handle(self, task: Task) -> bool:
        """Check if this specialist can handle a task."""
        return task.task_type in self.capability.task_types

    async def execute(self, task: Task) -> TaskResult:
        """Execute a task."""
        if not self.can_handle(task):
            return TaskResult(
                success=False,
                error=f"Specialist {self.name} cannot handle task type: {task.task_type}",
                specialist_id=self.id,
            )

        start_time = datetime.now()
        self._current_tasks[task.id] = task
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = start_time

        try:
            result = await self._do_execute(task)
            result.specialist_id = self.id
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            if result.success:
                self._tasks_completed += 1
                task.status = TaskStatus.COMPLETED
            else:
                self._tasks_failed += 1
                task.status = TaskStatus.FAILED

            task.completed_at = datetime.now()
            task.result = result
            return result

        except Exception as e:
            self._tasks_failed += 1
            logger.error(f"Specialist {self.id} failed on task {task.id}: {e}")

            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()

            result = TaskResult(
                success=False,
                error=str(e),
                specialist_id=self.id,
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
            task.result = result
            return result

        finally:
            self._current_tasks.pop(task.id, None)

    @abstractmethod
    async def _do_execute(self, task: Task) -> TaskResult:
        """Implement the actual task execution."""
        pass


# ============================================================================
# MANAGER - SPECIALIST COORDINATOR
# ============================================================================

class Manager(ABC):
    """
    Base class for Managers - specialist coordinators.

    Managers are the middle tier. They:
    - Receive tasks from Agents
    - Break down tasks into subtasks
    - Assign subtasks to appropriate Specialists
    - Aggregate and validate results
    - Use small-medium LLMs for planning
    """

    def __init__(
        self,
        manager_id: Optional[str] = None,
        llm_router: Optional["LLMRouter"] = None
    ):
        self.id = manager_id or f"{self.__class__.__name__}_{uuid4().hex[:8]}"
        self._llm_router = llm_router
        self._specialists: Dict[str, Specialist] = {}
        self._running = False
        self._tasks_completed = 0
        self._tasks_failed = 0

    def set_llm_router(self, router: "LLMRouter") -> None:
        """Set LLM router for this manager and all specialists."""
        self._llm_router = router
        for specialist in self._specialists.values():
            specialist.set_llm_router(router)

    async def llm_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tier: str = "planning",
        context: Dict[str, Any] = None
    ) -> Optional[str]:
        """Request LLM completion at planning tier."""
        if not self._llm_router:
            return None

        try:
            result = await self._llm_router.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                tier=tier,
                context=context or {}
            )
            return result.text
        except Exception as e:
            logger.warning(f"LLM completion failed for {self.id}: {e}")
            return None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable manager name."""
        pass

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain this manager handles."""
        pass

    @property
    @abstractmethod
    def handled_task_types(self) -> List[str]:
        """Task types this manager can handle."""
        pass

    @property
    def specialists(self) -> List[Specialist]:
        """Get all registered specialists."""
        return list(self._specialists.values())

    @property
    def available_specialists(self) -> List[Specialist]:
        """Get specialists that can accept work."""
        return [s for s in self._specialists.values() if s.is_available]

    @property
    def stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "specialists": len(self._specialists),
            "available_specialists": len(self.available_specialists),
            "specialist_stats": {s.id: s.stats for s in self._specialists.values()},
        }

    def register_specialist(self, specialist: Specialist) -> None:
        """Register a specialist with this manager."""
        self._specialists[specialist.id] = specialist
        if self._llm_router:
            specialist.set_llm_router(self._llm_router)
        logger.info(f"Manager {self.id} registered specialist: {specialist.name}")

    def unregister_specialist(self, specialist_id: str) -> None:
        """Unregister a specialist."""
        if specialist_id in self._specialists:
            del self._specialists[specialist_id]

    def can_handle(self, task: Task) -> bool:
        """Check if this manager can handle a task."""
        return task.task_type in self.handled_task_types

    def find_specialist(self, task: Task) -> Optional[Specialist]:
        """Find the best available specialist for a task."""
        candidates = [
            s for s in self._specialists.values()
            if s.can_handle(task) and s.is_available
        ]

        if not candidates:
            return None

        # Sort by success rate and current load
        candidates.sort(
            key=lambda s: (s.stats["success_rate"], -len(s._current_tasks)),
            reverse=True
        )
        return candidates[0]

    async def execute(self, task: Task) -> TaskResult:
        """Execute a task by coordinating specialists."""
        if not self.can_handle(task):
            return TaskResult(
                success=False,
                error=f"Manager {self.name} cannot handle task type: {task.task_type}",
            )

        start_time = datetime.now()
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = start_time

        try:
            # Break down into subtasks if needed
            subtasks = await self._decompose_task(task)

            if not subtasks:
                result = await self._execute_single(task)
            else:
                result = await self._execute_subtasks(task, subtasks)

            # Validate result
            result = await self._validate_result(task, result)
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            if result.success:
                self._tasks_completed += 1
                task.status = TaskStatus.COMPLETED
            else:
                self._tasks_failed += 1
                task.status = TaskStatus.FAILED

            task.completed_at = datetime.now()
            task.result = result
            return result

        except Exception as e:
            self._tasks_failed += 1
            logger.error(f"Manager {self.id} failed on task {task.id}: {e}")

            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()

            return TaskResult(
                success=False,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    async def _execute_single(self, task: Task) -> TaskResult:
        """Execute a single task via a specialist."""
        specialist = self.find_specialist(task)

        if not specialist:
            return TaskResult(
                success=False,
                error=f"No available specialist for task type: {task.task_type}",
            )

        task.assigned_to = specialist.id
        return await specialist.execute(task)

    async def _execute_subtasks(self, parent: Task, subtasks: List[Task]) -> TaskResult:
        """Execute multiple subtasks in parallel."""
        for subtask in subtasks:
            subtask.parent_task_id = parent.id
            parent.subtask_ids.append(subtask.id)

        parallel_tasks = []
        failed_results = []

        for subtask in subtasks:
            specialist = self.find_specialist(subtask)
            if specialist:
                subtask.assigned_to = specialist.id
                parallel_tasks.append(specialist.execute(subtask))
            else:
                failed_results.append(TaskResult(
                    success=False,
                    error=f"No specialist for: {subtask.task_type}",
                ))

        if parallel_tasks:
            results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    failed_results.append(TaskResult(success=False, error=str(result)))
                else:
                    failed_results.append(result)

        return await self._aggregate_results(parent, failed_results)

    async def _decompose_task(self, task: Task) -> List[Task]:
        """Break down a task into subtasks. Override in subclasses."""
        return []

    async def _aggregate_results(self, parent: Task, results: List[TaskResult]) -> TaskResult:
        """Aggregate results from multiple subtasks."""
        if not results:
            return TaskResult(success=False, error="No results to aggregate")

        all_success = all(r.success for r in results)
        outputs = [r.output for r in results if r.output is not None]
        errors = [r.error for r in results if r.error]
        avg_confidence = sum(r.confidence for r in results) / len(results)
        total_tokens = sum(r.tokens_used for r in results)

        return TaskResult(
            success=all_success,
            output=outputs if len(outputs) > 1 else (outputs[0] if outputs else None),
            error="; ".join(errors) if errors else None,
            confidence=avg_confidence,
            tokens_used=total_tokens,
            metadata={"subtask_count": len(results)},
        )

    async def _validate_result(self, task: Task, result: TaskResult) -> TaskResult:
        """Validate and potentially improve a result. Override in subclasses."""
        return result


# ============================================================================
# C-SUITE AGENT - EXECUTIVE / DOMAIN OWNER
# ============================================================================

class CSuiteAgent(ABC):
    """
    Base class for C-Suite Agents - domain executives.

    Agents are the top tier. They:
    - Own an entire domain (operations, technology, content, etc.)
    - Receive high-level goals and tasks
    - Delegate to appropriate Managers
    - Coordinate cross-manager workflows
    - Report outcomes to COO/CEO
    - Use medium LLMs for strategic understanding
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        llm_router: Optional["LLMRouter"] = None
    ):
        self.id = agent_id or f"{self.__class__.__name__}_{uuid4().hex[:8]}"
        self._llm_router = llm_router
        self._managers: Dict[str, Manager] = {}
        self._running = False
        self._initialized = False
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._started_at: Optional[datetime] = None

    def set_llm_router(self, router: "LLMRouter") -> None:
        """Set LLM router for this agent and propagate to managers."""
        self._llm_router = router
        for manager in self._managers.values():
            manager.set_llm_router(router)
        logger.info(f"Agent {self.name} LLM router set, propagated to managers")

    async def llm_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tier: str = "strategic",
        context: Dict[str, Any] = None
    ) -> Optional[str]:
        """Request LLM completion at strategic tier."""
        if not self._llm_router:
            return None

        try:
            result = await self._llm_router.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                tier=tier,
                context=context or {}
            )
            return result.text
        except Exception as e:
            logger.warning(f"LLM completion failed for {self.id}: {e}")
            return None

    @property
    def llm_router(self) -> Optional["LLMRouter"]:
        """Get the LLM router for this agent."""
        return self._llm_router

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name (e.g., 'Chief Operating Officer')."""
        pass

    @property
    @abstractmethod
    def code(self) -> str:
        """Short code for this agent (e.g., 'COO', 'CTO')."""
        pass

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain this agent owns (e.g., 'operations', 'technology')."""
        pass

    @property
    @abstractmethod
    def handled_task_types(self) -> List[str]:
        """Top-level task types this agent handles."""
        pass

    @property
    def managers(self) -> List[Manager]:
        """Get all registered managers."""
        return list(self._managers.values())

    @property
    def stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        uptime = None
        if self._started_at:
            uptime = (datetime.now() - self._started_at).total_seconds()

        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "domain": self.domain,
            "running": self._running,
            "initialized": self._initialized,
            "uptime_seconds": uptime,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "managers": len(self._managers),
            "manager_stats": {m.id: m.stats for m in self._managers.values()},
        }

    def register_manager(self, manager: Manager) -> None:
        """Register a manager with this agent."""
        self._managers[manager.id] = manager
        if self._llm_router:
            manager.set_llm_router(self._llm_router)
        logger.info(f"Agent {self.code} registered manager: {manager.name}")

    def unregister_manager(self, manager_id: str) -> None:
        """Unregister a manager."""
        if manager_id in self._managers:
            del self._managers[manager_id]

    async def initialize(self) -> bool:
        """Initialize the agent and its managers/specialists."""
        if self._initialized:
            return True

        try:
            await self._setup_managers()
            self._initialized = True
            self._running = True
            self._started_at = datetime.now()
            logger.info(f"Agent {self.name} initialized with {len(self._managers)} managers")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize agent {self.name}: {e}")
            return False

    @abstractmethod
    async def _setup_managers(self) -> None:
        """Set up managers and specialists for this agent."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        self._running = False
        self._initialized = False
        logger.info(f"Agent {self.name} shutdown")

    def can_handle(self, task: Task) -> bool:
        """Check if this agent can handle a task."""
        if task.task_type in self.handled_task_types:
            return True

        for manager in self._managers.values():
            if manager.can_handle(task):
                return True

        return False

    def find_manager(self, task: Task) -> Optional[Manager]:
        """Find the best manager for a task."""
        for manager in self._managers.values():
            if manager.can_handle(task):
                return manager
        return None

    async def execute(self, task: Task) -> TaskResult:
        """Execute a task by coordinating managers."""
        if not self._initialized:
            init_success = await self.initialize()
            if not init_success:
                return TaskResult(
                    success=False,
                    error=f"Agent {self.name} failed to initialize",
                )

        start_time = datetime.now()
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = start_time

        try:
            # Plan execution
            execution_plan = await self._plan_execution(task)

            # Execute via managers
            if execution_plan.get("parallel"):
                result = await self._execute_parallel(task, execution_plan)
            else:
                result = await self._execute_sequential(task, execution_plan)

            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            if result.success:
                self._tasks_completed += 1
                task.status = TaskStatus.COMPLETED
            else:
                self._tasks_failed += 1
                task.status = TaskStatus.FAILED

            task.completed_at = datetime.now()
            task.result = result
            return result

        except Exception as e:
            self._tasks_failed += 1
            logger.error(f"Agent {self.id} failed on task {task.id}: {e}")

            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()

            return TaskResult(
                success=False,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    async def _plan_execution(self, task: Task) -> Dict[str, Any]:
        """Plan how to execute a task. Override for custom planning."""
        manager = self.find_manager(task)
        if manager:
            return {
                "parallel": False,
                "steps": [{"manager_id": manager.id, "task": task}],
            }
        return {"parallel": False, "steps": []}

    async def _execute_sequential(self, task: Task, plan: Dict[str, Any]) -> TaskResult:
        """Execute steps sequentially."""
        results = []

        for step in plan.get("steps", []):
            manager_id = step.get("manager_id")
            step_task = step.get("task", task)

            manager = self._managers.get(manager_id)
            if not manager:
                results.append(TaskResult(
                    success=False,
                    error=f"Manager not found: {manager_id}",
                ))
                continue

            result = await manager.execute(step_task)
            results.append(result)

            if not result.success and not plan.get("continue_on_failure"):
                break

        return self._aggregate_results(results)

    async def _execute_parallel(self, task: Task, plan: Dict[str, Any]) -> TaskResult:
        """Execute steps in parallel."""
        tasks = []

        for step in plan.get("steps", []):
            manager_id = step.get("manager_id")
            step_task = step.get("task", task)

            manager = self._managers.get(manager_id)
            if manager:
                tasks.append(manager.execute(step_task))

        if not tasks:
            return TaskResult(success=False, error="No managers to execute")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(TaskResult(success=False, error=str(result)))
            else:
                processed_results.append(result)

        return self._aggregate_results(processed_results)

    def _aggregate_results(self, results: List[TaskResult]) -> TaskResult:
        """Aggregate results from multiple managers."""
        if not results:
            return TaskResult(success=False, error="No results")

        all_success = all(r.success for r in results)
        outputs = [r.output for r in results if r.output is not None]
        errors = [r.error for r in results if r.error]
        avg_confidence = sum(r.confidence for r in results) / len(results)
        total_tokens = sum(r.tokens_used for r in results)

        return TaskResult(
            success=all_success,
            output=outputs if len(outputs) > 1 else (outputs[0] if outputs else None),
            error="; ".join(errors) if errors else None,
            confidence=avg_confidence,
            tokens_used=total_tokens,
            metadata={"manager_count": len(results)},
        )
