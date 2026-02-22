"""
COO Agent - Chief Operating Officer.

The neural center of the organization, coordinating all C-suite agents
and translating CEO goals into actionable work across the enterprise.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nexus.csuite.base import (
    CSuiteAgent,
    Task,
    TaskResult,
    TaskStatus,
    Goal,
    Objective,
)
from nexus.csuite.coo.managers import (
    GoalManager,
    RoutingManager,
    HealthManager,
    ReportingManager,
)

logger = logging.getLogger(__name__)


class COOAgent(CSuiteAgent):
    """
    Chief Operating Officer Agent.

    Responsibilities:
    1. Goal Management - Capture, decompose, and track strategic goals
    2. Task Routing - Direct work to appropriate C-suite agents
    3. Health Monitoring - Ensure all agents are operational
    4. Executive Reporting - Synthesize insights for CEO

    The COO operates as a coordinator, not an executor. All domain-specific
    work is delegated to specialized C-suite agents.
    """

    def __init__(self, agent_id: Optional[str] = None, llm_router=None):
        super().__init__(agent_id=agent_id or "coo", llm_router=llm_router)

        # Registry of other C-suite agents
        self._csuite_agents: Dict[str, CSuiteAgent] = {}

        # Data stores (will be replaced with proper persistence)
        self._goals: Dict[str, Goal] = {}
        self._objectives: Dict[str, Objective] = {}
        self._tasks: Dict[str, Task] = {}

        # Operation state
        self._observation_cycle = 0
        self._last_health_check: Optional[datetime] = None

    @property
    def name(self) -> str:
        return "Chief Operating Officer"

    @property
    def code(self) -> str:
        return "COO"

    @property
    def domain(self) -> str:
        return "operations"

    @property
    def handled_task_types(self) -> List[str]:
        return [
            # Goal management
            "goal.create",
            "goal.decompose",
            "goal.track",
            "goal.update",

            # Task routing
            "task.route",
            "task.create",
            "task.assign",

            # Health monitoring
            "health.check",
            "health.report",

            # Reporting
            "report.status",
            "report.executive",
            "report.metrics",
        ]

    async def _setup_managers(self) -> None:
        """Set up the four core COO managers."""
        # Goal Manager - handles goal lifecycle and task generation
        goal_manager = GoalManager()
        self.register_manager(goal_manager)

        # Routing Manager - determines where tasks should go
        routing_manager = RoutingManager()
        self.register_manager(routing_manager)

        # Health Manager - monitors organizational health
        health_manager = HealthManager()
        self.register_manager(health_manager)

        # Reporting Manager - creates executive summaries
        reporting_manager = ReportingManager()
        self.register_manager(reporting_manager)

        logger.info("COO managers initialized")

    # =========================================================================
    # C-SUITE AGENT REGISTRY
    # =========================================================================

    def register_agent(self, agent: CSuiteAgent) -> None:
        """Register a C-suite agent that COO can delegate to."""
        self._csuite_agents[agent.code] = agent
        logger.info(f"COO registered agent: {agent.code} ({agent.name})")

    def unregister_agent(self, code: str) -> None:
        """Unregister a C-suite agent."""
        if code in self._csuite_agents:
            del self._csuite_agents[code]

    def get_agent(self, code: str) -> Optional[CSuiteAgent]:
        """Get a registered C-suite agent by code."""
        return self._csuite_agents.get(code)

    @property
    def registered_agents(self) -> Dict[str, CSuiteAgent]:
        """Get all registered C-suite agents."""
        return dict(self._csuite_agents)

    # =========================================================================
    # GOAL MANAGEMENT
    # =========================================================================

    async def create_goal(
        self,
        title: str,
        description: str,
        target_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
    ) -> Goal:
        """
        Create a new strategic goal.

        Args:
            title: Goal title
            description: Detailed description
            target_date: Optional target completion date
            tags: Optional categorization tags

        Returns:
            Created Goal object
        """
        goal = Goal(
            title=title,
            description=description,
            target_date=target_date,
            tags=tags or [],
        )
        self._goals[goal.id] = goal
        logger.info(f"Created goal: {goal.id} - {title}")
        return goal

    async def decompose_goal(self, goal_id: str) -> List[Objective]:
        """
        Decompose a goal into measurable objectives.

        Uses LLM at strategic tier to understand the goal and
        generate appropriate objectives.
        """
        goal = self._goals.get(goal_id)
        if not goal:
            raise ValueError(f"Goal not found: {goal_id}")

        # Use LLM to decompose the goal
        prompt = f"""Analyze this strategic goal and break it down into 2-5 measurable objectives.

Goal: {goal.title}
Description: {goal.description}

For each objective, provide:
1. Title (concise)
2. Description (what needs to be achieved)
3. Success criteria (how we know it's done)
4. Suggested target metric (if quantifiable)

Format as a structured list."""

        system_prompt = """You are a strategic operations expert helping decompose
high-level goals into actionable objectives. Focus on measurable outcomes."""

        response = await self.llm_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            tier="strategic"
        )

        if not response:
            logger.warning(f"LLM decomposition failed for goal {goal_id}")
            return []

        # Parse response and create objectives
        # For now, create a placeholder - real implementation would parse LLM output
        objectives = await self._parse_objectives(goal_id, response)

        # Link objectives to goal
        goal.objectives = [obj.id for obj in objectives]
        goal.status = "active"

        return objectives

    async def _parse_objectives(self, goal_id: str, llm_response: str) -> List[Objective]:
        """Parse LLM response into Objective objects."""
        # Simple parsing - production would use structured output
        objectives = []

        # Create at least one objective from the response
        obj = Objective(
            goal_id=goal_id,
            title="Objective from decomposition",
            description=llm_response[:500],
            success_criteria="To be refined",
        )
        objectives.append(obj)
        self._objectives[obj.id] = obj

        return objectives

    async def generate_tasks_for_objective(self, objective_id: str) -> List[Task]:
        """
        Generate actionable tasks for an objective.

        Uses LLM at planning tier to create tasks and determine
        which C-suite agent should handle each.
        """
        objective = self._objectives.get(objective_id)
        if not objective:
            raise ValueError(f"Objective not found: {objective_id}")

        # Get available agents for routing context
        available_agents = [
            f"- {code}: {agent.name} ({agent.domain})"
            for code, agent in self._csuite_agents.items()
        ]
        agents_str = "\n".join(available_agents) if available_agents else "No agents registered"

        prompt = f"""Create actionable tasks to achieve this objective.

Objective: {objective.title}
Description: {objective.description}
Success Criteria: {objective.success_criteria}

Available C-suite agents to assign tasks to:
{agents_str}

For each task, provide:
1. Title
2. Description
3. Recommended agent (code)
4. Priority (critical/high/medium/low)
5. Dependencies (other task numbers, if any)

Create 2-5 concrete tasks."""

        system_prompt = """You are an operations coordinator creating actionable tasks.
Each task should be specific enough to be executed by a single agent."""

        response = await self.llm_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            tier="planning"
        )

        if not response:
            logger.warning(f"LLM task generation failed for objective {objective_id}")
            return []

        # Parse and create tasks
        tasks = await self._parse_tasks(objective_id, response)

        # Link tasks to objective
        objective.tasks = [t.id for t in tasks]

        return tasks

    async def _parse_tasks(self, objective_id: str, llm_response: str) -> List[Task]:
        """Parse LLM response into Task objects."""
        tasks = []

        # Simple parsing - production would use structured output
        task = Task(
            objective_id=objective_id,
            task_type="task.generated",
            title="Task from objective",
            description=llm_response[:500],
        )
        tasks.append(task)
        self._tasks[task.id] = task

        return tasks

    # =========================================================================
    # TASK ROUTING
    # =========================================================================

    async def route_task(self, task: Task) -> Optional[str]:
        """
        Determine which C-suite agent should handle a task.

        Returns the agent code or None if no suitable agent found.
        """
        if not self._csuite_agents:
            logger.warning("No C-suite agents registered for task routing")
            return None

        # Check if task already has an assignment
        if task.assigned_to:
            return task.assigned_to

        # Use routing manager to determine best agent
        routing_manager = self._managers.get("RoutingManager")
        if routing_manager:
            # TODO: Implement proper routing via manager
            pass

        # Simple keyword-based routing as fallback
        description_lower = task.description.lower()
        title_lower = task.title.lower()
        combined = f"{title_lower} {description_lower}"

        routing_rules = {
            "CIO": ["infrastructure", "security", "network", "server", "firewall"],
            "CTO": ["code", "develop", "build", "implement", "architect", "deploy"],
            "CSO": ["content", "write", "publish", "ebook", "article", "blog"],
            "CKO": ["knowledge", "memory", "learn", "document", "research synthesis"],
            "CRO": ["research", "analyze", "discover", "investigate", "study"],
            "CFO": ["budget", "cost", "finance", "revenue", "expense"],
        }

        for agent_code, keywords in routing_rules.items():
            if agent_code in self._csuite_agents:
                if any(kw in combined for kw in keywords):
                    task.assigned_to = agent_code
                    return agent_code

        # Default to first available agent
        if self._csuite_agents:
            first_agent = list(self._csuite_agents.keys())[0]
            task.assigned_to = first_agent
            return first_agent

        return None

    async def dispatch_task(self, task: Task) -> TaskResult:
        """
        Dispatch a task to its assigned agent for execution.
        """
        agent_code = task.assigned_to or await self.route_task(task)

        if not agent_code:
            return TaskResult(
                success=False,
                error="No agent available to handle task",
            )

        agent = self._csuite_agents.get(agent_code)
        if not agent:
            return TaskResult(
                success=False,
                error=f"Agent not found: {agent_code}",
            )

        logger.info(f"Dispatching task {task.id} to {agent_code}")
        return await agent.execute(task)

    # =========================================================================
    # HEALTH MONITORING
    # =========================================================================

    async def check_organization_health(self) -> Dict[str, Any]:
        """
        Check the health of all registered C-suite agents.
        """
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "coo_status": "healthy" if self._running else "stopped",
            "agents": {},
            "issues": [],
        }

        for code, agent in self._csuite_agents.items():
            agent_health = {
                "name": agent.name,
                "initialized": agent._initialized,
                "running": agent._running,
                "stats": agent.stats,
            }

            if not agent._initialized:
                health_report["issues"].append(f"{code} not initialized")
            elif not agent._running:
                health_report["issues"].append(f"{code} not running")

            health_report["agents"][code] = agent_health

        self._last_health_check = datetime.now()
        return health_report

    # =========================================================================
    # EXECUTIVE REPORTING
    # =========================================================================

    async def generate_executive_report(self) -> Dict[str, Any]:
        """
        Generate an executive summary for the CEO.
        """
        # Gather data
        health = await self.check_organization_health()

        # Calculate goal progress
        goal_summary = []
        for goal in self._goals.values():
            obj_count = len(goal.objectives)
            completed_objs = sum(
                1 for oid in goal.objectives
                if self._objectives.get(oid, Objective()).status == "completed"
            )
            goal_summary.append({
                "title": goal.title,
                "status": goal.status.value if hasattr(goal.status, 'value') else goal.status,
                "objectives": f"{completed_objs}/{obj_count}",
            })

        # Build report
        report = {
            "generated_at": datetime.now().isoformat(),
            "organization_health": "healthy" if not health["issues"] else "issues_detected",
            "issues_count": len(health["issues"]),
            "issues": health["issues"][:5],  # Top 5 issues
            "active_goals": len([g for g in self._goals.values() if g.status == "active"]),
            "goal_summary": goal_summary[:10],  # Top 10 goals
            "agents_online": len([a for a in self._csuite_agents.values() if a._running]),
            "agents_total": len(self._csuite_agents),
            "coo_stats": self.stats,
        }

        # Use LLM to generate natural language summary if available
        if self._llm_router and goal_summary:
            summary_prompt = f"""Generate a brief executive summary (2-3 sentences) based on:
- {len(goal_summary)} active goals
- {len(health['issues'])} issues detected
- {report['agents_online']}/{report['agents_total']} agents online

Focus on what the CEO needs to know."""

            summary = await self.llm_complete(
                prompt=summary_prompt,
                tier="execution"  # Simple summarization task
            )
            if summary:
                report["executive_summary"] = summary

        return report

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive COO status."""
        return {
            "agent": self.stats,
            "goals": {
                "total": len(self._goals),
                "active": len([g for g in self._goals.values() if g.status == "active"]),
                "achieved": len([g for g in self._goals.values() if g.status == "achieved"]),
            },
            "objectives": {
                "total": len(self._objectives),
                "in_progress": len([o for o in self._objectives.values() if o.status == "in_progress"]),
                "completed": len([o for o in self._objectives.values() if o.status == "completed"]),
            },
            "tasks": {
                "total": len(self._tasks),
                "pending": len([t for t in self._tasks.values() if t.status == TaskStatus.PENDING]),
                "in_progress": len([t for t in self._tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
                "completed": len([t for t in self._tasks.values() if t.status == TaskStatus.COMPLETED]),
            },
            "registered_agents": list(self._csuite_agents.keys()),
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
        }
