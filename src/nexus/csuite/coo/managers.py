"""
COO Managers - Coordinators for COO's four core functions.

- GoalManager: Goal lifecycle and task generation
- RoutingManager: Task classification and agent assignment
- HealthManager: Organizational health monitoring
- ReportingManager: Executive reporting and insights
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nexus.csuite.base import (
    Manager,
    Task,
    TaskResult,
    TaskStatus,
    SpecialistCapability,
)
from nexus.csuite.coo.specialists import (
    GoalDecomposerSpecialist,
    TaskGeneratorSpecialist,
    ProgressTrackerSpecialist,
    TaskClassifierSpecialist,
    DependencyResolverSpecialist,
    AgentMonitorSpecialist,
    AlertSpecialist,
    SummarySpecialist,
    InsightSpecialist,
)

logger = logging.getLogger(__name__)


# ============================================================================
# GOAL MANAGER
# ============================================================================

class GoalManager(Manager):
    """
    Manages goal lifecycle: capture, decomposition, task generation, tracking.

    Specialists:
    - GoalDecomposerSpecialist: Breaks goals into objectives
    - TaskGeneratorSpecialist: Creates tasks from objectives
    - ProgressTrackerSpecialist: Monitors completion progress
    """

    def __init__(self, manager_id: Optional[str] = None, llm_router=None):
        super().__init__(manager_id=manager_id or "GoalManager", llm_router=llm_router)
        self._setup_specialists()

    def _setup_specialists(self):
        """Initialize goal management specialists."""
        self.register_specialist(GoalDecomposerSpecialist())
        self.register_specialist(TaskGeneratorSpecialist())
        self.register_specialist(ProgressTrackerSpecialist())

    @property
    def name(self) -> str:
        return "Goal Manager"

    @property
    def domain(self) -> str:
        return "goal_management"

    @property
    def handled_task_types(self) -> List[str]:
        return [
            "goal.decompose",
            "goal.generate_tasks",
            "goal.track_progress",
            "objective.create",
            "objective.update",
        ]

    async def _decompose_task(self, task: Task) -> List[Task]:
        """Decompose goal management tasks if needed."""
        if task.task_type == "goal.decompose":
            # Goal decomposition might need multiple steps
            # 1. Analyze goal
            # 2. Generate objectives
            # 3. Validate objectives
            return []  # For now, handle as single task

        return []


# ============================================================================
# ROUTING MANAGER
# ============================================================================

class RoutingManager(Manager):
    """
    Manages task routing: classification, agent matching, dependency resolution.

    Specialists:
    - TaskClassifierSpecialist: Determines task domain/type
    - DependencyResolverSpecialist: Orders tasks by dependencies
    """

    def __init__(self, manager_id: Optional[str] = None, llm_router=None):
        super().__init__(manager_id=manager_id or "RoutingManager", llm_router=llm_router)
        self._setup_specialists()

        # Agent capability registry
        self._agent_capabilities: Dict[str, List[str]] = {}

    def _setup_specialists(self):
        """Initialize routing specialists."""
        self.register_specialist(TaskClassifierSpecialist())
        self.register_specialist(DependencyResolverSpecialist())

    @property
    def name(self) -> str:
        return "Routing Manager"

    @property
    def domain(self) -> str:
        return "task_routing"

    @property
    def handled_task_types(self) -> List[str]:
        return [
            "task.classify",
            "task.route",
            "task.assign",
            "dependency.resolve",
            "dependency.check",
        ]

    def register_agent_capabilities(self, agent_code: str, capabilities: List[str]):
        """Register what an agent can do for routing decisions."""
        self._agent_capabilities[agent_code] = capabilities
        logger.info(f"Registered capabilities for {agent_code}: {capabilities}")

    async def classify_task(self, task: Task) -> Dict[str, Any]:
        """
        Classify a task to determine its domain and characteristics.

        Returns classification with:
        - domain: Primary domain (infrastructure, technology, content, etc.)
        - complexity: low/medium/high
        - suggested_agent: Recommended C-suite agent code
        """
        # Use TaskClassifierSpecialist
        classifier = self.find_specialist(Task(task_type="task.classify"))
        if classifier:
            classify_task = Task(
                task_type="task.classify",
                parameters={"task": task.to_dict()},
            )
            result = await classifier.execute(classify_task)
            if result.success:
                return result.output

        # Fallback: simple keyword classification
        return self._simple_classify(task)

    def _simple_classify(self, task: Task) -> Dict[str, Any]:
        """Simple keyword-based classification."""
        text = f"{task.title} {task.description}".lower()

        domain_keywords = {
            "infrastructure": ["server", "network", "firewall", "security", "backup"],
            "technology": ["code", "develop", "build", "api", "database", "deploy"],
            "content": ["write", "publish", "article", "ebook", "marketing"],
            "knowledge": ["document", "learn", "memory", "research"],
            "research": ["analyze", "investigate", "study", "discover"],
            "finance": ["budget", "cost", "revenue", "expense", "profit"],
        }

        domain_to_agent = {
            "infrastructure": "CIO",
            "technology": "CTO",
            "content": "CSO",
            "knowledge": "CKO",
            "research": "CRO",
            "finance": "CFO",
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                return {
                    "domain": domain,
                    "complexity": "medium",
                    "suggested_agent": domain_to_agent.get(domain),
                }

        return {
            "domain": "general",
            "complexity": "medium",
            "suggested_agent": None,
        }


# ============================================================================
# HEALTH MANAGER
# ============================================================================

class HealthManager(Manager):
    """
    Monitors organizational health across all C-suite agents.

    Specialists:
    - AgentMonitorSpecialist: Checks individual agent status
    - AlertSpecialist: Identifies and prioritizes issues
    """

    def __init__(self, manager_id: Optional[str] = None, llm_router=None):
        super().__init__(manager_id=manager_id or "HealthManager", llm_router=llm_router)
        self._setup_specialists()

        # Health tracking
        self._health_history: List[Dict[str, Any]] = []
        self._active_alerts: List[Dict[str, Any]] = []

    def _setup_specialists(self):
        """Initialize health monitoring specialists."""
        self.register_specialist(AgentMonitorSpecialist())
        self.register_specialist(AlertSpecialist())

    @property
    def name(self) -> str:
        return "Health Manager"

    @property
    def domain(self) -> str:
        return "health_monitoring"

    @property
    def handled_task_types(self) -> List[str]:
        return [
            "health.check",
            "health.check_agent",
            "health.report",
            "alert.create",
            "alert.resolve",
        ]

    async def check_agent_health(self, agent_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an agent's health from its stats.
        """
        issues = []
        status = "healthy"

        # Check initialization
        if not agent_stats.get("initialized", False):
            issues.append("Agent not initialized")
            status = "unhealthy"

        # Check running state
        if not agent_stats.get("running", False):
            issues.append("Agent not running")
            status = "unhealthy"

        # Check success rate
        completed = agent_stats.get("tasks_completed", 0)
        failed = agent_stats.get("tasks_failed", 0)
        total = completed + failed
        if total > 0:
            success_rate = completed / total
            if success_rate < 0.5:
                issues.append(f"Low success rate: {success_rate:.1%}")
                status = "degraded" if status == "healthy" else status

        return {
            "status": status,
            "issues": issues,
            "stats": agent_stats,
            "checked_at": datetime.now().isoformat(),
        }

    def add_alert(self, alert: Dict[str, Any]):
        """Add an alert to the active alerts list."""
        alert["created_at"] = datetime.now().isoformat()
        self._active_alerts.append(alert)
        logger.warning(f"Alert added: {alert.get('message', 'Unknown alert')}")

    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        self._active_alerts = [a for a in self._active_alerts if a.get("id") != alert_id]

    @property
    def active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return list(self._active_alerts)


# ============================================================================
# REPORTING MANAGER
# ============================================================================

class ReportingManager(Manager):
    """
    Generates executive reports and insights for CEO consumption.

    Specialists:
    - SummarySpecialist: Creates executive summaries
    - InsightSpecialist: Extracts key takeaways
    """

    def __init__(self, manager_id: Optional[str] = None, llm_router=None):
        super().__init__(manager_id=manager_id or "ReportingManager", llm_router=llm_router)
        self._setup_specialists()

    def _setup_specialists(self):
        """Initialize reporting specialists."""
        self.register_specialist(SummarySpecialist())
        self.register_specialist(InsightSpecialist())

    @property
    def name(self) -> str:
        return "Reporting Manager"

    @property
    def domain(self) -> str:
        return "executive_reporting"

    @property
    def handled_task_types(self) -> List[str]:
        return [
            "report.status",
            "report.executive",
            "report.metrics",
            "report.insights",
            "summary.create",
        ]

    async def generate_summary(self, data: Dict[str, Any]) -> str:
        """
        Generate an executive summary from organizational data.
        """
        if not self._llm_router:
            # Simple template-based summary
            return self._template_summary(data)

        # Use LLM for natural language summary
        prompt = f"""Create a brief executive summary (3-4 sentences) from this data:

Goals: {data.get('goals', {})}
Health Issues: {data.get('issues', [])}
Agents Online: {data.get('agents_online', 0)}/{data.get('agents_total', 0)}

Focus on: What's working, what needs attention, recommended actions."""

        response = await self.llm_complete(
            prompt=prompt,
            tier="execution",  # Simple summarization
        )

        return response or self._template_summary(data)

    def _template_summary(self, data: Dict[str, Any]) -> str:
        """Generate a template-based summary when LLM is unavailable."""
        goals = data.get('goals', {})
        issues = data.get('issues', [])
        agents_online = data.get('agents_online', 0)
        agents_total = data.get('agents_total', 0)

        parts = []

        if agents_total > 0:
            if agents_online == agents_total:
                parts.append(f"All {agents_total} agents operational.")
            else:
                parts.append(f"{agents_online}/{agents_total} agents online.")

        active_goals = goals.get('active', 0)
        if active_goals > 0:
            parts.append(f"{active_goals} active goals in progress.")

        if issues:
            parts.append(f"{len(issues)} issues require attention.")
        else:
            parts.append("No critical issues detected.")

        return " ".join(parts)

    async def extract_insights(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract key insights from organizational data.
        """
        insights = []

        # Check agent utilization
        for agent_code, stats in data.get('agent_stats', {}).items():
            completed = stats.get('tasks_completed', 0)
            failed = stats.get('tasks_failed', 0)

            if failed > completed:
                insights.append(f"{agent_code} has high failure rate - investigate")

            if completed == 0 and failed == 0:
                insights.append(f"{agent_code} has no task activity - may be underutilized")

        # Check goal progress
        goals = data.get('goals', {})
        if goals.get('active', 0) > 0 and goals.get('achieved', 0) == 0:
            insights.append("No goals achieved yet - review progress")

        return insights
