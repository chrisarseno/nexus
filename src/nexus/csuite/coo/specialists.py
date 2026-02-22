"""
COO Specialists - Atomic task executors for COO functions.

These specialists handle the smallest executable units of work
within the COO's domain, using small/specialized LLMs when needed.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from nexus.csuite.base import (
    Specialist,
    Task,
    TaskResult,
    SpecialistCapability,
)

logger = logging.getLogger(__name__)


# ============================================================================
# GOAL MANAGEMENT SPECIALISTS
# ============================================================================

class GoalDecomposerSpecialist(Specialist):
    """
    Decomposes strategic goals into measurable objectives.

    Uses LLM at planning tier to understand goal intent and
    generate appropriate objectives with success criteria.
    """

    @property
    def capability(self) -> SpecialistCapability:
        return SpecialistCapability(
            name="Goal Decomposer",
            task_types=["goal.decompose", "objective.create"],
            description="Breaks strategic goals into measurable objectives",
            requires_llm=True,
            preferred_model_tier="planning",
        )

    async def _do_execute(self, task: Task) -> TaskResult:
        """Decompose a goal into objectives."""
        goal_data = task.parameters.get("goal", {})
        goal_title = goal_data.get("title", "Unknown goal")
        goal_description = goal_data.get("description", "")

        # Use LLM to decompose
        prompt = f"""Decompose this strategic goal into 2-4 measurable objectives.

Goal: {goal_title}
Description: {goal_description}

For each objective, provide JSON format:
{{
  "title": "Objective title",
  "description": "What needs to be achieved",
  "success_criteria": "How we measure success",
  "target_metric": null or number
}}

Return a JSON array of objectives."""

        system = "You are a strategic planning expert. Output valid JSON only."

        response = await self.llm_complete(prompt, system_prompt=system, tier="planning")

        if not response:
            return TaskResult(
                success=False,
                error="LLM unavailable for goal decomposition",
            )

        # Parse response
        try:
            # Try to extract JSON from response
            objectives = self._parse_json_response(response)
            return TaskResult(
                success=True,
                output=objectives,
                confidence=0.8,
            )
        except Exception as e:
            logger.warning(f"Failed to parse objectives: {e}")
            return TaskResult(
                success=True,
                output=[{
                    "title": f"Objective for: {goal_title}",
                    "description": response[:500],
                    "success_criteria": "To be defined",
                    "target_metric": None,
                }],
                confidence=0.5,
            )

    def _parse_json_response(self, response: str) -> List[Dict]:
        """Extract JSON from LLM response."""
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in response
        start = response.find('[')
        end = response.rfind(']') + 1
        if start != -1 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not parse JSON from response")


class TaskGeneratorSpecialist(Specialist):
    """
    Generates actionable tasks from objectives.

    Uses LLM at planning tier to create specific tasks
    and recommend which agent should handle each.
    """

    @property
    def capability(self) -> SpecialistCapability:
        return SpecialistCapability(
            name="Task Generator",
            task_types=["goal.generate_tasks", "task.create"],
            description="Creates actionable tasks from objectives",
            requires_llm=True,
            preferred_model_tier="planning",
        )

    async def _do_execute(self, task: Task) -> TaskResult:
        """Generate tasks for an objective."""
        objective = task.parameters.get("objective", {})
        agents = task.parameters.get("available_agents", [])

        obj_title = objective.get("title", "Unknown objective")
        obj_description = objective.get("description", "")
        obj_criteria = objective.get("success_criteria", "")

        agents_str = ", ".join(agents) if agents else "CIO, CTO, CSO, CKO, CRO, CFO"

        prompt = f"""Create 2-4 actionable tasks to achieve this objective.

Objective: {obj_title}
Description: {obj_description}
Success Criteria: {obj_criteria}

Available agents: {agents_str}

For each task, provide JSON:
{{
  "title": "Task title",
  "description": "What needs to be done",
  "assigned_to": "Agent code (CIO/CTO/etc)",
  "priority": "high/medium/low"
}}

Return a JSON array."""

        system = "You are an operations coordinator. Output valid JSON only."

        response = await self.llm_complete(prompt, system_prompt=system, tier="planning")

        if not response:
            return TaskResult(
                success=False,
                error="LLM unavailable for task generation",
            )

        try:
            tasks = self._parse_json_response(response)
            return TaskResult(
                success=True,
                output=tasks,
                confidence=0.8,
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Could not parse task JSON, using fallback: {e}")
            return TaskResult(
                success=True,
                output=[{
                    "title": f"Task for: {obj_title}",
                    "description": response[:500],
                    "assigned_to": None,
                    "priority": "medium",
                }],
                confidence=0.5,
            )

    def _parse_json_response(self, response: str) -> List[Dict]:
        """Extract JSON from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        start = response.find('[')
        end = response.rfind(']') + 1
        if start != -1 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not parse JSON")


class ProgressTrackerSpecialist(Specialist):
    """
    Tracks progress on goals and objectives.

    Calculates completion percentages and identifies blockers.
    Does not typically require LLM - uses data aggregation.
    """

    @property
    def capability(self) -> SpecialistCapability:
        return SpecialistCapability(
            name="Progress Tracker",
            task_types=["goal.track_progress", "objective.track"],
            description="Monitors goal and objective completion",
            requires_llm=False,
            preferred_model_tier="execution",
        )

    async def _do_execute(self, task: Task) -> TaskResult:
        """Calculate progress for a goal or objective."""
        target_type = task.parameters.get("type", "goal")
        target_data = task.parameters.get("data", {})
        related_tasks = task.parameters.get("tasks", [])

        if not related_tasks:
            return TaskResult(
                success=True,
                output={
                    "progress_percent": 0,
                    "status": "no_tasks",
                    "message": "No tasks associated",
                },
                confidence=1.0,
            )

        # Calculate progress from task statuses
        completed = sum(1 for t in related_tasks if t.get("status") == "completed")
        total = len(related_tasks)
        progress = (completed / total * 100) if total > 0 else 0

        # Identify blockers
        blocked = [t for t in related_tasks if t.get("status") == "blocked"]

        status = "on_track"
        if blocked:
            status = "blocked"
        elif progress >= 100:
            status = "completed"
        elif progress == 0:
            status = "not_started"

        return TaskResult(
            success=True,
            output={
                "progress_percent": round(progress, 1),
                "tasks_completed": completed,
                "tasks_total": total,
                "tasks_blocked": len(blocked),
                "status": status,
                "blockers": [b.get("title") for b in blocked],
            },
            confidence=1.0,
        )


# ============================================================================
# ROUTING SPECIALISTS
# ============================================================================

class TaskClassifierSpecialist(Specialist):
    """
    Classifies tasks to determine domain and appropriate agent.

    Can use LLM for ambiguous cases or work with keywords.
    """

    # Domain keywords for classification
    DOMAIN_KEYWORDS = {
        "infrastructure": [
            "server", "network", "firewall", "security", "backup",
            "monitoring", "alert", "incident", "vpn", "dns", "ssl",
        ],
        "technology": [
            "code", "develop", "build", "api", "database", "deploy",
            "implement", "architect", "refactor", "test", "debug",
        ],
        "content": [
            "write", "publish", "article", "ebook", "blog", "content",
            "marketing", "copy", "editorial", "seo",
        ],
        "knowledge": [
            "document", "knowledge", "memory", "learn", "catalog",
            "taxonomy", "ontology", "wiki",
        ],
        "research": [
            "research", "analyze", "investigate", "study", "discover",
            "experiment", "hypothesis", "data",
        ],
        "finance": [
            "budget", "cost", "revenue", "expense", "profit", "invoice",
            "payment", "financial", "accounting",
        ],
    }

    DOMAIN_TO_AGENT = {
        "infrastructure": "CIO",
        "technology": "CTO",
        "content": "CSO",
        "knowledge": "CKO",
        "research": "CRO",
        "finance": "CFO",
    }

    @property
    def capability(self) -> SpecialistCapability:
        return SpecialistCapability(
            name="Task Classifier",
            task_types=["task.classify", "task.route"],
            description="Determines task domain and appropriate agent",
            requires_llm=False,  # Can work without, uses LLM for ambiguous
            preferred_model_tier="execution",
        )

    async def _do_execute(self, task: Task) -> TaskResult:
        """Classify a task."""
        task_data = task.parameters.get("task", {})
        text = f"{task_data.get('title', '')} {task_data.get('description', '')}".lower()

        # Try keyword matching first
        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            # Pick domain with highest score
            best_domain = max(domain_scores, key=domain_scores.get)
            confidence = min(0.9, 0.5 + (domain_scores[best_domain] * 0.1))

            return TaskResult(
                success=True,
                output={
                    "domain": best_domain,
                    "suggested_agent": self.DOMAIN_TO_AGENT.get(best_domain),
                    "confidence": confidence,
                    "method": "keyword",
                    "scores": domain_scores,
                },
                confidence=confidence,
            )

        # If no keywords match and LLM available, use it
        if self._llm_router:
            return await self._classify_with_llm(task_data)

        # Default fallback
        return TaskResult(
            success=True,
            output={
                "domain": "general",
                "suggested_agent": None,
                "confidence": 0.3,
                "method": "default",
            },
            confidence=0.3,
        )

    async def _classify_with_llm(self, task_data: Dict) -> TaskResult:
        """Use LLM for classification when keywords fail."""
        prompt = f"""Classify this task into one domain:
- infrastructure (servers, security, networks)
- technology (code, development, deployment)
- content (writing, publishing, marketing)
- knowledge (documentation, learning, memory)
- research (analysis, investigation, discovery)
- finance (budgets, costs, revenue)

Task: {task_data.get('title', '')}
Description: {task_data.get('description', '')}

Reply with just the domain name."""

        response = await self.llm_complete(prompt, tier="execution")

        if response:
            domain = response.strip().lower()
            if domain in self.DOMAIN_TO_AGENT:
                return TaskResult(
                    success=True,
                    output={
                        "domain": domain,
                        "suggested_agent": self.DOMAIN_TO_AGENT.get(domain),
                        "confidence": 0.7,
                        "method": "llm",
                    },
                    confidence=0.7,
                )

        return TaskResult(
            success=True,
            output={
                "domain": "general",
                "suggested_agent": None,
                "confidence": 0.3,
                "method": "fallback",
            },
            confidence=0.3,
        )


class DependencyResolverSpecialist(Specialist):
    """
    Resolves task dependencies and determines execution order.
    """

    @property
    def capability(self) -> SpecialistCapability:
        return SpecialistCapability(
            name="Dependency Resolver",
            task_types=["dependency.resolve", "dependency.check"],
            description="Orders tasks by dependencies",
            requires_llm=False,
            preferred_model_tier="execution",
        )

    async def _do_execute(self, task: Task) -> TaskResult:
        """Resolve dependencies and return execution order."""
        tasks = task.parameters.get("tasks", [])

        if not tasks:
            return TaskResult(success=True, output=[], confidence=1.0)

        # Build dependency graph
        task_map = {t.get("id"): t for t in tasks}
        resolved = []
        pending = list(tasks)

        # Simple topological sort
        max_iterations = len(tasks) * 2
        iteration = 0

        while pending and iteration < max_iterations:
            iteration += 1
            made_progress = False

            for t in pending[:]:
                deps = t.get("dependencies", [])
                deps_resolved = all(
                    d in [r.get("id") for r in resolved]
                    for d in deps
                )

                if deps_resolved:
                    resolved.append(t)
                    pending.remove(t)
                    made_progress = True

            if not made_progress and pending:
                # Circular dependency detected, add remaining as-is
                resolved.extend(pending)
                break

        return TaskResult(
            success=True,
            output={
                "ordered_tasks": [t.get("id") for t in resolved],
                "has_cycles": iteration >= max_iterations,
            },
            confidence=1.0,
        )


# ============================================================================
# HEALTH MONITORING SPECIALISTS
# ============================================================================

class AgentMonitorSpecialist(Specialist):
    """
    Monitors individual agent health status.
    """

    @property
    def capability(self) -> SpecialistCapability:
        return SpecialistCapability(
            name="Agent Monitor",
            task_types=["health.check_agent", "health.check"],
            description="Checks agent health status",
            requires_llm=False,
            preferred_model_tier="execution",
        )

    async def _do_execute(self, task: Task) -> TaskResult:
        """Check agent health from stats."""
        stats = task.parameters.get("agent_stats", {})
        agent_code = task.parameters.get("agent_code", "unknown")

        issues = []
        status = "healthy"

        # Check basic health indicators
        if not stats.get("initialized", False):
            issues.append("Not initialized")
            status = "unhealthy"

        if not stats.get("running", False):
            issues.append("Not running")
            status = "unhealthy"

        # Check success rate
        completed = stats.get("tasks_completed", 0)
        failed = stats.get("tasks_failed", 0)
        total = completed + failed

        if total > 5:  # Only check if meaningful sample
            success_rate = completed / total
            if success_rate < 0.5:
                issues.append(f"Low success rate: {success_rate:.0%}")
                if status == "healthy":
                    status = "degraded"

        # Check manager count
        if stats.get("managers", 0) == 0:
            issues.append("No managers registered")
            if status == "healthy":
                status = "degraded"

        return TaskResult(
            success=True,
            output={
                "agent_code": agent_code,
                "status": status,
                "issues": issues,
                "stats_summary": {
                    "tasks_completed": completed,
                    "tasks_failed": failed,
                    "managers": stats.get("managers", 0),
                },
                "checked_at": datetime.now().isoformat(),
            },
            confidence=1.0,
        )


class AlertSpecialist(Specialist):
    """
    Identifies and prioritizes organizational alerts.
    """

    @property
    def capability(self) -> SpecialistCapability:
        return SpecialistCapability(
            name="Alert Specialist",
            task_types=["alert.create", "alert.prioritize"],
            description="Identifies and prioritizes alerts",
            requires_llm=False,
            preferred_model_tier="execution",
        )

    async def _do_execute(self, task: Task) -> TaskResult:
        """Create or prioritize alerts."""
        health_data = task.parameters.get("health_data", {})
        alerts = []

        # Check for unhealthy agents
        for agent_code, agent_health in health_data.get("agents", {}).items():
            if agent_health.get("status") == "unhealthy":
                alerts.append({
                    "id": f"alert_{agent_code}_{datetime.now().timestamp()}",
                    "severity": "high",
                    "source": agent_code,
                    "message": f"{agent_code} is unhealthy",
                    "issues": agent_health.get("issues", []),
                })
            elif agent_health.get("status") == "degraded":
                alerts.append({
                    "id": f"alert_{agent_code}_{datetime.now().timestamp()}",
                    "severity": "medium",
                    "source": agent_code,
                    "message": f"{agent_code} is degraded",
                    "issues": agent_health.get("issues", []),
                })

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        alerts.sort(key=lambda a: severity_order.get(a["severity"], 99))

        return TaskResult(
            success=True,
            output={
                "alerts": alerts,
                "total": len(alerts),
                "high_severity": len([a for a in alerts if a["severity"] in ["critical", "high"]]),
            },
            confidence=1.0,
        )


# ============================================================================
# REPORTING SPECIALISTS
# ============================================================================

class SummarySpecialist(Specialist):
    """
    Creates executive summaries from organizational data.
    """

    @property
    def capability(self) -> SpecialistCapability:
        return SpecialistCapability(
            name="Summary Specialist",
            task_types=["summary.create", "report.summary"],
            description="Creates executive summaries",
            requires_llm=True,
            preferred_model_tier="execution",
        )

    async def _do_execute(self, task: Task) -> TaskResult:
        """Generate executive summary."""
        data = task.parameters.get("data", {})

        # Extract key metrics
        goals_active = data.get("goals", {}).get("active", 0)
        issues_count = len(data.get("issues", []))
        agents_online = data.get("agents_online", 0)
        agents_total = data.get("agents_total", 0)

        if self._llm_router:
            prompt = f"""Write a 2-sentence executive summary:
- Active goals: {goals_active}
- Issues: {issues_count}
- Agents online: {agents_online}/{agents_total}

Be concise and focus on what matters most."""

            response = await self.llm_complete(prompt, tier="execution")
            if response:
                return TaskResult(
                    success=True,
                    output={"summary": response.strip()},
                    confidence=0.9,
                )

        # Template fallback
        parts = []
        if agents_total > 0:
            if agents_online == agents_total:
                parts.append(f"All {agents_total} agents operational.")
            else:
                parts.append(f"{agents_online}/{agents_total} agents online.")

        if issues_count > 0:
            parts.append(f"{issues_count} issues need attention.")
        else:
            parts.append("No issues detected.")

        if goals_active > 0:
            parts.append(f"{goals_active} goals in progress.")

        return TaskResult(
            success=True,
            output={"summary": " ".join(parts)},
            confidence=0.7,
        )


class InsightSpecialist(Specialist):
    """
    Extracts actionable insights from organizational data.
    """

    @property
    def capability(self) -> SpecialistCapability:
        return SpecialistCapability(
            name="Insight Specialist",
            task_types=["report.insights", "insight.extract"],
            description="Extracts key insights and recommendations",
            requires_llm=True,
            preferred_model_tier="planning",
        )

    async def _do_execute(self, task: Task) -> TaskResult:
        """Extract insights from data."""
        data = task.parameters.get("data", {})
        insights = []

        # Analyze agent performance
        agent_stats = data.get("agent_stats", {})
        for code, stats in agent_stats.items():
            completed = stats.get("tasks_completed", 0)
            failed = stats.get("tasks_failed", 0)

            if failed > completed and (completed + failed) > 3:
                insights.append({
                    "type": "warning",
                    "message": f"{code} has more failures than successes",
                    "recommendation": f"Review {code} task assignments and capabilities",
                })

            if completed == 0 and failed == 0:
                insights.append({
                    "type": "info",
                    "message": f"{code} has no task activity",
                    "recommendation": f"Consider if {code} is being utilized effectively",
                })

        # Analyze goals
        goals = data.get("goals", {})
        if goals.get("active", 0) > 5 and goals.get("achieved", 0) == 0:
            insights.append({
                "type": "warning",
                "message": "Many active goals but none achieved",
                "recommendation": "Focus on completing existing goals before adding new ones",
            })

        # Use LLM for deeper insights if available
        if self._llm_router and data:
            prompt = f"""Based on this organizational data, provide 1-2 actionable insights:

Agents: {list(agent_stats.keys())}
Goals: {goals}
Issues: {data.get('issues', [])}

Format each insight as: [Type] Message - Recommendation"""

            response = await self.llm_complete(prompt, tier="planning")
            if response:
                insights.append({
                    "type": "analysis",
                    "message": response.strip(),
                    "recommendation": "Review and act on insights",
                })

        return TaskResult(
            success=True,
            output={
                "insights": insights,
                "total": len(insights),
            },
            confidence=0.8,
        )
