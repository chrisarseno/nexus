"""Session continuity and context handoff."""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field

from nexus.storage import SQLiteStore
from nexus.intelligence.tasks import TaskManager, Task, TaskStatus
from nexus.intelligence.goals import GoalManager, Goal
from nexus.intelligence.decisions import DecisionLog, Decision


@dataclass
class Session:
    id: str
    project_path: Optional[str] = None
    focus_task_id: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None
    handoff_notes: Optional[str] = None
    insights: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


@dataclass
class FocusContext:
    active_tasks: List[Task]
    blocked_tasks: List[Task]
    active_goals: List[Goal]
    recent_decisions: List[Decision]
    blockers_summary: List[str]
    suggested_action: str
    last_session_notes: Optional[str] = None


@dataclass
class HandoffReport:
    session_id: str
    summary: str
    in_progress_tasks: List[Dict]
    blockers: List[str]
    recent_decisions: List[Dict]
    suggested_next_steps: List[str]
    open_questions: List[str]


class ContinuityManager:
    """Manage session continuity and context handoff."""

    def __init__(self, sqlite: SQLiteStore, tasks: TaskManager,
                 goals: GoalManager, decisions: DecisionLog):
        self.sqlite = sqlite
        self.tasks = tasks
        self.goals = goals
        self.decisions = decisions
        self._current_session: Optional[Session] = None

    async def start_session(self, project_path: Optional[str] = None,
                           focus_task_id: Optional[str] = None) -> Session:
        """Start a new work session."""
        session = Session(
            id=Session.generate_id(),
            project_path=project_path,
            focus_task_id=focus_task_id
        )

        await self.sqlite.insert("sessions", {
            "id": session.id,
            "project_path": project_path,
            "focus_task_id": focus_task_id,
            "started_at": session.started_at.isoformat()
        })

        self._current_session = session
        return session

    async def end_session(self, summary: Optional[str] = None,
                         handoff_notes: Optional[str] = None,
                         insights: Optional[List[str]] = None,
                         open_questions: Optional[List[str]] = None) -> Session:
        """End current session with handoff info."""
        if not self._current_session:
            # Create a placeholder session
            self._current_session = Session(id=Session.generate_id())

        self._current_session.ended_at = lambda: datetime.now(timezone.utc)()
        self._current_session.summary = summary
        self._current_session.handoff_notes = handoff_notes
        self._current_session.insights = insights or []
        self._current_session.open_questions = open_questions or []

        await self.sqlite.update("sessions", self._current_session.id, {
            "ended_at": self._current_session.ended_at.isoformat(),
            "summary": summary,
            "handoff_notes": handoff_notes,
            "insights": ",".join(insights) if insights else None,
            "open_questions": ",".join(open_questions) if open_questions else None
        })

        session = self._current_session
        self._current_session = None
        return session

    async def get_current_focus(self, project_path: Optional[str] = None) -> FocusContext:
        """Get current focus context for continuity."""
        # Get active tasks
        in_progress = await self.tasks.list_tasks(
            project_path=project_path,
            status=TaskStatus.IN_PROGRESS
        )

        blocked = await self.tasks.list_tasks(
            project_path=project_path,
            status=TaskStatus.BLOCKED
        )

        # Get active goals
        goals = await self.goals.list_goals(project_path=project_path)

        # Get recent decisions
        decisions = await self.decisions.list_decisions(
            project_path=project_path, limit=5
        )

        # Get last session notes
        last_session = await self._get_last_session(project_path)

        # Compile blockers
        blockers_summary = []
        for task in blocked:
            for blocker in task.blockers:
                if not blocker.resolved:
                    blockers_summary.append(f"{task.title}: {blocker.description}")

        # Suggest next action
        suggested = self._suggest_next_action(in_progress, blocked, goals)

        return FocusContext(
            active_tasks=in_progress,
            blocked_tasks=blocked,
            active_goals=goals,
            recent_decisions=decisions,
            blockers_summary=blockers_summary,
            suggested_action=suggested,
            last_session_notes=last_session.handoff_notes if last_session else None
        )

    async def generate_handoff(self, project_path: Optional[str] = None) -> HandoffReport:
        """Generate handoff report for session transition."""
        focus = await self.get_current_focus(project_path)

        in_progress_info = [
            {"id": t.id, "title": t.title, "priority": t.priority.value}
            for t in focus.active_tasks
        ]

        recent_decisions_info = [
            {"question": d.question, "decision": d.decision}
            for d in focus.recent_decisions[:3]
        ]

        next_steps = []
        if focus.blockers_summary:
            next_steps.append(f"Resolve blockers: {len(focus.blockers_summary)} pending")
        if focus.active_tasks:
            next_steps.append(f"Continue: {focus.active_tasks[0].title}")
        if focus.active_goals:
            next_steps.append(f"Progress toward: {focus.active_goals[0].title}")

        session_id = self._current_session.id if self._current_session else "none"

        return HandoffReport(
            session_id=session_id,
            summary=f"Active work on {len(focus.active_tasks)} tasks",
            in_progress_tasks=in_progress_info,
            blockers=focus.blockers_summary,
            recent_decisions=recent_decisions_info,
            suggested_next_steps=next_steps,
            open_questions=[]
        )

    async def get_continuation_prompt(self, project_path: Optional[str] = None,
                                      token_budget: int = 1500) -> str:
        """Generate context prompt for new session."""
        focus = await self.get_current_focus(project_path)

        lines = ["## Current Context\n"]

        if focus.last_session_notes:
            lines.append(f"**Last Session:** {focus.last_session_notes}\n")

        if focus.active_tasks:
            lines.append("**In Progress:**")
            for task in focus.active_tasks[:3]:
                lines.append(f"- {task.title} [{task.priority.value}]")
            lines.append("")

        if focus.blockers_summary:
            lines.append("**Blockers:**")
            for blocker in focus.blockers_summary[:3]:
                lines.append(f"- {blocker}")
            lines.append("")

        if focus.recent_decisions:
            lines.append("**Recent Decisions:**")
            for dec in focus.recent_decisions[:2]:
                lines.append(f"- Q: {dec.question[:50]}... -> {dec.decision[:50]}...")
            lines.append("")

        lines.append(f"**Suggested:** {focus.suggested_action}")

        prompt = "\n".join(lines)

        # Rough token estimate (4 chars per token)
        if len(prompt) > token_budget * 4:
            prompt = prompt[:token_budget * 4] + "..."

        return prompt

    def _suggest_next_action(self, in_progress: List[Task],
                            blocked: List[Task], goals: List[Goal]) -> str:
        """Suggest next action based on context."""
        if in_progress:
            return f"Resume: {in_progress[0].title}"

        if blocked:
            blocker = blocked[0].blockers[0] if blocked[0].blockers else None
            if blocker:
                return f"Resolve blocker for '{blocked[0].title}': {blocker.description}"

        if goals:
            return f"Work toward goal: {goals[0].title}"

        return "No active work found. Create a new task or goal."

    async def _get_last_session(self, project_path: Optional[str] = None) -> Optional[Session]:
        """Get the last completed session."""
        sql = "SELECT * FROM sessions WHERE ended_at IS NOT NULL"
        params = []

        if project_path:
            sql += " AND project_path = ?"
            params.append(project_path)

        sql += " ORDER BY ended_at DESC LIMIT 1"

        rows = await self.sqlite.execute_raw(sql, params)
        if not rows:
            return None

        row = rows[0]
        return Session(
            id=row["id"],
            project_path=row.get("project_path"),
            focus_task_id=row.get("focus_task_id"),
            started_at=datetime.fromisoformat(row["started_at"]) if row.get("started_at") else lambda: datetime.now(timezone.utc)(),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row.get("ended_at") else None,
            summary=row.get("summary"),
            handoff_notes=row.get("handoff_notes"),
            insights=row.get("insights", "").split(",") if row.get("insights") else [],
            open_questions=row.get("open_questions", "").split(",") if row.get("open_questions") else []
        )
