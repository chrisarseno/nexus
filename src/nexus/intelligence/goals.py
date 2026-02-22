"""Goal hierarchy management."""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

from nexus.storage import SQLiteStore
from nexus.core.exceptions import NotFoundError


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    PAUSED = "paused"


class MilestoneStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class Milestone:
    id: str
    goal_id: str
    title: str
    description: Optional[str] = None
    status: MilestoneStatus = MilestoneStatus.PENDING
    target_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


@dataclass
class Goal:
    id: str
    title: str
    description: Optional[str] = None
    project_path: Optional[str] = None
    status: GoalStatus = GoalStatus.ACTIVE
    milestones: List[Milestone] = field(default_factory=list)
    target_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

    @property
    def progress(self) -> float:
        if not self.milestones:
            return 0.0
        completed = sum(1 for m in self.milestones if m.status == MilestoneStatus.COMPLETED)
        return completed / len(self.milestones)


class GoalManager:
    """Manage goals and milestones."""

    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    async def create_goal(self, goal: Goal) -> str:
        """Create a new goal."""
        await self.sqlite.insert("goals", {
            "id": goal.id,
            "title": goal.title,
            "description": goal.description,
            "project_path": goal.project_path,
            "status": goal.status.value,
            "target_date": goal.target_date.isoformat() if goal.target_date else None,
            "created_at": goal.created_at.isoformat()
        })

        for milestone in goal.milestones:
            await self.add_milestone(goal.id, milestone)

        return goal.id

    async def get_goal(self, goal_id: str) -> Goal:
        """Get goal by ID."""
        data = await self.sqlite.get("goals", goal_id)
        if not data:
            raise NotFoundError(f"Goal not found: {goal_id}")

        milestones = await self._get_milestones(goal_id)
        return self._row_to_goal(data, milestones)

    async def list_goals(self, project_path: Optional[str] = None,
                        status: Optional[GoalStatus] = None,
                        include_completed: bool = False) -> List[Goal]:
        """List goals with filters."""
        where = {}
        if project_path:
            where["project_path"] = project_path
        if status:
            where["status"] = status.value

        rows = await self.sqlite.query("goals", where=where if where else None, order_by="created_at DESC")

        goals = []
        for row in rows:
            if not include_completed and row.get("status") == GoalStatus.COMPLETED.value:
                continue
            milestones = await self._get_milestones(row["id"])
            goals.append(self._row_to_goal(row, milestones))

        return goals

    async def update_goal(self, goal_id: str, updates: Dict[str, Any]):
        """Update goal."""
        if "status" in updates and isinstance(updates["status"], GoalStatus):
            updates["status"] = updates["status"].value
        await self.sqlite.update("goals", goal_id, updates)

    async def complete_goal(self, goal_id: str) -> Goal:
        """Mark goal as completed."""
        await self.sqlite.update("goals", goal_id, {
            "status": GoalStatus.COMPLETED.value,
            "completed_at": lambda: datetime.now(timezone.utc)().isoformat()
        })
        return await self.get_goal(goal_id)

    async def add_milestone(self, goal_id: str, milestone: Milestone) -> str:
        """Add milestone to goal."""
        milestone.goal_id = goal_id
        await self.sqlite.insert("milestones", {
            "id": milestone.id,
            "goal_id": goal_id,
            "title": milestone.title,
            "description": milestone.description,
            "status": milestone.status.value,
            "target_date": milestone.target_date.isoformat() if milestone.target_date else None
        })
        return milestone.id

    async def complete_milestone(self, milestone_id: str) -> Milestone:
        """Complete a milestone."""
        await self.sqlite.update("milestones", milestone_id, {
            "status": MilestoneStatus.COMPLETED.value,
            "completed_at": lambda: datetime.now(timezone.utc)().isoformat()
        })

        # Check if goal should auto-complete
        data = await self.sqlite.get("milestones", milestone_id)
        if data:
            milestones = await self._get_milestones(data["goal_id"])
            all_done = all(m.status == MilestoneStatus.COMPLETED for m in milestones)
            if all_done:
                await self.complete_goal(data["goal_id"])

        return await self._get_milestone(milestone_id)

    async def _get_milestones(self, goal_id: str) -> List[Milestone]:
        rows = await self.sqlite.query("milestones", where={"goal_id": goal_id})
        return [self._row_to_milestone(r) for r in rows]

    async def _get_milestone(self, milestone_id: str) -> Milestone:
        data = await self.sqlite.get("milestones", milestone_id)
        if not data:
            raise NotFoundError(f"Milestone not found: {milestone_id}")
        return self._row_to_milestone(data)

    def _row_to_goal(self, row: Dict, milestones: List[Milestone]) -> Goal:
        return Goal(
            id=row["id"], title=row["title"], description=row.get("description"),
            project_path=row.get("project_path"),
            status=GoalStatus(row.get("status", "active")),
            milestones=milestones,
            target_date=datetime.fromisoformat(row["target_date"]) if row.get("target_date") else None,
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else lambda: datetime.now(timezone.utc)(),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row.get("completed_at") else None
        )

    def _row_to_milestone(self, row: Dict) -> Milestone:
        return Milestone(
            id=row["id"], goal_id=row["goal_id"], title=row["title"],
            description=row.get("description"),
            status=MilestoneStatus(row.get("status", "pending")),
            target_date=datetime.fromisoformat(row["target_date"]) if row.get("target_date") else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row.get("completed_at") else None
        )
