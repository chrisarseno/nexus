"""Task management with dependencies and blockers."""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

from nexus.storage import SQLiteStore
from nexus.core.exceptions import NotFoundError


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKLOG = "backlog"


class BlockerType(str, Enum):
    DECISION_NEEDED = "decision_needed"
    DEPENDENCY = "dependency"
    EXTERNAL = "external"
    CLARIFICATION = "clarification"
    TECHNICAL = "technical"


@dataclass
class Blocker:
    id: str
    task_id: str
    blocker_type: BlockerType
    description: str
    resolved: bool = False
    resolution: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


@dataclass
class TaskNote:
    id: str
    task_id: str
    note: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


@dataclass
class Task:
    id: str
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    project_path: Optional[str] = None
    parent_task_id: Optional[str] = None
    context: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: List[TaskNote] = field(default_factory=list)
    blockers: List[Blocker] = field(default_factory=list)
    linked_files: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

    @property
    def is_blocked(self) -> bool:
        return any(not b.resolved for b in self.blockers)


class TaskManager:
    """Manage tasks with full lifecycle support."""

    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    async def create_task(self, task: Task) -> str:
        """Create a new task."""
        await self.sqlite.insert("tasks", {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "status": task.status.value,
            "priority": task.priority.value,
            "project_path": task.project_path,
            "parent_task_id": task.parent_task_id,
            "context": task.context,
            "tags": ",".join(task.tags) if task.tags else None,
            "created_at": task.created_at.isoformat()
        })
        return task.id

    async def get_task(self, task_id: str) -> Task:
        """Get task by ID with all related data."""
        data = await self.sqlite.get("tasks", task_id)
        if not data:
            raise NotFoundError(f"Task not found: {task_id}")

        notes = await self._get_notes(task_id)
        blockers = await self._get_blockers(task_id)
        files = await self._get_files(task_id)

        return self._row_to_task(data, notes, blockers, files)

    async def list_tasks(self, project_path: Optional[str] = None,
                        status: Optional[TaskStatus] = None,
                        priority: Optional[TaskPriority] = None,
                        include_completed: bool = False,
                        limit: int = 50) -> List[Task]:
        """List tasks with filters."""
        sql = "SELECT * FROM tasks WHERE 1=1"
        params = []

        if project_path:
            sql += " AND project_path = ?"
            params.append(project_path)
        if status:
            sql += " AND status = ?"
            params.append(status.value)
        elif not include_completed:
            sql += " AND status NOT IN ('completed', 'cancelled')"
        if priority:
            sql += " AND priority = ?"
            params.append(priority.value)

        sql += " ORDER BY CASE priority WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 WHEN 'low' THEN 4 ELSE 5 END"
        sql += f" LIMIT {limit}"

        rows = await self.sqlite.execute_raw(sql, params)

        tasks = []
        for row in rows:
            notes = await self._get_notes(row["id"])
            blockers = await self._get_blockers(row["id"])
            files = await self._get_files(row["id"])
            tasks.append(self._row_to_task(row, notes, blockers, files))

        return tasks

    async def update_task(self, task_id: str, updates: Dict[str, Any]):
        """Update task fields."""
        if "status" in updates and isinstance(updates["status"], TaskStatus):
            updates["status"] = updates["status"].value
        if "priority" in updates and isinstance(updates["priority"], TaskPriority):
            updates["priority"] = updates["priority"].value
        if "tags" in updates and isinstance(updates["tags"], list):
            updates["tags"] = ",".join(updates["tags"])

        await self.sqlite.update("tasks", task_id, updates)

    async def start_task(self, task_id: str) -> Task:
        """Start working on a task."""
        task = await self.get_task(task_id)

        if task.is_blocked:
            raise ValueError("Cannot start blocked task")

        await self.update_task(task_id, {"status": TaskStatus.IN_PROGRESS.value})
        return await self.get_task(task_id)

    async def complete_task(self, task_id: str, note: Optional[str] = None) -> Task:
        """Mark task as completed."""
        await self.update_task(task_id, {
            "status": TaskStatus.COMPLETED.value,
            "completed_at": lambda: datetime.now(timezone.utc)().isoformat()
        })

        if note:
            await self.add_note(task_id, note)

        return await self.get_task(task_id)

    async def block_task(self, task_id: str, blocker_type: BlockerType,
                        description: str) -> Blocker:
        """Add blocker to task."""
        blocker = Blocker(
            id=Blocker.generate_id(),
            task_id=task_id,
            blocker_type=blocker_type,
            description=description
        )

        await self.sqlite.insert("blockers", {
            "id": blocker.id,
            "task_id": task_id,
            "blocker_type": blocker_type.value,
            "description": description,
            "resolved": False,
            "created_at": blocker.created_at.isoformat()
        })

        await self.update_task(task_id, {"status": TaskStatus.BLOCKED.value})
        return blocker

    async def resolve_blocker(self, blocker_id: str, resolution: str,
                             resume_task: bool = True) -> Blocker:
        """Resolve a blocker."""
        await self.sqlite.update("blockers", blocker_id, {
            "resolved": True,
            "resolution": resolution,
            "resolved_at": lambda: datetime.now(timezone.utc)().isoformat()
        })

        data = await self.sqlite.get("blockers", blocker_id)
        task_id = data["task_id"]

        # Check if task should resume
        if resume_task:
            blockers = await self._get_blockers(task_id)
            active_blockers = [b for b in blockers if not b.resolved]
            if not active_blockers:
                await self.update_task(task_id, {"status": TaskStatus.IN_PROGRESS.value})

        return self._row_to_blocker(data)

    async def add_note(self, task_id: str, note: str) -> TaskNote:
        """Add note to task."""
        task_note = TaskNote(
            id=TaskNote.generate_id(),
            task_id=task_id,
            note=note
        )

        await self.sqlite.insert("task_notes", {
            "id": task_note.id,
            "task_id": task_id,
            "note": note,
            "created_at": task_note.created_at.isoformat()
        })

        return task_note

    async def link_files(self, task_id: str, files: List[str]):
        """Link files to task."""
        for file_path in files:
            try:
                await self.sqlite.insert("task_files", {
                    "task_id": task_id,
                    "file_path": file_path
                })
            except Exception:
                pass  # Already linked

    async def get_subtasks(self, parent_id: str) -> List[Task]:
        """Get subtasks of a task."""
        rows = await self.sqlite.query("tasks", where={"parent_task_id": parent_id})
        tasks = []
        for row in rows:
            notes = await self._get_notes(row["id"])
            blockers = await self._get_blockers(row["id"])
            files = await self._get_files(row["id"])
            tasks.append(self._row_to_task(row, notes, blockers, files))
        return tasks

    async def _get_notes(self, task_id: str) -> List[TaskNote]:
        rows = await self.sqlite.query("task_notes", where={"task_id": task_id},
                                       order_by="created_at DESC")
        return [TaskNote(id=r["id"], task_id=task_id, note=r["note"],
                        created_at=datetime.fromisoformat(r["created_at"])) for r in rows]

    async def _get_blockers(self, task_id: str) -> List[Blocker]:
        rows = await self.sqlite.query("blockers", where={"task_id": task_id})
        return [self._row_to_blocker(r) for r in rows]

    async def _get_files(self, task_id: str) -> List[str]:
        rows = await self.sqlite.query("task_files", where={"task_id": task_id})
        return [r["file_path"] for r in rows]

    def _row_to_task(self, row: Dict, notes: List[TaskNote],
                    blockers: List[Blocker], files: List[str]) -> Task:
        return Task(
            id=row["id"], title=row["title"], description=row.get("description"),
            status=TaskStatus(row.get("status", "pending")),
            priority=TaskPriority(row.get("priority", "medium")),
            project_path=row.get("project_path"),
            parent_task_id=row.get("parent_task_id"),
            context=row.get("context"),
            tags=row.get("tags", "").split(",") if row.get("tags") else [],
            notes=notes, blockers=blockers, linked_files=files,
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else lambda: datetime.now(timezone.utc)(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else lambda: datetime.now(timezone.utc)(),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row.get("completed_at") else None
        )

    def _row_to_blocker(self, row: Dict) -> Blocker:
        return Blocker(
            id=row["id"], task_id=row["task_id"],
            blocker_type=BlockerType(row["blocker_type"]),
            description=row["description"],
            resolved=bool(row.get("resolved")),
            resolution=row.get("resolution"),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else lambda: datetime.now(timezone.utc)(),
            resolved_at=datetime.fromisoformat(row["resolved_at"]) if row.get("resolved_at") else None
        )
