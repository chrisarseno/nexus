"""Decision logging and retrieval."""

import uuid
from typing import List, Optional, Dict
from datetime import datetime, timezone
from dataclasses import dataclass, field

from nexus.storage import SQLiteStore
from nexus.core.exceptions import NotFoundError


@dataclass
class Decision:
    id: str
    question: str
    decision: str
    reasoning: str
    alternatives: List[str] = field(default_factory=list)
    project_path: Optional[str] = None
    task_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


class DecisionLog:
    """Log and retrieve decisions."""

    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    async def record_decision(self, decision: Decision) -> str:
        """Record a decision."""
        await self.sqlite.insert("decisions", {
            "id": decision.id,
            "question": decision.question,
            "decision": decision.decision,
            "reasoning": decision.reasoning,
            "alternatives": ",".join(decision.alternatives) if decision.alternatives else None,
            "project_path": decision.project_path,
            "task_id": decision.task_id,
            "created_at": decision.created_at.isoformat()
        })
        return decision.id

    async def get_decision(self, decision_id: str) -> Decision:
        """Get decision by ID."""
        data = await self.sqlite.get("decisions", decision_id)
        if not data:
            raise NotFoundError(f"Decision not found: {decision_id}")
        return self._row_to_decision(data)

    async def list_decisions(self, project_path: Optional[str] = None,
                            task_id: Optional[str] = None,
                            search_term: Optional[str] = None,
                            limit: int = 20) -> List[Decision]:
        """List decisions with filters."""
        sql = "SELECT * FROM decisions WHERE 1=1"
        params = []

        if project_path:
            sql += " AND project_path = ?"
            params.append(project_path)
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        if search_term:
            sql += " AND (question LIKE ? OR decision LIKE ? OR reasoning LIKE ?)"
            params.extend([f"%{search_term}%"] * 3)

        sql += f" ORDER BY created_at DESC LIMIT {limit}"

        rows = await self.sqlite.execute_raw(sql, params)
        return [self._row_to_decision(r) for r in rows]

    def _row_to_decision(self, row: Dict) -> Decision:
        return Decision(
            id=row["id"], question=row["question"],
            decision=row["decision"], reasoning=row["reasoning"],
            alternatives=row.get("alternatives", "").split(",") if row.get("alternatives") else [],
            project_path=row.get("project_path"), task_id=row.get("task_id"),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else lambda: datetime.now(timezone.utc)()
        )
