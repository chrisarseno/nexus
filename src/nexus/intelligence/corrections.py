"""Correction tracking for accuracy feedback."""

import uuid
from typing import List, Optional, Dict
from datetime import datetime, timezone
from dataclasses import dataclass, field

from nexus.storage import SQLiteStore


@dataclass
class Correction:
    id: str
    original: str
    corrected: str
    reason: str
    topic: Optional[str] = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


class CorrectionLog:
    """Track corrections for accuracy feedback."""

    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    async def record_correction(self, correction: Correction) -> str:
        """Record a correction."""
        await self.sqlite.insert("corrections", {
            "id": correction.id,
            "original": correction.original,
            "corrected": correction.corrected,
            "reason": correction.reason,
            "topic": correction.topic,
            "confidence": correction.confidence,
            "created_at": correction.created_at.isoformat()
        })
        return correction.id

    async def get_corrections(self, topic: Optional[str] = None,
                             search_term: Optional[str] = None,
                             limit: int = 20) -> List[Correction]:
        """Get corrections."""
        sql = "SELECT * FROM corrections WHERE 1=1"
        params = []

        if topic:
            sql += " AND topic = ?"
            params.append(topic)
        if search_term:
            sql += " AND (original LIKE ? OR corrected LIKE ?)"
            params.extend([f"%{search_term}%"] * 2)

        sql += f" ORDER BY created_at DESC LIMIT {limit}"

        rows = await self.sqlite.execute_raw(sql, params)
        return [self._row_to_correction(r) for r in rows]

    async def get_correction_stats(self) -> Dict:
        """Get correction statistics."""
        total = await self.sqlite.count("corrections")
        by_topic = await self.sqlite.execute_raw(
            "SELECT topic, COUNT(*) as count FROM corrections GROUP BY topic"
        )
        return {"total": total, "by_topic": {r["topic"] or "general": r["count"] for r in by_topic}}

    def _row_to_correction(self, row: Dict) -> Correction:
        return Correction(
            id=row["id"], original=row["original"],
            corrected=row["corrected"], reason=row["reason"],
            topic=row.get("topic"), confidence=row.get("confidence", 1.0),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else lambda: datetime.now(timezone.utc)()
        )
