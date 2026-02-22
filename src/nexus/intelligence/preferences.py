"""User preference learning and retrieval."""

import uuid
from typing import List, Optional, Dict
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

from nexus.storage import SQLiteStore


class PreferenceCategory(str, Enum):
    CODING_STYLE = "coding_style"
    LIBRARIES = "libraries"
    ARCHITECTURE = "architecture"
    WORKFLOW = "workflow"
    COMMUNICATION = "communication"
    TOOLS = "tools"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


@dataclass
class Preference:
    id: str
    category: PreferenceCategory
    key: str
    value: str
    confidence: float = 0.5
    observation_count: int = 1
    last_observed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


class PreferenceStore:
    """Learn and store user preferences."""

    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    async def observe_preference(self, category: PreferenceCategory,
                                key: str, value: str) -> Preference:
        """Observe a preference (create or reinforce)."""
        existing = await self.sqlite.execute_raw(
            "SELECT * FROM preferences WHERE category = ? AND key = ?",
            [category.value, key]
        )

        if existing:
            row = existing[0]
            new_count = row.get("observation_count", 1) + 1
            new_confidence = min(1.0, 0.5 + new_count * 0.1)

            await self.sqlite.update("preferences", row["id"], {
                "value": value,
                "confidence": new_confidence,
                "observation_count": new_count,
                "last_observed": lambda: datetime.now(timezone.utc)().isoformat()
            })

            return Preference(
                id=row["id"], category=category, key=key, value=value,
                confidence=new_confidence, observation_count=new_count
            )
        else:
            pref = Preference(
                id=Preference.generate_id(),
                category=category, key=key, value=value
            )

            await self.sqlite.insert("preferences", {
                "id": pref.id,
                "category": category.value,
                "key": key,
                "value": value,
                "confidence": pref.confidence,
                "observation_count": 1,
                "last_observed": lambda: datetime.now(timezone.utc)().isoformat()
            })

            return pref

    async def get_preferences(self, category: Optional[PreferenceCategory] = None,
                             min_confidence: float = 0.0) -> List[Preference]:
        """Get preferences."""
        sql = "SELECT * FROM preferences WHERE confidence >= ?"
        params = [min_confidence]

        if category:
            sql += " AND category = ?"
            params.append(category.value)

        sql += " ORDER BY confidence DESC"

        rows = await self.sqlite.execute_raw(sql, params)
        return [self._row_to_preference(r) for r in rows]

    async def recall_preference(self, topic: str) -> Optional[Preference]:
        """Recall preference for a topic."""
        rows = await self.sqlite.execute_raw(
            "SELECT * FROM preferences WHERE key LIKE ? ORDER BY confidence DESC LIMIT 1",
            [f"%{topic}%"]
        )
        return self._row_to_preference(rows[0]) if rows else None

    async def get_preference_summary(self) -> Dict:
        """Get preference summary."""
        prefs = await self.get_preferences(min_confidence=0.6)
        by_category = {}
        for pref in prefs:
            cat = pref.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({"key": pref.key, "value": pref.value, "confidence": pref.confidence})
        return {"total": len(prefs), "by_category": by_category}

    def _row_to_preference(self, row: Dict) -> Preference:
        return Preference(
            id=row["id"],
            category=PreferenceCategory(row["category"]),
            key=row["key"], value=row["value"],
            confidence=row.get("confidence", 0.5),
            observation_count=row.get("observation_count", 1),
            last_observed=datetime.fromisoformat(row["last_observed"]) if row.get("last_observed") else lambda: datetime.now(timezone.utc)()
        )
