"""Knowledge graph for entity and fact management."""

import re
import uuid
from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

from nexus.storage import SQLiteStore, VectorStore, LocalEmbedder, VectorChunk
from nexus.core.exceptions import NotFoundError, DuplicateError


class EntityType(str, Enum):
    PROJECT = "project"
    TECHNOLOGY = "technology"
    PERSON = "person"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    FILE_PATH = "file_path"
    DECISION = "decision"
    TASK = "task"


class RelationType(str, Enum):
    USES = "uses"
    DEPENDS_ON = "depends_on"
    CREATED_BY = "created_by"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    DECIDED_FOR = "decided_for"
    IMPLEMENTS = "implements"
    REPLACES = "replaces"


@dataclass
class Entity:
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


@dataclass
class Relation:
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


@dataclass
class Fact:
    id: str
    statement: str
    topic: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    confidence: float = 0.5
    verification_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


class KnowledgeGraph:
    """Knowledge graph for entities, relationships, and facts."""

    def __init__(self, sqlite_store: SQLiteStore, vector_store: VectorStore, embedder: LocalEmbedder):
        self.sqlite = sqlite_store
        self.vector_store = vector_store
        self.embedder = embedder

        # Entity extraction patterns
        self._tech_patterns = [
            r'\b(Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|Ruby)\b',
            r'\b(React|Vue|Angular|FastAPI|Django|Flask|Next\.js)\b',
            r'\b(PostgreSQL|MySQL|MongoDB|Redis|SQLite|ChromaDB)\b',
            r'\b(Docker|Kubernetes|AWS|GCP|Azure|Ollama)\b',
        ]
        self._path_pattern = r'[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*'

    # Entity CRUD
    async def create_entity(self, entity: Entity) -> str:
        existing = await self.sqlite.query("entities",
            where={"name": entity.name, "entity_type": entity.entity_type.value})
        if existing:
            raise DuplicateError(f"Entity exists: {entity.name}")

        await self.sqlite.insert("entities", {
            "id": entity.id, "name": entity.name,
            "entity_type": entity.entity_type.value,
            "properties": str(entity.properties),
            "created_at": entity.created_at.isoformat()
        })
        return entity.id

    async def get_entity(self, entity_id: str) -> Entity:
        data = await self.sqlite.get("entities", entity_id)
        if not data:
            raise NotFoundError(f"Entity not found: {entity_id}")
        return self._row_to_entity(data)

    async def find_entity(self, name: str, entity_type: Optional[EntityType] = None) -> Optional[Entity]:
        where = {"name": name}
        if entity_type:
            where["entity_type"] = entity_type.value
        results = await self.sqlite.query("entities", where=where, limit=1)
        return self._row_to_entity(results[0]) if results else None

    async def search_entities(self, query: str, entity_type: Optional[EntityType] = None,
                             limit: int = 20) -> List[Entity]:
        sql = "SELECT * FROM entities WHERE name LIKE ?"
        params = [f"%{query}%"]
        if entity_type:
            sql += " AND entity_type = ?"
            params.append(entity_type.value)
        sql += f" ORDER BY name LIMIT {limit}"

        results = await self.sqlite.execute_raw(sql, params)
        return [self._row_to_entity(r) for r in results]

    def _row_to_entity(self, row: Dict) -> Entity:
        props = {}
        if row.get("properties"):
            try:
                props = eval(row["properties"])
            except (SyntaxError, NameError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse entity properties: {e}")
                props = {}
        return Entity(
            id=row["id"], name=row["name"],
            entity_type=EntityType(row["entity_type"]),
            properties=props,
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else lambda: datetime.now(timezone.utc)()
        )

    # Relations
    async def create_relation(self, relation: Relation) -> str:
        await self.sqlite.insert("entity_relations", {
            "id": relation.id, "source_id": relation.source_id,
            "target_id": relation.target_id,
            "relation_type": relation.relation_type.value,
            "properties": str(relation.properties),
            "confidence": relation.confidence
        })
        return relation.id

    async def get_relations(self, entity_id: str,
                           relation_type: Optional[RelationType] = None) -> List[Tuple[Relation, Entity]]:
        relations = []

        sql = "SELECT * FROM entity_relations WHERE source_id = ? OR target_id = ?"
        params = [entity_id, entity_id]
        if relation_type:
            sql += " AND relation_type = ?"
            params.append(relation_type.value)

        rows = await self.sqlite.execute_raw(sql, params)
        for row in rows:
            rel = self._row_to_relation(row)
            other_id = rel.target_id if rel.source_id == entity_id else rel.source_id
            try:
                other = await self.get_entity(other_id)
                relations.append((rel, other))
            except NotFoundError:
                pass

        return relations

    def _row_to_relation(self, row: Dict) -> Relation:
        props = {}
        if row.get("properties"):
            try:
                props = eval(row["properties"])
            except (SyntaxError, NameError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse relation properties: {e}")
                props = {}
        return Relation(
            id=row["id"], source_id=row["source_id"], target_id=row["target_id"],
            relation_type=RelationType(row["relation_type"]),
            properties=props,
            confidence=row.get("confidence", 1.0)
        )

    # Facts
    async def add_fact(self, fact: Fact) -> str:
        await self.sqlite.insert("facts", {
            "id": fact.id, "statement": fact.statement, "topic": fact.topic,
            "source_type": fact.source_type, "source_id": fact.source_id,
            "confidence": fact.confidence, "verification_count": fact.verification_count,
            "created_at": fact.created_at.isoformat()
        })

        # Also add to vector store
        embedding = await self.embedder.embed(fact.statement)
        self.vector_store.add([VectorChunk(
            id=f"fact-{fact.id}", text=fact.statement, embedding=embedding,
            metadata={"type": "fact", "fact_id": fact.id, "topic": fact.topic or "", "confidence": fact.confidence}
        )])

        return fact.id

    async def search_facts(self, query: str, topic: Optional[str] = None,
                          min_confidence: float = 0.0, limit: int = 10) -> List[Fact]:
        query_embedding = await self.embedder.embed(query)

        where = {"type": "fact"}
        if topic:
            where["topic"] = topic

        results = self.vector_store.search(query_embedding, limit, where)

        facts = []
        for result in results:
            fact_id = result.chunk.metadata.get("fact_id")
            if fact_id:
                fact_data = await self.sqlite.get("facts", fact_id)
                if fact_data and fact_data.get("confidence", 0) >= min_confidence:
                    facts.append(self._row_to_fact(fact_data))

        return facts

    async def verify_fact(self, fact_id: str):
        """Increment verification count."""
        fact = await self.sqlite.get("facts", fact_id)
        if not fact:
            raise NotFoundError(f"Fact not found: {fact_id}")

        await self.sqlite.update("facts", fact_id, {
            "verification_count": fact.get("verification_count", 0) + 1,
            "last_verified": lambda: datetime.now(timezone.utc)().isoformat(),
            "confidence": min(1.0, fact.get("confidence", 0.5) + 0.1)
        })

    def _row_to_fact(self, row: Dict) -> Fact:
        return Fact(
            id=row["id"], statement=row["statement"], topic=row.get("topic"),
            source_type=row.get("source_type"), source_id=row.get("source_id"),
            confidence=row.get("confidence", 0.5),
            verification_count=row.get("verification_count", 0),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else lambda: datetime.now(timezone.utc)()
        )

    # Entity extraction
    async def extract_entities_from_text(self, text: str) -> List[Entity]:
        entities = []
        seen_names: Set[str] = set()

        for pattern in self._tech_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(0)
                if name.lower() not in seen_names:
                    seen_names.add(name.lower())
                    entities.append(Entity(
                        id=Entity.generate_id(), name=name, entity_type=EntityType.TECHNOLOGY
                    ))

        for match in re.finditer(self._path_pattern, text):
            path = match.group(0)
            if path not in seen_names:
                seen_names.add(path)
                entities.append(Entity(
                    id=Entity.generate_id(), name=path, entity_type=EntityType.FILE_PATH
                ))

        return entities

    async def get_stats(self) -> Dict[str, Any]:
        entity_count = await self.sqlite.count("entities")
        relation_count = await self.sqlite.count("entity_relations")
        fact_count = await self.sqlite.count("facts")

        type_breakdown = await self.sqlite.execute_raw(
            "SELECT entity_type, COUNT(*) as count FROM entities GROUP BY entity_type"
        )

        return {
            "total_entities": entity_count,
            "total_relations": relation_count,
            "total_facts": fact_count,
            "entities_by_type": {r["entity_type"]: r["count"] for r in type_breakdown}
        }
