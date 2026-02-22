"""Data models for intelligence layer."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import uuid


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None


@dataclass
class Conversation:
    id: str
    title: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    project_path: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryChunk:
    id: str
    conversation_id: str
    text: str
    chunk_index: int
    message_indices: List[int]
    project_path: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


@dataclass
class SearchHit:
    chunk: "MemoryChunk"
    score: float
    snippet: str
    conversation_id: str
    conversation_title: Optional[str] = None

    @property
    def relevance(self) -> float:
        return 1.0 / (1.0 + self.score)


@dataclass
class TopicMention:
    conversation_id: str
    conversation_title: Optional[str]
    chunk_text: str
    timestamp: datetime
    relevance: float
