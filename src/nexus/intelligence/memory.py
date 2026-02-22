"""Memory module for conversation indexing and semantic search."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from nexus.storage import LocalEmbedder, VectorStore, VectorChunk, SQLiteStore
from nexus.intelligence.models import (
    Conversation, Message, MemoryChunk, SearchHit, TopicMention, MessageRole
)


@dataclass
class ChunkingConfig:
    max_chunk_size: int = 1500
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    include_context_messages: int = 2


class NexusMemory:
    """Main memory module."""

    def __init__(self, vector_store: VectorStore, sqlite_store: SQLiteStore,
                 embedder: LocalEmbedder, chunking_config: Optional[ChunkingConfig] = None):
        self.vector_store = vector_store
        self.sqlite_store = sqlite_store
        self.embedder = embedder
        self.chunking_config = chunking_config or ChunkingConfig()

    async def index_conversation(self, conversation: Conversation) -> int:
        """Index a conversation for semantic search."""
        await self._store_conversation_metadata(conversation)
        chunks = self._chunk_conversation(conversation)

        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = await self.embedder.embed_batch(texts)

        vector_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            vector_chunks.append(VectorChunk(
                id=chunk.id, text=chunk.text, embedding=embedding,
                metadata={
                    "conversation_id": chunk.conversation_id,
                    "chunk_index": chunk.chunk_index,
                    "project_path": chunk.project_path or "",
                    "created_at": chunk.created_at.isoformat()
                }
            ))

        self.vector_store.add(vector_chunks)
        return len(chunks)

    async def _store_conversation_metadata(self, conversation: Conversation):
        """Store conversation metadata."""
        conv_data = {
            "id": conversation.id, "title": conversation.title,
            "project_path": conversation.project_path,
            "message_count": len(conversation.messages),
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "metadata": str(conversation.metadata)
        }

        existing = await self.sqlite_store.get("conversations", conversation.id)
        if existing:
            await self.sqlite_store.update("conversations", conversation.id, conv_data)
        else:
            await self.sqlite_store.insert("conversations", conv_data)

    def _chunk_conversation(self, conversation: Conversation) -> List[MemoryChunk]:
        """Chunk conversation into indexable pieces."""
        chunks = []
        current_text = []
        current_indices = []
        chunk_index = 0

        for i, message in enumerate(conversation.messages):
            role_prefix = "Human: " if message.role == MessageRole.USER else "Assistant: "
            message_text = f"{role_prefix}{message.content}"

            test_text = "\n\n".join(current_text + [message_text])

            if len(test_text) > self.chunking_config.max_chunk_size and current_text:
                chunks.append(MemoryChunk(
                    id=MemoryChunk.generate_id(),
                    conversation_id=conversation.id,
                    text="\n\n".join(current_text),
                    chunk_index=chunk_index,
                    message_indices=current_indices.copy(),
                    project_path=conversation.project_path,
                    created_at=conversation.created_at
                ))

                overlap_start = max(0, len(current_text) - self.chunking_config.include_context_messages)
                current_text = current_text[overlap_start:]
                current_indices = current_indices[overlap_start:]
                chunk_index += 1

            current_text.append(message_text)
            current_indices.append(i)

        if current_text and len("\n\n".join(current_text)) >= self.chunking_config.min_chunk_size:
            chunks.append(MemoryChunk(
                id=MemoryChunk.generate_id(),
                conversation_id=conversation.id,
                text="\n\n".join(current_text),
                chunk_index=chunk_index,
                message_indices=current_indices,
                project_path=conversation.project_path,
                created_at=conversation.created_at
            ))

        return chunks

    async def search(self, query: str, n_results: int = 10,
                    project_filter: Optional[str] = None) -> List[SearchHit]:
        """Semantic search across conversations."""
        query_embedding = await self.embedder.embed(query)

        where = {"project_path": project_filter} if project_filter else None
        results = self.vector_store.search(query_embedding, n_results, where)

        hits = []
        for result in results:
            conv = await self.sqlite_store.get(
                "conversations", result.chunk.metadata.get("conversation_id", "")
            )

            hits.append(SearchHit(
                chunk=MemoryChunk(
                    id=result.chunk.id,
                    conversation_id=result.chunk.metadata.get("conversation_id", ""),
                    text=result.chunk.text,
                    chunk_index=result.chunk.metadata.get("chunk_index", 0),
                    message_indices=result.chunk.metadata.get("message_indices", []),
                    project_path=result.chunk.metadata.get("project_path")
                ),
                score=result.score,
                snippet=self._extract_snippet(result.chunk.text, query),
                conversation_id=result.chunk.metadata.get("conversation_id", ""),
                conversation_title=conv.get("title") if conv else None
            ))

        return hits

    def _extract_snippet(self, text: str, query: str, context_chars: int = 150) -> str:
        """Extract relevant snippet."""
        query_words = query.lower().split()
        text_lower = text.lower()

        best_pos = 0
        for word in query_words:
            pos = text_lower.find(word)
            if pos != -1:
                best_pos = pos
                break

        start = max(0, best_pos - context_chars)
        end = min(len(text), best_pos + context_chars)

        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet.strip()

    async def topic_history(self, topic: str, limit: int = 20) -> List[TopicMention]:
        """Get chronological history of a topic."""
        hits = await self.search(topic, n_results=limit * 2)

        mentions = []
        for hit in hits:
            created_at = hit.chunk.created_at
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)

            mentions.append(TopicMention(
                conversation_id=hit.conversation_id,
                conversation_title=hit.conversation_title,
                chunk_text=hit.snippet,
                timestamp=created_at,
                relevance=hit.relevance
            ))

        mentions.sort(key=lambda m: m.timestamp)
        return mentions[:limit]

    async def recall(self, topic: str, context_type: str = "any") -> Dict[str, Any]:
        """Recall specific context about a topic."""
        hits = await self.search(topic, n_results=10)

        if context_type != "any":
            filtered = []
            for hit in hits:
                text_lower = hit.chunk.text.lower()
                if context_type == "decision" and any(w in text_lower for w in ["decided", "decision", "chose"]):
                    filtered.append(hit)
                elif context_type == "preference" and any(w in text_lower for w in ["prefer", "like", "want"]):
                    filtered.append(hit)
                elif context_type == "implementation" and any(w in text_lower for w in ["implemented", "built", "code"]):
                    filtered.append(hit)
                elif context_type == "discussion":
                    filtered.append(hit)
            hits = filtered if filtered else hits

        return {
            "topic": topic,
            "context_type": context_type,
            "results": [
                {"text": h.snippet, "relevance": h.relevance, "conversation_id": h.conversation_id}
                for h in hits[:5]
            ],
            "total_found": len(hits)
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        chunk_count = self.vector_store.count()
        conv_count = await self.sqlite_store.count("conversations")

        projects = await self.sqlite_store.execute_raw(
            "SELECT project_path, COUNT(*) as count FROM conversations GROUP BY project_path"
        )

        return {
            "total_chunks": chunk_count,
            "total_conversations": conv_count,
            "projects": {p["project_path"] or "unassigned": p["count"] for p in projects}
        }

    async def delete_conversation(self, conversation_id: str) -> int:
        """Delete conversation and chunks."""
        chunks = self.vector_store.list_all(where={"conversation_id": conversation_id})
        chunk_ids = [c.id for c in chunks]

        if chunk_ids:
            self.vector_store.delete(chunk_ids)

        await self.sqlite_store.delete("conversations", conversation_id)
        return len(chunk_ids)
