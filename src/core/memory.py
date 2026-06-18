"""
RAG Memory - Conversation History via LangGraph InMemoryStore

This module manages conversation context using LangGraph's native InMemoryStore.
The store handles cross-thread memory persistence and structured retrieval.

The public interface (ConversationMemory.add / get_active_context) is preserved so
all callers (memory_service.py) work without modification.

Replaces: custom cosine similarity, numpy decay math, MemoryEntry, tiktoken budgeting.
"""

import logging
import time
from typing import Optional
from langgraph.store.memory import InMemoryStore

from src.core.config import MEMORY_TOKEN_BUDGET, TOKENIZER_ENCODING

logger = logging.getLogger("RAG.Memory")

# Shared store instance — one store per process, namespaced by session
_store = InMemoryStore()


class ConversationMemory:
    """
    Manages conversation history for a single session using LangGraph InMemoryStore.

    Provides the same public interface as before:
    - add(text, importance, role): Add a turn to memory
    - get_active_context(): Retrieve the formatted context string

    Deduplication is handled by namespace+key scoping in the store.
    Token budgeting uses tiktoken for backward compatibility.
    """

    def __init__(self, session_id: str = None, max_tokens: int = MEMORY_TOKEN_BUDGET):
        import uuid
        actual_session_id = session_id or f"default_{uuid.uuid4().hex[:8]}"
        self._session_id = actual_session_id
        self._namespace = ("memory", actual_session_id)
        self._turn_counter = 0
        self._max_tokens = max_tokens
        logger.debug(f"ConversationMemory initialised for session '{actual_session_id}'")

    def add(self, text: str, importance: float = 1.0, role: str = "user") -> None:
        """
        Adds a turn to memory. Duplicate texts (same role+content) are silently ignored
        via deterministic key generation.
        """
        self._turn_counter += 1
        # Deterministic key: role + hash of text prevents storing exact duplicates
        import hashlib
        key = f"{role}_{hashlib.md5(text.encode()).hexdigest()[:12]}"
        _store.put(
            self._namespace,
            key,
            {
                "text": text,
                "role": role,
                "importance": importance,
                "turn": self._turn_counter,
            },
        )
        logger.debug(f"Memory stored: role={role}, turn={self._turn_counter}, key={key}")

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def entries(self) -> list['MemoryEntry']:
        items = _store.search(self._namespace, limit=10000)
        sorted_items = sorted(items, key=lambda x: x.value.get("turn", 0))
        return [
            MemoryEntry(
                text=i.value.get("text", ""),
                importance=i.value.get("importance", 1.0),
                role=i.value.get("role", "user"),
                turn_count=i.value.get("turn", 0),
            )
            for i in sorted_items
        ]

    @entries.setter
    def entries(self, value: list['MemoryEntry']):
        try:
            items = _store.search(self._namespace, limit=10000)
            for item in items:
                _store.delete(self._namespace, item.key)
        except Exception as e:
            logger.warning(f"Failed to clear store during prune: {e}")

        max_turn = 0
        for entry in value:
            if entry.turn_count > max_turn:
                max_turn = entry.turn_count
            import hashlib
            key = f"{entry.role}_{hashlib.md5(entry.text.encode()).hexdigest()[:12]}"
            _store.put(
                self._namespace,
                key,
                {
                    "text": entry.text,
                    "role": entry.role,
                    "importance": entry.base_importance,
                    "turn": entry.turn_count,
                },
            )
        if max_turn > 0:
            self._turn_counter = max_turn

    def get_active_context(self) -> str:
        """
        Retrieves all stored memory items, sorts by turn order descending (newest first),
        and returns a formatted string within the token budget.
        """
        t_start = time.time()
        try:
            import tiktoken
            tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)
        except Exception:
            tokenizer = None

        try:
            items = _store.search(self._namespace, limit=10000)
        except Exception as e:
            logger.warning(f"Memory store search failed: {e}")
            return ""

        # Sort by turn number descending to prioritize recent turns
        entries = sorted(
            [i.value for i in items],
            key=lambda x: x.get("turn", 0),
            reverse=True,
        )

        from src.core.config import MEMORY_TURN_DECAY_RATE, MEMORY_WEIGHT_THRESHOLD

        context_parts = []
        total_tokens = 0
        for entry in entries:
            # Apply turn-based importance decay
            entry_turn = entry.get("turn", 0)
            importance = entry.get("importance", 1.0)
            age = max(0, self._turn_counter - entry_turn)
            weight = importance * ((1.0 - MEMORY_TURN_DECAY_RATE) ** age)
            
            if weight < MEMORY_WEIGHT_THRESHOLD:
                continue

            line = f"[{entry.get('role', 'user')}]: {entry.get('text', '')}\n"
            if tokenizer:
                line_tokens = len(tokenizer.encode(line))
                if total_tokens + line_tokens > self._max_tokens:
                    break
                total_tokens += line_tokens
            context_parts.insert(0, line)  # Prepend to restore chronological order

        t_ms = (time.time() - t_start) * 1000
        logger.info(
            f"Extracted context: {len(context_parts)}/{len(entries)} entries "
            f"({total_tokens} tokens) in {t_ms:.1f}ms"
        )
        return "".join(context_parts)


# Backward-compat: MemoryEntry is no longer used internally but kept as a stub
# so any code that imports it doesn't crash.
class MemoryEntry:
    """Stub preserved for backward compatibility. Not used by ConversationMemory."""
    def __init__(self, text: str, importance: float = 1.0, role: str = "user", turn_count: int = 0):
        self.text = text
        self.base_importance = importance
        self.role = role
        self.turn_count = turn_count