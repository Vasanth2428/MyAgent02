"""
RAG Memory - Conversation History That Forgets Gracefully

This module handles remembering what's been said during your conversation. It works
like a smart notepad that:
- Keeps track of important facts from your chat
- Forgets old information automatically (like a fading memory)
- Recognizes when you're repeating something similar and just refreshes that memory

The memory helps the AI give more contextual answers that consider what you've
already discussed.
"""

import re
import time
import logging
import numpy as np
import tiktoken
from datetime import datetime
from typing import List, Optional
from functools import lru_cache

from src.core.config import (
    TOKENIZER_ENCODING, MEMORY_DECAY_RATE, MEMORY_TOKEN_BUDGET,
    MEMORY_WEIGHT_THRESHOLD, SEMANTIC_DEDUP_THRESHOLD, SEMANTIC_DEDUP_MIN_WORDS
)

# Re-export for backward compatibility with tests
def _get_embedding_model():
    from src.core.services.grounding_service import _get_shared_embedding_model
    return _get_shared_embedding_model()

logger = logging.getLogger("RAG.Memory")
tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def _encode_text(text: str) -> np.ndarray:
    """Encodes text to embedding using shared model."""
    from src.core.services.grounding_service import _get_shared_embedding_model
    return _get_shared_embedding_model().encode(text, normalize_embeddings=True)


class MemoryEntry:
    """
    Represents one message in the conversation.
    
    Each time you or the AI says something, we store it here. The entry tracks:
    - What was said (text)
    - Who said it (role: 'user' or 'assistant')
    - How important it is (importance - helps decide what to keep)
    - When it was last mentioned (last_seen - used for forgetting old info)
    - A mathematical fingerprint for finding similar messages (embedding)
    """

    def __init__(self, text: str, importance: float = 1.0, role: str = "user"):
        self.text = text
        self.base_importance = importance
        self.role = role
        self.last_seen = datetime.now()
        self._embedding: Optional[np.ndarray] = None

    @property
    def embedding(self) -> np.ndarray:
        if self._embedding is None:
            self._embedding = _encode_text(self.text)
        return self._embedding

    def current_weight(self, decay_rate: float = MEMORY_DECAY_RATE) -> float:
        """
        How relevant this memory is right now.
        
        Memories fade over time. This calculates how important this entry still is
        based on how long ago it was mentioned. The formula uses exponential decay:
        if something was important but mentioned hours ago, it becomes less important.
        """
        hours = (datetime.now() - self.last_seen).total_seconds() / 3600
        return self.base_importance * np.exp(-decay_rate * hours)

    def touch(self):
        """Resets the access timer to the current time."""
        self.last_seen = datetime.now()


class ConversationMemory:
    """
    Manages conversation history with smart forgetting.
    
    This class holds all the messages in your current conversation. It automatically
    removes old messages that have become "faded" (less important), keeping your
    context focused on recent and relevant discussion while staying within token limits.
    
    Args:
        decay_rate: How quickly memories fade (higher = forget faster)
        max_tokens: Maximum space allocated for memory in the AI's context
    """

    def __init__(self, decay_rate: float = MEMORY_DECAY_RATE, max_tokens: int = MEMORY_TOKEN_BUDGET):
        self.entries: List[MemoryEntry] = []
        self.decay_rate = decay_rate
        self.max_tokens = max_tokens

    def add(self, text: str, importance: float = 1.0, role: str = "user"):
        """
        Adds a turn to memory. If semantically similar text exists, resets its timer.
        Uses embedding-based cosine similarity for semantic deduplication.
        """
        word_count = len(text.split())

        # For longer texts, use semantic similarity
        if word_count >= SEMANTIC_DEDUP_MIN_WORDS:
            try:
                new_embedding = _encode_text(text)
                for existing in self.entries:
                    existing_embedding = existing.embedding
                    similarity = _cosine_similarity(existing_embedding, new_embedding)
                    if similarity > SEMANTIC_DEDUP_THRESHOLD:
                        logger.debug(f"Semantic deduplicating {role} entry (Similarity: {similarity:.2f}).")
                        existing.touch()
                        existing.base_importance = max(existing.base_importance, importance)
                        return
            except Exception as e:
                logger.warning(f"Semantic deduplication failed, falling back to lexical: {e}")

        # Fallback to lexical Jaccard for short texts or errors
        for existing in self.entries:
            overlap = self._text_overlap(existing.text, text)
            if overlap > 0.7:
                logger.debug(f"Lexical deduplicating {role} entry (Overlap: {overlap:.2f}).")
                existing.touch()
                existing.base_importance = max(existing.base_importance, importance)
                return

        logger.debug(f"Adding new {role} entry. Total active: {len(self.entries) + 1}")
        self.entries.append(MemoryEntry(text, importance, role))

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """Calculates Jaccard similarity between two strings (fallback for short texts)."""
        set_a = set(re.findall(r'\w+', a.lower()))
        set_b = set(re.findall(r'\w+', b.lower()))
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def get_active_context(self) -> str:
        """
        Retrieves top relevant memories that haven't 'faded' (weight > threshold),
        and formats them in chronological order.
        """
        t_start = time.time()
        indexed_entries = list(enumerate(self.entries))

        active = [
            (idx, e) for idx, e in indexed_entries 
            if e.current_weight(self.decay_rate) > MEMORY_WEIGHT_THRESHOLD
        ]

        active.sort(key=lambda x: x[1].current_weight(self.decay_rate), reverse=True)

        selected_indexed = []
        total_tokens = 0
        for idx, entry in active:
            entry_text = f"[{entry.role}]: {entry.text}\n"
            entry_tokens = len(tokenizer.encode(entry_text))
            if total_tokens + entry_tokens < self.max_tokens:
                selected_indexed.append((idx, entry_text))
                total_tokens += entry_tokens
            else:
                break

        selected_indexed.sort(key=lambda x: x[0])
        context_parts = [text for idx, text in selected_indexed]

        t_ms = (time.time() - t_start) * 1000
        logger.info(f"Extracted context: {len(context_parts)}/{len(self.entries)} entries ({total_tokens} tokens) in {t_ms:.1f}ms")
        return "".join(context_parts)