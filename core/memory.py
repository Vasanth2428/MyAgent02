"""
================================================================================
RAG CONTEXT ENGINE - MEMORY MODULE
================================================================================
Short-term conversational memory with temporal decay and semantic deduplication.
"""

import re
import time
import logging
import numpy as np
import tiktoken
from datetime import datetime
from typing import List

from core.config import (
    TOKENIZER_ENCODING, MEMORY_DECAY_RATE, MEMORY_TOKEN_BUDGET,
    MEMORY_WEIGHT_THRESHOLD
)

logger = logging.getLogger("RAG.Memory")
tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)


class MemoryEntry:
    """
    Represents a single turn in a conversation.

    Attributes:
        text: The content of the conversation turn.
        base_importance: Initial importance score (default: 1.0).
        role: The speaker's role ('user' or 'assistant').
        last_seen: Timestamp of when this entry was last accessed.
    """

    def __init__(self, text: str, importance: float = 1.0, role: str = "user"):
        self.text = text
        self.base_importance = importance
        self.role = role
        self.last_seen = datetime.now()

    def current_weight(self, decay_rate: float = MEMORY_DECAY_RATE) -> float:
        """
        Calculates the current relevance based on time elapsed.
        Formula: W = I * e^(-R * H) where H is hours since last seen.
        """
        hours = (datetime.now() - self.last_seen).total_seconds() / 3600
        return self.base_importance * np.exp(-decay_rate * hours)

    def touch(self):
        """Resets the access timer to the current time."""
        self.last_seen = datetime.now()


class ConversationMemory:
    """
    Manages short-term context using temporal decay and semantic deduplication.

    Args:
        decay_rate: Speed at which memories fade (default from config).
        max_tokens: Maximum token budget for memory context.
    """

    def __init__(self, decay_rate: float = MEMORY_DECAY_RATE, max_tokens: int = MEMORY_TOKEN_BUDGET):
        self.entries: List[MemoryEntry] = []
        self.decay_rate = decay_rate
        self.max_tokens = max_tokens

    def add(self, text: str, importance: float = 1.0, role: str = "user"):
        """
        Adds a turn to memory. If semantically similar text exists, resets its timer.
        """
        for existing in self.entries:
            overlap = self._text_overlap(existing.text, text)
            if overlap > 0.7:
                logger.debug(f"Deduplicating {role} entry (Overlap: {overlap:.2f}).")
                existing.touch()
                existing.base_importance = max(existing.base_importance, importance)
                return
        logger.debug(f"Adding new {role} entry. Total active: {len(self.entries) + 1}")
        self.entries.append(MemoryEntry(text, importance, role))

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """Calculates Jaccard similarity between two strings."""
        set_a = set(re.findall(r'\w+', a.lower()))
        set_b = set(re.findall(r'\w+', b.lower()))
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def get_active_context(self) -> str:
        """
        Retrieves top relevant memories that haven't 'faded' (weight > threshold).
        Returns a formatted string for LLM context.
        """
        t_start = time.time()
        active = [e for e in self.entries if e.current_weight(self.decay_rate) > MEMORY_WEIGHT_THRESHOLD]
        active.sort(key=lambda x: x.current_weight(self.decay_rate), reverse=True)

        context_parts = []
        total_tokens = 0
        for entry in active:
            entry_text = f"[{entry.role}]: {entry.text}\n"
            entry_tokens = len(tokenizer.encode(entry_text))
            if total_tokens + entry_tokens < self.max_tokens:
                context_parts.append(entry_text)
                total_tokens += entry_tokens
            else:
                break

        t_ms = (time.time() - t_start) * 1000
        logger.info(f"Extracted context: {len(context_parts)}/{len(self.entries)} entries ({total_tokens} tokens) in {t_ms:.1f}ms")
        return "".join(context_parts)
