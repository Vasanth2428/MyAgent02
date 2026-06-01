"""
Memory Service - Bridge Between Short-term Memory and Long-term Storage

This service connects the conversation memory (what we remember during chat) with
the persistent storage (what gets saved to disk). It ensures that when you add
something to memory, it also gets saved to the database.
"""

import logging
import json
from typing import Dict
from core.memory import ConversationMemory
from core.persistence import PersistentMemoryStore

logger = logging.getLogger("RAG.Services.Memory")


class MemoryService:
    """
    Connects in-memory conversation history with database storage.
    
    When you chat with the AI, this service makes sure your messages are remembered
    during the conversation AND saved permanently to the database. When you return
    to the same conversation later, it restores your history from the database.
    """

    def __init__(self, persistent_store: PersistentMemoryStore):
        self.persistent_store = persistent_store
        self.memories: Dict[str, ConversationMemory] = {}  # Active memories by session ID

    def get_memory(self, session_id: str) -> ConversationMemory:
        """
        Get or create the conversation memory for a session.
        
        If we've seen this conversation before, we restore the history from the
        database. Otherwise, we start fresh with an empty memory.
        """
        if session_id not in self.memories:
            logger.info(f"Restoring context for session: {session_id}")
            memory = ConversationMemory()
            for entry in self.persistent_store.get_history(session_id):
                memory.add(entry["text"], importance=entry["importance"], role=entry["role"])
            self.memories[session_id] = memory
        return self.memories[session_id]

    def save_memory(self, session_id: str, text: str, role: str, importance: float = 1.0, telemetry: dict = None):
        """
        Add a message to memory and save it to the database.
        
        Args:
            session_id: Which conversation this belongs to
            text: The message content
            role: Who said it ('user' or 'assistant')
            importance: How relevant this message is (affects forgetting)
            telemetry: Extra data about the response for debugging
        """
        self.get_memory(session_id).add(text, importance, role)
        telemetry_json = json.dumps(telemetry) if telemetry else None
        self.persistent_store.add_entry(session_id, text, role, importance, telemetry=telemetry_json)
