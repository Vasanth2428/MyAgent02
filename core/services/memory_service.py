import logging
from typing import Dict
from core.memory import ConversationMemory
from core.persistence import PersistentMemoryStore

logger = logging.getLogger("RAG.Services.Memory")

class MemoryService:
    """
    Handles memory storage, retrieval, synchronization, and pruning operations.
    """
    def __init__(self, persistent_store: PersistentMemoryStore):
        self.persistent_store = persistent_store
        self.memories: Dict[str, ConversationMemory] = {}

    def get_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self.memories:
            logger.info(f"Restoring context for session: {session_id}")
            memory = ConversationMemory()
            for entry in self.persistent_store.get_history(session_id):
                memory.add(entry["text"], importance=entry["importance"], role=entry["role"])
            self.memories[session_id] = memory
        return self.memories[session_id]

    def save_memory(self, session_id: str, text: str, role: str, importance: float = 1.0, telemetry: dict = None):
        import json
        self.get_memory(session_id).add(text, importance, role)
        telemetry_json = json.dumps(telemetry) if telemetry else None
        self.persistent_store.add_entry(session_id, text, role, importance, telemetry=telemetry_json)
