"""
================================================================================
RAG CONTEXT ENGINE - PERSISTENCE MODULE
================================================================================
SQLite-backed storage for conversation history across server restarts.
"""

import sqlite3
import time
import logging
from typing import List, Dict

from core.config import DB_PATH, HISTORY_LIMIT

logger = logging.getLogger("RAG.Persistence")


class PersistentMemoryStore:
    """
    Service for SQLite-backed storage of conversation history.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Creates the memory table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    session_id TEXT, 
                    role TEXT, 
                    text TEXT, 
                    importance REAL, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_entry(self, session_id: str, text: str, role: str, importance: float = 1.0):
        """Persists a single conversation turn to the database."""
        t_start = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memory (session_id, role, text, importance) VALUES (?, ?, ?, ?)",
                (session_id, role, text, importance)
            )
        logger.debug(f"Inserted entry for session {session_id} in {(time.time() - t_start)*1000:.1f}ms")

    def get_history(self, session_id: str, limit: int = HISTORY_LIMIT) -> List[Dict]:
        """Retrieves history for a session from the database."""
        t_start = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT role, text, importance, timestamp FROM memory WHERE session_id = ? ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                (session_id, limit)
            )
            res = [dict(row) for row in cursor.fetchall()][::-1]
        logger.info(f"Restored {len(res)} entries for session {session_id} in {(time.time() - t_start)*1000:.1f}ms")
        return res
