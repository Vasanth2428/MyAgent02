"""
================================================================================
RAG CONTEXT ENGINE - PERSISTENCE MODULE
================================================================================
SQLite-backed storage for conversation history across server restarts.
"""

import sqlite3
import time
import logging
import random
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

    def execute_with_retry(self, func, *args, **kwargs):
        """
        Executes a database operations block with auto-retries on locking.
        """
        max_retries = 5
        base_delay = 0.05
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    return func(conn, *args, **kwargs)
            except sqlite3.OperationalError as e:
                err_msg = str(e).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    if attempt == max_retries - 1:
                        logger.error(f"SQLite operation failed after {max_retries} attempts: {e}")
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.02)
                    logger.warning(f"Database locked or busy ({e}). Retrying in {delay:.3f}s...")
                    time.sleep(delay)
                else:
                    raise

    def _init_db(self):
        """Creates the memory table and indexes if they don't exist."""
        def _init(conn):
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    session_id TEXT, 
                    role TEXT, 
                    text TEXT, 
                    importance REAL, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Check if telemetry column exists, if not add it
            try:
                conn.execute("SELECT telemetry FROM memory LIMIT 1")
            except sqlite3.OperationalError:
                try:
                    conn.execute("ALTER TABLE memory ADD COLUMN telemetry TEXT")
                    logger.info("Migrated SQLite database: added telemetry column to memory table.")
                except Exception as e:
                    logger.error(f"Failed to add telemetry column: {e}")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_session_timestamp 
                ON memory (session_id, timestamp)
            """)
            conn.commit()
        self.execute_with_retry(_init)

    def add_entry(self, session_id: str, text: str, role: str, importance: float = 1.0, telemetry: str = None):
        """Persists a single conversation turn to the database."""
        t_start = time.time()
        def _insert(conn):
            conn.execute(
                "INSERT INTO memory (session_id, role, text, importance, telemetry) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, text, importance, telemetry)
            )
            conn.commit()
        self.execute_with_retry(_insert)
        logger.debug(f"Inserted entry for session {session_id} in {(time.time() - t_start)*1000:.1f}ms")

    def get_history(self, session_id: str, limit: int = HISTORY_LIMIT) -> List[Dict]:
        """Retrieves history for a session from the database."""
        t_start = time.time()
        def _fetch(conn):
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT role, text, importance, telemetry, timestamp FROM memory WHERE session_id = ? ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                (session_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()][::-1]
        res = self.execute_with_retry(_fetch)
        logger.info(f"Restored {len(res)} entries for session {session_id} in {(time.time() - t_start)*1000:.1f}ms")
        return res
