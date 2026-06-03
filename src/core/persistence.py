"""
RAG Persistence - Saving Conversations to Disk

This module saves your conversation history to a SQLite database, so it persists
across server restarts. When you come back later, the AI can continue the
conversation where you left off.
"""

import sqlite3
import time
import logging
import random
from typing import List, Dict

from src.core.config import DB_PATH, HISTORY_LIMIT

logger = logging.getLogger("RAG.Persistence")


class PersistentMemoryStore:
    """
    Saves conversation history to a SQLite database file.
    
    Instead of keeping conversation in memory (which is lost when the server stops),
    this class writes everything to disk. Your chat history will still be available
    when you restart the application.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def execute_with_retry(self, func, *args, **kwargs):
        """
        Executes a database operations block with auto-retries on locking.
        """
        def is_db_transient(e):
            if isinstance(e, sqlite3.OperationalError):
                err_msg = str(e).lower()
                return "locked" in err_msg or "busy" in err_msg
            return False

        from src.core.retry import retry

        @retry(
            retries=5,
            backoff=0.05,
            jitter=0.02,
            is_transient_fn=is_db_transient,
            logger_name="RAG.Persistence"
        )
        def _execute():
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                return func(conn, *args, **kwargs)

        return _execute()

    def _init_db(self):
        """Creates the memory and sessions tables and indexes if they don't exist."""
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
            # Sessions metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title      TEXT DEFAULT 'New Chat',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
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

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------

    def create_session(self, session_id: str, title: str = "New Chat") -> None:
        """Creates or upserts a session row in the sessions table."""
        def _upsert(conn):
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, title) VALUES (?, ?)",
                (session_id, title)
            )
            conn.commit()
        self.execute_with_retry(_upsert)

    def list_sessions(self) -> List[Dict]:
        """Returns all sessions ordered by most recently updated, with message counts."""
        def _fetch(conn):
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.rowid) as message_count
                FROM sessions s
                LEFT JOIN memory m ON m.session_id = s.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
        return self.execute_with_retry(_fetch)

    def rename_session(self, session_id: str, title: str) -> None:
        """Updates the title of an existing session."""
        def _update(conn):
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (title, session_id)
            )
            conn.commit()
        self.execute_with_retry(_update)

    def delete_session(self, session_id: str) -> None:
        """Deletes a session and all its message history."""
        def _delete(conn):
            conn.execute("DELETE FROM memory WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        self.execute_with_retry(_delete)

    def touch_session(self, session_id: str) -> None:
        """Upserts a session and bumps its updated_at timestamp."""
        def _touch(conn):
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                (session_id,)
            )
            conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
        self.execute_with_retry(_touch)
