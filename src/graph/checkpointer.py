# Checkpointing for multi-agent workflow.
import os
import logging

logger = logging.getLogger("MultiAgent.Checkpointer")

import re

def validate_db_path(path: str) -> str:
    """Validate database path to prevent path traversal attacks."""
    safe_path = os.path.basename(path)
    if not safe_path.endswith('.db'):
        safe_path = safe_path + '.db' if safe_path else 'checkpoints.db'
    # Only allow alphanumeric, dash, underscore, and .db extension
    if not re.match(r'^[\w\-]+\.db$', safe_path):
        safe_path = 'checkpoints.db'
    return safe_path


def setup_checkpointer():
    """Setup SqliteSaver or MemorySaver based on environment."""
    use_sqlite = os.getenv("USE_SQLITE_CHECKPOINTER", "false").lower() == "true"
    db_path = os.getenv("CHECKPOINTER_DB_PATH", "checkpoints.db")
    db_path = validate_db_path(db_path)
    
    if use_sqlite:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            import sqlite3
            
            # Use safe directory for database
            safe_dir = os.path.join(os.getcwd(), 'checkpoints')
            os.makedirs(safe_dir, exist_ok=True)
            full_path = os.path.join(safe_dir, db_path)
            
            conn = sqlite3.connect(full_path, check_same_thread=False)
            saver = SqliteSaver(conn)
            logger.info(f"Using SqliteSaver with database: {full_path}")
            return saver
        except ImportError as e:
            logger.warning(f"SqliteSaver not available: {e}. Falling back to MemorySaver.")
    
    from langgraph.checkpoint.memory import MemorySaver
    logger.info("Using MemorySaver for development.")
    return MemorySaver()
