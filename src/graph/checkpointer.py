# Checkpointing for multi-agent workflow using LangGraph native checkpointer.
import os
import logging
import re
import sqlite3
import contextlib
from typing import AsyncIterator

logger = logging.getLogger("MultiAgent.Checkpointer")


def validate_db_path(path: str) -> str:
    """Validate database path to prevent path traversal attacks."""
    safe_path = os.path.basename(path)
    if not safe_path.endswith('.db'):
        safe_path = safe_path + '.db' if safe_path else 'checkpoints.db'
    if not re.match(r'^[\w\-]+\.db$', safe_path):
        safe_path = 'checkpoints.db'
    return safe_path


def setup_checkpointer():
    """Setup SqliteSaver for true persistent memory using LangGraph's native checkpointer."""
    db_path = os.getenv("CHECKPOINTER_DB_PATH", "checkpoints.db")
    db_path = validate_db_path(db_path)
    
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        
        safe_dir = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(safe_dir, exist_ok=True)
        full_path = os.path.join(safe_dir, db_path)
        
        conn = sqlite3.connect(full_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        logger.info(f"Using LangGraph SqliteSaver with database: {full_path}")
        return saver
    except ImportError as e:
        logger.error(f"FATAL: SqliteSaver not available: {e}. Persistent memory cannot be initialized.")
        raise RuntimeError("True persistent memory requires langgraph-checkpoint-sqlite. Please install it.") from e


@contextlib.asynccontextmanager
async def setup_async_checkpointer():
    """Setup AsyncSqliteSaver for async contexts."""
    db_path = os.getenv("CHECKPOINTER_DB_PATH", "checkpoints.db")
    db_path = validate_db_path(db_path)
    
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        
        safe_dir = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(safe_dir, exist_ok=True)
        full_path = os.path.join(safe_dir, db_path)
        
        async with AsyncSqliteSaver.from_conn_string(full_path) as saver:
            logger.info(f"Using LangGraph AsyncSqliteSaver with database: {full_path}")
            yield saver
    except ImportError as e:
        logger.error(f"FATAL: AsyncSqliteSaver not available: {e}. Persistent memory cannot be initialized.")
        raise RuntimeError("True persistent memory requires langgraph-checkpoint-sqlite. Please install it.") from e