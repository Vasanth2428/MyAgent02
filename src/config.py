# Configuration for multi-agent system.
import os
from typing import Optional


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with default."""
    return os.getenv(key, default)


SUPERVISOR_MODEL = get_env("SUPERVISOR_MODEL", "gpt-4o-mini")
REASONING_MODEL = get_env("REASONING_MODEL", "gpt-4o")
TAVILY_API_KEY = get_env("TAVILY_API_KEY")
OPENAI_API_KEY = get_env("OPENAI_API_KEY")

RECURSION_LIMIT = int(get_env("RECURSION_LIMIT", "20"))
MAX_STEPS = int(get_env("MAX_STEPS", "10"))

LANGCHAIN_TRACING_V2 = get_env("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = get_env("LANGCHAIN_API_KEY")

USE_SQLITE_CHECKPOINTER = get_env("USE_SQLITE_CHECKPOINTER", "false").lower() == "true"
CHECKPOINTER_DB_PATH = get_env("CHECKPOINTER_DB_PATH", "checkpoints.db")
