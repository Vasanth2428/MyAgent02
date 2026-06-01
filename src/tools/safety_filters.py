# Safety filters for multi-agent system.
import re
import logging
from typing import Any, List, Dict

logger = logging.getLogger("MultiAgent.Safety")

# Maximum context size to prevent memory exhaustion
MAX_CONTEXT_TOKENS = 4000
MAX_RESULTS = 10
MAX_QUERY_LENGTH = 500


def sanitize_user_input(text: str) -> str:
    """Sanitize user input to prevent prompt injection."""
    if not text:
        return ""
    
    dangerous_patterns = [
        r"(?i)ignore (previous|above|all) (instructions|prompt)",
        r"(?i)you are (now|actually)",
        r"(?i)new (instructions|system prompt)",
        r"(?i)(system|assistant|user):",
        r"(?i)</?(?:script|style|iframe)",
    ]
    
    sanitized = text[:MAX_QUERY_LENGTH]
    for pattern in dangerous_patterns:
        if re.search(pattern, sanitized):
            logger.warning(f"Potential prompt injection detected, sanitizing input")
            sanitized = re.sub(pattern, "[REDACTED]", sanitized)
    
    return sanitized


def validate_tool_output(output: Any, max_length: int = 10000) -> str:
    """Validate and sanitize tool output."""
    if output is None:
        return ""
    
    output_str = str(output)[:max_length]
    
    # Remove any potential script injection
    dangerous_patterns = [
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
    ]
    
    for pattern in dangerous_patterns:
        output_str = re.sub(pattern, "[REMOVED]", output_str, flags=re.IGNORECASE)
    
    return output_str


def truncate_results(results: List[Dict], max_items: int = MAX_RESULTS) -> List[Dict]:
    """Truncate search results to prevent context overflow."""
    return results[:max_items]


def safe_extract(text: str, start: int = 0, end: int = None) -> str:
    """Safely extract text within bounds."""
    if not text:
        return ""
    if end is None:
        end = len(text)
    return text[start:end]
