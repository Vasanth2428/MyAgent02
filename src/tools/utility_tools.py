# Utility tools for calculator, datetime, and summarization.
import logging
from datetime import datetime

logger = logging.getLogger("MultiAgent.UtilityTools")


def get_current_datetime() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def evaluate_math(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    from src.core.tools import evaluate_math as safe_evaluate
    return safe_evaluate(expression)


def summarize_text(text: str, max_tokens: int = 200) -> str:
    """Summarize text to a max token limit."""
    import tiktoken
    from src.core.config import TOKENIZER_ENCODING
    
    tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens]) + "..."
