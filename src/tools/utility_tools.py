# Utility tools for calculator, datetime, and summarization.
import logging
from datetime import datetime

logger = logging.getLogger("MultiAgent.UtilityTools")


def get_current_datetime() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def evaluate_math(expression: str) -> str:
    """Safely evaluate a mathematical expression using simpleeval."""
    try:
        from simpleeval import simple_eval, NumberTooHigh
        result = simple_eval(expression.strip())
        return str(result)
    except Exception as e:
        logger.warning(f"Math evaluation failed for '{expression}': {e}")
        return f"Error: {e}"


def summarize_text(text: str, max_tokens: int = 200) -> str:
    """Summarize text to a max token limit."""
    import tiktoken
    from src.core.config import TOKENIZER_ENCODING
    
    tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens]) + "..."
