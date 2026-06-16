# Utility tools for calculator, datetime, and summarization.
import logging
from datetime import datetime

logger = logging.getLogger("MultiAgent.UtilityTools")


def get_current_datetime() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def evaluate_math(expression: str) -> str:
    """Safely evaluate a mathematical expression using ast.literal_eval with operator support."""
    import ast
    import operator

    _SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return _SAFE_OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
        else:
            raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
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
