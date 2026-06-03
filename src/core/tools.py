"""
Agent Tools - Helper Functions the Agent Can Use

This module provides simple, safe tools for the agent to use:
- Calculator: Do math without using dangerous eval()
- Time: Get the current date/time

All tools are designed to work safely even when given untrusted input.
"""

import ast
import operator
import logging
from datetime import datetime

logger = logging.getLogger("RAG.Tools")


class SecureEvaluator:
    """
    Safely calculates mathematical expressions.
    
    Instead of using Python's dangerous eval() function, this class parses
    the expression into a tree and evaluates only the safe math operations
    (+, -, *, /, etc.). This prevents people from running dangerous code.
    """
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: lambda x: x
    }

    def evaluate(self, expression: str) -> str:
        """Calculate a math expression and return the result as text."""
        if not expression or not expression.strip():
            return "Error: Empty expression received."
            
        try:
            tree = ast.parse(expression.strip(), mode='eval')
            result = self._eval_node(tree.body)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return str(result)
        except Exception as e:
            logger.error(f"Calculator failed for '{expression}': {e}")
            return f"Error: Could not calculate '{expression}': {type(e).__name__}"

    def _eval_node(self, node):
        """Recursively evaluate nodes in the expression tree."""
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError(f"Unsupported type: {type(node.value).__name__}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.operators:
                left = self._eval_node(node.left)
                right = self._eval_node(node.right)
                if op_type == ast.Pow and (left > 10000 or right > 100):
                    raise ValueError("Too large: exponents are capped to prevent hangs.")
                return self.operators[op_type](left, right)
            raise TypeError(f"Unsupported operation: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in self.operators:
                return self.operators[op_type](self._eval_node(node.operand))
            raise TypeError(f"Unsupported operation: {op_type.__name__}")
        else:
            raise TypeError(f"Unsupported: {type(node).__name__}")


def get_current_time() -> str:
    """Get the current date and time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def evaluate_math(expression: str) -> str:
    """Calculate a math expression (e.g., '2 + 2' returns '4')."""
    evaluator = SecureEvaluator()
    return evaluator.evaluate(expression)
