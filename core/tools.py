import ast
import operator
import logging
from datetime import datetime

logger = logging.getLogger("RAG.Tools")

class SecureEvaluator:
    """
    Evaluates mathematical expressions safely by parsing the AST and restricting
    operators and constants. Prevents use of eval() and associated security concerns.
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
        if not expression or not expression.strip():
            return "Error: Empty expression received."
            
        try:
            # Parse the string into an AST
            tree = ast.parse(expression.strip(), mode='eval')
            result = self._eval_node(tree.body)
            # Format large numbers cleanly
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return str(result)
        except Exception as e:
            logger.error(f"Calculator evaluation failed for '{expression}': {e}")
            return f"Error: Failed to evaluate mathematical expression '{expression}': {type(e).__name__}: {e}"

    def _eval_node(self, node):
        if isinstance(node, ast.Num):  # Python < 3.8 support
            return node.n
        elif isinstance(node, ast.Constant):  # Python >= 3.8
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError(f"Unsupported constant type: {type(node.value).__name__}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.operators:
                left = self._eval_node(node.left)
                right = self._eval_node(node.right)
                # Protect against excessive memory/CPU consumption on giant exponentiation
                if op_type == ast.Pow and (left > 10000 or right > 100):
                    raise ValueError("Calculation limit exceeded: exponents are capped to prevent CPU lockup.")
                return self.operators[op_type](left, right)
            raise TypeError(f"Unsupported binary operator: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in self.operators:
                operand = self._eval_node(node.operand)
                return self.operators[op_type](operand)
            raise TypeError(f"Unsupported unary operator: {op_type.__name__}")
        else:
            raise TypeError(f"Unsupported expression node: {type(node).__name__}")


def get_current_time() -> str:
    """
    Returns the current local date and time in format YYYY-MM-DD HH:MM:SS.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def evaluate_math(expression: str) -> str:
    """
    Interface function to securely evaluate mathematical expressions.
    """
    evaluator = SecureEvaluator()
    return evaluator.evaluate(expression)
