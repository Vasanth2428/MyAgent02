import logging
from typing import Tuple

from src.tools.coding_tools import execute_command

logger = logging.getLogger("RAG.CodeValidation")

def validate_syntax(code_content: str, filename: str) -> Tuple[bool, str]:
    """
    Compiles Python code content in-memory to ensure syntax correctness.
    """
    logger.info(f"Validating syntax for: {filename}")
    try:
        compile(code_content, filename, "exec")
        return True, "Syntax compilation successful."
    except SyntaxError as e:
        err_msg = f"Syntax Error: {e.msg} on line {e.lineno}, column {e.offset} in {filename}"
        logger.warning(err_msg)
        return False, err_msg
    except Exception as e:
        err_msg = f"Unexpected compilation error: {e}"
        logger.error(err_msg)
        return False, err_msg


def validate_tests(test_command: str) -> Tuple[bool, str]:
    """
    Runs unit tests through execute_command and checks for success.
    """
    logger.info(f"Executing test validation: {test_command}")
    res = execute_command(test_command)
    
    # Analyze output for failures/errors
    res_lower = res.lower()
    
    # Check if command failed or raised NameError/SyntaxError
    if "error" in res_lower or "failed" in res_lower or "failures" in res_lower:
        # Check if it still exited with code 0 (which is unlikely if there's failure)
        if "[command exited with status 0]" in res_lower:
            # Succeeded despite warning keyword
            return True, res
        return False, res
        
    return True, res
