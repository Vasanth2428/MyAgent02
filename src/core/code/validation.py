import logging
from typing import Tuple

from src.tools.coding_tools import execute_command

logger = logging.getLogger("RAG.CodeValidation")

def validate_syntax(code_content: str, filename: str) -> Tuple[bool, str]:
    """
    Validates Python and JS/JSX/TS/TSX syntax.
    """
    logger.info(f"Validating syntax for: {filename}")
    
    # Python validation
    if filename.lower().endswith(".py"):
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

    # JS/JSX/TS/TSX validation
    if filename.lower().endswith((".js", ".jsx", ".ts", ".tsx")):
        import tempfile
        import os
        import subprocess

        _, ext = os.path.splitext(filename)
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w", encoding="utf-8") as tmp:
                tmp.write(code_content)
                temp_path = tmp.name

            try:
                cmd = f"npx esbuild \"{temp_path}\""
                res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                
                # Check for esbuild warnings/errors
                if res.returncode != 0:
                    raw_err = res.stderr if res.stderr else res.stdout
                    clean_msg = raw_err.replace(temp_path, filename).replace(os.path.basename(temp_path), filename)
                    logger.warning(f"esbuild validation failed for {filename}: {clean_msg}")
                    return False, f"esbuild Syntax Error:\n{clean_msg}"
                return True, "Syntax compilation successful."
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            err_msg = f"Unexpected esbuild validation error: {e}"
            logger.error(err_msg)
            return False, err_msg

    # Other files
    logger.info(f"Skipping compilation syntax check for non-compilable file: {filename}")
    return True, "Skipping compilation check for non-compilable file."


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
