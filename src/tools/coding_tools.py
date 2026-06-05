# Coding tools for agentic file manipulation and subprocess execution under strict security policies.
import os
import subprocess
import logging
import shlex
import time
import psutil
import re
import tempfile
from typing import List, Dict

logger = logging.getLogger("MultiAgent.CodingTools")

# Resolve workspace root to './workspace' folder inside the project folder
PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.realpath(os.path.join(PROJECT_ROOT, "workspace"))

# Policy: Create the workspace directory if missing
if not os.path.exists(WORKSPACE_ROOT):
    try:
        os.makedirs(WORKSPACE_ROOT, exist_ok=True)
        logger.info(f"Created workspace root directory at '{WORKSPACE_ROOT}'")
    except Exception as e:
        logger.error(f"Failed to create workspace root directory: {e}")

# Policy: Allowed extensions
ALLOWED_EXTENSIONS = {
    ".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".json", ".md", ".txt", ".py"
}

# Policy: Forbidden path fragments
FORBIDDEN_PATH_FRAGMENTS = {
    "/", "/etc", "/root", "/usr", "/var", "../"
}

# Policy: Command allowlist
ALLOWED_COMMANDS = [
    "python -m py_compile",
    "python -m http.server",
    "pytest",
    "npm run lint",
    "npm run test",
    "npm run build",
    "git diff"
]

def sanitize_file_content_for_llm(content: str) -> str:
    """Sanitize read file content to prevent prompt injections."""
    dangerous_phrases = [
        "system note", "system message", "system prompt", "ignore previous instructions",
        "ignore the instructions", "new instructions", "override rules", "jailbreak",
        "developer mode", "do not follow", "you must print", "reveal prompt",
        "secret key", "api key", "password", "token"
    ]
    sanitized = content
    for phrase in dangerous_phrases:
        sanitized = re.sub(re.escape(phrase), "[REDACTED_SECURE]", sanitized, flags=re.IGNORECASE)
    return sanitized


def _is_safe_path(filepath: str) -> bool:
    """Check if filepath is safe (within workspace root and doesn't contain forbidden paths)."""
    if not filepath or not filepath.strip():
        return False
        
    # Normalize to forward slashes for checking
    norm_path = filepath.replace("\\", "/")
    
    # Check for direct forbidden prefixes/traversals
    forbidden_prefixes = ("/", "/etc", "/root", "/usr", "/var", "..")
    if any(norm_path.startswith(p) for p in forbidden_prefixes) or "../" in norm_path:
        return False
        
    real_workspace = os.path.realpath(WORKSPACE_ROOT)
    
    # If the file exists relative to PROJECT_ROOT and is outside WORKSPACE_ROOT, block it
    proj_path = os.path.realpath(os.path.join(PROJECT_ROOT, filepath))
    if proj_path != os.path.realpath(PROJECT_ROOT) and os.path.exists(proj_path):
        if os.name == 'nt':
            if not proj_path.lower().startswith(real_workspace.lower()):
                return False
        else:
            if not proj_path.startswith(real_workspace):
                return False

    abs_path = os.path.realpath(os.path.join(WORKSPACE_ROOT, filepath))
    
    # 1. Traversal check (must stay strictly inside ./workspace)
    if os.name == 'nt':
        if not abs_path.lower().startswith(real_workspace.lower()):
            return False
    else:
        if not abs_path.startswith(real_workspace):
            return False
            
    # Restrict package.json modifications completely
    basename = os.path.basename(abs_path).lower()
    if basename == "package.json":
        return False
        
    # 2. Forbidden path fragments check in the relative path
    rel_path = os.path.relpath(abs_path, WORKSPACE_ROOT)
    normalized_rel = rel_path.replace("\\", "/").lower()
    
    if normalized_rel == ".." or normalized_rel.startswith("../") or ".." in normalized_rel:
        return False
        
    # Check against system root level folders
    abs_path_norm = abs_path.replace("\\", "/").lower()
    system_forbidden = ["/etc", "/root", "/usr", "/var"]
    for sys_p in system_forbidden:
        if abs_path_norm == sys_p or abs_path_norm.startswith(sys_p + "/"):
            return False
            
    # Also block drive letter root levels on Windows (e.g. C:/etc, C:/usr)
    if re.match(r"^[a-zA-Z]:/(etc|root|usr|var)(/|$)", abs_path_norm):
        return False
        
    return True


def _has_allowed_extension(filepath: str) -> bool:
    """Check if the file has an approved extension."""
    _, ext = os.path.splitext(filepath)
    return ext.lower() in ALLOWED_EXTENSIONS


def _get_absolute_path(filepath: str) -> str:
    """Get absolute path to a file in the `./workspace` folder."""
    return os.path.realpath(os.path.join(WORKSPACE_ROOT, filepath))


def view_code_file(filepath: str, start_line: int = 1, end_line: int = 100) -> str:
    """
    Safely view a range of lines inside a file in `./workspace`.
    
    Args:
        filepath: Path relative to './workspace'.
        start_line: 1-indexed start line.
        end_line: 1-indexed end line (inclusive).
    """
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety or path policies."
        
    if not _has_allowed_extension(filepath):
        return f"Error: Access denied. File extension not allowed. Approved extensions: {', '.join(ALLOWED_EXTENSIONS)}"
    
    abs_path = _get_absolute_path(filepath)
    if not os.path.isfile(abs_path):
        return f"Error: File '{filepath}' does not exist."
    
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)
        
        output = []
        for i in range(start_idx, end_idx):
            output.append(f"{i + 1}: {lines[i]}")
            
        header = f"--- Viewing {filepath} (Lines {start_line}-{end_line} of {total_lines}) ---\n"
        return sanitize_file_content_for_llm(header + "".join(output))
    except Exception as e:
        return f"Error reading file '{filepath}': {e}"


def read_files(filepath: str, start_line: int = 1, end_line: int = 100) -> str:
    """
    Safely view a range of lines inside a file in `./workspace`.
    
    Args:
        filepath: Path relative to './workspace'.
        start_line: 1-indexed start line.
        end_line: 1-indexed end line (inclusive).
    """
    return view_code_file(filepath, start_line, end_line)


def search_code(query: str, directory: str = ".") -> str:
    """
    Find occurrences of a text query inside source code files in `./workspace`.
    
    Args:
        query: Text to search for.
        directory: Target subdirectory relative to './workspace'.
    """
    if not _is_safe_path(directory):
        return "Error: Access denied. Target directory violates safety policies."
    
    abs_dir = _get_absolute_path(directory)
    if not os.path.isdir(abs_dir):
        return f"Error: Directory '{directory}' does not exist."
    
    results = []
    max_matches = 50
    matches_count = 0
    
    try:
        for root, dirs, files in os.walk(abs_dir):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules", "checkpoints"}]
            
            for file in files:
                if not _has_allowed_extension(file):
                    continue
                
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, WORKSPACE_ROOT)
                
                # Double check path safety
                if not _is_safe_path(rel_path):
                    continue
                
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        if query in line:
                            results.append(f"{rel_path}:{line_num}: {line.strip()}")
                            matches_count += 1
                            if matches_count >= max_matches:
                                break
                if matches_count >= max_matches:
                    break
            if matches_count >= max_matches:
                break
                
        if not results:
            return f"No matches found for '{query}'."
            
        header = f"--- Search results for '{query}' (Max {max_matches} matches) ---\n"
        return sanitize_file_content_for_llm(header + "\n".join(results))
    except Exception as e:
        return f"Error searching code: {e}"


def create_files(filepath: str, content: str) -> str:
    """
    Create a new file with the specified content inside `./workspace`.
    Fails if the file already exists.
    
    Args:
        filepath: File path relative to './workspace'.
        content: The content to write to the file.
    """
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety or path policies."
        
    if not _has_allowed_extension(filepath):
        return f"Error: Access denied. File extension not allowed. Approved extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        
    abs_path = _get_absolute_path(filepath)
    if os.path.exists(abs_path):
        return f"Error: File '{filepath}' already exists. Use modify_files to make changes."
        
    dir_name = os.path.dirname(abs_path)
    if not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            return f"Error creating parent directory: {e}"
            
    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: Created new file '{filepath}'."
    except Exception as e:
        return f"Error creating file '{filepath}': {e}"


def edit_code_file(filepath: str, target: str, replacement: str) -> str:
    """
    Search and replace a specific block of text in a file inside `./workspace`.
    Creates file if missing (if create_if_missing policy is true).
    
    Args:
        filepath: File to modify.
        target: Exact target block of code to search for.
        replacement: Code block to replace the target code.
    """
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety or path policies."
        
    if not _has_allowed_extension(filepath):
        return f"Error: Access denied. File extension not allowed. Approved extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        
    abs_path = _get_absolute_path(filepath)
    
    # Create file if missing
    if not os.path.isfile(abs_path):
        dir_name = os.path.dirname(abs_path)
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                return f"Error creating parent directory: {e}"
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(replacement)
            return f"Success: Created new file '{filepath}'."
        except Exception as e:
            return f"Error creating file '{filepath}': {e}"
            
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        occurrences = content.count(target)
        if occurrences == 0:
            return (
                f"Error: Target text not found in '{filepath}'. "
                "Make sure spacing, newlines, and indentation match exactly."
            )
        if occurrences > 1:
            return (
                f"Error: Target text matches {occurrences} times in '{filepath}'. "
                "Provide a larger context block to make the target query unique."
            )
            
        new_content = content.replace(target, replacement, 1)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return f"Success: Modified '{filepath}' successfully."
    except Exception as e:
        return f"Error editing file '{filepath}': {e}"


def modify_files(filepath: str, target_code: str, replacement_code: str) -> str:
    """
    Search and replace a specific block of text in a file inside `./workspace`.
    Fails if the file does not exist.
    
    Args:
        filepath: File path relative to './workspace'.
        target_code: Exact target block of code to search for.
        replacement_code: Code block to replace the target code.
    """
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety or path policies."
        
    if not _has_allowed_extension(filepath):
        return f"Error: Access denied. File extension not allowed. Approved extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        
    abs_path = _get_absolute_path(filepath)
    if not os.path.isfile(abs_path):
        return f"Error: File '{filepath}' does not exist. Use create_files to create it first."
        
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        occurrences = content.count(target_code)
        if occurrences == 0:
            return (
                f"Error: Target text not found in '{filepath}'. "
                "Make sure spacing, newlines, and indentation match exactly."
            )
        if occurrences > 1:
            return (
                f"Error: Target text matches {occurrences} times in '{filepath}'. "
                "Provide a larger context block to make the target query unique."
            )
            
        new_content = content.replace(target_code, replacement_code, 1)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return f"Success: Modified '{filepath}' successfully."
    except Exception as e:
        return f"Error editing file '{filepath}': {e}"




def delete_file(filepath: str) -> str:
    """
    Delete a file in the workspace.
    
    Args:
        filepath: File path relative to './workspace'.
    """
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety or path policies."
        
    abs_path = _get_absolute_path(filepath)
    if not os.path.isfile(abs_path):
        return f"Error: File '{filepath}' does not exist."
        
    try:
        os.remove(abs_path)
        return f"Success: Deleted file '{filepath}'."
    except Exception as e:
        return f"Error deleting file '{filepath}': {e}"


def list_files(directory: str = ".") -> str:
    """
    List files and subdirectories inside the `./workspace` folder.
    
    Args:
        directory: Directory path relative to './workspace'.
    """
    if not _is_safe_path(directory):
        return "Error: Access denied. Target directory violates safety policies."
        
    abs_dir = _get_absolute_path(directory)
    if not os.path.isdir(abs_dir):
        return f"Error: Directory '{directory}' does not exist."
        
    try:
        items = os.listdir(abs_dir)
        output = []
        for item in items:
            full_path = os.path.join(abs_dir, item)
            rel_path = os.path.relpath(full_path, WORKSPACE_ROOT)
            
            # Double check path safety for the child item
            if not _is_safe_path(rel_path):
                continue
                
            if os.path.isdir(full_path):
                output.append(f"[DIR]  {item}")
            else:
                sz = os.path.getsize(full_path)
                output.append(f"[FILE] {item} ({sz} bytes)")
                
        if not output:
            return f"Directory '{directory}' is empty."
            
        header = f"--- Listing contents of {directory if directory != '.' else 'workspace root'} ---\n"
        return header + "\n".join(output)
    except Exception as e:
        return f"Error listing directory '{directory}': {e}"


def _is_safe_command(cmd_args: List[str]) -> bool:
    """Check if the command and arguments are safely within strict allowlist limits."""
    if not cmd_args:
        return False
        
    executable = cmd_args[0]
    
    # 1. First argument must be on the allowlist
    if executable not in ["python", "pytest", "npm", "git"]:
        return False
        
    # 2. Strict validation per executable
    if executable == "python":
        # Allowed formats:
        # python -m py_compile <filepath>
        # python -m http.server <port> (optional port)
        if len(cmd_args) >= 3 and cmd_args[1] == "-m":
            module = cmd_args[2]
            if module == "py_compile":
                if len(cmd_args) == 4:
                    return _is_safe_path(cmd_args[3])
                return False
            elif module == "http.server":
                if len(cmd_args) == 3:
                    return True
                elif len(cmd_args) == 4:
                    return cmd_args[3].isdigit()
                return False
        return False
        
    if executable == "pytest":
        # Allowed arguments for pytest:
        # Only options starting with '-' (e.g. -v, -s, -q, --version, --tb=short)
        # Or specific test files: 'tests/unit/test_coding_tools.py', 'tests/unit/test_coding_worker.py', 'tests/unit/test_workflow.py'
        allowed_pytest_args = {
            "tests/unit/test_coding_tools.py",
            "tests/unit/test_coding_worker.py",
            "tests/unit/test_workflow.py",
            "-v", "-s", "-q", "--version", "--tb=short", "--tb=line"
        }
        for arg in cmd_args[1:]:
            if arg not in allowed_pytest_args:
                if arg.startswith("tests/unit/test_") and arg.endswith(".py"):
                    continue
                return False
        return True
        
    if executable == "npm":
        # Only allowed formats:
        # npm run lint
        # npm run test
        # npm run build
        if len(cmd_args) == 3 and cmd_args[1] == "run" and cmd_args[2] in ["lint", "test", "build"]:
            return True
        return False
        
    if executable == "git":
        # Only allowed formats:
        # git diff
        if len(cmd_args) == 2 and cmd_args[1] == "diff":
            return True
        return False
        
    return False


def execute_command(command: str) -> str:
    """
    Execute a command in the `./workspace` folder securely.
    Only approved command structures are allowed.
    Enforces strict execution limits (CPU, memory, timeout) and runs without shell.
    """
    cmd_clean = command.strip()
    if not cmd_clean:
        return "Error: Empty command provided."
        
    try:
        cmd_args = shlex.split(cmd_clean)
    except Exception as e:
        return f"Error parsing command line: {e}"
        
    if not cmd_args:
        return "Error: Empty command provided."
        
    # Check strict safety allowlist
    if not _is_safe_command(cmd_args):
        return f"Error: Command '{command}' blocked by safety policy. Command is not in allowlist."
        
    # Executing the command
    print(f"\n[SECURE RUN] Executing command: {cmd_args} in '{WORKSPACE_ROOT}'")
    
    try:
        # Create temp files to capture stdout/stderr safely without pipe deadlock issues
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, \
             tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
             
            proc = subprocess.Popen(
                cmd_args,
                cwd=WORKSPACE_ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True
            )
            
            try:
                p = psutil.Process(proc.pid)
            except psutil.NoSuchProcess:
                p = None
                
            timeout = 15.0
            start_time = time.time()
            
            while proc.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    try:
                        if p:
                            for child in p.children(recursive=True):
                                try:
                                    child.kill()
                                except Exception:
                                    pass
                        proc.kill()
                    except Exception:
                        pass
                    return f"Error: Command execution timed out after {timeout} seconds."
                    
                if p:
                    # Check memory usage (100MB limit)
                    try:
                        mem_info = p.memory_info()
                        mem_mb = mem_info.rss / (1024 * 1024)
                        
                        for child in p.children(recursive=True):
                            try:
                                mem_mb += child.memory_info().rss / (1024 * 1024)
                            except Exception:
                                pass
                                
                        if mem_mb > 100.0:
                            try:
                                for child in p.children(recursive=True):
                                    try:
                                        child.kill()
                                    except Exception:
                                        pass
                                proc.kill()
                            except Exception:
                                pass
                            return f"Error: Command execution exceeded memory limit of 100MB (used {mem_mb:.2f}MB)."
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
                    # Check CPU time limit (5.0 seconds limit)
                    try:
                        cpu_times = p.cpu_times()
                        cpu_time_used = cpu_times.user + cpu_times.system
                        for child in p.children(recursive=True):
                            try:
                                c_times = child.cpu_times()
                                cpu_time_used += c_times.user + c_times.system
                            except Exception:
                                pass
                        if cpu_time_used > 5.0:
                            try:
                                for child in p.children(recursive=True):
                                    try:
                                        child.kill()
                                    except Exception:
                                        pass
                                proc.kill()
                            except Exception:
                                pass
                            return f"Error: Command execution exceeded CPU time limit of 5 seconds (used {cpu_time_used:.2f}s CPU time)."
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
                time.sleep(0.1)
                
            # Read captured outputs from tempfiles
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout_data = stdout_file.read()
            stderr_data = stderr_file.read()
            
            output = []
            if stdout_data:
                output.append("--- stdout ---")
                output.append(stdout_data)
            if stderr_data:
                output.append("--- stderr ---")
                output.append(stderr_data)
                
            status = f"\n[Command exited with status {proc.returncode}]"
            return "\n".join(output) + status
            
    except Exception as e:
        return f"Error executing command: {e}"


def run_safe_commands(command: str) -> str:
    """
    Execute a shell command in the `./workspace` folder.
    Only approved commands (pytest, npm run build, npm run test, git diff, etc.) are allowed.
    """
    return execute_command(command)
