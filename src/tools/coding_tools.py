import os
import subprocess
import logging
import shlex
import time
import psutil
import re
import tempfile
from typing import List, Dict, Optional

logger = logging.getLogger("MultiAgent.CodingTools")

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.realpath(os.path.join(PROJECT_ROOT, "workspace"))

if not os.path.exists(WORKSPACE_ROOT):
    try:
        os.makedirs(WORKSPACE_ROOT, exist_ok=True)
        logger.info(f"Created workspace root directory at '{WORKSPACE_ROOT}'")
    except Exception as e:
        logger.error(f"Failed to create workspace root directory: {e}")

ALLOWED_EXTENSIONS = {".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".json", ".md", ".txt", ".py"}

FORBIDDEN_PATH_FRAGMENTS = {"/", "/etc", "/root", "/usr", "/var", "../"}

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
        
    norm_path = filepath.replace("\\", "/")
    forbidden_prefixes = ("/", "/etc", "/root", "/usr", "/var", "..")
    if any(norm_path.startswith(p) for p in forbidden_prefixes) or "../" in norm_path:
        return False
        
    real_workspace = os.path.realpath(WORKSPACE_ROOT)
    abs_path = os.path.realpath(os.path.join(WORKSPACE_ROOT, filepath))
    
    if os.name == 'nt':
        if not abs_path.lower().startswith(real_workspace.lower()):
            return False
    else:
        if not abs_path.startswith(real_workspace):
            return False
            
    basename = os.path.basename(abs_path).lower()
    if basename == "package.json":
        return False
        
    rel_path = os.path.relpath(abs_path, WORKSPACE_ROOT)
    normalized_rel = rel_path.replace("\\", "/").lower()
    
    if normalized_rel == ".." or normalized_rel.startswith("../") or ".." in normalized_rel:
        return False
        
    abs_path_norm = abs_path.replace("\\", "/").lower()
    system_forbidden = ["/etc", "/root", "/usr", "/var"]
    for sys_p in system_forbidden:
        if abs_path_norm == sys_p or abs_path_norm.startswith(sys_p + "/"):
            return False
            
    if re.match(r"^[a-zA-Z]:/(etc|root|usr|var)(/|$)", abs_path_norm):
        return False
        
    return True


def _has_allowed_extension(filepath: str) -> bool:
    """Check if the file has an approved extension."""
    _, ext = os.path.splitext(filepath)
    return ext.lower() in ALLOWED_EXTENSIONS


def _get_absolute_path(filepath: str) -> str:
    """Get absolute path to a file in the `./workspace` folder."""
    p = filepath.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if p.startswith("workspace/"):
        p = p[len("workspace/"):]
    while p.startswith("./"):
        p = p[2:]
    return os.path.realpath(os.path.join(WORKSPACE_ROOT, p))


def view_code_file(filepath: str, start_line: int = 1, end_line: int = 100) -> str:
    """Safely view a range of lines inside a file in `./workspace`."""
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
    """Safely view a range of lines inside a file in `./workspace`."""
    return view_code_file(filepath, start_line, end_line)


def search_code(query: str, directory: str = ".") -> str:
    """Find occurrences of a text query inside source code files in `./workspace`."""
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
        
        if not results:
            return f"No matches found for '{query}'."
            
        header = f"--- Search results for '{query}' (Max {max_matches} matches) ---\n"
        return sanitize_file_content_for_llm(header + "\n".join(results))
    except Exception as e:
        return f"Error searching code: {e}"


def create_files(filepath: str, content: str) -> str:
    """Create a new file with the specified content inside `./workspace`. Fails if the file already exists."""
    from src.core.code.validation import validate_syntax
    
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
            
    filename = os.path.basename(filepath)
    if filename.endswith(".py"):
        ok, msg = validate_syntax(content, filepath)
        if not ok:
            return f"Error: {msg}"
            
    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: Created new file '{filepath}'."
    except Exception as e:
        return f"Error creating file '{filepath}': {e}"


def try_fuzzy_replace(content: str, target: str, replacement: str) -> Optional[str]:
    """Attempts to replace the target block of code in the content using normalized matches."""
    target_norm = target.replace("\r\n", "\n").rstrip()
    content_norm = content.replace("\r\n", "\n")
    
    if target_norm in content_norm:
        if content_norm.count(target_norm) == 1:
            parts = content_norm.split(target_norm, 1)
            return parts[0] + replacement + parts[1]

    target_lines = [line.strip() for line in target.replace("\r\n", "\n").split("\n")]
    if not target_lines or (len(target_lines) == 1 and not target_lines[0]):
        return None
        
    content_lines = [line.replace("\r\n", "\n") for line in content.split("\n")]
    content_lines_stripped = [line.strip() for line in content_lines]
    
    target_len = len(target_lines)
    match_index = -1
    match_count = 0
    
    for i in range(len(content_lines_stripped) - target_len + 1):
        sub_window = content_lines_stripped[i : i + target_len]
        if sub_window == target_lines:
            match_count += 1
            match_index = i
            
    if match_count == 1:
        before = content_lines[:match_index]
        after = content_lines[match_index + target_len :]
        return "\n".join(before + [replacement] + after)
        
    return None


def edit_code_file(filepath: str, target: str, replacement: str) -> str:
    """Search and replace a specific block of text in a file inside `./workspace`. Creates file if missing."""
    from src.tools.rollback import backup_file
    from src.core.code.validation import validate_syntax
    
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety or path policies."
        
    if not _has_allowed_extension(filepath):
        return f"Error: Access denied. File extension not allowed. Approved extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        
    abs_path = _get_absolute_path(filepath)
    
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
    
    if not target:
        backup_file(filepath)
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(replacement)
            return f"Success: Overwrote '{filepath}' completely."
        except Exception as e:
            return f"Error overwriting file '{filepath}': {e}"

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        occurrences = content.count(target)
        if occurrences == 0:
            fuzzy_content = try_fuzzy_replace(content, target, replacement)
            if fuzzy_content is not None:
                backup_file(filepath)
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(fuzzy_content)
                return f"Success: Modified '{filepath}' successfully using relaxed matching."
                
            return (
                f"Error: Target text not found in '{filepath}'. "
                "Make sure spacing, newlines, and indentation match exactly.\n"
                "Fuzzy matching also failed.\n"
                "Fallback: If you cannot match the exact surgical block, you can overwrite the entire file "
                "by passing an empty target string (target='') and providing the complete new file content in replacement."
            )
        if occurrences > 1:
            return (
                f"Error: Target text matches {occurrences} times in '{filepath}'. "
                "Provide a larger context block to make the target query unique."
            )
            
        backup_file(filepath)
        new_content = content.replace(target, replacement, 1)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return f"Success: Modified '{filepath}' successfully."
    except Exception as e:
        return f"Error editing file '{filepath}': {e}"


def modify_files(filepath: str, target_code: str, replacement_code: str) -> str:
    """Search and replace a specific block of text in a file inside `./workspace`. Fails if file does not exist."""
    from src.tools.rollback import backup_file
    
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety or path policies."
        
    if not _has_allowed_extension(filepath):
        return f"Error: Access denied. File extension not allowed. Approved extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        
    abs_path = _get_absolute_path(filepath)
    if not os.path.isfile(abs_path):
        return f"Error: File '{filepath}' does not exist. Use create_files to create it first."
        
    if not target_code:
        backup_file(filepath)
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(replacement_code)
            return f"Success: Overwrote '{filepath}' completely."
        except Exception as e:
            return f"Error overwriting file '{filepath}': {e}"

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        occurrences = content.count(target_code)
        if occurrences == 0:
            fuzzy_content = try_fuzzy_replace(content, target_code, replacement_code)
            if fuzzy_content is not None:
                backup_file(filepath)
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(fuzzy_content)
                return f"Success: Modified '{filepath}' successfully using relaxed matching."
                
            return (
                f"Error: Target text not found in '{filepath}'. "
                "Make sure spacing, newlines, and indentation match exactly.\n"
                "Fuzzy matching also failed.\n"
                "Fallback: If you cannot match the exact surgical block, you can overwrite the entire file "
                "by passing an empty target string (target_code='') and providing the complete new file content in replacement_code."
            )
        if occurrences > 1:
            return (
                f"Error: Target text matches {occurrences} times in '{filepath}'. "
                "Provide a larger context block to make the target query unique."
            )
            
        backup_file(filepath)
        new_content = content.replace(target_code, replacement_code, 1)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return f"Success: Modified '{filepath}' successfully."
    except Exception as e:
        return f"Error editing file '{filepath}': {e}"


def delete_file(filepath: str) -> str:
    """Delete a file in the workspace."""
    from src.tools.rollback import backup_file
    
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety or path policies."
        
    abs_path = _get_absolute_path(filepath)
    if not os.path.isfile(abs_path):
        return f"Error: File '{filepath}' does not exist."
        
    backup_file(filepath)
    
    try:
        os.remove(abs_path)
        return f"Success: Deleted file '{filepath}'."
    except Exception as e:
        return f"Error deleting file '{filepath}': {e}"


def list_files(directory: str = ".") -> str:
    """List files and subdirectories inside the `./workspace` folder."""
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
    
    if executable not in ["python", "pytest", "npm", "git"]:
        return False
        
    if executable == "python":
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
        if len(cmd_args) == 3 and cmd_args[1] == "run" and cmd_args[2] in ["lint", "test", "build"]:
            return True
        return False
        
    if executable == "git":
        if len(cmd_args) == 2 and cmd_args[1] == "diff":
            return True
        return False
        
    return False


def execute_command(command: str) -> str:
    """Execute a command in the `./workspace` folder securely."""
    cmd_clean = command.strip()
    if not cmd_clean:
        return "Error: Empty command provided."
        
    try:
        cmd_args = shlex.split(cmd_clean)
    except Exception as e:
        return f"Error parsing command line: {e}"
        
    if not cmd_args:
        return "Error: Empty command provided."
        
    if not _is_safe_command(cmd_args):
        return f"Error: Command '{command}' blocked by safety policy. Command is not in allowlist."
        
    print(f"\n[SECURE RUN] Executing command: {cmd_args} in '{WORKSPACE_ROOT}'")
    
    try:
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
    """Execute a shell command in the `./workspace` folder."""
    return execute_command(command)