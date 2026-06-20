import os
import logging
from typing import List, Dict, Any, Optional

from src.core.code.parser import extract_symbol_slices, get_symbol_tokens
from src.tools.coding_tools import _is_safe_path, _has_allowed_extension, _get_absolute_path, view_code_file, edit_code_file

logger = logging.getLogger("MultiAgent.TokenSavingTools")

# Approximate token count per line (used for estimation)
TOKENS_PER_LINE = 5

def get_pruned_context(filepath: str, token_budget: int) -> str:
    """Return a context string consisting of symbol snippets that fit within the token_budget.

    The function parses the file, extracts symbol slices, sorts them by start_line (newest first),
    and includes as many symbols as possible without exceeding the token_budget.
    Returns a formatted string with each symbol's source code.
    """
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety policies."
    if not _has_allowed_extension(filepath):
        return f"Error: Access denied. File extension not allowed."

    abs_path = _get_absolute_path(filepath)
    if not os.path.isfile(abs_path):
        return f"Error: File '{filepath}' does not exist."

    # Read raw content for parsing and token estimation
    try:
        with open(abs_path, "rb") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file '{filepath}': {e}"

    # Get symbol slices with token estimates
    slices = extract_symbol_slices(content, abs_path)
    # Sort by start_line descending (newest first). Could also sort by relevance later.
    slices.sort(key=lambda s: s["start_line"], reverse=True)

    selected: List[Dict[str, Any]] = []
    used_tokens = 0
    for sym in slices:
        sym_tokens = get_symbol_tokens(sym)
        if used_tokens + sym_tokens > token_budget:
            continue
        selected.append(sym)
        used_tokens += sym_tokens
        if used_tokens >= token_budget:
            break

    if not selected:
        return ""  # Nothing fits the budget

    # Build context string by retrieving each symbol's source lines
    lines = []
    for sym in selected:
        start = sym["start_line"]
        end = sym["end_line"]
        # Use view_code_file to safely fetch the range
        snippet = view_code_file(filepath, start_line=start, end_line=end)
        lines.append(f"--- {sym['type']} {sym['name']} ({start}-{end}) ---\n{snippet}\n")

    return "\n".join(lines)

def retrieve_symbol_context(symbol_name: str, workspace_root: str = ".") -> Dict[str, Any]:
    """Search the workspace for a symbol by name and return its full source snippet.

    Returns a dictionary with keys: 'filepath', 'start_line', 'end_line', 'code'.
    If multiple matches are found, the first one (by lexical order) is returned.
    """
    # Walk the workspace and parse each file (limited to allowed extensions)
    for root, dirs, files in os.walk(workspace_root):
        # Prune hidden/virtual/system directories in-place to speed up walking
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', 'venv', '.venv')]
        for fname in files:
            if not _has_allowed_extension(fname):
                continue
            rel_path = os.path.relpath(os.path.join(root, fname), workspace_root)
            if not _is_safe_path(rel_path):
                continue
            abs_path = os.path.join(workspace_root, rel_path)
            try:
                with open(abs_path, "rb") as f:
                    content = f.read()
                slices = extract_symbol_slices(content, abs_path)
                for sym in slices:
                    if sym.get("name") == symbol_name:
                        # Retrieve full source block
                        snippet = view_code_file(rel_path, start_line=sym["start_line"], end_line=sym["end_line"])
                        return {
                            "filepath": rel_path,
                            "start_line": sym["start_line"],
                            "end_line": sym["end_line"],
                            "code": snippet,
                        }
            except Exception as e:
                logger.warning(f"Failed to process file {rel_path}: {e}")
    return {}

def apply_surgical_edit(filepath: str, target_symbol: str, new_code: str) -> str:
    """Replace the source of a given symbol with new_code using a minimal edit.

    The function backs up the original file, extracts the target symbol range, and writes the
    new code in place of the old block via `edit_code_file`. Returns a success/failure message.
    """
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety policies."
    if not _has_allowed_extension(filepath):
        return f"Error: Access denied. File extension not allowed."

    # Locate the symbol in the file
    abs_path = _get_absolute_path(filepath)
    try:
        with open(abs_path, "rb") as f:
            content = f.read()
        slices = extract_symbol_slices(content, abs_path)
        target = next((s for s in slices if s.get("name") == target_symbol), None)
        if not target:
            return f"Error: Symbol '{target_symbol}' not found in '{filepath}'."
    except Exception as e:
        return f"Error reading file '{filepath}': {e}"

    # Build the target block's source text from raw lines (avoiding line numbers and headers from view_code_file)
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        old_snippet = "".join(lines[target["start_line"] - 1 : target["end_line"]])
    except Exception as e:
        return f"Error reading file '{filepath}' for surgical edit: {e}"

    # Use edit_code_file with fuzzy replace (empty target triggers full replace)
    # We replace the exact block using the original snippet as target
    result = edit_code_file(filepath, target=old_snippet, replacement=new_code)
    return result


# ---------------------------------------------------------------------------
# Token-Saving Utilities (Phase 2)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Return the exact BPE token count for *text* using the project tokenizer.

    This is a lightweight local computation (no LLM call).  Agents should use
    it to pre-check whether reading a large file or appending tool output will
    exceed their context budget.
    """
    if not text:
        return 0
    import tiktoken
    from src.core.config import TOKENIZER_ENCODING
    enc = tiktoken.get_encoding(TOKENIZER_ENCODING)
    return len(enc.encode(text))


def get_token_budget_remaining(
    current_prompt_tokens: int,
    context_limit: int = 0,
) -> Dict[str, Any]:
    """Compute how many tokens remain before the context limit is reached.

    Parameters
    ----------
    current_prompt_tokens : int
        Tokens already consumed by the current prompt / context.
    context_limit : int, optional
        Maximum token budget.  Falls back to ``TOTAL_CONTEXT_BUDGET`` from
        config when 0 or omitted.

    Returns
    -------
    dict
        ``context_limit``, ``used``, ``remaining``, ``utilization_pct``,
        and a human-readable ``warning`` string (empty when healthy).
    """
    from src.core.config import TOTAL_CONTEXT_BUDGET

    limit = context_limit if context_limit > 0 else TOTAL_CONTEXT_BUDGET
    remaining = max(0, limit - current_prompt_tokens)
    utilization = round(current_prompt_tokens / limit * 100, 1) if limit > 0 else 100.0

    warning = ""
    if utilization >= 90:
        warning = "CRITICAL: Less than 10% budget remaining — avoid loading more context."
    elif utilization >= 80:
        warning = "WARNING: Over 80% budget used — consider summarising before adding more."

    return {
        "context_limit": limit,
        "used": current_prompt_tokens,
        "remaining": remaining,
        "utilization_pct": utilization,
        "warning": warning,
    }


_MAX_HEADER_LINES = 50  # Hard cap to prevent agent circumvention


def fetch_file_headers(filepath: str, max_lines: int = 20) -> str:
    """Return only the first *max_lines* lines of a workspace file.

    This is much cheaper than a full ``read_files`` call and typically
    captures imports and class / function signatures — enough to understand
    a file's public API without loading the body.

    Parameters
    ----------
    filepath : str
        Relative path inside ``./workspace``.
    max_lines : int
        Number of leading lines to return (capped at 50).
    """
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety policies."
    if not _has_allowed_extension(filepath):
        return f"Error: Access denied. File extension not allowed."

    abs_path = _get_absolute_path(filepath)
    if not os.path.isfile(abs_path):
        return f"Error: File '{filepath}' does not exist."

    capped = min(max(1, max_lines), _MAX_HEADER_LINES)

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            
        total_lines = len(all_lines)
        lines = [f"{i + 1}: {line}" for i, line in enumerate(all_lines[:capped])]
        header = f"--- Headers of {filepath} (first {capped} of {total_lines} lines) ---\n"

        from src.tools.coding_tools import sanitize_file_content_for_llm
        return sanitize_file_content_for_llm(header + "".join(lines))
    except Exception as e:
        return f"Error reading file headers '{filepath}': {e}"


def summarize_tool_output(text: str, max_tokens: int = 200) -> str:
    """Truncate verbose tool output to fit within *max_tokens*.

    Unlike the semantic :class:`Compressor`, this performs fast deterministic
    line-level truncation — designed for grep results, file listings, and
    command output where preserving the first lines is most useful.

    Parameters
    ----------
    text : str
        Raw tool output string.
    max_tokens : int
        Target token budget (default 200).
    """
    if not text:
        return ""

    import tiktoken
    from src.core.config import TOKENIZER_ENCODING
    enc = tiktoken.get_encoding(TOKENIZER_ENCODING)

    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text  # Already fits — fast path

    # Line-level truncation: keep lines from the top until budget exhausted
    lines = text.splitlines(keepends=True)
    kept: List[str] = []
    running = 0
    for line in lines:
        line_tokens = len(enc.encode(line))
        if running + line_tokens > max_tokens:
            break
        kept.append(line)
        running += line_tokens

    truncated_count = len(lines) - len(kept)
    suffix = f"\n[...{truncated_count} more lines truncated to fit {max_tokens}-token budget]"
    return "".join(kept) + suffix
