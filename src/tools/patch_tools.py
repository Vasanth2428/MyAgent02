import difflib
import os
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("RAG.PatchTools")

def generate_diff_patch(filepath: str, original_code: str, replacement_code: str) -> str:
    """
    Generates a unified diff patch between the original code and the replacement code.
    """
    logger.info(f"Generating unified diff patch for: {filepath}")
    
    orig_lines = original_code.splitlines(keepends=True)
    repl_lines = replacement_code.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        orig_lines,
        repl_lines,
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
        lineterm="\n"
    )
    
    return "".join(diff)


def apply_patch(original_text: str, patch_text: str) -> str:
    """
    Pure Python implementation of unified diff patch applier.
    Validates context and raises ValueError if hunks do not match.
    """
    original_lines = original_text.splitlines()
    patch_lines = patch_text.splitlines()
    
    output_lines: List[str] = []
    original_idx = 0
    patch_idx = 0
    
    # Skip headers
    while patch_idx < len(patch_lines) and (patch_lines[patch_idx].startswith("---") or patch_lines[patch_idx].startswith("+++")):
        patch_idx += 1
        
    while patch_idx < len(patch_lines):
        line = patch_lines[patch_idx]
        if line.startswith("@@"):
            # Parse hunk header: @@ -start,len +start,len @@
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed hunk header: {line}")
                
            orig_start_str = parts[1].split(",")[0].replace("-", "")
            orig_start = int(orig_start_str) if orig_start_str else 1
            orig_idx = max(0, orig_start - 1)
            
            # Catch up original lines to hunk start
            while original_idx < orig_idx and original_idx < len(original_lines):
                output_lines.append(original_lines[original_idx])
                original_idx += 1
                
            patch_idx += 1
            
            # Process hunk content
            while patch_idx < len(patch_lines):
                hunk_line = patch_lines[patch_idx]
                if hunk_line.startswith("@@") or hunk_line.startswith("---") or hunk_line.startswith("+++"):
                    break
                    
                if hunk_line.startswith(" "):
                    if original_idx >= len(original_lines):
                        raise ValueError("Patch hunk mismatch: reached end of file expecting context.")
                    orig_l = original_lines[original_idx]
                    if orig_l != hunk_line[1:]:
                        raise ValueError(f"Patch hunk mismatch: expected context '{hunk_line[1:]}', got '{orig_l}'")
                    output_lines.append(orig_l)
                    original_idx += 1
                elif hunk_line.startswith("-"):
                    if original_idx >= len(original_lines):
                        raise ValueError("Patch hunk mismatch: reached end of file expecting line to delete.")
                    orig_l = original_lines[original_idx]
                    if orig_l != hunk_line[1:]:
                        raise ValueError(f"Patch hunk mismatch: expected deletion '{hunk_line[1:]}', got '{orig_l}'")
                    original_idx += 1
                elif hunk_line.startswith("+"):
                    output_lines.append(hunk_line[1:])
                elif hunk_line.startswith("\\ No newline"):
                    pass
                    
                patch_idx += 1
        else:
            patch_idx += 1
            
    # Append remainder
    while original_idx < len(original_lines):
        output_lines.append(original_lines[original_idx])
        original_idx += 1
        
    result = "\n".join(output_lines)
    if original_text.endswith("\n") or not original_text:
        result += "\n"
    return result


def dry_run_patch(abs_filepath: str, patch_diff: str) -> Tuple[bool, str]:
    """
    Applies the patch diff to the target file in memory and returns (success, patched_content_or_error).
    """
    logger.info(f"Dry-running patch for: {abs_filepath}")
    
    if not os.path.exists(abs_filepath):
        original_content = ""
    else:
        try:
            with open(abs_filepath, "r", encoding="utf-8", errors="replace") as f:
                original_content = f.read()
        except Exception as e:
            return False, f"Failed to read target file: {e}"
            
    try:
        patched_content = apply_patch(original_content, patch_diff)
        return True, patched_content
    except Exception as e:
        logger.error(f"Patch application failed for {abs_filepath}: {e}")
        return False, str(e)
