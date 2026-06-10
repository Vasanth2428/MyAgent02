import os
import shutil
import logging
from datetime import datetime
from typing import Optional

from src.tools.coding_tools import WORKSPACE_ROOT

logger = logging.getLogger("MultiAgent.Rollback")

BACKUP_ROOT = os.path.join(WORKSPACE_ROOT, ".kilo", "backups")


def _ensure_backup_dir() -> None:
    """Ensure the backup directory exists."""
    if not os.path.exists(BACKUP_ROOT):
        os.makedirs(BACKUP_ROOT, exist_ok=True)


def backup_file(filepath: str) -> Optional[str]:
    """
    Create a backup of a file before modification.
    Returns the backup path on success, None on failure.
    """
    _ensure_backup_dir()
    
    abs_path = os.path.realpath(os.path.join(WORKSPACE_ROOT, filepath))
    if not os.path.isfile(abs_path):
        return None
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{ts}_{os.path.basename(filepath)}"
    backup_path = os.path.join(BACKUP_ROOT, filepath.replace("/", os.sep), backup_name)
    
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    
    try:
        shutil.copy2(abs_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to backup {filepath}: {e}")
        return None


def rollback_file(filepath: str, backup_path: Optional[str] = None) -> str:
    """
    Restore a file from its most recent backup.
    If backup_path is provided, use that specific backup.
    Otherwise, find the most recent backup for the file.
    """
    from src.tools.coding_tools import _is_safe_path
    
    if not _is_safe_path(filepath):
        return f"Error: Access denied. Filepath '{filepath}' violates safety or path policies."
    
    abs_path = os.path.realpath(os.path.join(WORKSPACE_ROOT, filepath))
    
    if backup_path and os.path.isfile(backup_path):
        backup = backup_path
    else:
        backup_dir = os.path.join(BACKUP_ROOT, filepath.replace("/", os.sep))
        if not os.path.isdir(backup_dir):
            return f"Error: No backups found for '{filepath}'."
        
        backups = sorted(
            [f for f in os.listdir(backup_dir) if f.endswith(os.path.basename(filepath)) or "." in f],
            reverse=True
        )
        
        if not backups:
            return f"Error: No backups found for '{filepath}'."
        
        backup = os.path.join(backup_dir, backups[0])
    
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        shutil.copy2(backup, abs_path)
        return f"Success: Rolled back '{filepath}' from backup."
    except Exception as e:
        return f"Error rolling back '{filepath}': {e}"


def list_backups(filepath: str) -> str:
    """List all available backups for a file."""
    backup_dir = os.path.join(BACKUP_ROOT, filepath.replace("/", os.sep))
    if not os.path.isdir(backup_dir):
        return f"No backups found for '{filepath}'."
    
    backups = sorted(
        [f for f in os.listdir(backup_dir)],
        reverse=True
    )
    
    if not backups:
        return f"No backups found for '{filepath}'."
    
    output = [f"--- Backups for '{filepath}' ---"]
    for b in backups:
        full_path = os.path.join(backup_dir, b)
        mtime = datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M:%S")
        output.append(f"{b} (modified: {mtime})")
    
    return "\n".join(output)