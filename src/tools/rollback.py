import os
import shutil
import logging
from datetime import datetime
from typing import Optional

from src.tools.coding_tools import WORKSPACE_ROOT

logger = logging.getLogger("MultiAgent.Rollback")

BACKUP_ROOT = os.path.join(WORKSPACE_ROOT, ".backups")


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