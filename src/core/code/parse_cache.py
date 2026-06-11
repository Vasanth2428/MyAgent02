"""
Parse Cache - SHA-256 based file hash cache for incremental code indexing.

Issue #9: Avoids re-parsing unchanged files on every startup by caching
parse results keyed by file content hash. Only files whose content has
changed since the last index run are re-parsed.
"""

import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("RAG.ParseCache")

# Default cache file location (relative to project root)
_CACHE_FILENAME = ".code_index_cache.json"


class ParseCache:
    """
    File-hash-based cache for parsed code results.
    
    Stores {filepath: {"hash": str, "symbols": [...], "imports": [...], "calls": [...]}}
    in a JSON sidecar file. On startup, compares file hashes to detect changes.
    """
    
    def __init__(self, project_root: str, cache_filename: str = _CACHE_FILENAME):
        self.project_root = project_root
        self.cache_path = os.path.join(project_root, cache_filename)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
    
    def load(self) -> None:
        """Load the cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded parse cache with {len(self._cache)} entries from {self.cache_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load parse cache, starting fresh: {e}")
                self._cache = {}
        else:
            logger.debug("No parse cache file found, starting fresh.")
            self._cache = {}
    
    def save(self) -> None:
        """Persist the cache to disk (only if modified)."""
        if not self._dirty:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, default=str)
            self._dirty = False
            logger.info(f"Saved parse cache with {len(self._cache)} entries to {self.cache_path}")
        except IOError as e:
            logger.error(f"Failed to save parse cache: {e}")
    
    @staticmethod
    def _compute_hash(filepath: str) -> str:
        """Compute SHA-256 hash of a file's contents."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def is_stale(self, filepath: str) -> bool:
        """Check if a file needs re-parsing (not cached or content changed)."""
        rel_path = os.path.relpath(filepath, self.project_root).replace("\\", "/")
        cached = self._cache.get(rel_path)
        if cached is None:
            return True
        try:
            current_hash = self._compute_hash(filepath)
            return current_hash != cached.get("hash")
        except (IOError, OSError):
            return True
    
    def get_cached(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Get cached parse result for a file (or None if stale/missing)."""
        rel_path = os.path.relpath(filepath, self.project_root).replace("\\", "/")
        cached = self._cache.get(rel_path)
        if cached is None:
            return None
        # Return the parse result portion (without the hash)
        return {
            "symbols": cached.get("symbols", []),
            "imports": cached.get("imports", []),
            "calls": cached.get("calls", []),
            "filepath": filepath,
            "lines_count": cached.get("lines_count", 0),
        }
    
    def update(self, filepath: str, result: Dict[str, Any]) -> None:
        """Update the cache entry for a file after parsing."""
        rel_path = os.path.relpath(filepath, self.project_root).replace("\\", "/")
        try:
            file_hash = self._compute_hash(filepath)
        except (IOError, OSError):
            return
        self._cache[rel_path] = {
            "hash": file_hash,
            "symbols": result.get("symbols", []),
            "imports": result.get("imports", []),
            "calls": result.get("calls", []),
            "lines_count": result.get("lines_count", 0),
        }
        self._dirty = True
    
    def remove(self, filepath: str) -> None:
        """Remove a file from the cache (e.g. if deleted)."""
        rel_path = os.path.relpath(filepath, self.project_root).replace("\\", "/")
        if rel_path in self._cache:
            del self._cache[rel_path]
            self._dirty = True
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._dirty = True
