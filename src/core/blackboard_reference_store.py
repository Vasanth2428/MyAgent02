"""
Blackboard Reference Store - Prevents state bloat by storing references instead of full content.

GRAPH-01: Blackboard Memory Contamination Fix

Instead of storing full document content in the shared state (scratchpad),
this module stores references/IDs and retrieves full content on-demand.
"""

import os
import json
import logging
import time
from typing import Dict, Optional, Any

logger = logging.getLogger("RAG.BlackboardRef")

# In-memory store for large content (would be persisted to disk in production)
_REFERENCE_CACHE: Dict[str, str] = {}
_CACHE_TTL_SECONDS = 3600 * 4  # 4 hours
_MAX_CACHE_SIZE = 1000

# Persistent file for cache
_CACHE_FILE = ".blackboard_cache.json"


def _evict_stale() -> None:
    """Remove expired entries from cache."""
    cutoff = time.time() - _CACHE_TTL_SECONDS
    stale = [k for k, v in list(_REFERENCE_CACHE.items()) if isinstance(v, dict) and v.get("timestamp", 0) < cutoff]
    for key in stale:
        del _REFERENCE_CACHE[key]
    
    # Also enforce size limit
    if len(_REFERENCE_CACHE) > _MAX_CACHE_SIZE:
        # Remove oldest entries
        sorted_items = sorted(
            [(k, v) for k, v in _REFERENCE_CACHE.items() if isinstance(v, dict)],
            key=lambda x: x[1].get("timestamp", 0)
        )
        for key, _ in sorted_items[:len(_REFERENCE_CACHE) - _MAX_CACHE_SIZE]:
            del _REFERENCE_CACHE[key]


def _get_cache_path() -> str:
    """Get path to persistent cache file."""
    return os.path.join(os.getcwd(), _CACHE_FILE)


def load_cache() -> None:
    """Load cache from disk."""
    try:
        path = _get_cache_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _REFERENCE_CACHE.update(json.load(f))
            logger.debug(f"Loaded blackboard reference cache with {len(_REFERENCE_CACHE)} entries")
    except Exception as e:
        logger.warning(f"Failed to load blackboard cache: {e}")


def save_cache() -> None:
    """Persist cache to disk."""
    try:
        path = _get_cache_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_REFERENCE_CACHE, f)
    except Exception as e:
        logger.warning(f"Failed to save blackboard cache: {e}")


def store_reference(worker_name: str, output: str) -> str:
    """
    Store output and return a reference ID.
    
    Args:
        worker_name: Name of the worker storing the reference
        output: Full output content to store
        
    Returns:
        Reference ID that can be used to retrieve the content
    """
    _evict_stale()
    
    # Generate unique reference ID
    ref_id = f"ref_{worker_name}_{abs(hash(output)) & 0xFFFFF:x}_{int(time.time())}"
    
    _REFERENCE_CACHE[ref_id] = {
        "text": output,
        "worker": worker_name,
        "timestamp": time.time()
    }
    save_cache()
    
    return ref_id


def get_reference(ref_id: str) -> Optional[str]:
    """Retrieve full content by reference ID."""
    entry = _REFERENCE_CACHE.get(ref_id)
    if entry and isinstance(entry, dict):
        return entry.get("text")
    return None


def get_summary(ref_id: str, max_len: int = 150) -> str:
    """Get a short summary of referenced content."""
    text = get_reference(ref_id)
    if not text:
        return f"[Reference {ref_id} not found]"
    
    summary = text[:max_len].replace("\n", " ")
    if len(text) > max_len:
        summary += "..."
    return f"[{ref_id[:16]}] {summary}"


def compact_scratchpad(scratchpad: str, max_refs: int = 20) -> str:
    """
    Compact scratchpad by replacing full content with reference summaries.
    
    This prevents state bloat while keeping context manageable.
    """
    # Extract reference IDs from scratchpad
    import re
    ref_pattern = r'\[REF:([^\]]+)\]'
    refs = re.findall(ref_pattern, scratchpad)
    
    # Build compact view
    lines = scratchpad.split("\n")
    compact_lines = []
    ref_count = 0
    
    for line in lines:
        # If line contains a reference and we haven't hit limit, keep it
        if "[REF:" in line and ref_count < max_refs:
            # Replace full content with summary
            match = re.search(r'\[REF:([^\]]+)\]', line)
            if match:
                ref_id = match.group(1)
                summary = get_summary(ref_id)
                # Extract the rest of the line (e.g., "- [Worker Name]:")
                prefix_match = re.search(r'^(.*\[REF:[^\]]+\])(.*)$', line)
                if prefix_match:
                    compact_lines.append(f"{prefix_match.group(1)} {summary}")
                    ref_count += 1
        elif line.startswith("- [") or not line.strip():
            # Keep headers and empty lines
            compact_lines.append(line)
    
    return "\n".join(compact_lines)


# Initialize cache on module load
load_cache()