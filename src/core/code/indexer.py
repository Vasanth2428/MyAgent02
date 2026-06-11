import os
import logging
from typing import Set, Dict, Any, List

from src.core.code.parser import parse_code_file, ParserInitializationException
from src.core.code.symbol_table import SymbolTable
from src.core.code.dependency_graph import DependencyGraph
from src.core.code.parse_cache import ParseCache

logger = logging.getLogger("RAG.CodeIndexer")

# Multi-language support configuration
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
}

class CodeIndexer:
    """
    Scans repository directory structures and populates symbol tables and dependency graphs.
    Uses Tree-sitter for robust multi-language parsing (upgrade from raw ast).
    """
    def __init__(self, project_root: str) -> None:
        self.project_root = os.path.realpath(project_root)
        self.symbol_table = SymbolTable()
        self.dependency_graph = DependencyGraph()
        self.indexed_files: Set[str] = set()
        self.skipped_files: Set[str] = set()  # Issue #8: Track files with partial/failed parses
        self._parse_cache = ParseCache(self.project_root)  # Issue #9: Incremental indexing

        # Folders to exclude from indexing
        self.exclude_dirs = {
            ".git", "__pycache__", ".venv", "venv", "node_modules", "checkpoints", 
            ".pytest_cache", ".ruff_cache", "logs", "reports", ".next", "dist", "build"
        }

    def index_repository(self) -> None:
        """Walks the repository directory tree, parsing all supported source files."""
        logger.info(f"Initiating Tree-sitter code indexing of repository at: {self.project_root}")
        self.symbol_table.clear()
        self.dependency_graph.clear()
        self.indexed_files.clear()
        self.skipped_files.clear()

        # Issue #9: Load parse cache for incremental indexing
        self._parse_cache.load()
        cache_hits = 0
        
        files_count = 0
        language_counts: Dict[str, int] = {}
        
        for root, dirs, files in os.walk(self.project_root):
            # Exclude folders in-place
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    full_path = os.path.realpath(os.path.join(root, file))
                    rel_path = os.path.relpath(full_path, self.project_root)
                    
                    lang = SUPPORTED_EXTENSIONS[ext]
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                    
                    self._index_file(full_path, rel_path, lang)
                    files_count += 1
                    
        logger.info(f"Tree-sitter indexing completed. Total files parsed: {files_count} ({language_counts})")
        # Issue #8: Log summary of files with partial/failed parses
        if self.skipped_files:
            logger.warning(f"Partial/failed parses ({len(self.skipped_files)} files): {', '.join(sorted(self.skipped_files))}")

        # Issue #9: Save updated parse cache
        self._parse_cache.save()
        logger.info(f"Parse cache: {files_count} files indexed")

    def _index_file(self, full_path: str, rel_path: str, language: str = "python") -> None:
        """Parse a single file and load definitions into symbol table and dependency graph."""
        # Issue #9: Use cached result if file hasn't changed
        if not self._parse_cache.is_stale(full_path):
            res = self._parse_cache.get_cached(full_path)
            if res is not None:
                self.indexed_files.add(rel_path)
                self._load_result_into_tables(res, rel_path)
                return

        try:
            res = parse_code_file(full_path, language=language)
        except ParserInitializationException as e:
            logger.warning(f"Failed to parse {rel_path}: {e}")
            self.skipped_files.add(rel_path)
            return

        self.indexed_files.add(rel_path)

        # Update cache with fresh parse result
        self._parse_cache.update(full_path, res)

        # Issue #8: Track files with partial or failed parses
        if res.get("partial") or res.get("error"):
            warning = res.get("warning", res.get("error", "Unknown parse issue"))
            logger.warning(f"Partial/failed parse for {rel_path}: {warning}")
            self.skipped_files.add(rel_path)
        self._load_result_into_tables(res, rel_path)

    def _load_result_into_tables(self, res: Dict[str, Any], rel_path: str) -> None:
        """Load parsed symbols, imports, and calls into the symbol table and dependency graph."""
        # 1. Populate Symbol Table
        for sym in res.get("symbols", []):
            # Convert filepath in symbol to relative path for portability
            sym_rel = sym.copy()
            sym_rel["filepath"] = rel_path.replace("\\", "/")
            self.symbol_table.add_symbol(sym_rel)
            
        # 2. Populate Dependency Graph (Imports)
        for imp in res.get("imports", []):
            module_name = imp.get("name") or imp.get("module") or ""
            self.dependency_graph.add_import(rel_path.replace("\\", "/"), module_name)
            
        # 3. Populate Dependency Graph (Calls)
        for call in res.get("calls", []):
            self.dependency_graph.add_call(rel_path.replace("\\", "/"), call["name"])
