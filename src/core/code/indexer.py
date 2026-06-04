import os
import logging
from typing import Set, Dict, Any, List

from src.core.code.parser import parse_code_file
from src.core.code.symbol_table import SymbolTable
from src.core.code.dependency_graph import DependencyGraph

logger = logging.getLogger("RAG.CodeIndexer")

class CodeIndexer:
    """
    Scans repository directory structures and populates symbol tables and dependency graphs.
    """
    def __init__(self, project_root: str) -> None:
        self.project_root = os.path.realpath(project_root)
        self.symbol_table = SymbolTable()
        self.dependency_graph = DependencyGraph()
        self.indexed_files: Set[str] = set()

        # Folders to exclude from indexing
        self.exclude_dirs = {
            ".git", "__pycache__", ".venv", "venv", "node_modules", "checkpoints", 
            ".pytest_cache", ".ruff_cache", "logs", "reports"
        }

    def index_repository(self) -> None:
        """Walks the repository directory tree, parsing all Python source files."""
        logger.info(f"Initiating AST code indexing of repository at: {self.project_root}")
        self.symbol_table.clear()
        self.dependency_graph.clear()
        self.indexed_files.clear()
        
        py_files_count = 0
        
        for root, dirs, files in os.walk(self.project_root):
            # Exclude folders in-place
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.realpath(os.path.join(root, file))
                    rel_path = os.path.relpath(full_path, self.project_root)
                    
                    self._index_file(full_path, rel_path)
                    py_files_count += 1
                    
        logger.info(f"AST code indexing completed. Total files parsed: {py_files_count}")

    def _index_file(self, full_path: str, rel_path: str) -> None:
        """Parse a single file and load definitions into symbol table and dependency graph."""
        res = parse_code_file(full_path)
        self.indexed_files.add(rel_path)
        
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
        # We can map calls if we have caller functions/classes.
        # For simplicity, we register function call nodes.
        for call in res.get("calls", []):
            # Match callers based on active symbols inside file if needed.
            # Here we just record that caller (filepath) calls the symbol.
            self.dependency_graph.add_call(rel_path.replace("\\", "/"), call["name"])
