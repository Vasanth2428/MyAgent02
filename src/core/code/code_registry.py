import json
import os
import logging
from typing import Dict, Any, List

from src.core.code.indexer import CodeIndexer

logger = logging.getLogger("RAG.CodeRegistry")

class CodeRegistry:
    """
    Handles saving and loading the compiled AST index to disk to prevent re-indexing overhead on restarts.
    """
    def __init__(self, index_file_path: str, indexer: CodeIndexer) -> None:
        self.index_file_path = index_file_path
        self.indexer = indexer

    def save_index(self) -> None:
        """Serializes and saves the indexer's symbol table and dependency graph to JSON."""
        logger.info(f"Saving compiled code index to: {self.index_file_path}")
        
        try:
            # Prepare serialization payload
            payload = {
                "indexed_files": list(self.indexer.indexed_files),
                "symbols": self.indexer.symbol_table.get_all_symbols(),
                "imports": {
                    filepath: list(imported)
                    for filepath, imported in self.indexer.dependency_graph.imports.items()
                },
                "imported_by": {
                    module: list(importers)
                    for module, importers in self.indexer.dependency_graph.imported_by.items()
                },
                "calls": {
                    caller: list(callees)
                    for caller, callees in self.indexer.dependency_graph.calls.items()
                },
                "called_by": {
                    callee: list(callers)
                    for callee, callers in self.indexer.dependency_graph.called_by.items()
                }
            }
            
            # Ensure parent directories exist
            os.makedirs(os.path.dirname(self.index_file_path), exist_ok=True)
            with open(self.index_file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("Code index successfully saved.")
        except Exception as e:
            logger.error(f"Failed to save code index: {e}")

    def load_index(self) -> bool:
        """Loads and populates the indexer state from JSON if it exists."""
        if not os.path.exists(self.index_file_path):
            logger.info("No pre-compiled code index file found. Scan required.")
            return False
            
        logger.info(f"Loading pre-compiled code index from: {self.index_file_path}")
        try:
            with open(self.index_file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                
            self.indexer.symbol_table.clear()
            self.indexer.dependency_graph.clear()
            self.indexer.indexed_files = set(payload.get("indexed_files", []))
            
            # Populate Symbol Table
            for sym in payload.get("symbols", []):
                self.indexer.symbol_table.add_symbol(sym)
                
            # Populate Dependency Graph (Imports)
            for filepath, imported in payload.get("imports", {}).items():
                for imp in imported:
                    self.indexer.dependency_graph.add_import(filepath, imp)
                    
            # Populate Dependency Graph (Calls)
            for caller, callees in payload.get("calls", {}).items():
                for callee in callees:
                    self.indexer.dependency_graph.add_call(caller, callee)
                    
            logger.info(f"Successfully loaded code index containing {len(self.indexer.indexed_files)} files.")
            return True
        except Exception as e:
            logger.error(f"Failed to load pre-compiled code index: {e}")
            return False
