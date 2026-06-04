import logging
import os
from typing import List, Dict, Any, Optional

from src.core.code.indexer import CodeIndexer
from src.core.code.code_registry import CodeRegistry
from src.core.retriever import WeaviateRetriever

logger = logging.getLogger("RAG.CodeRetrievalService")

class CodeRetrievalService:
    """
    Coordinates repository-level code semantic search and symbol analysis.
    """
    def __init__(self, project_root: str, retriever: WeaviateRetriever) -> None:
        self.project_root = project_root
        self.retriever = retriever
        
        self.indexer = CodeIndexer(self.project_root)
        index_file = os.path.join(self.project_root, "checkpoints", "code_index.json")
        self.registry = CodeRegistry(index_file, self.indexer)
        
        # Load pre-existing index if available, otherwise compile one
        if not self.registry.load_index():
            self.indexer.index_repository()
            self.registry.save_index()

    def sync_index(self) -> None:
        """Forces a directory scan and persists the updated index."""
        logger.info("Synchronizing code index...")
        self.indexer.index_repository()
        self.registry.save_index()

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Fuzzy searches the symbol table for matching functions, methods, or classes."""
        logger.info(f"Symbol query lookup: '{query}'")
        return self.indexer.symbol_table.search_symbols(query)

    def get_symbol_definition(self, name: str) -> List[Dict[str, Any]]:
        """Retrieves exact symbol metadata instances matching name."""
        return self.indexer.symbol_table.get_symbols_by_name(name)

    def get_file_imports(self, filepath: str) -> List[str]:
        """Gets imports declared inside the target file."""
        return self.indexer.dependency_graph.get_imports(filepath)

    def get_file_imported_by(self, filepath: str) -> List[str]:
        """Gets files importing the target file module."""
        return self.indexer.dependency_graph.get_imported_by(filepath)

    def get_symbol_dependencies(self, symbol_name: str) -> Dict[str, List[str]]:
        """Retrieves calls and caller lists for the symbol."""
        return {
            "callees": self.indexer.dependency_graph.get_callees(symbol_name),
            "callers": self.indexer.dependency_graph.get_callers(symbol_name)
        }

    async def hybrid_retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves matching code snippets using Weaviate vector index and symbol lookup.
        """
        logger.info(f"Hybrid code retrieval query: '{query}'")
        
        # 1. Query Weaviate RAGCode collection if retriever is active
        vector_results = []
        if hasattr(self.retriever, "search_code_chunks"):
            try:
                # We search Weaviate code chunks
                vector_results = self.retriever.search_code_chunks(query, limit=limit)
            except Exception as e:
                logger.error(f"Failed to query Weaviate RAGCode collection: {e}")

        # 2. Extract static symbol matches
        symbol_matches = self.search_symbols(query)
        
        # 3. Combine results
        combined_results = []
        for res in vector_results:
            combined_results.append({
                "type": "code_snippet",
                "text": res.get("text"),
                "filepath": res.get("filepath"),
                "start_line": res.get("start_line"),
                "end_line": res.get("end_line"),
                "symbol_name": res.get("symbol_name"),
                "symbol_type": res.get("symbol_type")
            })

        for sym in symbol_matches[:limit]:
            combined_results.append({
                "type": "symbol_def",
                "text": f"class {sym['name']}" if sym['type'] == 'class' else f"def {sym['name']}",
                "filepath": sym.get("filepath"),
                "start_line": sym.get("start_line"),
                "end_line": sym.get("end_line"),
                "symbol_name": sym.get("name"),
                "symbol_type": sym.get("type"),
                "docstring": sym.get("docstring")
            })

        return combined_results
