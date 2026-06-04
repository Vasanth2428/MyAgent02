import logging
from typing import Dict, Set, List

logger = logging.getLogger("RAG.DependencyGraph")

class DependencyGraph:
    """
    Tracks imports and calls dependency relationships between modules/files and symbols.
    """
    def __init__(self) -> None:
        # File/module level imports: file_path -> Set of imported module names
        self.imports: Dict[str, Set[str]] = {}
        # File/module level dependents: module_name -> Set of file paths importing it
        self.imported_by: Dict[str, Set[str]] = {}
        # Call relationships: caller_symbol -> Set of callee_symbols
        self.calls: Dict[str, Set[str]] = {}
        # Reverse call relationships: callee_symbol -> Set of caller_symbols
        self.called_by: Dict[str, Set[str]] = {}

    def add_import(self, filepath: str, imported_module: str) -> None:
        """Record that a file imports a module."""
        if filepath not in self.imports:
            self.imports[filepath] = set()
        self.imports[filepath].add(imported_module)

        if imported_module not in self.imported_by:
            self.imported_by[imported_module] = set()
        self.imported_by[imported_module].add(filepath)

    def add_call(self, caller: str, callee: str) -> None:
        """Record that caller calls callee."""
        if caller not in self.calls:
            self.calls[caller] = set()
        self.calls[caller].add(callee)

        if callee not in self.called_by:
            self.called_by[callee] = set()
        self.called_by[callee].add(caller)

    def get_imports(self, filepath: str) -> List[str]:
        """Get list of modules imported by the file."""
        return list(self.imports.get(filepath, set()))

    def get_imported_by(self, module: str) -> List[str]:
        """Get list of files importing the module."""
        return list(self.imported_by.get(module, set()))

    def get_callees(self, symbol: str) -> List[str]:
        """Get list of symbols called by the symbol."""
        return list(self.calls.get(symbol, set()))

    def get_callers(self, symbol: str) -> List[str]:
        """Get list of symbols that call the symbol."""
        return list(self.called_by.get(symbol, set()))

    def clear(self) -> None:
        """Clear the dependency graph."""
        self.imports.clear()
        self.imported_by.clear()
        self.calls.clear()
        self.called_by.clear()
        logger.debug("Dependency graph cleared.")
