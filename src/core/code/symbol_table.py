import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("RAG.SymbolTable")

class SymbolTable:
    """
    In-memory symbol table tracking classes, functions, and methods across the codebase.
    """
    def __init__(self) -> None:
        self.symbols: Dict[str, List[Dict[str, Any]]] = {}

    def add_symbol(self, symbol: Dict[str, Any]) -> None:
        """Add a symbol definition to the table."""
        name = symbol.get("name")
        if not name:
            return
            
        if name not in self.symbols:
            self.symbols[name] = []
        self.symbols[name].append(symbol)

    def get_symbols_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Retrieve all defined instances of a symbol by name."""
        return self.symbols.get(name, [])

    def get_symbol(self, name: str, filepath: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a specific symbol instance, optionally filtered by filepath."""
        instances = self.symbols.get(name, [])
        if not instances:
            return None
        if filepath:
            for inst in instances:
                if inst.get("filepath") == filepath:
                    return inst
        return instances[0]

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Fuzzy/substring search for symbols matching query."""
        results = []
        query_lower = query.lower()
        for name, instances in self.symbols.items():
            if query_lower in name.lower():
                results.extend(instances)
        return results

    def get_all_symbols(self) -> List[Dict[str, Any]]:
        """Returns all registered symbols."""
        all_syms = []
        for instances in self.symbols.values():
            all_syms.extend(instances)
        return all_syms

    def clear(self) -> None:
        """Clear the symbol table."""
        self.symbols.clear()
        logger.debug("Symbol table cleared.")
