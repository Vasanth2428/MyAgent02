"""
Processor - Legacy Compatibility Layer

This file exists to help older code that imports from the old location.
All the processing logic has been moved to dedicated modules:
- src.core.compressor: Handles shortening documents
- src.core.reranker: Handles re-scoring search results  
- src.core.expander: Handles creating query variations
- src.core.hyde: Handles generating hypothetical answers

For new code, import directly from those modules instead.
"""

# Backward-compatible re-exports
from src.core.compressor import Compressor
from src.core.reranker import NeuralReranker
from src.core.expander import QueryExpander
from src.core.hyde import HyDEGenerator

__all__ = ["Compressor", "NeuralReranker", "QueryExpander", "HyDEGenerator"]
