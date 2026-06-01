"""
Processor - Legacy Compatibility Layer

This file exists to help older code that imports from the old location.
All the processing logic has been moved to dedicated modules:
- core.compressor: Handles shortening documents
- core.reranker: Handles re-scoring search results  
- core.expander: Handles creating query variations
- core.hyde: Handles generating hypothetical answers

For new code, import directly from those modules instead.
"""

# Backward-compatible re-exports
from core.compressor import Compressor
from core.reranker import NeuralReranker
from core.expander import QueryExpander
from core.hyde import HyDEGenerator

__all__ = ["Compressor", "NeuralReranker", "QueryExpander", "HyDEGenerator"]
