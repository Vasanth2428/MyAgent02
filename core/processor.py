"""
================================================================================
RAG CONTEXT ENGINE - PROCESSOR (LEGACY COMPATIBILITY)
================================================================================
This module re-exports the classes that were extracted into dedicated modules.
Import directly from core.compressor, core.reranker, core.expander, core.hyde
for new code.
"""

# Backward-compatible re-exports
from core.compressor import Compressor
from core.reranker import NeuralReranker
from core.expander import QueryExpander
from core.hyde import HyDEGenerator

__all__ = ["Compressor", "NeuralReranker", "QueryExpander", "HyDEGenerator"]
