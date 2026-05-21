"""
RAG Context Engine - Core Package
==================================
Public API surface for the core pipeline modules.
"""

from core.engine import RAGContextEngine
from core.retriever import WeaviateRetriever
from core.memory import ConversationMemory, MemoryEntry
from core.persistence import PersistentMemoryStore
from core.compressor import Compressor
from core.reranker import NeuralReranker
from core.expander import QueryExpander
from core.hyde import HyDEGenerator
from core.splitter import RecursiveCharacterSplitter

__all__ = [
    "RAGContextEngine",
    "WeaviateRetriever",
    "ConversationMemory",
    "MemoryEntry",
    "PersistentMemoryStore",
    "Compressor",
    "NeuralReranker",
    "QueryExpander",
    "HyDEGenerator",
    "RecursiveCharacterSplitter",
]
