"""
RAG Context Engine - Core Package

This package contains the main components of the RAG system. Each module handles
a specific part of answering questions from your documents:

- RAGContextEngine: The main coordinator that brings everything together
- WeaviateRetriever: Stores and searches your document database
- ConversationMemory: Remembers what's been discussed
- PersistentMemoryStore: Saves chat history to disk
- Compressor: Shortens documents to fit in the AI's context
- NeuralReranker: Re-scores search results for better accuracy
- QueryExpander: Creates alternative search queries
- HyDEGenerator: Generates hypothetical answers to improve search
- RecursiveCharacterSplitter: Breaks documents into searchable chunks

Code Intelligence Layer (for repository analysis and security auditing):
- CodeIndexer: Scans and indexes repository AST structures
- SymbolTable: In-memory symbol lookup for functions, classes, methods
- DependencyGraph: Tracks imports and call relationships
- CodeRetrievalService: Repository-aware code search and analysis
"""

from src.core.engine import RAGContextEngine
from src.core.retriever import WeaviateRetriever
from src.core.memory import ConversationMemory, MemoryEntry
from src.core.persistence import PersistentMemoryStore
from src.core.compressor import Compressor
from src.core.reranker import NeuralReranker
from src.core.expander import QueryExpander
from src.core.hyde import HyDEGenerator
from src.core.splitter import RecursiveCharacterSplitter
from src.core.code.indexer import CodeIndexer
from src.core.code.symbol_table import SymbolTable
from src.core.code.dependency_graph import DependencyGraph

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
    "CodeIndexer",
    "SymbolTable",
    "DependencyGraph",
]
