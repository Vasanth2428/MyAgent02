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
