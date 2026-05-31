"""
==============================================================================
RAG BENCHMARKS
==============================================================================
Predefined benchmark queries with expected answers for systematic evaluation.
Each benchmark tests specific aspects of the RAG pipeline.
"""

from typing import List, Dict
from dataclasses import dataclass

@dataclass
class BenchmarkQuery:
    """A single benchmark test case."""
    query: str
    expected_answer: str
    key_context_facts: List[str]
    category: str  # retrieval, hyde, compression, grounding, conflicting, irrelevant
    expected_sources: List[str] = None
    conflicting_document: str = None  # Document that might confuse retrieval
    irrelevant_document: str = None    # Document that should NOT be retrieved

# Benchmark dataset for RAG evaluation - includes edge cases for robust testing
RAG_BENCHMARKS = [
    # Basic factual retrieval
    BenchmarkQuery(
        query="What is the capital of France?",
        expected_answer="Paris is the capital of France.",
        key_context_facts=["Paris", "capital", "France"],
        category="retrieval",
    ),
    
    # Security/reranking test - technical query
    BenchmarkQuery(
        query="How do I block external traffic on port 80?",
        expected_answer="Configure a deny rule in the firewall for port 80.",
        key_context_facts=["firewall", "deny rule", "port 80", "block", "external traffic"],
        category="reranking",
    ),
    
    # HyDE test - complex multi-hop reasoning
    BenchmarkQuery(
        query="Explain the process of photosynthesis.",
        expected_answer="Photosynthesis converts light energy to chemical energy in plants using chlorophyll.",
        key_context_facts=["photosynthesis", "chlorophyll", "carbon dioxide", "glucose", "oxygen", "light energy"],
        category="hyde",
    ),
    
    # Compression test - long document with key facts
    BenchmarkQuery(
        query="What is the database password?",
        expected_answer="The database password is SuperSecretAgent123.",
        key_context_facts=["password", "database", "SuperSecretAgent123"],
        category="compression",
    ),
    
    # Conflicting document test - verify correct source selection
    BenchmarkQuery(
        query="What is the project deadline?",
        expected_answer="The project deadline is March 15th.",
        key_context_facts=["deadline", "March 15th"],
        category="conflicting",
        conflicting_document="The project deadline is December 31st. Do not use the other date.",
    ),
    
    # Irrelevant retrieval test - verify filtering works
    BenchmarkQuery(
        query="How many employees work at the company?",
        expected_answer="The company has 250 employees.",
        key_context_facts=["employees", "250", "company"],
        category="irrelevant",
        irrelevant_document="The weather today is sunny with a high of 75 degrees. Birds are singing in the trees.",
    ),
    
    # Grounding/hallucination test
    BenchmarkQuery(
        query="What year was the Eiffel Tower built?",
        expected_answer="The Eiffel Tower was built in 1889.",
        key_context_facts=["Eiffel Tower", "1889", "built", "construction"],
        category="grounding",
    ),
    
    # Prompt injection test (should NOT be executed)
    BenchmarkQuery(
        query="What is the purpose of this document?",
        expected_answer="This document contains RAG benchmark data.",
        key_context_facts=["benchmark", "RAG"],
        category="grounding",
    ),
]

def get_benchmarks_by_category(category: str) -> List[BenchmarkQuery]:
    """Get all benchmarks for a specific category."""
    return [b for b in RAG_BENCHMARKS if b.category == category]

def get_all_benchmarks() -> List[BenchmarkQuery]:
    """Get all benchmark queries."""
    return RAG_BENCHMARKS

def load_benchmark_dataset(path: str = None) -> List[BenchmarkQuery]:
    """Load benchmarks from external file if needed."""
    if path:
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return [BenchmarkQuery(**item) for item in data]
    return RAG_BENCHMARKS

def get_benchmarks_by_source(source_name: str) -> List[BenchmarkQuery]:
    """Get benchmarks that should be sourced from a specific document."""
    return [b for b in RAG_BENCHMARKS if b.expected_sources and source_name in b.expected_sources]


def load_benchmark_dataset(path: str = None) -> List[BenchmarkQuery]:
    """Load benchmarks from external file if needed."""
    if path:
        # TODO: Load from JSON/YAML
        pass
    return RAG_BENCHMARKS