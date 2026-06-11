"""
RAG Configuration - All the Settings in One Place

This file contains all the knobs you can tweak to change how the system works.
Values can be overridden with environment variables for easy testing and tuning
without touching code.

Key settings:
- LLM: Which AI model to use and how creative it should be
- Embedding: Which model converts text to mathematical vectors
- Retrieval: How many results to return and search behavior
- Memory: How long to remember conversation history
- Compression: How aggressively to shorten document content
"""

import os

# --- LLM ---
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.1
CONTEXT_WINDOW_LIMIT = 131072

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Reranker ---
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Retrieval ---
HYBRID_ALPHA_DEFAULT = 0.50
HYBRID_ALPHA_KEYWORD = 0.20
MAX_CANDIDATES = 12
DEFAULT_TOP_K = 5

# --- Token Budgets ---
TOTAL_CONTEXT_BUDGET = int(os.getenv("RAG_TOTAL_CONTEXT_BUDGET", "16384"))
MEMORY_TOKEN_BUDGET = int(os.getenv("RAG_MEMORY_TOKEN_BUDGET", "4096"))
MIN_KNOWLEDGE_BUDGET = int(os.getenv("RAG_MIN_KNOWLEDGE_BUDGET", "8192"))
TOKENIZER_ENCODING = "cl100k_base"

# --- Compression ---
COMPRESSION_SCORE_THRESHOLD = float(os.getenv("RAG_COMPRESSION_SCORE_THRESHOLD", "0.02"))
SAFETY_CHAR_LIMIT = int(os.getenv("RAG_SAFETY_CHAR_LIMIT", "16000"))

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

# --- Memory ---
MEMORY_DECAY_RATE = float(os.getenv("RAG_MEMORY_DECAY_RATE", "0.1"))
MEMORY_WEIGHT_THRESHOLD = float(os.getenv("RAG_MEMORY_WEIGHT_THRESHOLD", "0.1"))

# --- Turn-Based Memory Decay (Issue #10) ---
MEMORY_TURN_DECAY_RATE = float(os.getenv("RAG_MEMORY_TURN_DECAY_RATE", "0.15"))

# --- Semantic Deduplication ---
SEMANTIC_DEDUP_THRESHOLD = float(os.getenv("RAG_SEMANTIC_DEDUP_THRESHOLD", "0.85"))
SEMANTIC_DEDUP_MIN_WORDS = int(os.getenv("RAG_SEMANTIC_DEDUP_MIN_WORDS", "4"))

# --- Database ---
DB_PATH = os.getenv("RAG_DB_PATH", "data/memory.db")
HISTORY_LIMIT = int(os.getenv("RAG_HISTORY_LIMIT", "10"))

# --- HyDE ---
HYDE_MAX_TOKENS = int(os.getenv("RAG_HYDE_MAX_TOKENS", "150"))
HYDE_TEMPERATURE = float(os.getenv("RAG_HYDE_TEMPERATURE", "0.3"))

# --- Retrieval Concurrency (Issue #7: Concurrency Pool Starvation) ---
MAX_CONCURRENT_RETRIEVALS = int(os.getenv("RAG_MAX_CONCURRENT_RETRIEVALS", "3"))

# --- Query Expansion ---
EXPANSION_MIN_WORDS = int(os.getenv("RAG_EXPANSION_MIN_WORDS", "5"))

# --- Cost (Groq Llama 3.1 8B Pricing) ---
COST_PER_INPUT_TOKEN = 0.05 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.08 / 1_000_000

# --- Agent Model Strategy ---
# Primary and fallback models for each agent type
# Development environment: GROQ_CORE_KEY for most agents, GROQ_VALIDATION_KEY for critics
SUPERVISOR_MODEL_PRIMARY = "llama-3.3-70b-versatile"
SUPERVISOR_MODEL_FALLBACK = "llama-3.1-8b-instant"

RAG_WORKER_MODEL_PRIMARY = "llama-3.1-8b-instant"
RAG_WORKER_MODEL_FALLBACK = "llama-3.1-8b-instant"

CODING_WORKER_MODEL_PRIMARY = "llama-3.1-8b-instant"
CODING_WORKER_MODEL_FALLBACK = "llama-3.1-8b-instant"

CODE_CRITIC_MODEL_PRIMARY = "llama-3.1-8b-instant"
CODE_CRITIC_MODEL_FALLBACK = "llama-3.1-8b-instant"

CRITIC_MODEL_PRIMARY = "llama-3.1-8b-instant"
CRITIC_MODEL_FALLBACK = "llama-3.1-8b-instant"

SYNTHESIZER_MODEL_PRIMARY = "llama-3.1-8b-instant"
SYNTHESIZER_MODEL_FALLBACK = "llama-3.1-8b-instant"

REPORT_WORKER_MODEL_PRIMARY = "llama-3.1-8b-instant"
REPORT_WORKER_MODEL_FALLBACK = "llama-3.1-8b-instant"

WEB_WORKER_MODEL_PRIMARY = "llama-3.1-8b-instant"
WEB_WORKER_MODEL_FALLBACK = "llama-3.1-8b-instant"

SCRAPER_WORKER_MODEL_PRIMARY = "llama-3.1-8b-instant"
SCRAPER_WORKER_MODEL_FALLBACK = "llama-3.1-8b-instant"

UTILITY_WORKER_MODEL_PRIMARY = "llama-3.1-8b-instant"
UTILITY_WORKER_MODEL_FALLBACK = "llama-3.1-8b-instant"

# --- Pipeline Feature Flags ---
# These control conditional execution of expensive pipeline stages
# Features are triggered based on retrieval confidence or mode settings
ENABLE_HYDE = os.getenv("RAG_ENABLE_HYDE", "true").lower() == 'true'
ENABLE_QUERY_EXPANSION = os.getenv("RAG_ENABLE_QUERY_EXPANSION", "true").lower() == 'true'
ENABLE_RERANKING = os.getenv("RAG_ENABLE_RERANKING", "true").lower() == 'true'
ENABLE_COMPRESSION = os.getenv("RAG_ENABLE_COMPRESSION", "true").lower() == 'true'

# Confidence thresholds for conditional feature execution
# If top result score is above threshold, skip expensive features
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_LOW_CONF_THRESH", "0.3"))
MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_MED_CONF_THRESH", "0.5"))

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PipelineConfig:
    """
    Controls when to use expensive AI features.
    
    These settings let us skip costly operations (like re-ranking) when we're
    already confident in our search results. This makes simple queries faster.
    
    Attributes control whether to enable:
    - HyDE: Generate hypothetical answers for better search
    - Query expansion: Create multiple search variations
    - Reranking: Re-score results for better accuracy
    - Compression: Shorten documents to fit in context
    """
    enable_hyde: bool = True
    enable_expansion: bool = True
    enable_reranking: bool = True
    enable_compression: bool = True
    low_confidence_threshold: float = 0.3
    medium_confidence_threshold: float = 0.5

    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        import os
        return cls(
            enable_hyde=os.getenv('RAG_ENABLE_HYDE', 'true').lower() == 'true',
            enable_expansion=os.getenv('RAG_ENABLE_EXPANSION', 'true').lower() == 'true',
            enable_reranking=os.getenv('RAG_ENABLE_RERANKING', 'true').lower() == 'true',
            enable_compression=os.getenv('RAG_ENABLE_COMPRESSION', 'true').lower() == 'true',
            low_confidence_threshold=float(os.getenv('RAG_LOW_CONF_THRESH', '0.3')),
            medium_confidence_threshold=float(os.getenv('RAG_MED_CONF_THRESH', '0.5')),
        )

    @classmethod
    def development(cls) -> 'PipelineConfig':
        return cls(enable_hyde=False, enable_expansion=False, enable_reranking=False)

    @classmethod
    def production(cls) -> 'PipelineConfig':
        return cls()

    def should_use_full_pipeline(self, top_score: float) -> bool:
        """Use all features when initial search isn't confident (low score)."""
        return top_score < self.low_confidence_threshold

    def should_use_partial_pipeline(self, top_score: float) -> bool:
        """Use some features when search has moderate confidence."""
        return self.low_confidence_threshold <= top_score < self.medium_confidence_threshold