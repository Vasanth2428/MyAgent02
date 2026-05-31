"""
================================================================================
RAG CONTEXT ENGINE - CONFIGURATION
================================================================================
Centralized constants and tuning parameters for the entire pipeline.
Change values here instead of hunting through multiple files.
"""

# --- LLM ---
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.1
CONTEXT_WINDOW_LIMIT = 8192

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
TOTAL_CONTEXT_BUDGET = 1500       # Total tokens for memory + knowledge
MEMORY_TOKEN_BUDGET = 300         # Max tokens for conversation memory
MIN_KNOWLEDGE_BUDGET = 300        # Floor for knowledge even if memory is large
TOKENIZER_ENCODING = "cl100k_base"

# --- Compression ---
COMPRESSION_SCORE_THRESHOLD = 0.02
SAFETY_CHAR_LIMIT = 16000        # Hard character limit for Simple RAG mode

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- Memory ---
MEMORY_DECAY_RATE = 0.1
MEMORY_WEIGHT_THRESHOLD = 0.1

# --- Semantic Deduplication ---
SEMANTIC_DEDUP_THRESHOLD = 0.85  # Cosine similarity threshold for embedding-based dedup
SEMANTIC_DEDUP_MIN_WORDS = 4     # Minimum words before semantic dedup is used

# --- Database ---
DB_PATH = "memory.db"
HISTORY_LIMIT = 10

# --- HyDE ---
HYDE_MAX_TOKENS = 150
HYDE_TEMPERATURE = 0.3

# --- Query Expansion ---
EXPANSION_MIN_WORDS = 5           # Queries shorter than this skip expansion

# --- Cost (Groq Llama 3.1 8B Pricing) ---
COST_PER_INPUT_TOKEN = 0.05 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.08 / 1_000_000

# --- Pipeline Feature Flags ---
# These control conditional execution of expensive pipeline stages
# Features are triggered based on retrieval confidence or mode settings
ENABLE_HYDE = True              # Hypothetical Document Embeddings
ENABLE_QUERY_EXPANSION = True   # Semantic query expansion
ENABLE_RERANKING = True         # Cross-encoder reranking
ENABLE_COMPRESSION = True       # Context compression

# Confidence thresholds for conditional feature execution
# If top result score is above threshold, skip expensive features
LOW_CONFIDENCE_THRESHOLD = 0.3    # Below this, use full pipeline
MEDIUM_CONFIDENCE_THRESHOLD = 0.5 # Between low and high, use partial pipeline

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PipelineConfig:
    """Configuration for conditional pipeline feature execution."""
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
        """Determine if full pipeline is needed based on retrieval confidence."""
        return top_score < self.low_confidence_threshold

    def should_use_partial_pipeline(self, top_score: float) -> bool:
        """Determine if partial pipeline features are needed."""
        return self.low_confidence_threshold <= top_score < self.medium_confidence_threshold