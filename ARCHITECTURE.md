# RAG Context Engine — System Architecture

A production-grade Retrieval-Augmented Generation system with intelligent pipeline routing, grounding verification, agentic reasoning, and security hardening.

> For detailed rationale behind each design choice, see [DESIGN_AND_TRADEOFFS.md](DESIGN_AND_TRADEOFFS.md).

---

## Table of Contents

1. [Overview](#overview)
2. [File Map](#file-map)
3. [High-Level Architecture](#high-level-architecture)
4. [Entry Points — API Layer](#entry-points--api-layer)
5. [Orchestration — Engine](#orchestration--engine)
6. [Retrieval Pipeline Components](#retrieval-pipeline-components)
7. [Service Layer](#service-layer)
8. [Memory Subsystem](#memory-subsystem)
9. [Security Layer](#security-layer)
10. [Agentic Mode — ReAct + LangGraph](#agentic-mode--react--langgraph)
11. [LLM Client Architecture](#llm-client-architecture)
12. [Evaluation & Benchmarking](#evaluation--benchmarking)
13. [Configuration System](#configuration-system)
14. [Async Pipeline Flow](#async-pipeline-flow)
15. [Key Design Patterns](#key-design-patterns)
16. [Testing Structure](#testing-structure)

---

## Overview

The RAG Context Engine processes user queries through a multi-stage pipeline:

1. **Expand** — Generate semantic query variations and hypothetical documents
2. **Retrieve** — Hybrid search (vector + BM25) against a Weaviate vector database
3. **Refine** — Rerank, compress, and merge with conversation memory
4. **Generate** — Produce grounded answers via LLM with hallucination verification
5. **Agent** — (Optional) Multi-step reasoning with tool use via a ReAct state machine

The system supports three query modes:
- **`context_engine`** — Full RAG pipeline with all optimizations
- **`normal`** — Simplified retrieval without expansion, HyDE, or reranking
- **`agentic`** — ReAct reasoning loop with web search, scraping, and calculator tools

---

## File Map

```
RAG/
├── main.py                          # FastAPI app, endpoints, lifespan
├── index.html                       # Frontend UI dashboard
├── ARCHITECTURE.md                  # This document
├── DESIGN_AND_TRADEOFFS.md          # Design rationale & trade-offs
├── requirements.txt                 # Python dependencies
├── memory.db                        # SQLite conversation persistence
│
├── core/
│   ├── __init__.py                  # Public API surface exports
│   ├── engine.py                    # Central orchestrator (847 lines)
│   ├── config.py                    # All tunable parameters & PipelineConfig
│   ├── llm.py                       # Centralized LLM client with retry wrappers
│   │
│   ├── retriever.py                 # Weaviate hybrid search & indexing
│   ├── compressor.py                # Extractive context compression
│   ├── reranker.py                  # Cross-encoder neural reranking
│   ├── expander.py                  # LLM-based query expansion
│   ├── hyde.py                      # Hypothetical Document Embeddings
│   ├── splitter.py                  # Recursive character text splitting
│   │
│   ├── memory.py                    # Temporal decay conversation memory
│   ├── persistence.py               # SQLite-backed history storage
│   │
│   ├── security.py                  # Prompt injection sanitization
│   ├── scraper.py                   # SSRF-hardened web scraper
│   ├── tools.py                     # AST-safe calculator & time utility
│   ├── retry.py                     # Unified retry decorator (sync/async)
│   │
│   ├── agent.py                     # ReAct agent controller
│   ├── graph.py                     # LangGraph state machine (919 lines)
│   ├── registry.py                  # Knowledge base source registry
│   │
│   ├── evaluator.py                 # RAG evaluation framework
│   ├── benchmarks.py                # Predefined benchmark test cases
│   ├── processor.py                 # Legacy re-exports (backward compat)
│   │
│   └── services/
│       ├── __init__.py              # Service exports
│       ├── generation_service.py    # LLM generation + grounding check
│       ├── grounding_service.py     # Hallucination detection & verification
│       ├── retrieval_service.py     # Parallel multi-query retrieval
│       ├── memory_service.py        # Memory lifecycle management
│       ├── overflow_service.py      # Context window overflow recovery
│       └── telemetry_service.py     # Performance metrics tracking
│
├── static/                          # CSS/JS assets for the UI
├── tests/                           # Unit, integration, stress, diagnostic tests
└── logs/                            # Rotating application logs
```

---

## High-Level Architecture

```
                              ┌──────────────────────────────────┐
                              │       main.py (FastAPI)          │
                              │  /query  /query_stream  /upload  │
                              │  /stats  /history/{id}   /       │
                              └──────────────┬───────────────────┘
                                             │
                    ┌────────────────────────▼─────────────────────────┐
                    │           core/engine.py (RAGContextEngine)      │
                    │     Central orchestrator — routes all queries    │
                    └────────┬─────────┬──────────┬──────────┬────────┘
                             │         │          │          │
              ┌──────────────▼──┐  ┌───▼────┐  ┌──▼───┐  ┌──▼───────────┐
              │  Retrieval      │  │ Refine │  │ Gen  │  │  Agent       │
              │  Pipeline       │  │ Layer  │  │ Svc  │  │  (ReAct +    │
              │  ┌───────────┐  │  │        │  │      │  │   LangGraph) │
              │  │ Expander  │  │  │Reranker│  │LLM + │  │  ┌────────┐ │
              │  │ HyDE      │  │  │Compress│  │Ground│  │  │ Tools  │ │
              │  │ Retriever │  │  │Memory  │  │Verify│  │  │ Search │ │
              │  └───────────┘  │  │Overflow│  │      │  │  │ Scrape │ │
              └─────────────────┘  └────────┘  └──────┘  │  └────────┘ │
                                                          └────────────┘
```

---

## Entry Points — API Layer

### `main.py`

FastAPI application with lifespan-managed startup/shutdown.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the UI dashboard (`index.html`) |
| `/query` | POST | Synchronous question answering |
| `/query_stream` | POST | Server-Sent Events (SSE) streaming responses |
| `/upload` | POST | Document ingestion (`.pdf`, `.txt`) → chunk → index |
| `/stats` | GET | System performance metrics (CPU, RAM, doc count) |
| `/history/{session_id}` | GET | Retrieve conversation history for a session |

**Request schema** (`QueryRequest`):
```python
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    mode: Literal["context_engine", "normal", "agentic"] = "context_engine"
    source_filter: Optional[str] = None
    context_limit: Optional[int] = None
```

**Lifespan flow**:
1. Startup → Initialize `WeaviateRetriever` → Load `PipelineConfig.from_env()` → Create `RAGContextEngine`
2. Log Knowledge Registry summary (sources, domains, document count)
3. Shutdown → Close Weaviate connection → Close aiohttp session

---

## Orchestration — Engine

### `core/engine.py` — RAGContextEngine

The central brain of the system. All queries — sync, async, streaming, agentic — flow through this class.

**Initialization** creates all sub-components:
```
RAGContextEngine.__init__(retriever, pipeline_config)
    ├── WeaviateRetriever        (injected)
    ├── PipelineConfig           (injected or default)
    ├── PersistentMemoryStore    (SQLite)
    ├── Compressor               (extractive)
    ├── NeuralReranker           (cross-encoder, lazy-loaded)
    ├── KnowledgeRegistry        (source introspection)
    ├── LLMService               (centralized Groq client)
    │   ├── RobustLLMClient      (sync with retry)
    │   └── RobustAsyncLLMClient (async with retry)
    ├── GenerationService        (prompt building + generation)
    ├── QueryExpander            (LLM-based expansion)
    ├── HyDEGenerator            (hypothetical documents)
    ├── RetrievalService         (parallel multi-query search)
    ├── MemoryService            (lifecycle management)
    ├── ContextOverflowService   (budget recovery)
    ├── TelemetryService         (metrics tracking)
    └── RAGAgent                 (ReAct agentic mode)
        └── RAGLangGraph         (state machine)
```

**Public API**:

| Method | Type | Description |
|---|---|---|
| `ask(query, ...)` | Sync | Wraps `ask_async` via `asyncio.run()` |
| `ask_async(query, ...)` | Async | Primary query entry point |
| `ask_stream(query, ...)` | Sync generator | Wraps `ask_stream_async` |
| `ask_stream_async(query, ...)` | Async generator | Streaming with SSE events |

**Registry query detection**: Before running the full pipeline, the engine checks if the query asks about available documents (e.g., "what files do you have?") and returns a registry listing directly — skipping retrieval entirely.

---

## Retrieval Pipeline Components

### `core/retriever.py` — WeaviateRetriever

Manages the Weaviate Cloud vector database connection and search operations.

- **Hybrid search**: `α × VectorSimilarity + (1-α) × BM25`
- **Dynamic alpha**: Technical/code queries shift to keyword-heavy search (`α = 0.20`), general queries use balanced (`α = 0.50`)
- **Deterministic UUIDs**: `uuid5(NAMESPACE_DNS, text)` prevents duplicate indexing
- **Shared embedding model**: Uses the singleton from `grounding_service._get_shared_embedding_model()`
- **Retry wrapper**: All database operations wrapped with transient error retry logic
- **Schema**: `RAGKnowledge` collection with properties: `text`, `tags`, `source`, `content_hash`, `upload_timestamp`, `document_id`

### `core/expander.py` — QueryExpander

Generates 3 diverse search variations of the user's query using the LLM.

- Returns `[original_query, variation_1, variation_2, variation_3]`
- Uses JSON response format for reliable parsing
- Skipped for queries under `EXPANSION_MIN_WORDS` (5 words)
- Both sync (`expand`) and async (`expand_async`) methods

### `core/hyde.py` — HyDEGenerator

Creates hypothetical answer documents to improve retrieval alignment.

- Generates a brief paragraph answering the query (max 150 tokens, temp 0.3)
- The hypothetical document is embedded and used as an additional search query
- Only activated on low-confidence queries (< 0.3 threshold)
- Runs concurrently with query expansion via `asyncio.create_task()`

### `core/compressor.py` — Compressor

Extractive context compression to fit relevant content into the token budget.

- **Segment splitting**: Documents → paragraphs + intact code blocks (never splits mid-code-block)
- **Scoring**: Lexical overlap between query words and segment words
- **Selection**: Greedy top-K by score until token budget exhausted
- **Ordering**: Selected segments reassembled in original document order
- **Fast path**: If total tokens already under budget, returns text unchanged
- Minimum score threshold `COMPRESSION_SCORE_THRESHOLD = 0.02` filters noise

### `core/reranker.py` — NeuralReranker

Cross-encoder model for deep semantic relevance scoring.

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Lazy singleton**: Loaded on first use via `_get_cross_encoder()`, not at startup
- **Score normalization**: Raw logits → sigmoid → `[0, 1]` range
- **Async support**: `rerank_async()` runs model prediction in a thread pool
- Conditionally skipped when retrieval confidence is already high (score > 0.5)

### `core/splitter.py` — RecursiveCharacterSplitter

Splits uploaded documents into indexable chunks.

- **Separator priority**: `\n\n` → `\n` → `. ` → ` ` → `""` (empty = hard break)
- **Overlap**: Configurable overlap (default 100 chars) to preserve cross-boundary context
- **Chunk size**: Default 1000 characters per chunk
- Used in the `/upload` endpoint during document ingestion

---

## Service Layer

### `core/services/generation_service.py` — GenerationService

Builds LLM prompts, generates answers, and runs grounding verification.

- **Prompt structure**: `SYSTEM_INSTRUCTIONS → CONTEXT (memory + knowledge) → QUESTION → ANSWER`
- **Security prompt**: Explicitly warns the LLM that context is untrusted data
- **Grounding check**: After generation, each sentence is verified against context chunks
- **Return type**: `GenerationResult` dataclass with response, token usage, grounding score, unsupported claims
- **Thread-safe verifier**: Shared `GroundingVerifier` singleton via double-checked locking
- Methods: `generate()`, `generate_async()`, `generate_stream()`, `generate_stream_async()`

### `core/services/grounding_service.py` — GroundingVerifier

Verifies that LLM answers are supported by retrieved context. Detects hallucinations.

- **Sentence-level verification**: Each answer sentence is embedded and compared against all context chunks via cosine similarity
- **Grounding score formula**: `0.6 × (supported_ratio) + 0.4 × (avg_max_similarity)`
- **Hallucination detection**: Flags sentences with absolute claims, superlatives, or specific year references when unsupported
- **Citation extraction**: Parses `[source: ...]` markers from answers
- **Shared embedding model**: Thread-safe lazy singleton used across all components:
  ```python
  _embedding_model_instance = None
  _embedding_model_lock = threading.Lock()

  def _get_shared_embedding_model():
      if _embedding_model_instance is None:
          with _embedding_model_lock:
              if _embedding_model_instance is None:
                  _embedding_model_instance = SentenceTransformer("all-MiniLM-L6-v2")
      return _embedding_model_instance
  ```

### `core/services/retrieval_service.py` — RetrievalService

Orchestrates parallel search across multiple query variations.

- Accepts a list of search queries (original + expansions + HyDE)
- Runs parallel retrieval for each query
- Deduplicates results by text content
- Returns `(results, embed_latency_ms, db_search_latency_ms, total_ms)`

### `core/services/memory_service.py` — MemoryService

Manages the lifecycle of conversation memory per session.

- Creates `ConversationMemory` instances on demand
- Persists turns to SQLite via `PersistentMemoryStore`
- Restores history from database on first access per session

### `core/services/overflow_service.py` — ContextOverflowService

Handles context window overflow when total tokens exceed a user-specified limit.

Three-phase recovery cascade:
```
Phase 1: 🧹 Memory Pruning
  → Evict oldest memory turns until under budget

Phase 2: 🗜️ Aggressive Re-compression
  → Re-compress knowledge chunks with tighter token budget

Phase 3: ✂️ Hard Truncation
  → Tokenize → slice → decode (last resort)

✅ Recovery complete
```

### `core/services/telemetry_service.py` — TelemetryService

Tracks performance metrics across the pipeline.

- Running averages: latency, compression ratio
- Query counter
- System metrics via `psutil`: CPU %, RAM %
- Cost computation based on Groq token pricing

---

## Memory Subsystem

### `core/memory.py` — ConversationMemory

Short-term conversational context with intelligent management.

- **Temporal decay**: `Weight = Importance × e^(-DecayRate × HoursElapsed)`
- **Semantic deduplication**: Embedding cosine similarity (threshold 0.85) for long texts, Jaccard overlap (threshold 0.70) for short texts
- **Token budget**: Top-weighted entries selected until `MEMORY_TOKEN_BUDGET` (300) reached
- **Chronological output**: After weight-based selection, entries re-sorted by original order
- **Lazy embeddings**: `MemoryEntry.embedding` computed on first access and cached

### `core/persistence.py` — PersistentMemoryStore

SQLite-backed storage for conversation history across server restarts.

- **WAL journaling**: `PRAGMA journal_mode=WAL` for concurrent read performance
- **Retry logic**: Handles `OperationalError` (locked/busy) with exponential backoff
- **Schema migration**: Runtime detection and addition of the `telemetry` column
- **Index**: `(session_id, timestamp)` for efficient history retrieval
- **CRUD**: `add_entry()`, `get_history(session_id, limit)`

---

## Security Layer

### `core/security.py` — sanitize_document_text

Multi-pattern prompt injection defense applied to all retrieved document text.

```
Layer 1: XML/HTML Escaping
  < → &lt;   > → &gt;

Layer 2: Pattern Matching (25+ regex patterns)
  ├── Instruction overrides   ("ignore previous instructions")
  ├── System prompt revelation ("reveal your system prompt")
  ├── Role manipulation        ("pretend you are", "act as")
  ├── Jailbreak attempts       ("DAN mode", "developer mode")
  ├── Context-based injection  ("In this document, you must...")
  └── Template injection       ({{...}}, {%...%})

Layer 3: Replacement
  All matches → [CLEANED INSTRUCTION DETECTED]
```

### `core/scraper.py` — Web Scraper

SSRF-hardened web page scraping with both sync and async implementations.

- **SSRF protection**: DNS resolution → check all resolved IPs against private/loopback/link-local ranges
- **Custom HTML parser**: `HTMLTextExtractor` using stdlib `HTMLParser` (zero dependencies)
- **Tag filtering**: Ignores `script`, `style`, `head`, `title`, `meta`, `link`, `noscript`
- **Connection pooling**: Shared `aiohttp.ClientSession` singleton for async requests
- **Concurrent scraping**: `scrape_multiple_pages_async()` with semaphore-controlled concurrency
- **Truncation**: Content capped at `max_chars` (default 6000)

### `core/tools.py` — SecureEvaluator

AST-based safe mathematical expression evaluation.

- Parses expressions into an AST and walks only allowed node types
- **Whitelist**: `Num`, `Constant`, `BinOp` (`+`, `-`, `*`, `/`, `//`, `%`, `**`), `UnaryOp` (`-`, `+`)
- **DoS protection**: Exponentiation capped at `base > 10000` or `exponent > 100`
- `get_current_time()`: Returns formatted local datetime string

---

## Agentic Mode — ReAct + LangGraph

### `core/agent.py` — RAGAgent

Controller for the agentic reasoning mode.

- Parses `Action: tool_name[arguments]` from LLM responses via regex
- Delegates execution to `RAGLangGraph` state machine
- Supports both sync (`run_stream`) and async (`run_stream_async`) interfaces
- Maximum 3 reasoning iterations per query

### `core/graph.py` — RAGLangGraph

LangGraph-based state machine implementing the ReAct reasoning loop.

```
┌──────────────┐     ┌─────────────────┐     ┌─────────────┐
│ early_exit   │────→│ overflow        │────→│ reasoning   │◀──┐
│ _check       │     │ _recovery       │     │             │   │
└──────┬───────┘     └─────────────────┘     └──────┬──────┘   │
       │                                            │          │
  exit detected?                           ┌────────┼────────┐ │
       │                                   │        │        │ │
       ▼                              has action  has FA  format│
┌──────────────┐                     ┌─────▼─────┐  │   error │
│ early_exit   │                     │ execute   │  │  ┌──▼──┐│
│ _execute     │                     │ _tool     │──┘  │fmt  ││
└──────────────┘                     └───────────┘     │error│┘
       │                                               └─────┘
       ▼                              ┌──────────────────┐
      END                             │ streaming_final  │
                                      │ _answer          │
       ┌──────────────┐               └──────────────────┘
       │ synthesis    │                       │
       │ (fallback)   │                      END
       └──────────────┘
              │
             END
```

**State schema** (`AgentState`):
```python
class AgentState(TypedDict):
    query: str
    session_id: str
    context_limit: Optional[int]
    source_filter: Optional[str]
    memory_text: str
    scratchpad: str           # Accumulated Thought/Action/Observation log
    iteration: int            # Current reasoning step (max 3)
    llm_call_count: int
    goals_set: List[str]
    actions_taken: List[dict]
    final_response: str
    overflow_occurred: bool
    overflow_steps: List[str]
    retrieved_context: List[dict]
    events_queue: List[dict]  # SSE events to stream to client
    early_exit_type: Optional[str]  # "greeting" or "registry"
    parsed_action: Optional[tuple]
    is_direct: bool
    raw_response: str
    initial_tokens: int
    final_tokens: int
    search_cache: Dict[str, str]  # Deduplicates repeated searches
```

**Available tools**:

| Tool | Arguments | Description |
|---|---|---|
| `web_search` | query string | Mock web search (deterministic responses) |
| `web_scrape` | URL | Async web page scraping with compression |
| `get_system_stats` | none | CPU, RAM, indexed document count |
| `get_registry` | none | List all indexed sources |
| `get_current_time` | none | Current local datetime |
| `calculator` | expression | AST-safe math evaluation |
| `direct_response` | text | Direct reply without tools |

**Security in agentic mode**: Tool observations are sanitized through `sanitize_document_text()` before appending to the scratchpad. The system prompt explicitly warns the LLM to ignore instructions found in observation data.

### `core/registry.py` — KnowledgeRegistry

Provides introspection into the indexed knowledge base.

- `get_sources()`: Unique source names from Weaviate
- `get_document_domains()`: Inferred domains (documentation, sales_analytics, database_schema, etc.)
- `get_available_schemas()`: Hardcoded schema definitions for known datasets
- `get_topics()`: Unique tags/topics from document metadata
- `get_registry_summary()`: Consolidated summary dict

---

## LLM Client Architecture

### `core/llm.py` — LLMService

Centralized LLM wrapper providing a unified interface with transparent retry logic.

```
Application Code
      │
      ▼
RobustLLMClient ─────────────────→ RobustChat ─────→ RobustCompletions
      │  (sync)                        │                    │
      │                                │              .create() calls
      │                                │              llm_service.execute_with_retry()
      │                                │                    │
      ▼                                ▼                    ▼
RobustAsyncLLMClient ────────→ RobustAsyncChat ──→ RobustAsyncCompletions
      (async)                                            │
                                                   .create() calls
                                                   llm_service.execute_with_retry_async()
```

- **Provider**: Groq (`Groq` / `AsyncGroq` clients)
- **Model**: `llama-3.1-8b-instant` (configurable)
- **Retry**: 5 attempts, exponential backoff (1s base), 0.5s jitter
- **Transient errors**: `RateLimitError`, `APIConnectionError`, `InternalServerError`, `APITimeoutError`
- **Convenience methods**: `complete()`, `complete_async()`, `complete_text()`, `complete_text_async()`
- **Backward compatibility**: `raw_client` property exposes the underlying Groq client

### `core/retry.py` — Unified Retry Decorator

Generic retry utility supporting both sync and async functions.

```python
@retry(retries=5, backoff=1.0, jitter=0.5, is_transient_fn=custom_check)
async def my_operation():
    ...
```

- **Auto-detection**: Inspects `inspect.iscoroutinefunction()` to choose sync/async wrapper
- **Backoff**: `delay = backoff × 2^(attempt-1)` (exponential)
- **Jitter options**: `bool` (10% of delay), `float` (0 to value), `tuple` (min, max)
- **Custom transience**: `is_transient_fn` callback determines which errors are retryable

Used by: LLM calls (5 retries), Weaviate operations (3 retries), SQLite operations (5 retries), LangGraph LLM calls (3 retries).

---

## Evaluation & Benchmarking

### `core/evaluator.py` — RAGEvaluator

Systematic evaluation framework for pipeline components.

**Metric dataclasses**:

| Dataclass | Measures |
|---|---|
| `RetrievalMetrics` | MRR, recall@K, precision@K |
| `RerankingMetrics` | MRR improvement, score delta, correct ranking |
| `HyDEMetrics` | Baseline vs. HyDE recall improvement |
| `CompressionMetrics` | Facts preserved, noise dropped, compression ratio |
| `MemoryMetrics` | Entries surviving decay, deduplication effectiveness |
| `GroundingMetrics` | Grounding score, citations found, hallucinations detected |

**Evaluation methods**: `evaluate_retrieval()`, `evaluate_reranking()`, `evaluate_hyde()`, `evaluate_compression()`, `evaluate_memory()`, `evaluate_grounding()`, `run_full_evaluation()`

### `core/benchmarks.py` — BenchmarkQuery

Predefined test cases covering pipeline edge cases.

| Category | Tests |
|---|---|
| `retrieval` | Basic factual retrieval |
| `reranking` | Technical query needing precise keyword matching |
| `hyde` | Complex multi-hop reasoning benefiting from hypothetical docs |
| `compression` | Key fact preservation under tight budgets |
| `conflicting` | Correct source selection when contradictory docs exist |
| `irrelevant` | Filtering out noise documents |
| `grounding` | Hallucination detection and citation accuracy |

---

## Configuration System

### `core/config.py`

All tunable values centralized in one file. Environment variables override defaults.

| Category | Variable | Default | Purpose |
|---|---|---|---|
| **LLM** | `LLM_MODEL` | `llama-3.1-8b-instant` | Model identifier |
| | `LLM_TEMPERATURE` | `0.1` | Generation randomness |
| | `CONTEXT_WINDOW_LIMIT` | `8192` | Max context window tokens |
| **Embedding** | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name |
| **Reranker** | `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| **Retrieval** | `HYBRID_ALPHA_DEFAULT` | `0.50` | Default vector/BM25 balance |
| | `HYBRID_ALPHA_KEYWORD` | `0.20` | Alpha for technical queries |
| | `MAX_CANDIDATES` | `12` | Max retrieval candidates |
| | `DEFAULT_TOP_K` | `5` | Default top-K results |
| **Budget** | `RAG_TOTAL_CONTEXT_BUDGET` | `1500` | Max tokens: memory + knowledge |
| | `RAG_MEMORY_TOKEN_BUDGET` | `300` | Conversation memory limit |
| | `RAG_MIN_KNOWLEDGE_BUDGET` | `300` | Minimum knowledge allocation |
| **Compression** | `RAG_COMPRESSION_SCORE_THRESHOLD` | `0.02` | Min score to keep a segment |
| | `RAG_SAFETY_CHAR_LIMIT` | `16000` | Hard character safety limit |
| **Chunking** | `RAG_CHUNK_SIZE` | `1000` | Characters per chunk |
| | `RAG_CHUNK_OVERLAP` | `100` | Overlap between chunks |
| **Memory** | `RAG_MEMORY_DECAY_RATE` | `0.1` | Temporal decay speed |
| | `RAG_MEMORY_WEIGHT_THRESHOLD` | `0.1` | Minimum weight to keep entry |
| **Dedup** | `RAG_SEMANTIC_DEDUP_THRESHOLD` | `0.85` | Cosine similarity threshold |
| **HyDE** | `RAG_HYDE_MAX_TOKENS` | `150` | Max tokens for hypothetical doc |
| | `RAG_HYDE_TEMPERATURE` | `0.3` | HyDE generation temperature |
| **Pipeline** | `RAG_ENABLE_HYDE` | `true` | Enable/disable HyDE |
| | `RAG_ENABLE_QUERY_EXPANSION` | `true` | Enable/disable expansion |
| | `RAG_ENABLE_RERANKING` | `true` | Enable/disable reranking |
| | `RAG_ENABLE_COMPRESSION` | `true` | Enable/disable compression |
| | `RAG_LOW_CONF_THRESH` | `0.3` | Full pipeline threshold |
| | `RAG_MED_CONF_THRESH` | `0.5` | Partial pipeline threshold |

**PipelineConfig** dataclass with factory methods:
- `PipelineConfig.from_env()` — Load from environment variables
- `PipelineConfig.development()` — Disable expensive features for fast iteration
- `PipelineConfig.production()` — All features enabled

---

## Async Pipeline Flow

### Context Engine Mode (`ask_async`)

```
ask_async(query, session_id, mode, source_filter, top_k, context_limit)
    │
    ├── Registry query? ─── YES ──→ Return registry listing (early exit)
    │
    ├── Initial quick retrieval (top-1) to assess confidence
    │       │
    │       ├── score < 0.3 (low confidence)
    │       │       ├──→ asyncio.create_task(_phase_expand_async)  ─┐
    │       │       └──→ asyncio.create_task(_phase_hyde_async)    ─┤ concurrent
    │       │                                                       │
    │       └── score ≥ 0.3 (high confidence)                      │
    │               └── Skip expansion & HyDE                      │
    │                                                               │
    ├──→ _phase_retrieve_async()                                   │
    │       └── Parallel hybrid search across all queries ◀────────┘
    │
    ├──→ _phase_refine_async()
    │       ├──→ Reranking    (if enabled AND score < 0.5)
    │       ├──→ Memory sync  (temporal decay + token budget)
    │       └──→ Compression  (if enabled AND score < 0.7)
    │
    ├──→ overflow_service.handle_context_overflow_async()
    │       └── 3-phase recovery if context_limit exceeded
    │
    └──→ _phase_generate_async()
            ├── LLM completion
            └── Grounding verification (async in thread pool)
```

### Agentic Mode (`ask_stream_async` with mode="agentic")

```
ask_stream_async(query, mode="agentic")
    │
    └──→ agent.run_stream_async()
            │
            └──→ graph.compiled_graph.astream(initial_state)
                    │
                    ├── early_exit_check ──→ greeting/registry? ──→ early_exit_execute ──→ END
                    │
                    ├── overflow_recovery (prune memory if over limit)
                    │
                    └── reasoning ◀──────────────────────────────────┐
                            │                                        │
                            ├── Final Answer found ──→ streaming_final_answer ──→ END
                            ├── Action parsed ──→ execute_tool ──────┘
                            ├── Format error ──→ execute_formatting_error ──→ (retry)
                            └── Max iterations ──→ synthesis ──→ END
```

---

## Key Design Patterns

### 1. Thread-Safe Lazy Singleton Models

All ML models (embedding, cross-encoder) loaded once on first use with double-checked locking:

```python
_instance = None
_lock = threading.Lock()

def _get_shared_model():
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = load_model()
    return _instance
```

Components sharing the singleton: Retriever, Memory, GroundingVerifier, Evaluator.

### 2. Confidence-Gated Pipeline

```python
if self.pipeline_config.should_use_full_pipeline(top_score):
    # Low confidence → expansion + HyDE + reranking + compression
else:
    # High confidence → fast path (direct retrieval + generation)
```

High-confidence queries skip ~60% of pipeline stages for 3–4× latency reduction.

### 3. Request-Scoped Telemetry

```python
latencies = {}  # Each request gets its own dict
await self._phase_expand_async(query, mode, latencies)
# latencies['phase_1_expansion_ms'] = 142.3
```

No race conditions — each request writes to its own isolated dictionary.

### 4. Structured Context Isolation

```python
f'<document source="{source}">\n{sanitized_text}\n</document>'
```

XML-style wrapping prevents LLM confusion between document boundaries and metadata.

### 5. Proxy-Based Retry Injection

```python
# Transparent retry via proxy objects
client.chat.completions.create()  # → RobustCompletions.create()
                                   #   → execute_with_retry(raw.create, ...)
```

All LLM calls get retry logic without modifying call sites.

### 6. Dual Sync/Async API

Native async implementations with sync wrappers for backward compatibility:
```python
def ask(self, ...):
    return asyncio.run(self.ask_async(...))
```

---

## Testing Structure

```
tests/
├── unit/           # Component-level isolation tests
├── integration/    # End-to-end pipeline flow tests
├── stress/         # Concurrency and memory pressure tests
└── diagnostics/    # Introspective analysis and profiling tools
```

Run unit tests:
```bash
python -m pytest tests/unit/ -v
```