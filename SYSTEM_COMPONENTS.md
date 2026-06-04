# RAG Multi-Agent System - Component Architecture

## Overview

This system is a **hybrid RAG (Retrieval-Augmented Generation) and multi-agent architecture** that combines document-based question answering with cooperative AI agents. It can operate in two modes:

1. **Context Engine Mode** - Direct RAG pipeline with optimized features
2. **Agentic Mode** - Multi-agent cooperative workflow with specialized workers

---

## Core Components

### 1. Main Entry Point (`src/main.py`)

**Purpose**: System initialization and query execution entry point.

**Principles**:
- Single entry point for both CLI and programmatic usage
- Lazy-initializes the multi-agent graph on first query
- Sanitizes all inputs before processing

---

## Multi-Agent Architecture

### 2. State Schema (`src/graph/state.py`)

**Purpose**: Defines the shared state structure for agent coordination.

**Attributes**:
| Field | Type | Purpose |
|-------|------|---------|
| `messages` | List[BaseMessage] | Conversation history |
| `next_agent` | str | Routing decision from supervisor |
| `context_notes` | List[str] | External notes for context engine variant |
| `steps_remaining` | int | Bounded loop counter (prevents infinite loops) |
| `final_answer` | str | Synthesized response |
| `plan` | List[str] | Step-by-step execution plan |
| `scratchpad` | str | Blackboard for accumulating findings |
| `current_task` | str | Specific instruction for next worker |
| `worker_complete` | Dict[str, bool] | Completion tracking per worker |
| `worker_outputs` | Dict[str, str] | Raw outputs from workers |

**Principles**:
- Uses annotated types for state merging
- Implements bounded execution to prevent runaway costs
- Blackboard pattern enables information sharing between workers

---

### 3. Workflow Graph (`src/graph/workflow.py`)

**Purpose**: Orchestrates the flow between agents using LangGraph.

**Components**:
- **Nodes**: supervisor, 6 workers, aggregator, synthesizer
- **Edges**: Conditional routing from supervisor to workers, return to aggregator
- **Flow**: Supervisor → Worker → Aggregator → Supervisor (loop) → Synthesizer → END

**Principles**:
- Conditional edges enable dynamic routing based on supervisor decisions
- Aggregator merges parallel/sequential worker results
- Recursion limit prevents infinite loops

---

### 4. Supervisor Node (`src/graph/supervisor.py`)

**Purpose**: Central planner that constructs plans and dispatches workers.

**Responsibilities**:
1. Builds step-by-step plan (up to 3 steps) when starting
2. Evaluates scratchpad findings against plan
3. Routes to appropriate workers or synthesizer
4. Detects when critic flags severe issues (RETRY_REQUIRED)

**Principles**:
- Structured output via Pydantic for reliable routing decisions
- Blackboard context provides full state visibility
- Temperature=0 for deterministic planning

---

### 5. Worker Nodes (`src/agents/`)

#### RAG Worker (`src/agents/rag_worker.py`)
- **Purpose**: Answers questions using only uploaded documents
- **Principle**: Zero hallucination - only responses backed by document content
- **Outputs**: "I don't know based on the provided documents" when no match

#### Web Worker (`src/agents/web_worker.py`)
- **Purpose**: Fetches current web information via search
- **Principle**: Citations required - always provides source URLs
- **Outputs**: Structured responses with headers and bullet points

#### Utility Worker (`src/agents/utility_worker.py`)
- **Purpose**: Handles deterministic tasks (math, datetime, summarization)
- **Principle**: Specialized scope - rejects general knowledge queries
- **Tools**: Calculator, datetime lookup, text summarization

#### Scraper Worker (`src/agents/scraper_worker.py`)
- **Purpose**: Fetches and extracts text content from specific URLs
- **Principle**: SSRF protection - blocks private/internal network access
- **Output Format**: Summarizes scraped content relevant to query

#### Critic Worker (`src/agents/critic_worker.py`)
- **Purpose**: Fact-checks and cross-references findings
- **Principle**: Identifies contradictions and gaps between sources
- **Special Feature**: RETRY_REQUIRED token forces supervisor to re-route

#### Report Worker (`src/agents/report_worker.py`)
- **Purpose**: Generates comprehensive markdown reports
- **Principle**: Long-form, well-structured documentation
- **Output**: Saves reports to disk with automatic cleanup

---

### 6. Synthesizer Node (`src/graph/synthesizer.py`)
- **Purpose**: Compiles final response from accumulated findings
- **Principle**: Unified assistant persona - hides internal agent structure
- **Features**: Filters worker messages, uses markdown formatting

---

## RAG Core Pipeline

### 7. Engine (`src/core/engine.py`)

**Purpose**: Main orchestrator for the RAG pipeline.

**Phases**:
| Phase | Component | Purpose |
|-------|-----------|---------|
| P1 | QueryExpander | Generates semantic query variations |
| P1.5 | HyDE | Creates hypothetical document for better retrieval |
| P2 | RetrievalService | Searches Weaviate vector database |
| P3 | NeuralReranker | Re-scores results with Cross-Encoder |
| P4 | MemoryService | Synchronizes conversation history |
| P5 | Compressor | Intelligently shortens documents |
| P6 | GenerationService | Produces final answer via LLM |

**Principles**:
- Conditional execution based on confidence thresholds
- High confidence = skip expensive features
- Both sync and async implementations

---

### 8. Weaviate Retriever (`src/core/retriever.py`)

**Purpose**: Connects to vector database for document storage/search.

**Features**:
- Hybrid search (semantic + keyword)
- Dynamic alpha adjustment for technical queries
- Deterministic UUIDs prevent duplicate entries
- Automatic retry on transient errors

**Security**: SSRF protection via URL validation

---

### 9. Retrieval Service (`src/core/services/retrieval_service.py`)

**Purpose**: Normalizes and executes document retrieval.

**Principles**:
- Runs multiple search queries concurrently
- Deduplicates results by text content
- Returns top candidates sorted by relevance score

---

### 10. Neural Reranker (`src/core/reranker.py`)

**Purpose**: Improves search result ranking accuracy.

**Principles**:
- Uses Cross-Encoder model (query + document together)
- Lazy loading for faster startup
- Normalizes scores to 0.0-1.0 range

---

### 11. Compressor (`src/core/compressor.py`)

**Purpose**: Reduces document size to fit context window.

**Principles**:
- Hybrid scoring (semantic + lexical overlap)
- Preserves code blocks and tables intact
- Greedy selection by descending score within budget

---

### 12. Query Expander (`src/core/expander.py`)

**Purpose**: Creates alternative search queries.

**Principles**:
- Only for queries with sufficient word count
- Generates 3 variations focusing on different keywords
- Increases recall for complex questions

---

### 13. HyDE Generator (`src/core/hyde.py`)

**Purpose**: Hypothetical Document Embedding for better retrieval.

**Principle**: Generate a hypothetical answer, then use it as additional search query

---

## Memory & Persistence

### 14. Memory (`src/core/memory.py`)

**Purpose**: Manages conversation history with intelligent forgetting.

**Features**:
- Exponential decay: `weight = base_importance * e^(-decay_rate * hours)`
- Semantic deduplication prevents similar message repetition
- Token-budgeted context extraction

**Principles**:
- Memory fades over time
- Recent/important messages prioritized
- Automatic cleanup prevents memory bloat

---

### 15. Persistence (`src/core/persistence.py`)

**Purpose**: SQLite database storage for conversation history.

**Schema**:
- `memory`: (session_id, role, text, importance, telemetry, timestamp)
- `sessions`: (session_id, title, created_at, updated_at)

**Features**:
- WAL mode for concurrent access
- Retry logic for locked databases
- Session management (create, rename, delete, list)

---

### 16. Memory Service (`src/core/services/memory_service.py`)

**Purpose**: Bridge between in-memory conversation and persistent storage.

**Principles**:
- Lazy restoration - only loads when needed
- Every add/push goes to both memory and database

---

## Security & Safety

### 17. Safety Filters (`src/tools/safety_filters.py`)

**Purpose**: Prevents prompt injection and output sanitization.

**Protection**:
- Blocks: "ignore instructions", role impersonation, script injection
- Truncates: Context size, result count, query length
- Validates: All tool inputs/outputs

---

### 18. Document Security (`src/core/security.py`)

**Purpose**: Sanitizes document content before LLM consumption.

**Protections**:
- Removes embedded instructions from documents
- Flags suspicious patterns (jailbreak attempts)
- Escapes/neutralizes dangerous content

---

### 19. Web Scraper Security (`src/core/scraper.py`)

**Purpose**: SSRF protection for web scraping.

**Protection**:
- Validates URLs before fetching
- Blocks private IP ranges (127.0.0.1, 192.168.x.x, etc.)
- Blocks localhost and internal hostnames
- Async + sync implementations with controlled concurrency

---

## Services Layer

### 20. Grounding Service (`src/core/services/grounding_service.py`)

**Purpose**: Verifies LLM answers are based on source documents.

**Components**:
- `GroundingVerifier`: Sentence-level support checking via embeddings
- `GroundingEnforcer`: Adds citations and warnings

**Principles**:
- Semantic similarity for claim verification
- Entity hallucination detection (numbers, proper nouns)
- 0.0-1.0 grounding score with unsupported claims list

---

### 21. Generation Service (`src/core/services/generation_service.py`)

**Purpose**: LLM interaction with response validation.

**Features**:
- Sync, async, and streaming variants
- Token usage tracking
- Grounding verification on all responses
- Context window monitoring

---

### 22. Context Overflow Service (`src/core/services/overflow_service.py`)

**Purpose**: Handles context window overflow gracefully.

**Recovery Steps**:
1. **Memory Pruning**: Remove oldest conversation turns
2. **Aggressive Compression**: Re-compress documents to smaller budget
3. **Hard Truncation**: Final fallback to fit limits

---

### 23. Telemetry Service (`src/core/services/telemetry_service.py`)

**Purpose**: Performance and cost tracking.

**Metrics**:
- Query count and latency
- Token usage and cost calculation
- Compression ratios
- System resource usage (CPU, RAM)

---

## Knowledge Management

### 24. Knowledge Registry (`src/core/registry.py`)

**Purpose**: Tracks available documents and sources.

**Provides**:
- List of indexed sources
- Document domains (documentation, sales_analytics, etc.)
- Available schemas (sales_database, system_metrics)
- Topics/tags from metadata

---

## Configuration (`src/core/config.py`)

**Key Settings**:
| Category | Setting | Default | Purpose |
|----------|---------|---------|---------|
| LLM | MODEL | llama-3.1-8b-instant | AI model selection |
| LLM | TEMPERATURE | 0.1 | Creativity control |
| Embedding | MODEL | all-MiniLM-L6-v2 | Vector generation |
| Retrieval | MAX_CANDIDATES | 12 | Top results limit |
| Memory | DECAY_RATE | 0.1 | Forgetting speed |
| Compression | THRESHOLD | 0.02 | Minimum relevance score |

---

## Data Flow Summary

```
User Query
    │
    ├──────────────────────┐
    │                      │
    ▼                      ▼
Context Engine      Agentic Mode
    │                      │
    ▼                      ▼
┌─────────────────────────────────────┐
│     RAG Pipeline Phases             │
│  1. Expansion/HyDE (conditional)    │
│  2. Retrieval                       │
│  3. Reranking (conditional)         │
│  4. Memory                          │
│  5. Compression (conditional)        │
│  6. Generation                      │
└─────────────────────────────────────┘
    │
    ▼
Synthesizer → Final Answer
```

---

## Agentic Mode Flow

```
User Query
    │
    ▼
supervisor_node ←──┐
    │              │
    ▼              │ (if more work needed)
[route to worker]  │
    │              │
    ▼              │
aggregate_parallel_results_node
    │
    ▼
supervisor_node → synthesizer_node → Final Answer
(or loop back for more workers)
```

---

## Key Principles by Layer

### Orchestration Layer
- Bounded execution prevents runaway costs
- Blackboard pattern enables worker collaboration
- Conditional routing based on confidence/state

### Retrieval Layer
- Hybrid search balances semantic + keyword matching
- Conditional expensive features (rerank, compress) for low confidence
- Query expansion increases recall for complex questions

### Memory Layer
- Exponential decay mimics human forgetting
- Semantic deduplication prevents redundant storage
- Token budgeting ensures context window compliance

### Security Layer
- Zero trust: sanitize all inputs/outputs
- SSRF protection for web scraping
- Hallucination detection via grounding verification

### Agentic Layer
- Specialized workers with strict boundaries
- Critic validates before synthesis
- Report worker creates persistent artifacts