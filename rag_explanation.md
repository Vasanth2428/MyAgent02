# RAG Context Engine — Complete Technical Guide

**Version 3.1 | Last Updated: May 2026**

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Core Concepts for Beginners](#2-core-concepts-for-beginners)
3. [System Architecture](#3-system-architecture)
4. [The Multi-Phase Neural Pipeline](#4-the-multi-phase-neural-pipeline)
5. [Directory Structure &amp; File Map](#5-directory-structure--file-map)
6. [How Documents Are Processed (Upload Flow)](#6-how-documents-are-processed-upload-flow)
7. [How Questions Are Answered (Query Flow)](#7-how-questions-are-answered-query-flow)
8. [Token Management &amp; Budget System](#8-token-management--budget-system)
9. [Memory System (How the Engine &#34;Remembers&#34;)](#9-memory-system-how-the-engine-remembers)
10. [The Frontend Dashboard](#10-the-frontend-dashboard)
11. [Neural Models Used](#11-neural-models-used)
12. [API Reference](#12-api-reference)
13. [Performance Optimizations](#13-performance-optimizations)
14. [Security](#14-security)
15. [Developer Setup](#15-developer-setup)
16. [Troubleshooting](#16-troubleshooting)
17. [Glossary](#17-glossary)

---

## 1. What Is This Project?

### The Problem

Large Language Models (LLMs) like ChatGPT or Llama are trained on a fixed dataset. They cannot access your private documents, company manuals, or recently created files. If you ask an LLM about content it has never seen, it will either hallucinate (make up an answer) or say "I don't know."

### The Solution: RAG

**Retrieval-Augmented Generation (RAG)** solves this by adding a retrieval step before generation:

1. **You upload your documents** → the system stores them in a searchable database.
2. **You ask a question** → the system searches those documents for relevant passages.
3. **The relevant passages are injected into the LLM's prompt** → the LLM reads them and generates an answer grounded in your actual data.

This means the LLM never needs to have "memorized" your documents during training. It reads them on-the-fly, every time you ask a question.

### What Makes THIS System Different?

A basic RAG system does: `search → paste into prompt → generate`. This project goes far beyond that with a **multi-phase neural pipeline**:

| Basic RAG                  | This Context Engine                                     |
| -------------------------- | ------------------------------------------------------- |
| Single vector search       | Multi-query expansion + HyDE + hybrid search            |
| No ranking                 | Neural Cross-Encoder reranking                          |
| No memory                  | Session-aware conversational memory with temporal decay |
| Dumps all text into prompt | Extractive compression with token budgeting             |
| No observability           | Full Glass Box telemetry (tokens, latency, cost)        |

---

## 2. Core Concepts for Beginners

### What Is an Embedding?

An **embedding** is a way to represent text as a list of numbers (a "vector"). For example:

- `"The cat sat on the mat"` → `[0.12, -0.45, 0.78, ..., 0.33]` (384 numbers)
- `"A kitten rested on the rug"` → `[0.11, -0.44, 0.77, ..., 0.34]` (very similar numbers!)

Sentences with similar meanings produce vectors that are close together in mathematical space. This lets us find relevant documents by calculating which stored vectors are closest to the query vector — even if the exact words are different.

We use a model called **all-MiniLM-L6-v2** to generate these embeddings locally on your machine.

### What Is a Vector Database?

A **vector database** is a specialized database optimized for storing and searching embeddings. Instead of SQL queries like `WHERE text LIKE '%cat%'`, you give it a vector and ask "find the 5 closest vectors to this one."

We use **Weaviate Cloud** as our vector database. It also supports traditional keyword search (BM25), so we combine both approaches — this is called **hybrid search**.

### What Is a Token?

LLMs don't process text character-by-character. They split text into **tokens** — small pieces that are roughly ¾ of a word on average.

- `"Hello world"` = 2 tokens
- `"Retrieval-Augmented Generation"` = 4 tokens
- A typical sentence = 15-25 tokens

Every LLM has a **context window** — a maximum number of tokens it can process at once. Our model (Llama 3.1 8B) has an 8,192-token window. Everything — the instruction, the retrieved documents, the user's question, AND the generated answer — must fit within this limit. This is why **token budgeting** is critical.

We use **tiktoken** (OpenAI's tokenizer library) to count tokens exactly, rather than guessing with `len(text) / 4`.

### What Is a Cross-Encoder?

A **Cross-Encoder** is a neural network that takes a pair of texts (a query and a document) and outputs a relevance score. Unlike embeddings (which encode query and document separately), a Cross-Encoder reads both texts together, allowing it to understand deep semantic relationships.

This makes Cross-Encoders much more accurate than vector similarity, but also slower — so we only use them to re-score a small set of candidates (not the entire database).

### What Is BM25?

**BM25** is a traditional keyword-matching algorithm. It scores documents based on how many query words appear in them, adjusted for document length and word rarity. It's fast and excellent at finding exact matches (error codes, function names, specific terms).

---

## 3. System Architecture

### High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                     BROWSER (index.html)                         │
│  ┌────────────┐ ┌──────────────┐ ┌────────────────────────────┐  │
│  │ Process    │ │ Chat Thread  │ │ Performance Metrics (9-slot)│  │
│  │ Logs       │ │              │ │ Source Context Dump         │  │
│  │ Glass Box  │ │              │ │ Mode Selector               │  │
│  │ Inspector  │ │              │ │ Document Upload              │  │
│  └────────────┘ └──────┬───────┘ └────────────────────────────┘  │
│                        │ HTTP (fetch)                             │
└────────────────────────┼─────────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                    main.py (FastAPI Server)                     │
│                                                                │
│  POST /query   → Runs the full retrieval + generation pipeline │
│  POST /upload  → Chunks documents and indexes into Weaviate    │
│  GET  /stats   → Returns live system metrics                   │
│  GET  /history → Returns conversation history from SQLite      │
└────────────────────────┬───────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Weaviate     │ │ SQLite       │ │ Groq API     │
│ Cloud        │ │ (memory.db)  │ │ (LLM)        │
│              │ │              │ │              │
│ Stores       │ │ Stores       │ │ Generates    │
│ document     │ │ conversation │ │ text, query  │
│ vectors      │ │ history      │ │ expansions,  │
│ + text       │ │ per session  │ │ HyDE docs    │
└──────────────┘ └──────────────┘ └──────────────┘
```

### The `core/` Module Breakdown

| File               | Role                                                                                      | Key Classes                             |
| ------------------ | ----------------------------------------------------------------------------------------- | --------------------------------------- |
| `config.py`      | **Configuration** — single source of truth for all constants and tuning parameters | *(module-level constants)*            |
| `engine.py`      | **Orchestrator** — coordinates all phases via decomposed private methods           | `RAGContextEngine`                    |
| `compressor.py`  | **Compression** — extractive sentence selection within a token budget              | `Compressor`                          |
| `reranker.py`    | **Reranking** — Cross-Encoder deep semantic relevance scoring                      | `NeuralReranker`                      |
| `expander.py`    | **Expansion** — LLM-driven multi-query generation                                  | `QueryExpander`                       |
| `hyde.py`        | **HyDE** — hypothetical document generation for retrieval alignment                | `HyDEGenerator`                       |
| `retriever.py`   | **Data Access** — Weaviate Cloud interface with dynamic alpha tuning               | `WeaviateRetriever`                   |
| `memory.py`      | **Short-term Memory** — temporal decay and semantic deduplication                  | `ConversationMemory`, `MemoryEntry` |
| `persistence.py` | **Long-term Storage** — SQLite-backed session history across restarts              | `PersistentMemoryStore`               |
| `splitter.py`    | **Text Preprocessing** — recursive boundary-aware text chunking                    | `RecursiveCharacterSplitter`          |
| `processor.py`   | **Legacy Shim** — backward-compatible re-exports from old monolithic module        | *(re-exports only)*                   |

### Frontend Structure

| File                  | Role                                             |
| --------------------- | ------------------------------------------------ |
| `index.html`        | Slim HTML skeleton (structure only)              |
| `static/styles.css` | Design system and all CSS                        |
| `static/app.js`     | Application logic, API calls, inspector renderer |

---

## 4. The Multi-Phase Neural Pipeline

When you type a question and hit "EXECUTE," the engine runs through these phases **in sequence**. Each phase transforms the data before handing it to the next.

### Phase 1: Query Expansion

**File:** `core/expander.py` → `QueryExpander`

**The Problem:** Users and documents often use different words for the same concept. You might ask "How do I fix the login bug?" but the document says "Authentication error resolution procedure."

**The Solution:** The LLM generates 3 alternative phrasings of your question:

- Original: `"How do I fix the login bug?"`
- Variation 1: `"Authentication error troubleshooting steps"`
- Variation 2: `"Login failure resolution procedure"`
- Variation 3: `"User sign-in issue debugging guide"`

Now we search for ALL of these, dramatically increasing the chance of finding relevant documents.

**Smart Skip:** If your query is very short (under 5 words), expansion is skipped to save ~1 second of latency, because short queries are usually direct enough.

### Phase 1.5: Hypothetical Document Embeddings (HyDE)

**File:** `core/hyde.py` → `HyDEGenerator`

**The Problem:** Questions and answers are fundamentally different types of text. The embedding for "What causes memory leaks?" is quite different from the embedding for a paragraph that explains memory leaks. This means vector search can miss highly relevant documents.

**The Solution:** Before searching, the LLM writes a brief **hypothetical answer** to your question — a "draft" of what a good answer might look like. We then search the database using this draft's embedding instead of (in addition to) the question's embedding.

Since the hypothetical answer and the actual stored documents are both declarative paragraphs, their embeddings align much better, significantly improving retrieval accuracy.

**Example:**

- Query: `"What causes memory leaks in Python?"`
- HyDE Output: `"Memory leaks in Python commonly occur when objects maintain circular references, when global variables hold large data structures indefinitely, or when C extensions fail to properly deallocate memory..."`
- This paragraph's vector is now used to search — and it matches stored documents about memory management far better than the raw question would.

### Phase 2: Hybrid Retrieval

**File:** `core/retriever.py` → `WeaviateRetriever.retrieve()`

The system searches Weaviate using **hybrid search** — a combination of two complementary approaches:

| Approach                    | How It Works                  | Good At                              | Bad At                   |
| --------------------------- | ----------------------------- | ------------------------------------ | ------------------------ |
| **Vector (Semantic)** | Compares embedding similarity | Finding conceptually related content | Exact names, error codes |
| **BM25 (Keyword)**    | Counts matching words         | Exact matches, specific terms        | Synonyms, paraphrasing   |

**Dynamic Alpha Tuning:** The `alpha` parameter controls the balance. Rather than using a fixed 50/50 split, the system inspects the query:

- If the query contains code syntax (`{}`, `()`, `.`, `_`), error terms, or programming keywords → alpha shifts to **0.20** (heavily favoring keyword matching)
- If the query is conversational → alpha stays at **0.50** (balanced hybrid)

This is implemented in `_detect_alpha()` in `retriever.py`.

**Candidate Cap:** All search variations (original + expansions + HyDE) are searched, but the total unique results are capped at **12 candidates** to prevent downstream phases from becoming too slow.

### Phase 3: Neural Reranking

**File:** `core/reranker.py` → `NeuralReranker`

**The Problem:** Hybrid search returns candidates ranked by a combination of vector similarity and keyword overlap. These scores are decent but not deeply accurate — they don't truly "understand" whether a passage answers the specific question.

**The Solution:** A **Cross-Encoder** model (`ms-marco-MiniLM-L-6-v2`) reads each (query, candidate) pair together and assigns a deep relevance score from 0.0 to 1.0.

**Before reranking** (by hybrid score): The paragraph about "memory allocation strategies" might rank #7.
**After reranking** (by Cross-Encoder): That same paragraph jumps to #1 because the model understood it directly answers the query about memory leaks.

The top 5 candidates after reranking proceed to the next phase.

**Resilience:** If the reranker crashes (e.g., out of memory), the system gracefully falls back to the original hybrid scores instead of failing entirely.

### Phase 4: Memory Recall

**File:** `core/memory.py` → `ConversationMemory` (RAM) & `core/persistence.py` → `PersistentMemoryStore` (SQLite)

The system retrieves your recent conversation history to give the LLM context about what you've been discussing.

**Temporal Decay Formula:** `Weight = Importance × e^(-DecayRate × HoursElapsed)`

- A message from 5 minutes ago has nearly full weight.
- A message from 2 hours ago has reduced weight.
- A message from yesterday is nearly invisible.

This ensures the LLM focuses on your current topic, not something you discussed hours ago.

**Deduplication:** If you say the same thing twice, memory doesn't store it again — it "touches" the existing entry to reset its decay timer.

**Budget:** Memory is allocated **300 tokens** of the total 1,500-token context budget.

### Phase 5: Extractive Compression

**File:** `core/compressor.py` → `Compressor`

**The Problem:** After retrieval, we might have 5 document chunks totaling 3,000 tokens, but our knowledge budget is only ~1,200 tokens (1,500 total minus memory). We need to keep the most relevant sentences and discard filler.

**How It Works:**

1. All retrieved text is split into individual sentences using a regex pattern that correctly handles abbreviations ("Dr. Smith") and decimal numbers ("3.14").
2. Each sentence is scored by **lexical overlap** with the query — how many query words appear in the sentence.
3. Sentences are selected in order of relevance score until the token budget is filled.
4. Selected sentences are reassembled **in their original document order** to maintain logical flow.

**Fast Path:** If the retrieved text already fits within the budget, this entire phase is skipped to save time.

**Token Counting:** Uses `tiktoken` (BPE tokenizer) for exact token measurements, not character-count approximations.

### Phase 6: Response Generation

**File:** `core/engine.py` → LLM API call

The final assembled prompt is sent to **Groq's Llama 3.1 8B Instant** model:

```
Answer the user question using ONLY the provided context.
If the information is missing, state that you don't know.

### CONTEXT:
### MEMORY
[user]: What causes memory leaks?
[assistant]: Memory leaks occur when...

### KNOWLEDGE
Memory management in Python involves garbage collection...
The gc module can detect circular references...

### QUESTION:
How do I detect memory leaks in production?

### ANSWER:
```

The instruction `"Answer ONLY using the provided context"` is critical — it prevents the LLM from making up information that isn't in your documents.

---

## 5. Directory Structure & File Map

```text
/RAG
├── main.py              # FastAPI server — HTTP endpoints, CORS, request routing, serving UI
├── index.html           # Slim dashboard UI HTML skeleton (structure only)
├── .env                 # API keys for Weaviate and Groq (NEVER commit this)
├── .gitignore           # Prevents secrets, DB, and build artifacts from Git tracking
├── requirements.txt     # Python dependency manifest
├── memory.db            # SQLite database — conversation history (auto-created)
├── rag_explanation.md   # This document
│
├── core/                # Python package — modularized backend components
│   ├── __init__.py      # Package marker and public API exports (__all__)
│   ├── config.py        # Centralized system constants and tuning parameters
│   ├── engine.py        # Pipeline orchestrator decomposed into helper methods
│   ├── compressor.py    # Extractive sentence selection & token budget enforcement
│   ├── reranker.py      # Cross-Encoder deep semantic scoring (ms-marco-MiniLM-L-6-v2)
│   ├── expander.py      # LLM-driven query expansion variations
│   ├── hyde.py          # Hypothetical Document Generator
│   ├── retriever.py     # Weaviate interface with dynamic hybrid alpha matching
│   ├── memory.py        # Short-term conversation context (with decay & Jaccard deduplication)
│   ├── persistence.py   # Long-term SQLite history storage manager
│   ├── splitter.py      # Recursive, boundary-aware text splitter
│   └── processor.py     # Legacy compatibility shim (re-exports)
│
├── static/              # Decoupled frontend static assets
│   ├── styles.css       # Controls dashboard style and CSS variables
│   └── app.js           # Controls dashboard API requests, logic, and rendering
│
└── tests/               # Test suites
    ├── run_suite.py      # Discover and run all integration/unit tests
    ├── run_modular_suite.py  # Script to spin up server and run tests in sequence
    ├── integration/      # End-to-end integration tests (search, upload, concurrency)
    └── unit/             # Isolated unit tests for the core modules
```

---

## 6. How Documents Are Processed (Upload Flow)

When you drop a PDF or TXT file into the dashboard:

/*-p/-File (PDF/TXT)
    │
    ▼
[1] Text Extraction
    - PDF: pypdf extracts text from each page
    - TXT: raw UTF-8 decode
    │
    ▼
[2] Recursive Chunking (core/splitter.py)
    - Splits into ~1000-character chunks
    - Tries to break at: paragraphs → newlines → sentences → spaces
    - 100-character overlap between chunks (so context isn't lost at boundaries)
    │
    ▼
[3] Batch Embedding (core/retriever.py)
    - All chunks are embedded in a single batch using all-MiniLM-L6-v2
    - Each chunk becomes a 384-dimensional vector
    │
    ▼
[4] Weaviate Indexing
    - Each chunk is stored with: text, embedding vector, source filename
    - Deterministic UUIDs prevent duplicate indexing of the same text

---

## 7. How Questions Are Answered (Query Flow)

A complete end-to-end example:

**User asks:** `"What were the test results for the new authentication module?"`

| Phase                    | What Happens                                                                                                                  | Time            |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | --------------- |
| **1. Expansion**   | LLM generates: "Authentication module test outcomes," "QA results for login system," "Verification report for auth component" | ~800ms          |
| **1.5. HyDE**      | LLM writes: "The authentication module test results showed a 98% pass rate across unit tests, with 2 edge cases..."           | ~600ms          |
| **2. Retrieval**   | 4 queries × hybrid search → 10 unique document chunks found                                                                 | ~400ms          |
| **3. Reranking**   | Cross-Encoder scores all 10 chunks, selects top 5                                                                             | ~150ms          |
| **4. Memory**      | Retrieves last 3 conversation turns (within 300-token budget)                                                                 | ~1ms            |
| **5. Compression** | 5 chunks (2,400 tokens) compressed to fit 1,200-token knowledge budget                                                        | ~5ms            |
| **6. Generation**  | Prompt sent to Groq → answer generated                                                                                       | ~500ms          |
| **Total**          |                                                                                                                               | **~2.5s** |

---

## 8. Token Management & Budget System

### The Context Window Problem

Llama 3.1 8B has an **8,192-token context window**. Everything must fit:

```
┌─────────────────────────────────────────────┐
│              8,192 TOKEN WINDOW             │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │ System Instructions      (~40 tkn)  │    │
│  ├─────────────────────────────────────┤    │
│  │ Memory Context          (≤300 tkn)  │    │
│  ├─────────────────────────────────────┤    │
│  │ Knowledge Context      (≤1200 tkn)  │    │
│  ├─────────────────────────────────────┤    │
│  │ User Question           (~20 tkn)   │    │
│  ├─────────────────────────────────────┤    │
│  │ Generated Answer       (≤512 tkn)   │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  Total prompt ≈ 1,560 tokens               │
│  Remaining capacity ≈ 6,632 tokens (81%)   │
└─────────────────────────────────────────────┘
```

### How tiktoken Works

Instead of guessing tokens with `len(text) / 4`, we use OpenAI's `tiktoken` library with the `cl100k_base` encoding to count exact BPE (Byte-Pair Encoding) tokens:

```python
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = tokenizer.encode("Hello, world!")  # Returns [9906, 11, 1917, 0]
print(len(tokens))  # 4 exact tokens
```

This is used in:

- `core/engine.py` — budgeting memory and knowledge slots
- `core/compressor.py` — compression sentence selection
- `core/memory.py` — memory context assembly

---

## 9. Memory System (How the Engine "Remembers")

### Two-Layer Architecture

**Layer 1: RAM (Fast, Volatile)**
`ConversationMemory` in `core/memory.py` holds recent turns in Python objects. Instant access, but lost if the server restarts.

**Layer 2: SQLite (Persistent)**
`PersistentMemoryStore` in `core/persistence.py` writes every turn to `memory.db`. On server restart, RAM is automatically restored from SQLite.

### Session Isolation

Each user gets a unique **Session ID** (stored in the browser's `localStorage`). Memory from Session A never leaks into Session B.

### SQLite Concurrency & Reliability

The persistence layer (`core/persistence.py`) incorporates advanced mechanisms to handle high concurrency and write-locking:
- **Write-Ahead Logging (WAL)**: Initialized via `PRAGMA journal_mode=WAL;`. This decouples reads from writes, allowing multiple parallel readers to execute queries even when a transaction is writing.
- **Composite Database Indexing**: An index `idx_memory_session_timestamp` is created on `memory (session_id, timestamp)`. This prevents table scans and optimizes the performance of retrieving conversation histories under high load.
- **Transaction Retry Helper**: Wraps operations in a retry handler (`execute_with_retry`) using exponential backoff with random jitter. If a `sqlite3.OperationalError` (specifically `database is locked` or `database is busy`) is encountered, the connection retries up to 5 times (base delay of 50ms, doubling with each attempt, plus up to 20ms jitter) before raising.

### Temporal Decay

The formula: **W = I × e^(-0.1 × H)**

| Time Since Message | Weight (Importance=1.0) |
| ------------------ | ----------------------- |
| Just now           | 1.00                    |
| 30 minutes         | 0.95                    |
| 2 hours            | 0.82                    |
| 6 hours            | 0.55                    |
| 24 hours           | 0.09 (nearly invisible) |

This ensures the LLM always focuses on your current conversation topic.

### Chronological Sorting

To ensure the conversation flow remains coherent to the LLM, the retrieval of memory context preserves chronological order:
1. All stored entries for the current session are analyzed.
2. Entries with relevance weights higher than `MEMORY_WEIGHT_THRESHOLD` are filtered in.
3. The filtered active entries are ranked by current decay-adjusted weights.
4. The highest-ranked entries are selected sequentially until the memory token budget (`MEMORY_TOKEN_BUDGET` = 300 tokens) is saturated.
5. **Critical Step**: Before formulating the final prompt segment, the selected entries are re-sorted chronologically according to their original input sequence (relative insertion order). This guarantees that the LLM does not receive scrambled dialogue.

---

## 10. The Frontend Dashboard

The frontend consists of a slim HTML structure (`index.html`), a styling system (`static/styles.css`), and dynamic application logic (`static/app.js`). It displays a control console with 5 panels:

| Panel                         | Purpose                                                                                                                                                  |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PROCESS_LOGS**        | Real-time event feed showing requests, uploads, and errors                                                                                               |
| **GLASS_BOX_INSPECTOR** | Deep telemetry: exact token breakdown, HyDE output, query expansions, latencies, raw prompt                                                              |
| **INTERACTION_THREAD**  | Chat interface with Markdown rendering (via marked.js)                                                                                                   |
| **PERFORMANCE_METRICS** | 9-slot dashboard: Total Queries, Context Reduction, Round Trip, Active Thread Memory, Host CPU, Host RAM, Prompt Window, LLM Speed (TPS), Estimated Cost |
| **SOURCE_CONTEXT_DUMP** | Shows the actual document chunks used to generate each answer                                                                                            |

### Glass Box Inspector Sections

After each query, the inspector shows:

1. **Token Consumption** — Exact prompt/completion/total tokens + breakdown (Instructions, Memory, Knowledge, Query)
2. **Generation Speed** — Tokens per second + estimated USD cost
3. **Budget Allocation** — Memory tokens used/limit, Knowledge tokens used/limit
4. **Pipeline Telemetry** — Mode, hybrid alpha, peak reranker score, compression ratio, embed/search/HyDE latencies
5. **Query Expansions** — The 3 LLM-generated search variations
6. **HyDE Output** — The hypothetical answer paragraph
7. **Raw Prompt** — The exact text sent to the LLM

---

## 11. Neural Models Used

| Model                      | Type                 | Runs On    | Purpose                                     | Size      |
| -------------------------- | -------------------- | ---------- | ------------------------------------------- | --------- |
| `all-MiniLM-L6-v2`       | Sentence Transformer | Local CPU  | Converts text → 384-dim vectors for search | 80MB      |
| `ms-marco-MiniLM-L-6-v2` | Cross-Encoder        | Local CPU  | Scores query-document relevance (0.0–1.0)  | 80MB      |
| `Llama-3.1-8B-Instant`   | Large Language Model | Groq Cloud | Text generation, query expansion, HyDE      | 8B params |
| `cl100k_base` (tiktoken) | BPE Tokenizer        | Local CPU  | Exact token counting for budget management  | <1MB      |

---

## 12. API Reference

### POST `/query`

Runs the full retrieval + generation pipeline.

**Request:**

```json
{
  "question": "What causes memory leaks?",
  "session_id": "SID-A1B2C3",
  "mode": "context_engine",
  "source_filter": null
}
```

- `mode`: `"context_engine"` (full pipeline) or `"normal"` (simple search + generate)
- `source_filter`: Optional filename to restrict search to a single document

**Response includes:** `query`, `response`, `mode`, `search_queries`, `hyde_doc`, `tps`, `query_cost`, `retrieved_context`, `compressed_context`, `memory_context`, `raw_prompt`, `stats` (with full telemetry)

### POST `/upload`

Indexes a document into the vector database.

- **Accepts:** `.pdf` or `.txt` files via `multipart/form-data`
- **Returns:** `{"status": "success", "message": "Indexed 12 chunks from report.pdf"}`

### GET `/stats`

Live system metrics for the dashboard.

- **Returns:** `queries_handled`, `avg_compression`, `avg_latency_ms`, `document_count`, `cpu_usage_percent`, `memory_usage_percent`

### GET `/history/{session_id}`

Retrieves the last 10 conversation turns for a session from SQLite.

### Database Persistence (SQLite)

All conversation logs and metrics are backed by SQLite. Under the hood, database operations are managed with production-grade safety:
- **Journal Mode**: WAL (Write-Ahead Logging) is enabled dynamically on all connections to permit concurrent reads during write locks.
- **Connection Isolation**: Connects to `memory.db` with a connection timeout of 30 seconds.
- **Retry Mechanism**: Read/write actions are executed via a transaction retry helper implementing binary exponential backoff with randomized jitter to resolve potential write contentions gracefully.

---

## 13. Performance Optimizations

| Optimization                        | Where    | Impact                                                            |
| ----------------------------------- | -------- | ----------------------------------------------------------------- |
| **Smart Skip**                | Phase 1  | Short queries (<5 words) skip expansion, saving ~1s               |
| **Candidate Cap (12)**        | Phase 2  | Prevents reranking from processing hundreds of results            |
| **Fast Path Compression**     | Phase 5  | If text already fits budget, skip sentence splitting entirely     |
| **Batch Embedding**           | Upload   | All chunks embedded in one GPU/CPU pass instead of one-by-one     |
| **asyncio.to_thread**         | main.py  | CPU-heavy ML work runs in thread pool, keeping the API responsive |
| **Deterministic UUIDs**       | Upload   | Same text → same UUID → no duplicate entries in Weaviate        |
| **127.0.0.1 (not localhost)** | Frontend | Avoids DNS resolution delay on some systems                       |
| **SQLite WAL Mode**                 | Persistence | Decouples read/write processes, allowing parallel reads without locking blocks. |
| **SQLite Transaction Retries**      | Persistence | Implements exponential backoff + jitter to resolve concurrent DB lock contentions. |
| **Weaviate Connection Retries**     | Retriever | Handshakes Weaviate Cloud up to 3 times on startup to handle cold starts. |
| **Weaviate Operation Retries**      | Retriever | Retries transient timeouts, rate limits, and network glitches with backoff & jitter. |
| **Tolerant ReAct Parsing**          | Agent    | Flexibly parses quotes, brackets, and extra spaces in actions, preventing parsing crashes. |
| **ReAct loop self-correction**      | Agent    | Feeds formatting errors back into LLM context dynamically to auto-recover without aborting. |

---

## 14. Security

- **API Keys:** Stored in `.env`, blocked from Git by `.gitignore`
- **CORS:** Configurable via `ALLOWED_ORIGINS` environment variable (defaults to `*` for development)
- **Input Validation:** Pydantic `Literal` type restricts `mode` to only `"context_engine"` or `"normal"`
- **Output Sanitization:** AI responses are rendered through `marked.js` which escapes HTML

---

## 15. Developer Setup

### Prerequisites

- Python 3.10+
- A [Weaviate Cloud](https://console.weaviate.cloud/) cluster (free tier works)
- A [Groq API Key](https://console.groq.com/) (free tier: 14,400 tokens/min)

### Installation

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd RAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file
# WEAVIATE_API_KEY=your_key_here
# WEAVIATE_URL=https://your-cluster.weaviate.cloud
# GROQ_API_KEY=gsk_your_key_here

# 4. Start the server
python main.py

# 5. Open the dashboard
# Open index.html in your browser
```

### Running Tests

```bash
set PYTHONPATH=.
python tests/run_suite.py
```

---

## 16. Troubleshooting

| Error                                       | Cause                                                      | Fix                                                                                                        |
| ------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **500 Internal Server Error**         | Engine failed to start (Weaviate timeout, missing API key) | Check terminal for "Engine successfully initialized." If missing, verify `.env` and restart.             |
| **413 Payload Too Large**             | Too many tokens sent to Groq                               | Advanced Pipeline handles this automatically. In Simple mode, context is hard-capped at 16,000 characters. |
| **Network Timeout**                   | Weaviate Cloud handshake >5s                               | Connection timeouts are set to 60s. Check your internet connection.                                        |
| **UI shows no history after refresh** | Session ID mismatch                                        | The SID is stored in `localStorage`. Clearing browser data creates a new session.                        |
| **"UNSAFE ATTEMPT" in browser**       | Opening `index.html` via `file://` protocol            | Normal browser security warning. Does not affect API calls to `127.0.0.1:8000`.                          |
| **SQLite `database is locked` error** | High concurrency lock contention during writes              | Handled automatically by retry helper (5 attempts with backoff). Verify no external DB locks exist.          |
| **Weaviate Non-Transient Query Error** | Invalid query parameters, syntax errors, or schema mismatches | Immediately raises error (fails fast) to prevent useless retries and surface the actual bug immediately.   |
| **ReAct Agent Format Violation**     | LLM output violates ReAct spec (`Action: tool[arg]`)        | Loop intercepts violation and posts a self-correction observation to guide LLM back to valid format.       |

---

## 17. Glossary

| Term                             | Definition                                                                                   |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| **RAG**                    | Retrieval-Augmented Generation — injecting retrieved documents into an LLM prompt           |
| **Embedding**              | A fixed-length vector (list of numbers) representing text meaning                            |
| **Vector Database**        | A database optimized for nearest-neighbor search over embeddings                             |
| **BM25**                   | A keyword-matching scoring algorithm (like a smarter version of CTRL+F)                      |
| **Hybrid Search**          | Combining vector search + keyword search for better coverage                                 |
| **Alpha**                  | The parameter controlling the vector/keyword balance (1.0 = pure vector, 0.0 = pure keyword) |
| **Cross-Encoder**          | A neural model that reads a query+document pair together to score relevance                  |
| **HyDE**                   | Hypothetical Document Embeddings — generating a fake answer to improve search vectors       |
| **Temporal Decay**         | Memory entries lose relevance weight over time via exponential decay                         |
| **Context Window**         | The maximum number of tokens an LLM can process in one request                               |
| **Token**                  | The smallest unit of text an LLM processes (roughly ¾ of a word)                            |
| **tiktoken**               | OpenAI's library for exact BPE token counting                                                |
| **TPS**                    | Tokens Per Second — measures LLM generation speed                                           |
| **Extractive Compression** | Selecting the most relevant sentences from a larger text                                     |

---

*RAG Context Engine v3.1 — Built with FastAPI, Weaviate, Groq, and local neural models.*
