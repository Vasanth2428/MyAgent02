"""
================================================================================
RAG CONTEXT ENGINE - ORCHESTRATION MODULE (ASYNC REFACTORED)
================================================================================
The Engine is the central brain of the system, delegating to helper services:
1. Memory Service (core/services/memory_service.py)
2. Retrieval Service (core/services/retrieval_service.py)
3. Generation Service (core/services/generation_service.py)
4. Context Overflow Service (core/services/overflow_service.py)
5. Telemetry Service (core/services/telemetry_service.py)
"""

import logging
import time
import asyncio
from typing import Dict, Generator, AsyncGenerator, Optional, Any, List
import tiktoken
from core.llm import LLMService

from core.config import (
    LLM_MODEL, LLM_TEMPERATURE, CONTEXT_WINDOW_LIMIT,
    TOKENIZER_ENCODING, TOTAL_CONTEXT_BUDGET, MEMORY_TOKEN_BUDGET,
    MIN_KNOWLEDGE_BUDGET, MAX_CANDIDATES, EXPANSION_MIN_WORDS,
    SAFETY_CHAR_LIMIT, COST_PER_INPUT_TOKEN, COST_PER_OUTPUT_TOKEN,
    PipelineConfig
)
from core.retriever import WeaviateRetriever
from core.persistence import PersistentMemoryStore
from core.compressor import Compressor
from core.reranker import NeuralReranker
from core.expander import QueryExpander
from core.hyde import HyDEGenerator
from core.registry import KnowledgeRegistry
from core.services import RetrievalService, MemoryService, GenerationService, ContextOverflowService, TelemetryService
from core.security import sanitize_document_text

logger = logging.getLogger("RAG.Engine")
tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    """Counts exact BPE tokens for a given string."""
    if not text:
        return 0
    return len(tokenizer.encode(text))


class RAGContextEngine:
    """
    Main orchestrator for the RAG pipeline. Delegates to modular services.
    Supports both synchronous and native asynchronous interfaces.
    """

    def __init__(self, retriever: WeaviateRetriever, pipeline_config: Optional[PipelineConfig] = None):
        self.retriever = retriever
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.persistent_memory = PersistentMemoryStore()
        self.compressor = Compressor()
        self.reranker = NeuralReranker()
        self.registry = KnowledgeRegistry(self)

        # Initialize modular services
        self.telemetry = TelemetryService()
        self.stats = self.telemetry.stats  # Keep self.stats for backward compatibility
        self.memory_service = MemoryService(self.persistent_memory)
        self.overflow_service = ContextOverflowService(self.compressor)

        # Initialize centralized LLM service
        self.llm_service = LLMService()
        self.client = self.llm_service.raw_client  # backward compat for generation
        self.async_client = self.llm_service.async_client
        self.generation_service = GenerationService(self.client, LLM_MODEL, LLM_TEMPERATURE)
        self.expander = QueryExpander(self.client)
        self.hyde = HyDEGenerator(self.client)
        self.retrieval_service = RetrievalService(self.retriever)

        # Initialize ReAct Agent
        from core.agent import RAGAgent
        self.agent = RAGAgent(self)

    # ------------------------------------------------------------------
    # Memory Delegations
    # ------------------------------------------------------------------

    def get_memory(self, session_id: str):
        """Retrieves memory for a session."""
        return self.memory_service.get_memory(session_id)

    def save_memory(self, session_id: str, text: str, role: str, importance: float = 1.0, telemetry: dict = None):
        """Saves a turn to active memory and database."""
        self.memory_service.save_memory(session_id, text, role, importance, telemetry)

    # ------------------------------------------------------------------
    # Registry query detection helper
    # ------------------------------------------------------------------

    def _is_registry_query(self, query: str) -> bool:
        """Detects if query asks about available documents/sources."""
        q = query.lower().strip()
        registry_patterns = [
            "what document", "which document", "list document", "show document", "available document",
            "documents do you", "documents contain", "document do you contain",
            "what file", "which file", "list file", "show file", "show me file", "show me files", "available file", "files do you", "files contain",
            "what dataset", "which dataset", "list dataset", "show dataset", "available dataset", "datasets do you",
            "what database", "which database", "list database", "show database", "available database", "databases do you",
            "what schema", "which schema", "list schema", "show schema", "available schema", "schemas do you",
            "what source", "which source", "list source", "show source", "show me source", "show me sources", "available source", "sources do you",
            "your document", "your file", "your dataset", "your database", "your schema", "your source",
            "my document", "my file", "my dataset", "my database", "my schema", "my source"
        ]
        words = q.split()
        if "show" in words and "me" in words:
            try:
                me_idx = words.index("me")
                for offset in range(1, 4):
                    if me_idx + offset < len(words) and ("source" in words[me_idx + offset] or "sources" in words[me_idx + offset]):
                        return True
            except ValueError:
                pass
        return any(pat in q for pat in registry_patterns)

    def _get_registry_context_text(self) -> str:
        """Returns formatted registry text for querying what documents are available."""
        try:
            summary = self.registry.get_registry_summary()
            sources = summary.get("sources", [])
            total_docs = summary.get("total_documents_count", 0)
            
            if not sources:
                return "I don't have any documents indexed at the moment. You can upload documents via the /upload endpoint."
            
            result = ["### AVAILABLE DOCUMENTS ###\n"]
            for i, src in enumerate(sources, 1):
                result.append(f"{i}. {src}")
            result.append(f"\nTotal indexed documents: {total_docs}")
            return "\n".join(result)
        except Exception as e:
            logger.error(f"Error getting registry context: {e}")
            return "Unable to retrieve document listing at this time."

    # ------------------------------------------------------------------
    # Pipeline phases (Synchronous compatibility)
    # ------------------------------------------------------------------

    def _phase_expand(self, query: str, mode: str, latencies: dict) -> list:
        t = time.time()
        if mode == "context_engine":
            if len(query.split()) < EXPANSION_MIN_WORDS:
                logger.info("[P1: EXPANSION] Skip: Query is direct.")
                queries = [query]
            else:
                logger.info("[P1: EXPANSION] Generating semantic variations...")
                queries = self.expander.expand(query)
                logger.info(f" -> Generated {len(queries) - 1} variations")
        else:
            logger.info("[P1: EXPANSION] Disabled: base query only.")
            queries = [query]
        latencies['phase_1_expansion_ms'] = round((time.time() - t) * 1000, 2)
        return queries

    def _phase_hyde(self, query: str, mode: str, search_queries: list, latencies: dict) -> str:
        t = time.time()
        hyde_doc = ""
        if mode == "context_engine":
            logger.info("[P1.5: HyDE] Generating hypothetical document...")
            hyde_doc = self.hyde.generate_hypothetical_doc(query)
            logger.info(f" -> HyDE: {hyde_doc[:60]}...")
            search_queries.append(hyde_doc)
        latencies['phase_1_5_hyde_ms'] = round((time.time() - t) * 1000, 2)
        return hyde_doc

    def _phase_retrieve(self, search_queries: list, top_k: int, source_filter, latencies: dict) -> list:
        results, embed_total, db_total, total_ms = self.retrieval_service.retrieve(
            search_queries, top_k, source_filter
        )
        latencies['phase_2_retrieval_ms'] = total_ms
        latencies['phase_2_embed_generation_ms'] = embed_total
        latencies['phase_2_weaviate_search_ms'] = db_total
        return results

    def _phase_refine(self, query: str, mode: str, memory, all_raw_results: list, top_k: int, latencies: dict):
        peak_score = 0.0

        top_raw_score = float(all_raw_results[0]["score"]) if all_raw_results else 0.0

        if mode == "context_engine":
            # Phase 3: Reranking - conditionally skip if high confidence
            t = time.time()
            if self.pipeline_config.enable_reranking and top_raw_score < 0.5:
                try:
                    logger.info("[P3: RERANKING] Applying Cross-Encoder...")
                    reranked = self.reranker.rerank(query, all_raw_results)[:top_k]
                    peak_score = float(reranked[0]['cross_score']) if reranked else 0.0
                    logger.info(f" -> Peak score: {peak_score:.4f}")
                except Exception as e:
                    logger.warning(f" !! Reranking failed: {e}. Falling back.")
                    reranked = sorted(all_raw_results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
            else:
                logger.info(f"[P3: RERANKING] Skipped: high confidence ({top_raw_score:.2f}) or disabled.")
                reranked = sorted(all_raw_results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
                peak_score = top_raw_score
            latencies['phase_3_reranking_ms'] = round((time.time() - t) * 1000, 2)

            # Phase 4: Memory
            t = time.time()
            logger.info("[P4: MEMORY] Synchronizing history...")
            memory_text = memory.get_active_context()
            memory_tokens = count_tokens(memory_text)
            latencies['phase_4_memory_sync_ms'] = round((time.time() - t) * 1000, 2)

            # Phase 5: Compression - conditionally skip if high confidence
            t = time.time()
            if self.pipeline_config.enable_compression and peak_score < 0.7:
                doc_budget = max(MIN_KNOWLEDGE_BUDGET, TOTAL_CONTEXT_BUDGET - memory_tokens)
                logger.info(f"[P5: COMPRESSION] Budget: {doc_budget} tokens...")
                compressed = self.compressor.compress(
                    [r["text"] for r in reranked], query, max_tokens=doc_budget
                )
                
                compressed_segments = compressed.split("\n\n")
                formatted_parts = []
                for seg in compressed_segments:
                    seg_strip = seg.strip()
                    if not seg_strip:
                        continue
                    source = "unknown"
                    for r in reranked:
                        if seg_strip in r["text"]:
                            source = r.get("source", "unknown")
                            break
                    seg_sanitized = sanitize_document_text(seg_strip)
                    formatted_parts.append(f'<document source="{source}">\n{seg_sanitized}\n</document>')
                
                final_knowledge = "\n\n".join(formatted_parts)
                doc_tokens = count_tokens(final_knowledge)
                final_context = f"### MEMORY\n{memory_text}\n\n### KNOWLEDGE\n{final_knowledge}"

                # Update stats via Telemetry
                raw_len = sum(len(r["text"]) for r in all_raw_results) + 1
                ratio = len(compressed) / raw_len
                self.telemetry.update_compression_ratio(ratio)
                logger.info(f" -> Compression ratio: {ratio:.2%}")
                latencies['phase_5_compression_ms'] = round((time.time() - t) * 1000, 2)

                return final_context, memory_text, final_knowledge, ratio, memory_tokens, doc_tokens, doc_budget, peak_score
            else:
                logger.info(f"[P5: COMPRESSION] Skipped: high confidence ({peak_score:.2f}) or disabled.")
                formatted_parts = []
                for r in reranked[:top_k]:
                    source = r.get("source", "unknown")
                    text_sanitized = sanitize_document_text(r["text"])
                    formatted_parts.append(f'<document source="{source}">\n{text_sanitized}\n</document>')
                
                final_knowledge = "\n\n".join(formatted_parts)
                doc_tokens = count_tokens(final_knowledge)
                final_context = f"### MEMORY\n{memory_text}\n\n### KNOWLEDGE\n{final_knowledge}"
                ratio = 0.0
                latencies['phase_5_compression_ms'] = 0.0

                return final_context, memory_text, final_knowledge, ratio, memory_tokens, doc_tokens, TOTAL_CONTEXT_BUDGET, peak_score

        else:
            # Simple RAG bypass
            logger.info("[P3-5: BYPASS] Simple RAG mode.")
            formatted_parts = []
            for r in all_raw_results[:top_k]:
                source = r.get("source", "unknown")
                text_sanitized = sanitize_document_text(r["text"])
                formatted_parts.append(f'<document source="{source}">\n{text_sanitized}\n</document>')
            
            raw_context = "\n\n".join(formatted_parts)
            if len(raw_context) > SAFETY_CHAR_LIMIT:
                logger.warning(f" -> Truncating ({len(raw_context)} chars).")
                raw_context = raw_context[:SAFETY_CHAR_LIMIT] + "... [TRUNCATED]"

            latencies['phase_3_reranking_ms'] = 0.0
            latencies['phase_4_memory_sync_ms'] = 0.0
            latencies['phase_5_compression_ms'] = 0.0

            return (
                "### KNOWLEDGE\n" + raw_context, "N/A", "N/A", 1.0,
                0, count_tokens(raw_context), TOTAL_CONTEXT_BUDGET, 0.0
            )

    def _phase_generate(self, query: str, final_context: str, latencies: dict, context_chunks: List[str] = None):
        response, prompt, exact_tokens, ctx_used_pct, grounding_score = self.generation_service.generate(
            query, final_context, count_tokens, context_chunks
        )
        latencies['grounding_score'] = grounding_score
        return response, prompt, exact_tokens, ctx_used_pct, grounding_score

    def _phase_generate_stream(self, query: str, final_context: str, latencies: dict) -> Generator[Dict, None, None]:
        t = time.time()
        yield from self.generation_service.generate_stream(query, final_context)
        latencies['phase_6_generation_ms'] = round((time.time() - t) * 1000, 2)

    def _handle_context_overflow(self, query: str, final_context: str, memory_text: str,
                                 compressed_docs: str, memory, all_raw: list,
                                 context_limit: int) -> tuple:
        return self.overflow_service.handle_context_overflow(
            query, final_context, memory_text, compressed_docs, memory, all_raw, context_limit, count_tokens
        )

    # ------------------------------------------------------------------
    # Pipeline phases (Asynchronous native)
    # ------------------------------------------------------------------

    async def _phase_expand_async(self, query: str, mode: str, latencies: dict) -> list:
        t = time.time()
        if mode == "context_engine":
            if len(query.split()) < EXPANSION_MIN_WORDS:
                logger.info("[P1: EXPANSION] Skip: Query is direct.")
                queries = [query]
            else:
                logger.info("[P1: EXPANSION] Generating semantic variations...")
                queries = await self.expander.expand_async(query)
                logger.info(f" -> Generated {len(queries) - 1} variations")
        else:
            logger.info("[P1: EXPANSION] Disabled: base query only.")
            queries = [query]
        latencies['phase_1_expansion_ms'] = round((time.time() - t) * 1000, 2)
        return queries

    async def _phase_hyde_async(self, query: str, mode: str, latencies: dict) -> str:
        t = time.time()
        hyde_doc = ""
        if mode == "context_engine":
            logger.info("[P1.5: HyDE] Generating hypothetical document...")
            hyde_doc = await self.hyde.generate_hypothetical_doc_async(query)
            logger.info(f" -> HyDE: {hyde_doc[:60]}...")
        latencies['phase_1_5_hyde_ms'] = round((time.time() - t) * 1000, 2)
        return hyde_doc

    async def _phase_retrieve_async(self, search_queries: list, top_k: int, source_filter, latencies: dict) -> list:
        results, embed_total, db_total, total_ms = await self.retrieval_service.retrieve_async(
            search_queries, top_k, source_filter
        )
        latencies['phase_2_retrieval_ms'] = total_ms
        latencies['phase_2_embed_generation_ms'] = embed_total
        latencies['phase_2_weaviate_search_ms'] = db_total
        return results

    async def _phase_refine_async(self, query: str, mode: str, memory, all_raw_results: list, top_k: int, latencies: dict):
        peak_score = 0.0

        # Check if we have any results to determine confidence
        top_raw_score = float(all_raw_results[0]["score"]) if all_raw_results else 0.0
        high_confidence = top_raw_score > 0.5

        if mode == "context_engine":
            # Phase 3: Reranking - conditionally skip if high confidence
            t = time.time()
            if self.pipeline_config.enable_reranking and not high_confidence:
                try:
                    logger.info("[P3: RERANKING] Applying Cross-Encoder (Async)...")
                    reranked = await asyncio.to_thread(self.reranker.rerank, query, all_raw_results)
                    reranked = reranked[:top_k]
                    peak_score = float(reranked[0]['cross_score']) if reranked else 0.0
                    logger.info(f" -> Peak score: {peak_score:.4f}")
                except Exception as e:
                    logger.warning(f" !! Reranking failed: {e}. Falling back.")
                    reranked = sorted(all_raw_results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
            else:
                logger.info(f"[P3: RERANKING] Skipped: high confidence ({top_raw_score:.2f}) or disabled.")
                reranked = sorted(all_raw_results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
                peak_score = top_raw_score
            latencies['phase_3_reranking_ms'] = round((time.time() - t) * 1000, 2)

            # Phase 4: Memory
            t = time.time()
            logger.info("[P4: MEMORY] Synchronizing history...")
            memory_text = memory.get_active_context()
            memory_tokens = count_tokens(memory_text)
            latencies['phase_4_memory_sync_ms'] = round((time.time() - t) * 1000, 2)

            # Phase 5: Compression - conditionally skip if high confidence and small context
            t = time.time()
            if self.pipeline_config.enable_compression and peak_score < 0.7:
                doc_budget = max(MIN_KNOWLEDGE_BUDGET, TOTAL_CONTEXT_BUDGET - memory_tokens)
                logger.info(f"[P5: COMPRESSION] Budget: {doc_budget} tokens...")
                compressed = await asyncio.to_thread(
                    self.compressor.compress, [r["text"] for r in reranked], query, max_tokens=doc_budget
                )
                
                compressed_segments = compressed.split("\n\n")
                formatted_parts = []
                for seg in compressed_segments:
                    seg_strip = seg.strip()
                    if not seg_strip:
                        continue
                    source = "unknown"
                    for r in reranked:
                        if seg_strip in r["text"]:
                            source = r.get("source", "unknown")
                            break
                    seg_sanitized = sanitize_document_text(seg_strip)
                    formatted_parts.append(f'<document source="{source}">\n{seg_sanitized}\n</document>')
                
                final_knowledge = "\n\n".join(formatted_parts)
                doc_tokens = count_tokens(final_knowledge)
                final_context = f"### MEMORY\n{memory_text}\n\n### KNOWLEDGE\n{final_knowledge}"

                # Update stats via Telemetry
                raw_len = sum(len(r["text"]) for r in all_raw_results) + 1
                ratio = len(compressed) / raw_len
                self.telemetry.update_compression_ratio(ratio)
                logger.info(f" -> Compression ratio: {ratio:.2%}")
                latencies['phase_5_compression_ms'] = round((time.time() - t) * 1000, 2)

                return final_context, memory_text, final_knowledge, ratio, memory_tokens, doc_tokens, doc_budget, peak_score
            else:
                logger.info(f"[P5: COMPRESSION] Skipped: high confidence ({peak_score:.2f}) or disabled.")
                # Use raw results without compression
                formatted_parts = []
                for r in reranked[:top_k]:
                    source = r.get("source", "unknown")
                    text_sanitized = sanitize_document_text(r["text"])
                    formatted_parts.append(f'<document source="{source}">\n{text_sanitized}\n</document>')
                
                final_knowledge = "\n\n".join(formatted_parts)
                doc_tokens = count_tokens(final_knowledge)
                final_context = f"### MEMORY\n{memory_text}\n\n### KNOWLEDGE\n{final_knowledge}"
                ratio = 0.0
                latencies['phase_5_compression_ms'] = 0.0

                return final_context, memory_text, final_knowledge, ratio, memory_tokens, doc_tokens, TOTAL_CONTEXT_BUDGET, peak_score

        else:
            # Simple RAG bypass
            logger.info("[P3-5: BYPASS] Simple RAG mode.")
            formatted_parts = []
            for r in all_raw_results[:top_k]:
                source = r.get("source", "unknown")
                text_sanitized = sanitize_document_text(r["text"])
                formatted_parts.append(f'<document source="{source}">\n{text_sanitized}\n</document>')
            
            raw_context = "\n\n".join(formatted_parts)
            if len(raw_context) > SAFETY_CHAR_LIMIT:
                logger.warning(f" -> Truncating ({len(raw_context)} chars).")
                raw_context = raw_context[:SAFETY_CHAR_LIMIT] + "... [TRUNCATED]"

            latencies['phase_3_reranking_ms'] = 0.0
            latencies['phase_4_memory_sync_ms'] = 0.0
            latencies['phase_5_compression_ms'] = 0.0

            return (
                "### KNOWLEDGE\n" + raw_context, "N/A", "N/A", 1.0,
                0, count_tokens(raw_context), TOTAL_CONTEXT_BUDGET, 0.0
            )

    async def _phase_generate_async(self, query: str, final_context: str, latencies: dict, context_chunks: List[str] = None):
        t = time.time()
        response, prompt, exact_tokens, ctx_used_pct, grounding_score = await self.generation_service.generate_async(
            query, final_context, count_tokens, context_chunks
        )
        latencies['phase_6_generation_ms'] = round((time.time() - t) * 1000, 2)
        latencies['grounding_score'] = grounding_score
        return response, prompt, exact_tokens, ctx_used_pct, grounding_score

    async def _phase_generate_stream_async(self, query: str, final_context: str, latencies: dict) -> AsyncGenerator[Dict, None]:
        t = time.time()
        async for chunk in self.generation_service.generate_stream_async(query, final_context):
            yield chunk
        latencies['phase_6_generation_ms'] = round((time.time() - t) * 1000, 2)

    # ------------------------------------------------------------------
    # Public API (Synchronous backward compatibility wrappers)
    # ------------------------------------------------------------------

    def ask(self, query: str, session_id: str = "default", mode: str = "context_engine",
            source_filter: str = None, top_k: int = 5, context_limit: Optional[int] = None) -> Dict:
        """
        The primary entry point for querying the RAG system synchronously.
        """
        # Run the async ask method inside asyncio run loop (safe outside FastAPI loops)
        try:
            return asyncio.run(self.ask_async(
                query, session_id, mode, source_filter, top_k, context_limit
            ))
        except RuntimeError:
            # Fallback if loop is already running (e.g. in some nested setups)
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.ask_async(
                query, session_id, mode, source_filter, top_k, context_limit
            ))

    def ask_stream(self, query: str, session_id: str = "default", mode: str = "context_engine",
                   source_filter: str = None, top_k: int = 5, context_limit: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        Streaming query endpoint synchronously wrapping the async implementation.
        """
        loop = asyncio.new_event_loop()
        try:
            async_gen = self.ask_stream_async(
                query, session_id, mode, source_filter, top_k, context_limit
            )
            while True:
                try:
                    event = loop.run_until_complete(async_gen.__anext__())
                    yield event
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Public API (Asynchronous native implementation)
    # ------------------------------------------------------------------

    async def ask_async(self, query: str, session_id: str = "default", mode: str = "context_engine",
                  source_filter: str = None, top_k: int = 5, context_limit: Optional[int] = None) -> Dict:
        """
        The primary async entry point for querying the RAG system.
        """
        logger.info(f"\n[INIT ASYNC] Query: '{query[:60]}...' | Session: {session_id} | Mode: {mode} | Limit: {context_limit}")
        self.telemetry.increment_queries()
        
        # Early exit for registry queries
        if self._is_registry_query(query):
            logger.info("Early exit: Registry query detected.")
            response = self._get_registry_context_text()
            self.save_memory(session_id, query, "user")
            self.save_memory(session_id, response, "assistant", 0.8)
            sys_metrics = self.telemetry.get_system_metrics()
            return {
                "query": query,
                "response": response,
                "mode": mode,
                "search_queries": [query],
                "raw_prompt": response,
                "retrieved_context": [],
                "compressed_context": response,
                "memory_context": "",
                "hyde_doc": "",
                "tps": 0.0,
                "query_cost": "$0.00000000",
                "stats": {
                    "compression_ratio": 1.0,
                    "avg_compression_ratio": round(self.stats["avg_compression_ratio"], 3),
                    "queries_handled": self.stats["queries"],
                    "active_memories": 0,
                    "instantaneous_latency_ms": {"total_execution_ms": 0.0},
                    "avg_latency_ms": round(self.stats["avg_latency_ms"], 2),
                    "cpu_usage_percent": sys_metrics["cpu"],
                    "memory_usage_percent": sys_metrics["ram"],
                    "context_used_percent": 0.0,
                    "exact_tokens": {"prompt": 0, "completion": 0, "total": 0},
                    "budget_tracking": {"memory_tokens_used": 0, "memory_tokens_limit": MEMORY_TOKEN_BUDGET, "document_tokens_used": 0, "document_tokens_limit": TOTAL_CONTEXT_BUDGET},
                    "overflow_telemetry": {"overflow_occurred": False, "limit": None, "initial_tokens": 0, "final_tokens": 0, "steps": []},
                    "mode": mode,
                    "alpha": 0.5,
                    "reranker_peak_score": 0.0
                }
}
         
        memory = self.get_memory(session_id)
        latencies = {}
        t_start = time.time()

        search_queries = [query]
        hyde_doc = ""

        # Initial quick retrieval to assess confidence
        if mode == "context_engine":
            initial_results, _, _, _ = await self.retrieval_service.retrieve_async([query], top_k=1, source_filter=source_filter)
            top_score = float(initial_results[0]["score"]) if initial_results else 0.0
            
            # Conditionally run expensive features based on confidence
            if self.pipeline_config.should_use_full_pipeline(top_score):
                logger.info(f"[P1: EXPANSION] Low confidence ({top_score:.2f}), running full pipeline...")
                # Concurrent expansion and HyDE Tasks
                expand_task = asyncio.create_task(self._phase_expand_async(query, mode, latencies))
                hyde_task = asyncio.create_task(self._phase_hyde_async(query, mode, latencies))
                
                search_queries = await expand_task
                hyde_doc = await hyde_task
                if hyde_doc:
                    search_queries.append(hyde_doc)
            else:
                logger.info(f"[P1: EXPANSION] High confidence ({top_score:.2f}), skipping expansion/HyDE for speed.")
        
        # Retrieve in parallel
        all_raw = await self._phase_retrieve_async(search_queries, top_k, source_filter, latencies)
        
        # Refine context
        (final_context, memory_text, compressed_docs, ratio,
         mem_tokens, doc_tokens, doc_budget, peak_score) = await self._phase_refine_async(
            query, mode, memory, all_raw, top_k, latencies
        )

        # Apply overflow handling
        overflow_occurred = False
        overflow_steps = []
        initial_tokens = mem_tokens + doc_tokens + 350
        final_prompt_tokens = initial_tokens
        if context_limit:
            (final_context, memory_text, compressed_docs, overflow_occurred, overflow_steps,
             initial_tokens, final_prompt_tokens, mem_tokens, doc_tokens) = await self.overflow_service.handle_context_overflow_async(
                query, final_context, memory_text, compressed_docs, memory, all_raw, context_limit, count_tokens
            )

        context_chunks = [r["text"] for r in all_raw[:top_k]]
        response, prompt, exact_tokens, ctx_used_pct, grounding_score = await self._phase_generate_async(query, final_context, latencies, context_chunks)

        # Persist interaction
        self.save_memory(session_id, query, "user")
        
        telemetry_data = {
            "query": query,
            "raw_prompt": f"### CONTEXT:\n{final_context}\n\n### QUESTION:\n{query}\n\n### ANSWER:",
            "overflow_occurred": overflow_occurred,
            "limit": context_limit,
            "initial_tokens": initial_tokens,
            "final_tokens": final_prompt_tokens,
            "steps": overflow_steps,
            "budget_tracking": {
                "memory_tokens_used": mem_tokens,
                "memory_tokens_limit": MEMORY_TOKEN_BUDGET,
                "document_tokens_used": doc_tokens,
                "document_tokens_limit": doc_budget
            },
            "compression_ratio": round(ratio, 3)
        }
        self.save_memory(session_id, response, "assistant", 0.8, telemetry=telemetry_data)

        # Compute telemetry
        total_ms = round((time.time() - t_start) * 1000, 2)
        latencies['total_execution_ms'] = total_ms
        gen_sec = latencies.get('phase_6_generation_ms', 0.0) / 1000.0
        tps = round(exact_tokens["completion"] / gen_sec, 1) if gen_sec > 0 else 0.0
        cost = self.telemetry.compute_cost(exact_tokens["prompt"], exact_tokens["completion"])

        self.telemetry.update_latency(total_ms)
        logger.info(f"[FINISH] Complete async in {total_ms}ms.\n")

        sys_metrics = self.telemetry.get_system_metrics()

        return {
            "query": query,
            "response": response,
            "mode": mode,
            "search_queries": search_queries,
            "raw_prompt": prompt,
            "retrieved_context": all_raw[:top_k],
            "compressed_context": compressed_docs,
            "memory_context": memory_text,
            "hyde_doc": hyde_doc,
            "tps": tps,
            "query_cost": f"${cost:.8f}",
            "stats": {
                "compression_ratio": round(ratio, 3),
                "avg_compression_ratio": round(self.stats["avg_compression_ratio"], 3),
                "queries_handled": self.stats["queries"],
                "active_memories": len(memory.entries),
                "instantaneous_latency_ms": latencies,
                "avg_latency_ms": round(self.stats["avg_latency_ms"], 2),
                "cpu_usage_percent": sys_metrics["cpu"],
                "memory_usage_percent": sys_metrics["ram"],
                "context_used_percent": ctx_used_pct,
                "exact_tokens": exact_tokens,
                "budget_tracking": {
                    "memory_tokens_used": mem_tokens,
                    "memory_tokens_limit": MEMORY_TOKEN_BUDGET,
                    "document_tokens_used": doc_tokens,
                    "document_tokens_limit": doc_budget
                },
                "overflow_telemetry": {
                    "overflow_occurred": overflow_occurred,
                    "limit": context_limit,
                    "initial_tokens": initial_tokens,
                    "final_tokens": final_prompt_tokens,
                    "steps": overflow_steps
                },
                "mode": mode,
                "alpha": getattr(self.retriever, "alpha", 0.5),
                "reranker_peak_score": round(peak_score, 4),
                "grounding_score": round(grounding_score, 4)
            }
        }

    async def ask_stream_async(self, query: str, session_id: str = "default", mode: str = "context_engine",
                         source_filter: str = None, top_k: int = 5, context_limit: Optional[int] = None) -> AsyncGenerator[Dict, None]:
        """
        Asynchronous streaming query endpoint. Yields progress updates and LLM output tokens.
        """
        from typing import Dict as TypedDict
        import time as time_module
        
        if mode == "agentic":
            async for event in self.agent.run_stream_async(query, session_id, source_filter, context_limit=context_limit):
                yield event
            return
        
        # Early exit for registry queries
        if self._is_registry_query(query):
            logger.info("Early exit: Registry query detected (stream).")
            yield {"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"}
            registry_text = self._get_registry_context_text()
            for i in range(0, len(registry_text), 20):
                yield {"event": "answer_chunk", "text": registry_text[i:i+20]}
                await asyncio.sleep(0.01)
            self.save_memory(session_id, query, "user")
            self.save_memory(session_id, registry_text, "assistant", 0.8)
            yield {"event": "done", "response": registry_text, "stats": {
                "compression_ratio": 1.0,
                "overflow_telemetry": {"overflow_occurred": False, "limit": None, "initial_tokens": 0, "final_tokens": 0, "steps": []},
                "budget_tracking": {"memory_tokens_used": 0, "memory_tokens_limit": 1500, "document_tokens_used": 0, "document_tokens_limit": 0},
                "exact_tokens": {"prompt": 0, "completion": 0, "total": 0},
                "mode": mode
            }}
            return

        logger.info(f"\n[INIT STREAM ASYNC] Query: '{query[:60]}...' | Session: {session_id} | Mode: {mode} | Limit: {context_limit}")
        self.telemetry.increment_queries()
        memory = self.get_memory(session_id)
        latencies = {}
        t_start = time.time()

        # Step 1: Expand - conditionally based on confidence
        search_queries = [query]
        hyde_doc = ""
        
        if mode == "context_engine":
            # Quick check for confidence
            initial_results, _, _, _ = await self.retrieval_service.retrieve_async([query], top_k=1, source_filter=source_filter)
            top_score = float(initial_results[0]["score"]) if initial_results else 0.0
            
            if self.pipeline_config.should_use_full_pipeline(top_score):
                yield {"event": "thought", "text": "Low confidence detected, expanding query and generating HyDE document..."}
                expand_task = asyncio.create_task(self._phase_expand_async(query, mode, latencies))
                hyde_task = asyncio.create_task(self._phase_hyde_async(query, mode, latencies))
                
                search_queries = await expand_task
                hyde_doc = await hyde_task
                if hyde_doc:
                    search_queries.append(hyde_doc)
                yield {"event": "action", "tool": "Query Expansion & HyDE", "input": f"Variations: {search_queries}"}
            else:
                yield {"event": "thought", "text": f"High confidence ({top_score:.2f}), using fast path..."}
        else:
            yield {"event": "thought", "text": "Analyzing query..."}

        # Step 2: Retrieve
        yield {"event": "thought", "text": "Retrieving contextually relevant documents from Vector Database concurrently..."}
        all_raw = await self._phase_retrieve_async(search_queries, top_k, source_filter, latencies)
        yield {"event": "observation", "output": f"Retrieved {len(all_raw)} document chunks."}

        # Step 3-5: Refine (Reranking, Memory Sync, Compression)
        yield {"event": "thought", "text": "Reranking document candidates and performing dynamic memory decay budget sizing..."}
        (final_context, memory_text, compressed_docs, ratio,
         mem_tokens, doc_tokens, doc_budget, peak_score) = await self._phase_refine_async(
            query, mode, memory, all_raw, top_k, latencies
         )

        # Apply overflow handling
        overflow_occurred = False
        overflow_steps = []
        initial_tokens = mem_tokens + doc_tokens + 350
        final_prompt_tokens = initial_tokens
        
        if context_limit:
            (final_context, memory_text, compressed_docs, overflow_occurred, overflow_steps,
             initial_tokens, final_prompt_tokens, mem_tokens, doc_tokens) = await self.overflow_service.handle_context_overflow_async(
                query, final_context, memory_text, compressed_docs, memory, all_raw, context_limit, count_tokens
            )

        if overflow_occurred:
            yield {
                "event": "overflow_detected",
                "limit": context_limit,
                "initial": initial_tokens,
                "final": final_prompt_tokens,
                "steps": overflow_steps
            }
            for step in overflow_steps:
                yield {"event": "overflow_step", "text": step}
                await asyncio.sleep(0.1)

        yield {"event": "observation", "output": f"Reranking score: {peak_score:.4f}. Context compressed down by {round((1-ratio)*100)}%."}

        # Step 6: Generation (yielding chunks)
        yield {"event": "thought", "text": "Generating final grounded answer from compressed context..."}
        
        accumulated_response = ""
        async for chunk in self._phase_generate_stream_async(query, final_context, latencies):
            if chunk["event"] == "answer_chunk":
                accumulated_response += chunk["text"]
            yield chunk

        # Persist interaction
        self.save_memory(session_id, query, "user")
        
        telemetry_data = {
            "query": query,
            "raw_prompt": f"### CONTEXT:\n{final_context}\n\n### QUESTION:\n{query}\n\n### ANSWER:",
            "overflow_occurred": overflow_occurred,
            "limit": context_limit,
            "initial_tokens": initial_tokens,
            "final_tokens": final_prompt_tokens,
            "steps": overflow_steps,
            "budget_tracking": {
                "memory_tokens_used": mem_tokens,
                "memory_tokens_limit": MEMORY_TOKEN_BUDGET,
                "document_tokens_used": doc_tokens,
                "document_tokens_limit": doc_budget
            },
            "compression_ratio": round(ratio, 3)
        }
        self.save_memory(session_id, accumulated_response, "assistant", 0.8, telemetry=telemetry_data)

        total_ms = round((time.time() - t_start) * 1000, 2)
        latencies['total_execution_ms'] = total_ms
        self.telemetry.update_latency(total_ms)

        sys_metrics = self.telemetry.get_system_metrics()

        yield {"event": "done", "response": accumulated_response, "stats": {
            "compression_ratio": round(ratio, 3),
            "avg_compression_ratio": round(self.stats["avg_compression_ratio"], 3),
            "queries_handled": self.stats["queries"],
            "active_memories": len(memory.entries),
            "instantaneous_latency_ms": latencies,
            "avg_latency_ms": round(self.stats["avg_latency_ms"], 2),
            "cpu_usage_percent": sys_metrics["cpu"],
            "memory_usage_percent": sys_metrics["ram"],
            "overflow_telemetry": {
                "overflow_occurred": overflow_occurred,
                "limit": context_limit,
                "initial_tokens": initial_tokens,
                "final_tokens": final_prompt_tokens,
                "steps": overflow_steps
            },
            "budget_tracking": {
                "memory_tokens_used": mem_tokens,
                "memory_tokens_limit": MEMORY_TOKEN_BUDGET,
                "document_tokens_used": doc_tokens,
                "document_tokens_limit": doc_budget
            },
            "retrieved_context": all_raw[:top_k],
            "raw_prompt": f"### CONTEXT:\n{final_context}\n\n### QUESTION:\n{query}\n\n### ANSWER:"
        }}

    def close(self):
        """Cleanup resources."""
        self.retriever.close()
