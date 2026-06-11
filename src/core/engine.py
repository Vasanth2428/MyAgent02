"""
RAG Context Engine - The Brain of the System

This engine coordinates all the steps needed to answer your questions:
1. Memory Service - Remembers what was said in the conversation
2. Retrieval Service - Finds relevant document chunks from the database
3. Generation Service - Asks the AI to write the answer
4. Context Overflow Service - Handles situations when there's too much context
5. Telemetry Service - Tracks performance and costs

Think of it like a librarian who remembers your conversation history, finds relevant
books, and helps you craft a response to your question.
"""

import logging
import time
import asyncio
from typing import Dict, Generator, AsyncGenerator, Optional, Any, List
import tiktoken
from src.core.llm import LLMService

from src.core.config import (
    LLM_MODEL, LLM_TEMPERATURE, CONTEXT_WINDOW_LIMIT,
    TOKENIZER_ENCODING, TOTAL_CONTEXT_BUDGET, MEMORY_TOKEN_BUDGET,
    MIN_KNOWLEDGE_BUDGET, MAX_CANDIDATES, EXPANSION_MIN_WORDS,
    SAFETY_CHAR_LIMIT, COST_PER_INPUT_TOKEN, COST_PER_OUTPUT_TOKEN,
    PipelineConfig
)
from src.core.retriever import WeaviateRetriever
from src.core.persistence import PersistentMemoryStore
from src.core.compressor import Compressor
from src.core.reranker import NeuralReranker
from src.core.expander import QueryExpander
from src.core.hyde import HyDEGenerator
from src.core.registry import KnowledgeRegistry
from src.core.services import RetrievalService, MemoryService, GenerationService, ContextOverflowService, TelemetryService
from src.core.services.generation_service import GenerationResult
from src.core.security import sanitize_document_text

logger = logging.getLogger("RAG.Engine")
tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    """Counts exact BPE tokens for a given string."""
    if not text:
        return 0
    return len(tokenizer.encode(text))


class RAGContextEngine:
    """
    The main orchestrator that handles question answering.
    
    This class brings everything together - it finds relevant documents from your
    knowledge base, remembers your conversation, and generates helpful answers.
    It can work synchronously (one step at a time) or asynchronously (multiple steps
    happening at once for better performance).
    """

    def __init__(self, retriever: WeaviateRetriever, pipeline_config: Optional[PipelineConfig] = None, checkpointer: Optional[any] = None):
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


        # Initialize Multi-Agent Graph
        from src.graph.workflow import build_multi_agent_graph
        if checkpointer is not None:
            self.multi_agent_checkpointer = checkpointer
        else:
            from src.graph.checkpointer import setup_checkpointer
            self.multi_agent_checkpointer = setup_checkpointer()
        self.multi_agent_graph = build_multi_agent_graph(self.multi_agent_checkpointer)

    # ------------------------------------------------------------------
    # Memory Delegations
    # ------------------------------------------------------------------

    def get_memory(self, session_id: str):
        """Retrieves memory for a session."""
        return self.memory_service.get_memory(session_id)

    def delete_session(self, session_id: str) -> None:
        """Deletes a session's history from memory.db and its checkpoints from checkpoints.db."""
        logger.info(f"Deleting session history and checkpoints for session: {session_id}")
        self.persistent_memory.delete_session(session_id)
        self._clear_session_checkpoints(session_id)

    def _clear_session_checkpoints(self, session_id: str):
        """Clears all LangGraph checkpoints for the given session ID from the SQLite checkpointer database."""
        import os
        import sqlite3
        from src.graph.checkpointer import validate_db_path
        
        db_path = os.getenv("CHECKPOINTER_DB_PATH", "checkpoints.db")
        db_path = validate_db_path(db_path)
        safe_dir = os.path.join(os.getcwd(), 'checkpoints')
        full_path = os.path.join(safe_dir, db_path)
        
        if not os.path.exists(full_path):
            return
            
        try:
            with sqlite3.connect(full_path, timeout=10.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
                conn.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
                conn.commit()
                logger.info(f"Cleared checkpoints for thread {session_id} from {full_path}")
        except Exception as e:
            logger.error(f"Error clearing checkpoints for session {session_id}: {e}")

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
        result: GenerationResult = self.generation_service.generate(
            query, final_context, count_tokens, context_chunks
        )
        latencies['grounding_score'] = result.grounding_score
        return result.response, result.prompt, result.token_usage, result.context_used_percent, result.grounding_score

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
            
            t = time.time()
            logger.info("[P4: MEMORY] Synchronizing history for Simple RAG...")
            memory_text = memory.get_active_context()
            memory_tokens = count_tokens(memory_text)
            latencies['phase_4_memory_sync_ms'] = round((time.time() - t) * 1000, 2)
            
            latencies['phase_5_compression_ms'] = 0.0

            final_context = f"### MEMORY\n{memory_text}\n\n### KNOWLEDGE\n{raw_context}"

            return (
                final_context, memory_text, "N/A", 1.0,
                memory_tokens, count_tokens(raw_context), TOTAL_CONTEXT_BUDGET, 0.0
            )

    async def _phase_generate_async(self, query: str, final_context: str, latencies: dict, context_chunks: List[str] = None):
        t = time.time()
        result: GenerationResult = await self.generation_service.generate_async(
            query, final_context, count_tokens, context_chunks
        )
        latencies['phase_6_generation_ms'] = round((time.time() - t) * 1000, 2)
        latencies['grounding_score'] = result.grounding_score
        return result.response, result.prompt, result.token_usage, result.context_used_percent, result.grounding_score

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
                    "reranker_peak_score": 0.0
                }
            }
         
        # Multi-Agent Mode Execution
        if mode == "agentic":
            # Clear checkpoints in checkpoints.db for this session to avoid duplication/stale state
            self._clear_session_checkpoints(session_id)

            from langchain_core.messages import HumanMessage, AIMessage
            from src.graph.workflow import get_graph_config
            import time as time_module
            
            t_start = time_module.time()
            config = get_graph_config(session_id)
            
            # Fetch history from memory.db
            memory_history = self.persistent_memory.get_history(session_id)
            messages = []
            for entry in memory_history:
                role = entry.get("role")
                text = entry.get("text")
                if role == "user":
                    messages.append(HumanMessage(content=text))
                elif role == "assistant":
                    messages.append(AIMessage(content=text))
            
            # Append the current query
            messages.append(HumanMessage(content=query))

            initial_state = {
                "messages": messages,
                "next_agent": "supervisor",
                "context_notes": [],
                "steps_remaining": 10,
                "final_answer": "",
                "plan": [],
                "scratchpad": "",
                "current_task": "",
                "worker_complete": {},
                "worker_outputs": {},
                "parallel_tasks": [],
                "critic_retry_count": 0,
                "waiting_for_approval": False,
                "approval_filepath": "",
                "approval_tool": "",
                "pending_file_approvals": {}
            }
            
            if hasattr(self.multi_agent_checkpointer, "aget_tuple"):
                result = await self.multi_agent_graph.ainvoke(initial_state, config=config)
            else:
                result = await asyncio.to_thread(self.multi_agent_graph.invoke, initial_state, config=config)
            final_answer = result.get("final_answer", "")
            if not final_answer and result.get("messages"):
                final_answer = result["messages"][-1].content
                
            total_ms = round((time_module.time() - t_start) * 1000, 2)
            self.telemetry.update_latency(total_ms)
            sys_metrics = self.telemetry.get_system_metrics()
            
            self.save_memory(session_id, query, "user")
            telemetry_data = {
                "query": query,
                "raw_prompt": f"Multi-Agent system query: {query}",
                "overflow_occurred": False,
                "limit": context_limit,
                "initial_tokens": 0,
                "final_tokens": 0,
                "steps": [],
                "budget_tracking": {
                    "memory_tokens_used": 0,
                    "memory_tokens_limit": 1500,
                    "document_tokens_used": 0,
                    "document_tokens_limit": 0
                },
                "compression_ratio": 1.0,
                "debug_info": {
                    "llm_calls": 0,
                    "goals_set": result.get("plan", []),
                    "actions_taken": []
                }
            }
            self.save_memory(session_id, final_answer, "assistant", 0.8, telemetry=telemetry_data)
            
            return {
                "query": query,
                "response": final_answer,
                "mode": mode,
                "search_queries": [query],
                "raw_prompt": telemetry_data["raw_prompt"],
                "retrieved_context": [],
                "compressed_context": "",
                "memory_context": "",
                "hyde_doc": "",
                "tps": 0.0,
                "query_cost": "$0.00000000",
                "stats": {
                    "compression_ratio": 1.0,
                    "avg_compression_ratio": round(self.stats["avg_compression_ratio"], 3),
                    "queries_handled": self.stats["queries"],
                    "active_memories": len(self.get_memory(session_id).entries),
                    "instantaneous_latency_ms": {"total_execution_ms": total_ms},
                    "avg_latency_ms": round(self.stats["avg_latency_ms"], 2),
                    "cpu_usage_percent": sys_metrics["cpu"],
                    "memory_usage_percent": sys_metrics["ram"],
                    "context_used_percent": 0.0,
                    "exact_tokens": {"prompt": 0, "completion": 0, "total": 0},
                    "budget_tracking": telemetry_data["budget_tracking"],
                    "overflow_telemetry": telemetry_data,
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
                # Issue #7: Run expansion and HyDE sequentially instead of concurrently
                # to prevent LLM API contention and vector store connection pool exhaustion.
                search_queries = await self._phase_expand_async(query, mode, latencies)
                
                # Only run HyDE if expansion didn't produce sufficient variations
                if len(search_queries) < 3:
                    hyde_doc = await self._phase_hyde_async(query, mode, latencies)
                    if hyde_doc:
                        search_queries.append(hyde_doc)
                else:
                    logger.info("[P1.5: HyDE] Skipped: expansion already produced sufficient variations.")
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

    async def _run_multi_agent_stream_async(self, query: str, session_id: str, source_filter: Optional[str] = None, context_limit: Optional[int] = None) -> AsyncGenerator[Dict, None]:
        # Clear checkpoints in checkpoints.db for this session to avoid duplication/stale state
        self._clear_session_checkpoints(session_id)

        from langchain_core.messages import HumanMessage, AIMessage
        from src.graph.workflow import get_graph_config
        import time as time_module
        import psutil
        
        logger.info(f"Multi-Agent starting (Async) for query: {query[:50]}")
        
        config = get_graph_config(session_id)
        
        # Fetch history from memory.db
        memory_history = self.persistent_memory.get_history(session_id)
        messages = []
        for entry in memory_history:
            role = entry.get("role")
            text = entry.get("text")
            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "assistant":
                messages.append(AIMessage(content=text))
        
        # Append the current query
        messages.append(HumanMessage(content=query))

        initial_state = {
            "messages": messages,
            "next_agent": "supervisor",
            "context_notes": [],
            "steps_remaining": 10,
            "final_answer": "",
            "plan": [],
            "scratchpad": "",
            "current_task": "",
            "worker_complete": {},
            "worker_outputs": {},
            "parallel_tasks": [],
            "critic_retry_count": 0,
            "waiting_for_approval": False,
            "approval_filepath": "",
            "approval_tool": "",
            "pending_file_approvals": {}
        }
        
        actions_taken = []
        goals_set = []
        llm_call_count = 0
        current_plan = []
        current_task = ""
        
        t_start = time_module.time()
        yield {"event": "state_change", "state": "WAITING_FOR_REASONING"}
        
        async def get_stream():
            if hasattr(self.multi_agent_checkpointer, "aget_tuple"):
                async for ev in self.multi_agent_graph.astream(initial_state, config=config):
                    yield ev
            else:
                for ev in self.multi_agent_graph.stream(initial_state, config=config):
                    yield ev

        try:
            async for event in get_stream():
                for node_name, state_delta in event.items():
                    yield {"event": "node_start", "node": node_name}
                    
                    # Check if this is coding_worker waiting for approval - emit blocked_tool then
                    # a waiting_for_approval event (NOT done) so the frontend keeps its interactive
                    # approval UI alive instead of collapsing the stream.
                    if node_name == "coding_worker_node" and state_delta.get("waiting_for_approval"):
                        approval_filepath = state_delta.get("approval_filepath", "")
                        approval_tool = state_delta.get("approval_tool", "")
                        yield {"event": "blocked_tool", "filepath": approval_filepath, "tool": approval_tool}
                        yield {"event": "waiting_for_approval", "filepath": approval_filepath, "tool": approval_tool}
                    
                    if node_name == "supervisor_node":
                        llm_call_count += 1
                        current_plan = state_delta.get("plan", current_plan)
                        next_agent = state_delta.get("next_agent", "")
                        current_task = state_delta.get("current_task", "")
                        parallel_tasks = state_delta.get("parallel_tasks", [])
                        
                        for step in current_plan:
                            if step not in goals_set:
                                goals_set.append(step)
                                
                        thought_msg = f"Supervisor: Evaluated findings. Current Plan: {current_plan}. Next Agent: {next_agent}."
                        if current_task:
                            thought_msg += f" Task: '{current_task}'."
                        elif parallel_tasks:
                            thought_msg += f" Parallel tasks: {parallel_tasks}."
                            
                        yield {"event": "thought", "text": thought_msg}
                        
                        if next_agent == "parallel":
                            action_input = ", ".join([f"{t.get('worker')}: '{t.get('task')}'" for t in parallel_tasks])
                            yield {"event": "action", "tool": "Parallel Dispatch", "input": action_input}
                            
                            actions_taken.append({
                                "step": len(actions_taken),
                                "thought": f"Supervisor evaluated progress. Plan: {current_plan}",
                                "tool": "Parallel Dispatch",
                                "input": action_input,
                                "observation": f"Dispatched tasks to parallel workers."
                            })
                        elif next_agent == "synthesizer":
                            yield {"event": "action", "tool": "Synthesizer Routing", "input": "Synthesizing final response"}
                            
                            actions_taken.append({
                                "step": len(actions_taken),
                                "thought": f"Supervisor evaluated progress. Plan: {current_plan}. All tasks completed.",
                                "tool": "Synthesizer Routing",
                                "input": "compile answer",
                                "observation": "Routing execution to Synthesizer."
                            })
                        else:
                            yield {"event": "action", "tool": f"Route to {next_agent}", "input": current_task}
                            
                            actions_taken.append({
                                "step": len(actions_taken),
                                "thought": f"Supervisor evaluated progress. Plan: {current_plan}",
                                "tool": f"Route to {next_agent}",
                                "input": current_task,
                                "observation": f"Routing execution to {next_agent}."
                            })
                            
                    elif node_name in ["rag_worker_node", "web_worker_node", "utility_worker_node", "scraper_worker_node", "critic_worker_node", "coding_worker_node", "code_critic_worker_node"]:
                        llm_call_count += 1
                        worker_type = state_delta.get("worker_type", "")
                        if not worker_type:
                            worker_type = node_name.replace("_node", "")
                        
                        # Note: blocked_tool is already emitted at node_start level for coding_worker
                        
                        response = ""
                        if "worker_outputs" in state_delta and worker_type:
                            response = state_delta["worker_outputs"].get(worker_type, "")
                        if not response and "messages" in state_delta and state_delta["messages"]:
                            response = state_delta["messages"][-1].content
                            
                        specialist_labels = {
                            "rag_worker": "RAG Specialist",
                            "web_worker": "Web Search Specialist",
                            "utility_worker": "Utility Specialist",
                            "scraper_worker": "Scraper Specialist",
                            "critic_worker": "Critic Specialist",
                            "coding_worker": "Coding Specialist",
                            "code_critic_worker": "Code Critic Specialist"
                        }
                        specialist = specialist_labels.get(worker_type, worker_type.replace("_", " ").title())
                        
                        thought_msg = f"{specialist}: Finished executing sub-task: '{current_task}'."
                        yield {"event": "thought", "text": thought_msg}
                        yield {"event": "observation", "output": response}
                        
                        actions_taken.append({
                            "step": len(actions_taken),
                            "thought": f"{specialist} processing sub-task: '{current_task}'",
                            "tool": worker_type,
                            "input": current_task,
                            "observation": response
                        })
                        
                    elif node_name == "synthesizer_node":
                        llm_call_count += 1
                        final_answer = state_delta.get("final_answer", "")
                        
                        yield {"event": "thought", "text": "Synthesizer: Compiling final response from cooperative findings."}
                        yield {"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"}
                        
                        chunk_size = 12
                        for i in range(0, len(final_answer), chunk_size):
                            yield {"event": "answer_chunk", "text": final_answer[i:i+chunk_size]}
                            await asyncio.sleep(0.01)
                            
                        actions_taken.append({
                            "step": len(actions_taken),
                            "thought": "Synthesizer compiled final response",
                            "tool": "Synthesizer",
                            "input": "final compilation",
                            "observation": "Response generated."
                        })
                        
                        self.save_memory(session_id, query, "user")
                        
                        telemetry_data = {
                            "query": query,
                            "raw_prompt": f"Multi-Agent system query: {query}",
                            "overflow_occurred": False,
                            "limit": context_limit,
                            "initial_tokens": 0,
                            "final_tokens": 0,
                            "steps": [],
                            "budget_tracking": {
                                "memory_tokens_used": 0,
                                "memory_tokens_limit": 1500,
                                "document_tokens_used": 0,
                                "document_tokens_limit": 0
                            },
                            "compression_ratio": 1.0,
                            "debug_info": {
                                "llm_calls": llm_call_count,
                                "goals_set": goals_set,
                                "actions_taken": actions_taken
                            }
                        }
                        self.save_memory(session_id, final_answer, "assistant", 0.8, telemetry=telemetry_data)
                        
                        cpu_val = psutil.cpu_percent(interval=None)
                        ram_val = psutil.virtual_memory().percent
                        total_ms = round((time_module.time() - t_start) * 1000, 2)
                        self.telemetry.update_latency(total_ms)
                        
                        yield {"event": "done", "response": final_answer, "stats": {
                            "queries_handled": self.stats["queries"],
                            "compression_ratio": 1.0,
                            "active_memories": len(self.get_memory(session_id).entries),
                            "instantaneous_latency_ms": {"total_execution_ms": total_ms},
                            "avg_latency_ms": round(self.stats["avg_latency_ms"], 2),
                            "cpu_usage_percent": cpu_val,
                            "memory_usage_percent": ram_val,
                            "context_used_percent": 0.0,
                            "exact_tokens": {"prompt": 0, "completion": 0, "total": 0},
                            "query_cost": "$0.00000000",
                            "budget_tracking": telemetry_data["budget_tracking"],
                            "overflow_telemetry": telemetry_data,
                            "retrieved_context": [],
                            "raw_prompt": telemetry_data["raw_prompt"],
                            "mode": "agentic",
                            "alpha": 0.5,
                            "reranker_peak_score": 0.0,
                            "debug_info": telemetry_data["debug_info"]
                        }}
                        
        except Exception as e:
            logger.error(f"Multi-Agent workflow stream error: {e}", exc_info=True)
            err_msg = f"Multi-Agent error: {str(e)}"
            yield {"event": "error", "message": err_msg}

    async def ask_stream_async(self, query: str, session_id: str = "default", mode: str = "context_engine",
                         source_filter: str = None, top_k: int = 5, context_limit: Optional[int] = None) -> AsyncGenerator[Dict, None]:
        """
        Asynchronous streaming query endpoint. Yields progress updates and LLM output tokens.
        """
        from typing import Dict as TypedDict
        import time as time_module
        
        if mode == "agentic":
            async for event in self._run_multi_agent_stream_async(query, session_id, source_filter, context_limit=context_limit):
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
        queue = asyncio.Queue()
        
        def on_retry(attempt, delay, exc):
            msg = f"[Rate Limit / API Error] Attempt {attempt} failed: {type(exc).__name__}. Retrying in {delay:.2f}s."
            queue.put_nowait({"event": "thought", "text": msg})
            
        self.llm_service.retry_callback = on_retry
        
        async def run_generation():
            try:
                async for chunk in self._phase_generate_stream_async(query, final_context, latencies):
                    await queue.put(chunk)
            except Exception as e:
                await queue.put({"event": "error", "message": str(e)})
            finally:
                await queue.put(None)
                
        gen_task = asyncio.create_task(run_generation())
        
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                if event.get("event") == "answer_chunk":
                    accumulated_response += event["text"]
                yield event
        finally:
            self.llm_service.retry_callback = None
            try:
                gen_task.cancel()
            except Exception:
                pass

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

