"""
================================================================================
RAG CONTEXT ENGINE - ORCHESTRATION MODULE
================================================================================
The Engine is the central brain of the system. It coordinates:
1. Query Expansion (via Expander)
2. Hypothetical Document Generation (via HyDE)
3. Hybrid Retrieval (via Retriever)
4. Neural Re-ranking (via Reranker)
5. Memory Management (via Memory + Persistence)
6. Context Compression (via Compressor)
7. Generation (via Groq LLM)
"""

import os
import logging
import time
import psutil
from typing import Dict
import tiktoken
from core.llm import LLMService

from core.config import (
    LLM_MODEL, LLM_TEMPERATURE, CONTEXT_WINDOW_LIMIT,
    TOKENIZER_ENCODING, TOTAL_CONTEXT_BUDGET, MEMORY_TOKEN_BUDGET,
    MIN_KNOWLEDGE_BUDGET, MAX_CANDIDATES, EXPANSION_MIN_WORDS,
    SAFETY_CHAR_LIMIT, COST_PER_INPUT_TOKEN, COST_PER_OUTPUT_TOKEN,
)
from core.retriever import WeaviateRetriever
from core.memory import ConversationMemory
from core.persistence import PersistentMemoryStore
from core.compressor import Compressor
from core.reranker import NeuralReranker
from core.expander import QueryExpander
from core.hyde import HyDEGenerator

logger = logging.getLogger("RAG.Engine")
tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    """Counts exact BPE tokens for a given string."""
    if not text:
        return 0
    return len(tokenizer.encode(text))


class RAGContextEngine:
    """
    Main orchestrator for the RAG pipeline.
    """

    def __init__(self, retriever: WeaviateRetriever):
        self.retriever = retriever
        self.memories: Dict[str, ConversationMemory] = {}
        self.persistent_memory = PersistentMemoryStore()
        self.compressor = Compressor()
        self.reranker = NeuralReranker()
        self.stats = {
            "queries": 0,
            "queries_compressed": 0,
            "avg_compression_ratio": 0,
            "avg_latency_ms": 0.0
        }

        # Initialize centralized LLM service
        self.llm_service = LLMService()
        self.client = self.llm_service.raw_client  # backward compat for generation
        self.expander = QueryExpander(self.client)
        self.hyde = HyDEGenerator(self.client)

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def get_memory(self, session_id: str) -> ConversationMemory:
        """Retrieves memory for a session. Restores from SQLite if not in RAM."""
        if session_id not in self.memories:
            logger.info(f"Restoring context for session: {session_id}")
            memory = ConversationMemory()
            for entry in self.persistent_memory.get_history(session_id):
                memory.add(entry["text"], importance=entry["importance"], role=entry["role"])
            self.memories[session_id] = memory
        return self.memories[session_id]

    def save_memory(self, session_id: str, text: str, role: str, importance: float = 1.0):
        """Saves a turn to both the active RAM context and persistent database."""
        self.get_memory(session_id).add(text, importance, role)
        self.persistent_memory.add_entry(session_id, text, role, importance)

    # ------------------------------------------------------------------
    # Pipeline phases (private helpers)
    # ------------------------------------------------------------------

    def _phase_expand(self, query: str, mode: str, latencies: dict) -> list:
        """Phase 1: Query Expansion."""
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
        """Phase 1.5: Hypothetical Document Generation."""
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
        """Phase 2: Hybrid Retrieval."""
        t = time.time()
        logger.info("[P2: RETRIEVAL] Executing hybrid search...")
        candidates_by_text = {}
        embed_total = 0.0
        db_total = 0.0

        for q in search_queries:
            retrieved = self.retriever.retrieve(q, top_k=top_k, source_filter=source_filter)
            for r in retrieved:
                text = r["text"]
                score = r.get("score", 0.0)
                if text not in candidates_by_text:
                    candidates_by_text[text] = r
                else:
                    # Keep the higher score if duplicate
                    if score > candidates_by_text[text].get("score", 0.0):
                        candidates_by_text[text]["score"] = score
            embed_total += getattr(self.retriever, "last_embed_latency_ms", 0.0)
            db_total += getattr(self.retriever, "last_search_latency_ms", 0.0)

        # Sort deduplicated candidates by score descending
        sorted_candidates = sorted(candidates_by_text.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        results = sorted_candidates[:MAX_CANDIDATES]

        logger.info(f" -> Fetched {len(results)} unique candidates (from {len(candidates_by_text)} total pool).")
        latencies['phase_2_retrieval_ms'] = round((time.time() - t) * 1000, 2)
        latencies['phase_2_embed_generation_ms'] = round(embed_total, 2)
        latencies['phase_2_weaviate_search_ms'] = round(db_total, 2)
        return results

    def _phase_refine(self, query: str, mode: str, memory, all_raw_results: list, top_k: int, latencies: dict):
        """Phases 3-5: Reranking, Memory, Compression (or bypass for simple mode)."""
        peak_score = 0.0

        if mode == "context_engine":
            # Phase 3: Reranking
            t = time.time()
            try:
                logger.info("[P3: RERANKING] Applying Cross-Encoder...")
                reranked = self.reranker.rerank(query, all_raw_results)[:top_k]
                peak_score = float(reranked[0]['cross_score']) if reranked else 0.0
                logger.info(f" -> Peak score: {peak_score:.4f}")
            except Exception as e:
                logger.warning(f" !! Reranking failed: {e}. Falling back.")
                reranked = sorted(all_raw_results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
            latencies['phase_3_reranking_ms'] = round((time.time() - t) * 1000, 2)

            # Phase 4: Memory
            t = time.time()
            logger.info("[P4: MEMORY] Synchronizing history...")
            memory_text = memory.get_active_context()
            memory_tokens = count_tokens(memory_text)
            latencies['phase_4_memory_sync_ms'] = round((time.time() - t) * 1000, 2)

            # Phase 5: Compression
            t = time.time()
            doc_budget = max(MIN_KNOWLEDGE_BUDGET, TOTAL_CONTEXT_BUDGET - memory_tokens)
            logger.info(f"[P5: COMPRESSION] Budget: {doc_budget} tokens...")
            compressed = self.compressor.compress(
                [r["text"] for r in reranked], query, max_tokens=doc_budget
            )
            
            # Format compressed segments into XML tags with document sources
            compressed_segments = compressed.split("\n\n")
            formatted_parts = []
            for seg in compressed_segments:
                seg_strip = seg.strip()
                if not seg_strip:
                    continue
                # Find the source of this segment
                source = "unknown"
                for r in reranked:
                    if seg_strip in r["text"]:
                        source = r.get("source", "unknown")
                        break
                formatted_parts.append(f'<document source="{source}">\n{seg_strip}\n</document>')
            
            final_knowledge = "\n\n".join(formatted_parts)
            doc_tokens = count_tokens(final_knowledge)
            final_context = f"### MEMORY\n{memory_text}\n\n### KNOWLEDGE\n{final_knowledge}"

            # Update compression stats
            raw_len = sum(len(r["text"]) for r in all_raw_results) + 1
            ratio = len(compressed) / raw_len
            self.stats["queries_compressed"] += 1
            self.stats["avg_compression_ratio"] = (
                self.stats["avg_compression_ratio"] * (self.stats["queries_compressed"] - 1) + ratio
            ) / self.stats["queries_compressed"]
            logger.info(f" -> Compression ratio: {ratio:.2%}")
            latencies['phase_5_compression_ms'] = round((time.time() - t) * 1000, 2)

            return final_context, memory_text, final_knowledge, ratio, memory_tokens, doc_tokens, doc_budget, peak_score

        else:
            # Simple RAG bypass
            logger.info("[P3-5: BYPASS] Simple RAG mode.")
            formatted_parts = []
            for r in all_raw_results[:top_k]:
                source = r.get("source", "unknown")
                formatted_parts.append(f'<document source="{source}">\n{r["text"]}\n</document>')
            
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

    def _phase_generate(self, query: str, final_context: str, latencies: dict):
        """Phase 6: LLM Generation."""
        t = time.time()
        logger.info("[P6: GENERATION] Sending to Groq...")
        prompt = (
            "Answer the user question using ONLY the provided context. "
            "If the information is missing, state that you don't know.\n\n"
            f"### CONTEXT:\n{final_context}\n\n"
            f"### QUESTION:\n{query}\n\n"
            "### ANSWER:"
        )
        prompt_tokens_est = count_tokens(prompt)
        ctx_used_pct = round((prompt_tokens_est / CONTEXT_WINDOW_LIMIT) * 100, 2)
        exact_tokens = {"prompt": 0, "completion": 0, "total": 0}

        try:
            completion = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE
            )
            response = completion.choices[0].message.content
            if hasattr(completion, 'usage') and completion.usage:
                exact_tokens = {
                    "prompt": completion.usage.prompt_tokens,
                    "completion": completion.usage.completion_tokens,
                    "total": completion.usage.total_tokens
                }
                ctx_used_pct = round((exact_tokens["prompt"] / CONTEXT_WINDOW_LIMIT) * 100, 2)
            logger.info(f" -> Tokens used: {exact_tokens['total']}")
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            response = f"I'm sorry, I encountered an error: {e}"

        latencies['phase_6_generation_ms'] = round((time.time() - t) * 1000, 2)
        return response, prompt, exact_tokens, ctx_used_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, query: str, session_id: str = "default", mode: str = "context_engine",
            source_filter: str = None, top_k: int = 5) -> Dict:
        """
        The primary entry point for querying the RAG system.
        """
        logger.info(f"\n[INIT] Query: '{query[:60]}...' | Session: {session_id} | Mode: {mode}")
        self.stats["queries"] += 1
        memory = self.get_memory(session_id)
        latencies = {}
        t_start = time.time()

        # Execute pipeline phases
        search_queries = self._phase_expand(query, mode, latencies)
        hyde_doc = self._phase_hyde(query, mode, search_queries, latencies)
        all_raw = self._phase_retrieve(search_queries, top_k, source_filter, latencies)
        (final_context, memory_text, compressed_docs, ratio,
         mem_tokens, doc_tokens, doc_budget, peak_score) = self._phase_refine(
            query, mode, memory, all_raw, top_k, latencies
        )
        response, prompt, exact_tokens, ctx_used_pct = self._phase_generate(query, final_context, latencies)

        # Persist interaction
        self.save_memory(session_id, query, "user")
        self.save_memory(session_id, response, "assistant", 0.8)

        # Compute telemetry
        total_ms = round((time.time() - t_start) * 1000, 2)
        latencies['total_execution_ms'] = total_ms
        gen_sec = latencies.get('phase_6_generation_ms', 0.0) / 1000.0
        tps = round(exact_tokens["completion"] / gen_sec, 1) if gen_sec > 0 else 0.0
        cost = (exact_tokens["prompt"] * COST_PER_INPUT_TOKEN) + (exact_tokens["completion"] * COST_PER_OUTPUT_TOKEN)

        self.stats["avg_latency_ms"] = (
            self.stats["avg_latency_ms"] * (self.stats["queries"] - 1) + total_ms
        ) / self.stats["queries"]
        logger.info(f"[FINISH] Complete in {total_ms}ms.\n")

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
                "cpu_usage_percent": psutil.cpu_percent(interval=None),
                "memory_usage_percent": psutil.virtual_memory().percent,
                "context_used_percent": ctx_used_pct,
                "exact_tokens": exact_tokens,
                "budget_tracking": {
                    "memory_tokens_used": mem_tokens,
                    "memory_tokens_limit": MEMORY_TOKEN_BUDGET,
                    "document_tokens_used": doc_tokens,
                    "document_tokens_limit": doc_budget
                },
                "mode": mode,
                "alpha": getattr(self.retriever, "alpha", 0.5),
                "reranker_peak_score": round(peak_score, 4)
            }
        }

    def close(self):
        """Cleanup resources."""
        self.retriever.close()
