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
from typing import Dict, Generator, Optional
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
from core.registry import KnowledgeRegistry

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
        self.registry = KnowledgeRegistry(self)
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

        # Initialize ReAct Agent
        from core.agent import RAGAgent
        self.agent = RAGAgent(self)

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

    def save_memory(self, session_id: str, text: str, role: str, importance: float = 1.0, telemetry: dict = None):
        """Saves a turn to both the active RAM context and persistent database."""
        import json
        self.get_memory(session_id).add(text, importance, role)
        telemetry_json = json.dumps(telemetry) if telemetry else None
        self.persistent_memory.add_entry(session_id, text, role, importance, telemetry=telemetry_json)

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
        # Handle "show me the source(s)" pattern - check for source near "show me"
        words = q.split()
        if "show" in words and "me" in words:
            try:
                me_idx = words.index("me")
                # Check if source/sources is within 2 words after "me"
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

    def _handle_context_overflow(self, query: str, final_context: str, memory_text: str,
                                 compressed_docs: str, memory, all_raw: list,
                                 context_limit: int) -> tuple:
        """
        Detects, processes, and recovers from context overflows step-by-step.
        Returns (new_final_context, new_memory_text, new_compressed_docs, overflow_occurred, overflow_steps, initial_tokens, final_prompt_tokens, new_mem_tokens, new_doc_tokens)
        """
        import time
        from core.config import MEMORY_TOKEN_BUDGET
        
        # Build prompt templates
        instruction_prompt = (
            "Answer the user question using ONLY the provided context. "
            "If the information is missing, state that you don't know.\n\n"
        )
        instruction_tokens = count_tokens(instruction_prompt)
        query_prompt = f"\n\n### QUESTION:\n{query}\n\n### ANSWER:"
        query_tokens = count_tokens(query_prompt)
        
        mem_tokens = count_tokens(memory_text)
        doc_tokens = count_tokens(compressed_docs)
        
        # Total prompt token computation
        total_prompt_tokens = instruction_tokens + mem_tokens + doc_tokens + query_tokens + 15
        
        overflow_occurred = False
        overflow_steps = []
        initial_tokens = total_prompt_tokens
        
        if total_prompt_tokens > context_limit:
            overflow_occurred = True
            overflow_steps.append(
                f"🚨 OVERFLOW DETECTED: Prompt size ({total_prompt_tokens} tokens) "
                f"exceeds target limit ({context_limit} tokens) by {total_prompt_tokens - context_limit} tokens."
            )
            
            # Step 1: Memory Pruning
            old_mem_tokens = mem_tokens
            if total_prompt_tokens > context_limit and len(memory.entries) > 0:
                overflow_steps.append("🧹 Phase 1: Pruning conversation memory turns...")
                temp_entries = list(memory.entries)
                pruned_count = 0
                while len(temp_entries) > 1 and total_prompt_tokens > context_limit:
                    removed = temp_entries.pop(0)
                    pruned_count += 1
                    # Recompute memory text and tokens
                    temp_memory_text = "".join([f"[{e.role}]: {e.text}\n" for e in temp_entries])
                    temp_mem_tokens = count_tokens(temp_memory_text)
                    total_prompt_tokens = instruction_tokens + temp_mem_tokens + doc_tokens + query_tokens + 15
                
                # Apply changes
                if pruned_count > 0:
                    memory.entries = temp_entries
                    memory_text = memory.get_active_context()
                    mem_tokens = count_tokens(memory_text)
                    overflow_steps.append(f"   - Evicted {pruned_count} oldest conversational turns. Memory shrunk from {old_mem_tokens} to {mem_tokens} tokens.")
                else:
                    overflow_steps.append("   - No historical memory turns available for eviction.")
            
            # Step 2: Aggressive Knowledge Compression
            old_doc_tokens = doc_tokens
            if total_prompt_tokens > context_limit and doc_tokens > 10:
                overflow_steps.append("🗜️ Phase 2: Aggressive Knowledge Compression...")
                # Calculate new budget for knowledge to squeeze inside limit
                allowed_doc_budget = context_limit - instruction_tokens - mem_tokens - query_tokens - 25
                allowed_doc_budget = max(20, allowed_doc_budget)
                
                # Re-run compressor with smaller budget
                raw_texts = [r["text"] for r in all_raw]
                compressed_docs = self.compressor.compress(raw_texts, query, max_tokens=allowed_doc_budget)
                
                # Format with source XML tags
                compressed_segments = compressed_docs.split("\n\n")
                formatted_parts = []
                for seg in compressed_segments:
                    seg_strip = seg.strip()
                    if not seg_strip:
                        continue
                    source = "unknown"
                    for r in all_raw:
                        if seg_strip in r["text"]:
                            source = r.get("source", "unknown")
                            break
                    formatted_parts.append(f'<document source="{source}">\n{seg_strip}\n</document>')
                
                compressed_docs = "\n\n".join(formatted_parts)
                doc_tokens = count_tokens(compressed_docs)
                total_prompt_tokens = instruction_tokens + mem_tokens + doc_tokens + query_tokens + 15
                overflow_steps.append(f"   - Re-compressed knowledge source from {old_doc_tokens} to {doc_tokens} tokens (Target budget: {allowed_doc_budget}).")
            
            # Step 3: Hard Truncation / Eviction
            if total_prompt_tokens > context_limit:
                overflow_steps.append("✂️ Phase 3: Hard Truncation of prompt payload...")
                allowed_doc_budget = context_limit - instruction_tokens - mem_tokens - query_tokens - 25
                allowed_doc_budget = max(5, allowed_doc_budget)
                
                # Convert final_knowledge back to tokens, truncate, and decode
                doc_tkn_list = tokenizer.encode(compressed_docs)
                truncated_list = doc_tkn_list[:allowed_doc_budget]
                compressed_docs = tokenizer.decode(truncated_list)
                
                doc_tokens = count_tokens(compressed_docs)
                total_prompt_tokens = instruction_tokens + mem_tokens + doc_tokens + query_tokens + 15
                overflow_steps.append(f"   - Hard truncated remaining context from {old_doc_tokens} to {doc_tokens} tokens.")
            
            # Rebuild final context
            final_context = f"### MEMORY\n{memory_text}\n\n### KNOWLEDGE\n{compressed_docs}"
            overflow_steps.append(f"✅ RECOVERY COMPLETE: Prompt size is now {total_prompt_tokens} tokens (under {context_limit} limit).")
        
        return final_context, memory_text, compressed_docs, overflow_occurred, overflow_steps, initial_tokens, total_prompt_tokens, mem_tokens, doc_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, query: str, session_id: str = "default", mode: str = "context_engine",
            source_filter: str = None, top_k: int = 5, context_limit: Optional[int] = None) -> Dict:
        """
        The primary entry point for querying the RAG system.
        """
        logger.info(f"\n[INIT] Query: '{query[:60]}...' | Session: {session_id} | Mode: {mode} | Limit: {context_limit}")
        self.stats["queries"] += 1
        
        # Early exit for registry queries
        if self._is_registry_query(query):
            logger.info("Early exit: Registry query detected.")
            response = self._get_registry_context_text()
            self.save_memory(session_id, query, "user")
            self.save_memory(session_id, response, "assistant", 0.8)
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
                    "cpu_usage_percent": psutil.cpu_percent(interval=None),
                    "memory_usage_percent": psutil.virtual_memory().percent,
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

        # Execute pipeline phases
        search_queries = self._phase_expand(query, mode, latencies)
        hyde_doc = self._phase_hyde(query, mode, search_queries, latencies)
        all_raw = self._phase_retrieve(search_queries, top_k, source_filter, latencies)
        (final_context, memory_text, compressed_docs, ratio,
         mem_tokens, doc_tokens, doc_budget, peak_score) = self._phase_refine(
            query, mode, memory, all_raw, top_k, latencies
        )

        # Apply overflow handling
        overflow_occurred = False
        overflow_steps = []
        initial_tokens = mem_tokens + doc_tokens + 350
        final_prompt_tokens = initial_tokens
        if context_limit:
            (final_context, memory_text, compressed_docs, overflow_occurred, overflow_steps,
             initial_tokens, final_prompt_tokens, mem_tokens, doc_tokens) = self._handle_context_overflow(
                query, final_context, memory_text, compressed_docs, memory, all_raw, context_limit
            )

        response, prompt, exact_tokens, ctx_used_pct = self._phase_generate(query, final_context, latencies)

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
                "overflow_telemetry": {
                    "overflow_occurred": overflow_occurred,
                    "limit": context_limit,
                    "initial_tokens": initial_tokens,
                    "final_tokens": final_prompt_tokens,
                    "steps": overflow_steps
                },
                "mode": mode,
                "alpha": getattr(self.retriever, "alpha", 0.5),
                "reranker_peak_score": round(peak_score, 4)
            }
        }

    def _phase_generate_stream(self, query: str, final_context: str, latencies: dict) -> Generator[Dict, None, None]:
        """Phase 6: LLM Generation (Streaming)."""
        t = time.time()
        logger.info("[P6: GENERATION] Sending to Groq (Streaming)...")
        prompt = (
            "Answer the user question using ONLY the provided context. "
            "If the information is missing, state that you don't know.\n\n"
            f"### CONTEXT:\n{final_context}\n\n"
            f"### QUESTION:\n{query}\n\n"
            "### ANSWER:"
        )
        try:
            stream = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield {"event": "answer_chunk", "text": content}
        except Exception as e:
            logger.error(f"LLM Stream Error: {e}")
            yield {"event": "answer_chunk", "text": f"\n[LLM Error: {e}]"}
        latencies['phase_6_generation_ms'] = round((time.time() - t) * 1000, 2)

    def ask_stream(self, query: str, session_id: str = "default", mode: str = "context_engine",
                   source_filter: str = None, top_k: int = 5, context_limit: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        Streaming query endpoint. Yields progress updates and LLM output tokens.
        """
        from typing import Dict as TypedDict
        import time as time_module
        
        if mode == "agentic":
            yield from self.agent.run_stream(query, session_id, source_filter, context_limit=context_limit)
            return
        
        # Early exit for registry queries
        if self._is_registry_query(query):
            logger.info("Early exit: Registry query detected (stream).")
            yield {"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"}
            registry_text = self._get_registry_context_text()
            for i in range(0, len(registry_text), 20):
                yield {"event": "answer_chunk", "text": registry_text[i:i+20]}
                time_module.sleep(0.01)
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

        logger.info(f"\n[INIT STREAM] Query: '{query[:60]}...' | Session: {session_id} | Mode: {mode} | Limit: {context_limit}")
        self.stats["queries"] += 1
        memory = self.get_memory(session_id)
        latencies = {}
        t_start = time.time()

        # Step 1: Expand
        yield {"event": "thought", "text": "Analyzing query and generating semantic search variations..."}
        search_queries = self._phase_expand(query, mode, latencies)
        yield {"event": "action", "tool": "Query Expansion", "input": f"Variations: {search_queries}"}

        # Step 1.5: HyDE
        hyde_doc = ""
        if mode == "context_engine":
            yield {"event": "thought", "text": "Generating hypothetical document response (HyDE) to capture latent semantics..."}
            hyde_doc = self._phase_hyde(query, mode, search_queries, latencies)
            yield {"event": "observation", "output": f"HyDE generated (~{len(hyde_doc)} chars)"}

        # Step 2: Retrieve
        yield {"event": "thought", "text": "Retrieving contextually relevant documents from Vector Database..."}
        all_raw = self._phase_retrieve(search_queries, top_k, source_filter, latencies)
        yield {"event": "observation", "output": f"Retrieved {len(all_raw)} document chunks."}

        # Step 3-5: Refine (Reranking, Memory Sync, Compression)
        yield {"event": "thought", "text": "Reranking document candidates and performing dynamic memory decay budget sizing..."}
        (final_context, memory_text, compressed_docs, ratio,
         mem_tokens, doc_tokens, doc_budget, peak_score) = self._phase_refine(
            query, mode, memory, all_raw, top_k, latencies
         )

        # Apply overflow handling
        overflow_occurred = False
        overflow_steps = []
        initial_tokens = mem_tokens + doc_tokens + 350
        final_prompt_tokens = initial_tokens
        
        if context_limit:
            (final_context, memory_text, compressed_docs, overflow_occurred, overflow_steps,
             initial_tokens, final_prompt_tokens, mem_tokens, doc_tokens) = self._handle_context_overflow(
                query, final_context, memory_text, compressed_docs, memory, all_raw, context_limit
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
                time.sleep(0.4)

        yield {"event": "observation", "output": f"Reranking score: {peak_score:.4f}. Context compressed down by {round((1-ratio)*100)}%."}

        # Step 6: Generation (yielding chunks)
        yield {"event": "thought", "text": "Generating final grounded answer from compressed context..."}
        
        # Save placeholder history (will be updated when final response is completed)
        accumulated_response = ""
        for chunk in self._phase_generate_stream(query, final_context, latencies):
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
        self.stats["avg_latency_ms"] = (
            self.stats["avg_latency_ms"] * (self.stats["queries"] - 1) + total_ms
        ) / self.stats["queries"]

        yield {"event": "done", "response": accumulated_response, "stats": {
            "compression_ratio": round(ratio, 3),
            "avg_compression_ratio": round(self.stats["avg_compression_ratio"], 3),
            "queries_handled": self.stats["queries"],
            "active_memories": len(memory.entries),
            "instantaneous_latency_ms": latencies,
            "avg_latency_ms": round(self.stats["avg_latency_ms"], 2),
            "cpu_usage_percent": psutil.cpu_percent(interval=None),
            "memory_usage_percent": psutil.virtual_memory().percent,
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
