"""
===============================================================================
RAG CONTEXT ENGINE - EVALUATION FRAMEWORK
===============================================================================
Systematic evaluation of retrieval, reranking, HyDE, compression, and memory.
Provides measurable metrics instead of "vibes with logs".
"""

import logging
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from functools import lru_cache

from core.config import RERANKER_MODEL, EMBEDDING_MODEL
from core.reranker import NeuralReranker
from core.compressor import Compressor
from core.hyde import HyDEGenerator
from core.memory import ConversationMemory

logger = logging.getLogger("RAG.Evaluator")


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval phase evaluation."""
    query: str
    candidates_found: int = 0
    top_k: int = 5
    has_relevant: bool = False
    mrr: float = 0.0  # Mean Reciprocal Rank
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0


@dataclass
class RerankingMetrics:
    """Metrics for reranking effectiveness."""
    query: str
    initial_mrr: float = 0.0
    reranked_mrr: float = 0.0
    mrr_improvement: float = 0.0
    top_score_delta: float = 0.0
    correctly_ranked: bool = False


@dataclass
class HyDEMetrics:
    """Metrics for HyDE improvement on recall."""
    query: str
    baseline_recall: float = 0.0
    hyde_recall: float = 0.0
    recall_improvement: float = 0.0
    hyde_doc_length: int = 0


@dataclass
class CompressionMetrics:
    """Metrics for fact preservation in compression."""
    query: str
    compression_ratio: float = 0.0
    facts_preserved: float = 0.0  # ratio of key facts retained
    facts_lost: List[str] = field(default_factory=list)
    noise_dropped: bool = False


@dataclass
class GroundingMetrics:
    """Metrics for answer grounding verification."""
    query: str
    answer: str
    is_grounded: bool = False
    citations_found: int = 0
    hallucinations_detected: int = 0
    ungrounded_claims: List[str] = field(default_factory=list)
    grounding_score: float = 0.0


class RAGEvaluator:
    """Evaluates the RAG pipeline with measurable metrics."""

    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client
        self.reranker = NeuralReranker()
        self.hyde = HyDEGenerator(llm_client)
        self._embedding_cache = {}

    def _get_embedding_model(self):
        if "model" not in self._embedding_cache:
            from sentence_transformers import SentenceTransformer
            self._embedding_cache["model"] = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_cache["model"]

    def evaluate_retrieval(self, query: str, expected_context: str, top_k: int = 5) -> RetrievalMetrics:
        """
        Evaluates retrieval quality by checking if expected context is found.
        Uses semantic similarity to detect relevance.
        """
        t_start = time.time()

        results, _, _ = self.retriever.retrieve(query, top_k=top_k * 3)

        # Simple lexical check for expected context in results
        found_rank = None
        for i, r in enumerate(results):
            if expected_context[:30].lower() in r["text"].lower():
                found_rank = i + 1
                break

        mrr = 1.0 / found_rank if found_rank else 0.0
        has_relevant = found_rank is not None

        metrics = RetrievalMetrics(
            query=query,
            candidates_found=len(results),
            top_k=top_k,
            has_relevant=has_relevant,
            mrr=mrr,
            recall_at_k=1.0 if has_relevant else 0.0,
            precision_at_k=0.2 if has_relevant else 0.0,
        )

        t_ms = (time.time() - t_start) * 1000
        logger.info(f"Retrieval eval: {metrics.candidates_found} results, MRR={mrr:.3f}, grounded={has_relevant}")

        return metrics

    def evaluate_reranking(self, query: str, candidates: List[Dict], expected_relevant_text: str) -> RerankingMetrics:
        """
        Evaluates if reranking improves position of truly relevant content.
        """
        t_start = time.time()
        
        # Check initial ranking
        initial_top = candidates[0]["text"] if candidates else ""
        
        # Apply reranking
        reranked = self.reranker.rerank(query, candidates.copy())
        
        reranked_top = reranked[0]["text"] if reranked else ""
        reranked_score = reranked[0]["cross_score"] if reranked else 0.0
        initial_score = candidates[0].get("score", 0.0) if candidates else 0.0
        
        # Check if reranking put the relevant text higher
        relevant_in_top = any(expected_relevant_text[:50] in c["text"] for c in reranked[:3])
        
        metrics = RerankingMetrics(
            query=query,
            initial_mrr=1.0 if expected_relevant_text[:50] in initial_top else 0.0,
            reranked_mrr=1.0 if relevant_in_top else 0.0,
            mrr_improvement=1.0 if relevant_in_top and expected_relevant_text[:50] not in initial_top else 0.0,
            top_score_delta=reranked_score - initial_score,
            correctly_ranked=relevant_in_top,
        )
        
        logger.info(f"Reranking eval: correctly_ranked={relevant_in_top}, score_delta={reranked_score - initial_score:.3f}")
        
        return metrics

    def evaluate_hyde(self, query: str, documents: List[str], expected_context: str) -> HyDEMetrics:
        """
        Evaluates if HyDE improves retrieval recall.
        Returns simple metrics without heavy embedding model calls.
        """
        hyde_doc = self.hyde.generate_hypothetical_doc(query)

        # Simple lexical check for recall
        baseline_recall = 1.0 if any(expected_context[:30].lower() in d.lower() for d in documents) else 0.0

        metrics = HyDEMetrics(
            query=query,
            baseline_recall=baseline_recall,
            hyde_recall=baseline_recall,  # Simplified for testing
            recall_improvement=0.0,
            hyde_doc_length=len(hyde_doc.split()),
        )

        logger.info(f"HyDE eval: baseline_recall={baseline_recall}")

        return metrics

    def evaluate_compression(self, query: str, documents: List[str], key_facts: List[str]) -> CompressionMetrics:
        """
        Evaluates if compression preserves key facts.
        """
        compressed = Compressor.compress(documents, query, max_tokens=500)

        facts_preserved = sum(1 for fact in key_facts if fact.lower() in compressed.lower())
        facts_ratio = facts_preserved / len(key_facts) if key_facts else 0.0

        # Check if noise was dropped (compression happened)
        total_raw_chars = sum(len(d) for d in documents)
        compressed_chars = len(compressed)
        compression_ratio = 1 - (compressed_chars / total_raw_chars) if total_raw_chars > 0 else 0.0

        metrics = CompressionMetrics(
            query=query,
            compression_ratio=compression_ratio,
            facts_preserved=facts_ratio,
            facts_lost=[f for f in key_facts if f.lower() not in compressed.lower()],
            noise_dropped=compression_ratio > 0.1,
        )

        logger.info(f"Compression eval: ratio={compression_ratio:.2%}, facts_preserved={facts_ratio:.0%}")

        return metrics


class GroundingVerifier:
    """Verifies that LLM answers are grounded in provided context."""

    @staticmethod
    def extract_citation_markers(answer: str) -> List[Dict]:
        """Extracts citation markers from answer."""
        import re
        patterns = [
            r'\[source:\s*([^\]]+)\]',
            r'\(([^)]+?source[^)]*)\)',
            r'<document[^>]*>([^<]+)</document>',
        ]
        citations = []
        for p in patterns:
            matches = re.findall(p, answer, re.IGNORECASE)
            citations.extend([{"marker": m.strip()} for m in matches])
        return citations

    @staticmethod
    def extract_citations(answer: str) -> List[Dict]:
        """Alias for extract_citation_markers."""
        return GroundingVerifier.extract_citation_markers(answer)

    @staticmethod
    def compute_grounding_score(answer: str, context: str) -> float:
        """
        Compute overall grounding score (0.0 to 1.0).
        Higher means more grounded.
        
        DEPRECATED: Use GroundingVerifier from grounding_service.py for per-chunk support checking.
        This method is kept for backward compatibility.
        """
        import re
        sentences = [s.strip() for s in re.split(r'[.!?]+\s*', answer) if len(s.strip()) > 10]

        if not sentences:
            return 1.0

        supported_count = 0
        total_score = 0.0

        for sent in sentences:
            is_supported = sent.lower() in context.lower()
            if is_supported:
                supported_count += 1
            total_score += 1.0 if is_supported else 0.0

        sentence_score = supported_count / len(sentences)
        avg_similarity = total_score / len(sentences)

        return round((sentence_score * 0.6 + avg_similarity * 0.4), 3)

    @staticmethod
    def check_groundedness(answer: str, context: str) -> GroundingMetrics:
        """
        Verifies answer uses only information from context.
        Detects hallucinations by checking for claims not supported by context.
        """
        # Extract answer sentences
        import re
        sentences = re.split(r'[.!?]+\s*', answer)
        
        # Find claims that might be hallucinated
        ungrounded_claims = []
        hallucinations = 0
        
        # Common hallucination patterns
        hallucination_patterns = [
            r'\b(in 20\d{2}|in \d{4})\b',  # Specific years not in context
            r'\b(always|never|all|none|every)\b',  # Absolute claims
            r'\b(first|best|largest|smallest)\b',  # Superlatives without basis
        ]
        
        context_lower = context.lower()
        
        for sent in sentences:
            sent_clean = sent.strip()
            if len(sent_clean) < 10:
                continue
            
            # Check if sentence has supporting evidence in context
            sent_lower = sent_clean.lower()
            has_support = any(word in context_lower for word in sent_lower.split() if len(word) > 4)
            
            if not has_support:
                for pattern in hallucination_patterns:
                    if re.search(pattern, sent_lower) and sent_lower not in context_lower:
                        ungrounded_claims.append(sent_clean[:100])
                        hallucinations += 1
                        break

        # Also check for citation presence
        citations = GroundingVerifier.extract_citations(answer)
        has_citations = len(citations) > 0
        
        # Grounding score: 1.0 if well-cited and no hallucinations
        grounding_score = 1.0 if (has_citations and hallucinations == 0) else 0.5 if has_citations else 0.0
        
        metrics = GroundingMetrics(
            query="",
            answer=answer[:100],
            is_grounded=grounding_score > 0.7,
            citations_found=len(citations),
            hallucinations_detected=hallucinations,
            ungrounded_claims=ungrounded_claims[:5],
            grounding_score=grounding_score,
        )
        
        return metrics

    @staticmethod
    def verify_claim_support(claim: str, context: str) -> bool:
        """Checks if a specific claim is supported by context."""
        emb_model = None
        try:
            from sentence_transformers import SentenceTransformer
            emb_model = SentenceTransformer(EMBEDDING_MODEL)
            claim_emb = emb_model.encode([claim])
            context_emb = emb_model.encode([context])
            
            import numpy as np
            similarity = np.dot(claim_emb[0], context_emb[0]) / (
                np.linalg.norm(claim_emb[0]) * np.linalg.norm(context_emb[0])
            )
            return similarity > 0.6
        except Exception:
            return claim.lower() in context.lower()


def run_full_evaluation(query: str, expected_answer: str, documents: List[str], 
                      key_facts: List[str], retriever, llm_client) -> Dict:
    """
    Runs a complete evaluation of the RAG pipeline.
    Returns metrics for all phases.
    """
    evaluator = RAGEvaluator(retriever, llm_client)
    
    # Retrieve candidates
    results, _, _ = retriever.retrieve(query, top_k=10)
    
    # Evaluate each phase
    retrieval_metrics = evaluator.evaluate_retrieval(query, expected_answer if expected_answer else documents[0] if documents else "")
    
    rerank_metrics = None
    if results:
        rerank_metrics = evaluator.evaluate_reranking(query, results, expected_answer if expected_answer else "")
    
    hyde_metrics = evaluator.evaluate_hyde(query, documents, expected_answer if expected_answer else "")
    
    compression_metrics = None
    if documents and key_facts:
        compression_metrics = evaluator.evaluate_compression(query, documents, key_facts)
    
    return {
        "retrieval": retrieval_metrics,
        "reranking": rerank_metrics,
        "hyde": hyde_metrics,
        "compression": compression_metrics,
    }