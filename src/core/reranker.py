"""
RAG Reranker - Finding the Best Documents

When we search for documents, we get a list of candidates. This module re-scores
them more carefully to find the truly best matches. It uses a specialized AI model
called a Cross-Encoder that looks at both your question AND each document together
to give a more accurate relevance score.
"""

import time
import logging
import asyncio
from typing import List, Dict, Optional
from functools import lru_cache

from src.core.config import RERANKER_MODEL

logger = logging.getLogger("RAG.Reranker")

_model_instance: Optional["CrossEncoder"] = None

def _get_cross_encoder(model_name: str = RERANKER_MODEL):
    global _model_instance
    if _model_instance is None:
        logger.info(f"Lazy-loading CrossEncoder model: {model_name}")
        from sentence_transformers import CrossEncoder
        _model_instance = CrossEncoder(model_name)
    return _model_instance


class NeuralReranker:
    """
    Uses a specialized AI model to re-score search results for better accuracy.
    
    After we find documents that match your question, this class looks at each one
    more carefully to rank them by true relevance. It uses a Cross-Encoder model
    that understands both your question and each document together.
    
    The model is loaded only when first needed (lazy loading) to keep startup fast.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model_name = model_name
        self._model: Optional["CrossEncoder"] = None
        logger.info(f"NeuralReranker ready (model will load on first use)")

    @property
    def model(self):
        """Get the cross-encoder model, loading it if this is the first time."""
        if self._model is None:
            self._model = _get_cross_encoder(self._model_name)
        return self._model

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Re-score document candidates by how well they match your question.
        
        This gives us a more accurate ranking than the initial search, helping the AI
        focus on the most useful documents.
        """
        if not candidates:
            return []

        t_start = time.time()
        
        # Create pairs of (query, document) for the model to score
        import math
        pairs = [[query, cand["text"]] for cand in candidates]
        scores = self.model.predict(pairs)

        # Convert raw scores to normalized values (0.0 to 1.0)
        for i, score in enumerate(scores):
            raw_val = float(score)
            normalized = 1 / (1 + math.exp(-raw_val))
            candidates[i]["cross_score"] = normalized
            candidates[i]["raw_score"] = raw_val

        # Sort by the new scores
        sorted_candidates = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)

        t_ms = (time.time() - t_start) * 1000
        if sorted_candidates:
            lo = sorted_candidates[-1]["cross_score"]
            hi = sorted_candidates[0]["cross_score"]
            logger.info(f"Re-ranked {len(candidates)} documents in {t_ms:.1f}ms. Best score: {hi:.4f}, Worst: {lo:.4f}")

        return sorted_candidates

    async def rerank_async(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Asynchronously scores query-document pairs and sorts candidates.
        Runs the blocking model.predict in a thread pool.
        """
        if not candidates:
            return []

        t_start = time.time()

        def _do_rerank():
            import math
            pairs = [[query, cand["text"]] for cand in candidates]
            scores = self.model.predict(pairs)
            for i, score in enumerate(scores):
                raw_val = float(score)
                normalized = 1 / (1 + math.exp(-raw_val))
                candidates[i]["cross_score"] = normalized
                candidates[i]["raw_score"] = raw_val
            return sorted(candidates, key=lambda x: x["cross_score"], reverse=True)

        sorted_candidates = await asyncio.to_thread(_do_rerank)

        t_ms = (time.time() - t_start) * 1000
        if sorted_candidates:
            lo = sorted_candidates[-1]["cross_score"]
            hi = sorted_candidates[0]["cross_score"]
            logger.info(f"Scored {len(candidates)} pairs async in {t_ms:.1f}ms. Range: [{lo:.4f}, {hi:.4f}]")

        return sorted_candidates
