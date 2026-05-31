"""
================================================================================
RAG CONTEXT ENGINE - RERANKER MODULE
================================================================================
Neural re-ranking via Cross-Encoder for deep semantic relevance scoring.
Uses lazy singleton initialization to defer model loading until first use.
"""

import time
import logging
import asyncio
from typing import List, Dict, Optional
from functools import lru_cache

from core.config import RERANKER_MODEL

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
    Uses a Cross-Encoder model to deeply evaluate the relevance
    of search candidates against a query.
    Lazy-loads the model on first use to minimize startup cost.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model_name = model_name
        self._model: Optional["CrossEncoder"] = None
        logger.info(f"NeuralReranker initialized (lazy load pending)")

    @property
    def model(self):
        if self._model is None:
            self._model = _get_cross_encoder(self._model_name)
        return self._model

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Scores query-document pairs and sorts candidates by semantic relevance.
        """
        if not candidates:
            return []

        t_start = time.time()
        import math
        pairs = [[query, cand["text"]] for cand in candidates]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            raw_val = float(score)
            normalized = 1 / (1 + math.exp(-raw_val))
            candidates[i]["cross_score"] = normalized
            candidates[i]["raw_score"] = raw_val

        sorted_candidates = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)

        t_ms = (time.time() - t_start) * 1000
        if sorted_candidates:
            lo = sorted_candidates[-1]["cross_score"]
            hi = sorted_candidates[0]["cross_score"]
            logger.info(f"Scored {len(candidates)} pairs in {t_ms:.1f}ms. Range: [{lo:.4f}, {hi:.4f}]")

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
