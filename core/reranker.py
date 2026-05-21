"""
================================================================================
RAG CONTEXT ENGINE - RERANKER MODULE
================================================================================
Neural re-ranking via Cross-Encoder for deep semantic relevance scoring.
"""

import time
import logging
from typing import List, Dict
from sentence_transformers import CrossEncoder

from core.config import RERANKER_MODEL

logger = logging.getLogger("RAG.Reranker")


class NeuralReranker:
    """
    Uses a Cross-Encoder model to deeply evaluate the relevance
    of search candidates against a query.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        logger.info(f"Initializing Neural Reranker: {model_name}")
        self.model = CrossEncoder(model_name)

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
            # Apply Sigmoid: 1 / (1 + e^-x) to normalize to [0, 1] range
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
