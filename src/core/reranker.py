"""
RAG Reranker - Finding the Best Documents

When we search for documents, we get a list of candidates. This module re-scores
them more carefully to find the truly best matches using FlashrankRerank — a
LangChain-native document compressor backed by the Flashrank cross-encoder library.

Flashrank is CPU-only, fast, and encapsulates model loading, tokenisation, and
sigmoid scoring internally.
"""

import time
import logging
import asyncio
from typing import List, Dict, Optional

from src.core.config import RERANKER_MODEL

logger = logging.getLogger("RAG.Reranker")

import threading

_reranker_lock = threading.Lock()
_reranker_instance = None


def _get_flashrank_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                model_name = RERANKER_MODEL.split("/")[-1] if "/" in RERANKER_MODEL else RERANKER_MODEL
                # Map config cross-encoder model to supported Flashrank models
                if "minilm" in model_name.lower():
                    model_name = "ms-marco-MiniLM-L-12-v2"
                elif "tinybert" in model_name.lower():
                    model_name = "ms-marco-TinyBERT-L-6-v2"
                else:
                    model_name = "ms-marco-MultiBERT-L-12"
                    
                logger.info(f"Lazy-loading FlashrankRerank (model: {model_name})")
                from langchain_community.document_compressors import FlashrankRerank
                _reranker_instance = FlashrankRerank(model=model_name, top_n=100)
    return _reranker_instance


def _candidates_to_docs(candidates: List[Dict]):
    """Convert raw candidate dicts to LangChain Document objects for reranking."""
    from langchain_core.documents import Document
    return [Document(page_content=c["text"], metadata={k: v for k, v in c.items() if k != "text"}) for c in candidates]


def _docs_to_candidates(docs, original_candidates: List[Dict]) -> List[Dict]:
    """Map reranked Document objects back to the original candidate dict shape."""
    # Build a lookup by text
    original_by_text = {c["text"]: c for c in original_candidates}
    result = []
    for i, doc in enumerate(docs):
        text = doc.page_content
        candidate = dict(original_by_text.get(text, {"text": text}))
        # FlashrankRerank stores relevance_score in metadata
        score = doc.metadata.get("relevance_score", 1.0 - (i * 0.01))
        candidate["cross_score"] = float(score)
        candidate["raw_score"] = float(score)
        result.append(candidate)
    return result


class NeuralReranker:
    """
    Uses FlashrankRerank to re-score search results for better accuracy.

    After we find documents that match your question, this class looks at each one
    more carefully to rank them by true relevance. Flashrank handles cross-encoder
    model loading, tokenisation, and scoring internally.

    The model is loaded only when first needed (lazy loading) to keep startup fast.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model_name = model_name
        self._model = None
        logger.info("NeuralReranker ready (model will load on first use via FlashrankRerank)")

    @property
    def model(self):
        """Get the model, supporting backward compatibility and test mock setting."""
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

        # Check if a custom mock model is injected
        model = getattr(self, "_model", None)
        if model is not None:
            try:
                import math
                pairs = [[query, cand["text"]] for cand in candidates]
                scores = model.predict(pairs)
                for i, score in enumerate(scores):
                    raw_val = float(score)
                    normalized = 1 / (1 + math.exp(-raw_val))
                    candidates[i]["cross_score"] = normalized
                    candidates[i]["raw_score"] = raw_val
                return sorted(candidates, key=lambda x: x["cross_score"], reverse=True)
            except Exception as e:
                logger.warning(f"Injected mock model rerank failed: {e}")
                return candidates

        try:
            reranker = _get_flashrank_reranker()
            docs = _candidates_to_docs(candidates)
            reranked_docs = reranker.compress_documents(docs, query)
            result = _docs_to_candidates(reranked_docs, candidates)
        except Exception as e:
            logger.warning(f"FlashrankRerank failed, returning original order: {e}")
            result = candidates

        t_ms = (time.time() - t_start) * 1000
        if result:
            lo = result[-1].get("cross_score", 0)
            hi = result[0].get("cross_score", 0)
            logger.info(
                f"Re-ranked {len(candidates)} documents in {t_ms:.1f}ms. "
                f"Best score: {hi:.4f}, Worst: {lo:.4f}"
            )
        return result

    async def rerank_async(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Asynchronously re-scores query-document pairs.
        Runs the blocking reranker in a thread pool via asyncio.to_thread.
        """
        if not candidates:
            return []

        t_start = time.time()

        # Check if a custom mock model is injected
        model = getattr(self, "_model", None)
        if model is not None:
            def _do_cc_rerank():
                import math
                pairs = [[query, cand["text"]] for cand in candidates]
                scores = model.predict(pairs)
                for i, score in enumerate(scores):
                    raw_val = float(score)
                    normalized = 1 / (1 + math.exp(-raw_val))
                    candidates[i]["cross_score"] = normalized
                    candidates[i]["raw_score"] = raw_val
                return sorted(candidates, key=lambda x: x["cross_score"], reverse=True)

            try:
                return await asyncio.to_thread(_do_cc_rerank)
            except RuntimeError:
                # Fallback for thread restricted sandbox testing environments
                return _do_cc_rerank()

        def _do_rerank():
            reranker = _get_flashrank_reranker()
            docs = _candidates_to_docs(candidates)
            reranked_docs = reranker.compress_documents(docs, query)
            return _docs_to_candidates(reranked_docs, candidates)

        try:
            result = await asyncio.to_thread(_do_rerank)
        except RuntimeError:
            # Fallback for thread restricted sandbox testing environments
            try:
                result = _do_rerank()
            except Exception as e:
                logger.warning(f"Fallback async FlashrankRerank failed: {e}")
                result = candidates
        except Exception as e:
            logger.warning(f"Async FlashrankRerank failed, returning original order: {e}")
            result = candidates

        t_ms = (time.time() - t_start) * 1000
        if result:
            lo = result[-1].get("cross_score", 0)
            hi = result[0].get("cross_score", 0)
            logger.info(
                f"Scored {len(candidates)} pairs async in {t_ms:.1f}ms. "
                f"Range: [{lo:.4f}, {hi:.4f}]"
            )
        return result
