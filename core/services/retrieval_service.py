import time
import logging
import asyncio
from typing import List, Dict, Tuple, Optional
from core.config import MAX_CANDIDATES
from core.retriever import WeaviateRetriever

logger = logging.getLogger("RAG.Services.Retrieval")

class RetrievalService:
    """
    Handles retrieving document candidates from Weaviate database.
    """
    def __init__(self, retriever: WeaviateRetriever):
        self.retriever = retriever

    def retrieve(self, search_queries: List[str], top_k: int, source_filter: Optional[str] = None) -> Tuple[List[Dict], float, float, float]:
        """
        Executes hybrid search across all search queries and compiles unique candidates.
        Returns:
            Tuple containing:
            - results (List[Dict]): Unique sorted document candidates.
            - embed_latency_ms (float): Total embedding generation latency.
            - db_search_latency_ms (float): Total database search latency.
            - total_phase_latency_ms (float): Total phase execution latency.
        """
        t = time.time()
        logger.info("[P2: RETRIEVAL] Executing hybrid search via RetrievalService...")
        candidates_by_text = {}
        embed_total = 0.0
        db_total = 0.0

        for q in search_queries:
            retrieved, embed_lat, db_lat = self.retriever.retrieve(q, top_k=top_k, source_filter=source_filter)
            for r in retrieved:
                text = r["text"]
                score = r.get("score", 0.0)
                if text not in candidates_by_text:
                    candidates_by_text[text] = r
                else:
                    if score > candidates_by_text[text].get("score", 0.0):
                        candidates_by_text[text]["score"] = score
            embed_total += embed_lat
            db_total += db_lat

        sorted_candidates = sorted(candidates_by_text.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        results = sorted_candidates[:MAX_CANDIDATES]

        logger.info(f" -> Fetched {len(results)} unique candidates (from {len(candidates_by_text)} total pool).")
        total_ms = round((time.time() - t) * 1000, 2)
        
        return results, round(embed_total, 2), round(db_total, 2), total_ms

    async def retrieve_async(self, search_queries: List[str], top_k: int, source_filter: Optional[str] = None) -> Tuple[List[Dict], float, float, float]:
        """
        Executes hybrid search across all search queries concurrently in threads and compiles unique candidates.
        Returns:
            Tuple containing:
            - results (List[Dict]): Unique sorted document candidates.
            - embed_latency_ms (float): Total embedding generation latency.
            - db_search_latency_ms (float): Total database search latency.
            - total_phase_latency_ms (float): Total phase execution latency.
        """
        t = time.time()
        logger.info("[P2: RETRIEVAL] Executing concurrent hybrid search via RetrievalService...")
        
        async def single_retrieve(q):
            retrieved, embed_lat, db_lat = await asyncio.to_thread(self.retriever.retrieve, q, top_k=top_k, source_filter=source_filter)
            return retrieved, embed_lat, db_lat

        tasks = [single_retrieve(q) for q in search_queries]
        query_results = await asyncio.gather(*tasks)

        candidates_by_text = {}
        embed_total = 0.0
        db_total = 0.0

        for retrieved, embed_lat, db_lat in query_results:
            for r in retrieved:
                text = r["text"]
                score = r.get("score", 0.0)
                if text not in candidates_by_text:
                    candidates_by_text[text] = r
                else:
                    if score > candidates_by_text[text].get("score", 0.0):
                        candidates_by_text[text]["score"] = score
            embed_total += embed_lat
            db_total += db_lat

        sorted_candidates = sorted(candidates_by_text.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        results = sorted_candidates[:MAX_CANDIDATES]

        logger.info(f" -> Fetched {len(results)} unique candidates (from {len(candidates_by_text)} total pool) in parallel.")
        total_ms = round((time.time() - t) * 1000, 2)
        
        return results, round(embed_total, 2), round(db_total, 2), total_ms
