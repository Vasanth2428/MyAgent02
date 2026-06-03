"""
Retrieval Service - Finding Relevant Documents

This service handles searching through your document database. When you ask a question,
it runs multiple search queries (the original question plus variations) and combines
the results to find the most relevant document chunks.
"""

import time
import logging
import asyncio
from typing import List, Dict, Tuple, Optional
from src.core.config import MAX_CANDIDATES
from src.core.retriever import WeaviateRetriever

logger = logging.getLogger("RAG.Services.Retrieval")


class RetrievalService:
    """
    Searches the document database for relevant content.
    
    When you ask a question, this service:
    1. Runs the original question through the search
    2. Runs any query variations through search too
    3. Combines and dedupes results
    4. Returns the top candidates for the AI to use
    
    This helps find the most relevant pieces of your documents to answer questions.
    """

    def __init__(self, retriever: WeaviateRetriever):
        self.retriever = retriever  # Our connection to the document database

    def retrieve(self, search_queries: List[str], top_k: int, source_filter: Optional[str] = None) -> Tuple[List[Dict], float, float, float]:
        """
        Search for documents using multiple query variations.
        
        Args:
            search_queries: List of search queries (original + variations)
            top_k: Maximum number of documents to return per query
            source_filter: Optional filter to only search specific documents
            
        Returns:
            - List of unique document results, sorted by relevance
            - Embedding generation time in ms
            - Database search time in ms
            - Total time in ms
        """
        t = time.time()
        logger.info("[P2: RETRIEVAL] Searching for relevant documents...")
        candidates_by_text = {}  # Track unique documents by their text content
        embed_total = 0.0
        db_total = 0.0

        for q in search_queries:
            retrieved, embed_lat, db_lat = self.retriever.retrieve(q, top_k=top_k, source_filter=source_filter)
            for r in retrieved:
                text = r["text"]
                score = r.get("score", 0.0)
                # Keep the highest-scoring version if we see the same text twice
                if text not in candidates_by_text:
                    candidates_by_text[text] = r
                else:
                    if score > candidates_by_text[text].get("score", 0.0):
                        candidates_by_text[text]["score"] = score
                embed_total += embed_lat
                db_total += db_lat

        # Sort by score and limit to MAX_CANDIDATES
        sorted_candidates = sorted(candidates_by_text.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        results = sorted_candidates[:MAX_CANDIDATES]

        logger.info(f" -> Found {len(results)} unique relevant document chunks.")
        total_ms = round((time.time() - t) * 1000, 2)
        
        return results, round(embed_total, 2), round(db_total, 2), total_ms

    async def retrieve_async(self, search_queries: List[str], top_k: int, source_filter: Optional[str] = None) -> Tuple[List[Dict], float, float, float]:
        """
        Search for documents concurrently using multiple query variations.
        
        This async version runs searches in parallel instead of one after another,
        making it faster when you have multiple search queries.
        
        Args:
            search_queries: List of search queries (original + variations)
            top_k: Maximum number of documents to return per query
            source_filter: Optional filter to only search specific documents
            
        Returns:
            - List of unique document results, sorted by relevance
            - Embedding generation time in ms
            - Database search time in ms
            - Total time in ms
        """
        t = time.time()
        logger.info("[P2: RETRIEVAL] Searching concurrently for relevant documents...")
        
        async def single_retrieve(q):
            # Run each search in a thread to avoid blocking
            retrieved, embed_lat, db_lat = await asyncio.to_thread(self.retriever.retrieve, q, top_k=top_k, source_filter=source_filter)
            return retrieved, embed_lat, db_lat

        # Run all searches at once
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

        logger.info(f" -> Found {len(results)} unique relevant document chunks (searched in parallel).")
        total_ms = round((time.time() - t) * 1000, 2)
        
        return results, round(embed_total, 2), round(db_total, 2), total_ms
