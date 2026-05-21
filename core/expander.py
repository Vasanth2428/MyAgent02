"""
================================================================================
RAG CONTEXT ENGINE - QUERY EXPANDER MODULE
================================================================================
LLM-driven multi-query generation to improve retrieval recall.
"""

import json
import time
import logging
from typing import List

from core.config import LLM_MODEL

logger = logging.getLogger("RAG.Expander")


class QueryExpander:
    """
    Uses the LLM to generate diverse search variations of the user's query.
    """

    def __init__(self, groq_client):
        self.client = groq_client

    def expand(self, query: str) -> List[str]:
        """
        Generates 3 diverse search variations of the input query.
        Returns [original_query, variation_1, variation_2, variation_3].
        """
        t_start = time.time()
        prompt = (
            "Generate 3 diverse search variations of the following query, "
            "focused on different keywords. Return ONLY a JSON list of strings "
            "under the key 'variations'.\n"
            f"Query: {query}"
        )
        try:
            completion = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            variations = json.loads(completion.choices[0].message.content).get("variations", [])
            t_ms = (time.time() - t_start) * 1000
            logger.info(f"Generated {len(variations)} variations in {t_ms:.1f}ms")
            return [query] + variations
        except Exception as e:
            t_ms = (time.time() - t_start) * 1000
            logger.error(f"Expansion failed in {t_ms:.1f}ms: {e}")
            return [query]
