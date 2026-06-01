import json
import time
import logging
from typing import List

from core.config import LLM_MODEL

logger = logging.getLogger("RAG.Expander")


class QueryExpander:
    """
    Creates different ways to search for the same question.
    
    Sometimes a question can be phrased multiple ways. This class asks the AI to
    generate alternative versions of your query, which helps find more documents
    that might be relevant but use different wording.
    """

    def __init__(self, groq_client, async_client=None):
        self.client = groq_client
        self.async_client = async_client
        if not self.async_client and hasattr(groq_client, "llm_service"):
            self.async_client = groq_client.llm_service.async_client

    def expand(self, query: str) -> List[str]:
        """
        Generate alternative ways to search for the same question.
        
        For longer questions, this creates 3 variations focused on different keywords.
        For example, "How does the sales forecasting model work?" might become:
        - "sales model algorithm"
        - "forecasting predictions database"
        - "sales analysis workflow"
        
        This helps find more relevant documents that might use different terminology.
        """
        t_start = time.time()
        prompt = (
            "Generate 3 different ways to search for information about this question. "
            "Focus on different keywords each time. Return only JSON.\n"
            f"Question: {query}"
        )
        try:
            completion = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            variations = json.loads(completion.choices[0].message.content).get("variations", [])
            t_ms = (time.time() - t_start) * 1000
            logger.info(f"Generated {len(variations)} search variations in {t_ms:.1f}ms")
            return [query] + variations  # Include original as first result
        except Exception as e:
            t_ms = (time.time() - t_start) * 1000
            logger.error(f"Failed to generate search variations in {t_ms:.1f}ms: {e}")
            return [query]  # Fall back to original query only

    async def expand_async(self, query: str) -> List[str]:
        """
        Generate search variations asynchronously.
        
        Same as expand() but runs in the background without blocking.
        """
        t_start = time.time()
        prompt = (
            "Generate 3 different ways to search for information about this question. "
            "Focus on different keywords each time. Return only JSON.\n"
            f"Question: {query}"
        )
        client = self.async_client or self.client
        try:
            completion = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            variations = json.loads(completion.choices[0].message.content).get("variations", [])
            t_ms = (time.time() - t_start) * 1000
            logger.info(f"Generated {len(variations)} search variations async in {t_ms:.1f}ms")
            return [query] + variations
        except Exception as e:
            t_ms = (time.time() - t_start) * 1000
            logger.error(f"Failed to generate search variations async in {t_ms:.1f}ms: {e}")
            return [query]
