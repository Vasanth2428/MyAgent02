"""
================================================================================
RAG CONTEXT ENGINE - HyDE MODULE
================================================================================
Hypothetical Document Embeddings: generates a draft response to align
dense retrieval vectors for improved semantic matching.
"""

import time
import logging

from core.config import LLM_MODEL, HYDE_MAX_TOKENS, HYDE_TEMPERATURE

logger = logging.getLogger("RAG.HyDE")


class HyDEGenerator:
    """
    Hypothetical Document Embeddings (HyDE) generator.
    Generates a draft response used to align dense retrieval vectors.
    """

    def __init__(self, groq_client):
        self.client = groq_client

    def generate_hypothetical_doc(self, query: str) -> str:
        """
        Creates a brief, hypothetical response answering the query.
        """
        t_start = time.time()
        prompt = (
            "Write a brief, hypothetical paragraph answering the query below. "
            "Focus on outlining the key concepts and terminology. "
            "Do not write conversational preamble. Go straight to the point.\n"
            f"Query: {query}"
        )
        try:
            completion = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=HYDE_MAX_TOKENS,
                temperature=HYDE_TEMPERATURE
            )
            hypothetical_doc = completion.choices[0].message.content.strip()
            t_ms = (time.time() - t_start) * 1000
            logger.info(f"Generated hypothetical document in {t_ms:.1f}ms")
            return hypothetical_doc
        except Exception as e:
            t_ms = (time.time() - t_start) * 1000
            logger.error(f"HyDE generation failed in {t_ms:.1f}ms: {e}")
            return query
