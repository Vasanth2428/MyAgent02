import time
import logging

from core.config import LLM_MODEL, HYDE_MAX_TOKENS, HYDE_TEMPERATURE

logger = logging.getLogger("RAG.HyDE")


class HyDEGenerator:
    """
    Creates a hypothetical answer to help find better documents.
    
    Often called "Hypothetical Document Embeddings" (HyDE). When we're not sure
    what documents to find, this class writes a sample answer first, then uses
    that to search for real documents that match. It's like brainstorming what
    a good answer might look like before we go find it.
    """

    def __init__(self, groq_client, async_client=None):
        self.client = groq_client
        self.async_client = async_client
        if not self.async_client and hasattr(groq_client, "llm_service"):
            self.async_client = groq_client.llm_service.async_client

    def generate_hypothetical_doc(self, query: str) -> str:
        """
        Create a sample answer to improve document search.
        
        Instead of searching for your exact question, we first create what a good
        answer might look like, then search for documents matching that answer.
        This often finds more relevant documents than searching the original query.
        """
        t_start = time.time()
        prompt = (
            "Write a brief paragraph answering this question. "
            "Focus on the key concepts and terms. No extra commentary.\n"
            f"Question: {query}"
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
            logger.info(f"Created hypothetical answer in {t_ms:.1f}ms")
            return hypothetical_doc
        except Exception as e:
            t_ms = (time.time() - t_start) * 1000
            logger.error(f"HyDE generation failed in {t_ms:.1f}ms: {e}")
            return query

    async def generate_hypothetical_doc_async(self, query: str) -> str:
        """
        Create a sample answer asynchronously.
        
        Same as generate_hypothetical_doc() but runs in the background.
        """
        t_start = time.time()
        prompt = (
            "Write a brief paragraph answering this question. "
            "Focus on the key concepts and terms. No extra commentary.\n"
            f"Question: {query}"
        )
        client = self.async_client or self.client
        try:
            completion = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=HYDE_MAX_TOKENS,
                temperature=HYDE_TEMPERATURE
            )
            hypothetical_doc = completion.choices[0].message.content.strip()
            t_ms = (time.time() - t_start) * 1000
            logger.info(f"Created hypothetical answer async in {t_ms:.1f}ms")
            return hypothetical_doc
        except Exception as e:
            t_ms = (time.time() - t_start) * 1000
            logger.error(f"Async HyDE generation failed in {t_ms:.1f}ms: {e}")
            return query
