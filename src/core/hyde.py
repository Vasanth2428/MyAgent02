import time
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.core.config import LLM_MODEL, HYDE_MAX_TOKENS, HYDE_TEMPERATURE

logger = logging.getLogger("RAG.HyDE")

_HYDE_PROMPT = (
    "Write a brief paragraph answering this question. "
    "Focus on the key concepts and terms. No extra commentary.\n"
    "Question: {query}"
)


def _get_hyde_model() -> ChatGroq:
    import os
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("AGENT_API_KEY")
    return ChatGroq(
        model=LLM_MODEL,
        temperature=HYDE_TEMPERATURE,
        max_tokens=HYDE_MAX_TOKENS,
        api_key=api_key,
    )


class HyDEGenerator:
    """
    Creates a hypothetical answer to help find better documents.

    Often called "Hypothetical Document Embeddings" (HyDE). When we're not sure
    what documents to find, this class writes a sample answer first, then uses
    that to search for real documents that match. It's like brainstorming what
    a good answer might look like before we go find it.

    Uses ChatGroq (LangChain-native) instead of raw groq client calls.
    """

    def __init__(self, client=None, *args, **kwargs):
        self.client = client
        self.async_client = client

    def generate_hypothetical_doc(self, query: str) -> str:
        """
        Create a sample answer to improve document search.

        Instead of searching for your exact question, we first create what a good
        answer might look like, then search for documents matching that answer.
        """
        t_start = time.time()
        try:
            if self.client and hasattr(self.client, "chat"):
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
                    temperature=HYDE_TEMPERATURE,
                    max_tokens=HYDE_MAX_TOKENS,
                )
                hypothetical_doc = response.choices[0].message.content.strip()
            else:
                model = _get_hyde_model()
                response = model.invoke([HumanMessage(content=_HYDE_PROMPT.format(query=query))])
                hypothetical_doc = response.content.strip()
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

        Same as generate_hypothetical_doc() but runs in the background using ainvoke.
        """
        t_start = time.time()
        try:
            async_client = getattr(self, "async_client", None) or self.client
            if async_client and hasattr(async_client, "chat"):
                response = await async_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
                    temperature=HYDE_TEMPERATURE,
                    max_tokens=HYDE_MAX_TOKENS,
                )
                hypothetical_doc = response.choices[0].message.content.strip()
            else:
                model = _get_hyde_model()
                response = await model.ainvoke([HumanMessage(content=_HYDE_PROMPT.format(query=query))])
                hypothetical_doc = response.content.strip()
            t_ms = (time.time() - t_start) * 1000
            logger.info(f"Created hypothetical answer async in {t_ms:.1f}ms")
            return hypothetical_doc
        except Exception as e:
            t_ms = (time.time() - t_start) * 1000
            logger.error(f"Async HyDE generation failed in {t_ms:.1f}ms: {e}")
            return query
