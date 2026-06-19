import time
import logging
from typing import List

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.core.config import LLM_MODEL
from src.core.model_provider import build_chat_model

logger = logging.getLogger("RAG.Expander")

_EXPAND_PROMPT = (
    "Generate 3 different ways to search for information about this question. "
    "Focus on different keywords each time.\n"
    "Question: {query}"
)


class _QueryVariations(BaseModel):
    variations: List[str] = Field(
        description="3 alternative search queries for the original question",
        min_length=1,
        max_length=5,
    )


def _get_expander_model():
    return build_chat_model(
        "expander",
        LLM_MODEL,
        temperature=0,
        api_key_envs=("GROQ_API_KEY", "AGENT_API_KEY"),
        structured_output=_QueryVariations,
    )


class QueryExpander:
    """
    Creates different ways to search for the same question.

    Sometimes a question can be phrased multiple ways. This class asks the AI to
    generate alternative versions of your query, which helps find more documents
    that might be relevant but use different wording.

    Uses a provider-neutral LangChain model with structured output.
    """

    def __init__(self, client=None, *args, **kwargs):
        self.client = client
        self.async_client = client

    def expand(self, query: str) -> List[str]:
        """
        Generate alternative ways to search for the same question.

        For longer questions, this creates 3 variations focused on different keywords.
        This helps find more relevant documents that might use different terminology.
        """
        t_start = time.time()
        try:
            if self.client and hasattr(self.client, "chat"):
                prompt = (
                    "Generate 3 different ways to search for information about this question. "
                    "Focus on different keywords each time. Return only JSON.\n"
                    f"Question: {query}"
                )
                completion = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                import json
                variations = json.loads(completion.choices[0].message.content).get("variations", [])
            else:
                model = _get_expander_model()
                result: _QueryVariations = model.invoke(
                    [HumanMessage(content=_EXPAND_PROMPT.format(query=query))]
                )
                variations = result.variations or []
            t_ms = (time.time() - t_start) * 1000
            logger.info(f"Generated {len(variations)} search variations in {t_ms:.1f}ms")
            return [query] + variations  # Include original as first result
        except Exception as e:
            t_ms = (time.time() - t_start) * 1000
            logger.error(f"Failed to generate search variations in {t_ms:.1f}ms: {e}")
            return [query]

    async def expand_async(self, query: str) -> List[str]:
        """
        Generate search variations asynchronously using ainvoke.

        Same as expand() but runs in the background without blocking.
        """
        t_start = time.time()
        try:
            async_client = getattr(self, "async_client", None) or self.client
            if async_client and hasattr(async_client, "chat"):
                prompt = (
                    "Generate 3 different ways to search for information about this question. "
                    "Focus on different keywords each time. Return only JSON.\n"
                    f"Question: {query}"
                )
                completion = await async_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                import json
                variations = json.loads(completion.choices[0].message.content).get("variations", [])
            else:
                model = _get_expander_model()
                result: _QueryVariations = await model.ainvoke(
                    [HumanMessage(content=_EXPAND_PROMPT.format(query=query))]
                )
                variations = result.variations or []
            t_ms = (time.time() - t_start) * 1000
            logger.info(f"Generated {len(variations)} search variations async in {t_ms:.1f}ms")
            return [query] + variations
        except Exception as e:
            t_ms = (time.time() - t_start) * 1000
            logger.error(f"Failed to generate search variations async in {t_ms:.1f}ms: {e}")
            return [query]
