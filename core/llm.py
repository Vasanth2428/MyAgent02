# LLM Service - Talking to the AI Assistant

import os
import time
import random
import logging
from typing import Optional, List, Dict
import groq
from groq import Groq, AsyncGroq

from core.config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger("RAG.LLM")
from core.retry import retry


class RobustLLMClient:
    def __init__(self, raw_client, llm_service):
        self.raw_client = raw_client
        self.llm_service = llm_service
        self.chat = RobustChat(raw_client.chat, llm_service)

    def __getattr__(self, name):
        return getattr(self.raw_client, name)


class RobustChat:
    def __init__(self, raw_chat, llm_service):
        self.raw_chat = raw_chat
        self.llm_service = llm_service
        self.completions = RobustCompletions(raw_chat.completions, llm_service)

    def __getattr__(self, name):
        return getattr(self.raw_chat, name)


class RobustCompletions:
    def __init__(self, raw_completions, llm_service):
        self.raw_completions = raw_completions
        self.llm_service = llm_service

    def create(self, *args, **kwargs):
        return self.llm_service.execute_with_retry(self.raw_completions.create, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.raw_completions, name)


class RobustAsyncLLMClient:
    def __init__(self, raw_client, llm_service):
        self.raw_client = raw_client
        self.llm_service = llm_service
        self.chat = RobustAsyncChat(raw_client.chat, llm_service)

    def __getattr__(self, name):
        return getattr(self.raw_client, name)


class RobustAsyncChat:
    def __init__(self, raw_chat, llm_service):
        self.raw_chat = raw_chat
        self.llm_service = llm_service
        self.completions = RobustAsyncCompletions(raw_chat.completions, llm_service)

    def __getattr__(self, name):
        return getattr(self.raw_chat, name)


class RobustAsyncCompletions:
    def __init__(self, raw_completions, llm_service):
        self.raw_completions = raw_completions
        self.llm_service = llm_service

    async def create(self, *args, **kwargs):
        return await self.llm_service.execute_with_retry_async(self.raw_completions.create, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.raw_completions, name)


class LLMService:
    """Centralized connection to the Groq AI language model."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable missing - please set your API key")
        self.model = model or LLM_MODEL

        self._raw_sync = Groq(api_key=self.api_key)
        self._raw_async = AsyncGroq(api_key=self.api_key)

        self.client = RobustLLMClient(self._raw_sync, self)
        self.async_client = RobustAsyncLLMClient(self._raw_async, self)

        logger.info(f"LLM Service connected to model: {self.model}")

    def execute_with_retry(self, func, *args, **kwargs):
        def is_llm_transient(e):
            if isinstance(e, (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError, groq.APITimeoutError)):
                return True
            if hasattr(e, "status_code"):
                return e.status_code == 429 or (e.status_code and e.status_code >= 500)
            return False

        wrapped = retry(
            retries=5,
            backoff=1.0,
            jitter=0.5,
            is_transient_fn=is_llm_transient,
            logger_name="RAG.LLM"
        )(func)
        return wrapped(*args, **kwargs)

    async def execute_with_retry_async(self, func, *args, **kwargs):
        def is_llm_transient(e):
            if isinstance(e, (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError, groq.APITimeoutError)):
                return True
            if hasattr(e, "status_code"):
                return e.status_code == 429 or (e.status_code and e.status_code >= 500)
            return False

        wrapped = retry(
            retries=5,
            backoff=1.0,
            jitter=0.5,
            is_transient_fn=is_llm_transient,
            logger_name="RAG.LLM"
        )(func)
        return await wrapped(*args, **kwargs)

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> object:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "frequency_penalty": 0.3,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        return self.client.chat.completions.create(**kwargs)

    async def complete_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> object:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "frequency_penalty": 0.3,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        return await self.async_client.chat.completions.create(**kwargs)

    def complete_text(
        self,
        prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        completion = self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return completion.choices[0].message.content.strip()

    async def complete_text_async(
        self,
        prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        completion = await self.complete_async(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return completion.choices[0].message.content.strip()

    @property
    def raw_client(self):
        return self.client
