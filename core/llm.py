"""
LLM Service - Talking to the AI Assistant

This module wraps the Groq API client to provide a reliable way to send prompts
to the language model. It handles:
- Retrying failed requests (network timeouts, rate limits)
- Both synchronous and asynchronous communication
- Smart wrapper classes that add error handling automatically
"""

import os
import time
import random
import logging
from typing import Optional, List, Dict
import groq
from groq import Groq, AsyncGroq

from core.config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger("RAG.LLM")


class LLMService:
    """
    Centralized connection to the Groq AI language model.
    
    This class provides a clean interface for talking to the AI:
    - Complete: Send a conversation and get a full response back
    - Complete text: Send a single prompt and get just the response text
    - Complete async: Same as complete but doesn't block while waiting
    
    All methods include automatic retries when the AI service is temporarily
    unavailable.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable missing - please set your API key")
        self.model = model or LLM_MODEL
        
        # Create both sync and async clients
        raw_sync = Groq(api_key=self.api_key)
        self.client = RobustLLMClient(raw_sync, self)
        
        raw_async = AsyncGroq(api_key=self.api_key)
        self.async_client = RobustAsyncLLMClient(raw_async, self)
        
        logger.info(f"LLM Service connected to model: {self.model}")

    def execute_with_retry(self, func, *args, **kwargs):
        """
        Call the AI with automatic retries for common failures.
        
        If the AI service returns rate limit errors, timeouts, or connection issues,
        we wait and try again instead of failing immediately. This makes the
        system more reliable when the AI service is busy.
        """
        def is_llm_transient(e):
            # Retry on temporary API issues
            if isinstance(e, (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError, groq.APITimeoutError)):
                return True
            if hasattr(e, "status_code"):
                return e.status_code == 429 or (e.status_code and e.status_code >= 500)
            return False

        from core.retry import retry
        wrapped = retry(
            retries=5,
            backoff=1.0,
            jitter=0.5,
            is_transient_fn=is_llm_transient,
            logger_name="RAG.LLM"
        )(func)
        return wrapped(*args, **kwargs)

    async def execute_with_retry_async(self, func, *args, **kwargs):
        """
        Executes an async client call with exponential backoff and jitter retry logic.
        """
        def is_llm_transient(e):
            if isinstance(e, (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError, groq.APITimeoutError)):
                return True
            if hasattr(e, "status_code"):
                return e.status_code == 429 or (e.status_code and e.status_code >= 500)
            return False

        from core.retry import retry
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
        """
        Sends a chat completion request to the LLM provider.
        """
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
        """
        Sends an async chat completion request to the LLM provider.
        """
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
        """
        Convenience method: sends a single user message and returns the response text.
        """
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
        """
        Convenience async method: sends a single user message and returns the response text.
        """
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
        """
        Provides direct access to the underlying Groq client (backward compatibility).
        """
        return self.client
