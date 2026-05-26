import os
import time
import random
import logging
from typing import Optional, List, Dict
import groq
from groq import Groq

from core.config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger("RAG.LLM")


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


class LLMService:
    """
    Centralized LLM wrapper. Provides a unified interface for chat completions
    so that provider changes only need to happen in one place.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable missing")
        self.model = model or LLM_MODEL
        raw = Groq(api_key=self.api_key)
        self.client = RobustLLMClient(raw, self)
        logger.info(f"LLMService initialized with model: {self.model}")

    def execute_with_retry(self, func, *args, **kwargs):
        """
        Executes a client call with exponential backoff and jitter retry logic.
        Catches RateLimitError, APIConnectionError, InternalServerError, and APITimeoutError.
        """
        max_retries = 5
        base_delay = 1.0  # seconds
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError, groq.APITimeoutError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    f"LLM call encountered retriable error: {e}. "
                    f"Retrying in {delay:.2f}s (Attempt {attempt+1}/{max_retries})...."
                )
                time.sleep(delay)
            except Exception as e:
                # Catch general API status errors that represent 429 (Rate Limit) or 5xx (Server Error)
                is_retriable = False
                if hasattr(e, "status_code"):
                    if e.status_code == 429 or (e.status_code and e.status_code >= 500):
                        is_retriable = True
                
                if is_retriable:
                    if attempt == max_retries - 1:
                        logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        f"LLM call encountered status code error ({e.status_code}): {e}. "
                        f"Retrying in {delay:.2f}s (Attempt {attempt+1}/{max_retries})...."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"LLM call encountered non-retriable error: {type(e).__name__}: {e}")
                    raise

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> object:
        """
        Sends a chat completion request to the LLM provider.

        Args:
            messages: List of {"role": ..., "content": ...} message dicts.
            temperature: Sampling temperature.
            max_tokens: Optional max tokens for the completion.
            response_format: Optional response format (e.g., {"type": "json_object"}).

        Returns:
            The raw completion object from the provider.
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "frequency_penalty": 0.3, # Added default frequency penalty to prevent repetition
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        return self.client.chat.completions.create(**kwargs)

    def complete_text(
        self,
        prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Convenience method: sends a single user message and returns the response text.

        Args:
            prompt: The user prompt string.
            temperature: Sampling temperature.
            max_tokens: Optional max tokens.
            response_format: Optional response format.

        Returns:
            The response text string.
        """
        messages = [{"role": "user", "content": prompt}]
        completion = self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return completion.choices[0].message.content.strip()

    @property
    def raw_client(self):
        """
        Provides direct access to the underlying Groq client for modules
        that need provider-specific features (backward compatibility).
        """
        return self.client
