"""
================================================================================
RAG CONTEXT ENGINE - CENTRALIZED LLM SERVICE
================================================================================
Single point of contact for all LLM interactions. All modules (Engine, Expander,
HyDE) use this service instead of creating their own Groq clients.

To switch providers (e.g., from Groq to OpenAI, Ollama, or Claude), modify
ONLY this file.
"""

import os
import logging
from typing import Optional, List, Dict

from groq import Groq

from core.config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger("RAG.LLM")


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
        self.client = Groq(api_key=self.api_key)
        logger.info(f"LLMService initialized with model: {self.model}")

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
