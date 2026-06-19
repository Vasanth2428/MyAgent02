"""Provider-neutral LangChain chat model construction.

All multi-agent workers should obtain models through this module instead of
importing a provider-specific chat class. Providers and models can then be
changed globally or for one worker through environment variables.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Sequence


_PROVIDER_DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "google_genai": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "cerebras": "gpt-oss-120b",
}

_PROVIDER_ALIASES = {
    "gemini": "google_genai",
    "google": "google_genai",
    "google-genai": "google_genai",
}


def message_text(message: Any) -> str:
    """Normalize LangChain message content across provider-specific formats."""
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):
        value = text()
        if isinstance(value, str):
            return value

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_text = block.get("text")
                if isinstance(block_text, str):
                    parts.append(block_text)
        return "".join(parts)
    return str(content or "")


def _env_prefix(role: str) -> str:
    return role.strip().upper().replace("-", "_").replace(" ", "_")


def resolve_provider(role: str, variant: str = "primary") -> str:
    """Resolve a worker-specific provider, falling back to the global provider."""
    prefix = _env_prefix(role)
    variant_name = variant.strip().upper()
    provider = (
        os.getenv(f"{prefix}_LLM_{variant_name}_PROVIDER")
        or os.getenv(f"{prefix}_LLM_PROVIDER")
        or (
            os.getenv("LLM_FALLBACK_PROVIDER")
            if variant_name == "FALLBACK"
            else None
        )
        or os.getenv("LLM_PROVIDER", "groq")
    )
    normalized = provider.strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def resolve_model(role: str, default_model: str, variant: str = "primary") -> str:
    """Resolve role/provider-specific model overrides."""
    prefix = _env_prefix(role)
    provider = resolve_provider(role, variant)
    variant_name = variant.strip().upper()
    return (
        os.getenv(f"{prefix}_MODEL_{variant_name}")
        or os.getenv(f"{provider.upper()}_MODEL_{variant_name}")
        or (
            default_model
            if provider == "groq"
            else _PROVIDER_DEFAULT_MODELS.get(provider, default_model)
        )
    )


def _first_env(names: Iterable[str]) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _create_base_model(
    *,
    provider: str,
    model: str,
    temperature: float,
    api_key_envs: Sequence[str],
    max_tokens: Optional[int] = None,
    **kwargs: Any,
):
    common: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        **kwargs,
    }
    if max_tokens is not None:
        common["max_tokens"] = max_tokens

    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = _first_env((*api_key_envs, "GROQ_API_KEY", "AGENT_API_KEY"))
        return ChatGroq(api_key=api_key, **common)

    if provider == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = _first_env((*api_key_envs, "GOOGLE_API_KEY", "GEMINI_API_KEY"))
        return ChatGoogleGenerativeAI(api_key=api_key, **common)

    if provider in {"openai", "cerebras"}:
        from langchain_openai import ChatOpenAI

        if provider == "cerebras":
            api_key = _first_env((*api_key_envs, "CEREBRAS_API_KEY",))
            base_url = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
        else:
            api_key = _first_env((*api_key_envs, "OPENAI_API_KEY",))
            base_url = os.getenv("OPENAI_BASE_URL")

        if base_url:
            common["base_url"] = base_url
        return ChatOpenAI(api_key=api_key, **common)

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. "
        "Supported providers: groq, google_genai, openai, cerebras."
    )


def build_chat_model(
    role: str,
    default_model: str,
    *,
    temperature: float = 0,
    variant: str = "primary",
    api_key_envs: Sequence[str] = (),
    tools: Optional[Sequence[Any]] = None,
    structured_output: Optional[type] = None,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
):
    """Build one configured model while preserving LangChain capabilities."""
    provider = resolve_provider(role, variant)
    model_name = resolve_model(role, default_model, variant)
    model = _create_base_model(
        provider=provider,
        model=model_name,
        temperature=temperature,
        api_key_envs=api_key_envs,
        max_tokens=max_tokens,
        **kwargs,
    )
    if tools:
        model = model.bind_tools(tools)
    if structured_output is not None:
        model = model.with_structured_output(structured_output)
    return model


def build_model_with_fallback(
    role: str,
    primary_model: str,
    fallback_model: str,
    *,
    temperature: float = 0,
    api_key_envs: Sequence[str] = (),
    tools: Optional[Sequence[Any]] = None,
    structured_output: Optional[type] = None,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
):
    """Build a primary model and a compatible fallback model."""
    primary = build_chat_model(
        role,
        primary_model,
        temperature=temperature,
        variant="primary",
        api_key_envs=api_key_envs,
        tools=tools,
        structured_output=structured_output,
        max_tokens=max_tokens,
        **kwargs,
    )
    fallback = build_chat_model(
        role,
        fallback_model,
        temperature=temperature,
        variant="fallback",
        api_key_envs=api_key_envs,
        tools=tools,
        structured_output=structured_output,
        max_tokens=max_tokens,
        **kwargs,
    )
    return primary.with_fallbacks([fallback])
