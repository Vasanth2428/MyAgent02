import os
from unittest.mock import MagicMock, patch

import pytest

from src.core.model_provider import (
    build_chat_model,
    build_model_with_fallback,
    message_text,
    resolve_model,
    resolve_provider,
)


def test_role_provider_overrides_global_provider():
    with patch.dict(
        os.environ,
        {"LLM_PROVIDER": "groq", "CODING_WORKER_LLM_PROVIDER": "gemini"},
        clear=True,
    ):
        assert resolve_provider("coding_worker") == "google_genai"
        assert resolve_provider("supervisor") == "groq"


def test_fallback_provider_can_differ_from_primary():
    with patch.dict(
        os.environ,
        {
            "CODING_WORKER_LLM_PROVIDER": "google_genai",
            "CODING_WORKER_LLM_FALLBACK_PROVIDER": "cerebras",
        },
        clear=True,
    ):
        assert resolve_provider("coding_worker", "primary") == "google_genai"
        assert resolve_provider("coding_worker", "fallback") == "cerebras"


def test_non_groq_provider_gets_provider_default_model():
    with patch.dict(
        os.environ,
        {"CODING_WORKER_LLM_PROVIDER": "cerebras"},
        clear=True,
    ):
        assert (
            resolve_model("coding_worker", "llama-3.1-8b-instant")
            == "gpt-oss-120b"
        )


def test_role_model_override_has_highest_priority():
    with patch.dict(
        os.environ,
        {
            "CODING_WORKER_LLM_PROVIDER": "google_genai",
            "CODING_WORKER_MODEL_PRIMARY": "custom-gemini-model",
        },
        clear=True,
    ):
        assert (
            resolve_model("coding_worker", "llama-3.1-8b-instant")
            == "custom-gemini-model"
        )


def test_groq_model_preserves_tool_binding():
    base = MagicMock()
    bound = MagicMock()
    base.bind_tools.return_value = bound
    tools = [MagicMock()]

    with patch.dict(os.environ, {"LLM_PROVIDER": "groq"}, clear=True), patch(
        "langchain_groq.ChatGroq", return_value=base
    ) as constructor:
        result = build_chat_model("test_role", "test-model", tools=tools)

    constructor.assert_called_once()
    base.bind_tools.assert_called_once_with(tools)
    assert result is bound


def test_fallback_chain_uses_two_provider_models():
    primary = MagicMock()
    fallback = MagicMock()
    chained = MagicMock()
    primary.with_fallbacks.return_value = chained

    with patch(
        "src.core.model_provider.build_chat_model",
        side_effect=[primary, fallback],
    ) as builder:
        result = build_model_with_fallback("coding_worker", "primary", "fallback")

    assert builder.call_count == 2
    primary.with_fallbacks.assert_called_once_with([fallback])
    assert result is chained


def test_unknown_provider_fails_with_actionable_message():
    with patch.dict(os.environ, {"LLM_PROVIDER": "unknown-provider"}, clear=True):
        with pytest.raises(ValueError, match="Supported providers"):
            build_chat_model("test_role", "test-model")


def test_message_text_normalizes_content_blocks():
    message = MagicMock()
    message.text = None
    message.content = [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
    ]
    assert message_text(message) == "hello world"


def test_google_genai_uses_custom_api_key_envs():
    base = MagicMock()
    
    with patch.dict(
        os.environ,
        {"LLM_PROVIDER": "google_genai", "CUSTOM_GEMINI_KEY": "test-key-value"},
        clear=True,
    ), patch(
        "langchain_google_genai.ChatGoogleGenerativeAI", return_value=base
    ) as constructor:
        result = build_chat_model(
            "test_role", "test-model", api_key_envs=("CUSTOM_GEMINI_KEY",)
        )

    constructor.assert_called_once()
    kwargs = constructor.call_args[1]
    assert kwargs.get("api_key") == "test-key-value"
    assert result is base
