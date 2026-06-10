import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import groq

from src.core.engine import RAGContextEngine
from src.core.retriever import WeaviateRetriever

class TestRetryVisibility(unittest.IsolatedAsyncioTestCase):
    @patch("src.core.llm.Groq")
    @patch("src.core.llm.AsyncGroq")
    async def test_ask_stream_async_bubbles_retry_warning(self, mock_async_groq_class, mock_groq_class):
        # Setup mock sync/async clients
        mock_async_client = MagicMock()
        mock_async_groq_class.return_value = mock_async_client
        mock_async_client.chat = MagicMock()
        mock_async_client.chat.completions = AsyncMock()
        
        # We need the first call to raise RateLimitError, second to return mock completion
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "0.1"}
        
        rate_limit_err = groq.RateLimitError(
            message="Rate limit reached. Limit: 3, Used: 4. Please try again in 0.1s.",
            response=mock_response,
            body={"error": {"message": "Rate limit reached"}}
        )
        
        # Mock successful completion choice
        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.content = "Paris is the capital of France."
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock()
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        mock_completion.usage.total_tokens = 15
        
        # Set the side_effect: first call raises RateLimitError, second succeeds
        mock_async_client.chat.completions.create.side_effect = [rate_limit_err, mock_completion]
        
        # Setup Retriever Mock
        mock_retriever = MagicMock(spec=WeaviateRetriever)
        mock_retriever.retrieve_async = AsyncMock(return_value=([{"text": "France capital is Paris", "score": 0.9, "source": "test.txt"}], 1.0, 1.0, 2.0))
        mock_retriever.retrieve = MagicMock(return_value=([{"text": "France capital is Paris", "score": 0.9, "source": "test.txt"}], 1.0, 1.0))
        
        # Initialize RAG Engine
        engine = RAGContextEngine(retriever=mock_retriever)
        # Ensure it is not in mock mode by setting is_mock to False
        engine.llm_service.is_mock = False
        # Use mock client
        engine.llm_service.async_client.raw_client = mock_async_client
        
        # Run ask_stream_async in context_engine mode
        events = []
        async for event in engine.ask_stream_async(
            query="What is the capital of France?",
            session_id="test_session",
            mode="context_engine",
            top_k=1
        ):
            events.append(event)
            
        # Verify that we received the thought event about rate limit warning
        thought_events = [e for e in events if e.get("event") == "thought"]
        retry_warnings = [t for t in thought_events if "[Rate Limit / API Error]" in t.get("text", "")]
        
        self.assertTrue(len(retry_warnings) > 0, "Should bubble up retry warning thought event")
        self.assertIn("Attempt 1 failed", retry_warnings[0]["text"])
