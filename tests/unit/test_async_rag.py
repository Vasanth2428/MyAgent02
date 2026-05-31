import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import os

from core.llm import LLMService
from core.expander import QueryExpander
from core.hyde import HyDEGenerator
from core.services import RetrievalService, GenerationService
from core.engine import RAGContextEngine

class TestAsyncRAG(unittest.IsolatedAsyncioTestCase):

    """
    Unit tests to verify async behavior of RAG components, supporting python's async/await execution.
    """

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.chat.completions.create = MagicMock()
        self.llm_service = MagicMock()
        self.llm_service.model = "test-model"
        self.mock_choice = MagicMock()
        self.mock_choice.message.content = "mocked async response"
        self.mock_choice.delta.content = "mocked async delta"
        self.mock_completion = MagicMock()
        self.mock_completion.choices = [self.mock_choice]
        self.mock_usage = MagicMock()
        self.mock_usage.prompt_tokens = 10
        self.mock_usage.completion_tokens = 20
        self.mock_usage.total_tokens = 30
        self.mock_completion.usage = self.mock_usage

    async def test_async_llm_complete(self):
        raw_async = AsyncMock()
        raw_async.chat.completions.create = AsyncMock(return_value=self.mock_completion)

        from core.llm import RobustAsyncLLMClient
        llm_service_mock = MagicMock()
        llm_service_mock.execute_with_retry_async = AsyncMock(return_value=self.mock_completion)
        client = RobustAsyncLLMClient(raw_async, llm_service_mock)

        res = await client.chat.completions.create(messages=[{"role": "user", "content": "hello"}])
        self.assertEqual(res, self.mock_completion)

    async def test_async_expander(self):
        raw_async = AsyncMock()
        raw_async.chat.completions.create = AsyncMock(return_value=self.mock_completion)

        from core.llm import RobustAsyncLLMClient
        llm_service_mock = MagicMock()
        llm_service_mock.execute_with_retry_async = AsyncMock(return_value=self.mock_completion)
        robust_async = RobustAsyncLLMClient(raw_async, llm_service_mock)

        expander = QueryExpander(MagicMock())
        expander.client = robust_async

        res = await expander.expand_async("test query")
        self.assertTrue(len(res) > 0)

    async def test_async_hyde(self):
        raw_async = AsyncMock()
        raw_async.chat.completions.create = AsyncMock(return_value=self.mock_completion)

        from core.llm import RobustAsyncLLMClient
        llm_service_mock = MagicMock()
        llm_service_mock.execute_with_retry_async = AsyncMock(return_value=self.mock_completion)
        
        robust_async = RobustAsyncLLMClient(raw_async, llm_service_mock)

        hyde = HyDEGenerator(robust_async)
        hyde.async_client = robust_async

        doc = await hyde.generate_hypothetical_doc_async("test query")
        self.assertEqual(doc, "mocked async response")

    async def test_async_retrieval_service(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = ([{"text": "result 1", "source": "file.txt"}], 1.0, 2.0)

        service = RetrievalService(mock_retriever)
        results, embed_lat, db_lat, total_ms = await service.retrieve_async(["query 1", "query 2"], top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(embed_lat, float)
        self.assertIsInstance(db_lat, float)

    async def test_async_generation_service(self):
        raw_async = AsyncMock()
        raw_async.chat.completions.create = AsyncMock(return_value=self.mock_completion)

        from core.llm import RobustAsyncLLMClient
        llm_service_mock = MagicMock()
        llm_service_mock.execute_with_retry_async = AsyncMock(return_value=self.mock_completion)
        robust_async = RobustAsyncLLMClient(raw_async, llm_service_mock)

        service = GenerationService(MagicMock(), model="test", temperature=0.7)
        service.async_client = robust_async

        def mock_count(text):
            return len(text.split())

        import unittest.mock
        with unittest.mock.patch.object(service, '_verify_grounding', return_value=0.85):
            response, prompt, exact_tokens, ctx_used_pct, grounding = await service.generate_async("query", "context", mock_count, ["context chunk"])
        self.assertEqual(response, "mocked async response")
        self.assertTrue(exact_tokens["total"] > 0)
        self.assertEqual(grounding, 0.85)

    async def test_async_scraper(self):
        from core.scraper import scrape_web_page_async

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = AsyncMock(return_value="<html><body>Test content</body></html>")
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_sess = AsyncMock()
        mock_sess.get = MagicMock(return_value=mock_resp)
        mock_sess.__aenter__ = AsyncMock(return_value=mock_sess)
        mock_sess.__aexit__ = AsyncMock(return_value=None)

        with patch('core.scraper._get_aiohttp_session', new=AsyncMock(return_value=mock_sess)):
            result = await scrape_web_page_async("https://example.com")
            self.assertIn("Test content", result)

    async def test_async_scraper_multiple(self):
        from core.scraper import scrape_multiple_pages_async

        with patch('core.scraper.scrape_web_page_async') as mock_scrape:
            mock_scrape.side_effect = ["result 1", "result 2", "result 3"]
            results = await scrape_multiple_pages_async(["url1", "url2", "url3"])
            self.assertEqual(results, ["result 1", "result 2", "result 3"])

    async def test_async_reranker(self):
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.8, 0.5, 0.3]

        from core.reranker import NeuralReranker
        reranker = NeuralReranker.__new__(NeuralReranker)
        reranker._model = mock_encoder

        candidates = [{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]
        results = await reranker.rerank_async("test query", candidates)
        self.assertEqual(len(results), 3)
        self.assertGreater(results[0]["cross_score"], results[-1]["cross_score"])

if __name__ == "__main__":
    unittest.main()