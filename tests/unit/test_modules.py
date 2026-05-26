import unittest
from unittest.mock import MagicMock, patch
import os
import shutil

# Set environment variable dummy key for tests if not present
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "gsk_dummy_key_for_testing_purposes"

from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from core.splitter import RecursiveCharacterSplitter
from core.compressor import Compressor
from core.memory import ConversationMemory
from core.persistence import PersistentMemoryStore
from core.expander import QueryExpander
from core.hyde import HyDEGenerator


class TestRecursiveCharacterSplitter(unittest.TestCase):
    def test_basic_splitting(self):
        splitter = RecursiveCharacterSplitter(chunk_size=20, overlap=5)
        text = "Hello world. This is a simple text splitter test."
        chunks = splitter.split_text(text)
        self.assertTrue(len(chunks) > 0)
        self.assertTrue(all(len(c) <= 25 for c in chunks)) # 20 + potential overlap boundary

    def test_empty_string(self):
        splitter = RecursiveCharacterSplitter()
        self.assertEqual(splitter.split_text(""), [])


class TestCompressor(unittest.TestCase):
    def test_compress_under_budget(self):
        text = ["Short sentence."]
        res = Compressor.compress(text, "query", max_tokens=100)
        self.assertEqual(res, "Short sentence.")

    def test_compress_over_budget(self):
        docs = [
            "This is a sentence about dogs.",
            "This is a sentence about cats.",
            "This is completely unrelated data."
        ]
        res = Compressor.compress(docs, "cats", max_tokens=10)
        self.assertIn("cats", res)
        self.assertNotIn("dogs", res)


class TestMemory(unittest.TestCase):
    def test_memory_deduplication(self):
        mem = ConversationMemory()
        mem.add("This is an entry about routing protocols.", role="user")
        mem.add("This is an entry about routing protocols.", role="user") # High semantic overlap
        self.assertEqual(len(mem.entries), 1)

    def test_decay_effect(self):
        mem = ConversationMemory(decay_rate=100.0) # Instant decay
        mem.add("Old memory.", role="user")
        # Artificially set timestamp to 1 hour ago
        from datetime import datetime, timedelta
        mem.entries[0].last_seen = datetime.now() - timedelta(hours=1)
        active = mem.get_active_context()
        self.assertEqual(active, "")


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_memory.db"
        self.store = PersistentMemoryStore(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_add_and_retrieve_history(self):
        self.store.add_entry("session-1", "Hello AI", "user")
        self.store.add_entry("session-1", "Hello User", "assistant")
        history = self.store.get_history("session-1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")


class TestExpanderAndHyDE(unittest.TestCase):
    def test_expander_error_handling(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API offline")
        expander = QueryExpander(mock_client)
        variations = expander.expand("test query")
        self.assertEqual(variations, ["test query"])

    def test_hyde_error_handling(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API offline")
        hyde = HyDEGenerator(mock_client)
        doc = hyde.generate_hypothetical_doc("test query")
        self.assertEqual(doc, "test query")


class TestCompressorSegmentation(unittest.TestCase):
    """Tests for the paragraph/code-block-level segmentation in Compressor."""

    def test_code_block_preserved(self):
        """Code blocks wrapped in triple backticks must stay intact."""
        doc = "Here is some code:\n\n```python\ndef hello():\n    print('hi')\n```\n\nEnd of doc."
        segments = Compressor._split_into_segments([doc])
        # Find the segment that contains the code block
        code_segments = [s for s in segments if "```" in s]
        self.assertEqual(len(code_segments), 1, "Code block should be a single segment")
        self.assertIn("def hello():", code_segments[0])
        self.assertIn("print('hi')", code_segments[0])

    def test_paragraphs_split_on_blank_lines(self):
        """Paragraphs separated by blank lines should become separate segments."""
        doc = "First paragraph about networking.\n\nSecond paragraph about routing.\n\nThird paragraph about DNS."
        segments = Compressor._split_into_segments([doc])
        self.assertEqual(len(segments), 3)
        self.assertIn("networking", segments[0])
        self.assertIn("routing", segments[1])
        self.assertIn("DNS", segments[2])

    def test_single_line_filtered(self):
        """Segments shorter than 10 chars should be filtered out."""
        doc = "OK\n\nThis is a valid paragraph with enough content."
        segments = Compressor._split_into_segments([doc])
        self.assertEqual(len(segments), 1)
        self.assertIn("valid paragraph", segments[0])

    def test_compress_preserves_formatting(self):
        """Compressed output should join with double newlines, not periods."""
        docs = [
            "Paragraph one about cats and dogs and animals in general.",
            "Paragraph two about cats specifically and their behavior patterns."
        ]
        result = Compressor.compress(docs, "cats", max_tokens=500)
        # Under budget, fast path returns joined text
        self.assertIn("cats", result)

    def test_compress_empty_input(self):
        result = Compressor.compress([], "query")
        self.assertEqual(result, "")


class TestRerankerNormalization(unittest.TestCase):
    """Tests that the reranker outputs sigmoid-normalized scores in [0, 1]."""

    @patch('core.reranker.NeuralReranker.__init__', return_value=None)
    def test_scores_bounded_zero_to_one(self, mock_init):
        """All cross_score values must be between 0.0 and 1.0 after sigmoid."""
        import math
        from core.reranker import NeuralReranker

        reranker = NeuralReranker.__new__(NeuralReranker)
        # Create a mock model that returns raw logits
        reranker.model = MagicMock()
        reranker.model.predict.return_value = [7.5, -3.2, 0.0, -11.4, 2.1]

        candidates = [
            {"text": "doc about cats", "score": 0.9},
            {"text": "doc about dogs", "score": 0.8},
            {"text": "doc about birds", "score": 0.7},
            {"text": "doc about fish", "score": 0.6},
            {"text": "doc about mice", "score": 0.5},
        ]

        result = reranker.rerank("cats", candidates)

        for r in result:
            self.assertGreaterEqual(r["cross_score"], 0.0, f"Score {r['cross_score']} is below 0")
            self.assertLessEqual(r["cross_score"], 1.0, f"Score {r['cross_score']} is above 1")
            self.assertIn("raw_score", r, "Raw score should be preserved")

    @patch('core.reranker.NeuralReranker.__init__', return_value=None)
    def test_scores_sorted_descending(self, mock_init):
        """Results must be sorted by cross_score descending."""
        from core.reranker import NeuralReranker

        reranker = NeuralReranker.__new__(NeuralReranker)
        reranker.model = MagicMock()
        reranker.model.predict.return_value = [1.0, 5.0, -2.0]

        candidates = [
            {"text": "low relevance doc", "score": 0.3},
            {"text": "high relevance doc", "score": 0.9},
            {"text": "negative relevance doc", "score": 0.1},
        ]

        result = reranker.rerank("test query", candidates)
        scores = [r["cross_score"] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    @patch('core.reranker.NeuralReranker.__init__', return_value=None)
    def test_sigmoid_of_zero_is_half(self, mock_init):
        """Sigmoid(0) should equal exactly 0.5."""
        from core.reranker import NeuralReranker

        reranker = NeuralReranker.__new__(NeuralReranker)
        reranker.model = MagicMock()
        reranker.model.predict.return_value = [0.0]

        candidates = [{"text": "neutral doc", "score": 0.5}]
        result = reranker.rerank("query", candidates)
        self.assertAlmostEqual(result[0]["cross_score"], 0.5, places=5)

    @patch('core.reranker.NeuralReranker.__init__', return_value=None)
    def test_empty_candidates(self, mock_init):
        """Reranker should return empty list for empty input."""
        from core.reranker import NeuralReranker

        reranker = NeuralReranker.__new__(NeuralReranker)
        reranker.model = MagicMock()

        result = reranker.rerank("query", [])
        self.assertEqual(result, [])


class TestXMLContextFormatting(unittest.TestCase):
    """Tests that context is formatted with XML document tags."""

    def test_xml_tags_present(self):
        """Verify that XML document tags are generated for context blocks."""
        # Simulate what _phase_refine does for simple mode
        results = [
            {"text": "Document about routing protocols.", "source": "routing.txt"},
            {"text": "Document about DNS resolution.", "source": "dns.txt"},
        ]
        formatted_parts = []
        for r in results:
            source = r.get("source", "unknown")
            formatted_parts.append(f'<document source="{source}">\n{r["text"]}\n</document>')
        context = "\n\n".join(formatted_parts)

        self.assertIn('<document source="routing.txt">', context)
        self.assertIn('<document source="dns.txt">', context)
        self.assertIn("</document>", context)
        self.assertEqual(context.count("<document"), 2)
        self.assertEqual(context.count("</document>"), 2)

    def test_unknown_source_fallback(self):
        """Documents without a source field should default to 'unknown'."""
        results = [{"text": "Some orphan text."}]
        for r in results:
            source = r.get("source", "unknown")
            tag = f'<document source="{source}">\n{r["text"]}\n</document>'

        self.assertIn('source="unknown"', tag)


class TestLLMService(unittest.TestCase):
    """Tests for the centralized LLM service wrapper."""

    @patch('core.llm.Groq')
    def test_llm_service_initializes(self, MockGroq):
        """LLMService should create a Groq client with the env API key."""
        from core.llm import LLMService
        service = LLMService(api_key="test-key-123")
        MockGroq.assert_called_once_with(api_key="test-key-123")
        self.assertIsNotNone(service.client)

    @patch('core.llm.Groq')
    def test_complete_text_returns_string(self, MockGroq):
        """complete_text should return a stripped string from the LLM."""
        from core.llm import LLMService

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "  Hello World  "
        MockGroq.return_value.chat.completions.create.return_value = mock_completion

        service = LLMService(api_key="test-key")
        result = service.complete_text("Say hello")
        self.assertEqual(result, "Hello World")

    @patch('core.llm.Groq')
    def test_raw_client_property(self, MockGroq):
        """raw_client should return the underlying Groq client."""
        from core.llm import LLMService
        service = LLMService(api_key="test-key")
        self.assertEqual(service.raw_client, service.client)

    def test_missing_api_key_raises(self):
        """LLMService should raise ValueError if no API key is available."""
        from core.llm import LLMService
        original = os.environ.get("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = ""
        try:
            with self.assertRaises(ValueError):
                LLMService(api_key=None)
        finally:
            if original:
                os.environ["GROQ_API_KEY"] = original
class TestLLMServiceRobustness(unittest.TestCase):
    @patch('core.llm.Groq')
    @patch('time.sleep')  # Mock sleep so tests run instantly
    def test_retry_on_rate_limit_eventual_success(self, mock_sleep, MockGroq):
        """Should retry upon encountering RateLimitError and succeed eventually."""
        import httpx
        import groq
        from core.llm import LLMService

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Success Answer"

        req = httpx.Request('POST', 'https://api.groq.com')
        r_429 = httpx.Response(status_code=429, request=req)

        # Mock side effect: 2 rate limits, then success
        mock_create = MockGroq.return_value.chat.completions.create
        mock_create.side_effect = [
            groq.RateLimitError("Rate limited", response=r_429, body=None),
            groq.RateLimitError("Rate limited again", response=r_429, body=None),
            mock_completion
        ]

        service = LLMService(api_key="test-key")
        result = service.complete_text("Test retry")

        self.assertEqual(result, "Success Answer")
        self.assertEqual(mock_create.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('core.llm.Groq')
    @patch('time.sleep')
    def test_retry_exhaustion_raises(self, mock_sleep, MockGroq):
        """Should raise error after exhausting all retries."""
        import httpx
        import groq
        from core.llm import LLMService

        req = httpx.Request('POST', 'https://api.groq.com')
        r_429 = httpx.Response(status_code=429, request=req)

        mock_create = MockGroq.return_value.chat.completions.create
        mock_create.side_effect = groq.RateLimitError("Rate limited constantly", response=r_429, body=None)

        service = LLMService(api_key="test-key")
        with self.assertRaises(groq.RateLimitError):
            service.complete_text("Test retry failure")

        self.assertEqual(mock_create.call_count, 5)
        self.assertEqual(mock_sleep.call_count, 4)

    @patch('core.llm.Groq')
    @patch('time.sleep')
    def test_no_retry_on_validation_error(self, mock_sleep, MockGroq):
        """Should NOT retry on non-retriable exceptions (e.g., normal ValueError)."""
        from core.llm import LLMService

        mock_create = MockGroq.return_value.chat.completions.create
        mock_create.side_effect = ValueError("Some custom user input error")

        service = LLMService(api_key="test-key")
        with self.assertRaises(ValueError):
            service.complete_text("Test non-retriable")

        self.assertEqual(mock_create.call_count, 1)
        self.assertEqual(mock_sleep.call_count, 0)

    @patch('core.llm.Groq')
    @patch('time.sleep')
    def test_retry_on_custom_status_code_error(self, mock_sleep, MockGroq):
        """Should retry on custom exceptions that carry a 429 or 5xx status_code attribute."""
        from core.llm import LLMService

        class CustomAPIStatusError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(f"Error {status_code}")

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Status Success Answer"

        mock_create = MockGroq.return_value.chat.completions.create
        mock_create.side_effect = [
            CustomAPIStatusError(status_code=502),
            mock_completion
        ]

        service = LLMService(api_key="test-key")
        result = service.complete_text("Test status code retry")

        self.assertEqual(result, "Status Success Answer")
        self.assertEqual(mock_create.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)


class TestPersistenceRobustness(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_robustness_memory.db"
        
    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    @patch('sqlite3.connect')
    @patch('time.sleep')
    def test_execute_with_retry_locks(self, mock_sleep, mock_connect):
        """Should retry upon encountering OperationalError (locked/busy) and eventually succeed."""
        from core.persistence import PersistentMemoryStore
        import sqlite3
        
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        store = PersistentMemoryStore(db_path=self.db_path)
        
        # Now let's test a custom function run through execute_with_retry
        mock_func = MagicMock()
        mock_func.side_effect = [
            sqlite3.OperationalError("database is locked"),
            sqlite3.OperationalError("database is locked"),
            "success_result"
        ]
        
        result = store.execute_with_retry(mock_func)
        self.assertEqual(result, "success_result")
        self.assertEqual(mock_func.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('sqlite3.connect')
    @patch('time.sleep')
    def test_execute_with_retry_exhaustion(self, mock_sleep, mock_connect):
        """Should raise OperationalError after exhausting all retries."""
        from core.persistence import PersistentMemoryStore
        import sqlite3
        
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        store = PersistentMemoryStore(db_path=self.db_path)
        
        mock_func = MagicMock()
        mock_func.side_effect = sqlite3.OperationalError("database is locked")
        
        with self.assertRaises(sqlite3.OperationalError):
            store.execute_with_retry(mock_func)
            
        self.assertEqual(mock_func.call_count, 5)
        self.assertEqual(mock_sleep.call_count, 4)

    def test_wal_mode_enabled(self):
        """Verify WAL journal mode is active in the sqlite database."""
        from core.persistence import PersistentMemoryStore
        import sqlite3
        
        store = PersistentMemoryStore(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA journal_mode;")
            mode = cursor.fetchone()[0]
            self.assertEqual(mode.lower(), "wal")


class TestRetrieverRobustness(unittest.TestCase):
    @patch('weaviate.connect_to_weaviate_cloud')
    @patch('core.retriever.SentenceTransformer')
    @patch('time.sleep')
    def test_weaviate_connection_retries(self, mock_sleep, mock_st, mock_connect):
        """Verify that Retriever tries to connect multiple times on failure."""
        from core.retriever import WeaviateRetriever
        
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True
        
        # 2 failures, then 1 success
        mock_connect.side_effect = [
            Exception("Connection timed out"),
            Exception("Connection refused"),
            mock_client
        ]
        
        retriever = WeaviateRetriever()
        
        self.assertEqual(mock_connect.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertEqual(retriever.client, mock_client)

    @patch('weaviate.connect_to_weaviate_cloud')
    @patch('core.retriever.SentenceTransformer')
    @patch('time.sleep')
    def test_weaviate_connection_exhaustion(self, mock_sleep, mock_st, mock_connect):
        """Verify that connection failure after max retries still allows retriever to be created in degraded mode."""
        from core.retriever import WeaviateRetriever
        
        mock_connect.side_effect = Exception("Permanent connection error")
        
        # Retriever should be created but in degraded mode (no exception raised)
        retriever = WeaviateRetriever()
        self.assertFalse(retriever._connected)
        self.assertEqual(mock_connect.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('weaviate.connect_to_weaviate_cloud')
    @patch('core.retriever.SentenceTransformer')
    @patch('time.sleep')
    def test_execute_with_retry_transient_failure(self, mock_sleep, mock_st, mock_connect):
        """Should retry on transient errors and succeed."""
        from core.retriever import WeaviateRetriever
        import weaviate.exceptions
        
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True
        mock_connect.return_value = mock_client
        
        retriever = WeaviateRetriever()
        
        mock_op = MagicMock()
        mock_op.side_effect = [
            weaviate.exceptions.WeaviateConnectionError("connection rate limit 429 error"),
            "success_data"
        ]
        
        result = retriever.execute_with_retry(mock_op)
        self.assertEqual(result, "success_data")
        self.assertEqual(mock_op.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch('weaviate.connect_to_weaviate_cloud')
    @patch('core.retriever.SentenceTransformer')
    @patch('time.sleep')
    def test_execute_with_retry_non_transient_raises(self, mock_sleep, mock_st, mock_connect):
        """Should NOT retry on non-transient errors."""
        from core.retriever import WeaviateRetriever
        
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True
        mock_connect.return_value = mock_client
        
        retriever = WeaviateRetriever()
        
        mock_op = MagicMock()
        mock_op.side_effect = ValueError("invalid query argument")
        
        with self.assertRaises(ValueError):
            retriever.execute_with_retry(mock_op)
            
        self.assertEqual(mock_op.call_count, 1)
        self.assertEqual(mock_sleep.call_count, 0)


if __name__ == "__main__":
    unittest.main()

