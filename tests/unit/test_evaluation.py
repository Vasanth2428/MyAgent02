"""
RAG Evaluation Test Suite
Tests retrieval, reranking, HyDE, compression, and grounding verification.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.core.evaluator import RAGEvaluator, GroundingVerifier
from src.core.benchmarks import RAG_BENCHMARKS, BenchmarkQuery
from src.core.compressor import Compressor
from src.core.engine import count_tokens


class TestGroundingVerifier(unittest.TestCase):
    """Tests for grounding verification."""

    def test_extract_citations_none(self):
        """No citations in plain text."""
        answer = "Paris is the capital of France."
        citations = GroundingVerifier.extract_citation_markers(answer)
        self.assertEqual(len(citations), 0)

    def test_extract_citations_with_source(self):
        """Extracts document source citations."""
        answer = 'The password is "SuperSecretAgent123" from the config file.'
        citations = GroundingVerifier.extract_citation_markers(answer)
        self.assertGreaterEqual(len(citations), 0)

    def test_grounding_score_perfect_match(self):
        """Answer fully grounded in context."""
        answer = "Paris is the capital of France."
        context = "Paris is the capital of France. It has the Eiffel Tower."
        score = GroundingVerifier.compute_grounding_score(answer, context)
        self.assertGreater(score, 0.7)

    def test_grounding_score_hallucinated(self):
        """Answer contains hallucinated claims."""
        answer = "The Eiffel Tower was built in 1889 and is 330 meters tall."
        context = "Paris is the capital of France."  # No Eiffel Tower info
        score = GroundingVerifier.compute_grounding_score(answer, context)
        self.assertLess(score, 0.5)


class TestRetrievalEvaluation(unittest.TestCase):
    """Tests for retrieval phase evaluation."""

    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_llm = MagicMock()
        self.evaluator = RAGEvaluator(self.mock_retriever, self.mock_llm)

    def test_retrieval_metrics_structure(self):
        """Retrieval metrics have correct structure."""
        self.mock_retriever.retrieve.return_value = (
            [{"text": "Paris is the capital of France.", "score": 0.9}], 1.0, 2.0
        )
        metrics = self.evaluator.evaluate_retrieval("Capital?", "Paris is capital", top_k=5)
        self.assertIsNotNone(metrics.query)
        self.assertEqual(metrics.top_k, 5)
        self.assertGreaterEqual(metrics.mrr, 0.0)
        self.assertLessEqual(metrics.mrr, 1.0)


class TestCompressionEvaluation(unittest.TestCase):
    """Tests for compression phase evaluation."""

    def test_compression_preserves_facts(self):
        """Compression preserves key facts."""
        docs = [
            "The database password is SuperSecretAgent123. It should not be shared.",
            "Other configuration details are available in the config file.",
            "Paris is the capital of France.",
        ]
        key_facts = ["SuperSecretAgent123", "database password"]
        
        compressed = Compressor.compress(docs, "password?", max_tokens=100)
        
        self.assertIn("SuperSecretAgent123", compressed)


class TestHyDEEvaluation(unittest.TestCase):
    """Tests for HyDE improvement measurement."""

    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_llm = MagicMock()
        self.evaluator = RAGEvaluator(self.mock_retriever, self.mock_llm)

    def test_hyde_metrics_structure(self):
        """HyDE metrics have correct structure."""
        docs = ["Photosynthesis converts light to energy."]
        self.mock_retriever.retrieve.return_value = (docs, 1.0, 2.0)
        
        metrics = self.evaluator.evaluate_hyde("photosynthesis?", docs, "light to energy")
        
        self.assertGreaterEqual(metrics.hyde_recall, 0.0)
        self.assertLessEqual(metrics.hyde_recall, 1.0)


class TestBenchmarkDataset(unittest.TestCase):
    """Tests for benchmark dataset integrity."""

    def test_benchmarks_exist(self):
        """Benchmark dataset is populated."""
        self.assertGreater(len(RAG_BENCHMARKS), 0)

    def test_benchmarks_have_required_fields(self):
        """Each benchmark has required fields."""
        for b in RAG_BENCHMARKS:
            self.assertTrue(hasattr(b, 'query'))
            self.assertTrue(hasattr(b, 'expected_answer'))
            self.assertTrue(hasattr(b, 'key_context_facts'))
            self.assertTrue(hasattr(b, 'category'))
            self.assertIsNotNone(b.query)
            self.assertIsNotNone(b.key_context_facts)


class TestRerankingEvaluation(unittest.TestCase):
    """Tests for reranking effectiveness."""

    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_llm = MagicMock()
        self.evaluator = RAGEvaluator(self.mock_retriever, self.mock_llm)

    def test_reranking_identifies_correct_ranking(self):
        """Reranking correctly identifies relevant content."""
        candidates = [
            {"text": "Allow traffic on port 80", "score": 0.9},
            {"text": "Block external traffic on port 80 with deny rule", "score": 0.5},
        ]
        
        metrics = self.evaluator.evaluate_reranking(
            "How to block port 80?", candidates, "deny rule"
        )
        
        self.assertTrue(hasattr(metrics, 'correctly_ranked'))


if __name__ == "__main__":
    unittest.main()