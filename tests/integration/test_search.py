import unittest
import sys
import os
from dotenv import load_dotenv

# Ensure we can import from src.core
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.retriever import WeaviateRetriever

class TestWeaviateRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        cls.retriever = WeaviateRetriever()

    @classmethod
    def tearDownClass(cls):
        cls.retriever.close()

    def test_retrieval_returns_results(self):
        query = "operating system"
        results, embed_latency, search_latency = self.retriever.retrieve(query, top_k=2)
        
        self.assertGreater(len(results), 0, "Should find at least 1 document candidate")
        self.assertIsInstance(embed_latency, float, "Embedding latency should be float")
        self.assertIsInstance(search_latency, float, "Search latency should be float")
        for res in results:
            self.assertIn("text", res, "Result missing 'text' property")
            self.assertIn("score", res, "Result missing 'score' property")
            self.assertIn("tags", res, "Result missing 'tags' property")

if __name__ == "__main__":
    unittest.main()
