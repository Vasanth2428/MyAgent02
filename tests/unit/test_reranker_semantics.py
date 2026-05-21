import unittest
from core.reranker import NeuralReranker

class TestRerankerSemantics(unittest.TestCase):
    def test_semantic_precision_over_keywords(self):
        """
        Simulate an agent searching for specific instructions.
        The reranker must distinguish between sentences with identical keywords but opposite meanings.
        """
        reranker = NeuralReranker()
        
        query = "How to block external traffic on port 80?"
        
        candidates = [
            {"text": "To allow external traffic on port 80, add an accept rule to the firewall.", "score": 0.9},
            {"text": "To block external traffic on port 80, add a deny rule to the firewall.", "score": 0.8},
            {"text": "Traffic on port 80 is often external.", "score": 0.7}
        ]
        
        # Rerank
        results = reranker.rerank(query, candidates)
        
        # Inference / Assertions
        # Even though "allow" candidate had a higher initial retrieval score (0.9), 
        # the neural reranker should realize it has the opposite meaning of the query and push "block" to the top.
        top_result = results[0]["text"]
        
        self.assertIn("deny rule", top_result, "Reranker failed semantic precision test. It selected keyword overlap instead of meaning.")
        
        print("Reranker Semantic Precision Test:")
        for r in results:
            print(f"  [{r['cross_score']:.3f}] {r['text'][:50]}...")

if __name__ == "__main__":
    unittest.main()
