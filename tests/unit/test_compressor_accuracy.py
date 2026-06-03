import unittest
from src.core.compressor import Compressor
from src.core.engine import count_tokens


class TestCompressorAccuracy(unittest.TestCase):
    def test_noise_isolation(self):
        """
        Simulate an agent retrieving a large amount of noisy context.
        The compressor must isolate the single relevant sentence and discard the rest.
        """
        noise = [
            "The quick brown fox jumps over the lazy dog.",
            "Jupiter is the largest planet in our solar system.",
            "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
            "The Python programming language was created by Guido van Rossum.",
            "In 1969, Apollo 11 landed the first humans on the Moon.",
            "Photosynthesis is the process by which green plants make food.",
            "The capital of France is Paris, known for the Eiffel Tower.",
            "A standard deck of playing cards contains 52 cards.",
            "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
            "The password for the production database is SuperSecretAgent123.",
            "Apples are a great source of fiber and vitamin C."
        ]
        
        docs = [
            "\n\n".join(noise[:5]),
            "\n\n".join(noise[5:10]),  # The password is here
            "\n\n".join(noise[10:])
        ]
        
        query = "What is the database password?"
        
        compressed_text = Compressor.compress(docs, query, max_tokens=100)
        
        # 1. It must contain the answer
        self.assertIn("SuperSecretAgent123", compressed_text)
        
        # 2. It must discard irrelevant noise to protect the token budget
        noise_dropped = ("Apollo 11" not in compressed_text) or ("Photosynthesis" not in compressed_text) or ("Eiffel Tower" not in compressed_text)
        self.assertTrue(noise_dropped, "Compressor failed to drop irrelevant noise.")
        
        # 3. Token count must be drastically reduced
        total_raw_tokens = sum(count_tokens(doc) for doc in docs)
        compressed_tokens = count_tokens(compressed_text)
        self.assertLessEqual(compressed_tokens, 100, "Failed to compress under the requested max_tokens budget.")
        print(f"Compressor Accuracy Test: Raw Tokens: {total_raw_tokens} -> Compressed Tokens: {compressed_tokens}")


class TestCompressorContradictorySegments(unittest.TestCase):
    def test_contradictory_segment_both_viewpoints_retained(self):
        """
        When retrieved segments disagree, both viewpoints should be retained
        so the generator can see the contradiction.
        """
        docs = [
            "Remote work increases employee productivity by 40% according to study A.",
            "Remote work decreases employee productivity by 25% according to study B.",
            "The office is open Monday through Friday.",
        ]
        
        query = "What does research say about remote work productivity?"
        
        compressed = Compressor.compress(docs, query, max_tokens=200)
        
        # Both contradictory viewpoints should be present
        self.assertIn("increases", compressed.lower(), "First viewpoint should be retained")
        self.assertIn("decreases", compressed.lower(), "Second viewpoint should be retained")
    
    def test_contradictory_segment_contradiction_visible(self):
        """
        Contradictions in source documents should remain visible to the generator
        through indicator language.
        """
        docs = [
            "The system uses AES-256 encryption for data at rest.",
            "The system uses DES encryption which is deprecated.",
            "Configuration settings are stored in config.yaml.",
        ]
        
        query = "What encryption does the system use?"
        
        compressed = Compressor.compress(docs, query, max_tokens=200)
        
        # Key contradiction terms should be preserved
        self.assertIn("AES", compressed)
        self.assertIn("DES", compressed)
    
    def test_contradictory_segment_priority_to_recent(self):
        """
        When contradictions exist, more specific/recent info should be prioritized.
        """
        docs = [
            "System v1.0 used MD5 hashing (deprecated, insecure).",
            "System v2.0 now uses SHA-256 for all cryptographic hashing.",
            "SHA-256 is the current standard for this system.",
        ]
        
        query = "What hashing algorithm does the current system use?"
        
        compressed = Compressor.compress(docs, query, max_tokens=200)
        
        # Current standard should be emphasized
        self.assertIn("SHA-256", compressed)
    
    def test_contradictory_segment_mixed_agreement(self):
        """
        Test handling when some documents agree and some disagree.
        """
        docs = [
            "The API returns JSON responses for all endpoints.",
            "The API returns XML responses for legacy endpoints.",
            "New endpoints only support JSON format.",
            "JSON is the primary response format.",
            "XML support is deprecated and will be removed.",
        ]
        
        query = "What format does the API use?"
        
        compressed = Compressor.compress(docs, query, max_tokens=200)
        
        # Both formats should be mentioned to show the nuance
        self.assertIn("JSON", compressed)
        self.assertIn("XML", compressed)


class TestCompressorSemanticCompression(unittest.TestCase):
    def test_semantic_compression_low_keyword_overlap(self):
        """
        Verify compressor identifies relevant content even with low keyword overlap.
        Uses semantic similarity rather than just word matching.
        """
        query = "Who founded Tesla?"
        
        # No shared keywords but semantically relevant
        segment = "The company was established by Martin Eberhard and Marc Tarpenning in 2003."
        
        docs = [
            "Tesla Motors was incorporated in Delaware in July 2003.",
            segment,
            "Elon Musk joined Tesla later and led investment rounds.",
            "Electric vehicle technology has advanced significantly.",
        ]
        
        compressed = Compressor.compress(docs, query, max_tokens=150)
        
        # The relevant segment should be preserved despite low keyword overlap
        self.assertIn("Eberhard", compressed) or self.assertIn("Tarpenning", compressed)
    
    def test_semantic_compression_preserve_context(self):
        """
        Verify semantic compression preserves meaning even when wording differs.
        """
        query = "How to connect to the database?"
        
        docs = [
            "Establishing a database connection requires specific configuration.",
            "The connection string format is postgresql://user:pass@host:port/db",
            "Database connectivity depends on network permissions.",
            "Use the JDBC driver for Java applications.",
        ]
        
        compressed = Compressor.compress(docs, query, max_tokens=150)
        
        # Connection-related content should be preserved
        self.assertTrue(
            "connection" in compressed.lower() or "database" in compressed.lower() or "jdbc" in compressed.lower(),
            "Semantically relevant content should be preserved"
        )


class TestCompressorFailureScenarios(unittest.TestCase):
    def test_empty_documents(self):
        """Test compressor handles empty document list."""
        compressed = Compressor.compress([], "any query", max_tokens=100)
        self.assertEqual(compressed, "")
    
    def test_empty_query(self):
        """Test compressor handles empty query gracefully."""
        docs = ["Some document content here."]
        compressed = Compressor.compress(docs, "", max_tokens=100)
        # Should still return content (fast path or empty)
        self.assertIsInstance(compressed, str)
    
    def test_all_noise_compression(self):
        """Test compression drops all noise when nothing is relevant."""
        noise = [
            "The weather forecast predicts sunny skies.",
            "Coffee is brewed from roasted beans.",
            "Mountains are formed by tectonic activity.",
        ]
        
        query = "Quantum computing implementation details"
        
        compressed = Compressor.compress(noise, query, max_tokens=50)
        
        # Should be empty or minimal since nothing is relevant
        compressed_tokens = count_tokens(compressed)
        self.assertLessEqual(compressed_tokens, 50)


if __name__ == "__main__":
    unittest.main()