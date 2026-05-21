import unittest
from core.compressor import Compressor
from core.engine import count_tokens

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
        
        # Combine all noise into a few large documents, with the relevant fact hidden inside
        docs = [
            "\n\n".join(noise[:5]),
            "\n\n".join(noise[5:10]),  # The password is here
            "\n\n".join(noise[10:])
        ]
        
        query = "What is the database password?"
        
        # Compress the context
        compressed_text = Compressor.compress(docs, query, max_tokens=100)
        
        # Inference / Assertions
        # 1. It must contain the answer
        self.assertIn("SuperSecretAgent123", compressed_text)
        
        # 2. It must discard irrelevant noise to protect the token budget
        # We check at least some of the noise is gone
        noise_dropped = ("Apollo 11" not in compressed_text) or ("Photosynthesis" not in compressed_text) or ("Eiffel Tower" not in compressed_text)
        self.assertTrue(noise_dropped, "Compressor failed to drop irrelevant noise.")
        
        # 3. Token count must be drastically reduced
        total_raw_tokens = sum(count_tokens(doc) for doc in docs)
        compressed_tokens = count_tokens(compressed_text)
        self.assertLessEqual(compressed_tokens, 100, "Failed to compress under the requested max_tokens budget.")
        print(f"Compressor Accuracy Test: Raw Tokens: {total_raw_tokens} -> Compressed Tokens: {compressed_tokens}")

if __name__ == "__main__":
    unittest.main()
