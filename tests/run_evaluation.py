import unittest
import sys
import os
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evaluator import RAGEvaluator, GroundingVerifier
from core.benchmarks import get_all_benchmarks, RAG_BENCHMARKS


def run_groundedness_tests():
    """Run grounding verification tests."""
    print("\n" + "="*60)
    print("GROUNDING VERIFICATION TESTS")
    print("="*60)

    # Test 1: Perfect grounding
    print("\n[TEST] Perfect grounding - answer in context")
    answer = "Paris is the capital of France."
    context = "Paris is the capital of France. It has the Eiffel Tower."
    score = GroundingVerifier.compute_grounding_score(answer, context)
    print(f"  Answer: {answer}")
    print(f"  Context: {context[:50]}...")
    print(f"  Grounding score: {score}")
    assert score > 0.7, f"Expected high grounding, got {score}"
    print("  [PASS]")

    # Test 2: Hallucinated claims
    print("\n[TEST] Hallucinated claims detection")
    answer = "The Eiffel Tower was built in 1889 and is 330 meters tall."
    context = "Paris is the capital of France."
    score = GroundingVerifier.compute_grounding_score(answer, context)
    print(f"  Answer: {answer}")
    print(f"  Context: {context}")
    print(f"  Grounding score: {score}")
    assert score < 0.5, f"Expected low grounding, got {score}"
    print("  [PASS]")

    # Test 3: Citation extraction
    print("\n[TEST] Citation extraction")
    answer = "The password is SuperSecretAgent123 [source: config.txt]"
    citations = GroundingVerifier.extract_citation_markers(answer)
    print(f"  Answer: {answer}")
    print(f"  Citations found: {len(citations)}")
    assert len(citations) > 0, "Expected to find citations"
    print("  [PASS]")

    return True


def run_retrieval_evaluation():
    """Run retrieval quality evaluation."""
    print("\n" + "="*60)
    print("RETRIEVAL QUALITY EVALUATION")
    print("="*60)

    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    evaluator = RAGEvaluator(mock_retriever, mock_llm)

    # Mock retrieval results
    mock_retriever.retrieve.return_value = [
        {"text": "Paris is the capital of France.", "score": 0.95},
        {"text": "The Eiffel Tower is in Paris.", "score": 0.85},
    ]

    metrics = evaluator.evaluate_retrieval("capital", "Paris capital France", top_k=5)

    print(f"\n  Query: {metrics.query}")
    print(f"  Candidates found: {metrics.candidates_found}")
    print(f"  MRR: {metrics.mrr:.3f}")
    print(f"  Has relevant: {metrics.has_relevant}")

    return metrics


def run_compression_evaluation():
    """Run compression fact preservation evaluation."""
    print("\n" + "="*60)
    print("COMPRESSION EVALUATION")
    print("="*60)

    docs = [
        "The database password is SuperSecretAgent123. Keep it secret.",
        "Other configuration options are available in settings.",
        "This is unrelated noise content about weather.",
        "Paris is the capital of France.",
    ]
    key_facts = ["SuperSecretAgent123", "database password"]

    from core.compressor import Compressor
    compressed = Compressor.compress(docs, "password?", max_tokens=100)
    facts_preserved = sum(1 for f in key_facts if f.lower() in compressed.lower())
    total_raw_chars = sum(len(d) for d in docs)
    compressed_chars = len(compressed)
    compression_ratio = 1 - (compressed_chars / total_raw_chars)

    print(f"\n  Raw content: {len(docs)} docs, {total_raw_chars} chars")
    print(f"  Compressed: {len(compressed)} chars")
    print(f"  Compression ratio: {compression_ratio:.2%}")
    print(f"  Facts preserved: {facts_preserved}/{len(key_facts)}")
    print(f"  Key facts found in compressed: {all(f.lower() in compressed.lower() for f in key_facts)}")

    return {"compression_ratio": compression_ratio, "facts_preserved": facts_preserved / len(key_facts)}


def run_benchmark_evaluation():
    """Run benchmark queries evaluation."""
    print("\n" + "="*60)
    print("BENCHMARK EVALUATION")
    print("="*60)

    benchmarks = get_all_benchmarks()

    for b in benchmarks[:3]:
        print(f"\n  Query: {b.query}")
        print(f"  Category: {b.category}")
        print(f"  Expected: {b.expected_answer}")

    return True


def main():
    print("\n" + "="*60)
    print("RAG EVALUATION FRAMEWORK - FULL RUN")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)

    try:
        run_groundedness_tests()
        run_retrieval_evaluation()
        run_compression_evaluation()
        run_benchmark_evaluation()

        print("\n" + "="*60)
        print("ALL EVALUATION TESTS COMPLETED")
        print("="*60)
        print("\nSummary:")
        print("  - Grounding verification: OK")
        print("  - Retrieval evaluation: OK")
        print("  - Compression evaluation: OK")
        print("  - Benchmark dataset: OK")
        return 0
    except Exception as e:
        print(f"\n[X] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())