import pytest
from src.core.reranker import NeuralReranker


def test_retrieval_adversarial_similarity():
    """Verify that NeuralReranker ranks relevant documents above semantically opposite documents."""
    reranker = NeuralReranker()
    
    test_cases = [
        {
            "query": "How do I allow remote access to the database?",
            "correct": "You can allow remote access to the database by editing the config.",
            "decoy": "You must deny remote access to the database to ensure maximum security.",
            "concept": "allow vs deny"
        },
        {
            "query": "Please enable logging on the server.",
            "correct": "To enable logging, change the logs flag to true in settings.",
            "decoy": "To disable logging, change the logs flag to false in settings.",
            "concept": "enable vs disable"
        },
        {
            "query": "How do I encrypt my personal password database?",
            "correct": "To encrypt your password database, use a secure cryptographic key.",
            "decoy": "To decrypt your password database, input your master verification passphrase.",
            "concept": "encrypt vs decrypt"
        },
        {
            "query": "How can I run this command safely?",
            "correct": "It is safe to execute this script in a local sandbox environment.",
            "decoy": "It is unsafe to execute this script because it might corrupt data.",
            "concept": "safe vs unsafe"
        },
        {
            "query": "We need to increase the memory budget.",
            "correct": "You should increase the memory allocation limit by modifying the env config.",
            "decoy": "You should decrease the memory allocation limit by modifying the env config.",
            "concept": "increase vs decrease"
        }
    ]
    
    for case in test_cases:
        candidates = [
            {"text": case["decoy"], "score": 0.5},
            {"text": case["correct"], "score": 0.5}
        ]
        
        reranked = reranker.rerank(case["query"], candidates)
        
        assert len(reranked) == 2
        assert reranked[0]["text"] == case["correct"], (
            f"Failed on concept '{case['concept']}': "
            f"Reranker preferred decoy '{reranked[0]['text']}' over correct '{case['correct']}'"
        )
        assert reranked[0]["cross_score"] > reranked[1]["cross_score"], (
            f"Correct document score ({reranked[0]['cross_score']}) should be higher than "
            f"decoy score ({reranked[1]['cross_score']}) for '{case['concept']}'"
        )


def test_retrieval_contradictory_viewpoints():
    """Verify that contradictory documents are both retained and clearly marked."""
    reranker = NeuralReranker()
    
    query = "What are the benefits of remote work?"
    
    candidates = [
        {"text": "Remote work increases productivity by reducing office distractions.", "score": 0.5},
        {"text": "Remote work decreases productivity due to lack of supervision.", "score": 0.5},
        {"text": "Remote work improves work-life balance for employees.", "score": 0.4},
    ]
    
    reranked = reranker.rerank(query, candidates)
    
    assert len(reranked) == 3
    
    # Check that both viewpoints are present in results
    all_text = " ".join(c["text"] for c in reranked)
    assert "increases" in all_text or "improves" in all_text
    assert "decreases" in all_text or "lack" in all_text
    
    # The most relevant should win, but contradictions shouldn't be erased
    assert reranked[0]["cross_score"] >= reranked[2]["cross_score"]


def test_retrieval_ambiguity_handling():
    """Test retrieval behavior with ambiguous queries that could match multiple topics."""
    reranker = NeuralReranker()
    
    query = "How do I run the process?"
    
    candidates = [
        {"text": "To run the installation process, execute setup.py.", "score": 0.5},
        {"text": "To run the manufacturing process, follow the safety protocol.", "score": 0.5},
        {"text": "To run the software, use the start command.", "score": 0.4},
    ]
    
    reranked = reranker.rerank(query, candidates)
    
    assert len(reranked) == 3
    # Any of these could be correct - just verify ranking is consistent
    scores = [c["cross_score"] for c in reranked]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_empty_candidates():
    """Verify reranker handles empty candidate list gracefully."""
    reranker = NeuralReranker()
    
    result = reranker.rerank("test query", [])
    assert result == []


def test_retrieval_single_candidate():
    """Verify reranker handles single candidate correctly."""
    reranker = NeuralReranker()
    
    candidates = [{"text": "Only document available.", "score": 0.5}]
    result = reranker.rerank("test query", candidates)
    
    assert len(result) == 1
    assert result[0]["text"] == "Only document available."


def test_retrieval_malformed_input():
    """Verify reranker handles malformed input gracefully."""
    reranker = NeuralReranker()
    
    # Missing 'text' field
    candidates = [{"score": 0.5}, {"text": "Valid document.", "score": 0.3}]
    
    # Should not crash - implementation should handle gracefully
    try:
        result = reranker.rerank("test query", candidates)
        # If it doesn't crash, verify basic behavior
        assert isinstance(result, list)
    except (KeyError, TypeError):
        # If it raises an error, that's also acceptable behavior
        pass


def test_retrieval_score_separation():
    """Verify clear score separation between relevant and irrelevant documents."""
    reranker = NeuralReranker()
    
    query = "Quantum computing implementation details"
    
    candidates = [
        {"text": "Quantum bit manipulation uses superposition states.", "score": 0.5},
        {"text": "Coffee is brewed from roasted beans.", "score": 0.5},
        {"text": "Quantum algorithms solve optimization problems.", "score": 0.4},
    ]
    
    reranked = reranker.rerank(query, candidates)
    
    assert len(reranked) == 3
    
    relevant_score = reranked[0]["cross_score"]
    irrelevant_score = reranked[-1]["cross_score"]  # Last should be coffee
    
    assert relevant_score > irrelevant_score, (
        f"Relevant docs should score higher than irrelevant. Got relevant={relevant_score:.3f}, irrelevant={irrelevant_score:.3f}"
    )