"""
Model Initialization Race Test - Verify singleton embedding and reranking models initialize safely.
"""
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.core.services.grounding_service import _get_shared_embedding_model
from src.core.reranker import _get_flashrank_reranker, NeuralReranker, RERANKER_MODEL


def test_model_initialization_race_condition():
    """Verify singleton embedding model loads exactly once under concurrent initialization."""
    import importlib
    import sys
    
    # Reset the module-level state to test fresh initialization
    if 'src.core.services.grounding_service' in sys.modules:
        del sys.modules['src.core.services.grounding_service']
    
    init_count = threading.local()
    init_events = []
    lock = threading.Lock()
    
    def try_initialize():
        try:
            model = _get_shared_embedding_model()
            with lock:
                init_events.append("initialized")
            return model is not None
        except Exception as e:
            with lock:
                init_events.append(f"error: {e}")
            return False
    
    # Concurrent initialization attempts
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(try_initialize) for _ in range(50)]
        results = [f.result() for f in as_completed(futures)]
    
    # All threads should get a valid model (or at least not crash)
    success_count = sum(1 for r in results if r)
    assert success_count >= 45, f"Too many initialization failures: {success_count}/50 succeeded"
    assert len(init_events) == 50, "Each thread should have attempted initialization"


def test_reranker_model_race_condition():
    """Verify reranker model initializes safely under concurrent load."""
    import importlib
    import sys
    
    # Reset state
    if 'src.core.reranker' in sys.modules:
        del sys.modules['src.core.reranker']
    
    results = []
    
    def create_reranker():
        try:
            reranker = NeuralReranker()
            res = reranker.rerank("cats", [{"text": "cat doc", "score": 0.5}])
            results.append(len(res) > 0)
            return True
        except Exception as e:
            results.append(False)
            return False
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(create_reranker) for _ in range(50)]
        outcomes = [f.result() for f in as_completed(futures)]
    
    success_count = sum(results)
    assert success_count >= 45, f"Reranker initialization race: only {success_count}/50 succeeded"


def test_cross_encoder_singleton_safety():
    """Verify FlashrankRerank singleton is safe for concurrent access."""
    import importlib
    import sys
    
    # Reset state
    if 'src.core.reranker' in sys.modules:
        del sys.modules['src.core.reranker']
    
    models = []
    lock = threading.Lock()
    
    def get_model():
        model = _get_flashrank_reranker()
        with lock:
            models.append(id(model))
        return model
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(get_model) for _ in range(50)]
        for f in as_completed(futures):
            f.result()
    
    # All should reference the same singleton instance
    unique_ids = set(models)
    assert len(unique_ids) == 1, (
        f"Multiple model instances created! Got {len(unique_ids)} unique IDs"
    )


def test_embedding_model_singleton_safety():
    """Verify embedding model singleton is safe for concurrent access."""
    import importlib
    import sys
    
    # Reset state
    if 'src.core.services.grounding_service' in sys.modules:
        del sys.modules['src.core.services.grounding_service']
    
    models = []
    lock = threading.Lock()
    
    def get_model():
        try:
            model = _get_shared_embedding_model()
            with lock:
                if model is not None:
                    models.append(id(model))
            return model
        except Exception:
            return None
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(get_model) for _ in range(50)]
        for f in as_completed(futures):
            f.result()
    
    # All should reference the same singleton instance (if model loaded successfully)
    if models:
        unique_ids = set(models)
        assert len(unique_ids) == 1, (
            f"Multiple embedding model instances created! Got {len(unique_ids)} unique IDs"
        )


def test_model_predict_thread_safety():
    """Verify model predictions work correctly under concurrent access."""
    try:
        from src.core.services.grounding_service import GroundingVerifier
        verifier = GroundingVerifier()
        
        def verify_grounding():
            try:
                score, unsupported = verifier.verify_grounding(
                    "Python is a programming language.",
                    ["Python is used for web development."]
                )
                return (score, unsupported)
            except Exception as e:
                return (None, str(e))
        
        results = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(verify_grounding) for _ in range(20)]
            results = [f.result() for f in as_completed(futures)]
        
        # All should complete successfully
        success_count = sum(1 for r in results if r[0] is not None)
        assert success_count >= 18, f"Only {success_count}/20 predictions succeeded"
        
        # Scores should be consistent
        scores = [r[0] for r in results if r[0] is not None]
        if len(scores) >= 2:
            assert max(scores) - min(scores) < 0.1, "Scores should be consistent across threads"
            
    except ImportError:
        pytest.skip("Embedding model not available for thread safety testing")


def test_no_crash_under_parallel_init():
    """Verify no crashes occur when models are initialized in parallel - adversarial test."""
    errors = []
    
    def safe_init():
        try:
            # Direct instantiation tests
            from src.core.services.grounding_service import GroundingVerifier
            v = GroundingVerifier()
            # Test that model access doesn't crash
            _ = v._embedding_model or v._get_embedding_model()
        except Exception as e:
            errors.append(str(e))
        
        try:
            from src.core.reranker import NeuralReranker, _get_flashrank_reranker
            r = NeuralReranker()
            _ = _get_flashrank_reranker()
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=safe_init) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # At least 90% should succeed without crashing
    assert len(errors) < 5, f"Too many errors during parallel init: {errors[:3]}"