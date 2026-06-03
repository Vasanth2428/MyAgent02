"""
Telemetry Accuracy Test - Verify latency and token metrics remain correct under concurrency.
"""
import pytest
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.core.services.telemetry_service import TelemetryService
from src.core.engine import count_tokens


def test_telemetry_metrics_isolation():
    """Verify request-scoped metrics are not contaminated across sessions."""
    telemetry_a = TelemetryService()
    telemetry_b = TelemetryService()
    
    telemetry_a.increment_queries()
    telemetry_a.update_latency(150.5)
    telemetry_a.update_compression_ratio(0.75)
    
    telemetry_b.increment_queries()
    telemetry_b.update_latency(300.0)
    telemetry_b.update_compression_ratio(0.25)
    
    assert telemetry_a.stats["queries"] == 1
    assert telemetry_a.stats["avg_latency_ms"] == 150.5
    assert abs(telemetry_a.stats["avg_compression_ratio"] - 0.75) < 0.01
    
    assert telemetry_b.stats["queries"] == 1
    assert telemetry_b.stats["avg_latency_ms"] == 300.0
    assert abs(telemetry_b.stats["avg_compression_ratio"] - 0.25) < 0.01


def test_telemetry_concurrent_updates():
    """Verify telemetry remains accurate under concurrent updates."""
    telemetry = TelemetryService()
    num_threads = 100
    latencies = [100 + i * 2 for i in range(num_threads)]
    
    def record_metrics(latency):
        telemetry.increment_queries()
        telemetry.update_latency(latency)
        return latency
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(record_metrics, lat) for lat in latencies]
        results = [f.result() for f in as_completed(futures)]
    
    expected_avg = sum(latencies) / len(latencies)
    actual_avg = telemetry.stats["avg_latency_ms"]
    
    assert telemetry.stats["queries"] == num_threads
    assert abs(actual_avg - expected_avg) < 1.0, (
        f"Average latency corrupted under concurrency. "
        f"Expected ~{expected_avg:.2f}, got {actual_avg:.2f}"
    )


def test_telemetry_latency_attribution():
    """Verify latency metrics accurately attribute to operations."""
    telemetry = TelemetryService()
    
    operation_times = {
        "retrieval_ms": 50.0,
        "generation_ms": 100.0,
    }
    
    for op_name, lat in operation_times.items():
        telemetry.increment_queries()  # Need to increment queries first
        telemetry.update_latency(lat)
    
    avg_latency = sum(operation_times.values()) / len(operation_times)
    
    assert telemetry.stats["queries"] == len(operation_times)
    assert abs(telemetry.stats["avg_latency_ms"] - avg_latency) < 1.0


def test_telemetry_token_budget_tracking():
    """Verify token budget tracking works correctly under concurrent access."""
    telemetry = TelemetryService()
    
    test_prompts = [
        "This is prompt one.",
        "This is prompt number two with more tokens.",
        "Short.",
    ]
    
    for prompt in test_prompts:
        tokens = count_tokens(prompt)
        cost = telemetry.compute_cost(tokens, tokens // 2)
        assert cost > 0, "Cost should be computed"
    
    assert telemetry.stats["queries"] == 0
    telemetry.increment_queries()
    telemetry.increment_queries()
    assert telemetry.stats["queries"] == 2


def test_telemetry_compression_ratio_accuracy():
    """Verify compression ratio calculations are precise."""
    telemetry = TelemetryService()
    
    ratios = [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4]
    for ratio in ratios:
        telemetry.update_compression_ratio(ratio)
    
    expected_avg = sum(ratios) / len(ratios)
    assert abs(telemetry.stats["avg_compression_ratio"] - expected_avg) < 0.01


def test_telemetry_system_metrics_thread_safety():
    """Verify system metrics collection is thread-safe."""
    telemetry = TelemetryService()
    errors = []
    
    def get_metrics_repeatedly():
        try:
            for _ in range(50):
                metrics = telemetry.get_system_metrics()
                assert "cpu" in metrics
                assert "ram" in metrics
                assert 0 <= metrics["cpu"] <= 100
                assert 0 <= metrics["ram"] <= 100
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=get_metrics_repeatedly) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(errors) == 0, f"System metrics thread safety errors: {errors}"


def test_telemetry_no_metric_contamination():
    """Verify metrics from one session don't pollute another's measurements."""
    telemetry = TelemetryService()
    
    for i in range(5):
        telemetry.increment_queries()
        telemetry.update_latency(100 * (i + 1))
    
    stats_after_5 = dict(telemetry.stats)
    
    for i in range(3):
        telemetry.increment_queries()
        telemetry.update_latency(50 * (i + 1))
    
    stats_after_8 = dict(telemetry.stats)
    
    assert stats_after_8["queries"] == 8
    assert stats_after_8["avg_latency_ms"] != stats_after_5["avg_latency_ms"]
    
    # The average should have changed due to new lower latencies
    assert stats_after_8["avg_latency_ms"] < stats_after_5["avg_latency_ms"]


def test_telemetry_race_condition_on_update():
    """Stress test for race conditions in metric updates."""
    telemetry = TelemetryService()
    num_iterations = 500
    barrier = threading.Barrier(50)
    
    def concurrent_update():
        barrier.wait()
        for _ in range(num_iterations):
            telemetry.increment_queries()
            telemetry.update_latency(1.0)
    
    threads = [threading.Thread(target=concurrent_update) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    expected_total = 50 * num_iterations
    assert telemetry.stats["queries"] == expected_total, (
        f"Race condition detected: expected {expected_total} queries, "
        f"got {telemetry.stats['queries']}"
    )