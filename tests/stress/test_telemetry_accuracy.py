import pytest
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.core.services.telemetry_service import TelemetryService


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