import logging
import psutil
from typing import Dict
from core.config import COST_PER_INPUT_TOKEN, COST_PER_OUTPUT_TOKEN

logger = logging.getLogger("RAG.Services.Telemetry")

class TelemetryService:
    """
    Manages search stats updates, latency aggregations, system metrics, and cost calculations.
    """
    def __init__(self):
        self.stats = {
            "queries": 0,
            "queries_compressed": 0,
            "avg_compression_ratio": 0.0,
            "avg_latency_ms": 0.0
        }

    def increment_queries(self):
        self.stats["queries"] += 1

    def update_compression_ratio(self, ratio: float):
        self.stats["queries_compressed"] += 1
        q_comp = self.stats["queries_compressed"]
        self.stats["avg_compression_ratio"] = (
            self.stats["avg_compression_ratio"] * (q_comp - 1) + ratio
        ) / q_comp

    def update_latency(self, total_ms: float):
        q = self.stats["queries"]
        if q > 0:
            self.stats["avg_latency_ms"] = (
                self.stats["avg_latency_ms"] * (q - 1) + total_ms
            ) / q

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * COST_PER_INPUT_TOKEN) + (completion_tokens * COST_PER_OUTPUT_TOKEN)

    def get_system_metrics(self) -> Dict[str, float]:
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
        except Exception:
            cpu = 0.0
            ram = 0.0
        return {"cpu": cpu, "ram": ram}
