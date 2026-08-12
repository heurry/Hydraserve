"""
Benchmark metrics collection and analysis.

Tracks:
  - TTFT (Time to First Token): latency from request to first generated token
  - TPOT (Time Per Output Token): average time between consecutive tokens
  - Throughput: tokens per second, requests per second
  - P50/P90/P99 latency percentiles
  - GPU utilization and memory usage
"""

import time
import statistics
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import threading


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: int
    prompt_len: int
    arrival_time: float
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    token_times: List[float] = field(default_factory=list)  # Time of each token generation
    num_generated_tokens: int = 0
    is_finished: bool = False
    routing_decision: str = ""
    transfer_time_ms: Optional[float] = None

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.arrival_time) * 1000

    @property
    def tpot_ms(self) -> Optional[float]:
        """Average time per output token in milliseconds."""
        if len(self.token_times) < 2:
            return None
        intervals = [self.token_times[i] - self.token_times[i - 1]
                     for i in range(1, len(self.token_times))]
        return statistics.mean(intervals) * 1000 if intervals else None

    @property
    def total_latency_ms(self) -> Optional[float]:
        """Total request latency in milliseconds."""
        if self.completion_time is None:
            return None
        return (self.completion_time - self.arrival_time) * 1000

    @property
    def tokens_per_second(self) -> float:
        """Generation throughput for this request."""
        if not self.token_times or self.first_token_time is None:
            return 0.0
        gen_time = (self.token_times[-1] - self.token_times[0]) if len(self.token_times) > 1 else 0
        return self.num_generated_tokens / max(0.001, gen_time) if gen_time > 0 else 0.0


class BenchmarkCollector:
    """
    Collects and aggregates metrics across multiple requests.

    Thread-safe for concurrent request tracking.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.requests: Dict[int, RequestMetrics] = {}
        self.start_time: float = time.time()

    def record_arrival(self, request_id: int, prompt_len: int) -> None:
        with self._lock:
            self.requests[request_id] = RequestMetrics(
                request_id=request_id,
                prompt_len=prompt_len,
                arrival_time=time.time(),
            )

    def record_first_token(self, request_id: int) -> None:
        with self._lock:
            req = self.requests.get(request_id)
            if req:
                req.first_token_time = time.time()
                req.token_times.append(time.time())

    def record_token(self, request_id: int) -> None:
        with self._lock:
            req = self.requests.get(request_id)
            if req:
                req.token_times.append(time.time())
                req.num_generated_tokens += 1

    def record_completion(self, request_id: int) -> None:
        with self._lock:
            req = self.requests.get(request_id)
            if req:
                req.completion_time = time.time()
                req.is_finished = True

    def record_routing(self, request_id: int, decision: str) -> None:
        with self._lock:
            req = self.requests.get(request_id)
            if req:
                req.routing_decision = decision

    def record_transfer(self, request_id: int, transfer_ms: float) -> None:
        with self._lock:
            req = self.requests.get(request_id)
            if req:
                req.transfer_time_ms = transfer_ms

    def get_summary(self) -> Dict:
        """Compute aggregate statistics across all requests."""
        with self._lock:
            finished = [r for r in self.requests.values() if r.is_finished]
            all_reqs = list(self.requests.values())

            if not all_reqs:
                return {"error": "No requests recorded"}

            # TTFT
            ttfts = [r.ttft_ms for r in finished if r.ttft_ms is not None]
            # TPOT
            tpots = [r.tpot_ms for r in finished if r.tpot_ms is not None]
            # Total tokens
            total_tokens = sum(r.num_generated_tokens for r in finished)
            # Duration
            duration = time.time() - self.start_time

            return {
                "num_requests_total": len(all_reqs),
                "num_requests_finished": len(finished),
                "duration_seconds": duration,
                "total_generated_tokens": total_tokens,
                "throughput_tok_per_s": total_tokens / max(0.001, duration),

                "ttft": {
                    "p50_ms": _percentile(ttfts, 50),
                    "p90_ms": _percentile(ttfts, 90),
                    "p99_ms": _percentile(ttfts, 99),
                    "mean_ms": statistics.mean(ttfts) if ttfts else 0,
                    "min_ms": min(ttfts) if ttfts else 0,
                    "max_ms": max(ttfts) if ttfts else 0,
                },

                "tpot": {
                    "p50_ms": _percentile(tpots, 50),
                    "p90_ms": _percentile(tpots, 90),
                    "p99_ms": _percentile(tpots, 99),
                    "mean_ms": statistics.mean(tpots) if tpots else 0,
                },

                "routing": {
                    "collocated": sum(1 for r in all_reqs if r.routing_decision == "collocated"),
                    "pd_separated": sum(1 for r in all_reqs if r.routing_decision == "pd_disaggregated"),
                },

                "transfer": {
                    "mean_ms": statistics.mean(
                        [r.transfer_time_ms for r in finished if r.transfer_time_ms]
                    ) if finished else 0,
                },
            }

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self.start_time = time.time()


def _percentile(data: List[float], p: float) -> float:
    """Compute p-th percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    return sorted_data[f]
