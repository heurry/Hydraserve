"""
Cost Model for Adaptive Routing.

Computes expected latency for collocated vs PD-separated paths
to make routing decisions.

Key equations:
    collocated_latency = prefill_time(prompt_len) + interference_penalty(n_decode)
    disaggregated_latency = prefill_time(prompt_len)
                           + max(0, transfer_time - prefill_time)
                           + decode_time

Decision matrix:
    | prompt < 2K                     | Collocated  (prefill too fast for transfer) |
    | prompt > 8K + decode has space  | PD separated (transfer hidden in prefill)   |
    | prompt > 8K + decode full       | Collocated  (transfer can't help)           |
    | prompt > 32K                    | PD separated (prefill long, transfer hidden) |
"""

import math
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from hydraserve.config import ModelSpec


@dataclass
class CostEstimate:
    """Latency estimate for a single routing path."""
    prefill_time_ms: float
    transfer_time_ms: float
    interference_penalty_ms: float
    total_latency_ms: float
    path: str  # "collocated" or "pd_disaggregated"


class CostModel:
    """
    Cost model for routing decisions.

    Uses profiled parameters from micro-benchmarks to estimate
    latency for each path. Parameters are measured at startup.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        prefill_tokens_per_ms: float = 320.0,    # Tokens/ms for prefill (32K in 100ms ≈ 320 tok/ms)
        decode_tokens_per_ms: float = 0.2,       # Tokens/ms per sequence (5ms/tok ≈ 0.2 tok/ms)
        transfer_bandwidth_gb_s: float = 112.0,   # GB/s for transfer
        transfer_overhead_ms: float = 0.01,       # Fixed transfer overhead
        interference_coefficient: float = 0.015,  # ms interference per decode request per ms prefill
    ):
        self.model_spec = model_spec
        self.prefill_tokens_per_ms = prefill_tokens_per_ms
        self.decode_tokens_per_ms = decode_tokens_per_ms
        self.transfer_bandwidth_gb_s = transfer_bandwidth_gb_s
        self.transfer_overhead_ms = transfer_overhead_ms
        self.interference_coefficient = interference_coefficient

        # Computed parameters
        self.transfer_size_per_token = model_spec.get_kv_cache_size_per_token()
        self.recurrent_state_size = (
            model_spec.get_ssm_state_size() + model_spec.get_conv_state_size()
        )

    def estimate_prefill_time(self, prompt_len: int) -> float:
        """Estimate prefill time in ms for given prompt length."""
        return prompt_len / self.prefill_tokens_per_ms

    def estimate_transfer_time(self, prompt_len: int) -> float:
        """Estimate state transfer time in ms."""
        total_bytes = (prompt_len * self.transfer_size_per_token +
                       self.recurrent_state_size)
        transfer_seconds = total_bytes / (self.transfer_bandwidth_gb_s * 1e9)
        return transfer_seconds * 1000 + self.transfer_overhead_ms

    def estimate_interference_penalty(self, n_decode_active: int, prefill_time_ms: float) -> float:
        """
        Estimate interference penalty from prefill on decode.

        When a prefill runs on the same GPU as decode, each active decode
        request experiences a stall proportional to the prefill duration.
        """
        return n_decode_active * self.interference_coefficient * prefill_time_ms

    def estimate_collocated_latency(
        self,
        prompt_len: int,
        n_decode_active: int,
        expected_output_tokens: int = 200,
    ) -> CostEstimate:
        """
        Estimate total latency for collocated (single GPU) path.

        Latency = prefill_time + interference_penalty + decode_time
        """
        prefill_ms = self.estimate_prefill_time(prompt_len)
        interference_ms = self.estimate_interference_penalty(n_decode_active, prefill_ms)
        decode_ms = expected_output_tokens / self.decode_tokens_per_ms

        return CostEstimate(
            prefill_time_ms=prefill_ms,
            transfer_time_ms=0.0,
            interference_penalty_ms=interference_ms,
            total_latency_ms=prefill_ms + interference_ms + decode_ms,
            path="collocated",
        )

    def estimate_pd_latency(
        self,
        prompt_len: int,
        expected_output_tokens: int = 200,
    ) -> CostEstimate:
        """
        Estimate total latency for PD-separated (two GPU) path.

        Latency = prefill_time + max(0, transfer_time - prefill_time) + decode_time
        Note: transfer overlaps with prefill, so only the uncovered portion counts.
        """
        prefill_ms = self.estimate_prefill_time(prompt_len)
        transfer_ms = self.estimate_transfer_time(prompt_len)

        # Transfer hidden by prefill: only count excess
        uncovered_transfer_ms = max(0, transfer_ms - prefill_ms)
        decode_ms = expected_output_tokens / self.decode_tokens_per_ms

        # No interference in PD mode (physical isolation)
        total_ms = prefill_ms + uncovered_transfer_ms + decode_ms

        return CostEstimate(
            prefill_time_ms=prefill_ms,
            transfer_time_ms=transfer_ms,
            interference_penalty_ms=0.0,
            total_latency_ms=total_ms,
            path="pd_disaggregated",
        )

    def compare(
        self,
        prompt_len: int,
        n_decode_active: int,
        expected_output_tokens: int = 200,
    ) -> Tuple[CostEstimate, CostEstimate, str]:
        """
        Compare collocated vs PD separated and return winner.

        Returns:
            (collocated_estimate, pd_estimate, winner_path)
        """
        collocated = self.estimate_collocated_latency(
            prompt_len, n_decode_active, expected_output_tokens
        )
        pd = self.estimate_pd_latency(prompt_len, expected_output_tokens)

        winner = "collocated" if collocated.total_latency_ms <= pd.total_latency_ms else "pd_disaggregated"
        return collocated, pd, winner

    def is_pd_beneficial(
        self,
        prompt_len: int,
        n_decode_active: int,
    ) -> bool:
        """Quick check: is PD separation beneficial for this request?"""
        _, _, winner = self.compare(prompt_len, n_decode_active)
        return winner == "pd_disaggregated"

    def update_parameters(self, profile_results: Dict[str, float]) -> None:
        """Update cost model parameters from profiling results."""
        if "prefill_tokens_per_ms" in profile_results:
            self.prefill_tokens_per_ms = profile_results["prefill_tokens_per_ms"]
        if "decode_ms_per_token" in profile_results:
            self.decode_tokens_per_ms = 1.0 / profile_results["decode_ms_per_token"]
        if "transfer_bandwidth_gb_s" in profile_results:
            self.transfer_bandwidth_gb_s = profile_results["transfer_bandwidth_gb_s"]
        if "interference_coefficient" in profile_results:
            self.interference_coefficient = profile_results["interference_coefficient"]
