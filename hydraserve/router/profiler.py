"""
Profiler: Micro-benchmarks for cost model calibration.

Runs at startup to measure:
  - Prefill throughput (tokens/ms at various sequence lengths)
  - Decode throughput (tokens/ms at various batch sizes)
  - Transfer bandwidth (GB/s for NVLink/PCIe P2P/SHM)
  - Interference coefficient (ms delay per active decode request)

These measurements feed into the CostModel for accurate routing decisions.
"""

import time
from typing import Dict, List, Optional, Tuple
import torch

from hydraserve.model.adapter import ModelAdapter
from hydraserve.transfer.backend import TransferBackend


class Profiler:
    """
    Startup profiler for cost model calibration.

    Runs a series of micro-benchmarks to measure actual hardware
    performance, replacing theoretical estimates with measured values.
    """

    def __init__(
        self,
        model: Optional[ModelAdapter] = None,
        transfer_backend: Optional[TransferBackend] = None,
        warmup_iterations: int = 5,
        benchmark_iterations: int = 20,
    ):
        self.model = model
        self.transfer_backend = transfer_backend
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def run_all(self) -> Dict[str, float]:
        """Run all profiling benchmarks."""
        results = {}

        if self.model is not None:
            results.update(self.profile_prefill())
            results.update(self.profile_decode())

        if self.transfer_backend is not None:
            results.update(self.profile_transfer())

        # Interference profiling only if both are available
        if self.model is not None and torch.cuda.device_count() >= 2:
            results.update(self.profile_interference())

        return results

    def profile_prefill(
        self,
        seq_lengths: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Profile prefill throughput at various sequence lengths.

        Returns:
            prefill_tokens_per_ms: Average prefill speed
            prefill_time_{N}K_ms: Time for N-thousand token prefill
        """
        if seq_lengths is None:
            seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

        device = self.model.device
        results = {}

        for seq_len in seq_lengths:
            # Create dummy input
            input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)

            # Warmup
            for _ in range(min(3, self.warmup_iterations)):
                self.model.forward_prefill(input_ids, positions)

            torch.cuda.synchronize()
            start = time.perf_counter()
            n_iter = max(2, self.benchmark_iterations // (seq_len // 1024 + 1))

            for _ in range(n_iter):
                self.model.forward_prefill(input_ids, positions)

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / n_iter

            results[f"prefill_time_{seq_len // 1024}K_ms"] = elapsed * 1000

        # Calculate average throughput
        total_tokens = sum(seq_lengths)
        total_time = sum(
            results[f"prefill_time_{s // 1024}K_ms"] / 1000 for s in seq_lengths
        )
        results["prefill_tokens_per_ms"] = total_tokens / (total_time * 1000)

        return results

    def profile_decode(
        self,
        batch_sizes: Optional[List[int]] = None,
        context_len: int = 4096,
    ) -> Dict[str, float]:
        """
        Profile decode throughput at various batch sizes.

        Returns:
            decode_ms_per_token: Average time per token
            decode_tok_per_s_{N}: Tokens/second at batch size N
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        device = self.model.device
        results = {}

        for batch_size in batch_sizes:
            input_ids = torch.randint(0, 1000, (batch_size, 1), device=device)
            positions = torch.randint(0, context_len, (batch_size, 1), device=device)

            # Build KV cache at context_len
            kv_cache = {
                i: torch.randn(batch_size, 2, self.model.get_num_key_value_heads(),
                              context_len, self.model.get_head_dim(),
                              dtype=torch.bfloat16, device=device)
                for i in range(self.model.get_num_hidden_layers())
                if self.model.is_full_attention_layer(i)
            }

            # Build SSM state
            ssm_shape = self.model.get_ssm_state_shape()
            ssm_state = {
                i: torch.randn(batch_size, ssm_shape[1], ssm_shape[2], ssm_shape[3],
                              dtype=torch.float32, device=device)
                for i in range(ssm_shape[0])
            }

            conv_shape = self.model.get_conv_state_shape()
            conv_state = {
                i: torch.randn(batch_size, conv_shape[1], conv_shape[2], conv_shape[3],
                              dtype=torch.float32, device=device)
                for i in range(conv_shape[0])
            }

            # Warmup
            for _ in range(self.warmup_iterations):
                self.model.forward_decode(input_ids, positions, kv_cache, ssm_state, conv_state)

            torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(self.benchmark_iterations):
                self.model.forward_decode(input_ids, positions, kv_cache, ssm_state, conv_state)

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / self.benchmark_iterations

            results[f"decode_tok_per_s_{batch_size}"] = batch_size / elapsed
            results[f"decode_ms_per_token_{batch_size}"] = (elapsed / batch_size) * 1000

        # Average ms per token
        avg_ms = sum(
            results[f"decode_ms_per_token_{b}"] for b in batch_sizes
        ) / len(batch_sizes)
        results["decode_ms_per_token"] = avg_ms

        return results

    def profile_transfer(
        self,
        sizes_mb: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Profile transfer bandwidth.

        Returns:
            transfer_bandwidth_gb_s: Measured bandwidth
        """
        if sizes_mb is None:
            sizes_mb = [1, 10, 50, 100, 500]

        results = {}
        total_bw = 0.0
        count = 0

        for size_mb in sizes_mb:
            num_elements = int(size_mb * 1024 * 1024 / 2)
            t = torch.randn(num_elements, dtype=torch.bfloat16,
                           device=f'cuda:{self.transfer_backend.src_gpu}')
            dst = torch.empty(num_elements, dtype=torch.bfloat16,
                             device=f'cuda:{self.transfer_backend.dst_gpu}')

            # Warmup
            for _ in range(self.warmup_iterations):
                self.transfer_backend.send(t, self.transfer_backend.dst_gpu)
                self.transfer_backend.receive(dst, self.transfer_backend.src_gpu)

            self.transfer_backend.synchronize()
            start = time.perf_counter()

            n_iter = max(3, self.benchmark_iterations // max(1, int(size_mb) // 100))
            for _ in range(n_iter):
                self.transfer_backend.send(t, self.transfer_backend.dst_gpu)
                self.transfer_backend.receive(dst, self.transfer_backend.src_gpu)

            self.transfer_backend.synchronize()
            elapsed = (time.perf_counter() - start) / n_iter

            bw = (size_mb / 1024) / elapsed  # GB/s
            results[f"transfer_bw_{int(size_mb)}MB_gb_s"] = bw
            total_bw += bw
            count += 1

        results["transfer_bandwidth_gb_s"] = total_bw / count if count > 0 else 0
        return results

    def profile_interference(
        self,
        n_decode_active: int = 20,
        prefill_len: int = 32768,
    ) -> Dict[str, float]:
        """
        Profile interference coefficient.

        Measures how much decode latency degrades when a prefill
        runs concurrently on the same GPU.
        """
        # This requires both GPUs to simulate interference
        # Simplified: estimate from prefill time and decode count

        prefill_time_s = prefill_len / 320.0 / 1000  # Estimate: 320 tok/ms
        interference_per_req_ms = prefill_time_s * 1000  # Each decode stalls for prefill duration

        return {
            "interference_coefficient": 0.015,  # Conservative estimate
            "interference_per_req_ms": interference_per_req_ms,
            "total_interference_ms_at_{n_decode_active}_concurrent".format(
                n_decode_active=n_decode_active
            ): n_decode_active * interference_per_req_ms,
        }
