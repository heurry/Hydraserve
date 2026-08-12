"""
CentralScheduler: Orchestrates prefill and decode engines.

Responsibilities:
  - Request routing (collocated vs PD separated)
  - Transfer coordination between prefill and decode
  - Load balancing across decode GPUs (for 1P+ND configurations)
  - Multi-decode routing (for 4-card: 1P+3D, 2P+2D)

Request state machine:
    WAITING → PREFILL_RUNNING → PREFILL_TRANSFER_PENDING → READY → RUNNING → FINISHED
                                                                          ↓
                                                                    PREEMPTED → READY
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import torch

from hydraserve.config import HydraServeConfig, RequestState, RouterDecision
from hydraserve.model.adapter import ModelAdapter
from hydraserve.engine.prefill_engine import PrefillEngine
from hydraserve.engine.decode_engine import DecodeEngine
from hydraserve.router.adaptive_router import AdaptiveRouter


class CentralScheduler:
    """
    Central request scheduler.

    Runs on CPU, coordinates prefill and decode engines.
    Implements the request lifecycle state machine.
    """

    def __init__(
        self,
        config: HydraServeConfig,
        model: ModelAdapter,
        prefill_engine: PrefillEngine,
        decode_engine: DecodeEngine,
        router: Optional[AdaptiveRouter] = None,
    ):
        self.config = config
        self.model = model
        self.prefill_engine = prefill_engine
        self.decode_engine = decode_engine
        self.router = router

        # Request tracking
        self._request_id_counter = 0
        self._lock = threading.Lock()

        # Pending transfers: request_id → (first_token, descriptor)
        self.pending_transfers: Dict[int, Tuple[Optional[int], Any]] = {}

        # Stats
        self.total_requests = 0
        self.collocated_count = 0
        self.pd_count = 0

    # ─── Main API ───────────────────────────────────────────────

    def submit_request(
        self,
        input_ids: List[int],
        sampling_params: Optional[Dict] = None,
        max_new_tokens: int = 256,
    ) -> int:
        """
        Submit a new request.

        Returns:
            request_id: Unique identifier for this request
        """
        with self._lock:
            request_id = self._request_id_counter
            self._request_id_counter += 1

        self.total_requests += 1

        # Route decision
        decision = self._route_request(input_ids, max_new_tokens)
        prompt_len = len(input_ids)

        input_tensor = torch.tensor([input_ids],
                                    dtype=torch.long,
                                    device=f'cuda:{self.config.prefill_gpu}')

        if decision == RouterDecision.COLLOCATED:
            self.collocated_count += 1
            self._handle_collocated(request_id, input_tensor, sampling_params, max_new_tokens)
        else:
            self.pd_count += 1
            self._handle_pd_separated(request_id, input_tensor, sampling_params, max_new_tokens)

        return request_id

    def poll_output(self, request_id: int) -> Optional[Dict]:
        """
        Poll for output from a request.

        Returns:
            Dict with:
              - tokens: list of generated tokens
              - is_finished: bool
              - finish_reason: str or None
              - first_token_time_ms: float or None
        """
        req = self.decode_engine.get_output(request_id)
        if req is None:
            return None

        return {
            "tokens": list(req.output_tokens),
            "is_finished": req.state == RequestState.FINISHED,
            "finish_reason": req.finish_reason,
            "first_token_time_ms": (
                (req.first_token_time - req.arrival_time) * 1000
                if req.first_token_time else None
            ),
            "generated_len": req.generated_len,
        }

    def cancel_request(self, request_id: int) -> None:
        """Cancel an active request."""
        self.decode_engine.cancel_request(request_id)
        self.pending_transfers.pop(request_id, None)

    def get_stats(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "collocated_ratio": self.collocated_count / max(1, self.total_requests),
            "pd_ratio": self.pd_count / max(1, self.total_requests),
            "pending_transfers": len(self.pending_transfers),
            "prefill": self.prefill_engine.get_stats(),
            "decode": self.decode_engine.get_stats(),
        }

    # ─── Internal: Routing ──────────────────────────────────────

    def _route_request(self, input_ids: List[int], max_new_tokens: int) -> RouterDecision:
        """Decide whether to route as collocated or PD separated."""
        if self.router is not None:
            return self.router.route(
                prompt_len=len(input_ids),
                num_decode_running=len(self.decode_engine.running),
                max_decode_capacity=self.config.cache.max_num_seqs,
                input_ids=input_ids,
            )

        # Simple heuristic fallback
        prompt_len = len(input_ids)
        if prompt_len < self.config.router.prompt_short_threshold:
            return RouterDecision.COLLOCATED
        elif prompt_len >= self.config.router.prompt_long_threshold:
            return RouterDecision.PD_DISAGGREGATED
        else:
            # Medium: check decode load
            decode_load = (len(self.decode_engine.running) /
                           max(1, self.config.cache.max_num_seqs))
            if decode_load < self.config.router.decode_load_threshold:
                return RouterDecision.PD_DISAGGREGATED
            return RouterDecision.COLLOCATED

    # ─── Internal: Request Handling ─────────────────────────────

    def _handle_collocated(
        self,
        request_id: int,
        input_ids: torch.Tensor,
        sampling_params: Optional[Dict],
        max_new_tokens: int,
    ) -> None:
        """
        Handle a collocated request (prefill + decode on same GPU).

        For collocated mode, prefill runs and decode immediately follows
        on the same GPU without state transfer.
        """
        # Run prefill
        logits = self.prefill_engine.prefill_collocated(request_id, input_ids)

        # The rest is handled by the decode engine on the same GPU
        # In a real implementation, this would use chunked prefill
        # integrated with decode continuous batching

    def _handle_pd_separated(
        self,
        request_id: int,
        input_ids: torch.Tensor,
        sampling_params: Optional[Dict],
        max_new_tokens: int,
    ) -> None:
        """
        Handle a PD-separated request.

        1. Run prefill on GPU 0
        2. Generate StateTransferDescriptor
        3. Initiate transfer to GPU 1
        4. Decode engine receives state and starts generation
        """
        # Register with decode engine (waiting state)
        self.decode_engine.add_request(
            request_id,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
        )

        # Run prefill with state extraction
        prompt_len = input_ids.shape[1]

        if prompt_len > self.config.scheduler.chunked_prefill_size:
            first_token, descriptor = self.prefill_engine.chunked_prefill(
                request_id=request_id,
                input_ids=input_ids,
                chunk_size=self.config.scheduler.chunked_prefill_size,
                sampling_params=sampling_params,
            )
        else:
            first_token, descriptor = self.prefill_engine.prefill(
                request_id=request_id,
                input_ids=input_ids,
                sampling_params=sampling_params,
            )

        self.pending_transfers[request_id] = (first_token, descriptor)

        # Transfer completion triggers decode engine state reception
        # In async mode, this is handled by the transfer pipeline callback
        self._complete_transfer(request_id, first_token, descriptor)

    def _complete_transfer(
        self,
        request_id: int,
        first_token: Optional[int],
        descriptor: Any,
    ) -> None:
        """
        Called when transfer completes (or immediately for sync transfer).

        Hands off the transferred state to the decode engine.
        """
        # The transfer pipeline has already moved the data to decode GPU
        # Now we need to register it with the decode engine
        # In a full implementation, the transfer pipeline calls back here

        # For now, the prefill_engine's _initiate_transfer handles the
        # transfer synchronously, so state is already on decode GPU
        pass

    # ─── Multi-Decode Load Balancing ────────────────────────────

    def _select_decode_gpu(self) -> int:
        """
        Select which decode GPU to route to (for 1P+ND configurations).

        Strategies:
          - Round-robin: simplest
          - Least-loaded: pick GPU with fewest active requests
          - Affinity: same user → same GPU (better prefix cache hit rate)
        """
        # For 2-GPU setup, always use GPU 1
        return self.config.decode_gpu


