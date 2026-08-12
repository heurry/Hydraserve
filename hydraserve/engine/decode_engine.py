"""
Decode Engine (GPU 1).

Receives transferred dual-state from the prefill engine and runs
continuous batching decode. Manages the decode-side state:
  - KV Cache (via PagedAttention block manager)
  - Linear attention recurrent states (via state pool)
  - Prefix cache (radix tree)

Key features:
  - Continuous batching: add/remove requests dynamically
  - N-1 truncation: replay last prefill token to advance recurrent state
  - First-token seeding: skip replay if prefill already sampled first token
  - Preemption: LRU eviction when memory is tight
"""

import time
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import torch

from hydraserve.model.adapter import ModelAdapter
from hydraserve.config import HydraServeConfig, RequestState
from hydraserve.cache.block_manager import BlockManager, BlockTable
from hydraserve.cache.state_pool import StatePool
from hydraserve.cache.prefix_cache import PrefixCache
from hydraserve.transfer.descriptor import StateTransferDescriptor


@dataclass
class DecodeRequest:
    """State for a single request in the decode engine."""
    request_id: int
    state: RequestState = RequestState.WAITING
    prompt_len: int = 0
    generated_len: int = 0
    max_new_tokens: int = 256

    # State references
    block_table: Optional[BlockTable] = None
    state_slot_id: int = -1

    # First-token seeding
    first_token_id: Optional[int] = None
    first_token_output: bool = False

    # Generation
    next_token_id: Optional[int] = None
    output_tokens: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None

    # Timing
    arrival_time: float = 0.0
    first_token_time: Optional[float] = None
    last_activity_time: float = 0.0

    # Sampling params
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    stop_token_ids: Set[int] = field(default_factory=set)


class DecodeEngine:
    """
    Decode engine running on GPU 1.

    Runs continuous batching decode loop on the decode GPU.
    Receives state via transfer pipeline, manages dual-state memory,
    generates tokens, and returns outputs.
    """

    def __init__(
        self,
        model: ModelAdapter,
        config: HydraServeConfig,
        block_manager: BlockManager,
        state_pool: StatePool,
        prefix_cache: Optional[PrefixCache] = None,
    ):
        self.model = model
        self.config = config
        self.block_manager = block_manager
        self.state_pool = state_pool
        self.prefix_cache = prefix_cache

        self.gpu_id = config.decode_gpu
        self.device = torch.device(f"cuda:{self.gpu_id}")

        # Active requests
        self.requests: Dict[int, DecodeRequest] = {}
        self.running: List[int] = []       # Request IDs in RUNNING state
        self.waiting: List[int] = []       # Request IDs awaiting transfer

        # EOS token (model-specific)
        self.eos_token_id = 151645  # Qwen default; should be from model config

        # Metrics
        self.total_tokens_generated = 0
        self.total_decode_steps = 0
        self.total_decode_time_ms = 0.0

    # ─── State Reception ────────────────────────────────────────

    def receive_state(
        self,
        descriptor: StateTransferDescriptor,
        kv_cache: Dict[int, torch.Tensor],
        ssm_state: torch.Tensor,
        conv_state: torch.Tensor,
        request_id: int,
        sampling_params: Optional[Dict] = None,
    ) -> None:
        """
        Receive transferred state from prefill engine.

        This is called after the transfer pipeline completes.
        It performs:
        1. Register the request with its state
        2. N-1 truncation or first-token seeding
        3. Move the request to READY state
        """
        # Create decode request
        req = DecodeRequest(
            request_id=request_id,
            state=RequestState.PREFILL_TRANSFER_PENDING,
            prompt_len=descriptor.metadata.get("prompt_len", 0),
            arrival_time=time.time(),
            last_activity_time=time.time(),
        )

        if sampling_params:
            req.temperature = sampling_params.get("temperature", 1.0)
            req.top_p = sampling_params.get("top_p", 0.9)
            req.top_k = sampling_params.get("top_k", 50)
            stop_ids = sampling_params.get("stop_token_ids", [])
            req.stop_token_ids = set(stop_ids)

        # Allocate state on decode GPU
        ssm_slot = self.state_pool.allocate(request_id)
        self.state_pool.write_state(ssm_slot, ssm_state, conv_state)
        req.state_slot_id = ssm_slot

        # Allocate KV blocks (or use received block table)
        # In full implementation, block table is transferred alongside
        max_blocks = (req.prompt_len + self.block_manager.block_size - 1) // self.block_manager.block_size
        block_table = self.block_manager.create_block_table(request_id, max_blocks)
        block_ids = self.block_manager.allocate_blocks(max_blocks)
        for bid in block_ids:
            block_table.append(bid)
        req.block_table = block_table

        # Write received KV cache to blocks
        self._write_received_kv(kv_cache, block_ids, block_table)

        # N-1 truncation or first-token seeding
        if descriptor.first_token_id is not None:
            # First-token seeding: skip recomputation
            req.first_token_id = descriptor.first_token_id
            req.next_token_id = descriptor.first_token_id
            req.first_token_time = time.time()
        else:
            # Replay last prefill token (N-1 truncation)
            last_token_id = descriptor.metadata.get("last_prefill_token_id")
            last_token_pos = req.prompt_len - 1
            if last_token_id is not None:
                self._replay_last_token(request_id, last_token_id, last_token_pos)
                req.next_token_id = last_token_id

        # Now ready for decode
        req.state = RequestState.READY
        self.requests[request_id] = req
        self.waiting.append(request_id)

    # ─── Continuous Batching Decode Loop ────────────────────────

    @torch.no_grad()
    def decode_step(self) -> Dict[int, Tuple[int, bool, Optional[str]]]:
        """
        Run one decode step for all running requests.

        Returns:
            Dict mapping request_id → (token_id, is_finished, finish_reason)

        Continuous batching: all running requests are batched together
        in a single forward pass.
        """
        start_time = time.perf_counter()

        # Move waiting requests to running
        newly_ready = self.waiting[:]
        for rid in newly_ready:
            self.requests[rid].state = RequestState.RUNNING
            self.running.append(rid)
            self.waiting.remove(rid)

        if not self.running:
            return {}

        # Batch prefill for newly ready requests (single token to advance state)
        # and decode for existing running requests

        # Build batch
        batch_input_ids = []
        batch_positions = []
        batch_ssm_state = {}
        batch_conv_state = {}
        batch_kv_cache = {}
        batch_block_tables = {}

        for rid in self.running:
            req = self.requests[rid]
            next_token = req.next_token_id
            if next_token is None:
                continue

            batch_input_ids.append(next_token)
            seq_pos = req.prompt_len + req.generated_len
            batch_positions.append(seq_pos)

            # Read state from pool
            ssm, conv = self.state_pool.read_state(req.state_slot_id)
            batch_ssm_state[rid] = ssm
            batch_conv_state[rid] = conv

            # KV cache from block manager
            # (simplified: direct tensor access; in production, use block table)
            batch_kv_cache[rid] = req.block_table

        if not batch_input_ids:
            return {}

        # Convert to tensors
        input_ids_t = torch.tensor(batch_input_ids, device=self.device).unsqueeze(-1)  # [batch, 1]
        positions_t = torch.tensor(batch_positions, device=self.device).unsqueeze(-1)

        # Build batched KV cache (simplified: concatenate along batch dim)
        kv_cache_batched = {}
        for layer_idx in range(self.model.get_num_hidden_layers()):
            if self.model.is_full_attention_layer(layer_idx):
                # Gather KV from blocks
                pass  # Simplified - full implementation reads from block manager

        # Build batched SSM/conv state
        ssm_batched = torch.stack(list(batch_ssm_state.values()), dim=0)
        conv_batched = torch.stack(list(batch_conv_state.values()), dim=0)

        # Run decode forward
        result = self.model.forward_decode(
            input_ids=input_ids_t,
            positions=positions_t,
            kv_cache=kv_cache_batched,
            ssm_state=ssm_batched,
            conv_state=conv_batched,
            block_tables=batch_block_tables,
        )

        logits = result["logits"]  # [batch, 1, vocab_size]
        new_ssm = result["ssm_state"]
        new_conv = result["conv_state"]

        # Sample and update each request
        outputs = {}
        finished_rids = []

        for i, rid in enumerate(self.running):
            req = self.requests[rid]
            token_logits = logits[i, -1, :]  # [vocab_size]

            # First-token seeding: output pre-sampled token directly
            if req.first_token_id is not None and not req.first_token_output:
                token_id = req.first_token_id
                req.first_token_output = True
            else:
                token_id, _ = self.model.sample_logits(
                    token_logits.unsqueeze(0),
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                )
                token_id = token_id.item()

            # Update state
            req.output_tokens.append(token_id)
            req.generated_len += 1
            req.next_token_id = token_id
            req.last_activity_time = time.time()
            self.total_tokens_generated += 1

            # Write back SSM state
            self.state_pool.write_state(req.state_slot_id,
                                        new_ssm[i] if new_ssm.dim() == 4 else new_ssm,
                                        new_conv[i] if new_conv.dim() == 4 else new_conv)

            # Check stop conditions
            is_finished = False
            finish_reason = None

            if token_id == self.eos_token_id or token_id in req.stop_token_ids:
                is_finished = True
                finish_reason = "stop"
            elif req.generated_len >= req.max_new_tokens:
                is_finished = True
                finish_reason = "length"

            if is_finished:
                req.state = RequestState.FINISHED
                req.finish_reason = finish_reason
                finished_rids.append(rid)

            outputs[rid] = (token_id, is_finished, finish_reason)

        # Cleanup finished requests
        for rid in finished_rids:
            self.running.remove(rid)
            self._cleanup_request(rid)

        self.total_decode_steps += 1
        elapsed = (time.perf_counter() - start_time) * 1000
        self.total_decode_time_ms += elapsed

        return outputs

    # ─── Request Management ─────────────────────────────────────

    def add_request(
        self,
        request_id: int,
        max_new_tokens: int = 256,
        sampling_params: Optional[Dict] = None,
    ) -> None:
        """Register a new request (waits for state transfer)."""
        req = DecodeRequest(
            request_id=request_id,
            state=RequestState.WAITING,
            arrival_time=time.time(),
            last_activity_time=time.time(),
            max_new_tokens=max_new_tokens,
        )
        if sampling_params:
            req.temperature = sampling_params.get("temperature", 1.0)
            req.top_p = sampling_params.get("top_p", 0.9)
            req.top_k = sampling_params.get("top_k", 50)
        self.requests[request_id] = req

    def cancel_request(self, request_id: int) -> None:
        """Cancel a request."""
        if request_id in self.running:
            self.running.remove(request_id)
        if request_id in self.waiting:
            self.waiting.remove(request_id)
        self._cleanup_request(request_id)

    def preempt_request(self, request_id: int) -> None:
        """
        Preempt a request (LRU eviction).

        When memory is tight, evict the least recently used request.
        Its KV Cache may be quantized to INT4 for compressed storage,
        and its recurrent state is preserved for later restoration.
        """
        req = self.requests.get(request_id)
        if req is None or req.state not in (RequestState.RUNNING, RequestState.READY):
            return

        req.state = RequestState.PREEMPTED
        if request_id in self.running:
            self.running.remove(request_id)
        if request_id in self.waiting:
            self.waiting.remove(request_id)

        # State is preserved in state pool; KV blocks may be quantized
        # When restored, the request goes back to READY

    def restore_request(self, request_id: int) -> None:
        """Restore a preempted request."""
        req = self.requests.get(request_id)
        if req is None or req.state != RequestState.PREEMPTED:
            return
        req.state = RequestState.READY
        self.waiting.append(request_id)

    # ─── Streaming Output ───────────────────────────────────────

    def get_output(self, request_id: int) -> Optional[DecodeRequest]:
        """Get the current state of a request."""
        return self.requests.get(request_id)

    def is_finished(self, request_id: int) -> bool:
        req = self.requests.get(request_id)
        return req is not None and req.state == RequestState.FINISHED

    # ─── Internal ───────────────────────────────────────────────

    def _replay_last_token(
        self,
        request_id: int,
        token_id: int,
        position: int,
    ) -> None:
        """
        N-1 truncation: replay the last prefill token to advance
        the recurrent state boundary.

        The recurrent state from prefill encodes tokens 0..N-1.
        We need to process token N to advance the state to 0..N
        before regular decode can start from token N+1.

        Cost: < 5ms (single token forward pass).
        """
        req = self.requests[request_id]
        ssm, conv = self.state_pool.read_state(req.state_slot_id)

        result = self.model.forward_single_token(
            token_id=token_id,
            position=position,
            ssm_state=ssm,
            conv_state=conv,
        )

        # Update state
        self.state_pool.write_state(
            req.state_slot_id,
            result["ssm_state"],
            result["conv_state"],
        )

    def _write_received_kv(
        self,
        kv_cache: Dict[int, torch.Tensor],
        block_ids: List[int],
        block_table: BlockTable,
    ) -> None:
        """Write received KV cache to blocks on decode GPU."""
        block_size = self.block_manager.block_size

        for layer_idx, kv_tensor in kv_cache.items():
            if kv_tensor.dim() < 3:
                continue
            # Move to decode GPU
            kv_tensor = kv_tensor.to(self.device)

            seq_len = kv_tensor.shape[2] if kv_tensor.dim() >= 3 else kv_tensor.shape[1]
            k = kv_tensor[0, 0] if kv_tensor.dim() >= 4 else kv_tensor[0]
            v = kv_tensor[0, 1] if kv_tensor.dim() >= 4 else kv_tensor[1]

            for i, start in enumerate(range(0, seq_len, block_size)):
                end = min(start + block_size, seq_len)
                n_tokens = end - start
                if i < len(block_ids):
                    self.block_manager.write_kv_block(
                        block_ids[i], layer_idx,
                        k[start:end], v[start:end], n_tokens
                    )

    def _cleanup_request(self, request_id: int) -> None:
        """Free all resources for a finished request."""
        req = self.requests.pop(request_id, None)
        if req is None:
            return
        self.block_manager.remove_block_table(request_id)
        self.state_pool.free(request_id)

    # ─── Memory Management ──────────────────────────────────────

    def get_memory_pressure(self) -> float:
        """Return memory pressure ratio (0-1)."""
        used = self.block_manager.get_used_blocks()
        total = self.block_manager.max_blocks
        return used / total if total > 0 else 0.0

    def maybe_evict(self) -> None:
        """Evict LRU request if memory pressure is high."""
        if self.get_memory_pressure() < 0.9:
            return
        # Find least recently active request
        if not self.running:
            return
        lru_rid = min(self.running, key=lambda rid: self.requests[rid].last_activity_time)
        self.preempt_request(lru_rid)

    # ─── Statistics ─────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "total_requests": len(self.requests),
            "running": len(self.running),
            "waiting": len(self.waiting),
            "total_tokens_generated": self.total_tokens_generated,
            "total_decode_steps": self.total_decode_steps,
            "avg_decode_step_ms": (self.total_decode_time_ms / self.total_decode_steps
                                   if self.total_decode_steps > 0 else 0),
            "tokens_per_second": (self.total_tokens_generated /
                                  (self.total_decode_time_ms / 1000)
                                  if self.total_decode_time_ms > 0 else 0),
            "gpu_memory_used_gb": torch.cuda.memory_allocated(self.device) / 1e9,
        }
