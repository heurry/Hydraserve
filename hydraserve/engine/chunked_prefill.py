"""
Chunked Prefill Scheduler (GPU 0).

Handles long prompts by splitting them into fixed-size chunks
and interleaving chunks from different requests for fairness.

Benefits:
  - Lower TTFT for short requests arriving during long prefill
  - Better GPU utilization (fill idle time between chunks)
  - Enables layer-level pipeline hiding

Without chunked prefill:
    [====== 32K prefill 100ms ======] [transfer 9ms] [decode starts]

With chunked prefill:
    [4K][4K][4K][4K][4K][4K][4K][4K]
         |    |    |    |    |    |    |    |
      [transfer L0-L3] ...                [done]
                                    [decode starts, transfer done]
"""

from typing import List, Dict, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from collections import deque
import torch

from hydraserve.model.adapter import ModelAdapter
from hydraserve.cache.block_manager import BlockManager
from hydraserve.cache.state_pool import StatePool


@dataclass
class ChunkedPrefillTask:
    """State for an in-progress chunked prefill."""
    request_id: int
    total_len: int
    chunks_processed: int
    total_chunks: int
    chunk_size: int
    next_token_pos: int

    # Accumulated state across chunks
    kv_cache: Dict[int, torch.Tensor] = field(default_factory=dict)
    ssm_state: Dict[int, torch.Tensor] = field(default_factory=dict)
    conv_state: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Input tensor (full prompt, processed in chunks)
    input_ids: Optional[torch.Tensor] = None
    sampling_params: Optional[Dict] = None

    # Block management
    block_ids: List[int] = field(default_factory=list)
    ssm_slot_id: int = -1

    @property
    def is_complete(self) -> bool:
        return self.chunks_processed >= self.total_chunks

    @property
    def remaining_tokens(self) -> int:
        return self.total_len - self.next_token_pos

    def get_next_chunk(self) -> torch.Tensor:
        """Get the next chunk of input tokens."""
        start = self.chunks_processed * self.chunk_size
        end = min(start + self.chunk_size, self.total_len)
        chunk = self.input_ids[:, start:end]
        self.chunks_processed += 1
        self.next_token_pos = end
        return chunk

    def get_next_positions(self) -> torch.Tensor:
        """Get position indices for the next chunk."""
        start = (self.chunks_processed - 1) * self.chunk_size
        end = min(start + self.chunk_size, self.total_len)
        return torch.arange(start, end, device=self.input_ids.device).unsqueeze(0)


class ChunkedPrefillScheduler:
    """
    Scheduler for chunked prefill.

    Maintains a queue of prefill tasks, processes one chunk at a time
    in round-robin fashion, allowing new requests to be interleaved.
    """

    def __init__(
        self,
        model: ModelAdapter,
        block_manager: BlockManager,
        state_pool: StatePool,
        chunk_size: int = 4096,
        max_chunks_per_step: int = 2,
    ):
        self.model = model
        self.block_manager = block_manager
        self.state_pool = state_pool
        self.chunk_size = chunk_size
        self.max_chunks_per_step = max_chunks_per_step

        # Task queue
        self.tasks: Deque[ChunkedPrefillTask] = deque()
        self.completed_tasks: Dict[int, ChunkedPrefillTask] = {}

    def submit(
        self,
        request_id: int,
        input_ids: torch.Tensor,
        sampling_params: Optional[Dict] = None,
    ) -> None:
        """
        Submit a new chunked prefill task.

        If prompt is shorter than chunk_size, it's processed immediately.
        Otherwise, it goes into the chunked queue.
        """
        total_len = input_ids.shape[1]

        if total_len <= self.chunk_size:
            # Short enough: don't chunk, just process
            self._process_immediate(request_id, input_ids, sampling_params)
            return

        total_chunks = (total_len + self.chunk_size - 1) // self.chunk_size
        ssm_slot = self.state_pool.allocate(request_id)

        max_blocks = (total_len + self.block_manager.block_size - 1) // self.block_manager.block_size
        block_ids = self.block_manager.allocate_blocks(max_blocks)

        task = ChunkedPrefillTask(
            request_id=request_id,
            total_len=total_len,
            chunks_processed=0,
            total_chunks=total_chunks,
            chunk_size=self.chunk_size,
            next_token_pos=0,
            input_ids=input_ids,
            sampling_params=sampling_params,
            block_ids=block_ids,
            ssm_slot_id=ssm_slot,
        )
        self.tasks.append(task)

    def process_step(self) -> List[int]:
        """
        Process one scheduling step.

        Picks up to max_chunks_per_step tasks and processes one chunk each.
        Returns list of completed request IDs.
        """
        if not self.tasks:
            return []

        completed = []
        chunks_this_step = 0

        for _ in range(min(len(self.tasks), self.max_chunks_per_step)):
            if not self.tasks:
                break

            task = self.tasks.popleft()

            # Process one chunk
            chunk = task.get_next_chunk()
            positions = task.get_next_positions()

            result = self.model.forward_prefill(
                input_ids=chunk,
                positions=positions,
                kv_cache=task.kv_cache if task.kv_cache else None,
                ssm_state=task.ssm_state if task.ssm_state else None,
                conv_state=task.conv_state if task.conv_state else None,
            )

            # Update accumulated state
            for layer_idx, kv in result["kv_cache"].items():
                if layer_idx in task.kv_cache:
                    task.kv_cache[layer_idx] = torch.cat([
                        task.kv_cache[layer_idx], kv], dim=-2)
                else:
                    task.kv_cache[layer_idx] = kv

            task.ssm_state = result["ssm_state"]
            task.conv_state = result["conv_state"]

            chunks_this_step += 1

            if task.is_complete:
                # Write KV to blocks
                self._write_kv_blocks(task.kv_cache, task.block_ids)
                # Write state to pool
                self._write_states(task.ssm_slot_id, task.ssm_state, task.conv_state)
                self.completed_tasks[task.request_id] = task
                completed.append(task.request_id)
            else:
                # Re-queue for next step
                self.tasks.append(task)

        return completed

    def get_completed_task(self, request_id: int) -> Optional[ChunkedPrefillTask]:
        """Get and remove a completed task."""
        return self.completed_tasks.pop(request_id, None)

    def cancel(self, request_id: int) -> None:
        """Cancel an in-progress task."""
        # Remove from queue
        self.tasks = deque([t for t in self.tasks if t.request_id != request_id])
        # Free resources
        self.block_manager.remove_block_table(request_id)
        self.state_pool.free(request_id)

    def get_queue_depth(self) -> int:
        return len(self.tasks)

    # ─── Internal ───────────────────────────────────────────────

    def _process_immediate(
        self,
        request_id: int,
        input_ids: torch.Tensor,
        sampling_params: Optional[Dict],
    ) -> None:
        """Process a short prompt without chunking."""
        task = ChunkedPrefillTask(
            request_id=request_id,
            total_len=input_ids.shape[1],
            chunks_processed=1,
            total_chunks=1,
            chunk_size=self.chunk_size,
            next_token_pos=input_ids.shape[1],
            input_ids=input_ids,
            sampling_params=sampling_params,
        )
        task.kv_cache = {}  # Will be filled by forward
        task.ssm_state = {}
        task.conv_state = {}
        self.completed_tasks[request_id] = task

    def _write_kv_blocks(self, kv_cache: Dict[int, torch.Tensor], block_ids: List[int]) -> None:
        """Write KV cache to blocks."""
        block_size = self.block_manager.block_size
        for layer_idx, kv_tensor in kv_cache.items():
            seq_len = kv_tensor.shape[2] if kv_tensor.dim() >= 3 else 0
            k = kv_tensor[0, 0] if kv_tensor.dim() >= 4 else kv_tensor[0]
            v = kv_tensor[0, 1] if kv_tensor.dim() >= 4 else kv_tensor[1]

            for i, start in enumerate(range(0, seq_len, block_size)):
                end = min(start + block_size, seq_len)
                if i < len(block_ids):
                    self.block_manager.write_kv_block(
                        block_ids[i], layer_idx,
                        k[start:end], v[start:end], end - start
                    )

    def _write_states(
        self,
        slot_id: int,
        ssm_state: Dict[int, torch.Tensor],
        conv_state: Dict[int, torch.Tensor],
    ) -> None:
        """Write linear attention states to state pool."""
        n_linear = self.model.get_num_linear_attn_layers()
        ssm_shape = self.model.get_ssm_state_shape()
        conv_shape = self.model.get_conv_state_shape()

        device = next(iter(ssm_state.values())).device
        ssm_buf = torch.zeros(ssm_shape, dtype=torch.float32, device=device)
        conv_buf = torch.zeros(conv_shape, dtype=torch.float32, device=device)

        for lin_idx, tensor in ssm_state.items():
            if tensor.dim() >= 4:
                ssm_buf[lin_idx] = tensor[0].to(torch.float32)
            else:
                ssm_buf[lin_idx] = tensor.to(torch.float32)

        for lin_idx, tensor in conv_state.items():
            if tensor.dim() >= 4:
                conv_buf[lin_idx] = tensor[0].to(torch.float32)
            else:
                conv_buf[lin_idx] = tensor.to(torch.float32)

        self.state_pool.write_state(slot_id, ssm_buf, conv_buf)
