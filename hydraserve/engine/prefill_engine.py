"""
Prefill Engine (GPU 0).

Processes incoming prompts, extracts dual-state (KV Cache + recurrent states),
and initiates async transfer to the decode GPU.

Responsibilities:
  - Chunked prefill for long prompts
  - Layer-level state extraction during forward pass
  - First-token seeding optimization
  - Generate StateTransferDescriptor for the transfer pipeline

Request flow:
  1. Receive tokenized prompt
  2. Run chunked prefill (forward pass with state accumulation)
  3. Sample first token (optional, for seeding)
  4. Build StateTransferDescriptor
  5. Submit to transfer pipeline
"""

import time
from typing import List, Dict, Optional, Tuple, Any
import torch

from hydraserve.model.adapter import ModelAdapter
from hydraserve.config import HydraServeConfig, RequestState
from hydraserve.cache.block_manager import BlockManager, BlockTable
from hydraserve.cache.state_pool import StatePool
from hydraserve.transfer.backend import TransferBackend
from hydraserve.transfer.descriptor import (
    StateTransferDescriptor, RegionDescriptor, RegionType
)
from hydraserve.transfer.pipeline import TransferPipeline


class PrefillEngine:
    """
    Prefill engine running on GPU 0.

    Manages the prefill phase: token processing, state extraction,
    and transfer initiation.

    For PD separation mode:
    - Runs chunked prefill on GPU 0
    - Extracts KV Cache and recurrent states per layer
    - Initiates async transfer to GPU 1
    - Optionally samples first token for seeding

    For collocated mode:
    - Runs full forward pass (prefill + decode) on same GPU
    """

    def __init__(
        self,
        model: ModelAdapter,
        config: HydraServeConfig,
        block_manager: BlockManager,
        state_pool: StatePool,
        transfer_backend: Optional[TransferBackend] = None,
    ):
        self.model = model
        self.config = config
        self.block_manager = block_manager
        self.state_pool = state_pool
        self.transfer_backend = transfer_backend

        if transfer_backend is not None:
            self.transfer_pipeline = TransferPipeline(
                transfer_backend,
                use_quantization=(transfer_backend.transfer_mode.value != "full")
            )
        else:
            self.transfer_pipeline = None

        self.gpu_id = config.prefill_gpu
        self.device = torch.device(f"cuda:{self.gpu_id}")

        # Metrics
        self.total_prefills = 0
        self.total_prefill_time_ms = 0.0

    # ─── Main Entry Point ───────────────────────────────────────

    @torch.no_grad()
    def prefill(
        self,
        request_id: int,
        input_ids: torch.Tensor,        # [1, prompt_len]
        positions: Optional[torch.Tensor] = None,
        sampling_params: Optional[Dict] = None,
    ) -> Tuple[Optional[int], StateTransferDescriptor]:
        """
        Run prefill for a single request.

        Args:
            request_id: Unique request identifier
            input_ids: Tokenized prompt [1, prompt_len]
            positions: Position indices (auto-generated if None)
            sampling_params: Dict with temperature, top_p, top_k

        Returns:
            (first_token_id, transfer_descriptor):
              first_token_id: Sampled first token (None if seeding disabled)
              transfer_descriptor: Complete state transfer description
        """
        prompt_len = input_ids.shape[1]
        start_time = time.perf_counter()

        if positions is None:
            positions = torch.arange(prompt_len, device=self.device).unsqueeze(0)

        if sampling_params is None:
            sampling_params = {"temperature": 1.0, "top_p": 0.9, "top_k": 50}

        # Allocate state slots
        ssm_slot = self.state_pool.allocate(request_id)

        # Create block table for KV cache
        max_blocks = (prompt_len + self.block_manager.block_size - 1) // self.block_manager.block_size
        block_table = self.block_manager.create_block_table(request_id, max_blocks)

        # Allocate KV blocks
        block_ids = self.block_manager.allocate_blocks(max_blocks)
        for bid in block_ids:
            block_table.append(bid)

        # Run prefill forward
        prefill_result = self.model.forward_prefill(
            input_ids=input_ids,
            positions=positions,
            kv_cache=None,  # Fresh cache
            ssm_state=None,  # Fresh state
            conv_state=None,
        )

        logits = prefill_result["logits"]       # [1, prompt_len, vocab_size]
        kv_cache = prefill_result["kv_cache"]   # {layer_idx: tensor}
        ssm_state = prefill_result["ssm_state"]  # {linear_layer_idx: tensor}
        conv_state = prefill_result["conv_state"]

        # Write KV cache to blocks
        self._write_kv_to_blocks(kv_cache, block_ids, block_table)

        # Write SSM/conv states to state pool
        self._write_states_to_pool(ssm_slot, ssm_state, conv_state)

        # First-token seeding (optional)
        first_token_id = None
        first_token_logprob = None
        if self.config.transfer.first_token_seeding:
            last_logits = logits[:, -1, :]  # [1, vocab_size]
            first_token_id, entropy = self.model.sample_logits(
                last_logits,
                temperature=sampling_params.get("temperature", 1.0),
                top_p=sampling_params.get("top_p", 0.9),
                top_k=sampling_params.get("top_k", 50),
            )
            first_token_id = first_token_id.item()

        # Build transfer descriptor
        descriptor = self._build_transfer_descriptor(
            request_id=request_id,
            prompt_len=prompt_len,
            kv_cache=kv_cache,
            ssm_state=ssm_state,
            conv_state=conv_state,
            first_token_id=first_token_id,
            first_token_logprob=first_token_logprob,
        )

        # Initiate transfer if in PD mode
        if self.transfer_pipeline is not None:
            self._initiate_transfer(descriptor, kv_cache, ssm_state, conv_state)

        self.total_prefills += 1
        elapsed = (time.perf_counter() - start_time) * 1000
        self.total_prefill_time_ms += elapsed

        return first_token_id, descriptor

    # ─── Chunked Prefill ────────────────────────────────────────

    @torch.no_grad()
    def chunked_prefill(
        self,
        request_id: int,
        input_ids: torch.Tensor,        # [1, total_prompt_len]
        chunk_size: int = 4096,
        sampling_params: Optional[Dict] = None,
    ) -> Tuple[Optional[int], StateTransferDescriptor]:
        """
        Chunked prefill for long prompts.

        Splits prompt into chunks, processes each chunk sequentially,
        accumulating KV Cache and recurrent states across chunks.

        This allows:
        - Interleaving multiple requests on the prefill GPU
        - Lower prefill latency for short requests mixed with long ones
        - Better GPU utilization
        """
        total_len = input_ids.shape[1]
        num_chunks = (total_len + chunk_size - 1) // chunk_size

        if num_chunks <= 1:
            return self.prefill(request_id, input_ids, sampling_params=sampling_params)

        if sampling_params is None:
            sampling_params = {"temperature": 1.0, "top_p": 0.9, "top_k": 50}

        start_time = time.perf_counter()

        # Allocate state slots
        ssm_slot = self.state_pool.allocate(request_id)

        # Allocate blocks
        max_blocks = (total_len + self.block_manager.block_size - 1) // self.block_manager.block_size
        block_table = self.block_manager.create_block_table(request_id, max_blocks)
        block_ids = self.block_manager.allocate_blocks(max_blocks)
        for bid in block_ids:
            block_table.append(bid)

        # Accumulate states across chunks
        acc_kv_cache: Dict[int, torch.Tensor] = {}
        acc_ssm_state: Dict[int, torch.Tensor] = {}
        acc_conv_state: Dict[int, torch.Tensor] = {}

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_len)
            chunk_input = input_ids[:, chunk_start:chunk_end]
            chunk_positions = torch.arange(chunk_start, chunk_end,
                                           device=self.device).unsqueeze(0)

            result = self.model.forward_prefill(
                input_ids=chunk_input,
                positions=chunk_positions,
                kv_cache=acc_kv_cache if acc_kv_cache else None,
                ssm_state=acc_ssm_state if acc_ssm_state else None,
                conv_state=acc_conv_state if acc_conv_state else None,
            )

            # Merge chunk results
            for layer_idx, kv in result["kv_cache"].items():
                if layer_idx in acc_kv_cache:
                    acc_kv_cache[layer_idx] = torch.cat([
                        acc_kv_cache[layer_idx], kv], dim=-2)
                else:
                    acc_kv_cache[layer_idx] = kv

            acc_ssm_state = result["ssm_state"]
            acc_conv_state = result["conv_state"]

            # Check for new requests between chunks (scheduler hook)
            # self._check_new_requests()

        # First-token seeding
        first_token_id = None
        first_token_logprob = None
        if self.config.transfer.first_token_seeding:
            last_logits = result["logits"][:, -1, :]
            first_token_id, _ = self.model.sample_logits(
                last_logits,
                temperature=sampling_params.get("temperature", 1.0),
                top_p=sampling_params.get("top_p", 0.9),
                top_k=sampling_params.get("top_k", 50),
            )
            first_token_id = first_token_id.item()

        # Write to blocks and state pool
        self._write_kv_to_blocks(acc_kv_cache, block_ids, block_table)
        self._write_states_to_pool(ssm_slot, acc_ssm_state, acc_conv_state)

        # Build descriptor
        descriptor = self._build_transfer_descriptor(
            request_id=request_id,
            prompt_len=total_len,
            kv_cache=acc_kv_cache,
            ssm_state=acc_ssm_state,
            conv_state=acc_conv_state,
            first_token_id=first_token_id,
            first_token_logprob=first_token_logprob,
        )

        self.total_prefills += 1
        elapsed = (time.perf_counter() - start_time) * 1000
        self.total_prefill_time_ms += elapsed

        return first_token_id, descriptor

    # ─── Internal: KV Cache Block Writing ───────────────────────

    def _write_kv_to_blocks(
        self,
        kv_cache: Dict[int, torch.Tensor],
        block_ids: List[int],
        block_table: BlockTable,
    ) -> None:
        """Write extracted KV cache tensors to PagedAttention blocks."""
        block_size = self.block_manager.block_size

        for layer_idx, kv_tensor in kv_cache.items():
            # kv_tensor: [batch=1, 2, seq_len, n_kv_heads, head_dim]
            seq_len = kv_tensor.shape[2]
            k = kv_tensor[0, 0]  # [seq_len, n_kv_heads, head_dim]
            v = kv_tensor[0, 1]

            # Write to blocks
            for i, start in enumerate(range(0, seq_len, block_size)):
                end = min(start + block_size, seq_len)
                n_tokens = end - start
                block_id = block_ids[i]

                k_block = k[start:end]  # [n_tokens, n_kv_heads, head_dim]
                v_block = v[start:end]

                self.block_manager.write_kv_block(
                    block_id, layer_idx, k_block, v_block, n_tokens
                )

    def _write_states_to_pool(
        self,
        slot_id: int,
        ssm_state: Dict[int, torch.Tensor],
        conv_state: Dict[int, torch.Tensor],
    ) -> None:
        """Write linear attention states to the state pool."""
        # Convert dict-of-tensors to stacked tensors
        n_linear = self.model.get_num_linear_attn_layers()
        ssm_shape = self.model.get_ssm_state_shape()  # (n_lin_layers, ...)
        conv_shape = self.model.get_conv_state_shape()

        ssm_buffer = torch.zeros(ssm_shape, dtype=torch.float32, device=self.device)
        conv_buffer = torch.zeros(conv_shape, dtype=torch.float32, device=self.device)

        for lin_idx, tensor in ssm_state.items():
            ssm_buffer[lin_idx] = tensor[0].to(torch.float32)  # Remove batch dim

        for lin_idx, tensor in conv_state.items():
            conv_buffer[lin_idx] = tensor[0].to(torch.float32)

        self.state_pool.write_state(slot_id, ssm_buffer, conv_buffer)

    # ─── Internal: Transfer ─────────────────────────────────────

    def _build_transfer_descriptor(
        self,
        request_id: int,
        prompt_len: int,
        kv_cache: Dict[int, torch.Tensor],
        ssm_state: Dict[int, torch.Tensor],
        conv_state: Dict[int, torch.Tensor],
        first_token_id: Optional[int] = None,
        first_token_logprob: Optional[float] = None,
    ) -> StateTransferDescriptor:
        """Build a StateTransferDescriptor from prefill results."""
        descriptor = StateTransferDescriptor(
            request_id=request_id,
            model_name=self.config.model_name,
            first_token_id=first_token_id,
            first_token_logprob=first_token_logprob,
        )

        model_spec = self.config.model_spec
        num_full_attn = model_spec.num_full_attn_layers

        # Full attention KV regions
        if kv_cache:
            full_attn_indices = [
                i for i in range(model_spec.num_hidden_layers)
                if (i + 1) % model_spec.full_attention_interval == 0
            ]

            # Split KV cache into regions (one per full attn layer, or grouped)
            for layer_idx in full_attn_indices:
                if layer_idx in kv_cache:
                    kv_tensor = kv_cache[layer_idx]
                    region = RegionDescriptor(
                        region_type=RegionType.FULL_ATTN_KV,
                        layer_indices=[layer_idx],
                        shape=tuple(kv_tensor.shape),
                        dtype="bfloat16",
                        quantized=(self.transfer_pipeline is not None and
                                   self.transfer_pipeline.use_quantization),
                        src_gpu=self.config.prefill_gpu,
                        dst_gpu=self.config.decode_gpu,
                        size_bytes=kv_tensor.numel() * kv_tensor.element_size(),
                    )
                    descriptor.regions.append(region)

        # Linear attention SSM regions
        if ssm_state:
            lin_indices = list(ssm_state.keys())
            sample_tensor = ssm_state[lin_indices[0]]
            region = RegionDescriptor(
                region_type=RegionType.LINEAR_SSM,
                layer_indices=lin_indices,
                shape=(len(lin_indices),) + tuple(sample_tensor.shape[1:]),
                dtype="float32",
                quantized=False,  # Never quantize recurrent state
                src_gpu=self.config.prefill_gpu,
                dst_gpu=self.config.decode_gpu,
            )
            descriptor.regions.append(region)

        # Conv state region
        if conv_state:
            conv_indices = list(conv_state.keys())
            sample_tensor = conv_state[conv_indices[0]]
            region = RegionDescriptor(
                region_type=RegionType.LINEAR_CONV,
                layer_indices=conv_indices,
                shape=(len(conv_indices),) + tuple(sample_tensor.shape[1:]),
                dtype="float32",
                quantized=False,
                src_gpu=self.config.prefill_gpu,
                dst_gpu=self.config.decode_gpu,
            )
            descriptor.regions.append(region)

        descriptor.metadata["prompt_len"] = prompt_len
        return descriptor

    def _initiate_transfer(
        self,
        descriptor: StateTransferDescriptor,
        kv_cache: Dict[int, torch.Tensor],
        ssm_state: Dict[int, torch.Tensor],
        conv_state: Dict[int, torch.Tensor],
    ) -> None:
        """Initiate async transfer of all state to decode GPU."""
        # Prepare state tensors dict
        state_tensors = {}
        for region in descriptor.regions:
            key = f"{region.region_type.value}_{region.layer_indices[0]}"
            if region.region_type == RegionType.FULL_ATTN_KV:
                state_tensors[key] = kv_cache.get(region.layer_indices[0])
            elif region.region_type == RegionType.LINEAR_SSM:
                state_tensors[key] = ssm_state.get(region.layer_indices[0])
            elif region.region_type == RegionType.LINEAR_CONV:
                state_tensors[key] = conv_state.get(region.layer_indices[0])

        # Execute transfer
        self.transfer_pipeline.transfer_full_descriptor(descriptor, state_tensors)

    # ─── Collocated Mode ───────────────────────────────────────

    @torch.no_grad()
    def prefill_collocated(
        self,
        request_id: int,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Collocated prefill (no transfer). Returns logits for immediate decode.
        """
        if positions is None:
            positions = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)

        result = self.model.forward_prefill(
            input_ids=input_ids,
            positions=positions,
        )

        return result["logits"]

    # ─── Statistics ─────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "total_prefills": self.total_prefills,
            "avg_prefill_time_ms": (self.total_prefill_time_ms / self.total_prefills
                                     if self.total_prefills > 0 else 0),
            "gpu_id": self.gpu_id,
            "gpu_memory_used_gb": torch.cuda.memory_allocated(self.device) / 1e9,
        }
