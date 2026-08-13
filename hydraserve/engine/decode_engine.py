from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from hydraserve.cache.block_manager import BlockAllocation, KVBlockManager
from hydraserve.cache.kv_quantizer import Int4Tensor
from hydraserve.cache.state_pool import LinearStatePool
from hydraserve.engine.scheduler import Request, RequestState
from hydraserve.transfer.descriptor import StateTransferDescriptor, TransferMode
from hydraserve.transfer.pipeline import HybridStateBundle, TransferPipeline


KVRecompute = Callable[[tuple[int, ...]], np.ndarray]


@dataclass(slots=True)
class InstalledRequest:
    descriptor: StateTransferDescriptor
    kv_allocation: BlockAllocation
    kv_cache: np.ndarray | Int4Tensor
    seeded_token_id: int | None


class DecodeEngine:
    def __init__(
        self,
        pipeline: TransferPipeline,
        block_manager: KVBlockManager,
        state_pool: LinearStatePool,
        kv_recompute: KVRecompute | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.block_manager = block_manager
        self.state_pool = state_pool
        self.kv_recompute = kv_recompute
        self._installed: dict[int, InstalledRequest] = {}

    def receive_and_install(self, request: Request, timeout: float | None = None) -> InstalledRequest:
        if request.state is not RequestState.TRANSFER_PENDING:
            raise RuntimeError("request is not waiting for state transfer")
        descriptor, bundle = self.pipeline.receive(request.request_id, timeout=timeout)
        kv_cache = self._materialize_kv(request, descriptor, bundle)

        slot_allocated = False
        blocks_allocated = False
        try:
            self.state_pool.allocate(request.request_id)
            slot_allocated = True
            self.state_pool.set(request.request_id, bundle.recurrent)
            allocation = self.block_manager.allocate(request.request_id, descriptor.prompt_length)
            blocks_allocated = True
        except Exception:
            if blocks_allocated:
                self.block_manager.free(request.request_id)
            if slot_allocated:
                self.state_pool.free(request.request_id)
            request.transition(RequestState.FAILED)
            raise

        installed = InstalledRequest(
            descriptor=descriptor,
            kv_allocation=allocation,
            kv_cache=kv_cache,
            seeded_token_id=descriptor.first_token_id,
        )
        self._installed[request.request_id] = installed
        if descriptor.first_token_id is not None:
            request.generated_token_ids.append(descriptor.first_token_id)
        request.transition(RequestState.READY)
        return installed

    def start(self, request: Request) -> None:
        if request.request_id not in self._installed:
            raise RuntimeError("request state has not been installed")
        request.transition(RequestState.RUNNING)

    def finish(self, request: Request) -> None:
        request.transition(RequestState.FINISHED)
        self._installed.pop(request.request_id, None)
        self.block_manager.free(request.request_id)
        self.state_pool.free(request.request_id)

    def _materialize_kv(
        self,
        request: Request,
        descriptor: StateTransferDescriptor,
        bundle: HybridStateBundle,
    ) -> np.ndarray | Int4Tensor:
        if descriptor.mode is TransferMode.PARTIAL_TRANSFER:
            if bundle.kv_cache is not None:
                raise RuntimeError("partial transfer unexpectedly included KV")
            if self.kv_recompute is None:
                raise RuntimeError("partial transfer requires a decode-side KV recompute hook")
            return self.kv_recompute(request.token_ids)
        if bundle.kv_cache is None:
            raise RuntimeError("full/quantized transfer did not contain KV")
        return bundle.kv_cache
