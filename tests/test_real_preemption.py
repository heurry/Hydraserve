from __future__ import annotations

import os
from threading import Event

import pytest

torch = pytest.importorskip("torch")

from hydraserve.cache import KVBlockManager, PagedKVCache
from hydraserve.engine import ContinuousGenerationLoop, RuntimeGenerationBackend
from hydraserve.model.runtime import QwenTextRuntime


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("HYDRASERVE_RUN_REAL_PREEMPTION") != "1",
        reason="set HYDRASERVE_RUN_REAL_PREEMPTION=1 to load the 8.8GB checkpoint",
    ),
]


def test_real_4b_preemption_recovery_matches_uninterrupted_generation() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    runtime = QwenTextRuntime.from_checkpoint(
        "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B",
        device="cuda:0",
        dtype=torch.bfloat16,
        use_triton=True,
        use_flash_attention=False,
    )
    cache = PagedKVCache(
        runtime.config,
        KVBlockManager(64, block_size=16, headroom_blocks=2),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    inner = RuntimeGenerationBackend(runtime, cache, max_state_slots=1)
    decode_entered = Event()
    allow_decode = Event()

    class BlockingBackend:
        def __init__(self):
            self.blocked = False

        def __getattr__(self, name):
            return getattr(inner, name)

        def decode(self, requests):
            if not self.blocked:
                self.blocked = True
                decode_entered.set()
                assert allow_decode.wait(30)
            return inner.decode(requests)

    loop = ContinuousGenerationLoop(BlockingBackend(), max_batch_size=1)
    prompt = [1, 42, 17, 9]
    try:
        background = loop.submit(prompt, max_new_tokens=4)
        first = background.get(timeout=60)
        assert first.token_id is not None
        assert decode_entered.wait(60)
        urgent = loop.submit([1, 7], max_new_tokens=1, priority=7)
        allow_decode.set()
        urgent_events = list(urgent)
        background_events = [first, *list(background)]
        reference_events = list(loop.submit(prompt, max_new_tokens=4))
    finally:
        loop.close(timeout=60)

    generated = [event.token_id for event in background_events if event.token_id is not None]
    reference = [event.token_id for event in reference_events if event.token_id is not None]
    assert generated == reference
    assert urgent_events[-1].finish_reason == "length"
    assert background_events[-1].finish_reason == "length"
    assert loop.preemptions_total == 1
    assert loop.recoveries_total == 1
    audit = inner.audit_resources()
    assert audit["active_allocations"] == 0
    assert audit["state_allocated_slots"] == 0
    assert audit["physical_free_blocks"] == audit["physical_total_blocks"]
