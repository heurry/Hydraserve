from __future__ import annotations

from pathlib import Path

import pytest

from hydraserve.model.tokenizer import IncrementalTextDecoder


class RevisingTokenizer:
    def decode(self, token_ids):
        table = {(1,): "�", (1, 2): "你", (1, 2, 3): "你好"}
        return table[tuple(token_ids)]


def test_incremental_decoder_handles_revised_byte_sequence() -> None:
    decoder = IncrementalTextDecoder(RevisingTokenizer())
    assert decoder.push(1) == ""
    assert decoder.push(2) == "你"
    assert decoder.push(3) == "好"
    assert decoder.text == "你好"


def test_real_qwen_tokenizer_loads_without_transformers_model() -> None:
    pytest.importorskip("tokenizers")
    from hydraserve.model.tokenizer import QwenTokenizer

    model_dir = Path("/mnt/nvme-data/models/LLM_model/Qwen3.5-4B")
    if not model_dir.is_dir():
        pytest.skip("local Qwen tokenizer is absent")
    tokenizer = QwenTokenizer(model_dir)
    ids = tokenizer.encode("HydraServe 测试")
    assert ids
    assert "HydraServe" in tokenizer.decode(ids)
    assert tokenizer.eos_token_id is not None
    rendered = tokenizer.render_chat([{"role": "user", "content": "hello"}])
    assert rendered.endswith("<|im_start|>assistant\n")
