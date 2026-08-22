"""Tokenizer boundary independent from model execution frameworks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class QwenTokenizer:
    """Load tokenizer.json directly, without a Transformers model backend."""

    def __init__(self, model_dir: str | Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("tokenizers is required; install hydraserve[serve]") from exc
        root = Path(model_dir)
        tokenizer_path = root / "tokenizer.json"
        config_path = root / "tokenizer_config.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(tokenizer_path)
        self.revision = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        self.model_max_length = int(config.get("model_max_length", 0)) or None
        eos_token = config.get("eos_token")
        self.eos_token_id = (
            self._tokenizer.token_to_id(eos_token) if isinstance(eos_token, str) else None
        )

    @property
    def base_vocab_size(self) -> int:
        """Vocabulary size excluding added/special tokens.

        Sampling IDs from ``[0, base_vocab_size)`` avoids special tokens, so a
        synthetic prompt of random IDs re-encodes to ordinary text.
        """
        return int(self._tokenizer.get_vocab_size(with_added_tokens=False))

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str) or not text:
            raise ValueError("prompt must be a non-empty string")
        return tuple(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, token_ids: Iterable[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def render_chat(self, messages: list[Mapping[str, Any]]) -> str:
        if not messages:
            raise ValueError("messages must be a non-empty list")
        pieces: list[str] = []
        allowed = {"system", "user", "assistant", "tool"}
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in allowed or not isinstance(content, str):
                raise ValueError("text chat messages require a valid role and string content")
            pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        pieces.append("<|im_start|>assistant\n")
        return "".join(pieces)


class IncrementalTextDecoder:
    """Return text deltas while retaining byte-level tokenizer context."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.text = ""
        self._emitted_text = ""

    def push(self, token_id: int) -> str:
        self.token_ids.append(int(token_id))
        decoded = self.tokenizer.decode(self.token_ids)
        stable = decoded.rstrip("\ufffd")
        if stable.startswith(self._emitted_text):
            delta = stable[len(self._emitted_text) :]
        else:
            # Tokenizer cleanup should not revise stable Qwen text, but avoid
            # duplicating the shared prefix if a custom tokenizer does.
            common = 0
            for old, new in zip(self._emitted_text, stable):
                if old != new:
                    break
                common += 1
            delta = stable[common:]
        self._emitted_text = stable
        self.text = decoded
        return delta

    def finish(self) -> str:
        delta = self.text[len(self._emitted_text) :] if self.text.startswith(self._emitted_text) else ""
        self._emitted_text = self.text
        return delta
