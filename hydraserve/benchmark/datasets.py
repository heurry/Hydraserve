"""Low-memory adapters for the benchmark datasets used by HydraServe.

The adapters deliberately return plain prompt/reference records. Tokenization is
owned by the model-facing benchmark runner and is not coupled to dataset I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import io
import json
from pathlib import Path
from random import Random
from typing import Any, BinaryIO, Iterator, TextIO
import zipfile


class DatasetFormatError(ValueError):
    """Raised when a local dataset does not have the expected structure."""


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    dataset: str
    sample_id: str
    prompt: str
    reference: str | list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetCatalog:
    """Resolved paths under one benchmark dataset directory."""

    root: Path

    FILES = {
        "gsm8k": "gsm8k.jsonl",
        "humaneval": "humaneval.jsonl",
        "sharegpt": "sharegpt.json",
        "wikitext": "wikitext-103-test.jsonl",
        "longbench": "longbench.zip",
    }

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "root", Path(root))

    def path(self, dataset: str) -> Path:
        try:
            filename = self.FILES[dataset.lower()]
        except KeyError as exc:
            raise KeyError(
                f"unknown dataset {dataset!r}; expected one of {sorted(self.FILES)}"
            ) from exc
        path = self.root / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"dataset is missing or empty: {path}")
        return path

    def available(self) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for name in self.FILES:
            try:
                result[name] = self.path(name)
            except FileNotFoundError:
                pass
        return result

    def longbench_subsets(self) -> tuple[str, ...]:
        path = self.path("longbench")
        with zipfile.ZipFile(path) as archive:
            return tuple(
                sorted(
                    Path(name).stem
                    for name in archive.namelist()
                    if name.startswith("data/") and name.endswith(".jsonl")
                )
            )


def _text_reader(raw: BinaryIO) -> TextIO:
    """Open UTF-8 text, detecting gzip by magic bytes rather than extension."""
    magic = raw.read(2)
    raw.seek(0)
    binary: BinaryIO = gzip.GzipFile(fileobj=raw) if magic == b"\x1f\x8b" else raw
    return io.TextIOWrapper(binary, encoding="utf-8")


def _iter_jsonl(raw: BinaryIO, *, source: str) -> Iterator[dict[str, Any]]:
    with _text_reader(raw) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetFormatError(
                    f"invalid JSON at {source}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise DatasetFormatError(
                    f"expected an object at {source}:{line_number}"
                )
            yield value


def _iter_json_array(path: Path, *, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Incrementally decode a top-level JSON array without loading it in memory."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        position = 0
        eof = False

        def refill() -> bool:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = stream.read(chunk_size)
            if not chunk:
                eof = True
                return False
            buffer += chunk
            return True

        refill()
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if not refill():
                raise DatasetFormatError(f"empty JSON document: {path}")
        if buffer[position] != "[":
            raise DatasetFormatError(f"expected a top-level JSON array: {path}")
        position += 1

        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise DatasetFormatError(f"unterminated JSON array: {path}")
            if buffer[position] == "]":
                return

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError as exc:
                    if eof or not refill():
                        raise DatasetFormatError(
                            f"invalid JSON array item in {path}: {exc.msg}"
                        ) from exc
            if not isinstance(value, dict):
                raise DatasetFormatError(f"expected array objects in {path}")
            yield value

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise DatasetFormatError(f"unterminated JSON array: {path}")
            marker = buffer[position]
            position += 1
            if marker == "]":
                return
            if marker != ",":
                raise DatasetFormatError(
                    f"expected ',' or ']' after JSON array item in {path}"
                )


def _required(record: dict[str, Any], key: str, source: str) -> Any:
    try:
        return record[key]
    except KeyError as exc:
        raise DatasetFormatError(f"missing {key!r} in {source}") from exc


def _iter_gsm8k(path: Path) -> Iterator[BenchmarkSample]:
    with path.open("rb") as raw:
        for index, record in enumerate(_iter_jsonl(raw, source=str(path))):
            yield BenchmarkSample(
                "gsm8k",
                str(index),
                str(_required(record, "question", "GSM8K record")),
                str(_required(record, "answer", "GSM8K record")),
            )


def _iter_humaneval(path: Path) -> Iterator[BenchmarkSample]:
    with path.open("rb") as raw:
        for record in _iter_jsonl(raw, source=str(path)):
            task_id = str(_required(record, "task_id", "HumanEval record"))
            yield BenchmarkSample(
                "humaneval",
                task_id,
                str(_required(record, "prompt", task_id)),
                str(_required(record, "canonical_solution", task_id)),
                {
                    "entry_point": record.get("entry_point"),
                    "test": record.get("test"),
                },
            )


def _iter_wikitext(path: Path) -> Iterator[BenchmarkSample]:
    with path.open("rb") as raw:
        output_index = 0
        for record in _iter_jsonl(raw, source=str(path)):
            text = str(_required(record, "text", "WikiText record"))
            if not text.strip():
                continue
            yield BenchmarkSample("wikitext", str(output_index), text)
            output_index += 1


def _iter_sharegpt(path: Path) -> Iterator[BenchmarkSample]:
    for index, record in enumerate(_iter_json_array(path)):
        conversations = _required(record, "conversations", "ShareGPT record")
        if not isinstance(conversations, list):
            raise DatasetFormatError("ShareGPT conversations must be a list")
        prompt = None
        reference = None
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).lower()
            value = turn.get("value")
            if not isinstance(value, str) or not value:
                continue
            if prompt is None and role in {"human", "user"}:
                prompt = value
            elif prompt is not None and role in {"gpt", "assistant"}:
                reference = value
                break
        if prompt is None:
            continue
        yield BenchmarkSample(
            "sharegpt",
            str(record.get("id", index)),
            prompt,
            reference,
            {"turn_count": len(conversations)},
        )


def _iter_longbench(path: Path, subset: str) -> Iterator[BenchmarkSample]:
    normalized = subset[:-6] if subset.endswith(".jsonl") else subset
    member = f"data/{normalized}.jsonl"
    with zipfile.ZipFile(path) as archive:
        try:
            raw = archive.open(member)
        except KeyError as exc:
            available = sorted(
                Path(name).stem
                for name in archive.namelist()
                if name.startswith("data/") and name.endswith(".jsonl")
            )
            raise KeyError(
                f"unknown LongBench subset {subset!r}; expected one of {available}"
            ) from exc
        with raw:
            for index, record in enumerate(_iter_jsonl(raw, source=f"{path}!{member}")):
                task_input = str(_required(record, "input", "LongBench record"))
                context = str(_required(record, "context", "LongBench record"))
                answers = record.get("answers")
                if isinstance(answers, list):
                    reference: str | list[str] | None = [str(item) for item in answers]
                elif answers is None:
                    reference = None
                else:
                    reference = str(answers)
                metadata = {
                    key: value
                    for key, value in record.items()
                    if key not in {"input", "context", "answers"}
                }
                yield BenchmarkSample(
                    f"longbench/{normalized}",
                    str(index),
                    f"{context}\n\n{task_input}",
                    reference,
                    metadata,
                )


def iter_dataset(
    root: str | Path,
    dataset: str,
    *,
    subset: str | None = None,
    limit: int | None = None,
) -> Iterator[BenchmarkSample]:
    """Yield normalized records from one local dataset.

    ``limit`` bounds records after filtering (for example blank WikiText rows).
    LongBench requires an explicit subset and is streamed directly from its ZIP.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    name = dataset.lower()
    catalog = DatasetCatalog(root)
    path = catalog.path(name)
    if name == "gsm8k":
        iterator = _iter_gsm8k(path)
    elif name == "humaneval":
        iterator = _iter_humaneval(path)
    elif name == "wikitext":
        iterator = _iter_wikitext(path)
    elif name == "sharegpt":
        iterator = _iter_sharegpt(path)
    elif name == "longbench":
        if not subset:
            raise ValueError("LongBench requires subset=...")
        iterator = _iter_longbench(path, subset)
    else:  # DatasetCatalog.path already validates this, for type narrowing.
        raise AssertionError(name)

    for index, sample in enumerate(iterator):
        if limit is not None and index >= limit:
            return
        yield sample


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """Recipe for a synthetic, length-controlled benchmark workload."""

    num_long: int = 0
    long_tokens: int = 0
    num_short: int = 0
    short_tokens: int = 0
    num_balanced: int = 0
    balanced_min_tokens: int = 0
    balanced_max_tokens: int = 0
    seed: int = 0

    def total_requests(self) -> int:
        return self.num_long + self.num_short + self.num_balanced

    def max_prompt_tokens(self) -> int:
        return max(self.long_tokens, self.short_tokens, self.balanced_max_tokens)

    def validate(self) -> None:
        if self.total_requests() <= 0:
            raise ValueError("synthetic spec requires at least one request")
        if min(self.num_long, self.num_short, self.num_balanced) < 0:
            raise ValueError("synthetic request counts cannot be negative")
        if self.num_long and self.long_tokens <= 0:
            raise ValueError("synthetic long requests require long_tokens > 0")
        if self.num_short and self.short_tokens <= 0:
            raise ValueError("synthetic short requests require short_tokens > 0")
        if self.num_balanced and not (
            0 < self.balanced_min_tokens <= self.balanced_max_tokens
        ):
            raise ValueError(
                "synthetic balanced requests require 0 < balanced_min_tokens "
                "<= balanced_max_tokens"
            )


def _base_vocab_size(tokenizer) -> int:
    """Return the tokenizer vocabulary size excluding added/special tokens."""
    base = getattr(tokenizer, "base_vocab_size", None)
    if base is not None:
        return int(base)
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None:
        return int(inner.get_vocab_size(with_added_tokens=False))
    raise TypeError("tokenizer does not expose a base vocabulary size")


def _random_prompt(tokenizer, target_tokens: int, rng: Random) -> str:
    """Sample a distinct prompt whose re-encoded length approaches ``target_tokens``.

    Random base-vocab IDs can decode to text that re-encodes to a different
    length (byte-fallback merges/splits). We resample with a proportional count
    correction, bounded to avoid pathological loops.
    """
    vocab_size = _base_vocab_size(tokenizer)
    count = max(1, target_tokens)
    text = ""
    for _ in range(4):
        token_ids = [rng.randrange(vocab_size) for _ in range(count)]
        text = tokenizer.decode(token_ids)
        encoded = len(tokenizer.encode(text))
        if encoded == target_tokens:
            return text
        if encoded <= 0:
            continue
        count = max(1, int(round(count * target_tokens / encoded)))
    return text


def iter_synthetic(tokenizer, spec: SyntheticSpec) -> Iterator[BenchmarkSample]:
    """Yield shuffled synthetic requests with reproducible, distinct prompts.

    Entries are ``long-*``/``short-*``/``balanced-*``, shuffled with
    ``Random(spec.seed)``; each prompt is generated from ``Random(f"{seed}:{id}")``
    so requests are distinct and reproducible.
    """
    spec.validate()
    entries: list[tuple[str, int]] = []
    for index in range(spec.num_long):
        entries.append((f"long-{index}", spec.long_tokens))
    for index in range(spec.num_short):
        entries.append((f"short-{index}", spec.short_tokens))
    rng = Random(spec.seed)
    balanced_tokens = [
        rng.randint(spec.balanced_min_tokens, spec.balanced_max_tokens)
        for _ in range(spec.num_balanced)
    ]
    for index, tokens in enumerate(balanced_tokens):
        entries.append((f"balanced-{index}", tokens))
    rng.shuffle(entries)
    for sample_id, target_tokens in entries:
        prompt_rng = Random(f"{spec.seed}:{sample_id}")
        prompt = _random_prompt(tokenizer, target_tokens, prompt_rng)
        yield BenchmarkSample(
            "synthetic",
            sample_id,
            prompt,
            metadata={"target_tokens": target_tokens},
        )
