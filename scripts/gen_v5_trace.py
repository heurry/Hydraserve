#!/usr/bin/env python3
"""Generate the V5 M1/B1 frozen traces from real multi-document QA content.

V5 (BENCHMARK_PLAN_V5.md) replaces the random-token R1/R2/R3 traces with real
content replayed identically across DP and H1.  This script freezes the two
official trace families:

M1  real mixed RAG (120 s window)
    48 short RAG requests  (prompt 1K-4K tokens, 256 output)  +
    16 long  RAG requests  (prompt 8K-16K tokens, 512 output)
    short/long arrivals come from independent, seeded Poisson-ish processes
    that both cover the full window; prompts are real LongBench QA records
    assembled as ``system + question + retrieved documents + instruction``.

B1  Long-heavy boundary (60 s window)
    8 short + 8 long RAG requests (same content distributions as M1); long
    requests arrive as four 2-request bursts at 10/25/40/55 s, short requests
    arrive Poisson-ish over the full window.  Exercises Hybrid saturation,
    queueing, the 5 s overflow fallback and no-starvation behaviour.

All entries run with ``ignore_eos=false`` (V5 real mode) and greedy sampling.
The trace is written with per-record/per-trace SHA256 (see
``hydraserve.benchmark.datasets.write_trace``); replaying it with ``iter_trace``
re-validates every hash and re-encoded token count.

Usage:
  python scripts/gen_v5_trace.py MODEL_DIR DATASETS_DIR OUT_DIR \
      [--seeds 42 43 44] [--m1-only] [--b1-only]

Example:
  python scripts/gen_v5_trace.py /root/autodl-tmp/Qwen3.5-4B \
      /root/autodl-tmp/data traces/v5
"""

from __future__ import annotations

import argparse
import json
import pathlib
from random import Random
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hydraserve.benchmark.datasets import iter_trace, write_trace  # noqa: E402
from hydraserve.model.tokenizer import QwenTokenizer  # noqa: E402

SYSTEM = (
    "You are a customer-service assistant. Answer the user's question using "
    "only the retrieved documents below."
)
INSTRUCTION = (
    "Answer concisely and accurately based only on the retrieved documents. "
    "If the documents do not contain the answer, say so directly."
)

# Real multi-document QA subsets (skip *_e translations and zh variants).
_SUBSETS = (
    "hotpotqa",
    "2wikimqa",
    "musique",
    "qasper",
    "multifieldqa_en",
    "narrativeqa",
    "multi_news",
    "gov_report",
    "dureader",
    "passage_retrieval_en",
    "trec",
    "triviaqa",
    "lsht",
    "lcc",
)

# M1 workload (V5 section 3.1).
M1_SHORT_COUNT = 48
M1_LONG_COUNT = 16
M1_WINDOW_S = 120.0
M1_SHORT_RANGE = (1024, 4096)
M1_LONG_RANGE = (8192, 16384)
M1_SHORT_MAX_NEW = 256
M1_LONG_MAX_NEW = 512

# B1 workload (V5 section 3.2).
B1_SHORT_COUNT = 8
B1_LONG_COUNT = 8
B1_WINDOW_S = 60.0
B1_LONG_BURST_OFFSETS_S = (10.0, 25.0, 40.0, 55.0)
B1_LONG_MAX_NEW = 512

# Documents to retrieve per prompt, by class.
_DOCS_PER_REQUEST = {"short": 3, "long": 9}


def load_records(datasets_root: pathlib.Path) -> list[dict]:
    """Load real (question, context) records from LongBench subsets."""
    zip_path = datasets_root / "longbench.zip"
    if not zip_path.is_file():
        raise SystemExit(f"LongBench zip not found: {zip_path}")
    records: list[dict] = []
    with zipfile.ZipFile(zip_path) as archive:
        for subset in _SUBSETS:
            member = f"data/{subset}.jsonl"
            try:
                raw = archive.open(member)
            except KeyError:
                continue
            with raw:
                for line in raw:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "input" in record and "context" in record:
                        records.append(
                            {
                                "subset": subset,
                                "question": str(record["input"]),
                                "context": str(record["context"]),
                            }
                        )
    if not records:
        raise SystemExit("no usable LongBench records found")
    return records


def _poisson_offsets_ms(rng: Random, count: int, window_s: float) -> list[float]:
    """Exactly ``count`` arrival offsets covering ``[0, window_s]``.

    Exponential inter-arrival gaps (Poisson-like), rescaled so the final
    arrival lands at the window edge.  Deterministic per ``rng``.
    """
    if count <= 0:
        return []
    if count == 1:
        return [rng.uniform(0.0, window_s) * 1000.0]
    gaps = [rng.expovariate(1.0) for _ in range(count)]
    scale = window_s / sum(gaps)
    offsets: list[float] = []
    cum = 0.0
    for gap in gaps[:-1]:
        cum += gap
        offsets.append(cum * scale * 1000.0)
    offsets.append(window_s * 1000.0)
    return offsets


def retrieve_docs(records: list[dict], rng: Random, count: int) -> tuple[str, str]:
    """Sample ``count`` records; return ``(question, concatenated docs)``."""
    chosen = rng.sample(records, min(count, len(records)))
    question = chosen[0]["question"].strip()
    docs = "\n\n".join(
        f"[{index}] {record['context'].strip()}" for index, record in enumerate(chosen, 1)
    )
    return question, docs


def fit_prompt(
    tokenizer, question: str, docs: str, target_tokens: int
) -> str:
    """Assemble a RAG prompt whose re-encoded length is ~``target_tokens``.

    Only the retrieved-documents section is adjusted (truncate or repeat real
    document text), so prompts stay realistic while hitting the length target.
    """
    head = f"{SYSTEM}\n\n# Question\n{question}\n\n# Retrieved documents\n"
    tail = f"\n\n# Instructions\n{INSTRUCTION}"
    budget = target_tokens - len(tokenizer.encode(head)) - len(tokenizer.encode(tail))
    if budget <= 8:
        raise ValueError(f"question too long for {target_tokens} token target")
    doc_ids = tokenizer.encode(docs)
    if len(doc_ids) >= budget:
        fitted = tokenizer.decode(doc_ids[:budget])
    else:
        fitted = tokenizer.decode((doc_ids * (budget // len(doc_ids) + 1))[:budget])
    return head + fitted + tail


def _sample_prompt(
    tokenizer,
    records: list[dict],
    rng: Random,
    target_lo: int,
    target_hi: int,
    klass: str,
) -> str:
    """Sample a real RAG prompt whose re-encoded length lands in the range."""
    for _ in range(8):
        target = rng.randint(target_lo, target_hi)
        question, docs = retrieve_docs(records, rng, _DOCS_PER_REQUEST[klass])
        prompt = fit_prompt(tokenizer, question, docs, target)
        size = len(tokenizer.encode(prompt))
        if target_lo <= size <= target_hi:
            return prompt
    raise ValueError(f"could not fit a {klass} prompt into {target_lo}-{target_hi} tokens")


def _build_entries(
    tokenizer,
    records: list[dict],
    rng: Random,
    *,
    sample_ids: list[str],
    klass: str,
    target_lo: int,
    target_hi: int,
    max_new_tokens: int,
    offsets_ms: list[float],
    seed: int,
) -> list:
    from hydraserve.benchmark.datasets import TraceSpec

    entries = []
    for sample_id, offset in zip(sample_ids, offsets_ms, strict=True):
        prompt = _sample_prompt(
            tokenizer, records, rng, target_lo, target_hi, klass
        )
        entries.append(
            TraceSpec(
                id=sample_id,
                klass=klass,
                prompt_tokens=len(tokenizer.encode(prompt)),
                max_new_tokens=max_new_tokens,
                arrival_offset_ms=offset,
                ignore_eos=False,
                seed=seed,
                prompt=prompt,
            )
        )
    return entries


def generate_m1(tokenizer, records, rng, seed: int) -> list:
    from hydraserve.benchmark.datasets import TraceSpec

    short_rng = Random(f"{seed}:m1-short")
    long_rng = Random(f"{seed}:m1-long")
    short_offsets = _poisson_offsets_ms(short_rng, M1_SHORT_COUNT, M1_WINDOW_S)
    long_offsets = _poisson_offsets_ms(long_rng, M1_LONG_COUNT, M1_WINDOW_S)
    entries = _build_entries(
        tokenizer,
        records,
        rng,
        sample_ids=[f"short-{i}" for i in range(M1_SHORT_COUNT)],
        klass="short",
        target_lo=M1_SHORT_RANGE[0],
        target_hi=M1_SHORT_RANGE[1],
        max_new_tokens=M1_SHORT_MAX_NEW,
        offsets_ms=short_offsets,
        seed=seed,
    )
    entries += _build_entries(
        tokenizer,
        records,
        rng,
        sample_ids=[f"long-{i}" for i in range(M1_LONG_COUNT)],
        klass="long",
        target_lo=M1_LONG_RANGE[0],
        target_hi=M1_LONG_RANGE[1],
        max_new_tokens=M1_LONG_MAX_NEW,
        offsets_ms=long_offsets,
        seed=seed,
    )
    return entries


def generate_b1(tokenizer, records, rng, seed: int) -> list:
    from hydraserve.benchmark.datasets import TraceSpec

    short_rng = Random(f"{seed}:b1-short")
    short_offsets = _poisson_offsets_ms(short_rng, B1_SHORT_COUNT, B1_WINDOW_S)
    # Four 2-request bursts: two Long arrive simultaneously at each marker.
    long_offsets = [
        offset_s * 1000.0
        for offset_s in B1_LONG_BURST_OFFSETS_S
        for _ in range(B1_LONG_COUNT // len(B1_LONG_BURST_OFFSETS_S))
    ]
    entries = _build_entries(
        tokenizer,
        records,
        rng,
        sample_ids=[f"short-{i}" for i in range(B1_SHORT_COUNT)],
        klass="short",
        target_lo=M1_SHORT_RANGE[0],
        target_hi=M1_SHORT_RANGE[1],
        max_new_tokens=M1_SHORT_MAX_NEW,
        offsets_ms=short_offsets,
        seed=seed,
    )
    entries += _build_entries(
        tokenizer,
        records,
        rng,
        sample_ids=[f"long-{i}" for i in range(B1_LONG_COUNT)],
        klass="long",
        target_lo=M1_LONG_RANGE[0],
        target_hi=M1_LONG_RANGE[1],
        max_new_tokens=B1_LONG_MAX_NEW,
        offsets_ms=long_offsets,
        seed=seed,
    )
    return entries


def validate(tokenizer, trace_path: pathlib.Path, seed: int) -> dict:
    """Replay the trace (re-validates every SHA256/recode) and report stats."""
    samples = list(iter_trace(tokenizer, trace_path, seed=seed))
    by_class: dict[str, dict] = {}
    for sample in samples:
        bucket = by_class.setdefault(
            sample.klass, {"count": 0, "tokens": [], "arrivals": []}
        )
        bucket["count"] += 1
        bucket["tokens"].append(sample.metadata["reencode_tokens"])
        bucket["arrivals"].append(sample.metadata["arrival_offset_ms"])
    report = {"path": str(trace_path), "samples": len(samples)}
    for klass, bucket in sorted(by_class.items()):
        report[klass] = {
            "count": bucket["count"],
            "reencode_tokens": [
                min(bucket["tokens"]),
                max(bucket["tokens"]),
            ],
            "arrival_offset_ms": [
                min(bucket["arrivals"]),
                max(bucket["arrivals"]),
            ],
            # True when every sample in the class runs real mode (EOS enabled).
            "eos_enabled": all(
                not s.metadata["ignore_eos"] for s in samples if s.klass == klass
            ),
        }
    meta = trace_path.with_suffix(trace_path.suffix + ".meta.json")
    if meta.is_file():
        report["meta"] = json.loads(meta.read_text(encoding="utf-8"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate V5 M1/B1 frozen traces from real LongBench content"
    )
    parser.add_argument("model_dir", type=pathlib.Path)
    parser.add_argument("datasets_dir", type=pathlib.Path)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--m1-only", action="store_true")
    mode.add_argument("--b1-only", action="store_true")
    args = parser.parse_args()

    tokenizer = QwenTokenizer(args.model_dir)
    records = load_records(args.datasets_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for seed in args.seeds:
        rng = Random(seed)
        if not args.b1_only:
            entries = generate_m1(tokenizer, records, rng, seed)
            trace_path = args.out_dir / f"m1_seed{seed}.jsonl"
            write_trace(tokenizer, entries, trace_path, seed=seed)
            report = validate(tokenizer, trace_path, seed)
            total += report["samples"]
            print(json.dumps({"workload": "m1", **report}, ensure_ascii=False, indent=2))
        if not args.m1_only:
            entries = generate_b1(tokenizer, records, rng, seed)
            trace_path = args.out_dir / f"b1_seed{seed}.jsonl"
            write_trace(tokenizer, entries, trace_path, seed=seed)
            report = validate(tokenizer, trace_path, seed)
            total += report["samples"]
            print(json.dumps({"workload": "b1", **report}, ensure_ascii=False, indent=2))
    print(f"wrote {total} trace records for seeds {args.seeds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
