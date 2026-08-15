#!/usr/bin/env python3
"""Generate a synthetic sharegpt dataset with prompts >= target token length (O(n))."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/Hydraserve")
from hydraserve.model.tokenizer import QwenTokenizer

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
    "HydraServe disaggregates prefill and decode across GPUs with dual-state transfer.",
    "Gated delta networks maintain a fixed-size recurrent state that never grows with context.",
    "Long-context inference stresses the paged KV cache and the chunked prefill scheduler.",
    "A cost-aware router picks between collocated and disaggregated serving per request.",
    "Attention scores are computed with online softmax over non-contiguous KV blocks.",
    "The memory planner reserves KV pages and recurrent slots before streaming begins.",
    "Preemption replays the exact prompt plus generated prefix to preserve sampling state.",
    "Radix prefix caching shares physical pages across requests with matching prompts.",
    "Continuous batching packs heterogeneous sequence lengths into a single decode step.",
    "Throughput and tail latency trade off against each other under bursty arrival patterns.",
    "Benchmark traces replay ShareGPT conversations and LongBench documents faithfully.",
]


def build_prompt(tokenizer: QwenTokenizer, target_tokens: int) -> str:
    chunk = " ".join(_SENTENCES) + " "
    # ~3.6 chars/token for English; use a generous factor so we overshoot the target.
    reps = int(target_tokens * 4.5 / len(chunk)) + 1
    text = chunk * reps
    # top up a few times if the estimate undershot (bounded)
    for _ in range(8):
        n = len(tokenizer.encode(text))
        if n >= target_tokens:
            return text, n
        text += chunk * max(1, (target_tokens - n) * 4 // len(chunk))
    return text, len(tokenizer.encode(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=262144)
    ap.add_argument("--count", type=int, default=8)
    args = ap.parse_args()

    tok = QwenTokenizer(args.model)
    prompt, actual = build_prompt(tok, args.tokens)
    records = [
        {"id": f"synthetic-{i}", "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "ACK"},
        ]}
        for i in range(args.count)
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records), encoding="utf-8")
    print(f"wrote {args.out}: {args.count} records x {actual} tokens/prompt (target {args.tokens})")


if __name__ == "__main__":
    main()
