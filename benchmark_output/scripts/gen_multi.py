#!/usr/bin/env python3
"""Generate `count` distinct sharegpt records with ~`tokens`-token prompts each."""
import sys
import json

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
_CHUNK = " ".join(_SENTENCES) + " "


def build(tok, i, target):
    text = f"Document {i} begins here. "
    n = 0
    while True:
        n = len(tok.encode(text))
        if n >= target:
            return text
        text += _CHUNK


def main():
    tok = QwenTokenizer("/root/autodl-tmp/Qwen3.5-4B")
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/csweep_data/sharegpt.json"

    records = []
    for i in range(count):
        prompt = build(tok, i, tokens)
        records.append({"id": f"doc-{i}", "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "ACK"},
        ]})
    with open(out, "w") as f:
        json.dump(records, f)
    print(f"wrote {out}: {count} records, ~{tokens} tokens each")


if __name__ == "__main__":
    main()
