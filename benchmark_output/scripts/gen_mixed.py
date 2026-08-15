#!/usr/bin/env python3
"""Generate a mixed short/long sharegpt dataset (interleaved)."""
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
    while True:
        n = len(tok.encode(text))
        if n >= target:
            return text
        text += _CHUNK


def main():
    tok = QwenTokenizer("/root/autodl-tmp/Qwen3.5-4B")
    n_long = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_short = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    long_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 131072
    short_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    out = sys.argv[5] if len(sys.argv) > 5 else "/tmp/mixed_data/sharegpt.json"

    records = []
    idx = 0
    # interleave: mostly short, a long every (n_short//n_long) shorts
    step = max(1, n_short // n_long)
    for s in range(n_short):
        records.append({"id": f"short-{s}", "conversations": [
            {"from": "human", "value": build(tok, idx, short_tokens)},
            {"from": "gpt", "value": "ACK"},
        ]})
        idx += 1
        if (s + 1) % step == 0 and len([r for r in records if r["id"].startswith("long")]) < n_long:
            records.append({"id": f"long-{s}", "conversations": [
                {"from": "human", "value": build(tok, idx, long_tokens)},
                {"from": "gpt", "value": "ACK"},
            ]})
            idx += 1
    with open(out, "w") as f:
        json.dump(records, f)
    long_cnt = len([r for r in records if r["id"].startswith("long")])
    print(f"wrote {out}: {len(records)} records ({long_cnt} long / {len(records)-long_cnt} short)")


if __name__ == "__main__":
    main()
