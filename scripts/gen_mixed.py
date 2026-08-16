#!/usr/bin/env python3
"""Generate the standard mixed workload for DP-vs-unified-pool experiments.

Shared prefix FIRST (identical across records), distinct tail LAST — the radix
prefix cache matches block-aligned prefixes, and a per-record header inside
block 0 makes every record a separate tree path (zero shared blocks).

Usage:
  python scripts/gen_mixed.py <model_dir> <out_dir> [64k|128k]

Defaults: 8 long + 64 short (1:8), longs 64K, shorts 2K, 80% shared prefix.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
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


def build(tok, i, target, shared_fraction=0.8):
    chunk_tokens = len(tok.encode(_CHUNK))
    shared_tokens = int(target * shared_fraction)
    shared_text = _CHUNK * ((shared_tokens + chunk_tokens - 1) // chunk_tokens)
    tail_base = f" Tail of document {i} is distinct from every other record. "
    remaining = max(0, target - len(tok.encode(shared_text + tail_base)))
    tail_fill = _CHUNK * (remaining // chunk_tokens + 1)
    return shared_text + tail_base + tail_fill


def main():
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: {sys.argv[0]} <model_dir> <out_dir> [64k|128k]")
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "64k"
    if mode == "128k":
        n_long, n_short, long_tokens = 8, 32, 131072
    else:
        n_long, n_short, long_tokens = 8, 64, 65536
    tok = QwenTokenizer(model_dir)
    records, idx = [], 0
    step = max(1, n_short // n_long)
    for s in range(n_short):
        records.append({"id": f"short-{s}", "conversations": [
            {"from": "human", "value": build(tok, idx, 2048)},
            {"from": "gpt", "value": "ACK"}]})
        idx += 1
        if (s + 1) % step == 0 and len(
            [r for r in records if r["id"].startswith("long")]
        ) < n_long:
            records.append({"id": f"long-{s}", "conversations": [
                {"from": "human", "value": build(tok, idx, long_tokens)},
                {"from": "gpt", "value": "ACK"}]})
            idx += 1
    out = pathlib.Path(out_dir) / "sharegpt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as stream:
        json.dump(records, stream)
    longs = [r for r in records if r["id"].startswith("long")]
    shorts = [r for r in records if r["id"].startswith("short")]
    shared = sum(
        1
        for x, y in zip(
            tok.encode(longs[0]["conversations"][0]["value"]),
            tok.encode(longs[1]["conversations"][0]["value"]),
        )
        if x == y
    )
    print(
        f"wrote {out}: {len(records)} records ({len(longs)} long / {len(shorts)} short), "
        f"longs share {shared} tokens = {shared // 16} blocks"
    )


if __name__ == "__main__":
    main()
