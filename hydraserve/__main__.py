from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import subprocess

from hydraserve.config import discover_model_configs, load_model_config
from hydraserve.diagnostics import enable_stall_diagnostics


def _collect_benchmark_metadata(args) -> dict:
    """Reproducibility metadata (P0.4, V3 plan): commit, CLI, versions, GPU."""
    import hashlib
    import json

    meta: dict = {
        "git_commit": None,
        "git_dirty": None,
        "cli": {k: str(v) for k, v in sorted(vars(args).items())},
        "env": {
            k: os.environ[k]
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "HYDRASERVE_PAGED_ATTENTION",
                "HYDRASERVE_PAGED_PREFILL",
            )
            if k in os.environ
        },
    }
    try:
        cwd = Path(__file__).resolve().parent
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_dirty"] = bool(dirty)
    except Exception:
        pass
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda
        available = torch.cuda.is_available()
        meta["gpu"] = torch.cuda.get_device_name(0) if available else None
        meta["gpu_count"] = torch.cuda.device_count() if available else 0
        meta["gpus"] = (
            [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
            if available
            else []
        )
    except Exception:
        pass
    try:
        import triton

        meta["triton"] = triton.__version__
    except Exception:
        pass
    try:
        import flash_attn

        meta["flash_attn"] = flash_attn.__version__
    except Exception:
        pass
    model_dir = Path(args.model).resolve()
    meta["model_dir"] = str(model_dir)
    model_manifest = []
    for pattern in ("*.json", "*.safetensors", "*.bin", "*.model"):
        for path in sorted(model_dir.glob(pattern)):
            if path.is_file():
                model_manifest.append(
                    {"name": path.name, "size_bytes": path.stat().st_size}
                )
    meta["model_manifest_sha256"] = hashlib.sha256(
        json.dumps(model_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    meta["model_manifest"] = model_manifest
    trace_path = getattr(args, "trace", None)
    if trace_path:
        trace_path = Path(trace_path).resolve()
        meta["trace_path"] = str(trace_path)
        meta["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        trace_meta = trace_path.with_suffix(trace_path.suffix + ".meta.json")
        if trace_meta.is_file():
            try:
                meta["trace_metadata"] = json.loads(
                    trace_meta.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                meta["trace_metadata"] = {"error": "unreadable"}
    try:
        meta["nvidia_smi_query"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,pci.bus_id,driver_version,pstate,"
                "temperature.gpu,clocks.sm,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip().splitlines()
        meta["nvidia_topology"] = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        pass
    return meta


def _describe(path: Path, result) -> str:
    if isinstance(result, Exception):
        return f"FAIL {path.name}: {result}"
    return (
        f"OK   {path.name}: layers={result.num_hidden_layers} "
        f"(linear={result.num_linear_layers}, full={result.num_full_attention_layers}), "
        f"KV={result.kv_bytes_per_token_bf16} B/token, "
        f"recurrent={result.recurrent_state_bytes / 1e6:.2f} MB"
    )


def main() -> int:
    enable_stall_diagnostics("coordinator")
    parser = argparse.ArgumentParser(prog="python -m hydraserve")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect-models", help="validate model config files")
    inspect_parser.add_argument("path", type=Path)
    dataset_parser = subparsers.add_parser(
        "inspect-datasets", help="validate and sample local benchmark datasets"
    )
    dataset_parser.add_argument("path", type=Path)
    dataset_parser.add_argument("--limit", type=int, default=1)
    fit_router_parser = subparsers.add_parser(
        "fit-router-profile",
        help="fit a cost-aware route profile from warmed concurrency-1 benchmarks",
    )
    fit_router_parser.add_argument("--collocated", type=Path, nargs="+", required=True)
    fit_router_parser.add_argument(
        "--pd-disaggregated", type=Path, nargs="+", required=True
    )
    fit_router_parser.add_argument("--minimum-pd-prompt-tokens", type=int, default=256)
    fit_router_parser.add_argument("--minimum-savings-ms", type=float, default=5.0)
    fit_router_parser.add_argument("--minimum-savings-ratio", type=float, default=0.05)
    fit_router_parser.add_argument(
        "--pd-uncertainty-multiplier", type=float, default=1.10
    )
    fit_router_parser.add_argument("--ewma-alpha", type=float, default=0.2)
    fit_router_parser.add_argument("--hysteresis-ms", type=float, default=5.0)
    fit_router_parser.add_argument("--hysteresis-ratio", type=float, default=0.02)
    fit_router_parser.add_argument("--drift-ratio-threshold", type=float, default=1.5)
    fit_router_parser.add_argument("--drift-min-observations", type=int, default=5)
    fit_router_parser.add_argument(
        "--allow-routing-during-drift", action="store_true"
    )
    fit_router_parser.add_argument("--output", type=Path)
    serve_parser = subparsers.add_parser(
        "serve", help="run the HydraServe OpenAI-compatible HTTP server"
    )
    serve_parser.add_argument("model", type=Path)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--device", default="cuda:0")
    serve_parser.add_argument("--decode-device", default="cuda:1")
    serve_parser.add_argument("--decode-devices", nargs="+")
    serve_parser.add_argument("--prefill-devices", nargs="+", help="multiple prefill workers (nP+mD)")
    serve_parser.add_argument("--worker-log-dir", default="", help="capture per-worker stderr into this directory")
    serve_parser.add_argument(
        "--pd-schedule",
        choices=("round-robin", "kv-aware", "load-aware"),
        default="round-robin",
        help="decode-worker selection policy for multi-card PD",
    )
    serve_mode = serve_parser.add_mutually_exclusive_group()
    serve_mode.add_argument("--pd", action="store_true")
    serve_mode.add_argument("--adaptive", action="store_true")
    serve_parser.add_argument("--router-profile", type=Path)
    serve_parser.add_argument(
        "--force-pd-tokens",
        type=int,
        default=0,
        help="force PD disaggregated routing when prompt token count reaches "
        "this threshold (0 disables)",
    )
    serve_parser.add_argument(
        "--conditional-pd-tokens",
        type=int,
        default=0,
        help="deterministically run prompts at or above this length through PD; "
        "shorter prompts stay collocated on decode workers (0 disables)",
    )
    serve_parser.add_argument(
        "--prefill-short-policy",
        choices=("never", "work-conserving"),
        default="work-conserving",
        help="whether an otherwise-idle prefill GPU may serve collocated short requests",
    )
    serve_parser.add_argument(
        "--prefill-preempt-max-ops",
        type=int,
        default=8,
        help="maximum queued short decode/release operations served per prefill chunk boundary",
    )
    serve_parser.add_argument(
        "--hybrid-prefill-reserve-tokens",
        type=int,
        default=-1,
        help="KV tokens kept free on a decode-role hybrid P worker for a future "
        "long prefill (-1 reserves min(32K, half the cache); 0 disables)",
    )
    serve_parser.add_argument(
        "--hybrid-long-overflow-ms",
        type=float,
        default=5000.0,
        help="wait this long for a busy Hybrid prefill slot before a Long request "
        "falls back to collocated execution on a D-bound worker",
    )
    serve_parser.add_argument(
        "--pd-prefill-token-budget",
        type=int,
        default=0,
        help="maximum outstanding Long prompt tokens admitted to the P/Hybrid pool "
        "(0 disables the token-aware guard)",
    )
    serve_parser.add_argument(
        "--hybrid-short-max-prefill-backlog-tokens",
        type=int,
        default=0,
        help="allow Hybrid to serve new Short requests only when its outstanding "
        "prefill backlog is at or below this token budget (0 disables)",
    )
    serve_parser.add_argument(
        "--hybrid-short-max-assigned-work",
        type=int,
        default=0,
        help="allow Hybrid to serve new Short requests only when its assigned "
        "token work is at or below this budget (0 disables)",
    )
    serve_parser.add_argument(
        "--hybrid-long-pressure-hold-ms",
        type=float,
        default=0.0,
        help="after a Long request is deferred waiting for Hybrid/P capacity, "
        "keep idle Hybrid workers out of the Short pool for this long (0 disables)",
    )
    serve_parser.add_argument("--cache-tokens", type=int, default=65536)
    serve_parser.add_argument("--kv-headroom-blocks", type=int, default=0)
    serve_parser.add_argument("--block-size", type=int, default=16)
    serve_parser.add_argument("--max-batch-size", type=int, default=64)
    serve_parser.add_argument("--max-active-requests", type=int)
    serve_parser.add_argument("--max-preemptions-per-request", type=int, default=2)
    serve_parser.add_argument("--max-queue-size", type=int, default=1024)
    serve_parser.add_argument("--max-queue-tokens", type=int, default=1048576)
    serve_parser.add_argument("--max-step-tokens", type=int, default=8192)
    serve_parser.add_argument("--dp-graph-sync", action="store_true")
    serve_parser.add_argument("--host-prefix-cache-gb", type=float, default=0.0)
    serve_parser.add_argument(
        "--pd-transfer-backend", choices=("shm-ring", "shm"), default="shm-ring"
    )
    serve_parser.add_argument("--pd-transfer-quant", choices=("int8",), default=None)
    serve_parser.add_argument("--pd-transfer-target-mb", type=float, default=8.0)
    serve_parser.add_argument("--pd-transfer-inflight", type=int, default=2)
    serve_parser.add_argument("--pd-max-concurrent-prepares", type=int, default=2)
    serve_parser.add_argument("--pd-receiver-dispatch-timeout-s", type=float, default=5.0)
    serve_parser.add_argument("--pd-receiver-arm-timeout-s", type=float, default=10.0)
    serve_parser.add_argument("--shm-ring-slots", type=int, default=3)
    serve_parser.add_argument("--shm-ring-slot-mb", type=float, default=64.0)
    serve_parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    serve_parser.add_argument("--kv-quant", choices=["int8"], default=None, help="compress KV cache to INT8")
    serve_parser.add_argument("--prefix-cache-blocks", type=int, default=0)
    serve_parser.add_argument("--prefix-cache-min-frequency", type=int, default=2)
    serve_parser.add_argument("--no-flash-attention", action="store_true")
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="run local datasets through the HydraServe runtime"
    )
    benchmark_parser.add_argument("model", type=Path)
    benchmark_parser.add_argument("datasets", type=Path)
    benchmark_parser.add_argument("--dataset", required=True)
    benchmark_parser.add_argument(
        "--trace",
        type=Path,
        help="replay a frozen JSONL trace (P0.1); overrides --dataset for sample generation",
    )
    benchmark_parser.add_argument(
        "--trace-out",
        type=Path,
        help="write a frozen JSONL trace from the synthetic spec (all ignore_eos=true) and exit",
    )
    benchmark_parser.add_argument("--subset")
    benchmark_parser.add_argument("--limit", type=int, default=100)
    benchmark_parser.add_argument("--max-new-tokens", type=int, default=32)
    benchmark_parser.add_argument(
        "--long-new-tokens", type=int, help="per-long-request output target for --trace-out"
    )
    benchmark_parser.add_argument(
        "--short-new-tokens", type=int, help="per-short-request output target for --trace-out"
    )
    benchmark_parser.add_argument(
        "--balanced-new-tokens",
        type=int,
        help="per-balanced-request output target for --trace-out",
    )
    benchmark_parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    benchmark_parser.add_argument("--num-long", type=int, default=0)
    benchmark_parser.add_argument("--long-tokens", type=int, default=0)
    benchmark_parser.add_argument("--num-short", type=int, default=0)
    benchmark_parser.add_argument("--short-tokens", type=int, default=0)
    benchmark_parser.add_argument("--num-balanced", type=int, default=0)
    benchmark_parser.add_argument("--balanced-min-tokens", type=int, default=0)
    benchmark_parser.add_argument("--balanced-max-tokens", type=int, default=0)
    benchmark_parser.add_argument(
        "--long-arrival-offsets-ms",
        help="comma-separated absolute arrival offsets for long trace requests",
    )
    benchmark_parser.add_argument(
        "--short-trace-request-rate",
        type=float,
        help="Poisson request/s used to freeze short-request arrival offsets",
    )
    benchmark_parser.add_argument("--concurrency", type=int, default=1)
    benchmark_parser.add_argument(
        "--closed-loop-clients",
        action="store_true",
        help="use legacy closed-loop client workers where each client submits "
        "the next request only after the previous request finishes",
    )
    benchmark_parser.add_argument("--max-preemptions-per-request", type=int, default=2)
    benchmark_parser.add_argument("--warmup", type=int, default=0)
    benchmark_parser.add_argument("--request-rate", type=float)
    benchmark_parser.add_argument(
        "--arrival-pattern", choices=("burst", "fixed", "poisson"), default="burst"
    )
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.add_argument(
        "--slo-short-ttft-ms", type=float, default=1000.0, help="short e2e TTFT SLO (V5 default 1000ms)"
    )
    benchmark_parser.add_argument(
        "--slo-short-tpot-ms", type=float, default=100.0, help="short TPOT SLO (V5 default 100ms)"
    )
    benchmark_parser.add_argument(
        "--slo-long-ttft-ms", type=float, default=10000.0, help="long e2e TTFT SLO (V5 default 10s)"
    )
    benchmark_parser.add_argument(
        "--slo-long-tpot-ms", type=float, default=150.0, help="long TPOT SLO (V5 default 150ms)"
    )
    benchmark_parser.add_argument(
        "--slo-long-admission-wait-ms",
        type=float,
        default=30000.0,
        help="long max admission wait SLO (V5 default 30s)",
    )
    benchmark_parser.add_argument(
        "--slo-starvation-wait-ms",
        type=float,
        default=30000.0,
        help="admission wait above which a request is reported as starved",
    )
    benchmark_parser.add_argument("--device", default="cuda:0")
    benchmark_parser.add_argument("--decode-device", default="cuda:1")
    benchmark_parser.add_argument("--decode-devices", nargs="+")
    benchmark_parser.add_argument("--prefill-devices", nargs="+", help="multiple prefill workers (nP+mD)")
    benchmark_parser.add_argument(
        "--dp-devices",
        nargs="+",
        help="multiple collocated data-parallel workers (engine-only 4xDP, no HTTP)",
    )
    benchmark_parser.add_argument("--worker-log-dir", default="", help="capture per-worker stderr into this directory")
    benchmark_parser.add_argument(
        "--pd-schedule",
        choices=("round-robin", "kv-aware", "load-aware"),
        default="round-robin",
        help="decode-worker selection policy for multi-card PD",
    )
    benchmark_mode = benchmark_parser.add_mutually_exclusive_group()
    benchmark_mode.add_argument("--pd", action="store_true")
    benchmark_mode.add_argument("--adaptive", action="store_true")
    benchmark_parser.add_argument("--router-profile", type=Path)
    benchmark_parser.add_argument(
        "--force-pd-tokens",
        type=int,
        default=0,
        help="force PD disaggregated routing when prompt token count reaches "
        "this threshold (0 disables)",
    )
    benchmark_parser.add_argument(
        "--conditional-pd-tokens",
        type=int,
        default=0,
        help="deterministically run prompts at or above this length through PD; "
        "shorter prompts stay collocated on decode workers (0 disables)",
    )
    benchmark_parser.add_argument(
        "--prefill-short-policy",
        choices=("never", "work-conserving"),
        default="work-conserving",
        help="whether an otherwise-idle prefill GPU may serve collocated short requests",
    )
    benchmark_parser.add_argument(
        "--prefill-preempt-max-ops",
        type=int,
        default=8,
        help="maximum queued short decode/release operations served per prefill chunk boundary",
    )
    benchmark_parser.add_argument(
        "--hybrid-prefill-reserve-tokens",
        type=int,
        default=-1,
        help="KV tokens kept free on a decode-role hybrid P worker for a future "
        "long prefill (-1 reserves min(32K, half the cache); 0 disables)",
    )
    benchmark_parser.add_argument(
        "--hybrid-long-overflow-ms",
        type=float,
        default=5000.0,
        help="wait this long for a busy Hybrid prefill slot before a Long request "
        "falls back to collocated execution on a D-bound worker",
    )
    benchmark_parser.add_argument(
        "--pd-prefill-token-budget",
        type=int,
        default=0,
        help="maximum outstanding Long prompt tokens admitted to the P/Hybrid pool "
        "(0 disables the token-aware guard)",
    )
    benchmark_parser.add_argument(
        "--hybrid-short-max-prefill-backlog-tokens",
        type=int,
        default=0,
        help="allow Hybrid to serve new Short requests only when its outstanding "
        "prefill backlog is at or below this token budget (0 disables)",
    )
    benchmark_parser.add_argument(
        "--hybrid-short-max-assigned-work",
        type=int,
        default=0,
        help="allow Hybrid to serve new Short requests only when its assigned "
        "token work is at or below this budget (0 disables)",
    )
    benchmark_parser.add_argument(
        "--hybrid-long-pressure-hold-ms",
        type=float,
        default=0.0,
        help="after a Long request is deferred waiting for Hybrid/P capacity, "
        "keep idle Hybrid workers out of the Short pool for this long (0 disables)",
    )
    benchmark_parser.add_argument("--cache-tokens", type=int, default=65536)
    benchmark_parser.add_argument("--kv-headroom-blocks", type=int, default=0)
    benchmark_parser.add_argument("--block-size", type=int, default=16)
    benchmark_parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    benchmark_parser.add_argument("--max-step-tokens", type=int, default=8192)
    benchmark_parser.add_argument("--dp-graph-sync", action="store_true")
    benchmark_parser.add_argument("--host-prefix-cache-gb", type=float, default=0.0)
    benchmark_parser.add_argument(
        "--pd-transfer-backend", choices=("shm-ring", "shm"), default="shm-ring"
    )
    benchmark_parser.add_argument("--pd-transfer-quant", choices=("int8",), default=None)
    benchmark_parser.add_argument("--pd-transfer-target-mb", type=float, default=8.0)
    benchmark_parser.add_argument("--pd-transfer-inflight", type=int, default=2)
    benchmark_parser.add_argument("--pd-max-concurrent-prepares", type=int, default=2)
    benchmark_parser.add_argument("--pd-receiver-dispatch-timeout-s", type=float, default=5.0)
    benchmark_parser.add_argument("--pd-receiver-arm-timeout-s", type=float, default=10.0)
    benchmark_parser.add_argument("--shm-ring-slots", type=int, default=3)
    benchmark_parser.add_argument("--shm-ring-slot-mb", type=float, default=64.0)
    benchmark_parser.add_argument("--kv-quant", choices=["int8"], default=None, help="compress KV cache to INT8")
    benchmark_parser.add_argument("--prefix-cache-blocks", type=int, default=0)
    benchmark_parser.add_argument("--prefix-cache-min-frequency", type=int, default=2)
    benchmark_parser.add_argument("--no-flash-attention", action="store_true")
    benchmark_parser.add_argument("--output", type=Path)
    benchmark_parser.add_argument(
        "--dp-proxy",
        metavar="URL",
        help="drive a remote endpoint (e.g. a load-aware DP proxy) over HTTP "
        "instead of an in-process backend",
    )
    args = parser.parse_args()

    if args.command == "fit-router-profile":
        import json

        from hydraserve.router import (
            CostRouterConfig,
            build_router_profile,
            load_calibration_points,
        )

        try:
            profile = build_router_profile(
                load_calibration_points(args.collocated),
                load_calibration_points(args.pd_disaggregated),
                minimum_pd_prompt_tokens=args.minimum_pd_prompt_tokens,
                minimum_savings_ms=args.minimum_savings_ms,
                minimum_savings_ratio=args.minimum_savings_ratio,
                pd_uncertainty_multiplier=args.pd_uncertainty_multiplier,
                ewma_alpha=args.ewma_alpha,
                hysteresis_ms=args.hysteresis_ms,
                hysteresis_ratio=args.hysteresis_ratio,
                drift_ratio_threshold=args.drift_ratio_threshold,
                drift_min_observations=args.drift_min_observations,
                fail_closed_on_drift=not args.allow_routing_during_drift,
            )
            CostRouterConfig.from_dict(profile)
        except (OSError, ValueError) as exc:
            parser.error(f"cannot fit router profile: {exc}")
        output = json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
            print(args.output)
        else:
            print(output, end="")
        return 0

    if args.command == "serve":
        from hydraserve.api import create_server
        from hydraserve.engine import (
            AdaptiveGenerationBackend,
            ContinuousGenerationLoop,
            DisaggregatedGenerationBackend,
            MultiWorkerGenerationBackend,
            PDClusterConfig,
            PDWorkerConfig,
        )
        from hydraserve.model import QwenTokenizer
        from hydraserve.router import CostAwareRouter, CostRouterConfig

        if min(
            args.cache_tokens,
            args.block_size,
            args.max_batch_size,
            args.max_queue_size,
            args.max_queue_tokens,
            args.max_step_tokens,
            args.prefix_cache_min_frequency,
            args.pd_transfer_target_mb,
            args.pd_transfer_inflight,
            args.shm_ring_slots,
            args.shm_ring_slot_mb,
            args.prefill_preempt_max_ops,
        ) <= 0:
            parser.error("cache, batch, and queue limits must be positive")
        max_active_requests = args.max_active_requests or args.max_batch_size
        if max_active_requests < args.max_batch_size:
            parser.error("--max-active-requests cannot be below --max-batch-size")
        if args.prefix_cache_blocks < 0:
            parser.error("prefix cache blocks cannot be negative")
        if args.host_prefix_cache_gb < 0:
            parser.error("host prefix cache size cannot be negative")
        if args.max_preemptions_per_request < 0:
            parser.error("--max-preemptions-per-request cannot be negative")
        if args.pd_prefill_token_budget < 0:
            parser.error("--pd-prefill-token-budget cannot be negative")
        if args.hybrid_short_max_prefill_backlog_tokens < 0:
            parser.error("--hybrid-short-max-prefill-backlog-tokens cannot be negative")
        if args.hybrid_short_max_assigned_work < 0:
            parser.error("--hybrid-short-max-assigned-work cannot be negative")
        if args.hybrid_long_pressure_hold_ms < 0:
            parser.error("--hybrid-long-pressure-hold-ms cannot be negative")
        cache_blocks = (args.cache_tokens + args.block_size - 1) // args.block_size
        if not 0 <= args.kv_headroom_blocks < cache_blocks:
            parser.error("--kv-headroom-blocks must be below physical cache blocks")
        tokenizer = QwenTokenizer(args.model)
        if args.decode_devices and not args.adaptive:
            parser.error("--decode-devices requires --adaptive")
        if args.router_profile and not args.adaptive:
            parser.error("--router-profile requires --adaptive")
        if args.force_pd_tokens and not args.adaptive:
            parser.error("--force-pd-tokens requires --adaptive")
        if args.conditional_pd_tokens and not (args.adaptive and args.decode_devices):
            parser.error("--conditional-pd-tokens requires multi-worker --adaptive")
        if args.conditional_pd_tokens < 0:
            parser.error("--conditional-pd-tokens cannot be negative")
        if args.conditional_pd_tokens and args.force_pd_tokens:
            parser.error(
                "--conditional-pd-tokens and --force-pd-tokens are mutually exclusive"
            )
        try:
            router = (
                CostAwareRouter.from_json(args.router_profile)
                if args.router_profile
                else None
            )
            if args.force_pd_tokens > 0:
                config = (
                    router.config
                    if router is not None
                    else CostRouterConfig.partial_transfer_default()
                )
                router = CostAwareRouter(
                    replace(config, force_pd_tokens=args.force_pd_tokens)
                )
        except (OSError, ValueError) as exc:
            parser.error(f"cannot load router profile: {exc}")
        if args.adaptive and args.decode_devices:
            backend = MultiWorkerGenerationBackend(
                PDClusterConfig(
                    str(args.model),
                    tuple(args.decode_devices),
                    prefill_device=args.device,
                    prefill_devices=tuple(args.prefill_devices) if args.prefill_devices else (),
                    cache_tokens_per_worker=args.cache_tokens,
                    block_size=args.block_size,
                    max_state_slots_per_worker=max_active_requests,
                    max_decode_batch_size_per_worker=args.max_batch_size,
                    use_flash_attention=not args.no_flash_attention,
                    prefill_chunk_size=args.prefill_chunk_size,
                    prefix_cache_blocks=args.prefix_cache_blocks,
                    prefix_cache_min_frequency=args.prefix_cache_min_frequency,
                    kv_headroom_blocks=args.kv_headroom_blocks,
                    kv_quant=args.kv_quant,
                    host_prefix_cache_bytes=int(args.host_prefix_cache_gb * (1 << 30)),
                    transfer_backend=args.pd_transfer_backend,
                    transfer_quant=args.pd_transfer_quant,
                    transfer_target_bytes=int(args.pd_transfer_target_mb * (1 << 20)),
                    max_inflight_transfer_chunks=args.pd_transfer_inflight,
                    max_concurrent_prepares_per_worker=args.pd_max_concurrent_prepares,
                    shm_ring_slots=args.shm_ring_slots,
                    shm_ring_slot_bytes=int(args.shm_ring_slot_mb * (1 << 20)),
                    worker_log_dir=args.worker_log_dir,
                    pd_schedule=args.pd_schedule,
                    conditional_pd_tokens=args.conditional_pd_tokens,
                    prefill_short_policy=args.prefill_short_policy,
                    prefill_preempt_max_ops=args.prefill_preempt_max_ops,
                    hybrid_prefill_reserve_tokens=args.hybrid_prefill_reserve_tokens,
                    hybrid_long_overflow_ms=args.hybrid_long_overflow_ms,
                    pd_prefill_token_budget=args.pd_prefill_token_budget,
                    hybrid_short_max_prefill_backlog_tokens=(
                        args.hybrid_short_max_prefill_backlog_tokens
                    ),
                    hybrid_short_max_assigned_work=args.hybrid_short_max_assigned_work,
                    hybrid_long_pressure_hold_ms=args.hybrid_long_pressure_hold_ms,
                ),
                router=router,
                receiver_dispatch_timeout=args.pd_receiver_dispatch_timeout_s,
                receiver_arm_timeout=args.pd_receiver_arm_timeout_s,
            )
            model_name = backend.model_name
        elif args.pd or args.adaptive:
            worker_config = PDWorkerConfig(
                str(args.model),
                prefill_device=args.device,
                decode_device=args.decode_device,
                cache_tokens=args.cache_tokens,
                block_size=args.block_size,
                use_flash_attention=not args.no_flash_attention,
                prefill_chunk_size=args.prefill_chunk_size,
                max_state_slots=max_active_requests,
                max_decode_batch_size=args.max_batch_size,
                prefix_cache_blocks=args.prefix_cache_blocks,
                prefix_cache_min_frequency=args.prefix_cache_min_frequency,
                kv_headroom_blocks=args.kv_headroom_blocks,
                kv_quant=args.kv_quant,
                host_prefix_cache_bytes=int(args.host_prefix_cache_gb * (1 << 30)),
                transfer_backend=args.pd_transfer_backend,
                transfer_quant=args.pd_transfer_quant,
                transfer_target_bytes=int(args.pd_transfer_target_mb * (1 << 20)),
                max_inflight_transfer_chunks=args.pd_transfer_inflight,
                max_concurrent_prepares=args.pd_max_concurrent_prepares,
                shm_ring_slots=args.shm_ring_slots,
                shm_ring_slot_bytes=int(args.shm_ring_slot_mb * (1 << 20)),
                worker_log_dir=args.worker_log_dir,
            )
            backend = (
                AdaptiveGenerationBackend(
                    worker_config,
                    router=router,
                    receiver_arm_timeout=args.pd_receiver_arm_timeout_s,
                )
                if args.adaptive
                else DisaggregatedGenerationBackend(
                    worker_config,
                    receiver_arm_timeout=args.pd_receiver_arm_timeout_s,
                )
            )
            model_name = backend.model_name
        else:
            import torch

            from hydraserve.cache import (
                CacheNamespace,
                CostAwarePrefixPolicy,
                KVBlockManager,
                PagedKVCache,
                PrefixCache,
                plan_paged_kv_blocks,
            )
            from hydraserve.engine import RuntimeGenerationBackend
            from hydraserve.model import QwenTextRuntime

            runtime = QwenTextRuntime.from_checkpoint(
                args.model,
                device=args.device,
                dtype=torch.bfloat16,
                use_triton=True,
                use_flash_attention=not args.no_flash_attention,
                requested_cache_tokens=args.cache_tokens,
            )
            requested_blocks = (
                args.cache_tokens + args.block_size - 1
            ) // args.block_size
            memory_plan = plan_paged_kv_blocks(
                runtime.config,
                requested_blocks,
                block_size=args.block_size,
                dtype=torch.bfloat16,
                device=args.device,
                state_slots=max_active_requests,
                state_workspace_slots=min(max_active_requests, args.max_batch_size),
                kv_quant=args.kv_quant,
            )
            blocks = memory_plan.planned_blocks
            if args.kv_headroom_blocks >= blocks:
                parser.error("KV headroom consumes the memory-planned cache")
            cache = PagedKVCache(
                runtime.config,
                KVBlockManager(
                    blocks,
                    block_size=args.block_size,
                    headroom_blocks=args.kv_headroom_blocks,
                ),
                device=args.device,
                dtype=torch.bfloat16,
                prefix_cache=(
                    PrefixCache(
                        args.block_size,
                        max_blocks=args.prefix_cache_blocks,
                        policy=CostAwarePrefixPolicy(
                            minimum_frequency=args.prefix_cache_min_frequency
                        ),
                    )
                    if args.prefix_cache_blocks
                    else None
                ),
                cache_namespace=CacheNamespace(
                    model=runtime.config.name,
                    tokenizer_revision=str(args.model.resolve()),
                    model_revision=str(args.model.resolve()),
                ),
                memory_plan=memory_plan,
                kv_quant=args.kv_quant,
            )
            backend = RuntimeGenerationBackend(
                runtime,
                cache,
                prefill_chunk_size=args.prefill_chunk_size,
                max_state_slots=max_active_requests,
                max_decode_batch_size=args.max_batch_size,
            )
            model_name = runtime.config.name
        loop = ContinuousGenerationLoop(
            backend,
            max_batch_size=args.max_batch_size,
            max_active_requests=max_active_requests,
            max_queue_size=args.max_queue_size,
            max_queue_tokens=args.max_queue_tokens,
            eos_token_id=tokenizer.eos_token_id,
            max_preemptions_per_request=args.max_preemptions_per_request,
            max_step_tokens=args.max_step_tokens,
            dp_graph_sync=args.dp_graph_sync,
        )
        server = create_server(
            args.host,
            args.port,
            generation_loop=loop,
            tokenizer=tokenizer,
            model_name=model_name,
        )
        print(
            f"HydraServe model={model_name} "
            f"mode={'adaptive-1p' + str(len(args.decode_devices)) + 'd' if args.decode_devices else ('adaptive' if args.adaptive else ('pd' if args.pd else 'collocated'))} "
            "listening on "
            f"http://{args.host}:{args.port}"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            loop.close()
        return 0

    if args.command == "benchmark":
        import json

        from hydraserve.benchmark import (
            SLOConfig,
            SyntheticSpec,
            TraceSpec,
            iter_dataset,
            iter_synthetic,
            iter_trace,
            run_benchmark,
            run_http_benchmark,
            write_trace,
        )
        from hydraserve.engine import (
            AdaptiveGenerationBackend,
            CollocatedClusterConfig,
            ContinuousGenerationLoop,
            DisaggregatedGenerationBackend,
            MultiGPUCollocatedBackend,
            MultiWorkerGenerationBackend,
            PDClusterConfig,
            PDWorkerConfig,
        )
        from hydraserve.model import QwenTokenizer
        from hydraserve.router import CostAwareRouter, CostRouterConfig

        if min(
            args.cache_tokens,
            args.block_size,
            args.prefix_cache_min_frequency,
            args.pd_transfer_target_mb,
            args.pd_transfer_inflight,
            args.shm_ring_slots,
            args.shm_ring_slot_mb,
            args.prefill_preempt_max_ops,
        ) <= 0:
            parser.error("cache limits must be positive")
        if args.prefix_cache_blocks < 0:
            parser.error("prefix cache blocks cannot be negative")
        if args.host_prefix_cache_gb < 0:
            parser.error("host prefix cache size cannot be negative")
        if args.max_preemptions_per_request < 0:
            parser.error("--max-preemptions-per-request cannot be negative")
        if args.pd_prefill_token_budget < 0:
            parser.error("--pd-prefill-token-budget cannot be negative")
        if args.hybrid_short_max_prefill_backlog_tokens < 0:
            parser.error("--hybrid-short-max-prefill-backlog-tokens cannot be negative")
        if args.hybrid_short_max_assigned_work < 0:
            parser.error("--hybrid-short-max-assigned-work cannot be negative")
        if args.hybrid_long_pressure_hold_ms < 0:
            parser.error("--hybrid-long-pressure-hold-ms cannot be negative")
        cache_blocks = (args.cache_tokens + args.block_size - 1) // args.block_size
        if not 0 <= args.kv_headroom_blocks < cache_blocks:
            parser.error("--kv-headroom-blocks must be below physical cache blocks")
        tokenizer = QwenTokenizer(args.model)
        slo_config = SLOConfig(
            short_ttft_ms=args.slo_short_ttft_ms,
            short_tpot_ms=args.slo_short_tpot_ms,
            long_ttft_ms=args.slo_long_ttft_ms,
            long_tpot_ms=args.slo_long_tpot_ms,
            long_max_admission_wait_ms=args.slo_long_admission_wait_ms,
            starvation_admission_wait_ms=args.slo_starvation_wait_ms,
        )

        if args.trace_out:
            if args.dataset.lower() != "synthetic":
                parser.error("--trace-out requires --dataset synthetic")
            spec = SyntheticSpec(
                num_long=args.num_long,
                long_tokens=args.long_tokens,
                num_short=args.num_short,
                short_tokens=args.short_tokens,
                num_balanced=args.num_balanced,
                balanced_min_tokens=args.balanced_min_tokens,
                balanced_max_tokens=args.balanced_max_tokens,
                seed=args.seed,
            )
            spec.validate()
            output_targets = tuple(
                value
                for value in (
                    args.long_new_tokens,
                    args.short_new_tokens,
                    args.balanced_new_tokens,
                    args.max_new_tokens,
                )
                if value is not None
            )
            if min(output_targets) <= 0:
                parser.error("trace output-token targets must be positive")
            long_offsets = [0.0] * spec.num_long
            if args.long_arrival_offsets_ms:
                try:
                    long_offsets = [
                        float(value.strip())
                        for value in args.long_arrival_offsets_ms.split(",")
                    ]
                except ValueError:
                    parser.error("--long-arrival-offsets-ms must contain numbers")
                if len(long_offsets) != spec.num_long or min(long_offsets) < 0:
                    parser.error(
                        "--long-arrival-offsets-ms needs one non-negative value per long request"
                    )
            if (
                args.short_trace_request_rate is not None
                and args.short_trace_request_rate <= 0
            ):
                parser.error("--short-trace-request-rate must be positive")
            from random import Random as _Random

            rng = _Random(args.seed)
            short_offsets: list[float] = []
            short_elapsed_ms = 0.0
            for index in range(spec.num_short):
                if index and args.short_trace_request_rate is not None:
                    short_elapsed_ms += (
                        rng.expovariate(args.short_trace_request_rate) * 1000.0
                    )
                short_offsets.append(short_elapsed_ms)
            entries: list[TraceSpec] = []
            for index in range(spec.num_long):
                entries.append(
                    TraceSpec(
                        f"long-{index}",
                        "long",
                        spec.long_tokens,
                        args.long_new_tokens or args.max_new_tokens,
                        arrival_offset_ms=long_offsets[index],
                        ignore_eos=True,
                        seed=args.seed,
                    )
                )
            for index in range(spec.num_short):
                entries.append(
                    TraceSpec(
                        f"short-{index}",
                        "short",
                        spec.short_tokens,
                        args.short_new_tokens or args.max_new_tokens,
                        arrival_offset_ms=short_offsets[index],
                        ignore_eos=True,
                        seed=args.seed,
                    )
                )
            for index in range(spec.num_balanced):
                tokens = rng.randint(spec.balanced_min_tokens, spec.balanced_max_tokens)
                entries.append(
                    TraceSpec(
                        f"balanced-{index}",
                        "balanced",
                        tokens,
                        args.balanced_new_tokens or args.max_new_tokens,
                        ignore_eos=True,
                        seed=args.seed,
                    )
                )
            meta = write_trace(tokenizer, entries, args.trace_out, seed=args.seed)
            print(json.dumps(meta, ensure_ascii=False, indent=2))
            return 0

        def build_samples():
            warmup_samples = []
            if args.trace:
                samples = list(iter_trace(tokenizer, args.trace, seed=args.seed))
                max_prompt_tokens = max(
                    args.max_prompt_tokens,
                    max(sample.metadata["target_tokens"] for sample in samples),
                )
                if args.warmup:
                    warmup_spec = SyntheticSpec(
                        num_short=args.warmup,
                        short_tokens=min(args.short_tokens or 512, max_prompt_tokens),
                        seed=args.seed + 1,
                    )
                    warmup_samples = [
                        replace(
                            sample,
                            sample_id=f"warmup-{index}",
                            max_new_tokens=min(args.max_new_tokens, 8),
                            ignore_eos=True,
                        )
                        for index, sample in enumerate(
                            iter_synthetic(tokenizer, warmup_spec)
                        )
                    ]
            elif args.dataset.lower() == "synthetic":
                spec = SyntheticSpec(
                    num_long=args.num_long,
                    long_tokens=args.long_tokens,
                    num_short=args.num_short,
                    short_tokens=args.short_tokens,
                    num_balanced=args.num_balanced,
                    balanced_min_tokens=args.balanced_min_tokens,
                    balanced_max_tokens=args.balanced_max_tokens,
                    seed=args.seed,
                )
                spec.validate()
                samples = list(iter_synthetic(tokenizer, spec))
                if args.warmup > 0:
                    warmup_tokens = args.short_tokens or 512
                    warmup_spec = SyntheticSpec(
                        num_short=args.warmup,
                        short_tokens=warmup_tokens,
                        seed=args.seed + 1,
                    )
                    warmup_samples = list(iter_synthetic(tokenizer, warmup_spec))
                # Avoid the default 8192 truncating long synthetic prompts.
                max_prompt_tokens = max(args.max_prompt_tokens, spec.max_prompt_tokens())
            else:
                loaded = list(
                    iter_dataset(
                        args.datasets,
                        args.dataset,
                        subset=args.subset,
                        limit=args.limit + args.warmup,
                    )
                )
                warmup_samples = loaded[: args.warmup]
                samples = loaded[args.warmup :]
                max_prompt_tokens = args.max_prompt_tokens
            return samples, warmup_samples, max_prompt_tokens

        if args.dp_devices and (args.dp_proxy or args.pd or args.adaptive):
            parser.error(
                "--dp-devices is engine-only 4xDP; combine with neither "
                "--dp-proxy nor --pd/--adaptive"
            )

        if args.dp_proxy:
            samples, warmup_samples, max_prompt_tokens = build_samples()
            summary = run_http_benchmark(
                args.dp_proxy,
                tokenizer,
                samples,
                max_new_tokens=args.max_new_tokens,
                concurrency=args.concurrency,
                max_prompt_tokens=max_prompt_tokens,
                warmup_requests=args.warmup,
                warmup_samples=warmup_samples,
                request_rate=args.request_rate,
                arrival_pattern=args.arrival_pattern,
                seed=args.seed,
                slo=slo_config,
                closed_loop_clients=args.closed_loop_clients,
            )
            summary = replace(summary, metadata=_collect_benchmark_metadata(args))
            output = json.dumps(summary.to_dict(), ensure_ascii=False, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(output + "\n", encoding="utf-8")
                print(args.output)
            else:
                print(output)
            return int(summary.failed > 0)

        if args.decode_devices and not args.adaptive:
            parser.error("--decode-devices requires --adaptive")
        if args.router_profile and not args.adaptive:
            parser.error("--router-profile requires --adaptive")
        if args.force_pd_tokens and not args.adaptive:
            parser.error("--force-pd-tokens requires --adaptive")
        if args.conditional_pd_tokens and not (args.adaptive and args.decode_devices):
            parser.error("--conditional-pd-tokens requires multi-worker --adaptive")
        if args.conditional_pd_tokens < 0:
            parser.error("--conditional-pd-tokens cannot be negative")
        if args.conditional_pd_tokens and args.force_pd_tokens:
            parser.error(
                "--conditional-pd-tokens and --force-pd-tokens are mutually exclusive"
            )
        try:
            router = (
                CostAwareRouter.from_json(args.router_profile)
                if args.router_profile
                else None
            )
            if args.force_pd_tokens > 0:
                config = (
                    router.config
                    if router is not None
                    else CostRouterConfig.partial_transfer_default()
                )
                router = CostAwareRouter(
                    replace(config, force_pd_tokens=args.force_pd_tokens)
                )
        except (OSError, ValueError) as exc:
            parser.error(f"cannot load router profile: {exc}")
        if args.dp_devices:
            backend = MultiGPUCollocatedBackend(
                CollocatedClusterConfig(
                    str(args.model),
                    tuple(args.dp_devices),
                    cache_tokens_per_worker=args.cache_tokens,
                    block_size=args.block_size,
                    max_state_slots_per_worker=args.concurrency,
                    max_decode_batch_size_per_worker=args.concurrency,
                    use_flash_attention=not args.no_flash_attention,
                    prefill_chunk_size=args.prefill_chunk_size,
                    prefix_cache_blocks=args.prefix_cache_blocks,
                    prefix_cache_min_frequency=args.prefix_cache_min_frequency,
                    kv_headroom_blocks=args.kv_headroom_blocks,
                    kv_quant=args.kv_quant,
                    worker_log_dir=args.worker_log_dir,
                ),
            )
        elif args.adaptive and args.decode_devices:
            backend = MultiWorkerGenerationBackend(
                PDClusterConfig(
                    str(args.model),
                    tuple(args.decode_devices),
                    prefill_device=args.device,
                    prefill_devices=tuple(args.prefill_devices) if args.prefill_devices else (),
                    cache_tokens_per_worker=args.cache_tokens,
                    block_size=args.block_size,
                    max_state_slots_per_worker=args.concurrency,
                    max_decode_batch_size_per_worker=args.concurrency,
                    use_flash_attention=not args.no_flash_attention,
                    prefill_chunk_size=args.prefill_chunk_size,
                    prefix_cache_blocks=args.prefix_cache_blocks,
                    prefix_cache_min_frequency=args.prefix_cache_min_frequency,
                    kv_headroom_blocks=args.kv_headroom_blocks,
                    kv_quant=args.kv_quant,
                    host_prefix_cache_bytes=int(args.host_prefix_cache_gb * (1 << 30)),
                    transfer_backend=args.pd_transfer_backend,
                    transfer_quant=args.pd_transfer_quant,
                    transfer_target_bytes=int(args.pd_transfer_target_mb * (1 << 20)),
                    max_inflight_transfer_chunks=args.pd_transfer_inflight,
                    max_concurrent_prepares_per_worker=args.pd_max_concurrent_prepares,
                    shm_ring_slots=args.shm_ring_slots,
                    shm_ring_slot_bytes=int(args.shm_ring_slot_mb * (1 << 20)),
                    worker_log_dir=args.worker_log_dir,
                    pd_schedule=args.pd_schedule,
                    conditional_pd_tokens=args.conditional_pd_tokens,
                    prefill_short_policy=args.prefill_short_policy,
                    prefill_preempt_max_ops=args.prefill_preempt_max_ops,
                    hybrid_prefill_reserve_tokens=args.hybrid_prefill_reserve_tokens,
                    hybrid_long_overflow_ms=args.hybrid_long_overflow_ms,
                    pd_prefill_token_budget=args.pd_prefill_token_budget,
                    hybrid_short_max_prefill_backlog_tokens=(
                        args.hybrid_short_max_prefill_backlog_tokens
                    ),
                    hybrid_short_max_assigned_work=args.hybrid_short_max_assigned_work,
                    hybrid_long_pressure_hold_ms=args.hybrid_long_pressure_hold_ms,
                ),
                router=router,
                receiver_dispatch_timeout=args.pd_receiver_dispatch_timeout_s,
                receiver_arm_timeout=args.pd_receiver_arm_timeout_s,
            )
        elif args.pd or args.adaptive:
            worker_config = PDWorkerConfig(
                str(args.model),
                prefill_device=args.device,
                decode_device=args.decode_device,
                cache_tokens=args.cache_tokens,
                block_size=args.block_size,
                use_flash_attention=not args.no_flash_attention,
                prefill_chunk_size=args.prefill_chunk_size,
                max_state_slots=args.concurrency,
                max_decode_batch_size=args.concurrency,
                prefix_cache_blocks=args.prefix_cache_blocks,
                prefix_cache_min_frequency=args.prefix_cache_min_frequency,
                kv_headroom_blocks=args.kv_headroom_blocks,
                kv_quant=args.kv_quant,
                host_prefix_cache_bytes=int(args.host_prefix_cache_gb * (1 << 30)),
                transfer_backend=args.pd_transfer_backend,
                transfer_quant=args.pd_transfer_quant,
                transfer_target_bytes=int(args.pd_transfer_target_mb * (1 << 20)),
                max_inflight_transfer_chunks=args.pd_transfer_inflight,
                max_concurrent_prepares=args.pd_max_concurrent_prepares,
                shm_ring_slots=args.shm_ring_slots,
                shm_ring_slot_bytes=int(args.shm_ring_slot_mb * (1 << 20)),
                worker_log_dir=args.worker_log_dir,
            )
            backend = (
                AdaptiveGenerationBackend(
                    worker_config,
                    router=router,
                    receiver_arm_timeout=args.pd_receiver_arm_timeout_s,
                )
                if args.adaptive
                else DisaggregatedGenerationBackend(
                    worker_config,
                    receiver_arm_timeout=args.pd_receiver_arm_timeout_s,
                )
            )
        else:
            import torch

            from hydraserve.cache import (
                CacheNamespace,
                CostAwarePrefixPolicy,
                KVBlockManager,
                PagedKVCache,
                PrefixCache,
                plan_paged_kv_blocks,
            )
            from hydraserve.engine import RuntimeGenerationBackend
            from hydraserve.model import QwenTextRuntime

            runtime = QwenTextRuntime.from_checkpoint(
                args.model,
                device=args.device,
                dtype=torch.bfloat16,
                use_triton=True,
                use_flash_attention=not args.no_flash_attention,
                requested_cache_tokens=args.cache_tokens,
            )
            requested_blocks = (
                args.cache_tokens + args.block_size - 1
            ) // args.block_size
            memory_plan = plan_paged_kv_blocks(
                runtime.config,
                requested_blocks,
                block_size=args.block_size,
                dtype=torch.bfloat16,
                device=args.device,
                state_slots=args.concurrency,
                state_workspace_slots=args.concurrency,
                kv_quant=args.kv_quant,
            )
            blocks = memory_plan.planned_blocks
            if args.kv_headroom_blocks >= blocks:
                parser.error("KV headroom consumes the memory-planned cache")
            cache = PagedKVCache(
                runtime.config,
                KVBlockManager(
                    blocks,
                    block_size=args.block_size,
                    headroom_blocks=args.kv_headroom_blocks,
                ),
                device=args.device,
                dtype=torch.bfloat16,
                prefix_cache=(
                    PrefixCache(
                        args.block_size,
                        max_blocks=args.prefix_cache_blocks,
                        policy=CostAwarePrefixPolicy(
                            minimum_frequency=args.prefix_cache_min_frequency
                        ),
                    )
                    if args.prefix_cache_blocks
                    else None
                ),
                cache_namespace=CacheNamespace(
                    model=runtime.config.name,
                    tokenizer_revision=str(args.model.resolve()),
                    model_revision=str(args.model.resolve()),
                ),
                memory_plan=memory_plan,
                kv_quant=args.kv_quant,
            )
            backend = RuntimeGenerationBackend(
                runtime,
                cache,
                prefill_chunk_size=args.prefill_chunk_size,
                max_state_slots=args.concurrency,
                max_decode_batch_size=args.concurrency,
            )
        loop = ContinuousGenerationLoop(
            backend,
            max_batch_size=args.concurrency,
            eos_token_id=tokenizer.eos_token_id,
            max_preemptions_per_request=args.max_preemptions_per_request,
            max_step_tokens=args.max_step_tokens,
            dp_graph_sync=args.dp_graph_sync,
        )
        try:
            samples, warmup_samples, max_prompt_tokens = build_samples()
            summary = run_benchmark(
                loop,
                tokenizer,
                samples,
                max_new_tokens=args.max_new_tokens,
                concurrency=args.concurrency,
                max_prompt_tokens=max_prompt_tokens,
                warmup_requests=args.warmup,
                warmup_samples=warmup_samples,
                request_rate=args.request_rate,
                arrival_pattern=args.arrival_pattern,
                seed=args.seed,
                slo=slo_config,
            )
        finally:
            loop.close()
        summary = replace(summary, metadata=_collect_benchmark_metadata(args))
        output = json.dumps(summary.to_dict(), ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
            print(args.output)
        else:
            print(output)
        return int(summary.failed > 0)

    if args.command == "inspect-datasets":
        from hydraserve.benchmark import DatasetCatalog, iter_dataset

        catalog = DatasetCatalog(args.path)
        failures = 0
        for name, path in catalog.available().items():
            try:
                if name == "longbench":
                    subsets = catalog.longbench_subsets()
                    samples = sum(
                        len(list(iter_dataset(args.path, name, subset=subset, limit=args.limit)))
                        for subset in subsets
                    )
                    detail = f"{len(subsets)} subsets, sampled={samples}"
                else:
                    samples = len(list(iter_dataset(args.path, name, limit=args.limit)))
                    detail = f"sampled={samples}"
                print(f"OK   {name}: {path.name}, {detail}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
        return int(failures > 0)

    if args.path.is_file() or (args.path / "config.json").is_file():
        try:
            result = load_model_config(args.path)
        except Exception as exc:
            print(_describe(args.path.parent if args.path.is_file() else args.path, exc))
            return 1
        print(_describe(args.path.parent if args.path.is_file() else args.path, result))
        return 0

    results = discover_model_configs(args.path)
    failures = 0
    for path, result in results.items():
        print(_describe(path, result))
        failures += isinstance(result, Exception)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
