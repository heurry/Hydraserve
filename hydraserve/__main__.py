from __future__ import annotations

import argparse
from pathlib import Path

from hydraserve.config import discover_model_configs, load_model_config


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
    serve_mode = serve_parser.add_mutually_exclusive_group()
    serve_mode.add_argument("--pd", action="store_true")
    serve_mode.add_argument("--adaptive", action="store_true")
    serve_parser.add_argument("--router-profile", type=Path)
    serve_parser.add_argument("--cache-tokens", type=int, default=65536)
    serve_parser.add_argument("--kv-headroom-blocks", type=int, default=0)
    serve_parser.add_argument("--block-size", type=int, default=16)
    serve_parser.add_argument("--max-batch-size", type=int, default=64)
    serve_parser.add_argument("--max-active-requests", type=int)
    serve_parser.add_argument("--max-preemptions-per-request", type=int, default=2)
    serve_parser.add_argument("--max-queue-size", type=int, default=1024)
    serve_parser.add_argument("--max-queue-tokens", type=int, default=1048576)
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
    benchmark_parser.add_argument("--subset")
    benchmark_parser.add_argument("--limit", type=int, default=100)
    benchmark_parser.add_argument("--max-new-tokens", type=int, default=32)
    benchmark_parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    benchmark_parser.add_argument("--concurrency", type=int, default=1)
    benchmark_parser.add_argument("--max-preemptions-per-request", type=int, default=2)
    benchmark_parser.add_argument("--warmup", type=int, default=0)
    benchmark_parser.add_argument("--request-rate", type=float)
    benchmark_parser.add_argument(
        "--arrival-pattern", choices=("burst", "fixed", "poisson"), default="burst"
    )
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.add_argument("--device", default="cuda:0")
    benchmark_parser.add_argument("--decode-device", default="cuda:1")
    benchmark_parser.add_argument("--decode-devices", nargs="+")
    benchmark_parser.add_argument("--prefill-devices", nargs="+", help="multiple prefill workers (nP+mD)")
    benchmark_parser.add_argument("--worker-log-dir", default="", help="capture per-worker stderr into this directory")
    benchmark_mode = benchmark_parser.add_mutually_exclusive_group()
    benchmark_mode.add_argument("--pd", action="store_true")
    benchmark_mode.add_argument("--adaptive", action="store_true")
    benchmark_parser.add_argument("--router-profile", type=Path)
    benchmark_parser.add_argument("--cache-tokens", type=int, default=65536)
    benchmark_parser.add_argument("--kv-headroom-blocks", type=int, default=0)
    benchmark_parser.add_argument("--block-size", type=int, default=16)
    benchmark_parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    benchmark_parser.add_argument("--kv-quant", choices=["int8"], default=None, help="compress KV cache to INT8")
    benchmark_parser.add_argument("--prefix-cache-blocks", type=int, default=0)
    benchmark_parser.add_argument("--prefix-cache-min-frequency", type=int, default=2)
    benchmark_parser.add_argument("--no-flash-attention", action="store_true")
    benchmark_parser.add_argument("--output", type=Path)
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
        from hydraserve.router import CostAwareRouter

        if min(
            args.cache_tokens,
            args.block_size,
            args.max_batch_size,
            args.max_queue_size,
            args.max_queue_tokens,
            args.prefix_cache_min_frequency,
        ) <= 0:
            parser.error("cache, batch, and queue limits must be positive")
        max_active_requests = args.max_active_requests or args.max_batch_size
        if max_active_requests < args.max_batch_size:
            parser.error("--max-active-requests cannot be below --max-batch-size")
        if args.prefix_cache_blocks < 0:
            parser.error("prefix cache blocks cannot be negative")
        if args.max_preemptions_per_request < 0:
            parser.error("--max-preemptions-per-request cannot be negative")
        cache_blocks = (args.cache_tokens + args.block_size - 1) // args.block_size
        if not 0 <= args.kv_headroom_blocks < cache_blocks:
            parser.error("--kv-headroom-blocks must be below physical cache blocks")
        tokenizer = QwenTokenizer(args.model)
        if args.decode_devices and not args.adaptive:
            parser.error("--decode-devices requires --adaptive")
        if args.router_profile and not args.adaptive:
            parser.error("--router-profile requires --adaptive")
        try:
            router = (
                CostAwareRouter.from_json(args.router_profile)
                if args.router_profile
                else None
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
                    worker_log_dir=args.worker_log_dir,
                ),
                router=router,
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
                worker_log_dir=args.worker_log_dir,
            )
            backend = (
                AdaptiveGenerationBackend(worker_config, router=router)
                if args.adaptive
                else DisaggregatedGenerationBackend(worker_config)
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

        from hydraserve.benchmark import iter_dataset, run_benchmark
        from hydraserve.engine import (
            AdaptiveGenerationBackend,
            ContinuousGenerationLoop,
            DisaggregatedGenerationBackend,
            MultiWorkerGenerationBackend,
            PDClusterConfig,
            PDWorkerConfig,
        )
        from hydraserve.model import QwenTokenizer
        from hydraserve.router import CostAwareRouter

        if min(
            args.cache_tokens,
            args.block_size,
            args.prefix_cache_min_frequency,
        ) <= 0:
            parser.error("cache limits must be positive")
        if args.prefix_cache_blocks < 0:
            parser.error("prefix cache blocks cannot be negative")
        if args.max_preemptions_per_request < 0:
            parser.error("--max-preemptions-per-request cannot be negative")
        cache_blocks = (args.cache_tokens + args.block_size - 1) // args.block_size
        if not 0 <= args.kv_headroom_blocks < cache_blocks:
            parser.error("--kv-headroom-blocks must be below physical cache blocks")
        tokenizer = QwenTokenizer(args.model)
        if args.decode_devices and not args.adaptive:
            parser.error("--decode-devices requires --adaptive")
        if args.router_profile and not args.adaptive:
            parser.error("--router-profile requires --adaptive")
        try:
            router = (
                CostAwareRouter.from_json(args.router_profile)
                if args.router_profile
                else None
            )
        except (OSError, ValueError) as exc:
            parser.error(f"cannot load router profile: {exc}")
        if args.adaptive and args.decode_devices:
            backend = MultiWorkerGenerationBackend(
                PDClusterConfig(
                    str(args.model),
                    tuple(args.decode_devices),
                    prefill_device=args.device,
                    cache_tokens_per_worker=args.cache_tokens,
                    block_size=args.block_size,
                    max_state_slots_per_worker=args.concurrency,
                    max_decode_batch_size_per_worker=args.concurrency,
                    use_flash_attention=not args.no_flash_attention,
                    prefill_chunk_size=args.prefill_chunk_size,
                    prefix_cache_blocks=args.prefix_cache_blocks,
                    prefix_cache_min_frequency=args.prefix_cache_min_frequency,
                    kv_headroom_blocks=args.kv_headroom_blocks,
                ),
                router=router,
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
                worker_log_dir=args.worker_log_dir,
            )
            backend = (
                AdaptiveGenerationBackend(worker_config, router=router)
                if args.adaptive
                else DisaggregatedGenerationBackend(worker_config)
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
        )
        try:
            samples = iter_dataset(
                args.datasets,
                args.dataset,
                subset=args.subset,
                limit=args.limit + args.warmup,
            )
            summary = run_benchmark(
                loop,
                tokenizer,
                samples,
                max_new_tokens=args.max_new_tokens,
                concurrency=args.concurrency,
                max_prompt_tokens=args.max_prompt_tokens,
                warmup_requests=args.warmup,
                request_rate=args.request_rate,
                arrival_pattern=args.arrival_pattern,
                seed=args.seed,
            )
        finally:
            loop.close()
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
