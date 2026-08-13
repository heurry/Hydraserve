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
    serve_parser = subparsers.add_parser(
        "serve", help="run the HydraServe OpenAI-compatible HTTP server"
    )
    serve_parser.add_argument("model", type=Path)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--device", default="cuda:0")
    serve_parser.add_argument("--decode-device", default="cuda:1")
    serve_parser.add_argument("--decode-devices", nargs="+")
    serve_mode = serve_parser.add_mutually_exclusive_group()
    serve_mode.add_argument("--pd", action="store_true")
    serve_mode.add_argument("--adaptive", action="store_true")
    serve_parser.add_argument("--router-profile", type=Path)
    serve_parser.add_argument("--cache-tokens", type=int, default=65536)
    serve_parser.add_argument("--block-size", type=int, default=16)
    serve_parser.add_argument("--max-batch-size", type=int, default=64)
    serve_parser.add_argument("--max-queue-size", type=int, default=1024)
    serve_parser.add_argument("--max-queue-tokens", type=int, default=1048576)
    serve_parser.add_argument("--prefill-chunk-size", type=int, default=4096)
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
    benchmark_parser.add_argument("--warmup", type=int, default=0)
    benchmark_parser.add_argument("--request-rate", type=float)
    benchmark_parser.add_argument(
        "--arrival-pattern", choices=("burst", "fixed", "poisson"), default="burst"
    )
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.add_argument("--device", default="cuda:0")
    benchmark_parser.add_argument("--decode-device", default="cuda:1")
    benchmark_parser.add_argument("--decode-devices", nargs="+")
    benchmark_mode = benchmark_parser.add_mutually_exclusive_group()
    benchmark_mode.add_argument("--pd", action="store_true")
    benchmark_mode.add_argument("--adaptive", action="store_true")
    benchmark_parser.add_argument("--router-profile", type=Path)
    benchmark_parser.add_argument("--cache-tokens", type=int, default=65536)
    benchmark_parser.add_argument("--block-size", type=int, default=16)
    benchmark_parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    benchmark_parser.add_argument("--prefix-cache-blocks", type=int, default=0)
    benchmark_parser.add_argument("--prefix-cache-min-frequency", type=int, default=2)
    benchmark_parser.add_argument("--no-flash-attention", action="store_true")
    benchmark_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

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
        if args.prefix_cache_blocks < 0:
            parser.error("prefix cache blocks cannot be negative")
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
                    max_state_slots_per_worker=args.max_batch_size,
                    use_flash_attention=not args.no_flash_attention,
                    prefill_chunk_size=args.prefill_chunk_size,
                    prefix_cache_blocks=args.prefix_cache_blocks,
                    prefix_cache_min_frequency=args.prefix_cache_min_frequency,
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
                max_state_slots=args.max_batch_size,
                prefix_cache_blocks=args.prefix_cache_blocks,
                prefix_cache_min_frequency=args.prefix_cache_min_frequency,
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
            )
            from hydraserve.engine import RuntimeGenerationBackend
            from hydraserve.model import QwenTextRuntime

            runtime = QwenTextRuntime.from_checkpoint(
                args.model,
                device=args.device,
                dtype=torch.bfloat16,
                use_triton=True,
                use_flash_attention=not args.no_flash_attention,
            )
            blocks = (args.cache_tokens + args.block_size - 1) // args.block_size
            cache = PagedKVCache(
                runtime.config,
                KVBlockManager(blocks, block_size=args.block_size),
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
            )
            backend = RuntimeGenerationBackend(
                runtime,
                cache,
                prefill_chunk_size=args.prefill_chunk_size,
                max_state_slots=args.max_batch_size,
            )
            model_name = runtime.config.name
        loop = ContinuousGenerationLoop(
            backend,
            max_batch_size=args.max_batch_size,
            max_queue_size=args.max_queue_size,
            max_queue_tokens=args.max_queue_tokens,
            eos_token_id=tokenizer.eos_token_id,
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
                    use_flash_attention=not args.no_flash_attention,
                    prefill_chunk_size=args.prefill_chunk_size,
                    prefix_cache_blocks=args.prefix_cache_blocks,
                    prefix_cache_min_frequency=args.prefix_cache_min_frequency,
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
                prefix_cache_blocks=args.prefix_cache_blocks,
                prefix_cache_min_frequency=args.prefix_cache_min_frequency,
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
            )
            from hydraserve.engine import RuntimeGenerationBackend
            from hydraserve.model import QwenTextRuntime

            runtime = QwenTextRuntime.from_checkpoint(
                args.model,
                device=args.device,
                dtype=torch.bfloat16,
                use_triton=True,
                use_flash_attention=not args.no_flash_attention,
            )
            blocks = (args.cache_tokens + args.block_size - 1) // args.block_size
            cache = PagedKVCache(
                runtime.config,
                KVBlockManager(blocks, block_size=args.block_size),
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
            )
            backend = RuntimeGenerationBackend(
                runtime,
                cache,
                prefill_chunk_size=args.prefill_chunk_size,
                max_state_slots=args.concurrency,
            )
        loop = ContinuousGenerationLoop(
            backend,
            max_batch_size=args.concurrency,
            eos_token_id=tokenizer.eos_token_id,
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
