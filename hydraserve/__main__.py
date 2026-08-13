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
    serve_parser.add_argument("--pd", action="store_true")
    serve_parser.add_argument("--cache-tokens", type=int, default=65536)
    serve_parser.add_argument("--block-size", type=int, default=16)
    serve_parser.add_argument("--max-batch-size", type=int, default=64)
    serve_parser.add_argument("--prefill-chunk-size", type=int, default=4096)
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
    benchmark_parser.add_argument("--device", default="cuda:0")
    benchmark_parser.add_argument("--decode-device", default="cuda:1")
    benchmark_parser.add_argument("--pd", action="store_true")
    benchmark_parser.add_argument("--cache-tokens", type=int, default=65536)
    benchmark_parser.add_argument("--block-size", type=int, default=16)
    benchmark_parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    benchmark_parser.add_argument("--no-flash-attention", action="store_true")
    benchmark_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "serve":
        from hydraserve.api import create_server
        from hydraserve.engine import (
            ContinuousGenerationLoop,
            DisaggregatedGenerationBackend,
            PDWorkerConfig,
        )
        from hydraserve.model import QwenTokenizer

        if args.cache_tokens <= 0 or args.block_size <= 0:
            parser.error("cache limits must be positive")
        tokenizer = QwenTokenizer(args.model)
        if args.pd:
            backend = DisaggregatedGenerationBackend(
                PDWorkerConfig(
                    str(args.model),
                    prefill_device=args.device,
                    decode_device=args.decode_device,
                    cache_tokens=args.cache_tokens,
                    block_size=args.block_size,
                    use_flash_attention=not args.no_flash_attention,
                    prefill_chunk_size=args.prefill_chunk_size,
                )
            )
            model_name = backend.model_name
        else:
            import torch

            from hydraserve.cache import KVBlockManager, PagedKVCache
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
            )
            backend = RuntimeGenerationBackend(
                runtime, cache, prefill_chunk_size=args.prefill_chunk_size
            )
            model_name = runtime.config.name
        loop = ContinuousGenerationLoop(
            backend,
            max_batch_size=args.max_batch_size,
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
            f"HydraServe model={model_name} mode={'pd' if args.pd else 'collocated'} listening on "
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
            ContinuousGenerationLoop,
            DisaggregatedGenerationBackend,
            PDWorkerConfig,
        )
        from hydraserve.model import QwenTokenizer

        if args.cache_tokens <= 0 or args.block_size <= 0:
            parser.error("cache limits must be positive")
        tokenizer = QwenTokenizer(args.model)
        if args.pd:
            backend = DisaggregatedGenerationBackend(
                PDWorkerConfig(
                    str(args.model),
                    prefill_device=args.device,
                    decode_device=args.decode_device,
                    cache_tokens=args.cache_tokens,
                    block_size=args.block_size,
                    use_flash_attention=not args.no_flash_attention,
                    prefill_chunk_size=args.prefill_chunk_size,
                )
            )
        else:
            import torch

            from hydraserve.cache import KVBlockManager, PagedKVCache
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
            )
            backend = RuntimeGenerationBackend(
                runtime, cache, prefill_chunk_size=args.prefill_chunk_size
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
                limit=args.limit,
            )
            summary = run_benchmark(
                loop,
                tokenizer,
                samples,
                max_new_tokens=args.max_new_tokens,
                concurrency=args.concurrency,
                max_prompt_tokens=args.max_prompt_tokens,
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
