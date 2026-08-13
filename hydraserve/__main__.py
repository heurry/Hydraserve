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
    args = parser.parse_args()

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
