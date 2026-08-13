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
    args = parser.parse_args()

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
