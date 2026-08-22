#!/usr/bin/env python3
"""Start HydraServe's ZeroMQ ROUTER/DEALER data-parallel broker."""

import argparse

from hydraserve.engine.zmq_proxy import run_zmq_broker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", default="tcp://127.0.0.1:9000")
    parser.add_argument("--backend", default="tcp://127.0.0.1:9001")
    args = parser.parse_args()
    run_zmq_broker(args.frontend, args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
