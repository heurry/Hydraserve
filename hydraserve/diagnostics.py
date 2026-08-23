"""Opt-in diagnostics for stalls that cannot be inspected with an external profiler."""

from __future__ import annotations

import faulthandler
import os
import sys


def enable_stall_diagnostics(role: str) -> None:
    """Periodically dump every Python thread when requested through the env.

    ``HYDRASERVE_STALL_DUMP_SECONDS`` is intentionally opt-in so normal
    benchmarks pay no timer or logging cost.  Worker processes call this only
    after redirecting stderr, which keeps each process's stacks in its existing
    worker log instead of interleaving five processes on the benchmark console.
    """

    raw = os.environ.get("HYDRASERVE_STALL_DUMP_SECONDS", "").strip()
    if not raw:
        return
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise ValueError(
            "HYDRASERVE_STALL_DUMP_SECONDS must be a positive number"
        ) from exc
    if seconds <= 0:
        raise ValueError("HYDRASERVE_STALL_DUMP_SECONDS must be positive")
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(
        seconds,
        repeat=True,
        file=sys.stderr,
    )
    print(
        f"[hydraserve] stall diagnostics enabled: role={role} "
        f"pid={os.getpid()} interval={seconds:g}s",
        file=sys.stderr,
        flush=True,
    )
