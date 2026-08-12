"""
Intra-GPU MPS Mode Manager (BulletServe-inspired).

BulletServe uses libsmctrl for SM masking on a single GPU to isolate
prefill and decode. Our environment has CUDA 12.8 (libsmctrl requires
<=12.6), so we use CUDA MPS (Multi-Process Service) instead:

- MPS allows multiple processes to share one GPU
- Prefill process and decode process both run on the same GPU
- State transfer becomes pointer passing (zero-copy, same GPU memory)
- N-1 truncation semantics unchanged

Limitations vs BulletServe:
- No precise SM partitioning (MPS schedules transparently)
- Some SM contention remains
- But: zero transfer overhead + hybrid DP/intra-GPU-PD possible

Usage:
    mps = MPSManager(gpu_id=0)
    mps.start()                      # Start MPS daemon
    prefill_proc = mps.launch_prefill_process()
    decode_proc = mps.launch_decode_process()
    ...
    mps.stop()
"""

import os
import subprocess
import signal
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MPSConfig:
    """Configuration for MPS-based intra-GPU mode."""
    gpu_id: int = 0
    active_thread_percentage: Optional[int] = None  # Limit MPS thread usage
    default_thread_percentage: Optional[int] = None
    pinned_device_mem_limit: Optional[int] = None   # MB


class MPSManager:
    """
    Manages CUDA MPS daemon lifecycle for intra-GPU PD separation.

    MPS pipeline:
        1. nvidia-cuda-mps-control -d    (daemon)
        2. echo start | nvidia-cuda-mps-control
        3. Run prefill + decode processes with CUDA_MPS_PIPE_DIRECTORY env
        4. echo quit | nvidia-cuda-mps-control (cleanup)
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.pipe_dir = f"/tmp/hydraserve_mps_gpu{gpu_id}"
        self.is_running = False
        self._daemon_proc: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        """Start MPS daemon for the target GPU."""
        if self.is_running:
            logger.warning("MPS already running")
            return True

        os.makedirs(self.pipe_dir, exist_ok=True)

        # Set environment for MPS
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["CUDA_MPS_LOG_DIRECTORY"] = self.pipe_dir

        # Start daemon
        try:
            self._daemon_proc = subprocess.Popen(
                ["nvidia-cuda-mps-control", "-d"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(1)  # Wait for daemon to start

            # Start MPS server
            subprocess.run(
                ["nvidia-cuda-mps-control", "-c", "start_server", "-uid", f"hydraserve"],
                env=env, capture_output=True, timeout=10
            )

            self.is_running = True
            logger.info(f"MPS daemon started for GPU {self.gpu_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start MPS: {e}")
            return False

    def stop(self) -> None:
        """Stop MPS daemon."""
        if not self.is_running:
            return

        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        try:
            subprocess.run(
                ["nvidia-cuda-mps-control", "-c", "quit"],
                env=env, capture_output=True, timeout=10
            )
            if self._daemon_proc:
                self._daemon_proc.terminate()
                self._daemon_proc.wait(timeout=5)
        except Exception as e:
            logger.error(f"Failed to stop MPS: {e}")

        self.is_running = False
        logger.info(f"MPS daemon stopped for GPU {self.gpu_id}")

    def get_client_env(self) -> Dict[str, str]:
        """Get environment variables for MPS client processes."""
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["CUDA_MPS_ENABLE_ACTIVE_THREAD_PERCENTAGE"] = "1"
        return env

    def is_mps_active(self) -> bool:
        """Check if MPS server is active."""
        try:
            result = subprocess.run(
                ["nvidia-cuda-mps-control", "-c", "get_server_list"],
                env=self.get_client_env(),
                capture_output=True, text=True, timeout=5
            )
            return "hydraserve" in result.stdout
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get MPS statistics."""
        try:
            result = subprocess.run(
                ["nvidia-cuda-mps-control", "-c", "get_device_server_stats", str(self.gpu_id)],
                env=self.get_client_env(),
                capture_output=True, text=True, timeout=5
            )
            stats = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    k, v = line.split(':', 1)
                    stats[k.strip()] = v.strip()
            return stats
        except Exception:
            return {}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def launch_process_under_mps(
    script_path: str,
    gpu_id: int,
    role: str,
    extra_args: Optional[list] = None,
) -> subprocess.Popen:
    """
    Launch a Python process under MPS.

    Args:
        script_path: Path to the Python script
        gpu_id: GPU to run on
        role: "prefill" or "decode" (for logging)
        extra_args: Extra command line args for the script

    Returns:
        The subprocess handle
    """
    mps = MPSManager(gpu_id)
    env = mps.get_client_env()

    cmd = [
        "python", "-u", script_path,
        "--gpu-id", str(gpu_id),
        "--role", role,
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    logger.info(f"Launched {role} process (PID {proc.pid}) under MPS on GPU {gpu_id}")
    return proc
