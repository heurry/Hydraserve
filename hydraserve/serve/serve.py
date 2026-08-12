"""
Main entry point for HydraServe.

Usage:
    python -m hydraserve.serve \
        --model /models/Qwen3.5-9B-AWQ \
        --mode pd_disaggregated \
        --prefill-gpu 0 --decode-gpu 1

Or programmatically:
    from hydraserve.serve import create_server
    server = create_server(config)
    server.start()
"""

import argparse
import logging
import sys
import time
from typing import Optional

import torch

from hydraserve.config import (
    HydraServeConfig, ServingMode, MODEL_SPECS,
    TransferConfig, CacheConfig, SchedulerConfig, RouterConfig,
)
from hydraserve.model import create_adapter
from hydraserve.model.adapter import ModelAdapter
from hydraserve.cache.block_manager import BlockManager
from hydraserve.cache.state_pool import StatePool
from hydraserve.cache.prefix_cache import PrefixCache
from hydraserve.transfer.backend import select_backend, TransferBackend
from hydraserve.transfer.pipeline import TransferPipeline
from hydraserve.engine.prefill_engine import PrefillEngine
from hydraserve.engine.decode_engine import DecodeEngine
from hydraserve.engine.scheduler import CentralScheduler
from hydraserve.router.cost_model import CostModel
from hydraserve.router.profiler import Profiler
from hydraserve.router.adaptive_router import AdaptiveRouter
from hydraserve.serve.api_server import serve

logger = logging.getLogger(__name__)


def create_server(config: HydraServeConfig):
    """
    Create a fully configured HydraServe server.

    This is the main initialization function that wires together
    all components: model, cache, transfer, engines, router, and API.
    """
    logger.info(f"Initializing HydraServe v0.1.0")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Mode: {config.mode.value}")
    logger.info(f"  Prefill GPU: {config.prefill_gpu}")
    logger.info(f"  Decode GPU: {config.decode_gpu}")
    logger.info(f"  Precision: {config.precision}")

    model_spec = config.model_spec
    logger.info(f"  Hidden size: {model_spec.hidden_size}")
    logger.info(f"  Layers: {model_spec.num_hidden_layers} "
                f"({model_spec.num_linear_attn_layers} linear + "
                f"{model_spec.num_full_attn_layers} full)")
    logger.info(f"  KV/token: {model_spec.get_kv_cache_size_per_token()} bytes")
    logger.info(f"  SSM state/req: {model_spec.get_ssm_state_size() / 1e6:.1f} MB")
    logger.info(f"  Weight size INT4: {model_spec.estimate_weight_size_int4():.1f} GB")

    # ─── Step 1: Load Model ───────────────────────────────────
    logger.info("Loading model...")
    prefill_device = torch.device(f"cuda:{config.prefill_gpu}")
    decode_device = torch.device(f"cuda:{config.decode_gpu}")

    model = create_adapter(
        model_name=config.model_name,
        model_path=config.model_path,
        device=prefill_device,
        precision=config.precision,
    )

    try:
        model.load_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load model from {config.model_path}: {e}")
        logger.warning("Running in mock mode (no actual model weights).")
        logger.warning("Set model_path to a valid HuggingFace model for full functionality.")

    # ─── Step 2: Initialize Cache ─────────────────────────────
    logger.info("Initializing dual-state memory management...")

    block_manager = BlockManager(
        model_spec=model_spec,
        cache_config=config.cache,
        device=decode_device,
    )
    logger.info(f"  Block manager: {block_manager.max_blocks} blocks "
                f"({block_manager.block_size} tokens each, "
                f"{block_manager.block_bytes / 1024:.0f} KB/block)")

    state_pool = StatePool(
        model_spec=model_spec,
        max_sequences=config.cache.max_num_seqs,
        device=decode_device,
    )
    state_pool.init_buffers()
    stats = state_pool.get_stats()
    logger.info(f"  State pool: {stats['total_slots']} slots "
                f"({stats['bytes_per_slot_mb']:.1f} MB/slot)")

    prefix_cache = None
    if config.cache.enable_prefix_cache:
        prefix_cache = PrefixCache(block_size=config.cache.block_size)
        logger.info(f"  Prefix cache: enabled (block_size={config.cache.block_size})")

    # ─── Step 3: Initialize Transfer ──────────────────────────
    logger.info("Initializing transfer backend...")

    if config.mode == ServingMode.PD_DISAGGREGATED:
        transfer_backend = select_backend(config.prefill_gpu, config.decode_gpu)
        logger.info(f"  Backend: {type(transfer_backend).__name__}")
        logger.info(f"  Bandwidth: {transfer_backend.get_bandwidth():.1f} GB/s")
        logger.info(f"  Mode: {transfer_backend.transfer_mode.value}")
        logger.info(f"  Layer pipeline: {transfer_backend.supports_layer_pipeline()}")
        logger.info(f"  Latency: {transfer_backend.get_latency():.3f} ms")

        transfer_pipeline = TransferPipeline(
            transfer_backend,
            use_quantization=(transfer_backend.transfer_mode.value == "quant")
        )
    else:
        transfer_backend = None
        transfer_pipeline = None
        logger.info("  No transfer needed (collocated mode)")

    # ─── Step 4: Initialize Engines ───────────────────────────
    logger.info("Initializing inference engines...")

    prefill_engine = PrefillEngine(
        model=model,
        config=config,
        block_manager=block_manager,
        state_pool=state_pool,
        transfer_backend=transfer_backend,
    )

    decode_engine = DecodeEngine(
        model=model,
        config=config,
        block_manager=block_manager,
        state_pool=state_pool,
        prefix_cache=prefix_cache,
    )

    # ─── Step 5: Initialize Router ────────────────────────────
    logger.info("Initializing adaptive router...")

    cost_model = CostModel(
        model_spec=model_spec,
        transfer_bandwidth_gb_s=(transfer_backend.get_bandwidth()
                                  if transfer_backend else 0),
    )

    # Run profiler to calibrate cost model
    profiler = Profiler(model=model, transfer_backend=transfer_backend)
    logger.info("Running micro-benchmarks for cost model calibration...")
    try:
        profile_results = profiler.run_all()
        cost_model.update_parameters(profile_results)
        logger.info(f"  Prefill speed: {cost_model.prefill_tokens_per_ms:.1f} tok/ms")
        logger.info(f"  Decode speed: {1.0 / cost_model.decode_tokens_per_ms:.1f} ms/tok")
        logger.info(f"  Transfer BW: {cost_model.transfer_bandwidth_gb_s:.1f} GB/s")
    except Exception as e:
        logger.warning(f"Profiling failed, using defaults: {e}")

    router = AdaptiveRouter(
        config=config.router,
        cost_model=cost_model,
    )

    # ─── Step 6: Initialize Scheduler ─────────────────────────
    logger.info("Initializing central scheduler...")

    scheduler = CentralScheduler(
        config=config,
        model=model,
        prefill_engine=prefill_engine,
        decode_engine=decode_engine,
        router=router,
    )

    # ─── Step 7: Create API Server ────────────────────────────
    logger.info(f"Starting API server on {config.host}:{config.port}...")

    return scheduler


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HydraServe: PD Disaggregated Inference Engine"
    )
    parser.add_argument("--model", type=str, default="/models/Qwen3.5-9B-AWQ",
                        help="Path to model weights")
    parser.add_argument("--model-name", type=str, default="Qwen3.5-9B",
                        choices=["Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.6-27B"])
    parser.add_argument("--mode", type=str, default="pd_disaggregated",
                        choices=["collocated", "pd_disaggregated", "dp"])
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--precision", type=str, default="int4",
                        choices=["int4", "bf16"])
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-seqs", type=int, default=256)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--no-first-token-seeding", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    config = HydraServeConfig(
        model_name=args.model_name,
        model_path=args.model,
        mode=ServingMode(args.mode),
        prefill_gpu=args.prefill_gpu,
        decode_gpu=args.decode_gpu,
        precision=args.precision,
        host=args.host,
        port=args.port,
        transfer=TransferConfig(
            first_token_seeding=not args.no_first_token_seeding,
        ),
        cache=CacheConfig(
            max_num_seqs=args.max_seqs,
            enable_prefix_cache=not args.no_prefix_cache,
        ),
        scheduler=SchedulerConfig(
            chunked_prefill_size=args.chunk_size,
        ),
    )

    scheduler = create_server(config)

    # Serve
    serve(config, scheduler)


if __name__ == "__main__":
    main()
