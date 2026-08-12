"""
Benchmark visualization using matplotlib.

Generates the 8 key plots specified in the design doc (§6.9):
  1. Throughput vs concurrency curve
  2. TPOT CDF cumulative distribution
  3. TTFT histogram (burst arrival)
  4. Context length vs P99 TPOT
  5. Routing decision distribution
  6. Multi-model comparison
  7. Transfer hiding effectiveness
  8. Accuracy comparison bar chart
"""

from typing import Dict, List, Optional, Tuple
import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ─── Color Palette ──────────────────────────────────────────────────

CONFIG_COLORS = {
    "A (1-GPU Collocated)": "#94a3b8",    # Slate (reference)
    "B (DP)": "#3b82f6",                  # Blue (baseline)
    "C (TP=2)": "#f59e0b",                # Amber
    "D (PD Separated)": "#10b981",        # Emerald (HydraServe)
    "E (vLLM unified)": "#8b5cf6",        # Violet
    "F (Intra-GPU MPS)": "#ec4899",       # Pink
}

MODEL_COLORS = {
    "Qwen3.5-4B": "#93c5fd",
    "Qwen3.5-9B": "#3b82f6",
    "Qwen3.6-27B": "#1e40af",
}


def setup_style():
    """Configure matplotlib style."""
    plt.rcParams.update({
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'grid.alpha': 0.3,
    })


def plot_throughput_vs_concurrency(
    results: Dict[str, Dict[int, float]],  # config → {concurrency: tok/s}
    output_path: str = "throughput_vs_concurrency.png",
    title: str = "Throughput vs Concurrency (9B, 32K)",
):
    """Plot 1: Throughput vs concurrency curve."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for config_name, data in results.items():
        concurrency = sorted(data.keys())
        throughput = [data[c] for c in concurrency]
        color = CONFIG_COLORS.get(config_name, "#64748b")
        linestyle = '--' if '1-GPU' in config_name else '-'
        alpha = 0.6 if '1-GPU' in config_name else 1.0
        ax.plot(concurrency, throughput, 'o-', label=config_name,
                color=color, linestyle=linestyle, alpha=alpha,
                markersize=6, linewidth=2)

    ax.set_xlabel("Concurrent Requests", fontweight='medium')
    ax.set_ylabel("Throughput (tokens/s)", fontweight='medium')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Annotate crossover point
    midpoint = (min(concurrency) + max(concurrency)) // 2
    ax.axvline(x=midpoint, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.annotate('crossover\n~10-15 concurrent',
                xy=(midpoint, ax.get_ylim()[1] * 0.3),
                fontsize=8, color='red', alpha=0.7,
                ha='center')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_tpot_cdf(
    results: Dict[str, List[float]],  # config → list of TPOT values
    output_path: str = "tpot_cdf.png",
    title: str = "TPOT CDF (30 Concurrent)",
):
    """Plot 2: TPOT cumulative distribution function."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for config_name, values in results.items():
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        color = CONFIG_COLORS.get(config_name, "#64748b")
        ax.plot(sorted_vals, cdf, label=config_name, color=color, linewidth=2)

    ax.set_xlabel("TPOT (ms)", fontweight='medium')
    ax.set_ylabel("Cumulative Probability", fontweight='medium')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    # P50, P90, P99 vertical lines
    for pct, ls, alpha in [(0.50, '--', 0.5), (0.90, '-.', 0.4), (0.99, ':', 0.3)]:
        ax.axhline(y=pct, color='gray', linestyle=ls, alpha=alpha)
        ax.text(ax.get_xlim()[1] * 0.02, pct, f'P{int(pct*100)}',
                fontsize=7, va='bottom', ha='left', alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_ttft_histogram(
    results: Dict[str, List[float]],  # config → list of TTFT values
    output_path: str = "ttft_histogram.png",
    title: str = "TTFT Distribution (Burst Arrival, 5x32K)",
):
    """Plot 3: TTFT histogram for burst arrival."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = 30
    for config_name, values in results.items():
        color = CONFIG_COLORS.get(config_name, "#64748b")
        ax.hist(values, bins=bins, alpha=0.6, label=config_name, color=color,
                edgecolor='white', linewidth=0.5)

    ax.set_xlabel("TTFT (ms)", fontweight='medium')
    ax.set_ylabel("Count", fontweight='medium')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_context_vs_p99_tpot(
    results: Dict[str, Dict[int, float]],  # config → {context_len: p99_tpot}
    output_path: str = "context_vs_p99_tpot.png",
    title: str = "Context Length vs P99 TPOT",
):
    """Plot 4: Context length vs P99 TPOT."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for config_name, data in results.items():
        ctx_lens = sorted(data.keys())
        p99s = [data[c] for c in ctx_lens]
        color = CONFIG_COLORS.get(config_name, "#64748b")
        ax.plot(ctx_lens, p99s, 's-', label=config_name, color=color,
                markersize=8, linewidth=2)

    ax.set_xlabel("Context Length (tokens)", fontweight='medium')
    ax.set_ylabel("P99 TPOT (ms)", fontweight='medium')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_routing_distribution(
    prompt_lengths: List[int],
    decisions: List[str],
    output_path: str = "routing_distribution.png",
    title: str = "Routing Decision Distribution",
):
    """Plot 5: Routing decision distribution by prompt length."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    collocated = [(pl, 1) for pl, d in zip(prompt_lengths, decisions) if d == "collocated"]
    pd_sep = [(pl, 1) for pl, d in zip(prompt_lengths, decisions) if d == "pd_disaggregated"]

    if collocated:
        x_c, _ = zip(*collocated)
        ax.scatter(x_c, [1] * len(x_c), c='#3b82f6', alpha=0.5, s=30,
                   label='Collocated', marker='o')
    if pd_sep:
        x_p, _ = zip(*pd_sep)
        ax.scatter(x_p, [2] * len(x_p), c='#10b981', alpha=0.5, s=30,
                   label='PD Separated', marker='s')

    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Collocated', 'PD Separated'])
    ax.set_xlabel("Prompt Length (tokens)", fontweight='medium')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3, axis='x')

    # Add threshold line
    ax.axvline(x=2048, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=8192, color='gray', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_multimodel_comparison(
    results: Dict[str, Dict[str, float]],  # model → {config: p99_tpot}
    output_path: str = "multimodel_comparison.png",
    title: str = "Multi-Model P99 TPOT Comparison (32K)",
):
    """Plot 6: Multi-model comparison."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(results.keys())
    configs = list(results[models[0]].keys())

    x = np.arange(len(models))
    width = 0.35
    colors = ['#3b82f6', '#10b981']

    for i, config_name in enumerate(configs):
        values = [results[m][config_name] for m in models]
        bars = ax.bar(x + i * width, values, width, label=config_name,
                      color=colors[i % len(colors)], edgecolor='white')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel("P99 TPOT (ms)", fontweight='medium')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_transfer_hiding(
    prefill_times: List[float],
    transfer_times: List[float],
    context_lengths: List[int],
    output_path: str = "transfer_hiding.png",
    title: str = "Transfer Hiding Effectiveness",
):
    """Plot 7: Transfer time vs prefill time by context length."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(context_lengths, 0, prefill_times, alpha=0.3,
                    color='#3b82f6', label='Prefill Time')
    ax.plot(context_lengths, prefill_times, 'o-', color='#3b82f6', linewidth=2,
            markersize=6)
    ax.plot(context_lengths, transfer_times, 's-', color='#f59e0b', linewidth=2,
            markersize=6, label='Transfer Time')

    # Uncovered transfer (when transfer > prefill)
    uncovered = [max(0, t - p) for t, p in zip(transfer_times, prefill_times)]
    ax.fill_between(context_lengths, prefill_times,
                    [p + u for p, u in zip(prefill_times, uncovered)],
                    alpha=0.3, color='#ef4444', label='Uncovered Transfer')

    ax.set_xlabel("Context Length (tokens)", fontweight='medium')
    ax.set_ylabel("Time (ms)", fontweight='medium')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate('100% hidden\n(NVLink)', xy=(8000, prefill_times[2]),
                fontsize=8, ha='center', color='#3b82f6')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_accuracy_comparison(
    results: Dict[str, Dict[str, float]],  # benchmark → {config: accuracy}
    output_path: str = "accuracy_comparison.png",
    title: str = "Accuracy Comparison",
):
    """Plot 8: Accuracy comparison bar chart."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    benchmarks = list(results.keys())
    configs = list(results[benchmarks[0]].keys())
    x = np.arange(len(benchmarks))
    width = 0.8 / len(configs)
    colors = ['#3b82f6', '#10b981', '#8b5cf6']

    for i, config_name in enumerate(configs):
        values = [results[b][config_name] for b in benchmarks]
        bars = ax.bar(x + i * width, values, width, label=config_name,
                      color=colors[i % len(colors)], edgecolor='white')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Accuracy / Score", fontweight='medium')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=max(0, min(min(results[b][c] for c in configs)
                                   for b in benchmarks) - 5))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_all_plots(plot_data: Dict, output_dir: str = "benchmark_output/") -> None:
    """Generate all 8 plots from benchmark results."""
    os.makedirs(output_dir, exist_ok=True)

    if "throughput" in plot_data:
        plot_throughput_vs_concurrency(
            plot_data["throughput"],
            os.path.join(output_dir, "1_throughput_vs_concurrency.png")
        )
    if "tpot_values" in plot_data:
        plot_tpot_cdf(
            plot_data["tpot_values"],
            os.path.join(output_dir, "2_tpot_cdf.png")
        )
    if "ttft_values" in plot_data:
        plot_ttft_histogram(
            plot_data["ttft_values"],
            os.path.join(output_dir, "3_ttft_histogram.png")
        )
    if "context_p99" in plot_data:
        plot_context_vs_p99_tpot(
            plot_data["context_p99"],
            os.path.join(output_dir, "4_context_vs_p99_tpot.png")
        )
    if "routing_data" in plot_data:
        plot_routing_distribution(
            plot_data["routing_data"]["prompt_lengths"],
            plot_data["routing_data"]["decisions"],
            os.path.join(output_dir, "5_routing_distribution.png")
        )
    if "multimodel" in plot_data:
        plot_multimodel_comparison(
            plot_data["multimodel"],
            os.path.join(output_dir, "6_multimodel_comparison.png")
        )
    if "transfer_data" in plot_data:
        plot_transfer_hiding(
            plot_data["transfer_data"]["prefill_times"],
            plot_data["transfer_data"]["transfer_times"],
            plot_data["transfer_data"]["context_lengths"],
            os.path.join(output_dir, "7_transfer_hiding.png")
        )
    if "accuracy" in plot_data:
        plot_accuracy_comparison(
            plot_data["accuracy"],
            os.path.join(output_dir, "8_accuracy_comparison.png")
        )
