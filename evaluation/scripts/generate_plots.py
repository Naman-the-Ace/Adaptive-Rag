"""
Visualization Script for Adaptive RAG Evaluation Results
=========================================================
Generates charts from actual evaluation results CSV/JSON files.

Charts generated:
  1. Precision@K and Recall@K bar chart comparison (baseline vs rewriting)
  2. MRR and nDCG@K comparison
  3. Routing confusion matrix heatmap
  4. Generation metric distribution (box plot)
  5. Latency per-component bar chart with error bars

Usage:
  python evaluation/scripts/generate_plots.py \\
      --reports evaluation/reports/ \\
      --output  evaluation/plots/

NOTE: Charts are only generated if the corresponding CSV/JSON result files exist.
      Run the evaluation scripts first.
"""

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed — pip install matplotlib")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed — pip install pandas")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_REPORTS_DIR = _REPO_ROOT / "evaluation" / "reports"
_PLOTS_DIR = _REPO_ROOT / "evaluation" / "plots"


def load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_csv(path: Path):
    if not HAS_PANDAS:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def plot_retrieval_comparison(reports_dir: Path, plots_dir: Path):
    """Bar chart: Precision@K and Recall@K, baseline vs rewriting."""
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        logger.warning("Skipping retrieval comparison plot (missing matplotlib/pandas)")
        return

    baseline = load_json(reports_dir / "retrieval_results_aggregate.json")
    rewriting = load_json(reports_dir / "retrieval_results_rewriting_aggregate.json")

    if not baseline:
        logger.warning("No baseline retrieval aggregate found — skipping plot")
        return

    k_values = baseline.get("k_values", [3, 5, 10])
    metrics = ["precision", "recall", "hit_rate", "ndcg"]
    metric_labels = ["Precision@K", "Recall@K", "Hit Rate@K", "nDCG@K"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Retrieval Metrics: Baseline vs With Query Rewriting", fontsize=14)

    for ax, metric, label in zip(axes.flatten(), metrics, metric_labels):
        baseline_vals = [baseline.get(f"mean_{metric}@{k}", 0) for k in k_values]
        rewriting_vals = [rewriting.get(f"mean_{metric}@{k}", 0) for k in k_values] if rewriting else []

        x = np.arange(len(k_values))
        width = 0.35

        ax.bar(x - width/2, baseline_vals, width, label="Baseline", color="#4472C4", alpha=0.85)
        if rewriting_vals:
            ax.bar(x + width/2, rewriting_vals, width, label="With Rewriting", color="#ED7D31", alpha=0.85)

        ax.set_xlabel("K")
        ax.set_ylabel("Score")
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in k_values])
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Annotate values
        for i, v in enumerate(baseline_vals):
            ax.text(x[i] - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        if rewriting_vals:
            for i, v in enumerate(rewriting_vals):
                ax.text(x[i] + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    out_path = plots_dir / "retrieval_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved retrieval comparison plot: %s", out_path)


def plot_mrr_ndcg(reports_dir: Path, plots_dir: Path):
    """Bar chart: MRR and nDCG@K comparison."""
    if not HAS_MATPLOTLIB:
        return

    baseline = load_json(reports_dir / "retrieval_results_aggregate.json")
    rewriting = load_json(reports_dir / "retrieval_results_rewriting_aggregate.json")

    if not baseline:
        return

    k_values = baseline.get("k_values", [3, 5, 10])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MRR and nDCG Comparison")

    # MRR (single value per config)
    ax = axes[0]
    configs = ["Baseline"]
    mrr_vals = [baseline.get("mean_mrr", 0)]
    if rewriting:
        configs.append("With Rewriting")
        mrr_vals.append(rewriting.get("mean_mrr", 0))
    colors = ["#4472C4", "#ED7D31"]
    bars = ax.bar(configs, mrr_vals, color=colors[:len(configs)], alpha=0.85)
    for bar, val in zip(bars, mrr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.4f}", ha="center")
    ax.set_title("Mean Reciprocal Rank (MRR)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("MRR")
    ax.grid(axis="y", alpha=0.3)

    # nDCG@K
    ax = axes[1]
    x = np.arange(len(k_values))
    width = 0.35
    baseline_ndcg = [baseline.get(f"mean_ndcg@{k}", 0) for k in k_values]
    ax.bar(x - width/2, baseline_ndcg, width, label="Baseline", color="#4472C4", alpha=0.85)
    if rewriting:
        rw_ndcg = [rewriting.get(f"mean_ndcg@{k}", 0) for k in k_values]
        ax.bar(x + width/2, rw_ndcg, width, label="With Rewriting", color="#ED7D31", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.set_title("nDCG@K")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = plots_dir / "mrr_ndcg_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved MRR/nDCG plot: %s", out_path)


def plot_routing_confusion_matrix(reports_dir: Path, plots_dir: Path):
    """Heatmap of routing confusion matrix."""
    if not HAS_MATPLOTLIB:
        return

    routing = load_json(reports_dir / "routing_results_summary.json")
    if not routing or "confusion_matrix" not in routing:
        logger.warning("No routing summary found — skipping confusion matrix")
        return

    matrix = routing["confusion_matrix"]
    routes = ["index", "general", "search"]
    data = [[matrix.get(r, {}).get(c, 0) for c in routes] for r in routes]

    fig, ax = plt.subplots(figsize=(7, 6))
    if HAS_SEABORN:
        import seaborn as sns
        sns.heatmap(
            data, annot=True, fmt="d", cmap="Blues",
            xticklabels=routes, yticklabels=routes, ax=ax
        )
    else:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(routes)))
        ax.set_yticks(range(len(routes)))
        ax.set_xticklabels(routes)
        ax.set_yticklabels(routes)
        for i in range(len(routes)):
            for j in range(len(routes)):
                ax.text(j, i, str(data[i][j]), ha="center", va="center", fontsize=12)
        plt.colorbar(im, ax=ax)

    accuracy = routing.get("accuracy", "N/A")
    ax.set_title(f"Routing Confusion Matrix\n(Accuracy: {accuracy})")
    ax.set_xlabel("Predicted Route")
    ax.set_ylabel("True Route")

    plt.tight_layout()
    out_path = plots_dir / "routing_confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix: %s", out_path)


def plot_generation_metrics(reports_dir: Path, plots_dir: Path):
    """Box plot of generation quality score distributions."""
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        return

    df = load_csv(reports_dir / "generation_results.csv")
    if df is None or df.empty:
        logger.warning("No generation results CSV — skipping generation plot")
        return

    metrics = ["faithfulness", "answer_relevancy", "answer_correctness",
               "context_precision", "context_recall"]
    # Filter to valid scores only
    valid_dfs = {m: df[df[m] >= 0][m] for m in metrics if m in df.columns}
    if not valid_dfs:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    data_to_plot = [valid_dfs[m].tolist() for m in metrics if m in valid_dfs]
    labels = [m.replace("_", " ").title() for m in metrics if m in valid_dfs]

    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
    colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000", "#7030A0"]
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Generation Quality Score Distributions (LLM-as-Judge)")
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, label="Good threshold (0.7)")
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Warning threshold (0.5)")
    ax.legend()

    plt.tight_layout()
    out_path = plots_dir / "generation_quality_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved generation distribution plot: %s", out_path)


def plot_latency_comparison(reports_dir: Path, plots_dir: Path):
    """Bar chart with error bars for per-component latency."""
    if not HAS_MATPLOTLIB:
        return

    latency = load_json(reports_dir / "latency_results.json")
    if not latency:
        logger.warning("No latency data found — skipping latency plot")
        return

    components = []
    means = []
    p95s = []

    for comp, stats in latency.items():
        if isinstance(stats, dict) and stats.get("mean") is not None:
            components.append(comp.replace("_", "\n").title())
            means.append(stats["mean"])
            p95s.append(stats.get("p95", stats["mean"]))

    if not components:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(components))
    bars = ax.bar(x, means, color="#4472C4", alpha=0.85, label="Mean")
    ax.bar(x, [p - m for p, m in zip(p95s, means)], bottom=means,
           color="#ED7D31", alpha=0.6, label="Mean → p95")

    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=9)
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Component Latency: Mean and p95")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (m, p) in enumerate(zip(means, p95s)):
        ax.text(i, p + 0.01, f"{m:.3f}s", ha="center", fontsize=8)

    plt.tight_layout()
    out_path = plots_dir / "latency_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved latency comparison plot: %s", out_path)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate evaluation plots")
    p.add_argument("--reports", default="evaluation/reports")
    p.add_argument("--output", default="evaluation/plots")
    args = p.parse_args()

    reports_dir = _REPO_ROOT / args.reports if not os.path.isabs(args.reports) else Path(args.reports)
    plots_dir = _REPO_ROOT / args.output if not os.path.isabs(args.output) else Path(args.output)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib is required for plotting.")
        print("  pip install matplotlib seaborn pandas numpy")
        sys.exit(1)

    plot_retrieval_comparison(reports_dir, plots_dir)
    plot_mrr_ndcg(reports_dir, plots_dir)
    plot_routing_confusion_matrix(reports_dir, plots_dir)
    plot_generation_metrics(reports_dir, plots_dir)
    plot_latency_comparison(reports_dir, plots_dir)

    print(f"\n[OK] Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
