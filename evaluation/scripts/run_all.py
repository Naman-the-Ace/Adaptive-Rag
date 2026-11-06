"""
Master Evaluation Runner for Adaptive RAG
==========================================
Orchestrates all evaluation phases and generates a summary report.

Phases:
  1. Document ingestion
  2. Retrieval evaluation (baseline + with_rewriting)
  3. Routing evaluation
  4. Latency benchmarking
  5. Generation quality evaluation (optional, expensive)
  6. Summary report generation

Usage:
  cd adaptive_rag
  python ../evaluation/scripts/run_all.py \\
      --doc  ../evaluation/datasets/sample_docs/rag_concepts.txt \\
      --skip-generation   # Skip expensive LLM-as-judge calls

  # Full run including generation eval (uses LLM API calls):
  python ../evaluation/scripts/run_all.py \\
      --doc  ../evaluation/datasets/sample_docs/rag_concepts.txt \\
      --generation-sample 15
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_APP_DIR = _REPO_ROOT / "adaptive_rag"
_EVAL_DIR = _REPO_ROOT / "evaluation"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_script(script_name: str, extra_args: list[str] = None) -> tuple[int, float]:
    """Run an evaluation script as a subprocess and return (returncode, elapsed)."""
    script_path = str(_HERE / script_name)
    cmd = [sys.executable, script_path] + (extra_args or [])
    logger.info("Running: %s", " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(_APP_DIR),
        capture_output=False,
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        logger.error("Script '%s' failed with code %d", script_name, result.returncode)
    else:
        logger.info("Script '%s' completed in %.1fs", script_name, elapsed)
    return result.returncode, elapsed


def load_json_safe(path: str) -> dict:
    """Load JSON file, return empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def generate_summary_report(
    args,
    retrieval_agg: dict,
    retrieval_rw_agg: dict,
    routing_summary: dict,
    latency_data: dict,
    gen_agg: dict,
    run_start: str,
    elapsed_total: float,
):
    """Generate the evaluation_summary.md report."""
    report_path = _EVAL_DIR / "reports" / "evaluation_summary.md"
    os.makedirs(report_path.parent, exist_ok=True)

    k_values = retrieval_agg.get("k_values", [3, 5, 10])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Adaptive RAG — Evaluation Summary\n\n")
        f.write(f"**Run Date:** {run_start}  \n")
        f.write(f"**Total Evaluation Time:** {elapsed_total:.1f}s  \n")
        f.write(f"**Dataset:** `evaluation/datasets/golden_dataset.jsonl`  \n")
        f.write(f"**Documents Ingested:** {', '.join(args.doc)}  \n\n")

        f.write("---\n\n")
        f.write("## [WARN] Important Notes\n\n")
        f.write("- Retrieval metrics (Precision, Recall, MRR, nDCG) are **deterministic** and computed from ground-truth labels.\n")
        f.write("- Generation metrics (Faithfulness, Relevancy, etc.) are **LLM-as-judge** estimates — not independently verified.\n")
        f.write("- Document-route queries have `relevant_doc_ids = []` by default. Update the golden dataset with real doc IDs after uploading your documents to get valid retrieval metrics.\n")
        f.write("- Web-search route queries are excluded from retrieval/generation metrics because web results change over time.\n\n")

        f.write("---\n\n")
        f.write("## 1. Retrieval Quality\n\n")
        f.write("### Configuration: Baseline (no query rewriting)\n\n")
        if retrieval_agg:
            f.write(f"- **Queries evaluated:** {retrieval_agg.get('n_queries', 'N/A')}\n")
            f.write(f"- **Queries with no relevant labels:** {retrieval_agg.get('no_label_queries', 'N/A')}\n\n")
            f.write("| K | Precision@K | Recall@K | Hit Rate@K | nDCG@K |\n")
            f.write("|---|---|---|---|---|\n")
            for k in k_values:
                f.write(f"| {k} | {retrieval_agg.get(f'mean_precision@{k}', 'N/A')} | "
                        f"{retrieval_agg.get(f'mean_recall@{k}', 'N/A')} | "
                        f"{retrieval_agg.get(f'mean_hit_rate@{k}', 'N/A')} | "
                        f"{retrieval_agg.get(f'mean_ndcg@{k}', 'N/A')} |\n")
            f.write(f"\n- **MRR:** {retrieval_agg.get('mean_mrr', 'N/A')}\n")
            f.write(f"- **Mean Retrieval Latency:** {retrieval_agg.get('mean_retrieval_latency_s', 'N/A')}s\n\n")
        else:
            f.write("_Retrieval evaluation not completed or failed._\n\n")

        f.write("### Configuration: With Query Rewriting\n\n")
        if retrieval_rw_agg:
            f.write("| K | Precision@K | Recall@K | Hit Rate@K | nDCG@K |\n")
            f.write("|---|---|---|---|---|\n")
            for k in k_values:
                f.write(f"| {k} | {retrieval_rw_agg.get(f'mean_precision@{k}', 'N/A')} | "
                        f"{retrieval_rw_agg.get(f'mean_recall@{k}', 'N/A')} | "
                        f"{retrieval_rw_agg.get(f'mean_hit_rate@{k}', 'N/A')} | "
                        f"{retrieval_rw_agg.get(f'mean_ndcg@{k}', 'N/A')} |\n")
            f.write(f"\n- **MRR (with rewriting):** {retrieval_rw_agg.get('mean_mrr', 'N/A')}\n\n")
        else:
            f.write("_Not run._\n\n")

        f.write("---\n\n")
        f.write("## 2. Adaptive Routing\n\n")
        if routing_summary:
            f.write(f"- **Total queries:** {routing_summary.get('n_queries', 'N/A')}\n")
            f.write(f"- **Overall accuracy:** {routing_summary.get('accuracy', 'N/A')}\n")
            f.write(f"- **Rewrite rate:** {routing_summary.get('rewrite_rate', 'N/A')}\n")
            f.write(f"- **Error rate:** {routing_summary.get('error_rate', 'N/A')}\n\n")
            pr = routing_summary.get("per_route", {})
            if pr:
                f.write("| Route | Precision | Recall | F1 |\n")
                f.write("|---|---|---|---|\n")
                for route, m in pr.items():
                    f.write(f"| {route} | {m.get('precision', 'N/A')} | {m.get('recall', 'N/A')} | {m.get('f1', 'N/A')} |\n")
            f.write("\n")
        else:
            f.write("_Routing evaluation not completed._\n\n")

        f.write("---\n\n")
        f.write("## 3. Latency Benchmarking\n\n")
        if latency_data:
            f.write("| Component | Mean (s) | Median (s) | p95 (s) | p99 (s) |\n")
            f.write("|---|---|---|---|---|\n")
            for component, stats in latency_data.items():
                if isinstance(stats, dict) and stats.get("n"):
                    f.write(f"| {component.replace('_', ' ').title()} | "
                            f"{stats.get('mean', 'N/A')} | "
                            f"{stats.get('median', 'N/A')} | "
                            f"{stats.get('p95', 'N/A')} | "
                            f"{stats.get('p99', 'N/A')} |\n")
                else:
                    f.write(f"| {component.replace('_', ' ').title()} | NOT MEASURED | | | |\n")
            f.write("\n")
        else:
            f.write("_Latency benchmarking not completed._\n\n")

        f.write("---\n\n")
        f.write("## 4. Generation Quality (LLM-as-Judge)\n\n")
        f.write("> **Scoring model:** Same LLM used by the application (Groq llama-3.3-70b or Gemini gemini-2.0-flash)  \n")
        f.write("> Scores range 0.0–1.0. Values of -1.0 indicate the metric could not be measured.\n\n")
        if gen_agg:
            f.write(f"- **Queries evaluated:** {gen_agg.get('n_total', 'N/A')}\n")
            f.write(f"- **Faithfulness:** {gen_agg.get('mean_faithfulness', 'N/A')} (n={gen_agg.get('n_faithfulness', 0)})\n")
            f.write(f"- **Answer Relevancy:** {gen_agg.get('mean_answer_relevancy', 'N/A')} (n={gen_agg.get('n_answer_relevancy', 0)})\n")
            f.write(f"- **Answer Correctness:** {gen_agg.get('mean_answer_correctness', 'N/A')} (n={gen_agg.get('n_answer_correctness', 0)})\n")
            f.write(f"- **Context Precision:** {gen_agg.get('mean_context_precision', 'N/A')} (n={gen_agg.get('n_context_precision', 0)})\n")
            f.write(f"- **Context Recall:** {gen_agg.get('mean_context_recall', 'N/A')} (n={gen_agg.get('n_context_recall', 0)})\n")
            f.write(f"- **Hallucination Rate:** {gen_agg.get('hallucination_rate', 'N/A')}\n\n")
        else:
            f.write("_Generation evaluation not run. Use `--generation-sample N` to enable._\n\n")

        f.write("---\n\n")
        f.write("## 5. How to Reproduce\n\n")
        f.write("```bash\n")
        f.write("# 1. Install evaluation dependencies\n")
        f.write("cd adaptive_rag && pip install -r requirements.txt\n")
        f.write("pip install -r ../evaluation/requirements.txt\n\n")
        f.write("# 2. Configure API keys\n")
        f.write("cp .env.example .env  # then fill in GROQ_API_KEY / GEMINI_API_KEY / TAVILY_API_KEY\n\n")
        f.write("# 3. Run all evaluations\n")
        f.write("python ../evaluation/scripts/run_all.py \\\n")
        f.write("    --doc ../evaluation/datasets/sample_docs/rag_concepts.txt \\\n")
        f.write("    --generation-sample 20\n")
        f.write("```\n\n")
        f.write("## 6. Files Generated\n\n")
        f.write("| File | Contents |\n")
        f.write("|---|---|\n")
        f.write("| `reports/retrieval_results.csv` | Per-query retrieval metrics |\n")
        f.write("| `reports/retrieval_results_aggregate.json` | Aggregated retrieval metrics |\n")
        f.write("| `reports/routing_results.csv` | Per-query routing predictions |\n")
        f.write("| `reports/routing_results_summary.json` | Routing accuracy + confusion matrix |\n")
        f.write("| `reports/latency_results.csv` | Per-component latency stats |\n")
        f.write("| `reports/generation_results.csv` | Per-query generation scores |\n")
        f.write("| `reports/generation_results_aggregate.json` | Mean generation metrics |\n")
        f.write("| `reports/failure_analysis.md` | Low-scoring query analysis |\n")

    logger.info("Summary report saved to %s", report_path)
    return str(report_path)


def parse_args():
    p = argparse.ArgumentParser(description="Run all Adaptive RAG evaluations")
    p.add_argument("--dataset", default="evaluation/datasets/golden_dataset.jsonl")
    p.add_argument("--doc", nargs="+",
                   default=["evaluation/datasets/sample_docs/rag_concepts.txt"])
    p.add_argument("--k", nargs="+", type=int, default=[3, 5, 10])
    p.add_argument("--latency-runs", type=int, default=5)
    p.add_argument("--generation-sample", type=int, default=None,
                   help="Number of queries for generation eval (None = skip)")
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip generation evaluation (overrides --generation-sample)")
    p.add_argument("--skip-routing", action="store_true")
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--skip-llm-latency", action="store_true",
                   help="Skip LLM-dependent latency benchmarks")
    p.add_argument("--ragas", action="store_true",
                   help="Also attempt RAGAS evaluation (requires pip install ragas)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_start = datetime.now().isoformat(timespec="seconds")
    t_total_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Adaptive RAG Evaluation Framework")
    logger.info("Run started: %s", run_start)
    logger.info("=" * 60)

    # Build common args
    dataset_flag = ["--dataset", args.dataset]
    doc_flag = []
    for d in args.doc:
        doc_flag += ["--doc", d]
    k_flag = ["--k"] + [str(k) for k in args.k]
    verbose_flag = ["--verbose"] if args.verbose else []

    # Phase 1: Baseline retrieval
    logger.info("\n── Phase 1: Retrieval Evaluation (Baseline) ──")
    ret_code, ret_time = run_script("evaluate_retrieval.py",
        dataset_flag + doc_flag + k_flag + verbose_flag)

    # Phase 2: Retrieval with query rewriting
    logger.info("\n── Phase 2: Retrieval Evaluation (With Rewriting) ──")
    ret_rw_code, ret_rw_time = run_script("evaluate_retrieval.py",
        dataset_flag + doc_flag + k_flag + verbose_flag + ["--rewriting",
         "--output", "evaluation/reports/retrieval_results_rewriting.csv"])

    # Phase 3: Routing evaluation
    if not args.skip_routing:
        logger.info("\n── Phase 3: Routing Evaluation ──")
        route_code, route_time = run_script("evaluate_routing.py",
            dataset_flag + doc_flag + verbose_flag)
    else:
        logger.info("Skipping routing evaluation.")

    # Phase 4: Latency benchmarking
    if not args.skip_latency:
        logger.info("\n── Phase 4: Latency Benchmarking ──")
        lat_extra = ["--runs", str(args.latency_runs)]
        if args.skip_llm_latency:
            lat_extra.append("--skip-llm")
        lat_code, lat_time = run_script("benchmark_latency.py",
            doc_flag + lat_extra + verbose_flag)
    else:
        logger.info("Skipping latency benchmarking.")

    # Phase 5: Generation evaluation
    if not args.skip_generation and args.generation_sample:
        logger.info("\n── Phase 5: Generation Quality Evaluation ──")
        gen_extra = ["--sample", str(args.generation_sample)]
        if args.ragas:
            gen_extra.append("--ragas")
        gen_code, gen_time = run_script("evaluate_generation.py",
            dataset_flag + doc_flag + gen_extra + verbose_flag)
    else:
        logger.info("Skipping generation evaluation (use --generation-sample N to enable).")

    # Phase 6: Summary report
    logger.info("\n── Phase 6: Summary Report ──")
    ret_agg = load_json_safe(str(_EVAL_DIR / "reports/retrieval_results_aggregate.json"))
    ret_rw_agg = load_json_safe(str(_EVAL_DIR / "reports/retrieval_results_rewriting_aggregate.json"))
    routing_summary = load_json_safe(str(_EVAL_DIR / "reports/routing_results_summary.json"))
    latency_data = load_json_safe(str(_EVAL_DIR / "reports/latency_results.json"))
    gen_agg = load_json_safe(str(_EVAL_DIR / "reports/generation_results_aggregate.json"))

    elapsed_total = time.perf_counter() - t_total_start
    report_path = generate_summary_report(
        args, ret_agg, ret_rw_agg, routing_summary, latency_data, gen_agg,
        run_start, elapsed_total
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed_total:.1f}s")
    print(f"Summary report: {report_path}")
    print(f"Reports directory: {_EVAL_DIR / 'reports'}")


if __name__ == "__main__":
    main()
