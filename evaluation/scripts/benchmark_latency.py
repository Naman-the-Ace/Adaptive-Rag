"""
Latency Benchmarking Script for Adaptive RAG
=============================================
Measures per-node and end-to-end latency statistics.

Reported statistics per node:
  mean, median (p50), p90, p95, p99, min, max

Nodes measured:
  1. Embedding generation
  2. FAISS retrieval
  3. Query classification (LLM call)
  4. Document grading (LLM call per doc)
  5. Query rewriting (LLM call)
  6. LLM generation (final answer)
  7. End-to-end pipeline

Usage:
  cd adaptive_rag
  python ../evaluation/scripts/benchmark_latency.py \\
      --doc  ../evaluation/datasets/sample_docs/rag_concepts.txt \\
      --runs 10 \\
      --output ../evaluation/reports/latency_results.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
import statistics

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_APP_DIR = _REPO_ROOT / "adaptive_rag"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def latency_stats(values: list[float]) -> dict:
    """Compute descriptive statistics for a list of latency values (seconds)."""
    if not values:
        return {k: None for k in ["mean", "median", "p90", "p95", "p99", "min", "max", "n"]}
    n = len(values)
    sorted_vals = sorted(values)

    def percentile(p):
        idx = max(0, min(n - 1, int(round(p / 100 * n)) - 1))
        return sorted_vals[idx]

    return {
        "n": n,
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "p90": round(percentile(90), 4),
        "p95": round(percentile(95), 4),
        "p99": round(percentile(99), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "stdev": round(statistics.stdev(values) if n > 1 else 0.0, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Component Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════


BENCHMARK_QUERIES = [
    "What is Retrieval-Augmented Generation?",
    "How does FAISS perform similarity search?",
    "What is the difference between semantic and keyword search?",
    "Explain vector embeddings in machine learning.",
    "What is query rewriting in RAG systems?",
    "How is chat history managed in this system?",
    "What file formats can be uploaded to the RAG system?",
    "What is faithfulness in LLM evaluation?",
    "What is 2 + 2?",
    "What is LangGraph and how does it work?",
]


def ingest_document(doc_path: str) -> bool:
    """Ingest a document file into FAISS."""
    try:
        import asyncio
        from src.rag.document_upload import documents as doc_upload
        with open(doc_path, "rb") as f:
            content = f.read()
        result = asyncio.run(doc_upload(content, os.path.basename(doc_path)))
        return result.get("status") == "indexed"
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        return False


def benchmark_embedding(texts: list[str], runs: int = 5) -> dict:
    """Benchmark embedding generation latency."""
    from src.llms.gemini import embeddings

    times = []
    for query in texts[:runs]:
        t0 = time.perf_counter()
        try:
            embeddings.embed_query(query)
            times.append(time.perf_counter() - t0)
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
    return latency_stats(times)


def benchmark_faiss_retrieval(queries: list[str], k: int = 4) -> dict:
    """Benchmark FAISS similarity search latency (excludes embedding time)."""
    from src.rag.retriever_setup import _faiss_vectorstore
    from src.llms.gemini import embeddings

    if _faiss_vectorstore is None:
        logger.warning("FAISS not initialized — retrieval latency not measured")
        return latency_stats([])

    times = []
    for query in queries:
        try:
            query_vec = embeddings.embed_query(query)
            t0 = time.perf_counter()
            _faiss_vectorstore.similarity_search_by_vector(query_vec, k=k)
            times.append(time.perf_counter() - t0)
        except Exception as exc:
            logger.warning("FAISS retrieval failed: %s", exc)
    return latency_stats(times)


def benchmark_query_classification(queries: list[str]) -> dict:
    """Benchmark query_classifier node latency (includes one retrieval + LLM call)."""
    from src.rag.graph_builder import query_classifier
    from langchain_core.messages import HumanMessage

    times = []
    for query in queries:
        state = {
            "messages": [HumanMessage(content=query)],
            "route": "",
            "latest_query": query,
            "documents": [],
            "generation": "",
            "session_id": "bench",
        }
        t0 = time.perf_counter()
        try:
            query_classifier(state)
            times.append(time.perf_counter() - t0)
        except Exception as exc:
            logger.warning("Classification failed: %s", exc)
    return latency_stats(times)


def benchmark_document_grading(queries: list[str], k: int = 4) -> dict:
    """Benchmark grade_docs latency (K LLM calls per query)."""
    from src.rag.graph_builder import grade_docs
    from src.rag.retriever_setup import get_retriever

    retriever = get_retriever()
    if retriever is None:
        logger.warning("No retriever — grading latency not measured")
        return latency_stats([])

    times = []
    for query in queries:
        try:
            docs = retriever.invoke(query)
            state = {
                "messages": [],
                "route": "index",
                "latest_query": query,
                "documents": docs,
                "generation": "",
                "session_id": "bench",
            }
            t0 = time.perf_counter()
            grade_docs(state)
            times.append(time.perf_counter() - t0)
        except Exception as exc:
            logger.warning("Grading failed: %s", exc)
    return latency_stats(times)


def benchmark_query_rewriting(queries: list[str]) -> dict:
    """Benchmark rewrite_query node latency (1 LLM call per query)."""
    from src.rag.graph_builder import rewrite_query
    from langchain_core.messages import HumanMessage

    times = []
    for query in queries:
        state = {
            "messages": [HumanMessage(content=query)],
            "route": "index",
            "latest_query": query,
            "documents": [],
            "generation": "",
            "session_id": "bench",
        }
        t0 = time.perf_counter()
        try:
            rewrite_query(state)
            times.append(time.perf_counter() - t0)
        except Exception as exc:
            logger.warning("Rewriting failed: %s", exc)
    return latency_stats(times)


def benchmark_generation(queries: list[str]) -> dict:
    """Benchmark generate_answer latency (1 LLM call with retrieved context)."""
    from src.rag.graph_builder import generate_answer
    from src.rag.retriever_setup import get_retriever
    from langchain_core.messages import HumanMessage

    retriever = get_retriever()
    times = []
    for query in queries:
        docs = retriever.invoke(query) if retriever else []
        state = {
            "messages": [HumanMessage(content=query)],
            "route": "index",
            "latest_query": query,
            "documents": docs,
            "generation": "",
            "session_id": "bench",
        }
        t0 = time.perf_counter()
        try:
            generate_answer(state)
            times.append(time.perf_counter() - t0)
        except Exception as exc:
            logger.warning("Generation failed: %s", exc)
    return latency_stats(times)


def benchmark_end_to_end(queries: list[str]) -> dict:
    """Benchmark complete pipeline end-to-end latency."""
    from src.rag.graph_builder import builder
    from langchain_core.messages import HumanMessage

    times = []
    for query in queries:
        input_state = {
            "messages": [HumanMessage(content=query)],
            "route": "",
            "latest_query": query,
            "documents": [],
            "generation": "",
            "session_id": "bench_e2e",
        }
        t0 = time.perf_counter()
        try:
            builder.invoke(input_state)
            times.append(time.perf_counter() - t0)
        except Exception as exc:
            logger.warning("E2E run failed: %s", exc)
    return latency_stats(times)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def print_stats_table(label: str, stats: dict):
    """Pretty-print latency statistics."""
    if stats.get("n") is None or stats.get("n") == 0:
        print(f"  {label}: NOT MEASURED")
        return
    print(f"  {label} (n={stats['n']}):")
    print(f"    mean={stats['mean']}s  median={stats['median']}s  "
          f"p90={stats['p90']}s  p95={stats['p95']}s  "
          f"p99={stats['p99']}s  min={stats['min']}s  max={stats['max']}s")


def resolve_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    cand = (_REPO_ROOT / path).resolve()
    if cand.exists():
        return str(cand)
    return str((_REPO_ROOT / p.lstrip("./\\")).resolve())


def main():
    p = argparse.ArgumentParser(description="Latency benchmarking for Adaptive RAG")
    p.add_argument("--doc", nargs="+",
                   default=["evaluation/datasets/sample_docs/rag_concepts.txt"])
    p.add_argument("--runs", type=int, default=5,
                   help="Number of measurement runs (after warmup)")
    p.add_argument("--warmup", type=int, default=1,
                   help="Number of warmup runs to discard")
    p.add_argument("--output", default="evaluation/reports/latency_results.csv")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--skip-llm", action="store_true",
                   help="Skip LLM-dependent benchmarks (classification, grading, etc.)")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    doc_paths = [resolve_path(d) for d in args.doc]
    output_path = resolve_path(args.output)

    # Ingest documents
    for dp in doc_paths:
        if os.path.exists(dp):
            ingest_document(dp)

    total_runs = args.warmup + args.runs
    queries_all = (BENCHMARK_QUERIES * ((total_runs // len(BENCHMARK_QUERIES)) + 1))[:total_runs]
    queries_warm = queries_all[args.warmup:]  # measurement queries only

    logger.info("Warmup runs: %d | Measurement runs: %d", args.warmup, args.runs)

    print("\n" + "=" * 60)
    print("LATENCY BENCHMARKING — Adaptive RAG")
    print("=" * 60)
    print(f"Warmup: {args.warmup} runs | Measured: {args.runs} runs")

    all_stats = {}

    # 1. Embedding latency
    logger.info("Benchmarking embedding generation ...")
    # Run warmup first
    _ = benchmark_embedding(queries_all[:args.warmup])
    emb_stats = benchmark_embedding(queries_warm, runs=args.runs)
    all_stats["embedding_generation"] = emb_stats
    print_stats_table("Embedding Generation", emb_stats)

    # 2. FAISS retrieval latency (excludes embedding)
    logger.info("Benchmarking FAISS retrieval ...")
    _ = benchmark_faiss_retrieval(queries_all[:args.warmup])
    faiss_stats = benchmark_faiss_retrieval(queries_warm)
    all_stats["faiss_retrieval"] = faiss_stats
    print_stats_table("FAISS Retrieval (excl. embedding)", faiss_stats)

    if not args.skip_llm:
        # 3. Query classification latency
        logger.info("Benchmarking query classification ...")
        _ = benchmark_query_classification(queries_all[:args.warmup])
        class_stats = benchmark_query_classification(queries_warm)
        all_stats["query_classification"] = class_stats
        print_stats_table("Query Classification", class_stats)

        # 4. Document grading latency
        logger.info("Benchmarking document grading ...")
        _ = benchmark_document_grading(queries_all[:args.warmup])
        grade_stats = benchmark_document_grading(queries_warm)
        all_stats["document_grading"] = grade_stats
        print_stats_table("Document Grading", grade_stats)

        # 5. Query rewriting latency
        logger.info("Benchmarking query rewriting ...")
        _ = benchmark_query_rewriting(queries_all[:args.warmup])
        rewrite_stats = benchmark_query_rewriting(queries_warm)
        all_stats["query_rewriting"] = rewrite_stats
        print_stats_table("Query Rewriting", rewrite_stats)

        # 6. LLM generation latency
        logger.info("Benchmarking LLM generation ...")
        _ = benchmark_generation(queries_all[:args.warmup])
        gen_stats = benchmark_generation(queries_warm)
        all_stats["llm_generation"] = gen_stats
        print_stats_table("LLM Generation", gen_stats)

        # 7. End-to-end latency
        logger.info("Benchmarking end-to-end pipeline ...")
        _ = benchmark_end_to_end(queries_all[:args.warmup])
        e2e_stats = benchmark_end_to_end(queries_warm)
        all_stats["end_to_end"] = e2e_stats
        print_stats_table("End-to-End Pipeline", e2e_stats)
    else:
        for key in ["query_classification", "document_grading",
                    "query_rewriting", "llm_generation", "end_to_end"]:
            all_stats[key] = {"n": 0, "note": "skipped via --skip-llm"}
            print(f"  {key}: SKIPPED (--skip-llm flag)")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for component, stats in all_stats.items():
        row = {"component": component}
        row.update({k: v for k, v in (stats or {}).items()})
        rows.append(row)

    if rows:
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        fieldnames = ["component"] + sorted(all_keys - {"component"})
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved latency results to %s", output_path)

    json_path = output_path.replace(".csv", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n[OK] Latency results: {output_path}")
    print(f"[OK] JSON results:   {json_path}")
    print("\nNOTE: Token counts and API cost estimation require vendor-specific")
    print("      usage metadata from the LLM response object. Use the LangChain")
    print("      callback system (get_openai_callback equivalent) for cost tracking.")


if __name__ == "__main__":
    main()
