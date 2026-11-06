"""
Retrieval Evaluation Script for Adaptive RAG
============================================
Computes Precision@K, Recall@K, Hit Rate@K, MRR, and nDCG@K.

Metric definitions:
  Precision@K  = |relevant ∩ retrieved@K| / K
  Recall@K     = |relevant ∩ retrieved@K| / |relevant|
  Hit Rate@K   = % queries with at least 1 relevant doc in top K
  MRR          = mean(1 / rank_of_first_relevant)
  nDCG@K       = DCG@K / IDCG@K

Usage:
  cd adaptive_rag
  python ../evaluation/scripts/evaluate_retrieval.py \\
      --dataset ../evaluation/datasets/golden_dataset.jsonl \\
      --doc    ../evaluation/datasets/sample_docs/rag_concepts.txt \\
      --k 3 5 10 \\
      --output ../evaluation/reports/retrieval_results.csv
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Allow running from either repo root or adaptive_rag directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_APP_DIR = _REPO_ROOT / "adaptive_rag"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# Load .env from repo root
from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Functions (fully deterministic, no LLM calls)
# ═══════════════════════════════════════════════════════════════════════════════


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-K retrieved items that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Fraction of all relevant items that appear in the top-K results.
    For document-level evaluation with chunked documents, counts distinct
    relevant document hits so recall is strictly bounded in [0.0, 1.0].
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & relevant_ids)
    return hits / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1 if at least one relevant item is in top-K, else 0."""
    top_k = retrieved_ids[:k]
    return float(any(doc_id in relevant_ids for doc_id in top_k))


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal of the rank of the first relevant item (0 if none found)."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Discounted Cumulative Gain at K using binary relevance.
    Deduplicates repeated document chunks so that subsequent chunks from an
    already-retrieved document do not artificially inflate the gain.
    """
    dcg = 0.0
    seen = set()
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids and doc_id not in seen:
            dcg += 1.0 / math.log2(i + 1)
            seen.add(doc_id)
    return dcg


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized DCG at K. IDCG computed assuming all relevant docs are at top."""
    actual_dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    # Ideal: relevant docs ranked first
    n_relevant = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


# ═══════════════════════════════════════════════════════════════════════════════
# Document Indexing
# ═══════════════════════════════════════════════════════════════════════════════


def ingest_document(doc_path: str) -> bool:
    """
    Ingest a document into the in-memory FAISS index using the app's pipeline.
    Returns True on success, False on failure.
    """
    try:
        import asyncio
        from src.rag.document_upload import documents as doc_upload

        with open(doc_path, "rb") as f:
            content = f.read()
        filename = os.path.basename(doc_path)
        logger.info("Ingesting '%s' into FAISS ...", filename)
        result = asyncio.run(doc_upload(content, filename, description="eval_doc"))
        logger.info("Ingestion result: %s", result)
        return result.get("status") == "indexed"
    except Exception as exc:
        logger.error("Document ingestion failed: %s", exc)
        return False


def get_faiss_retriever(k: int):
    """Get the FAISS retriever with specified K."""
    from src.rag.retriever_setup import _faiss_vectorstore
    if _faiss_vectorstore is None:
        return None
    return _faiss_vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_with_timing(query: str, k: int) -> tuple[list, float]:
    """
    Retrieve top-K docs for a query and return (docs, latency_seconds).
    """
    retriever = get_faiss_retriever(k)
    if retriever is None:
        return [], 0.0
    t0 = time.perf_counter()
    docs = retriever.invoke(query)
    latency = time.perf_counter() - t0
    return docs, latency


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Logic
# ═══════════════════════════════════════════════════════════════════════════════


def doc_id_from_retrieved(doc) -> str:
    """Extract a stable document identifier from a LangChain Document."""
    source = doc.metadata.get("source", "unknown")
    return source


def evaluate_retrieval(
    dataset_path: str,
    doc_paths: list[str],
    k_values: list[int],
    enable_rewriting: bool = False,
) -> list[dict]:
    """
    Run retrieval evaluation over all document-route queries.

    Returns:
        List of per-query result dicts with all metrics for each K value.
    """
    # Step 1: Ingest documents
    for dp in doc_paths:
        if not os.path.exists(dp):
            logger.warning("Document not found: %s — skipping", dp)
            continue
        ingest_document(dp)

    # Step 2: Load dataset — only doc-route queries with known relevant_doc_ids
    rows = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item["expected_route"] == "index":
                rows.append(item)

    logger.info("Loaded %d document-route queries for retrieval evaluation.", len(rows))

    max_k = max(k_values)
    results = []

    for item in rows:
        query = item["question"]
        relevant_docs = set(item.get("relevant_doc_ids", []))

        # Apply query rewriting if enabled
        effective_query = query
        rewrite_latency = 0.0
        was_rewritten = False
        if enable_rewriting:
            try:
                from src.rag.graph_builder import rewrite_query
                from src.models.state import State
                from langchain_core.messages import HumanMessage
                dummy_state = {
                    "messages": [HumanMessage(content=query)],
                    "route": "index",
                    "latest_query": query,
                    "documents": [],
                    "generation": "",
                    "session_id": "eval",
                }
                t_rw = time.perf_counter()
                new_state = rewrite_query(dummy_state)
                rewrite_latency = time.perf_counter() - t_rw
                effective_query = new_state["latest_query"]
                was_rewritten = (effective_query != query)
                logger.debug("Rewritten: '%s' → '%s'", query, effective_query)
            except Exception as exc:
                logger.warning("Query rewriting failed: %s", exc)

        # Retrieve at max K, then slice
        docs, retrieval_latency = retrieve_with_timing(effective_query, max_k)
        retrieved_ids = [doc_id_from_retrieved(d) for d in docs]

        row = {
            "query_id": item["query_id"],
            "question": query,
            "effective_query": effective_query,
            "was_rewritten": was_rewritten,
            "rewrite_latency_s": round(rewrite_latency, 4),
            "retrieval_latency_s": round(retrieval_latency, 4),
            "retrieved_docs": retrieved_ids,
            "relevant_docs": list(relevant_docs),
            "n_relevant_labeled": len(relevant_docs),
            "difficulty": item.get("difficulty", "unknown"),
            "category": item.get("category", "unknown"),
            "verified": item.get("verified", False),
        }

        # Compute metrics for each K value
        for k in k_values:
            prec = precision_at_k(retrieved_ids, relevant_docs, k)
            rec = recall_at_k(retrieved_ids, relevant_docs, k)
            hit = hit_rate_at_k(retrieved_ids, relevant_docs, k)
            rr = reciprocal_rank(retrieved_ids, relevant_docs)
            ndcg = ndcg_at_k(retrieved_ids, relevant_docs, k)
            row[f"precision@{k}"] = round(prec, 4)
            row[f"recall@{k}"] = round(rec, 4)
            row[f"hit_rate@{k}"] = round(hit, 4)
            row[f"mrr"] = round(rr, 4)
            row[f"ndcg@{k}"] = round(ndcg, 4)

        results.append(row)

    return results


def aggregate_metrics(results: list[dict], k_values: list[int]) -> dict:
    """Compute mean metrics across all query results."""
    n = len(results)
    if n == 0:
        return {}

    agg = {"n_queries": n}
    for k in k_values:
        for metric in [f"precision@{k}", f"recall@{k}", f"hit_rate@{k}", f"ndcg@{k}"]:
            vals = [r[metric] for r in results]
            agg[f"mean_{metric}"] = round(sum(vals) / n, 4)

    mrr_vals = [r["mrr"] for r in results]
    agg["mean_mrr"] = round(sum(mrr_vals) / n, 4)

    ret_lats = [r["retrieval_latency_s"] for r in results]
    agg["mean_retrieval_latency_s"] = round(sum(ret_lats) / n, 4)
    agg["median_retrieval_latency_s"] = round(sorted(ret_lats)[n // 2], 4)

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


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


def parse_args():
    p = argparse.ArgumentParser(description="Retrieval evaluation for Adaptive RAG")
    p.add_argument("--dataset", default="evaluation/datasets/golden_dataset.jsonl")
    p.add_argument("--doc", nargs="+",
                   default=["evaluation/datasets/sample_docs/rag_concepts.txt"],
                   help="Document(s) to ingest into FAISS")
    p.add_argument("--k", nargs="+", type=int, default=[3, 5, 10])
    p.add_argument("--output", default="evaluation/reports/retrieval_results.csv")
    p.add_argument("--config", default="evaluation/config/eval_config.yaml")
    p.add_argument("--rewriting", action="store_true",
                   help="Enable query rewriting before retrieval")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    k_values = args.k or cfg.get("retrieval", {}).get("k_values", [3, 5, 10])
    logger.info("K values: %s | Query rewriting: %s", k_values, args.rewriting)

    # Resolve doc paths
    doc_paths = [resolve_path(d) for d in args.doc]
    dataset_path = resolve_path(args.dataset)

    # Run baseline evaluation
    config_label = "with_rewriting" if args.rewriting else "baseline"
    logger.info("Running retrieval evaluation: config='%s'", config_label)
    results = evaluate_retrieval(dataset_path, doc_paths, k_values, enable_rewriting=args.rewriting)
    agg = aggregate_metrics(results, k_values)

    # Determine which queries have no labeled relevant docs
    no_label_count = sum(1 for r in results if r["n_relevant_labeled"] == 0)
    logger.warning(
        "%d/%d queries have no labeled relevant_doc_ids. "
        "Retrieval metrics will be 0 for these. "
        "Please verify golden_dataset.jsonl after uploading real documents.",
        no_label_count, len(results)
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVALUATION RESULTS — config={config_label}")
    print("=" * 60)
    print(f"Queries evaluated: {agg.get('n_queries', 0)}")
    print(f"Queries with no relevant labels: {no_label_count}")
    for k in k_values:
        print(f"\n  K = {k}")
        print(f"    Precision@{k}:  {agg.get(f'mean_precision@{k}', 'N/A')}")
        print(f"    Recall@{k}:     {agg.get(f'mean_recall@{k}', 'N/A')}")
        print(f"    Hit Rate@{k}:   {agg.get(f'mean_hit_rate@{k}', 'N/A')}")
        print(f"    nDCG@{k}:       {agg.get(f'mean_ndcg@{k}', 'N/A')}")
    print(f"\n  MRR:            {agg.get('mean_mrr', 'N/A')}")
    print(f"  Mean Retrieval Latency: {agg.get('mean_retrieval_latency_s', 'N/A')}s")
    print(f"  Median Retrieval Latency: {agg.get('median_retrieval_latency_s', 'N/A')}s")

    # Save CSV
    output_path = resolve_path(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        fieldnames.append("config")
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                r["config"] = config_label
                writer.writerow(r)
        logger.info("Saved per-query results to %s", output_path)

    # Save aggregate summary
    agg_path = output_path.replace(".csv", "_aggregate.json")
    agg["config"] = config_label
    agg["k_values"] = k_values
    agg["query_rewriting_enabled"] = args.rewriting
    agg["no_label_queries"] = no_label_count
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    logger.info("Saved aggregate results to %s", agg_path)

    print(f"\n[OK] Results saved to: {output_path}")
    print(f"[OK] Aggregate saved to: {agg_path}")
    print("\n[NOTE] Queries with empty relevant_doc_ids score 0 on all metrics.")
    print("       After uploading your real documents, update relevant_doc_ids in the dataset.")


if __name__ == "__main__":
    main()
