"""
Routing Evaluation Script for Adaptive RAG
==========================================
Measures the accuracy of LangGraph query routing against expected route labels.

Metrics:
  - Overall routing accuracy
  - Per-route Precision, Recall, F1
  - Confusion matrix
  - Query rewriting frequency
  - Document grader rejection rate
  - Web search fallback frequency

Usage:
  cd adaptive_rag
  python ../evaluation/scripts/evaluate_routing.py \\
      --dataset ../evaluation/datasets/golden_dataset.jsonl \\
      --doc    ../evaluation/datasets/sample_docs/rag_concepts.txt \\
      --output ../evaluation/reports/routing_results.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

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

ROUTES = ["index", "general", "search"]


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Functions
# ═══════════════════════════════════════════════════════════════════════════════


def compute_confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute confusion matrix as a nested dict {true_label: {pred_label: count}}."""
    matrix = {r: {c: 0 for c in ROUTES} for r in ROUTES}
    for true, pred in zip(y_true, y_pred):
        if true in matrix and pred in ROUTES:
            matrix[true][pred] += 1
    return matrix


def per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute precision, recall, F1 per class (route)."""
    results = {}
    for route in ROUTES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == route and p == route)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != route and p == route)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == route and p != route)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        results[route] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph Instrumented Runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_query_with_instrumentation(question: str) -> dict:
    """
    Run a query through the LangGraph pipeline and collect routing metadata.
    Returns a dict with route, documents, generation, timing, and grading info.
    """
    from src.rag.graph_builder import builder
    from langchain_core.messages import HumanMessage

    input_state = {
        "messages": [HumanMessage(content=question)],
        "route": "",
        "latest_query": question,
        "documents": [],
        "generation": "",
        "session_id": "eval_routing",
    }

    t0 = time.perf_counter()
    try:
        result = builder.invoke(input_state)
        latency = time.perf_counter() - t0
        return {
            "predicted_route": result.get("route", "unknown"),
            "generation": result.get("generation", ""),
            "n_docs": len(result.get("documents", [])),
            "latest_query": result.get("latest_query", question),
            "was_rewritten": result.get("latest_query", question) != question,
            "latency_s": round(latency, 4),
            "error": None,
        }
    except Exception as exc:
        latency = time.perf_counter() - t0
        logger.error("Query failed: %s — %s", question[:60], exc)
        return {
            "predicted_route": "error",
            "generation": "",
            "n_docs": 0,
            "latest_query": question,
            "was_rewritten": False,
            "latency_s": round(latency, 4),
            "error": str(exc),
        }


def ingest_document(doc_path: str) -> bool:
    """Ingest a document into the in-memory FAISS index."""
    try:
        import asyncio
        from src.rag.document_upload import documents as doc_upload
        with open(doc_path, "rb") as f:
            content = f.read()
        filename = os.path.basename(doc_path)
        result = asyncio.run(doc_upload(content, filename))
        return result.get("status") == "indexed"
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Main Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_routing(dataset_path: str, doc_paths: list[str]) -> list[dict]:
    """
    Run routing evaluation for all queries in the dataset.
    Returns per-query results.
    """
    # Ingest documents for FAISS index
    for dp in doc_paths:
        if os.path.exists(dp):
            ingest_document(dp)
        else:
            logger.warning("Document not found: %s", dp)

    rows = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    logger.info("Running routing evaluation on %d queries ...", len(rows))
    results = []

    for item in rows:
        question = item["question"]
        expected_route = item["expected_route"]
        logger.info("[%s] %s", item["query_id"], question[:70])

        run_result = run_query_with_instrumentation(question)
        predicted_route = run_result["predicted_route"]
        correct = (predicted_route == expected_route)

        results.append({
            "query_id": item["query_id"],
            "question": question,
            "expected_route": expected_route,
            "predicted_route": predicted_route,
            "correct": correct,
            "was_rewritten": run_result["was_rewritten"],
            "n_docs_retrieved": run_result["n_docs"],
            "latency_s": run_result["latency_s"],
            "error": run_result["error"],
            "category": item.get("category", ""),
            "difficulty": item.get("difficulty", ""),
            "generation_snippet": run_result["generation"][:100] if run_result["generation"] else "",
        })

    return results


def print_confusion_matrix(matrix: dict):
    """Pretty-print confusion matrix."""
    header = f"{'':>10}" + "".join(f"{r:>10}" for r in ROUTES)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(header)
    for true_route in ROUTES:
        row = f"{true_route:>10}" + "".join(
            f"{matrix[true_route][pred]:>10}" for pred in ROUTES
        )
        print(row)


def main():
    p = argparse.ArgumentParser(description="Routing evaluation for Adaptive RAG")
    p.add_argument("--dataset", default="evaluation/datasets/golden_dataset.jsonl")
    p.add_argument("--doc", nargs="+",
                   default=["evaluation/datasets/sample_docs/rag_concepts.txt"])
    p.add_argument("--output", default="evaluation/reports/routing_results.csv")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dataset_path = resolve_path(args.dataset)
    doc_paths = [resolve_path(d) for d in args.doc]
    output_path = resolve_path(args.output)

    results = evaluate_routing(dataset_path, doc_paths)

    y_true = [r["expected_route"] for r in results]
    y_pred = [r["predicted_route"] for r in results]
    n = len(results)

    # Accuracy
    correct_count = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct_count / n if n > 0 else 0.0

    # Per-class metrics
    class_metrics = per_class_metrics(y_true, y_pred)

    # Confusion matrix
    confusion = compute_confusion_matrix(y_true, y_pred)

    # Rewriting frequency
    rewrite_count = sum(1 for r in results if r["was_rewritten"])

    # Error rate
    error_count = sum(1 for r in results if r["error"])

    # Print results
    print("\n" + "=" * 60)
    print("ROUTING EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total queries:   {n}")
    print(f"Correct routes:  {correct_count}")
    print(f"Accuracy:        {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Rewrite triggered: {rewrite_count}/{n} ({rewrite_count/n*100:.1f}%)")
    print(f"Errors:          {error_count}")

    print("\nPer-Route Metrics:")
    print(f"{'Route':>10} {'Precision':>12} {'Recall':>10} {'F1':>8}")
    for route in ROUTES:
        m = class_metrics[route]
        print(f"{route:>10} {m['precision']:>12.4f} {m['recall']:>10.4f} {m['f1']:>8.4f}")

    print_confusion_matrix(confusion)

    # Save CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved per-query results to %s", output_path)

    # Save summary JSON
    summary = {
        "n_queries": n,
        "accuracy": round(accuracy, 4),
        "rewrite_rate": round(rewrite_count / n, 4) if n else 0,
        "error_rate": round(error_count / n, 4) if n else 0,
        "per_route": class_metrics,
        "confusion_matrix": confusion,
    }
    summary_path = output_path.replace(".csv", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    print(f"\n[OK] Per-query results: {output_path}")
    print(f"[OK] Summary: {summary_path}")


if __name__ == "__main__":
    main()
