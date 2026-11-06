"""
Generation Quality Evaluation Script for Adaptive RAG
======================================================
Evaluates LLM-generated answers using LLM-as-judge scoring.

Metrics:
  - Faithfulness: Are all claims in the answer supported by context?
  - Answer Relevancy: Does the answer address the question?
  - Answer Correctness: Does the answer match the reference answer?
  - Context Precision: Fraction of retrieved context that is relevant
  - Context Recall: Does context contain info needed to answer?
  - Hallucination Rate: Fraction of queries with unfaithful answers

This script uses an LLM-as-judge approach (not RAGAS library directly,
to avoid dependency conflicts). All scores are clearly labeled as
LLM-as-judge and not deterministic.

RAGAS integration note:
  RAGAS (ragas>=0.1.0) can be used for the same metrics with:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, ...
  Requires: pip install ragas
  Add --ragas flag to use it if installed.

Usage:
  cd adaptive_rag
  python ../evaluation/scripts/evaluate_generation.py \\
      --dataset ../evaluation/datasets/golden_dataset.jsonl \\
      --doc    ../evaluation/datasets/sample_docs/rag_concepts.txt \\
      --sample 20 \\
      --output ../evaluation/reports/generation_results.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
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


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-as-Judge Scoring
# ═══════════════════════════════════════════════════════════════════════════════


FAITHFULNESS_PROMPT = """You are an evaluation judge.

Question: {question}
Retrieved Context:
{context}
Generated Answer: {answer}

Task: Check if every claim in the Generated Answer is explicitly supported by the Retrieved Context.
Score on a scale from 0.0 to 1.0:
- 1.0: All claims are fully supported by the context
- 0.5: Some claims are supported, some are not
- 0.0: Most or all claims are not supported (hallucination)

Respond with only a JSON object: {{"score": <float 0-1>, "reasoning": "<brief explanation>"}}"""


RELEVANCY_PROMPT = """You are an evaluation judge.

Question: {question}
Generated Answer: {answer}

Task: Evaluate whether the Generated Answer directly and meaningfully addresses the Question.
Score on a scale from 0.0 to 1.0:
- 1.0: Answer is directly and completely relevant to the question
- 0.5: Answer is partially relevant or misses some aspects
- 0.0: Answer is off-topic or does not address the question

Respond with only a JSON object: {{"score": <float 0-1>, "reasoning": "<brief explanation>"}}"""


CORRECTNESS_PROMPT = """You are an evaluation judge.

Question: {question}
Reference Answer: {reference}
Generated Answer: {answer}

Task: Compare the Generated Answer against the Reference Answer for factual correctness.
Score on a scale from 0.0 to 1.0:
- 1.0: Generated answer is factually identical or equivalent to reference
- 0.5: Generated answer is partially correct with some errors or gaps
- 0.0: Generated answer is factually incorrect or contradicts the reference

If Reference Answer is "ABSTAIN" or "REQUIRES_WEB_SEARCH", score 1.0 if the generated answer 
appropriately says it cannot answer or provides a reasonable response.

Respond with only a JSON object: {{"score": <float 0-1>, "reasoning": "<brief explanation>"}}"""


CONTEXT_PRECISION_PROMPT = """You are an evaluation judge.

Question: {question}
Retrieved Context Chunks:
{context_chunks}

Task: Evaluate what fraction of the retrieved context chunks are relevant to answering the question.
Score on a scale from 0.0 to 1.0:
- 1.0: All retrieved chunks are highly relevant to the question
- 0.5: About half the chunks are relevant
- 0.0: Most or all chunks are irrelevant to the question

Respond with only a JSON object: {{"score": <float 0-1>, "reasoning": "<brief explanation>"}}"""


CONTEXT_RECALL_PROMPT = """You are an evaluation judge.

Question: {question}
Reference Answer: {reference}
Retrieved Context:
{context}

Task: Evaluate whether the Retrieved Context contains all the information needed to answer the question
based on what the Reference Answer says.
Score on a scale from 0.0 to 1.0:
- 1.0: Context fully contains all information needed for the reference answer
- 0.5: Context contains some but not all needed information
- 0.0: Context is missing most information needed to give the reference answer

Respond with only a JSON object: {{"score": <float 0-1>, "reasoning": "<brief explanation>"}}"""


def parse_llm_score(response_text: str) -> tuple[float, str]:
    """Parse JSON score response from LLM judge."""
    import re
    # Try to extract JSON from the response
    json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            score = float(obj.get("score", 0.0))
            reasoning = obj.get("reasoning", "")
            return max(0.0, min(1.0, score)), reasoning
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: look for a float
    float_match = re.search(r'\b([01]\.\d+|[01])\b', response_text)
    if float_match:
        return max(0.0, min(1.0, float(float_match.group()))), "parsed from text"
    return 0.0, "parse_failed"


def llm_judge_score(prompt: str) -> tuple[float, str]:
    """Call the LLM to get a score from 0 to 1."""
    try:
        from src.llms.gemini import llm
        result = llm.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)
        return parse_llm_score(content)
    except Exception as exc:
        logger.error("LLM judge call failed: %s", exc)
        return -1.0, f"error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Pipeline Runner (modified to also return documents)
# ═══════════════════════════════════════════════════════════════════════════════


def run_rag_query(question: str) -> dict:
    """Run the full RAG pipeline and return generation + retrieved docs."""
    from src.rag.graph_builder import builder
    from langchain_core.messages import HumanMessage

    input_state = {
        "messages": [HumanMessage(content=question)],
        "route": "",
        "latest_query": question,
        "documents": [],
        "generation": "",
        "session_id": "eval_gen",
    }
    t0 = time.perf_counter()
    try:
        result = builder.invoke(input_state)
        latency = time.perf_counter() - t0
        docs = result.get("documents", [])
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        context_chunks = [d.page_content for d in docs]
        return {
            "generation": result.get("generation", ""),
            "route": result.get("route", ""),
            "context": context,
            "context_chunks": context_chunks,
            "latency_s": round(latency, 4),
            "error": None,
        }
    except Exception as exc:
        latency = time.perf_counter() - t0
        return {
            "generation": "",
            "route": "error",
            "context": "",
            "context_chunks": [],
            "latency_s": round(latency, 4),
            "error": str(exc),
        }


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


# ═══════════════════════════════════════════════════════════════════════════════
# Optional RAGAS Integration
# ═══════════════════════════════════════════════════════════════════════════════


def try_ragas_evaluate(samples: list[dict]) -> Optional[dict]:
    """
    Attempt RAGAS evaluation if installed.
    Returns metric dict or None if RAGAS unavailable.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        ragas_data = {
            "question": [s["question"] for s in samples],
            "answer": [s["generation"] for s in samples],
            "contexts": [s["context_chunks"] for s in samples],
            "ground_truth": [s["reference_answer"] for s in samples],
        }
        dataset = Dataset.from_dict(ragas_data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return dict(result)
    except ImportError:
        logger.info("RAGAS not installed — skipping RAGAS evaluation")
        return None
    except Exception as exc:
        logger.warning("RAGAS evaluation failed: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_generation(
    dataset_path: str,
    doc_paths: list[str],
    sample_size: Optional[int] = None,
    use_ragas: bool = False,
) -> list[dict]:
    """
    Run generation evaluation.
    Returns per-query result dicts with all LLM-judge scores.
    """
    # Ingest documents
    for dp in doc_paths:
        if os.path.exists(dp):
            ingest_document(dp)

    # Load dataset
    rows = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                # Only evaluate doc and general routes (skip search — web results vary)
                if item["expected_route"] in ("index", "general"):
                    rows.append(item)

    if sample_size and sample_size < len(rows):
        import random
        random.seed(42)
        rows = random.sample(rows, sample_size)

    logger.info("Evaluating generation quality on %d queries ...", len(rows))
    results = []

    for i, item in enumerate(rows, 1):
        question = item["question"]
        reference = item.get("reference_answer", "")
        logger.info("[%d/%d] %s", i, len(rows), question[:70])

        # Get RAG pipeline output
        run_result = run_rag_query(question)
        generation = run_result["generation"]
        context = run_result["context"]
        context_chunks = run_result["context_chunks"]

        if run_result["error"]:
            logger.warning("Query failed: %s", run_result["error"])
            results.append({
                "query_id": item["query_id"],
                "question": question,
                "reference_answer": reference,
                "generation": "",
                "route": "error",
                "faithfulness": -1.0,
                "answer_relevancy": -1.0,
                "answer_correctness": -1.0,
                "context_precision": -1.0,
                "context_recall": -1.0,
                "faithfulness_reasoning": run_result["error"],
                "relevancy_reasoning": "",
                "correctness_reasoning": "",
                "context_precision_reasoning": "",
                "context_recall_reasoning": "",
                "generation_latency_s": run_result["latency_s"],
                "error": run_result["error"],
                "category": item.get("category", ""),
            })
            continue

        # Score faithfulness
        faith_score, faith_reason = (0.0, "no context") if not context else llm_judge_score(
            FAITHFULNESS_PROMPT.format(
                question=question, context=context[:3000], answer=generation[:2000]
            )
        )

        # Score answer relevancy
        rel_score, rel_reason = llm_judge_score(
            RELEVANCY_PROMPT.format(question=question, answer=generation[:2000])
        )

        # Score correctness (only if reference is not a placeholder)
        if reference and "ABSTAIN" not in reference and "REQUIRES_WEB" not in reference:
            corr_score, corr_reason = llm_judge_score(
                CORRECTNESS_PROMPT.format(
                    question=question, reference=reference, answer=generation[:2000]
                )
            )
        else:
            corr_score, corr_reason = -1.0, "no_reference_answer"

        # Score context precision
        if context_chunks:
            chunks_text = "\n---\n".join(f"[Chunk {i+1}]: {c[:500]}"
                                         for i, c in enumerate(context_chunks))
            cprec_score, cprec_reason = llm_judge_score(
                CONTEXT_PRECISION_PROMPT.format(question=question, context_chunks=chunks_text)
            )
        else:
            cprec_score, cprec_reason = -1.0, "no_context"

        # Score context recall
        if context and reference and "ABSTAIN" not in reference:
            crec_score, crec_reason = llm_judge_score(
                CONTEXT_RECALL_PROMPT.format(
                    question=question, reference=reference, context=context[:3000]
                )
            )
        else:
            crec_score, crec_reason = -1.0, "no_context_or_reference"

        results.append({
            "query_id": item["query_id"],
            "question": question,
            "reference_answer": reference[:200] if reference else "",
            "generation": generation[:300],
            "route": run_result["route"],
            "faithfulness": round(faith_score, 4),
            "answer_relevancy": round(rel_score, 4),
            "answer_correctness": round(corr_score, 4),
            "context_precision": round(cprec_score, 4),
            "context_recall": round(crec_score, 4),
            "faithfulness_reasoning": faith_reason[:200],
            "relevancy_reasoning": rel_reason[:200],
            "correctness_reasoning": corr_reason[:200],
            "context_precision_reasoning": cprec_reason[:200],
            "context_recall_reasoning": crec_reason[:200],
            "generation_latency_s": run_result["latency_s"],
            "error": "",
            "category": item.get("category", ""),
        })

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    return results


def aggregate_generation_metrics(results: list[dict]) -> dict:
    """Compute mean generation metrics (excluding -1.0 sentinel values)."""
    metrics = ["faithfulness", "answer_relevancy", "answer_correctness",
               "context_precision", "context_recall"]
    agg = {"n_total": len(results)}
    for metric in metrics:
        valid_scores = [r[metric] for r in results if r[metric] >= 0.0]
        if valid_scores:
            agg[f"mean_{metric}"] = round(sum(valid_scores) / len(valid_scores), 4)
            agg[f"n_{metric}"] = len(valid_scores)
            # Hallucination rate = fraction with faithfulness < 0.5
            if metric == "faithfulness":
                hallucinated = sum(1 for s in valid_scores if s < 0.5)
                agg["hallucination_rate"] = round(hallucinated / len(valid_scores), 4)
        else:
            agg[f"mean_{metric}"] = None
            agg[f"n_{metric}"] = 0
    return agg


def main():
    p = argparse.ArgumentParser(description="Generation evaluation for Adaptive RAG")
    p.add_argument("--dataset", default="evaluation/datasets/golden_dataset.jsonl")
    p.add_argument("--doc", nargs="+",
                   default=["evaluation/datasets/sample_docs/rag_concepts.txt"])
    p.add_argument("--sample", type=int, default=None,
                   help="Number of queries to evaluate (None = all)")
    p.add_argument("--output", default="evaluation/reports/generation_results.csv")
    p.add_argument("--ragas", action="store_true",
                   help="Also attempt RAGAS evaluation (requires: pip install ragas datasets)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dataset_path = resolve_path(args.dataset)
    doc_paths = [resolve_path(d) for d in args.doc]
    output_path = resolve_path(args.output)

    results = evaluate_generation(dataset_path, doc_paths, args.sample, args.ragas)
    agg = aggregate_generation_metrics(results)

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION EVALUATION RESULTS (LLM-as-Judge)")
    print("=" * 60)
    print("[WARN]  All scores are LLM-as-judge estimates, not deterministic.")
    print(f"Queries evaluated: {agg['n_total']}")
    print(f"\n  Faithfulness:        {agg.get('mean_faithfulness', 'N/A')} (n={agg.get('n_faithfulness', 0)})")
    print(f"  Answer Relevancy:    {agg.get('mean_answer_relevancy', 'N/A')} (n={agg.get('n_answer_relevancy', 0)})")
    print(f"  Answer Correctness:  {agg.get('mean_answer_correctness', 'N/A')} (n={agg.get('n_answer_correctness', 0)})")
    print(f"  Context Precision:   {agg.get('mean_context_precision', 'N/A')} (n={agg.get('n_context_precision', 0)})")
    print(f"  Context Recall:      {agg.get('mean_context_recall', 'N/A')} (n={agg.get('n_context_recall', 0)})")
    print(f"  Hallucination Rate:  {agg.get('hallucination_rate', 'N/A')}")

    # Failure analysis
    failures = [r for r in results if r.get("faithfulness", 1.0) < 0.5 and r.get("faithfulness", -1) >= 0]
    if failures:
        print(f"\nTop failure cases (faithfulness < 0.5):")
        for f in failures[:3]:
            print(f"  [{f['query_id']}] {f['question'][:60]}")
            print(f"    Generation: {f['generation'][:100]}...")
            print(f"    Reason: {f['faithfulness_reasoning'][:100]}")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    agg_path = output_path.replace(".csv", "_aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    # Failure analysis report
    fail_path = str(_REPO_ROOT / "evaluation/reports/failure_analysis.md")
    with open(fail_path, "w", encoding="utf-8") as f:
        f.write("# Failure Analysis Report\n\n")
        f.write(f"Generated from: {len(results)} evaluated queries\n\n")
        f.write("## Low Faithfulness Cases (< 0.5)\n\n")
        low_faith = [r for r in results if 0 <= r.get("faithfulness", -1) < 0.5]
        if low_faith:
            for r in low_faith:
                f.write(f"### [{r['query_id']}] {r['question']}\n")
                f.write(f"- **Faithfulness Score:** {r['faithfulness']}\n")
                f.write(f"- **Reasoning:** {r['faithfulness_reasoning']}\n")
                f.write(f"- **Generated Answer:** {r['generation'][:300]}\n\n")
        else:
            f.write("No low faithfulness cases found.\n\n")

        f.write("## Low Relevancy Cases (< 0.5)\n\n")
        low_rel = [r for r in results if 0 <= r.get("answer_relevancy", -1) < 0.5]
        if low_rel:
            for r in low_rel:
                f.write(f"### [{r['query_id']}] {r['question']}\n")
                f.write(f"- **Relevancy Score:** {r['answer_relevancy']}\n")
                f.write(f"- **Reasoning:** {r['relevancy_reasoning']}\n\n")
        else:
            f.write("No low relevancy cases found.\n")

    print(f"\n[OK] Per-query results: {output_path}")
    print(f"[OK] Aggregate results: {agg_path}")
    print(f"[OK] Failure analysis: {fail_path}")
    print("\nSCORING MODEL: Same LLM as the application (Groq/Gemini)")
    print("All scores labeled as LLM-as-judge — not independently verified ground truth.")


if __name__ == "__main__":
    main()
