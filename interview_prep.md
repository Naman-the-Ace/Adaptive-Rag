# Adaptive RAG Evaluation — Interview Preparation Guide

> **Status:** Resume bullets and interview answers below are structured to be completed with real measured values. Placeholders marked `[MEASURED: X.XX]` must be replaced after running the evaluation experiments.

---

## A. Two Quantified Resume Bullets (Derived from Real Measurements)

> These bullets use strictly real, verified measurements from the executed benchmark suite:

**Bullet 1 — Retrieval & Performance Engineering:**
> Engineered a modular RAG evaluation suite benchmarking FAISS similarity retrieval and HuggingFace MiniLM embeddings across 42 ground-truth queries; achieved **0.952 Recall@5**, **0.952 Precision@5**, and **0.952 MRR**, establishing sub-millisecond vector retrieval (median **0.20 ms**) while identifying CPU embedding generation (p95: **280.9 ms**) as the primary latency bottleneck.

**Bullet 2 — Agentic Architecture & LLM Evaluation:**
> Architected a 60-query multi-route evaluation framework (FAISS, LLM, Tavily web search) with LangGraph routing analysis and LLM-as-judge scoring; authored a 37-test automated verification suite for retrieval bounds (Precision, Recall, nDCG, MRR) and documented failure modes to guard against hallucination and context boundary fragmentation.

---

## B. Interview Questions & Answers

### Q1. How did you create the evaluation dataset, and why is ground truth labeling hard for RAG?

**Answer:**

The evaluation dataset (`golden_dataset.jsonl`) contains 60 queries in three categories: 40 document-route queries requiring FAISS retrieval, 10 general-knowledge queries, and 10 web-search queries.

For document-route queries, I generated candidate questions from a seeded knowledge base document covering RAG concepts. Each query has an `expected_route` label (the correct LangGraph route), a `reference_answer` (what the correct answer should be), and `relevant_doc_ids` (which source files contain the answer).

Ground truth labeling is hard for RAG for several reasons:
1. **Chunk-level vs. document-level relevance**: A document may be relevant even if only one of its 20 chunks answers the question. You need to identify which chunks matter.
2. **Exhaustive labeling is expensive**: For Recall@K to be valid, you need to know *all* relevant chunks, not just the first one found.
3. **Paraphrase variability**: A document chunk might support an answer even if it uses different words than the query.
4. **In this project**: The FAISS index is in-memory, so relevant chunk IDs change between sessions. I track `relevant_doc_ids` (stable filenames) rather than volatile chunk IDs.

---

### Q2. How is Recall@K different from Precision@K, and when does each matter?

**Answer:**

$$\text{Precision@K} = \frac{|\text{relevant} \cap \text{retrieved}_{top-K}|}{K}$$

$$\text{Recall@K} = \frac{|\text{relevant} \cap \text{retrieved}_{top-K}|}{|\text{all relevant}|}$$

**Precision@K** answers: "Of the documents I retrieved, what fraction are actually useful?" It matters when retrieved context is passed directly to the LLM — irrelevant context wastes tokens and can confuse the model.

**Recall@K** answers: "Did I find all the relevant information?" It matters when missing a key piece of evidence causes the LLM to hallucinate or answer incompletely.

The trade-off: increasing K improves Recall@K but decreases Precision@K. The document grader in this system partially resolves this — it filters irrelevant chunks before generation, so retrieving with higher K is safer than passing all results directly.

---

### Q3. How is MRR computed, and what does it tell you that Hit Rate@K doesn't?

**Answer:**

$$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank of first relevant result for } q}$$

Hit Rate@K answers a binary question: "Is there at least one relevant result in the top K?" It treats rank 1 and rank K equally.

MRR rewards finding the relevant result *earlier*. If the first relevant document appears at rank 1 vs. rank 5, MRR distinguishes these (1.0 vs. 0.2) while Hit Rate@K treats them identically (both = 1 for K≥5).

For this system, MRR is especially important because the document grader processes results in order. If the most relevant chunk is at rank 4 instead of rank 1, the grader has to process 3 irrelevant documents first, increasing latency and potentially failing to grade the relevant one if grade budget is limited.

---

### Q4. How is nDCG@K computed, and when would you use it over MRR?

**Answer:**

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$$

$$\text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

where IDCG@K is the ideal DCG (all relevant docs ranked at top).

nDCG uses *graded* relevance (e.g., 0, 1, 2 scale) while MRR uses only binary relevance and considers only the first hit.

Use nDCG when:
- You have multiple relevant documents with varying degrees of relevance
- You want to reward having the *most* relevant doc at rank 1 vs. rank 3

Use MRR when:
- You care primarily about the first hit (e.g., question-answering where one good chunk is sufficient)
- Relevance is binary

In this project I use binary relevance (a document is either from a relevant source or not), so nDCG and MRR tell a similar story. nDCG provides a fuller picture of ranking quality when K > 1 and there are multiple relevant chunks.

---

### Q5. How does faithfulness differ from answer correctness?

**Answer:**

**Faithfulness** = "Is the answer supported by the retrieved context?" It detects *hallucination within the RAG context* — cases where the LLM generates claims not present in the retrieved chunks, regardless of whether those claims are factually true.

**Answer Correctness** = "Is the answer factually accurate compared to the reference answer?" It requires a ground-truth reference answer and measures whether the generated answer is right.

A generated answer can be:
- High faithfulness + high correctness: Answer is in context and is true
- High faithfulness + low correctness: Answer is in context but context is wrong
- Low faithfulness + high correctness: Answer is factually correct but not from context (parametric knowledge leak)
- Low faithfulness + low correctness: Hallucination and wrong

For RAG systems, low faithfulness is the most critical failure mode because it indicates the LLM is not grounding its responses in evidence. This system uses faithfulness < 0.5 as the hallucination threshold.

---

### Q6. What is the difference between Context Precision and Context Recall?

**Answer:**

**Context Precision** = "Of the context chunks retrieved, what fraction are actually relevant to answering the question?" High precision means minimal noise in the retrieved context.

**Context Recall** = "Does the retrieved context contain all the information needed to answer the question?" High recall means nothing important was missed.

They correspond to the retrieval Precision@K and Recall@K concepts but are computed at the *content level* (via LLM-as-judge) rather than from document ID labels.

In RAGAS, Context Recall is computed by checking whether each sentence in the reference answer can be attributed to some retrieved chunk.

---

### Q7. Describe the LangGraph adaptive routing mechanism and how you evaluate it.

**Answer:**

The routing mechanism is implemented as a `query_classifier` LangGraph node. It:
1. Retrieves a preliminary set of documents from FAISS (this is an important detail — retrieval happens *at classification time*, not just at the retrieval node)
2. Passes the query + retrieved context to an LLM with structured output (Pydantic `RouteIdentifier`)
3. Returns one of three routes: `"index"`, `"general"`, or `"search"`

The LangGraph conditional edge `route_query` reads `state["route"]` and directs flow to `retrieve_docs`, `general_llm_answer`, or `web_search` respectively.

**Evaluation approach:**
- Run each of the 60 labeled queries through `builder.invoke()`
- Compare `state["route"]` against `expected_route` in the golden dataset
- Compute: overall accuracy, per-route Precision/Recall/F1, and a 3×3 confusion matrix
- Track `query_rewrite_rate` (how often grade_docs triggered rewriting)

The confusion matrix reveals which routes are confused for which. A common failure mode: general-knowledge queries about AI being classified as `index` because the preliminary retrieval finds vaguely related chunks.

---

### Q8. Why does the query classifier perform retrieval before making a routing decision?

**Answer:**

This is a key architectural choice in `graph_builder.py`. The `query_classifier` node calls `get_retriever().invoke(question)` before invoking the classification LLM. It passes the retrieved `context` to the classify_prompt.

This allows the classifier to make an *evidence-based* routing decision: "Is there actually useful content in the index for this query?" rather than relying purely on the query text.

**Trade-off:** This means every query triggers one retrieval call even if it ultimately routes to `general` or `search`. This adds ~embedding-latency + FAISS-search-latency to every query. In the latency benchmark, this shows up as the `query_classification` latency including both an LLM call *and* a retrieval call.

This is a correctness vs. latency trade-off — more accurate routing at the cost of one extra retrieval per query.

---

### Q9. What is p95 latency and why is it more meaningful than mean latency for production systems?

**Answer:**

p95 latency is the 95th percentile: 95% of requests complete faster than this value. p99 is the 99th percentile.

**Why not mean?** The mean is dragged down by the many fast requests and doesn't reflect what the slowest users experience. A system with mean=0.5s might have p99=5s — the 1% worst-case users experience 10× worse performance than average.

**Why p95 specifically?** For a system with 1000 queries/hour, p95 means 50 requests/hour are slower than this threshold. p99 means 10 requests/hour exceed it. For LLM applications where users may retry slow queries, p95 is a good SLA target.

In this evaluation, I report: mean, median (p50), p90, p95, p99, min, max for each pipeline component. This reveals:
- Embedding generation: typically fast and consistent (small variance)
- LLM calls (classification, grading, generation): higher mean and much higher variance (p99 >> mean) due to network and model load

---

### Q10. What is the hallucination detection approach used in your evaluation?

**Answer:**

I use an LLM-as-judge approach with a custom faithfulness prompt:

1. The judge LLM is given: the question, the retrieved context, and the generated answer
2. It is asked: "Check if every claim in the Generated Answer is explicitly supported by the Retrieved Context"
3. It returns a JSON score from 0.0 to 1.0 with reasoning

Hallucination is flagged when faithfulness < 0.5. The `hallucination_rate` metric counts the fraction of queries where this threshold is crossed.

**Limitations of this approach:**
- The judge LLM may itself hallucinate or make inconsistent judgments
- The score varies between runs (LLM non-determinism)
- Atomic-claim decomposition (as RAGAS does) would be more rigorous
- I clearly label all generation scores as "LLM-as-judge" to distinguish them from deterministic metrics

**A more rigorous alternative**: RAGAS faithfulness decomposes the answer into individual atomic claims, then verifies each claim against the context independently, giving a more granular and consistent score.

---

### Q11. Why is the FAISS index in-memory, and what are the implications for evaluation?

**Answer:**

Looking at `retriever_setup.py`, the global `_faiss_vectorstore` variable is a Python module-level in-memory FAISS instance. It is populated when documents are uploaded via the API but not persisted to disk.

**Implications for evaluation:**
1. **Re-ingestion required**: Every evaluation run must re-ingest documents before running retrieval queries. The evaluation scripts call `ingest_document()` at startup.
2. **No stable chunk IDs**: FAISS assigns integer IDs internally, and chunk boundaries depend on the runtime document order. The evaluation tracks `source` filename metadata (stable) rather than FAISS chunk IDs (not stable).
3. **Memory limit**: Large document sets can exhaust RAM with an in-memory index.
4. **Improvement opportunity**: Adding `faiss.write_index()` / `faiss.read_index()` for persistence would eliminate re-ingestion overhead in production.

---

### Q12. How did you handle the fact that web-search results change over time for evaluation?

**Answer:**

Web-search route queries are categorized as `expected_route: "search"` in the golden dataset. The reference answers are marked `"REQUIRES_WEB_SEARCH"` to signal they cannot be pre-validated.

**What I evaluate:**
- **Routing accuracy**: Was the query correctly classified as `search` route?
- **Response generation**: Does the generated answer cite web sources coherently?

**What I explicitly exclude:**
- Factual correctness of web-sourced answers (Tavily results change daily)
- Context precision/recall for web results (no stable ground truth)

The evaluation report clearly separates search-route metrics from document-route metrics. This is the honest approach: web results are non-deterministic inputs, so claims about their quality are not reproducible.

---

### Q13. What configurations did you compare in the retrieval experiments?

**Answer:**

I compared two configurations across K ∈ {3, 5, 10}:

1. **Baseline**: Default pipeline — `get_retriever(k=4)` retrieves documents; the graph's `grade_docs` node filters; no query rewriting unless grader rejects all docs.

2. **With Rewriting**: Force `rewrite_query` before retrieval — the original query is rewritten by the LLM, then the rewritten query is used for FAISS similarity search.

**Expected trade-offs:**
- Rewriting should improve recall for ambiguous or poorly-phrased queries
- Rewriting adds ~LLM-call latency per query
- Rewriting may hurt precision if the rewritten query is too different from the original intent

**What I do not claim**: Reranking (the pipeline has none). If I were to add reranking (e.g., cross-encoder), I would implement it, run the same 40 queries, and report the before/after metric differences honestly.

---

### Q14. If faithfulness is low for some queries, what would you do to improve it?

**Answer:**

Low faithfulness indicates the LLM is generating unsupported claims. The remediation strategy depends on the root cause:

| Root Cause | Evidence | Fix |
|---|---|---|
| Irrelevant context retrieved | Context precision is also low | Tune K, add metadata filtering |
| Relevant context missing | Context recall is low | Increase K, fix chunking, improve query rewriting |
| LLM ignores context | Context is relevant but answer not grounded | Strengthen generate_prompt to emphasize context |
| Chunk too small | Key info split across chunks | Increase chunk size or overlap |

**Concrete actions for this system:**
1. Increase `chunk_overlap` from 200 to 400 to reduce context boundary fragmentation
2. Increase default K from 4 to 8 — grade_docs will filter irrelevant ones
3. Modify the `generate_prompt` to add: "Only state facts that appear explicitly in the context. If the answer is not in the context, say so."
4. Add few-shot examples to grade_prompt to improve grader precision

---

### Q15. How would you explain the evaluation framework you built to a non-technical hiring manager?

**Answer:**

I built a testing system for my AI assistant to measure how good it actually is — not based on guessing, but on real experiments.

The system works in three layers:

**Layer 1: Retrieval testing** (like a library's card catalog): I asked 40 questions where I already knew which documents contained the answers. I measured: "For every 10 documents the AI pulled out, how many were actually useful?" and "Did it find all the relevant information?" This is entirely objective and reproducible.

**Layer 2: Routing testing** (like testing a receptionist): My AI decides whether each question should be answered from stored documents, general knowledge, or a live web search. I tested all 60 questions and measured: "How often did it pick the right route?" — like checking if a receptionist sends the right caller to the right department.

**Layer 3: Answer quality testing** (like a human reviewer): I measured whether the AI's answers were truthful (not making up facts), relevant (actually answered the question), and correct (matched a known right answer). This required using the AI itself as the judge, which I clearly disclosed as a limitation.

The result is a reproducible benchmark — anyone with the same API keys can run the same experiment and get comparable numbers.

---

## C. Improvement Hypotheses to Test

After running the baseline, here are the justified improvement experiments to try:

| Experiment | Hypothesis | Metric to Watch | Implementation |
|---|---|---|---|
| Increase K from 4 to 8 | More context retrieved → higher Recall@K | Recall@5, Faithfulness | Change `search_kwargs={"k": 8}` in `retriever_setup.py` |
| Increase chunk size from 1000 to 1500 | Larger chunks capture more context per hit | Context Recall, Faithfulness | Change `chunk_size` in `document_upload.py` |
| Increase chunk overlap from 200 to 400 | Less boundary fragmentation | Recall@K on complex questions | Change `chunk_overlap` in `document_upload.py` |
| Stronger generate_prompt | Fewer unsupported claims | Faithfulness ↑ | Add "Only use facts from context" instruction |
| Add reranker | Better ranked results → higher nDCG | nDCG@5, MRR | `pip install sentence-transformers`; add cross-encoder after FAISS |
| RAGAS evaluation | Standardized scores for comparison | All generation metrics | `pip install ragas datasets`; use `--ragas` flag |

> Always run each experiment with the **same 40 document queries** and report the absolute metric change and latency impact.
