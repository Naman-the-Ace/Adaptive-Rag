"""
Unit Tests for Retrieval Evaluation Metric Functions
=====================================================
Tests are deterministic — no LLM or API calls required.

Run with:
  cd evaluation
  python -m pytest tests/test_metrics.py -v
"""

import math
import sys
from pathlib import Path

import pytest

# Add evaluation/scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate_retrieval import (
    dcg_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Precision@K Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrecisionAtK:
    def test_all_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_none_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_half_relevant(self):
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, k=4) == pytest.approx(0.5)

    def test_k_less_than_retrieved(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        # top 3: a, x, b → 2/3
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    def test_k_zero(self):
        assert precision_at_k(["a", "b"], {"a"}, k=0) == 0.0

    def test_empty_relevant(self):
        # With no relevant docs, all precision values should be 0
        retrieved = ["a", "b", "c"]
        assert precision_at_k(retrieved, set(), k=3) == 0.0

    def test_k1_first_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, k=1) == 1.0

    def test_k1_first_irrelevant(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, k=1) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Recall@K Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_none_retrieved(self):
        retrieved = ["x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.5)

    def test_empty_relevant(self):
        # With no labeled relevant, recall is undefined → return 0
        assert recall_at_k(["a", "b"], set(), k=3) == 0.0

    def test_k_smaller_than_all_relevant(self):
        # Only 2 slots but 4 relevant docs
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b", "c", "d"}
        assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(2 / 4)

    def test_recall_at_larger_k_improves(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        r3 = recall_at_k(retrieved, relevant, k=3)
        r5 = recall_at_k(retrieved, relevant, k=5)
        assert r5 >= r3


# ═══════════════════════════════════════════════════════════════════════════════
# Hit Rate@K Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHitRateAtK:
    def test_hit(self):
        retrieved = ["x", "y", "a"]
        relevant = {"a", "b"}
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0

    def test_no_hit(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert hit_rate_at_k(retrieved, relevant, k=3) == 0.0

    def test_hit_at_first_position(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a"}
        assert hit_rate_at_k(retrieved, relevant, k=1) == 1.0

    def test_no_hit_at_k1_but_hit_at_k3(self):
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        assert hit_rate_at_k(retrieved, relevant, k=1) == 0.0
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0

    def test_empty_relevant(self):
        assert hit_rate_at_k(["a", "b"], set(), k=2) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# MRR Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestReciprocalRank:
    def test_first_position(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)

    def test_second_position(self):
        retrieved = ["x", "a", "c"]
        relevant = {"a"}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)

    def test_third_position(self):
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_multiple_relevant_uses_first(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        # First relevant is 'a' at rank 2
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)

    def test_empty_retrieved(self):
        assert reciprocal_rank([], {"a"}) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# nDCG@K Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNDCG:
    def test_perfect_ranking(self):
        # All relevant docs at top → nDCG = 1.0
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_no_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_suboptimal_ranking(self):
        # Relevant doc 'a' is at position 3 instead of 1
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        # DCG = 1/log2(4) = 1/2
        # IDCG = 1/log2(2) = 1
        expected = (1 / math.log2(4)) / (1 / math.log2(2))
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected, abs=1e-6)

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        assert ndcg_at_k(retrieved, set(), k=2) == 0.0

    def test_k_larger_than_retrieved(self):
        # Should not crash when K > len(retrieved)
        retrieved = ["a", "b"]
        relevant = {"a"}
        result = ndcg_at_k(retrieved, relevant, k=10)
        assert 0.0 <= result <= 1.0

    def test_dcg_at_k_manual(self):
        # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        expected = 1 / math.log2(2) + 1 / math.log2(4)
        assert dcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Consistency / Property Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetricProperties:
    """Property-based tests that hold for all valid inputs."""

    def test_precision_bounded(self):
        for retrieved, relevant, k in [
            (["a", "b", "c"], {"a"}, 3),
            ([], {"a"}, 5),
            (["x", "y"], set(), 2),
        ]:
            score = precision_at_k(retrieved, relevant, k)
            assert 0.0 <= score <= 1.0

    def test_recall_bounded(self):
        for retrieved, relevant, k in [
            (["a", "b"], {"a", "b", "c"}, 2),
            (["a"], {"a"}, 1),
        ]:
            score = recall_at_k(retrieved, relevant, k)
            assert 0.0 <= score <= 1.0

    def test_ndcg_bounded(self):
        for retrieved, relevant, k in [
            (["a", "x", "b"], {"a", "b"}, 3),
            (["x", "y", "z"], {"a"}, 3),
            (["a", "b"], {"a", "b"}, 2),
        ]:
            score = ndcg_at_k(retrieved, relevant, k)
            assert 0.0 <= score <= 1.0 + 1e-9  # small tolerance for float

    def test_mrr_bounded(self):
        for retrieved, relevant in [
            (["a", "b"], {"b"}),
            (["x", "y"], {"a"}),
            ([], {"a"}),
        ]:
            score = reciprocal_rank(retrieved, relevant)
            assert 0.0 <= score <= 1.0

    def test_recall_increases_with_k(self):
        """Recall@K is non-decreasing as K increases."""
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        prev = 0.0
        for k in range(1, len(retrieved) + 1):
            current = recall_at_k(retrieved, relevant, k)
            assert current >= prev - 1e-9
            prev = current

    def test_hit_rate_increases_with_k(self):
        """Hit Rate is non-decreasing as K increases."""
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        for k in range(1, len(retrieved)):
            assert hit_rate_at_k(retrieved, relevant, k) <= hit_rate_at_k(retrieved, relevant, k + 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
