import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.evaluate_retrieval import (
    calculate_rank,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_rank_delta
)


def test_calculate_rank_descending_order():
    scores = np.array([0.1, 0.9, 0.4, 0.7])
    # Target index 1 (score 0.9) is highest => rank 1
    assert calculate_rank(scores, 1) == 1
    # Target index 3 (score 0.7) is second highest => rank 2
    assert calculate_rank(scores, 3) == 2
    # Target index 2 (score 0.4) is third => rank 3
    assert calculate_rank(scores, 2) == 3
    # Target index 0 (score 0.1) is lowest => rank 4
    assert calculate_rank(scores, 0) == 4


def test_calculate_recall_at_k():
    assert calculate_recall_at_k(1, 5) == 1.0
    assert calculate_recall_at_k(5, 5) == 1.0
    assert calculate_recall_at_k(6, 5) == 0.0
    assert calculate_recall_at_k(10, 10) == 1.0
    assert calculate_recall_at_k(11, 10) == 0.0


def test_calculate_mrr():
    assert calculate_mrr(1) == 1.0
    assert calculate_mrr(2) == 0.5
    assert calculate_mrr(4) == 0.25
    assert calculate_mrr(0) == 0.0


def test_calculate_rank_delta():
    # Base rank 5, HyDE rank 1 => improved by 4
    assert calculate_rank_delta(5, 1) == 4
    # Both rank 1 => 0 delta
    assert calculate_rank_delta(1, 1) == 0
    # Base rank 1, HyDE rank 3 => degraded by -2
    assert calculate_rank_delta(1, 3) == -2
