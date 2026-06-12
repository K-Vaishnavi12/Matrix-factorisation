"""
Manual verification of NDCG@K and MAP@K against hand-calculated values.
Run with:  python -m recsys.evaluation.test_metrics
"""
import math
from recsys.evaluation.metrics import dcg_at_k, idcg_at_k, ndcg_at_k, average_precision_at_k


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def test_dcg():
    # ranked = [1, 2, 3, 4],  relevant = {1, 3},  k=4
    # rank 1: item 1 → hit  1/log2(2) = 1.0000
    # rank 2: item 2 → miss
    # rank 3: item 3 → hit  1/log2(4) = 0.5000
    # DCG@4 = 1.5
    result = dcg_at_k([1, 2, 3, 4], {1, 3}, k=4)
    expected = 1.0 / math.log2(2) + 1.0 / math.log2(4)   # 1.5
    assert approx(result, expected), f"dcg_at_k failed: {result} != {expected}"
    print(f"  dcg_at_k         = {result:.6f}  ✓  (expected {expected:.6f})")


def test_idcg():
    # relevant = {1, 3},  k=4  → ideal hits = min(2,4) = 2
    # IDCG = 1/log2(2) + 1/log2(3)
    result   = idcg_at_k({1, 3}, k=4)
    expected = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert approx(result, expected), f"idcg_at_k failed: {result} != {expected}"
    print(f"  idcg_at_k        = {result:.6f}  ✓  (expected {expected:.6f})")


def test_ndcg_perfect():
    # Perfect ranking: both relevant items at top → NDCG = 1.0
    result = ndcg_at_k([1, 3, 2, 4], {1, 3}, k=4)
    assert approx(result, 1.0), f"ndcg perfect failed: {result}"
    print(f"  ndcg_at_k perfect= {result:.6f}  ✓  (expected 1.000000)")


def test_ndcg_partial():
    # ranked=[1,2,3,4]  relevant={1,3}  k=4
    # DCG  = 1/log2(2) + 1/log2(4)          = 1.5
    # IDCG = 1/log2(2) + 1/log2(3)          ≈ 1.6309
    # NDCG = 1.5 / 1.6309                   ≈ 0.9197
    dcg      = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg     = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg
    result   = ndcg_at_k([1, 2, 3, 4], {1, 3}, k=4)
    assert approx(result, expected), f"ndcg partial failed: {result} != {expected}"
    print(f"  ndcg_at_k partial= {result:.6f}  ✓  (expected {expected:.6f})")


def test_ndcg_no_relevant():
    result = ndcg_at_k([1, 2, 3], set(), k=3)
    assert result == 0.0, f"ndcg empty relevant failed: {result}"
    print(f"  ndcg_at_k empty  = {result:.6f}  ✓  (expected 0.000000)")


def test_ap_k():
    # ranked=[1,2,3,4]  relevant={1,3}  k=4
    # rank 1: item 1 → hit  hits=1  P(1)=1/1=1.000  sum=1.000
    # rank 2: item 2 → miss
    # rank 3: item 3 → hit  hits=2  P(3)=2/3=0.667  sum=1.667
    # AP@4 = 1.667 / min(2,4) = 0.8333
    result   = average_precision_at_k([1, 2, 3, 4], {1, 3}, k=4)
    expected = (1.0 + 2.0 / 3.0) / min(2, 4)
    assert approx(result, expected), f"ap@k failed: {result} != {expected}"
    print(f"  ap_at_k          = {result:.6f}  ✓  (expected {expected:.6f})")


def test_ap_k_perfect():
    # Both relevant items at top → AP@2 = (1/1 + 2/2) / 2 = 1.0
    result = average_precision_at_k([1, 3, 2, 4], {1, 3}, k=4)
    expected = (1.0 / 1 + 2.0 / 2) / min(2, 4)   # (1 + 1) / 2 = 1.0
    assert approx(result, expected), f"ap perfect failed: {result} != {expected}"
    print(f"  ap_at_k perfect  = {result:.6f}  ✓  (expected {expected:.6f})")


def test_ap_no_relevant():
    result = average_precision_at_k([1, 2, 3], set(), k=3)
    assert result == 0.0
    print(f"  ap_at_k empty    = {result:.6f}  ✓  (expected 0.000000)")


if __name__ == "__main__":
    print("Running metric unit tests …\n")
    test_dcg()
    test_idcg()
    test_ndcg_perfect()
    test_ndcg_partial()
    test_ndcg_no_relevant()
    test_ap_k()
    test_ap_k_perfect()
    test_ap_no_relevant()
    print("\nAll tests passed.")
