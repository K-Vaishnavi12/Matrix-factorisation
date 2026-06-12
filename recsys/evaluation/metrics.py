"""
Ranking evaluation metrics — manual implementations of NDCG@K and MAP@K.

Standard definitions
────────────────────

DCG@K (Discounted Cumulative Gain)
    Measures ranking quality by rewarding relevant items placed higher.
    Binary relevance: rel(i) = 1 if item i is relevant, else 0.

        DCG@K = sum_{r=1}^{K}  rel(r) / log2(r + 1)

    where r is the 1-based rank position in the predicted list.

IDCG@K (Ideal DCG)
    The best achievable DCG@K — all relevant items packed at the top.

        IDCG@K = sum_{r=1}^{min(|relevant|, K)}  1 / log2(r + 1)

NDCG@K (Normalised DCG)
    Scales DCG into [0, 1] so scores are comparable across users.

        NDCG@K = DCG@K / IDCG@K       (0 if IDCG@K == 0)

AP@K (Average Precision at K)
    Rewards finding relevant items early in the ranked list.

        AP@K = (1 / min(|relevant|, K)) * sum_{r=1}^{K}  P(r) * rel(r)

    where P(r) = (number of relevant items in positions 1..r) / r.

MAP@K (Mean Average Precision)
    Mean of AP@K over all evaluated users.

        MAP@K = (1 / |users|) * sum_u  AP@K(u)

Public API
──────────
  dcg_at_k(ranked_items, relevant, k)             → float
  idcg_at_k(relevant, k)                          → float
  ndcg_at_k(ranked_items, relevant, k)             → float
  average_precision_at_k(ranked_items, relevant, k)→ float
  mean_average_precision_at_k(results, k)          → float   (corpus-level)

  evaluate_ranking_metrics_ncf(model, test_df, train_df, num_items, k)
  evaluate_ranking_metrics_svd(model, test_df, train_df, num_items, k)  ← imported from svd_model
  evaluate_model(model, test_df, train_df, num_items, k)               ← generic alias
"""

import math
import random
from typing import Dict, List, Set, Tuple

RELEVANCE_THRESHOLD = 4.0   # rating >= this threshold → item is relevant
DEFAULT_K           = 10


# ──────────────────────────────────────────────────────────────────────────────
# NDCG@K
# ──────────────────────────────────────────────────────────────────────────────

def dcg_at_k(
    ranked_items: List[int],
    relevant:     Set[int],
    k:            int = DEFAULT_K,
) -> float:
    """
    Discounted Cumulative Gain at K.

    Formula
    -------
        DCG@K = sum_{r=1}^{K}  rel(r) / log2(r + 1)

    Parameters
    ----------
    ranked_items : predicted item list ordered by score descending
    relevant     : ground-truth set of relevant item indices for this user
    k            : rank cut-off

    Returns
    -------
    DCG@K as a float ≥ 0.
    """
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        # rel(rank) is binary: 1 if item is in the relevant set, else 0
        if item in relevant:
            # Discount factor: 1 / log2(rank + 1)
            # rank=1 → 1/log2(2)=1.0   rank=2 → 1/log2(3)≈0.63   etc.
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def idcg_at_k(relevant: Set[int], k: int = DEFAULT_K) -> float:
    """
    Ideal Discounted Cumulative Gain at K.

    Assumes the best-case scenario: all relevant items appear at the
    very top of the ranking, in positions 1 … min(|relevant|, K).

    Formula
    -------
        IDCG@K = sum_{r=1}^{min(|relevant|, K)}  1 / log2(r + 1)

    Parameters
    ----------
    relevant : ground-truth relevant item set
    k        : rank cut-off

    Returns
    -------
    IDCG@K as a float ≥ 0.  Returns 0.0 when the relevant set is empty.
    """
    # Only min(|relevant|, K) positions can contribute
    ideal_hits = min(len(relevant), k)
    return sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))


def ndcg_at_k(
    ranked_items: List[int],
    relevant:     Set[int],
    k:            int = DEFAULT_K,
) -> float:
    """
    Normalised Discounted Cumulative Gain at K.

    Formula
    -------
        NDCG@K = DCG@K / IDCG@K

    Normalisation ensures the score lies in [0, 1] regardless of how
    many relevant items exist, making scores comparable across users.

    Parameters
    ----------
    ranked_items : predicted item list ordered by score descending
    relevant     : ground-truth set of relevant item indices for this user
    k            : rank cut-off (default 10)

    Returns
    -------
    NDCG@K in [0.0, 1.0].  Returns 0.0 if there are no relevant items.

    Example
    -------
    >>> ndcg_at_k([3, 1, 2, 4], relevant={1, 3}, k=4)
    # rank 1: item 3 → relevant  DCG += 1/log2(2) = 1.000
    # rank 2: item 1 → relevant  DCG += 1/log2(3) = 0.631
    # DCG@4  = 1.631
    # IDCG@4 = 1/log2(2) + 1/log2(3) = 1.631   (both relevant at top)
    # NDCG@4 = 1.631 / 1.631 = 1.0
    """
    idcg = idcg_at_k(relevant, k)
    if idcg == 0.0:
        return 0.0                              # no relevant items → undefined, return 0
    return dcg_at_k(ranked_items, relevant, k) / idcg


# ──────────────────────────────────────────────────────────────────────────────
# MAP@K
# ──────────────────────────────────────────────────────────────────────────────

def average_precision_at_k(
    ranked_items: List[int],
    relevant:     Set[int],
    k:            int = DEFAULT_K,
) -> float:
    """
    Average Precision at K for a single user.

    Formula
    -------
        AP@K = (1 / min(|relevant|, K))  *  sum_{r=1}^{K}  P(r) * rel(r)

    where
        P(r)   = number of relevant items in positions 1..r  /  r
        rel(r) = 1 if the item at rank r is relevant, else 0

    The normalisation by min(|relevant|, K) penalises a model that only
    finds some of the available relevant items.

    Parameters
    ----------
    ranked_items : predicted item list ordered by score descending
    relevant     : ground-truth set of relevant item indices for this user
    k            : rank cut-off (default 10)

    Returns
    -------
    AP@K in [0.0, 1.0].  Returns 0.0 if there are no relevant items.

    Example
    -------
    >>> average_precision_at_k([1, 2, 3, 4], relevant={1, 3}, k=4)
    # rank 1: item 1 → hit  hits=1  P(1)=1/1=1.000  precision_sum += 1.000
    # rank 2: item 2 → miss
    # rank 3: item 3 → hit  hits=2  P(3)=2/3=0.667  precision_sum += 0.667
    # rank 4: item 4 → miss
    # AP@4 = (1 / min(2,4)) * 1.667 = 1.667 / 2 = 0.833
    """
    if not relevant:
        return 0.0

    hits          = 0
    precision_sum = 0.0

    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in relevant:
            hits          += 1
            # Precision at this rank = running hits / current rank position
            precision_sum += hits / rank

    # Normalise by the number of relevant items we *could* have found (≤ K)
    return precision_sum / min(len(relevant), k)


def mean_average_precision_at_k(
    user_results: List[Tuple[List[int], Set[int]]],
    k:            int = DEFAULT_K,
) -> float:
    """
    Mean Average Precision at K across multiple users.

    Formula
    -------
        MAP@K = (1 / |users|)  *  sum_u  AP@K(u)

    Parameters
    ----------
    user_results : list of (ranked_items, relevant_set) per user
    k            : rank cut-off (default 10)

    Returns
    -------
    MAP@K as a float in [0.0, 1.0].
    """
    if not user_results:
        return 0.0
    ap_scores = [
        average_precision_at_k(ranked, relevant, k)
        for ranked, relevant in user_results
    ]
    return sum(ap_scores) / len(ap_scores)


# Backward-compat alias (no cut-off, used by old code paths)
def average_precision(ranked_items: List[int], relevant: Set[int]) -> float:
    """Average Precision with no rank cut-off (legacy alias)."""
    return average_precision_at_k(ranked_items, relevant, k=len(ranked_items))


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation engine (shared by NCF and SVD)
# ──────────────────────────────────────────────────────────────────────────────

def _run_evaluation(
    model,
    test_df,
    train_df,
    num_items:   int,
    k:           int = DEFAULT_K,
    neg_samples: int = 99,
    seed:        int = 42,
    label:       str = "model",
) -> Dict[str, float]:
    """
    Evaluation loop implementing the standard 1-vs-N protocol.

    For each user that has ≥ 1 relevant item in the test set:
      1. Candidate pool = all relevant test items
                        + `neg_samples` randomly sampled unseen negatives.
         (Unseen = not in training set and not in relevant set.)
      2. Score and rank the pool with model.top_k(user_idx, pool, k=k).
      3. Compute NDCG@K and AP@K using the functions above.
    Average both metrics across users.

    Returns
    -------
    {"ndcg@k": float, "map@k": float, "num_users": int}
    """
    rng       = random.Random(seed)
    all_items = list(range(num_items))

    train_seen: Dict[int, Set[int]] = (
        train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )

    relevant_map: Dict[int, Set[int]] = {}
    for uid, grp in test_df.groupby("user_idx"):
        rel = set(grp.loc[grp["rating"] >= RELEVANCE_THRESHOLD, "item_idx"].tolist())
        if rel:
            relevant_map[uid] = rel

    ndcg_list: List[float] = []
    ap_list:   List[float] = []

    for user_idx, relevant in relevant_map.items():
        seen      = train_seen.get(user_idx, set())
        negatives = [i for i in all_items if i not in seen and i not in relevant]
        sampled   = rng.sample(negatives, min(neg_samples, len(negatives)))
        eval_pool = list(relevant) + sampled

        # Score every candidate, get top-k ranked list
        ranked_pairs = model.top_k(user_idx, eval_pool, k=k)
        ranked_items = [item for item, _ in ranked_pairs]

        ndcg_list.append(ndcg_at_k(ranked_items, relevant, k))
        ap_list.append(average_precision_at_k(ranked_items, relevant, k))

    n = len(ndcg_list)
    result = {
        f"ndcg@{k}": round(sum(ndcg_list) / n, 4) if n else 0.0,
        f"map@{k}":  round(sum(ap_list)   / n, 4) if n else 0.0,
        "num_users": n,
    }
    print(
        f"[{label}] NDCG@{k}={result[f'ndcg@{k}']}  "
        f"MAP@{k}={result[f'map@{k}']}  users={n}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Model-specific public wrappers
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_ranking_metrics_ncf(
    model,
    test_df,
    train_df,
    num_items:   int,
    k:           int = DEFAULT_K,
    neg_samples: int = 99,
    seed:        int = 42,
) -> Dict[str, float]:
    """
    Evaluate a trained NCFModel with NDCG@K and MAP@K.

    Parameters
    ----------
    model       : NCFModel with a .top_k(user_idx, candidates, k) method
    test_df     : DataFrame [user_idx, item_idx, rating]
    train_df    : DataFrame [user_idx, item_idx, rating]
    num_items   : total item count
    k           : rank cut-off (default 10)
    neg_samples : negatives sampled per user for the eval pool (default 99)
    seed        : RNG seed

    Returns
    -------
    {"ndcg@k": float, "map@k": float, "num_users": int}
    """
    return _run_evaluation(
        model, test_df, train_df, num_items,
        k=k, neg_samples=neg_samples, seed=seed, label="NCF eval",
    )


def evaluate_model(
    model,
    test_df,
    train_df,
    num_items: int,
    k:         int = DEFAULT_K,
) -> Dict[str, float]:
    """Generic evaluator — works with any model that exposes .top_k()."""
    return _run_evaluation(
        model, test_df, train_df, num_items, k=k, label="eval",
    )
