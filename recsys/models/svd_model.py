"""
Matrix Factorization baseline using Surprise's SVD implementation.

Public API
──────────
  build_surprise_dataset(df)          → Surprise Dataset
  tune_svd(train_df)                  → best SVDModel (GridSearchCV over factors/lr/reg)
  SVDModel.fit(train_df)
  SVDModel.predict_score(uid, iid)
  SVDModel.top_k(uid, candidates, k)
  SVDModel.save() / SVDModel.load()

  get_top_n_surprise(model, user_idx, candidate_items, n=10)
  evaluate_ranking_metrics_svd(model, test_df, train_df, num_items, k=10)
"""
import pickle
import random
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from surprise import SVD, Dataset as SurpriseDataset, Reader
from surprise.model_selection import GridSearchCV

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH    = ARTIFACTS_DIR / "svd_model.pkl"
RATING_SCALE  = (1, 5)
RELEVANCE_THRESHOLD = 4.0


# ──────────────────────────────────────────────────────────────────────────────
# Dataset conversion
# ──────────────────────────────────────────────────────────────────────────────

def build_surprise_dataset(df: pd.DataFrame) -> SurpriseDataset:
    """
    Convert a DataFrame with columns [user_idx, item_idx, rating] into
    a Surprise Dataset object ready for trainset/testset extraction.
    """
    reader = Reader(rating_scale=RATING_SCALE)
    return SurpriseDataset.load_from_df(
        df[["user_idx", "item_idx", "rating"]], reader
    )


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter tuning
# ──────────────────────────────────────────────────────────────────────────────

# Light grid — keeps tuning fast while covering the important knobs.
_PARAM_GRID = {
    "n_factors": [20, 50, 100],
    "lr_all":    [0.005, 0.01],
    "reg_all":   [0.02, 0.1],
    "n_epochs":  [20],
}


def tune_svd(train_df: pd.DataFrame, cv: int = 3) -> "SVDModel":
    """
    Run a GridSearchCV over _PARAM_GRID using RMSE on cv folds of train_df.
    Returns a fitted SVDModel with the best hyperparameters.
    """
    print("[SVD] Starting hyperparameter search …")
    data = build_surprise_dataset(train_df)

    gs = GridSearchCV(SVD, _PARAM_GRID, measures=["rmse"], cv=cv, n_jobs=-1)
    gs.fit(data)

    best = gs.best_params["rmse"]
    print(
        f"[SVD] Best params → n_factors={best['n_factors']}  "
        f"lr_all={best['lr_all']}  reg_all={best['reg_all']}  "
        f"RMSE={gs.best_score['rmse']:.4f}"
    )

    model = SVDModel(
        n_factors=best["n_factors"],
        n_epochs=best["n_epochs"],
        lr_all=best["lr_all"],
        reg_all=best["reg_all"],
    )
    model.fit(train_df)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Model class
# ──────────────────────────────────────────────────────────────────────────────

class SVDModel:
    def __init__(
        self,
        n_factors: int   = 50,
        n_epochs:  int   = 20,
        lr_all:    float = 0.005,
        reg_all:   float = 0.02,
    ):
        self.algo = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42,
        )
        self._trained  = False
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr_all    = lr_all
        self.reg_all   = reg_all

    # ── train ──────────────────────────────────────────────────────────
    def fit(self, train_df: pd.DataFrame) -> "SVDModel":
        """
        Accepts a DataFrame with columns [user_idx, item_idx, rating].
        Converts to Surprise format and trains on the full trainset.
        """
        data     = build_surprise_dataset(train_df)
        trainset = data.build_full_trainset()
        self.algo.fit(trainset)
        self._trained = True
        print(
            f"[SVDModel] Trained  n_factors={self.n_factors}  "
            f"n_epochs={self.n_epochs}  lr={self.lr_all}  reg={self.reg_all}"
        )
        return self

    # ── predict ────────────────────────────────────────────────────────
    def predict_score(self, user_idx: int, item_idx: int) -> float:
        """Predicted rating for a single (user_idx, item_idx) pair."""
        return self.algo.predict(user_idx, item_idx).est

    def top_k(
        self,
        user_idx:        int,
        candidate_items: List[int],
        k:               int = 10,
    ) -> List[Tuple[int, float]]:
        """Return [(item_idx, score), …] sorted by score descending."""
        scores = [(i, self.predict_score(user_idx, i)) for i in candidate_items]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    # ── persist ────────────────────────────────────────────────────────
    def save(self) -> None:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        print(f"[SVDModel] Saved → {MODEL_PATH}")

    @classmethod
    def load(cls) -> "SVDModel":
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"[SVDModel] Loaded ← {MODEL_PATH}")
        return model


# ──────────────────────────────────────────────────────────────────────────────
# Standalone helper functions
# ──────────────────────────────────────────────────────────────────────────────

def get_top_n_surprise(
    model:           SVDModel,
    user_idx:        int,
    candidate_items: List[int],
    n:               int = 10,
) -> List[Tuple[int, float]]:
    """
    Convenience wrapper around SVDModel.top_k.

    Parameters
    ----------
    model            : trained SVDModel instance
    user_idx         : integer user index
    candidate_items  : list of integer item indices to score
    n                : number of recommendations to return

    Returns
    -------
    List of (item_idx, predicted_score) sorted descending, length ≤ n.
    """
    return model.top_k(user_idx, candidate_items, k=n)


# ── metric helpers (kept local to avoid circular imports) ─────────────────────

def _dcg(ranked: List[int], relevant: Set[int], k: int) -> float:
    return sum(
        1.0 / math.log2(r + 2)
        for r, item in enumerate(ranked[:k])
        if item in relevant
    )


def _ndcg(ranked: List[int], relevant: Set[int], k: int) -> float:
    ideal_len = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_len))
    return _dcg(ranked, relevant, k) / idcg if idcg else 0.0


def _ap(ranked: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = precision_sum = 0
    for r, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / r
    return precision_sum / min(len(relevant), k)


def evaluate_ranking_metrics_svd(
    model:     SVDModel,
    test_df:   pd.DataFrame,
    train_df:  pd.DataFrame,
    num_items: int,
    k:         int = 10,
    neg_samples: int = 99,
    seed:      int = 42,
) -> Dict[str, float]:
    """
    Evaluate SVD using NDCG@k and MAP@k on the test set.

    Protocol
    --------
    For each user with at least one relevant test item (rating ≥ 4):
      1. Build candidate pool = relevant items + `neg_samples` randomly
         sampled unseen negatives (standard 1-vs-N evaluation protocol).
      2. Score and rank the pool with get_top_n_surprise().
      3. Compute NDCG@k and AP@k for the ranked list.

    Parameters
    ----------
    model       : trained SVDModel
    test_df     : DataFrame with columns [user_idx, item_idx, rating]
    train_df    : DataFrame used to identify already-seen items
    num_items   : total number of unique items
    k           : cut-off rank (default 10)
    neg_samples : number of negative items to sample per user
    seed        : random seed for reproducibility

    Returns
    -------
    {
        "ndcg@k" : float,   mean NDCG@k across test users
        "map@k"  : float,   mean AP@k  across test users
        "num_users": int,   number of users evaluated
    }
    """
    rng = random.Random(seed)
    all_items = list(range(num_items))

    # Per-user sets of items seen during training
    train_seen: Dict[int, Set[int]] = (
        train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )

    # Per-user relevant items in test (binary relevance: rating >= threshold)
    relevant_map: Dict[int, Set[int]] = {}
    for uid, grp in test_df.groupby("user_idx"):
        rel = set(grp.loc[grp["rating"] >= RELEVANCE_THRESHOLD, "item_idx"].tolist())
        if rel:
            relevant_map[uid] = rel

    ndcg_list, ap_list = [], []

    for user_idx, relevant in relevant_map.items():
        seen       = train_seen.get(user_idx, set())
        negatives  = [i for i in all_items if i not in seen and i not in relevant]
        sampled    = rng.sample(negatives, min(neg_samples, len(negatives)))
        eval_pool  = list(relevant) + sampled

        ranked_pairs = get_top_n_surprise(model, user_idx, eval_pool, n=k)
        ranked_items = [item for item, _ in ranked_pairs]

        ndcg_list.append(_ndcg(ranked_items, relevant, k))
        ap_list.append(_ap(ranked_items, relevant, k))

    n_users = len(ndcg_list)
    results = {
        f"ndcg@{k}":  round(sum(ndcg_list) / n_users, 4) if n_users else 0.0,
        f"map@{k}":   round(sum(ap_list)   / n_users, 4) if n_users else 0.0,
        "num_users":  n_users,
    }
    print(
        f"[SVD eval] NDCG@{k}={results[f'ndcg@{k}']}  "
        f"MAP@{k}={results[f'map@{k}']}  users={n_users}"
    )
    return results
