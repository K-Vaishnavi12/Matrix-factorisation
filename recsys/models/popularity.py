"""
Popularity baseline — most-popular non-personalised recommender.

Why include it?
───────────────
A popularity model is the canonical sanity-check baseline in recsys.
If your fancy MF / NN model can't beat "show everyone the most popular
items", the model is broken or the dataset is too small to learn from.
Reporting popularity NDCG@K alongside SVD/NCF is standard practice in
research papers (He et al. 2017, Rendle et al. 2020).

Algorithm
─────────
1. Score each item by either:
     • count   — number of training interactions with that item
     • mean    — mean rating of that item (count-weighted via Bayesian prior)
2. The same global ranking is returned to every user (ignoring already-seen).

Public API mirrors SVDModel / NCFModel for drop-in evaluation:
   model.fit(train_df) → self
   model.predict_score(user_idx, item_idx) → float
   model.top_k(user_idx, candidate_items, k=10) → [(item_idx, score), ...]
   model.save() / model.load()
"""
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH    = ARTIFACTS_DIR / "popularity_model.pkl"


class PopularityModel:
    """
    Non-personalised most-popular baseline.

    Parameters
    ----------
    method : {"count", "mean"}, default "count"
        - "count": rank by raw interaction count (popularity).
        - "mean":  rank by Bayesian-smoothed mean rating to avoid
                   one-rating wonders dominating the ranking.
    prior_strength : float, default 5.0
        Pseudo-count used by the "mean" method
        (smoothed_mean = (sum + prior * global_mean) / (count + prior)).
    """

    def __init__(self, method: str = "count", prior_strength: float = 5.0):
        assert method in {"count", "mean"}, f"Unknown method '{method}'"
        self.method         = method
        self.prior_strength = prior_strength
        self._scores: Dict[int, float] = {}
        self._trained                  = False

    # ── train ──────────────────────────────────────────────────────────
    def fit(self, train_df: pd.DataFrame) -> "PopularityModel":
        """
        Compute item popularity scores from the training set.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training interactions with columns [user_idx, item_idx, rating].
        """
        if self.method == "count":
            counts       = train_df.groupby("item_idx").size()
            self._scores = counts.astype(float).to_dict()
        else:  # mean
            global_mean = float(train_df["rating"].mean())
            grp         = train_df.groupby("item_idx")["rating"].agg(["sum", "count"])
            smoothed    = (grp["sum"] + self.prior_strength * global_mean) / (grp["count"] + self.prior_strength)
            self._scores = smoothed.to_dict()

        self._trained = True
        print(
            f"[Popularity] Trained  method={self.method}  "
            f"items={len(self._scores):,}  "
            f"score_range=[{min(self._scores.values()):.3f}, {max(self._scores.values()):.3f}]"
        )
        return self

    # ── predict ────────────────────────────────────────────────────────
    def predict_score(self, user_idx: int, item_idx: int) -> float:
        """User-independent score (popularity is non-personalised)."""
        return self._scores.get(item_idx, 0.0)

    def top_k(
        self,
        user_idx:        int,
        candidate_items: List[int],
        k:               int = 10,
    ) -> List[Tuple[int, float]]:
        """Return the top-k popular items from `candidate_items` (ignores user_idx)."""
        scored = [(i, self._scores.get(i, 0.0)) for i in candidate_items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ── persist ────────────────────────────────────────────────────────
    def save(self) -> None:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        print(f"[Popularity] Saved → {MODEL_PATH}")

    @classmethod
    def load(cls) -> "PopularityModel":
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"[Popularity] Loaded ← {MODEL_PATH}")
        return model
