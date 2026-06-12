"""
Item-based collaborative filtering with cosine similarity.

Algorithm (classical Sarwar et al., WWW 2001)
─────────────────────────────────────────────
1. Build a sparse user-item interaction matrix R (binary: rated >= threshold).
2. Normalise columns to unit L2 norm: R_n = R / ||R_:i||_2.
3. Item-item similarity matrix S = R_n^T @ R_n   (cosine similarity).
4. Top-k recommendations for user u:
       score(u, i) = sum_{j in seen(u)} S[i, j]
   i.e. an item is scored by its similarity to items u has interacted with.

Why include it?
───────────────
- Strong, time-tested baseline that often rivals deep models on small datasets.
- Uses no embeddings / no SGD — a useful sanity check for model-based methods.
- Inference is a single sparse matrix-vector product.

Public API mirrors the other models:
   model.fit(train_df)
   model.predict_score(user_idx, item_idx)
   model.top_k(user_idx, candidate_items, k=10)
   model.save() / model.load()
"""
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH    = ARTIFACTS_DIR / "itemknn_model.pkl"
RELEVANCE_THRESHOLD = 4.0


class ItemKNNModel:
    """
    Item-item collaborative filtering with cosine similarity.

    Parameters
    ----------
    top_k_neighbors : int, default 50
        Keep only the top-k most similar items per item to keep S sparse.
    use_implicit : bool, default True
        If True, treat any rating >= RELEVANCE_THRESHOLD as 1.
        If False, use raw ratings (mean-centered for cosine).
    """

    def __init__(self, top_k_neighbors: int = 50, use_implicit: bool = True):
        self.top_k_neighbors = top_k_neighbors
        self.use_implicit    = use_implicit
        self._sim:        sparse.csr_matrix = None  # (num_items, num_items)
        self._user_items: Dict[int, Set[int]] = {}
        self._user_ratings: Dict[int, Dict[int, float]] = {}
        self._num_items                       = 0
        self._trained                         = False

    # ── train ──────────────────────────────────────────────────────────
    def fit(self, train_df: pd.DataFrame) -> "ItemKNNModel":
        """
        Build the item-item cosine similarity matrix from training data.
        """
        num_items     = int(train_df["item_idx"].max()) + 1
        num_users     = int(train_df["user_idx"].max()) + 1
        self._num_items = num_items

        # Build sparse user-item matrix R
        if self.use_implicit:
            mask   = train_df["rating"] >= RELEVANCE_THRESHOLD
            df_use = train_df[mask]
            data   = np.ones(len(df_use), dtype=np.float32)
        else:
            df_use = train_df
            data   = df_use["rating"].values.astype(np.float32)

        rows = df_use["user_idx"].values
        cols = df_use["item_idx"].values
        R    = sparse.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

        # Column-normalise so X^T @ X gives cosine similarity
        col_norms = sparse.linalg.norm(R, axis=0)
        col_norms[col_norms == 0] = 1.0  # avoid div-by-zero for cold items
        R_norm = R.multiply(1.0 / col_norms).tocsr()

        # Cosine sim: (num_items x num_users) @ (num_users x num_items) = (num_items x num_items)
        sim = (R_norm.T @ R_norm).tolil()
        sim.setdiag(0.0)        # ignore self-similarity (i, i)
        sim = sim.tocsr()

        # Keep only top_k_neighbors per row to control memory + improve quality
        sim = self._keep_top_k_per_row(sim, self.top_k_neighbors)

        self._sim = sim

        # Cache per-user seen items + their ratings for fast scoring
        self._user_items   = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        if self.use_implicit:
            ratings_df = train_df[train_df["rating"] >= RELEVANCE_THRESHOLD]
        else:
            ratings_df = train_df
        self._user_ratings = (
            ratings_df.groupby("user_idx")[["item_idx", "rating"]]
            .apply(lambda g: dict(zip(g["item_idx"], g["rating"].astype(float))))
            .to_dict()
        )

        nnz = sim.nnz
        density = nnz / (num_items * num_items) if num_items else 0.0
        self._trained = True
        print(
            f"[ItemKNN] Trained  items={num_items:,}  users={num_users:,}  "
            f"top_k_neighbors={self.top_k_neighbors}  "
            f"sim_nnz={nnz:,}  density={density:.4%}"
        )
        return self

    @staticmethod
    def _keep_top_k_per_row(sim: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
        """Zero out all but the top-k entries in each row of `sim` (sparse-safe)."""
        sim   = sim.tocsr().copy()
        rows, cols, data = [], [], []
        for i in range(sim.shape[0]):
            start, end = sim.indptr[i], sim.indptr[i + 1]
            if end - start <= k:
                rows.extend([i] * (end - start))
                cols.extend(sim.indices[start:end])
                data.extend(sim.data[start:end])
                continue
            row_data = sim.data[start:end]
            row_cols = sim.indices[start:end]
            top_idx  = np.argpartition(-row_data, kth=k - 1)[:k]
            rows.extend([i] * k)
            cols.extend(row_cols[top_idx])
            data.extend(row_data[top_idx])
        return sparse.csr_matrix((data, (rows, cols)), shape=sim.shape)

    # ── predict ────────────────────────────────────────────────────────
    def predict_score(self, user_idx: int, item_idx: int) -> float:
        """Score = sum of similarities between item_idx and user's seen items."""
        seen_ratings = self._user_ratings.get(user_idx, {})
        if not seen_ratings or item_idx >= self._num_items:
            return 0.0

        # Pull row item_idx of similarity matrix
        start, end = self._sim.indptr[item_idx], self._sim.indptr[item_idx + 1]
        if start == end:
            return 0.0

        nbrs   = self._sim.indices[start:end]
        sims   = self._sim.data[start:end]
        score  = 0.0
        for j_idx, s in zip(nbrs, sims):
            r = seen_ratings.get(int(j_idx))
            if r is not None:
                score += s * r
        return float(score)

    def top_k(
        self,
        user_idx:        int,
        candidate_items: List[int],
        k:               int = 10,
    ) -> List[Tuple[int, float]]:
        """Score every candidate and return the top k."""
        scored = [(i, self.predict_score(user_idx, i)) for i in candidate_items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ── persist ────────────────────────────────────────────────────────
    def save(self) -> None:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        print(f"[ItemKNN] Saved → {MODEL_PATH}")

    @classmethod
    def load(cls) -> "ItemKNNModel":
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"[ItemKNN] Loaded ← {MODEL_PATH}")
        return model
