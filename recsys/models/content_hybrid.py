"""
Content-aware hybrid recommender — addresses the cold-start problem.

Why include it?
───────────────
Pure collaborative filtering (SVD, NCF, BPR) cannot recommend items that
have zero training interactions. In e-commerce this is the **cold-start
problem** and is one of the biggest practical issues with classical recsys.

Solution
────────
Combine an NCF-style user embedding with a *content-derived item vector*:

    user_emb (learned)              ─┐
                                     ├─→ concat → MLP → sigmoid
    item_emb = f(item_text, item_cat)┘

For unseen items, item_emb is computed on-the-fly from their metadata,
so a fresh ASIN can be ranked even with zero training interactions.

Item features
─────────────
We use the hashing-trick over tokenised item titles + a one-hot main category.
This avoids requiring a language model and keeps the project dependency-light:

    title_hash_vec(d=128) ⊕ category_onehot(C) → item_content_vec
    item_content_vec → dense projection → item_emb (emb_dim)

This is the standard "tower" model used by Amazon, YouTube and others
(two-tower / dual-encoder architecture, simplified).

Public API
──────────
    model = ContentHybridModel(num_users, num_items, content_dim, emb_dim)
    model.fit(train_loader, val_loader, item_features, ...)
    model.predict_score_with_features(user_idx, item_features_vec)
    model.top_k(user_idx, candidate_items, k=10)
    model.save() / model.load()

Cold-start use:
    score = model.predict_score_with_features(uid, content_vec)
    # works even if item_idx == -1 (unseen)
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH    = ARTIFACTS_DIR / "content_hybrid_model.pt"


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction (hashing-trick)
# ──────────────────────────────────────────────────────────────────────────────

def hash_token(token: str, dim: int) -> int:
    """Deterministic hash → bucket index in [0, dim)."""
    return abs(hash(token)) % dim


def text_to_hash_vector(text: str, dim: int = 128) -> np.ndarray:
    """
    Convert a free-text string to a fixed-length hashed bag-of-words vector.

    Each whitespace-separated token increments one bucket; vector is then
    L2-normalised. No vocabulary required — works for any string.
    """
    vec = np.zeros(dim, dtype=np.float32)
    if not isinstance(text, str) or not text:
        return vec
    for tok in text.lower().split():
        vec[hash_token(tok, dim)] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def build_item_features(
    item_meta_df: pd.DataFrame,
    item2idx:     Dict[str, int],
    text_dim:     int = 128,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a content matrix [num_items, text_dim + num_categories].

    Parameters
    ----------
    item_meta_df : DataFrame with columns [item_id, title, main_category]
    item2idx     : mapping from raw item_id → integer item_idx
    text_dim     : dimension of the title hash vector

    Returns
    -------
    features : np.ndarray, shape (num_items, text_dim + num_categories)
    categories : list of category names (column ordering for the one-hot block)
    """
    num_items  = len(item2idx)
    cats       = sorted(item_meta_df["main_category"].fillna("UNKNOWN").unique())
    cat2col    = {c: i for i, c in enumerate(cats)}
    num_cats   = len(cats)

    feat_dim   = text_dim + num_cats
    features   = np.zeros((num_items, feat_dim), dtype=np.float32)

    # Index meta by item_id for O(1) lookup
    meta_by_id = item_meta_df.set_index("item_id")

    n_with_meta = 0
    for item_id, idx in item2idx.items():
        if item_id not in meta_by_id.index:
            continue
        row = meta_by_id.loc[item_id]
        # Some items may have multiple meta rows — take first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        title = row.get("title", "") or ""
        cat   = row.get("main_category", "UNKNOWN") or "UNKNOWN"
        features[idx, :text_dim] = text_to_hash_vector(title, text_dim)
        features[idx, text_dim + cat2col.get(cat, 0)] = 1.0
        n_with_meta += 1

    print(
        f"[content] Built features  shape={features.shape}  "
        f"items_with_meta={n_with_meta}/{num_items}  "
        f"text_dim={text_dim}  num_categories={num_cats}"
    )
    return features, cats


# ──────────────────────────────────────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────────────────────────────────────

class ContentHybridNet(nn.Module):
    """
    Two-tower-lite: user embedding tower + item content-projection tower.

    Architecture
    ────────────
        user → user_emb (emb_dim)
        item_features (content_dim) → Linear → ReLU → Linear → item_emb (emb_dim)
        concat → MLP → sigmoid
    """

    def __init__(
        self,
        num_users:   int,
        content_dim: int,
        emb_dim:     int = 64,
        hidden_dims: List[int] = None,
        dropout:     float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]
        self.user_emb = nn.Embedding(num_users, emb_dim)

        # Project the content vector to the same emb_dim as the user
        self.item_proj = nn.Sequential(
            nn.Linear(content_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        # Top MLP that combines both towers
        layers = []
        in_dim = emb_dim * 2
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 1)

        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)

    def forward(
        self,
        user:          torch.Tensor,  # (B,) long
        item_features: torch.Tensor,  # (B, content_dim) float
    ) -> torch.Tensor:
        u = self.user_emb(user)
        i = self.item_proj(item_features)
        x = torch.cat([u, i], dim=-1)
        h = self.mlp(x)
        return torch.sigmoid(self.out(h)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Trainer / wrapper (mirrors NCFModel API)
# ──────────────────────────────────────────────────────────────────────────────

class ContentHybridModel:
    """
    Hybrid CF + content recommender that gracefully handles cold-start items.

    Parameters
    ----------
    num_users    : total users
    num_items    : total items (indexed 0..num_items-1)
    item_features: ndarray (num_items, content_dim) — pre-built content vectors
    emb_dim      : embedding dim (default 64)
    hidden_dims  : MLP widths
    dropout      : dropout probability
    lr           : Adam learning rate
    device       : auto/cpu/cuda
    """

    def __init__(
        self,
        num_users:     int,
        num_items:     int,
        item_features: np.ndarray,
        emb_dim:       int = 64,
        hidden_dims:   List[int] = None,
        dropout:       float = 0.2,
        lr:            float = 1e-3,
        device:        str = "auto",
    ):
        assert item_features.shape[0] == num_items, "item_features rows must equal num_items"
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.content_dim   = item_features.shape[1]
        self.item_features = torch.from_numpy(item_features).float().to(self.device)
        self.net = ContentHybridNet(
            num_users   = num_users,
            content_dim = self.content_dim,
            emb_dim     = emb_dim,
            hidden_dims = hidden_dims or [128, 64, 32],
            dropout     = dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

        self._num_users   = num_users
        self._num_items   = num_items
        self._emb_dim     = emb_dim
        self._hidden_dims = hidden_dims or [128, 64, 32]
        self._dropout     = dropout
        self._trained     = False
        self.history      = []

    # ── training ───────────────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int = 10,
        patience:     int = 3,
    ) -> "ContentHybridModel":
        """
        Train with BCE loss. The DataLoader yields (user, item_idx, label);
        we look up content vectors via self.item_features.
        """
        best_val   = float("inf")
        no_improve = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._epoch(train_loader, train=True)
            val_loss   = self._epoch(val_loader,   train=False)
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            print(
                f"[ContentHybrid] Epoch {epoch:02d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

            if val_loss < best_val:
                best_val   = val_loss
                no_improve = 0
                self._save_weights()
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[ContentHybrid] Early stop at epoch {epoch}  best_val={best_val:.4f}")
                    break

        self._load_weights()
        self._trained = True
        return self

    def _epoch(self, loader: DataLoader, train: bool) -> float:
        self.net.train() if train else self.net.eval()
        total, n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for users, items, labels in loader:
                users  = users.to(self.device)
                items  = items.to(self.device)
                labels = labels.to(self.device)
                feats  = self.item_features[items]
                preds  = self.net(users, feats)
                loss   = self.criterion(preds, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total += loss.item() * len(labels)
                n     += len(labels)
        return total / n if n else 0.0

    # ── inference ──────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_score(self, user_idx: int, item_idx: int) -> float:
        self.net.eval()
        u = torch.LongTensor([user_idx]).to(self.device)
        f = self.item_features[item_idx].unsqueeze(0)
        return float(self.net(u, f).item())

    @torch.no_grad()
    def predict_score_with_features(
        self,
        user_idx:          int,
        content_vec:       np.ndarray,
    ) -> float:
        """Score a user against an *unseen* item via its content vector (cold-start)."""
        assert content_vec.shape == (self.content_dim,), (
            f"Expected content_vec of shape ({self.content_dim},), got {content_vec.shape}"
        )
        self.net.eval()
        u = torch.LongTensor([user_idx]).to(self.device)
        f = torch.from_numpy(content_vec).float().unsqueeze(0).to(self.device)
        return float(self.net(u, f).item())

    @torch.no_grad()
    def top_k(
        self,
        user_idx:        int,
        candidate_items: List[int],
        k:               int = 10,
    ) -> List[Tuple[int, float]]:
        self.net.eval()
        u     = torch.LongTensor([user_idx] * len(candidate_items)).to(self.device)
        feats = self.item_features[candidate_items]
        scores = self.net(u, feats).cpu().tolist()
        ranked = sorted(zip(candidate_items, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    # ── persist ────────────────────────────────────────────────────────
    def _save_weights(self) -> None:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        torch.save(
            {
                "state_dict":    self.net.state_dict(),
                "num_users":     self._num_users,
                "num_items":     self._num_items,
                "content_dim":   self.content_dim,
                "emb_dim":       self._emb_dim,
                "hidden_dims":   self._hidden_dims,
                "dropout":       self._dropout,
                "item_features": self.item_features.cpu().numpy(),
            },
            MODEL_PATH,
        )

    def _load_weights(self) -> None:
        ckpt = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["state_dict"])

    def save(self) -> None:
        self._save_weights()
        print(f"[ContentHybrid] Saved → {MODEL_PATH}")

    @classmethod
    def load(cls, device: str = "auto") -> "ContentHybridModel":
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = cls(
            num_users     = ckpt["num_users"],
            num_items     = ckpt["num_items"],
            item_features = ckpt["item_features"],
            emb_dim       = ckpt["emb_dim"],
            hidden_dims   = ckpt["hidden_dims"],
            dropout       = ckpt["dropout"],
            device        = device,
        )
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.to(model.device)
        model._trained = True
        print(f"[ContentHybrid] Loaded ← {MODEL_PATH}")
        return model
