"""
Bayesian Personalized Ranking (Rendle et al., UAI 2009).

Why include it?
───────────────
BPR is the canonical *pairwise* ranking baseline for implicit feedback.
SVD optimises for rating prediction (point-wise MSE) and NCF optimises
binary classification (point-wise BCE) — neither is directly optimising
for ranking. BPR maximises the probability that a positive item ranks
above a negative item:

    L = -sum log σ( score(u, i) - score(u, j) )  +  λ ||θ||^2

where i is a positive item, j is a sampled negative.

Architecture
────────────
Same matrix factorisation as SVD (user/item embeddings + biases),
but trained end-to-end with a pairwise hinge-style objective using
the BPR loss. Empirically this consistently outperforms point-wise
MF on top-k ranking tasks.

Public API mirrors the other models:
    model.fit(train_df, num_users, num_items)
    model.predict_score(u, i)
    model.top_k(u, candidates, k=10)
    model.save() / model.load()
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH    = ARTIFACTS_DIR / "bpr_model.pt"
RELEVANCE_THRESHOLD = 4.0


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch architecture
# ──────────────────────────────────────────────────────────────────────────────

class BPRNet(nn.Module):
    """Plain matrix factorisation with user / item biases (no MLP)."""

    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64) -> None:
        super().__init__()
        self.user_emb  = nn.Embedding(num_users, emb_dim)
        self.item_emb  = nn.Embedding(num_items, emb_dim)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.user_emb.weight,  mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight,  mean=0.0, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Score = u . i + b_i"""
        u_vec = self.user_emb(u)
        i_vec = self.item_emb(i)
        b_i   = self.item_bias(i).squeeze(-1)
        return (u_vec * i_vec).sum(dim=-1) + b_i

    def forward(
        self,
        u: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
    ) -> torch.Tensor:
        """Return positive minus negative score for every triplet."""
        return self.score(u, i) - self.score(u, j)


# ──────────────────────────────────────────────────────────────────────────────
# Pairwise dataset: (user, pos_item, neg_item)
# ──────────────────────────────────────────────────────────────────────────────

class BPRDataset(Dataset):
    """
    For each positive interaction (u, i), sample a random negative j (item u has
    not interacted with) at __getitem__ time. This re-samples every epoch which
    acts as implicit data augmentation.
    """

    def __init__(self, df: pd.DataFrame, num_items: int, seed: int = 42):
        mask    = df["rating"] >= RELEVANCE_THRESHOLD
        df_pos  = df[mask].reset_index(drop=True)
        self.users = df_pos["user_idx"].values.astype(np.int64)
        self.items = df_pos["item_idx"].values.astype(np.int64)
        self.num_items = num_items
        self.user_pos = df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        u   = int(self.users[idx])
        i   = int(self.items[idx])
        pos = self.user_pos.get(u, set())
        # Rejection sampling — usually 1 try suffices since #items >> |pos|
        while True:
            j = int(self.rng.integers(0, self.num_items))
            if j not in pos:
                return u, i, j


# ──────────────────────────────────────────────────────────────────────────────
# Trainer / wrapper
# ──────────────────────────────────────────────────────────────────────────────

class BPRModel:
    """
    Parameters
    ----------
    num_users  : total number of users
    num_items  : total number of items
    emb_dim    : embedding dimension (default 64)
    lr         : Adam learning rate (default 1e-3)
    weight_decay : L2 regularisation (BPR's λ) (default 1e-5)
    device     : "auto" | "cpu" | "cuda"
    """

    def __init__(
        self,
        num_users:    int,
        num_items:    int,
        emb_dim:      int   = 64,
        lr:           float = 1e-3,
        weight_decay: float = 1e-5,
        device:       str   = "auto",
    ):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.net = BPRNet(num_users, num_items, emb_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )
        self._num_users  = num_users
        self._num_items  = num_items
        self._emb_dim    = emb_dim
        self._trained    = False
        self.history     = []

    # ── train ──────────────────────────────────────────────────────────
    def fit(
        self,
        train_df:    pd.DataFrame,
        epochs:      int = 10,
        batch_size:  int = 1024,
        num_workers: int = 0,
    ) -> "BPRModel":
        """Train BPR with pairwise BPR-loss for `epochs`."""
        ds     = BPRDataset(train_df, num_items=self._num_items)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )

        for epoch in range(1, epochs + 1):
            self.net.train()
            total, n = 0.0, 0
            for u, i, j in loader:
                u = u.to(self.device)
                i = i.to(self.device)
                j = j.to(self.device)

                # BPR loss = -log σ(score_pos - score_neg)
                diff = self.net(u, i, j)
                loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total += loss.item() * len(u)
                n     += len(u)

            avg = total / n if n else 0.0
            self.history.append({"epoch": epoch, "loss": avg})
            print(f"[BPR] Epoch {epoch:02d}/{epochs}  bpr_loss={avg:.4f}")

        self._trained = True
        return self

    # ── inference ──────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_score(self, user_idx: int, item_idx: int) -> float:
        self.net.eval()
        u = torch.LongTensor([user_idx]).to(self.device)
        i = torch.LongTensor([item_idx]).to(self.device)
        return float(self.net.score(u, i).item())

    @torch.no_grad()
    def top_k(
        self,
        user_idx:        int,
        candidate_items: List[int],
        k:               int = 10,
    ) -> List[Tuple[int, float]]:
        """Batched top-k scoring over candidate set."""
        self.net.eval()
        u = torch.LongTensor([user_idx] * len(candidate_items)).to(self.device)
        i = torch.LongTensor(candidate_items).to(self.device)
        scores = self.net.score(u, i).cpu().tolist()
        ranked = sorted(zip(candidate_items, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    # ── persist ────────────────────────────────────────────────────────
    def save(self) -> None:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "num_users":  self._num_users,
                "num_items":  self._num_items,
                "emb_dim":    self._emb_dim,
            },
            MODEL_PATH,
        )
        print(f"[BPR] Saved → {MODEL_PATH}")

    @classmethod
    def load(cls, device: str = "auto") -> "BPRModel":
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = cls(
            num_users=ckpt["num_users"],
            num_items=ckpt["num_items"],
            emb_dim=ckpt["emb_dim"],
            device=device,
        )
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.to(model.device)
        model._trained = True
        print(f"[BPR] Loaded ← {MODEL_PATH}")
        return model
