"""
NCFModel — training wrapper around the NCF architecture.

Provides the same fit() / predict_score() / top_k() / save() / load()
interface as SVDModel so the API and evaluate.py work without changes.
"""
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from recsys.models.ncf import NCF

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH    = ARTIFACTS_DIR / "ncf_model.pt"


class NCFModel:
    """
    Parameters
    ----------
    num_users  : total number of unique users
    num_items  : total number of unique items
    emb_dim    : embedding dimension (default 64)
    hidden_dims: MLP hidden layer sizes (default [128, 64, 32])
    dropout    : dropout probability in MLP (default 0.2)
    lr         : Adam learning rate (default 1e-3)
    device     : "auto" | "cpu" | "cuda"
    """

    def __init__(
        self,
        num_users:   int,
        num_items:   int,
        emb_dim:     int        = 64,
        hidden_dims: List[int]  = None,
        dropout:     float      = 0.2,
        lr:          float      = 1e-3,
        device:      str        = "auto",
    ) -> None:
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.net = NCF(
            num_users=num_users,
            num_items=num_items,
            emb_dim=emb_dim,
            hidden_dims=hidden_dims or [128, 64, 32],
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self._trained  = False

        # stored for checkpoint reconstruction
        self._num_users  = num_users
        self._num_items  = num_items
        self._emb_dim    = emb_dim
        self._hidden_dims = hidden_dims or [128, 64, 32]
        self._dropout    = dropout

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int   = 10,
        patience:     int   = 3,
    ) -> "NCFModel":
        """
        Train with BCE loss and Adam.  Early stopping on validation loss.
        Best weights are checkpointed and restored at the end.

        Prints train_loss and val_loss every epoch.
        """
        best_val   = float("inf")
        no_improve = 0
        history    = []

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss   = self._eval_loss(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            print(
                f"[NCF] Epoch {epoch:02d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

            if val_loss < best_val:
                best_val   = val_loss
                no_improve = 0
                self._save_weights()          # checkpoint best
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[NCF] Early stopping triggered at epoch {epoch}  (best val={best_val:.4f})")
                    break

        self._load_weights()      # restore best checkpoint
        self._trained = True
        self.history  = history
        return self

    def _train_epoch(self, loader: DataLoader) -> float:
        self.net.train()
        total_loss, n = 0.0, 0
        for users, items, labels in loader:
            users  = users.to(self.device)
            items  = items.to(self.device)
            labels = labels.to(self.device)

            preds  = self.net(users, items)
            loss   = self.criterion(preds, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(labels)
            n          += len(labels)
        return total_loss / n if n else 0.0

    @torch.no_grad()
    def _eval_loss(self, loader: DataLoader) -> float:
        self.net.eval()
        total, n = 0.0, 0
        for users, items, labels in loader:
            users  = users.to(self.device)
            items  = items.to(self.device)
            labels = labels.to(self.device)
            preds  = self.net(users, items)
            total += self.criterion(preds, labels).item() * len(labels)
            n     += len(labels)
        return total / n if n else 0.0

    # ── Inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_score(self, user_idx: int, item_idx: int) -> float:
        """Predicted relevance probability for a single (user, item) pair."""
        self.net.eval()
        u = torch.LongTensor([user_idx]).to(self.device)
        i = torch.LongTensor([item_idx]).to(self.device)
        return self.net(u, i).item()

    @torch.no_grad()
    def top_k(
        self,
        user_idx:        int,
        candidate_items: List[int],
        k:               int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Score all candidate items in a single batched forward pass.
        Returns [(item_idx, score), …] sorted by score descending, length ≤ k.
        """
        self.net.eval()
        users  = torch.LongTensor([user_idx] * len(candidate_items)).to(self.device)
        items  = torch.LongTensor(candidate_items).to(self.device)
        scores = self.net(users, items).cpu().tolist()
        ranked = sorted(zip(candidate_items, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    # ── Persistence ────────────────────────────────────────────────────────

    def _save_weights(self) -> None:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        torch.save(
            {
                "state_dict":  self.net.state_dict(),
                "num_users":   self._num_users,
                "num_items":   self._num_items,
                "emb_dim":     self._emb_dim,
                "hidden_dims": self._hidden_dims,
                "dropout":     self._dropout,
            },
            MODEL_PATH,
        )

    def _load_weights(self) -> None:
        ckpt = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["state_dict"])

    def save(self) -> None:
        self._save_weights()
        print(f"[NCFModel] Saved → {MODEL_PATH}")

    @classmethod
    def load(cls, device: str = "auto") -> "NCFModel":
        ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = cls(
            num_users=ckpt["num_users"],
            num_items=ckpt["num_items"],
            emb_dim=ckpt["emb_dim"],
            hidden_dims=ckpt["hidden_dims"],
            dropout=ckpt["dropout"],
            device=device,
        )
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.to(model.device)
        model._trained = True
        print(f"[NCFModel] Loaded ← {MODEL_PATH}")
        return model
