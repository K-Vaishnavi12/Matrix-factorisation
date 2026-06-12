"""
Neural Collaborative Filtering — pure MLP variant.

Architecture (He et al., WWW 2017 — MLP branch)
────────────────────────────────────────────────
  user_emb  [num_users, emb_dim]   ← learnable
  item_emb  [num_items, emb_dim]   ← learnable

  concat = [user_emb(u) ‖ item_emb(i)]   (size: emb_dim * 2)
  h = MLP(concat)                          (ReLU + Dropout between layers)
  output = sigmoid( Linear(h → 1) )        (implicit relevance probability)

Loss : Binary Cross-Entropy (BCELoss)
Labels: 1 if rating ≥ 4, else 0  (implicit binary feedback)
"""

import torch
import torch.nn as nn
from typing import List


class NCF(nn.Module):
    """
    Parameters
    ----------
    num_users   : vocabulary size for users
    num_items   : vocabulary size for items
    emb_dim     : embedding dimension for both user and item (default 64)
    hidden_dims : list of hidden layer widths for the MLP tower (default [128, 64, 32])
    dropout     : dropout probability applied after every hidden layer (default 0.2)
    """

    def __init__(
        self,
        num_users:   int,
        num_items:   int,
        emb_dim:     int        = 64,
        hidden_dims: List[int]  = None,
        dropout:     float      = 0.2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]

        # ── Learnable embedding matrices ────────────────────────────────
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        # ── MLP tower: concat → hidden layers → output ─────────────────
        layers: List[nn.Module] = []
        in_dim = emb_dim * 2          # user ‖ item concat
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        # ── Final prediction layer ──────────────────────────────────────
        self.output_layer = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        user: torch.Tensor,   # (batch,) LongTensor of user indices
        item: torch.Tensor,   # (batch,) LongTensor of item indices
    ) -> torch.Tensor:        # (batch,) FloatTensor of probabilities in [0, 1]
        u = self.user_emb(user)              # (batch, emb_dim)
        i = self.item_emb(item)              # (batch, emb_dim)
        x = torch.cat([u, i], dim=-1)        # (batch, emb_dim*2)
        h = self.mlp(x)                      # (batch, hidden_dims[-1])
        return torch.sigmoid(self.output_layer(h)).squeeze(-1)
