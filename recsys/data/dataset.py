"""
PyTorch Dataset and DataLoader for NCF training.

Label convention
────────────────
  1  — positive interaction (rating >= RELEVANCE_THRESHOLD)
  0  — negative interaction (item the user never rated, sampled uniformly)

Negative sampling
─────────────────
For each positive (user, item) pair in the training set we sample
NEG_SAMPLE_RATIO random items that the user has NOT interacted with.
Validation / test sets contain only the original observed interactions
(no extra negatives) — the ranking evaluator handles negatives itself.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Set, Tuple

RELEVANCE_THRESHOLD = 4.0
NEG_SAMPLE_RATIO    = 4        # negatives per positive interaction


class NCFDataset(Dataset):
    """
    Holds (user_idx, item_idx, label) triples as pre-built tensors.

    Parameters
    ----------
    df          : DataFrame with columns [user_idx, item_idx, rating]
    num_items   : total item vocabulary size (used for negative sampling)
    neg_sample  : if True, generate NEG_SAMPLE_RATIO negatives per positive
    seed        : random seed for negative sampling reproducibility
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        num_items:  int,
        neg_sample: bool = False,
        seed:       int  = 42,
    ) -> None:
        # Binary relevance labels for the observed interactions
        users  = df["user_idx"].values.astype(np.int64)
        items  = df["item_idx"].values.astype(np.int64)
        labels = (df["rating"].values >= RELEVANCE_THRESHOLD).astype(np.float32)

        if neg_sample:
            users, items, labels = self._add_negatives(
                users, items, labels, df, num_items, seed
            )

        self.users  = torch.from_numpy(users)
        self.items  = torch.from_numpy(items)
        self.labels = torch.from_numpy(labels)

    @staticmethod
    def _add_negatives(
        users:     np.ndarray,
        items:     np.ndarray,
        labels:    np.ndarray,
        df:        pd.DataFrame,
        num_items: int,
        seed:      int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For every user, sample NEG_SAMPLE_RATIO * |positives| negatives.
        Uses rejection sampling to guarantee the sampled item was never
        rated by that user.
        """
        rng = np.random.default_rng(seed)

        # Per-user set of ALL observed items (to reject during sampling)
        pos_sets: Dict[int, Set[int]] = (
            df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )

        neg_users, neg_items = [], []

        for uid, pos_set in pos_sets.items():
            n_neg  = len(pos_set) * NEG_SAMPLE_RATIO
            found  = 0
            while found < n_neg:
                # Sample a batch for efficiency, then filter
                batch = rng.integers(0, num_items, size=n_neg * 2)
                for cand in batch:
                    if found >= n_neg:
                        break
                    if int(cand) not in pos_set:
                        neg_users.append(uid)
                        neg_items.append(int(cand))
                        found += 1

        neg_users  = np.array(neg_users,  dtype=np.int64)
        neg_items  = np.array(neg_items,  dtype=np.int64)
        neg_labels = np.zeros(len(neg_users), dtype=np.float32)

        return (
            np.concatenate([users, neg_users]),
            np.concatenate([items, neg_items]),
            np.concatenate([labels, neg_labels]),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.labels[idx]


def make_loaders(
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    test_df:     pd.DataFrame,
    num_items:   int,
    batch_size:  int = 1024,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for train / val / test splits.

    Negative sampling is applied only to the training set.
    Val and test loaders expose raw interactions for loss monitoring
    (ranking evaluation uses model.top_k directly, not these loaders).
    """
    train_ds = NCFDataset(train_df, num_items, neg_sample=True)
    val_ds   = NCFDataset(val_df,   num_items, neg_sample=False)
    test_ds  = NCFDataset(test_df,  num_items, neg_sample=False)

    print(
        f"[dataset] train={len(train_ds):,} samples "
        f"(~{NEG_SAMPLE_RATIO}:1 neg ratio)  "
        f"val={len(val_ds):,}  test={len(test_ds):,}"
    )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )
