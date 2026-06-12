"""
ID mapping + user-wise 80/10/10 train/val/test split.
"""
import pickle
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd


ARTIFACTS_DIR = Path("artifacts")


def build_mappings(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Map raw string IDs to contiguous 0-based integer indices."""
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())

    user2idx: Dict[str, int] = {u: i for i, u in enumerate(users)}
    item2idx: Dict[str, int] = {it: i for i, it in enumerate(items)}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["item_idx"] = df["item_id"].map(item2idx)
    return df, user2idx, item2idx


def user_wise_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each user sort interactions by implicit time order (row order within
    the dataset acts as a proxy since we can't guarantee timestamps exist).
    Put the last test_frac items in test, next val_frac in val, rest in train.
    """
    df = df.sort_values(["user_idx"]).copy()
    train_rows, val_rows, test_rows = [], [], []

    for _, grp in df.groupby("user_idx"):
        n = len(grp)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        train_rows.append(grp.iloc[: n - n_test - n_val])
        val_rows.append(grp.iloc[n - n_test - n_val : n - n_test])
        test_rows.append(grp.iloc[n - n_test :])

    train = pd.concat(train_rows).reset_index(drop=True)
    val   = pd.concat(val_rows).reset_index(drop=True)
    test  = pd.concat(test_rows).reset_index(drop=True)

    print(f"[preprocessor] train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


def save_artifacts(user2idx: Dict, item2idx: Dict) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(ARTIFACTS_DIR / "user2idx.pkl", "wb") as f:
        pickle.dump(user2idx, f)
    with open(ARTIFACTS_DIR / "item2idx.pkl", "wb") as f:
        pickle.dump(item2idx, f)
    print(f"[preprocessor] Mappings saved to {ARTIFACTS_DIR}/")


def load_artifacts() -> Tuple[Dict, Dict]:
    with open(ARTIFACTS_DIR / "user2idx.pkl", "rb") as f:
        user2idx = pickle.load(f)
    with open(ARTIFACTS_DIR / "item2idx.pkl", "rb") as f:
        item2idx = pickle.load(f)
    return user2idx, item2idx
