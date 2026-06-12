"""
Download and filter the Amazon Reviews 2023 dataset from HuggingFace.

Supports
────────
- Single category:    load_reviews(category="All_Beauty")
- Multiple categories: load_reviews(categories=["All_Beauty", "Office_Products"])
- Item metadata:       load_item_metadata(category)  → DataFrame[item_id, title, main_category]

Sparsity filter is iterative: drop users/items below MIN_*_INTERACTIONS,
re-check, repeat until convergence (otherwise removing items can re-create
sparse users and vice-versa).
"""
from typing import List, Optional, Union

from datasets import load_dataset
import pandas as pd


HF_REPO  = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_CATEGORY = "All_Beauty"

# ──────────────────────────────────────────────────────────────────────────────
# Tunable filters
# ──────────────────────────────────────────────────────────────────────────────

MIN_USER_INTERACTIONS = 5     # drop users with fewer than this many ratings
MIN_ITEM_INTERACTIONS = 5     # drop items rated by fewer than this many users


# ──────────────────────────────────────────────────────────────────────────────
# Reviews (interactions)
# ──────────────────────────────────────────────────────────────────────────────

def _download_one(category: str) -> pd.DataFrame:
    """Download a single category's review split."""
    print(f"[loader] Downloading reviews for '{category}' …")
    ds = load_dataset(
        HF_REPO,
        f"raw_review_{category}",
        split="full",
        trust_remote_code=True,
    )
    df = ds.to_pandas()[["user_id", "parent_asin", "rating"]]
    df.rename(columns={"parent_asin": "item_id"}, inplace=True)
    df["category"] = category   # tag origin so we can analyse per-cat later
    df.dropna(subset=["user_id", "item_id", "rating"], inplace=True)
    df["rating"] = df["rating"].astype(float)
    print(f"[loader]   raw rows: {len(df):,}")
    return df


def _filter_sparse(df: pd.DataFrame) -> pd.DataFrame:
    """Iteratively drop sparse users / items until convergence."""
    prev = -1
    iteration = 0
    while prev != len(df):
        prev = len(df)
        iteration += 1
        u_counts = df["user_id"].value_counts()
        i_counts = df["item_id"].value_counts()
        df = df[
            df["user_id"].isin(u_counts[u_counts >= MIN_USER_INTERACTIONS].index) &
            df["item_id"].isin(i_counts[i_counts >= MIN_ITEM_INTERACTIONS].index)
        ]
        print(
            f"[loader]   iter {iteration}: rows={len(df):,}  "
            f"users={df['user_id'].nunique():,}  items={df['item_id'].nunique():,}"
        )
    return df.reset_index(drop=True)


def load_reviews(
    category:   Optional[str]              = None,
    categories: Optional[List[str]]        = None,
) -> pd.DataFrame:
    """
    Return a cleaned, deduplicated, sparsity-filtered DataFrame.

    Parameters
    ----------
    category   : single category name (mutually exclusive with `categories`)
    categories : list of category names — concatenated before filtering
                 (use this to scale up to ~1M+ interactions)

    If neither is given, falls back to DEFAULT_CATEGORY.

    Returns
    -------
    DataFrame with columns [user_id, item_id, rating, category]
    """
    if categories:
        if category:
            raise ValueError("Provide `category` OR `categories`, not both.")
        cats = categories
    else:
        cats = [category or DEFAULT_CATEGORY]

    # Concatenate raw downloads
    parts = [_download_one(c) for c in cats]
    df    = pd.concat(parts, ignore_index=True)
    print(f"[loader] Combined raw rows across {len(cats)} category(ies): {len(df):,}")

    # Dedup (same user+item could be rated in two categories — keep latest)
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last").reset_index(drop=True)
    print(f"[loader] Dedup removed {before - len(df):,} duplicate (user, item) pairs")

    # Sparsity filter
    df = _filter_sparse(df)

    print(
        f"[loader] Final  rows={len(df):,}  "
        f"users={df['user_id'].nunique():,}  items={df['item_id'].nunique():,}  "
        f"categories={df['category'].nunique()}"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Item metadata (used by the content-hybrid model)
# ──────────────────────────────────────────────────────────────────────────────

def load_item_metadata(
    category:   Optional[str]              = None,
    categories: Optional[List[str]]        = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with [item_id, title, main_category] for the given
    category(ies). Used as side information by ContentHybridModel.

    Parameters mirror load_reviews().
    """
    if categories:
        cats = categories
    else:
        cats = [category or DEFAULT_CATEGORY]

    frames = []
    for cat in cats:
        print(f"[loader] Downloading item metadata for '{cat}' …")
        try:
            ds = load_dataset(
                HF_REPO,
                f"raw_meta_{cat}",
                split="full",
                trust_remote_code=True,
            )
            df = ds.to_pandas()[["parent_asin", "title", "main_category"]]
            df.rename(columns={"parent_asin": "item_id"}, inplace=True)
            df["main_category"] = df["main_category"].fillna(cat)
            frames.append(df)
            print(f"[loader]   raw meta rows: {len(df):,}")
        except Exception as e:
            print(f"[loader]   WARNING: failed to load metadata for {cat}: {e}")

    if not frames:
        return pd.DataFrame(columns=["item_id", "title", "main_category"])

    meta = pd.concat(frames, ignore_index=True)
    meta = meta.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"], keep="first")
    print(f"[loader] Item metadata final: {len(meta):,} unique items")
    return meta
