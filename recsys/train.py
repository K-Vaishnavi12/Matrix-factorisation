"""
Train all five recommender models on the same data splits.

Models trained
──────────────
1. Popularity        — non-personalised baseline
2. ItemKNN           — classical CF baseline
3. SVD               — Surprise matrix factorisation (rating prediction)
4. NCF               — Neural Collaborative Filtering (point-wise BCE)
5. BPR               — Bayesian Personalized Ranking (pairwise loss)
6. ContentHybrid     — two-tower with item content features (cold-start)

Each model exposes the same .top_k() interface so a single evaluation
loop can score them with NDCG@10 / MAP@10.

Usage
─────
    # default: All_Beauty (small, ~1 min)
    python -m recsys.train

    # larger (multiple categories, ~30 min):
    python -m recsys.train --categories All_Beauty Office_Products Toys_and_Games

    # skip slow content model:
    python -m recsys.train --skip-content
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from recsys.data.dataset      import make_loaders
from recsys.data.loader       import load_reviews, load_item_metadata
from recsys.data.preprocessor import build_mappings, save_artifacts, user_wise_split
from recsys.models.bpr            import BPRModel
from recsys.models.content_hybrid import (
    ContentHybridModel,
    build_item_features,
)
from recsys.models.item_knn       import ItemKNNModel
from recsys.models.ncf_model      import NCFModel
from recsys.models.popularity     import PopularityModel
from recsys.models.svd_model      import tune_svd

ARTIFACTS_DIR = Path("artifacts")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train all recsys baselines")
    p.add_argument(
        "--category",
        type=str,
        default=None,
        help="Single Amazon Reviews category (default: All_Beauty)",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Multiple categories to combine (overrides --category)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs for NCF / BPR / ContentHybrid (default: 10)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size (default: 1024)",
    )
    p.add_argument(
        "--skip-content",
        action="store_true",
        help="Skip the ContentHybrid model (faster, no metadata download)",
    )
    p.add_argument(
        "--skip-svd",
        action="store_true",
        help="Skip the SVD grid search (very slow on big data)",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_data(args) -> tuple:
    raw_df = load_reviews(category=args.category, categories=args.categories)
    df, user2idx, item2idx = build_mappings(raw_df)
    save_artifacts(user2idx, item2idx)

    train_df, val_df, test_df = user_wise_split(df)
    train_df.to_parquet(ARTIFACTS_DIR / "train.parquet", index=False)
    val_df.to_parquet(ARTIFACTS_DIR   / "val.parquet",   index=False)
    test_df.to_parquet(ARTIFACTS_DIR  / "test.parquet",  index=False)

    # Stash dataset stats for the README/results
    stats = {
        "categories":   args.categories or [args.category or "All_Beauty"],
        "interactions": len(df),
        "num_users":    len(user2idx),
        "num_items":    len(item2idx),
        "train_size":   len(train_df),
        "val_size":     len(val_df),
        "test_size":    len(test_df),
        "sparsity":     round(
            1 - len(df) / (len(user2idx) * len(item2idx)), 6
        ) if user2idx and item2idx else None,
    }
    with open(ARTIFACTS_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[train] dataset stats → {stats}")

    return df, train_df, val_df, test_df, user2idx, item2idx


def _train_popularity(train_df) -> float:
    print("\n[train] ── Popularity ─────────────────────────────────────────")
    t0 = time.perf_counter()
    pop = PopularityModel(method="count").fit(train_df)
    pop.save()
    return round(time.perf_counter() - t0, 2)


def _train_itemknn(train_df) -> float:
    print("\n[train] ── ItemKNN ─────────────────────────────────────────────")
    t0 = time.perf_counter()
    knn = ItemKNNModel(top_k_neighbors=50, use_implicit=True).fit(train_df)
    knn.save()
    return round(time.perf_counter() - t0, 2)


def _train_svd(train_df) -> float:
    print("\n[train] ── SVD ─────────────────────────────────────────────────")
    t0  = time.perf_counter()
    svd = tune_svd(train_df, cv=3)
    svd.save()
    return round(time.perf_counter() - t0, 2)


def _train_ncf(train_loader, val_loader, num_users, num_items, args) -> float:
    print("\n[train] ── NCF ─────────────────────────────────────────────────")
    ncf = NCFModel(
        num_users=num_users,
        num_items=num_items,
        emb_dim=64,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        lr=1e-3,
    )
    t0 = time.perf_counter()
    ncf.fit(train_loader, val_loader, epochs=args.epochs, patience=3)
    ncf.save()
    return round(time.perf_counter() - t0, 2)


def _train_bpr(train_df, num_users, num_items, args) -> float:
    print("\n[train] ── BPR ─────────────────────────────────────────────────")
    bpr = BPRModel(
        num_users=num_users,
        num_items=num_items,
        emb_dim=64,
        lr=1e-3,
        weight_decay=1e-5,
    )
    t0 = time.perf_counter()
    bpr.fit(train_df, epochs=args.epochs, batch_size=args.batch_size)
    bpr.save()
    return round(time.perf_counter() - t0, 2)


def _train_content_hybrid(
    train_loader,
    val_loader,
    num_users,
    num_items,
    item2idx,
    args,
) -> float:
    print("\n[train] ── ContentHybrid ───────────────────────────────────────")
    meta_df = load_item_metadata(category=args.category, categories=args.categories)
    if len(meta_df) == 0:
        print("[train] No metadata available — skipping ContentHybrid")
        return 0.0
    item_features, _ = build_item_features(meta_df, item2idx, text_dim=128)

    model = ContentHybridModel(
        num_users     = num_users,
        num_items     = num_items,
        item_features = item_features,
        emb_dim       = 64,
        hidden_dims   = [128, 64, 32],
        dropout       = 0.2,
        lr            = 1e-3,
    )
    t0 = time.perf_counter()
    model.fit(train_loader, val_loader, epochs=args.epochs, patience=3)
    model.save()
    return round(time.perf_counter() - t0, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # ── 1. Data prep ──────────────────────────────────────────────────────
    df, train_df, val_df, test_df, user2idx, item2idx = _prepare_data(args)
    num_users, num_items = len(user2idx), len(item2idx)

    train_loader, val_loader, _ = make_loaders(
        train_df, val_df, test_df,
        num_items=num_items,
        batch_size=args.batch_size,
    )

    # ── 2. Train every model ──────────────────────────────────────────────
    times: Dict[str, float] = {}
    times["popularity"] = _train_popularity(train_df)
    times["itemknn"]    = _train_itemknn(train_df)

    if not args.skip_svd:
        times["svd"] = _train_svd(train_df)

    times["ncf"] = _train_ncf(train_loader, val_loader, num_users, num_items, args)
    times["bpr"] = _train_bpr(train_df, num_users, num_items, args)

    if not args.skip_content:
        ch_time = _train_content_hybrid(
            train_loader, val_loader, num_users, num_items, item2idx, args,
        )
        if ch_time > 0:
            times["content_hybrid"] = ch_time

    # ── 3. Persist timings ────────────────────────────────────────────────
    timing_path = ARTIFACTS_DIR / "training_times.json"
    with open(timing_path, "w") as f:
        json.dump(times, f, indent=2)

    print("\n[train] Training summary")
    for name, t in times.items():
        print(f"  {name:<18} {t:>8.2f}s")
    print(f"[train] Times saved → {timing_path}")
    print("[train] All artifacts in ./artifacts/")


if __name__ == "__main__":
    main()
