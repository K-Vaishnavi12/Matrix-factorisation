"""
Load every saved model and evaluate on the same test split with NDCG@10 / MAP@10.

Standard 1-vs-99 protocol (same eval pool for every model)
─────────────────────────────────────────────────────────
For each test user with at least one relevant item (rating ≥ 4):
  1. Build candidate pool = relevant items + 99 random unseen negatives.
  2. Score & rank with model.top_k(user_idx, pool, k=10).
  3. Compute NDCG@10 and AP@10.

Cold-start sub-evaluation
─────────────────────────
We additionally compute the same metrics restricted to the **cold-tail**
items (items with <= COLD_THRESHOLD train interactions). This shows where
content-aware models pay off vs pure CF.

Outputs
───────
- Console table with all models side-by-side.
- artifacts/results.json — machine-readable, used by README.md auto-update.

Usage
─────
    python -m recsys.evaluate
    python -m recsys.evaluate --skip-content
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from recsys.data.preprocessor   import load_artifacts
from recsys.evaluation.metrics  import _run_evaluation as run_eval

ARTIFACTS_DIR = Path("artifacts")
RESULTS_PATH  = ARTIFACTS_DIR / "results.json"
K              = 10
COLD_THRESHOLD = 5     # items with ≤ N training interactions are "cold"


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

def _print_table(results: Dict[str, dict]) -> None:
    if not results:
        print("(no results to display)")
        return
    col_w   = 18
    models  = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    header_row = f"{'Metric':<{col_w}}" + "".join(f"{m:>{col_w}}" for m in models)
    sep        = "─" * len(header_row)

    print(f"\n{'Model Comparison':^{len(header_row)}}")
    print(sep)
    print(header_row)
    print(sep)
    for metric in metrics:
        row = f"{metric:<{col_w}}"
        for m in models:
            val = results[m].get(metric, "—")
            row += f"{str(val):>{col_w}}"
        print(row)
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Cold-start evaluation: restrict relevant set to cold items
# ──────────────────────────────────────────────────────────────────────────────

def _cold_test_df(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only test interactions whose item appears ≤ COLD_THRESHOLD times in train."""
    item_train_count = train_df.groupby("item_idx").size()
    cold_items = set(item_train_count[item_train_count <= COLD_THRESHOLD].index.tolist())
    cold_test  = test_df[test_df["item_idx"].isin(cold_items)].copy()
    print(
        f"[cold-start] cold_items={len(cold_items):,}/{train_df['item_idx'].nunique():,}  "
        f"cold_test_rows={len(cold_test):,}/{len(test_df):,}"
    )
    return cold_test


# ──────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _eval_one(
    name:       str,
    model,
    train_df:   pd.DataFrame,
    test_df:    pd.DataFrame,
    cold_df:    pd.DataFrame,
    num_items:  int,
    train_time: float | str,
) -> Dict[str, object]:
    print(f"\n[evaluate] Scoring {name} (K={K}) …")
    t0 = time.perf_counter()
    full = run_eval(model, test_df, train_df, num_items, k=K, label=name)
    infer_time = round(time.perf_counter() - t0, 2)

    cold = run_eval(model, cold_df, train_df, num_items, k=K, label=f"{name} (cold)")

    return {
        f"ndcg@{K}":         full[f"ndcg@{K}"],
        f"map@{K}":          full[f"map@{K}"],
        "users":             full["num_users"],
        f"ndcg@{K}_cold":    cold[f"ndcg@{K}"],
        f"map@{K}_cold":     cold[f"map@{K}"],
        "users_cold":        cold["num_users"],
        "train_time_s":      train_time,
        "infer_time_s":      infer_time,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports so that the script still runs when some artifacts are missing
# ──────────────────────────────────────────────────────────────────────────────

def _try_load(label: str, loader):
    try:
        return loader()
    except FileNotFoundError as e:
        print(f"[evaluate] Skipping {label}: {e}")
        return None
    except Exception as e:
        print(f"[evaluate] Skipping {label}: {type(e).__name__}: {e}")
        return None


def _load_models(skip_content: bool) -> Dict[str, object]:
    from recsys.models.bpr            import BPRModel
    from recsys.models.item_knn       import ItemKNNModel
    from recsys.models.ncf_model      import NCFModel
    from recsys.models.popularity     import PopularityModel
    from recsys.models.svd_model      import SVDModel
    if not skip_content:
        from recsys.models.content_hybrid import ContentHybridModel

    models: Dict[str, object] = {}
    candidates = [
        ("Popularity",    lambda: PopularityModel.load()),
        ("ItemKNN",       lambda: ItemKNNModel.load()),
        ("SVD",           lambda: SVDModel.load()),
        ("NCF",           lambda: NCFModel.load()),
        ("BPR",           lambda: BPRModel.load()),
    ]
    if not skip_content:
        candidates.append(("ContentHybrid", lambda: ContentHybridModel.load()))

    for name, fn in candidates:
        m = _try_load(name, fn)
        if m is not None:
            models[name] = m
    return models


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-content", action="store_true", help="Skip ContentHybrid")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load splits + mappings ────────────────────────────────────────────
    train_df = pd.read_parquet(ARTIFACTS_DIR / "train.parquet")
    test_df  = pd.read_parquet(ARTIFACTS_DIR / "test.parquet")
    _, item2idx = load_artifacts()
    num_items   = len(item2idx)

    cold_df = _cold_test_df(test_df, train_df)

    # ── Training-time metadata ────────────────────────────────────────────
    times_path = ARTIFACTS_DIR / "training_times.json"
    train_times: Dict[str, float] = {}
    if times_path.exists():
        with open(times_path) as f:
            train_times = json.load(f)

    name_to_key = {
        "Popularity":    "popularity",
        "ItemKNN":       "itemknn",
        "SVD":           "svd",
        "NCF":           "ncf",
        "BPR":           "bpr",
        "ContentHybrid": "content_hybrid",
    }

    # ── Evaluate every loaded model ───────────────────────────────────────
    models = _load_models(skip_content=args.skip_content)
    if not models:
        print("[evaluate] No models loaded — run `python -m recsys.train` first.")
        return

    results: Dict[str, dict] = {}
    for name, model in models.items():
        results[name] = _eval_one(
            name       = name,
            model      = model,
            train_df   = train_df,
            test_df    = test_df,
            cold_df    = cold_df,
            num_items  = num_items,
            train_time = train_times.get(name_to_key[name], "n/a"),
        )

    # ── Display + persist ─────────────────────────────────────────────────
    _print_table(results)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[evaluate] Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
