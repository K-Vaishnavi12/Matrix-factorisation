"""
Ablation study — measure how NDCG@10 / MAP@10 change with key hyperparameters.

Sweeps performed (NCF model, default unless varying)
────────────────────────────────────────────────────
  1. emb_dim    ∈ {16, 32, 64, 128}
  2. neg_ratio  ∈ {1, 2, 4, 8}
  3. dropout    ∈ {0.0, 0.2, 0.4}

Each cell trains a fresh NCF, evaluates on the held-out test set,
and writes the result to `artifacts/ablation_results.json`.

Why bother?
───────────
A single number per model is weak evidence. An ablation shows the *shape*
of model behaviour (how performance scales with capacity, regularisation,
and supervision density) — exactly what an Amazon ML Summer School panel
wants to see in a project. Keep the grid small; the goal is curves, not
SOTA.

Usage
─────
    # Quick sanity sweep (5 epochs each):
    python -m recsys.ablation --epochs 5

    # Customise grid:
    python -m recsys.ablation --emb-dims 16 64 --neg-ratios 4 --dropouts 0.2

Outputs
───────
- Console table per sweep
- artifacts/ablation_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from torch.utils.data import DataLoader

from recsys.data.dataset       import NCFDataset
from recsys.data.preprocessor  import load_artifacts
from recsys.evaluation.metrics import _run_evaluation as run_eval
from recsys.models.ncf_model   import NCFModel

ARTIFACTS_DIR = Path("artifacts")
RESULTS_PATH  = ARTIFACTS_DIR / "ablation_results.json"
K = 10


# ──────────────────────────────────────────────────────────────────────────────
# Loader builder (mirrors data/dataset.make_loaders but parametric in neg_ratio)
# ──────────────────────────────────────────────────────────────────────────────

def _build_loaders(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    num_items:  int,
    neg_ratio:  int,
    batch_size: int,
):
    """Build train/val DataLoaders with a custom negative-sampling ratio."""
    # Monkey-patch the module-level NEG_SAMPLE_RATIO temporarily so we don't
    # need to refactor NCFDataset's signature (keeps the public API stable).
    import recsys.data.dataset as ds_mod
    saved = ds_mod.NEG_SAMPLE_RATIO
    ds_mod.NEG_SAMPLE_RATIO = neg_ratio
    try:
        train_ds = NCFDataset(train_df, num_items, neg_sample=True)
        val_ds   = NCFDataset(val_df,   num_items, neg_sample=False)
    finally:
        ds_mod.NEG_SAMPLE_RATIO = saved

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
    )


# ──────────────────────────────────────────────────────────────────────────────
# A single training+eval cell of the ablation grid
# ──────────────────────────────────────────────────────────────────────────────

def _run_cell(
    label:      str,
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    num_users:  int,
    num_items:  int,
    emb_dim:    int,
    neg_ratio:  int,
    dropout:    float,
    epochs:     int,
    batch_size: int,
) -> Dict[str, float]:
    print(
        f"\n[ablation] === {label}  "
        f"emb_dim={emb_dim}  neg_ratio={neg_ratio}  dropout={dropout} ==="
    )
    train_loader, val_loader = _build_loaders(
        train_df, val_df, num_items, neg_ratio=neg_ratio, batch_size=batch_size,
    )
    model = NCFModel(
        num_users   = num_users,
        num_items   = num_items,
        emb_dim     = emb_dim,
        hidden_dims = [128, 64, 32],
        dropout     = dropout,
        lr          = 1e-3,
    )
    t0 = time.perf_counter()
    model.fit(train_loader, val_loader, epochs=epochs, patience=epochs)  # disable early-stop in ablation
    train_time = round(time.perf_counter() - t0, 2)

    metrics = run_eval(model, test_df, train_df, num_items, k=K, label=label)
    return {
        "emb_dim":      emb_dim,
        "neg_ratio":    neg_ratio,
        "dropout":      dropout,
        f"ndcg@{K}":    metrics[f"ndcg@{K}"],
        f"map@{K}":     metrics[f"map@{K}"],
        "users":        metrics["num_users"],
        "train_time_s": train_time,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing per sweep
# ──────────────────────────────────────────────────────────────────────────────

def _print_sweep(title: str, rows: List[dict], var_key: str) -> None:
    if not rows:
        return
    headers = [var_key, f"ndcg@{K}", f"map@{K}", "train_time_s"]
    widths  = [max(len(h), 12) for h in headers]
    line    = "  ".join(f"{h:>{w}}" for h, w in zip(headers, widths))
    sep     = "─" * len(line)
    print(f"\n[ablation] {title}")
    print(sep)
    print(line)
    print(sep)
    for r in rows:
        cells = [str(r[k]) for k in headers]
        print("  ".join(f"{c:>{w}}" for c, w in zip(cells, widths)))
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dims",   nargs="+", type=int,   default=[16, 32, 64, 128])
    p.add_argument("--neg-ratios", nargs="+", type=int,   default=[1, 2, 4, 8])
    p.add_argument("--dropouts",   nargs="+", type=float, default=[0.0, 0.2, 0.4])
    p.add_argument("--epochs",     type=int, default=5,    help="Epochs per cell (default: 5)")
    p.add_argument("--batch-size", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # ── Load splits ────────────────────────────────────────────────────────
    train_df = pd.read_parquet(ARTIFACTS_DIR / "train.parquet")
    val_df   = pd.read_parquet(ARTIFACTS_DIR / "val.parquet")
    test_df  = pd.read_parquet(ARTIFACTS_DIR / "test.parquet")
    user2idx, item2idx = load_artifacts()
    num_users, num_items = len(user2idx), len(item2idx)

    # Defaults held constant when sweeping a single hyperparameter
    DEFAULT_EMB     = 64
    DEFAULT_NEG     = 4
    DEFAULT_DROPOUT = 0.2

    results: Dict[str, list] = {"emb_dim": [], "neg_ratio": [], "dropout": []}

    # ── Sweep 1: emb_dim ───────────────────────────────────────────────────
    for d in args.emb_dims:
        results["emb_dim"].append(_run_cell(
            label=f"emb_dim={d}",
            train_df=train_df, val_df=val_df, test_df=test_df,
            num_users=num_users, num_items=num_items,
            emb_dim=d, neg_ratio=DEFAULT_NEG, dropout=DEFAULT_DROPOUT,
            epochs=args.epochs, batch_size=args.batch_size,
        ))

    # ── Sweep 2: neg_ratio ─────────────────────────────────────────────────
    for n in args.neg_ratios:
        results["neg_ratio"].append(_run_cell(
            label=f"neg_ratio={n}",
            train_df=train_df, val_df=val_df, test_df=test_df,
            num_users=num_users, num_items=num_items,
            emb_dim=DEFAULT_EMB, neg_ratio=n, dropout=DEFAULT_DROPOUT,
            epochs=args.epochs, batch_size=args.batch_size,
        ))

    # ── Sweep 3: dropout ───────────────────────────────────────────────────
    for d in args.dropouts:
        results["dropout"].append(_run_cell(
            label=f"dropout={d}",
            train_df=train_df, val_df=val_df, test_df=test_df,
            num_users=num_users, num_items=num_items,
            emb_dim=DEFAULT_EMB, neg_ratio=DEFAULT_NEG, dropout=d,
            epochs=args.epochs, batch_size=args.batch_size,
        ))

    # ── Print + persist ────────────────────────────────────────────────────
    _print_sweep("Sweep 1 — embedding dim",     results["emb_dim"],   var_key="emb_dim")
    _print_sweep("Sweep 2 — negative ratio",    results["neg_ratio"], var_key="neg_ratio")
    _print_sweep("Sweep 3 — dropout",           results["dropout"],   var_key="dropout")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ablation] Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
