"""
End-to-end orchestrator: train → evaluate → (optional) ablation.

This is a one-stop shop that:
  1. Downloads the Amazon Reviews dataset (or uses cached parquet splits)
  2. Trains all six models (Popularity, ItemKNN, SVD, NCF, BPR, ContentHybrid)
  3. Evaluates them on the held-out test set with NDCG@10 / MAP@10
  4. Optionally runs the hyperparameter ablation sweep

Examples
────────
  # Quickest reproducible end-to-end (All_Beauty, ~5 min on CPU):
  python run_all.py

  # Larger-scale with multiple categories (~30 min):
  python run_all.py --categories All_Beauty Office_Products Toys_and_Games

  # Skip heavy steps:
  python run_all.py --skip-content --skip-svd --skip-ablation

  # Add the ablation grid (5 epochs per cell):
  python run_all.py --run-ablation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], stage: str) -> None:
    print(f"\n┌────────────────────────────────────────────────────────────────")
    print(f"│ Stage: {stage}")
    print(f"│ Command: {' '.join(cmd)}")
    print(f"└────────────────────────────────────────────────────────────────")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    dt   = round(time.perf_counter() - t0, 2)
    if proc.returncode != 0:
        print(f"\n✗ Stage '{stage}' failed (exit {proc.returncode}) after {dt}s")
        sys.exit(proc.returncode)
    print(f"\n✓ Stage '{stage}' completed in {dt}s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + evaluate full recsys pipeline")
    # Pass-through to recsys.train
    p.add_argument("--category",   type=str)
    p.add_argument("--categories", nargs="+")
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--skip-content", action="store_true")
    p.add_argument("--skip-svd",     action="store_true")
    # Ablation control
    p.add_argument("--run-ablation",    action="store_true",
                   help="Run hyperparameter sweep after evaluation (slow)")
    p.add_argument("--ablation-epochs", type=int, default=5)
    p.add_argument("--skip-train",      action="store_true",
                   help="Skip training (use existing artifacts)")
    p.add_argument("--skip-eval",       action="store_true",
                   help="Skip evaluation")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    py   = sys.executable
    overall_t0 = time.perf_counter()

    # ── Stage 1: Train ─────────────────────────────────────────────────────
    if not args.skip_train:
        train_cmd = [py, "-m", "recsys.train",
                     "--epochs", str(args.epochs),
                     "--batch-size", str(args.batch_size)]
        if args.category:
            train_cmd += ["--category", args.category]
        if args.categories:
            train_cmd += ["--categories", *args.categories]
        if args.skip_content:
            train_cmd += ["--skip-content"]
        if args.skip_svd:
            train_cmd += ["--skip-svd"]
        _run(train_cmd, "train")

    # ── Stage 2: Evaluate ──────────────────────────────────────────────────
    if not args.skip_eval:
        eval_cmd = [py, "-m", "recsys.evaluate"]
        if args.skip_content:
            eval_cmd += ["--skip-content"]
        _run(eval_cmd, "evaluate")

    # ── Stage 3: Ablation ──────────────────────────────────────────────────
    if args.run_ablation:
        abl_cmd = [py, "-m", "recsys.ablation",
                   "--epochs", str(args.ablation_epochs)]
        _run(abl_cmd, "ablation")

    overall = round(time.perf_counter() - overall_t0, 2)
    print(f"\n=== Pipeline complete in {overall}s ===")
    print("    artifacts/results.json         — main eval table")
    if args.run_ablation:
        print("    artifacts/ablation_results.json — sweep results")


if __name__ == "__main__":
    main()
