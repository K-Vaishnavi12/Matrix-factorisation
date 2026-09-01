# B2B Reviews Recommender

An end-to-end product recommendation system that benchmarks **six** different
models — from a simple popularity baseline to a content-aware deep model —
on the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
dataset, and serves the trained models behind a REST API.

---

## About the project

Given a user's past product ratings, the system predicts which products
they're likely to engage with next (a top-K recommendation task). Six
models are trained on the same data and evaluated under an identical
protocol so the comparison is fair:

| # | Model | Type | What it brings |
|---|---|---|---|
| 1 | **Popularity** | Non-personalised | Sanity-check baseline |
| 2 | **Item-kNN** | Memory-based CF | Classical cosine-similarity baseline |
| 3 | **SVD** | Matrix factorisation (point-wise) | Rating prediction baseline |
| 4 | **NCF** | Neural CF (point-wise BCE) | Non-linear user-item interactions |
| 5 | **BPR** | Matrix factorisation (pairwise) | Directly optimises ranking |
| 6 | **ContentHybrid** | Two-tower with item content features | Cold-start handling |

Evaluation uses the standard **1-vs-99 protocol** with **NDCG@10** and
**MAP@10** computed from scratch, plus a separate **cold-start subset**
restricted to items with ≤ 5 training interactions.

After training, the models are exposed through a **FastAPI** service
with auto-generated Swagger docs.

---

## Tech stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data handling** | `pandas`, `numpy`, `pyarrow` (Parquet I/O), `scipy.sparse` (Item-kNN) |
| **Dataset source** | HuggingFace `datasets` — `McAuley-Lab/Amazon-Reviews-2023` |
| **Classical CF** | `scikit-surprise` (SVD with grid search) |
| **Deep models** | PyTorch 2.x — NCF, BPR, ContentHybrid |
| **REST API** | FastAPI + Uvicorn |
| **Validation** | Pydantic v2 |
| **Hardware** | Runs on CPU; auto-detects CUDA if available |

Pinned versions are in [`requirements.txt`](requirements.txt).

---

## Project layout

```
.
├── run_all.py                       # Top-level orchestrator (train + eval + ablation)
├── requirements.txt
├── README.md
│
├── recsys/
│   ├── train.py                     # Trains all 6 models
│   ├── evaluate.py                  # Evaluates all 6 on the test set
│   ├── ablation.py                  # Hyperparameter sweep
│   │
│   ├── data/
│   │   ├── loader.py                # HuggingFace download (multi-category)
│   │   ├── preprocessor.py          # ID mapping + user-wise split
│   │   └── dataset.py               # PyTorch Dataset + negative sampling
│   │
│   ├── models/
│   │   ├── popularity.py
│   │   ├── item_knn.py
│   │   ├── svd_model.py
│   │   ├── ncf.py / ncf_model.py
│   │   ├── bpr.py
│   │   └── content_hybrid.py
│   │
│   ├── evaluation/
│   │   └── metrics.py               # NDCG@K, MAP@K, 1-vs-99 evaluator
│   │
│   └── api/
│       ├── main.py                  # FastAPI service
│       └── API_REFERENCE.md         # Endpoint reference
│
└── artifacts/                       # Generated at runtime (ignored in VCS)
```

---

## How to execute the code

You need **Python 3.10+** and roughly **2 GB free disk** for the dataset cache.

### Step 1 — Set up the environment

```powershell
# From the project root
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2 — Run the full pipeline (one command)

```bash
python run_all.py
```

This downloads the default category (`All_Beauty`), trains all six models,
evaluates them, and writes results to `artifacts/`. Roughly **5 min on CPU**.

Useful flags:

```bash
# Larger experiment with multiple categories (~30 min)
python run_all.py --categories All_Beauty Office_Products Toys_and_Games

# Skip the slow grid-search SVD or the metadata-heavy ContentHybrid
python run_all.py --skip-svd --skip-content

# Re-evaluate without retraining (artifacts already present)
python run_all.py --skip-train

# Train + evaluate + run hyperparameter ablation
python run_all.py --run-ablation
```

### Step 3 — Inspect the results

Once `run_all.py` finishes, look at:

| File | What's in it |
|---|---|
| `artifacts/results.json`           | Per-model NDCG@10, MAP@10, cold-start variants, timings |
| `artifacts/dataset_stats.json`     | Interactions, users, items, sparsity |
| `artifacts/training_times.json`    | Wall-clock training time per model |
| `artifacts/ablation_results.json`  | Sweep curves (only if `--run-ablation` was passed) |

The console also prints a comparison table at the end of the evaluation step.

### Step 4 (optional) — Run pieces individually

```bash
# Train all models
python -m recsys.train

# Evaluate already-trained models
python -m recsys.evaluate

# Hyperparameter ablation (sweep emb_dim, neg_ratio, dropout)
python -m recsys.ablation --epochs 5
```

### Step 5 (optional) — Use a model from Python

Every model exposes the same five-method API:

```python
from recsys.models.bpr import BPRModel

bpr = BPRModel.load()  # reads artifacts/bpr_model.pt

# Single prediction
score = bpr.predict_score(user_idx=42, item_idx=137)

# Top-10 from a candidate pool
top10 = bpr.top_k(user_idx=42, candidate_items=list(range(1000)), k=10)
# → [(item_idx, score), ...] sorted descending
```

To translate raw Amazon IDs to indices:

```python
import pickle
with open("artifacts/user2idx.pkl", "rb") as f:
    user2idx = pickle.load(f)
uid = user2idx["A1EXAMPLE123"]
```

### Step 6 (optional) — Serve the API

After training, expose SVD and NCF as a REST service:

```bash
uvicorn recsys.api.main:app --host 0.0.0.0 --port 8000

# Production: multiple workers
uvicorn recsys.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

Endpoints:

| Method | Path                              | Purpose |
|--------|-----------------------------------|---------|
| GET    | `/health`                         | Liveness + model load state |
| GET    | `/recommend/svd/{user_id}?k=10`   | SVD top-K |
| GET    | `/recommend/ncf/{user_id}?k=10`   | NCF top-K |
| POST   | `/recommend/compare`              | Side-by-side SVD vs NCF |

Quick test:

```bash
# Health check
curl http://localhost:8000/health

# Get a valid user_id from the artifacts
python -c "import pickle; print(sorted(pickle.load(open('artifacts/user2idx.pkl','rb')).keys())[0])"

# Use it
curl "http://localhost:8000/recommend/ncf/<USER_ID>?k=5"
```

Full API contract: [`recsys/api/API_REFERENCE.md`](recsys/api/API_REFERENCE.md).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'surprise'`        | Missing dependency        | `pip install scikit-surprise`, or run with `--skip-svd` |
| `ModuleNotFoundError: No module named 'recsys'`          | Wrong working directory   | Run from repo root, and use `python -m recsys.train` (not `python recsys/train.py`) |
| `FileNotFoundError: artifacts/train.parquet`             | Evaluating before training | Run `python -m recsys.train` first, then `python -m recsys.evaluate` |
| HuggingFace download stalls or fails                     | Network / rate limit      | Re-run; set `HF_HUB_OFFLINE=1` after first success to use the cache |
| CUDA out-of-memory                                       | Large category, small GPU | Use `--batch-size 256` or train on CPU (default) |

---

## License

MIT
