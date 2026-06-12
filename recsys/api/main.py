"""
FastAPI Product Recommendation Service

Endpoints
─────────
  GET  /health
       Returns service health status and model load state.

  GET  /recommend/svd/{user_id}?k=10
       Top-K recommendations from the SVD model.

  GET  /recommend/ncf/{user_id}?k=10
       Top-K recommendations from the NCF model.

  POST /recommend/compare
       Side-by-side comparison of SVD vs NCF recommendations.
       Body: {"user_id": "A123...", "k": 10}

Usage
─────
  Local:
    uvicorn recsys.api.main:app --reload --port 8000

  Production (EC2):
    uvicorn recsys.api.main:app --host 0.0.0.0 --port 8000 --workers 2

Examples
────────
  curl http://localhost:8000/health

  curl http://localhost:8000/recommend/svd/A1EXAMPLE123?k=5

  curl -X POST http://localhost:8000/recommend/compare \\
    -H "Content-Type: application/json" \\
    -d '{"user_id": "A1EXAMPLE123", "k": 5}'
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from recsys.data.preprocessor import load_artifacts
from recsys.models.ncf_model  import NCFModel
from recsys.models.svd_model  import SVDModel

ARTIFACTS_DIR = Path("artifacts")

# Global state loaded at startup (shared across all requests)
_state: Dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan: Load models once at startup
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load models, mappings, and training item sets into memory.
    Shutdown: Clear state.
    """
    print("[api] Loading artifacts from disk …")

    # ID mappings
    user2idx, item2idx = load_artifacts()
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}

    # Per-user training item sets (for candidate filtering)
    train_df = pd.read_parquet(ARTIFACTS_DIR / "train.parquet")
    train_seen: Dict[int, set] = (
        train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )

    # Load trained models
    svd = SVDModel.load()
    ncf = NCFModel.load()

    _state.update(
        user2idx=user2idx,
        item2idx=item2idx,
        idx2user=idx2user,
        idx2item=idx2item,
        train_seen=train_seen,
        svd=svd,
        ncf=ncf,
        num_users=len(user2idx),
        num_items=len(item2idx),
    )
    print(f"[api] Loaded {len(user2idx):,} users, {len(item2idx):,} items")
    print("[api] Service ready.\n")

    yield  # Application runs here

    _state.clear()
    print("[api] Shutdown complete.")


app = FastAPI(
    title="Product Recommendation API",
    description="SVD vs NCF recommendation models trained on Amazon Reviews 2023",
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    """Single recommended item with score."""
    item_id: str = Field(..., description="Product ID (original from dataset)")
    score:   float = Field(..., description="Predicted relevance score")


class RecommendationResponse(BaseModel):
    """Standard recommendation response."""
    user_id:         str = Field(..., description="User ID (original from dataset)")
    model:           str = Field(..., description="Model name: SVD or NCF")
    k:               int = Field(..., description="Number of recommendations")
    recommendations: List[RecommendationItem] = Field(..., description="Ranked list")


class CompareRequest(BaseModel):
    """Request body for /recommend/compare."""
    user_id: str = Field(..., description="User ID", example="A1EXAMPLE123")
    k:       int = Field(default=10, ge=1, le=100, description="Number of recommendations")


class CompareResponse(BaseModel):
    """Side-by-side comparison of SVD and NCF recommendations."""
    user_id: str
    k:       int
    svd:     List[RecommendationItem] = Field(..., description="SVD recommendations")
    ncf:     List[RecommendationItem] = Field(..., description="NCF recommendations")
    overlap: int = Field(..., description="Number of items appearing in both lists")


class HealthResponse(BaseModel):
    """Health check response."""
    status:        str = Field(..., description="Service status: ok | error")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    num_users:     Optional[int] = Field(None, description="Total users in index")
    num_items:     Optional[int] = Field(None, description="Total items in index")


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _get_user_idx(user_id: str) -> int:
    """
    Map raw user_id string to internal integer index.
    Raises 404 if user is not in the training vocabulary.
    """
    user2idx = _state["user2idx"]
    if user_id not in user2idx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"User '{user_id}' not found in the training data. "
                f"The model was trained on {len(user2idx):,} users."
            ),
        )
    return user2idx[user_id]


def _get_candidates(user_idx: int) -> List[int]:
    """
    Return all item indices NOT seen by this user during training.
    This ensures we don't recommend items the user has already interacted with.
    """
    seen = _state["train_seen"].get(user_idx, set())
    return [i for i in range(_state["num_items"]) if i not in seen]


def _format_recommendations(ranked: List[tuple[int, float]]) -> List[RecommendationItem]:
    """Convert [(item_idx, score), …] to RecommendationItem list."""
    idx2item = _state["idx2item"]
    return [
        RecommendationItem(
            item_id=idx2item[item_idx],
            score=round(score, 4),
        )
        for item_idx, score in ranked
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
def health() -> HealthResponse:
    """
    Returns service health status.

    Use this endpoint to verify the service is running and models are loaded.
    """
    loaded = bool(_state)
    return HealthResponse(
        status="ok" if loaded else "error",
        models_loaded=loaded,
        num_users=_state.get("num_users"),
        num_items=_state.get("num_items"),
    )


@app.get(
    "/recommend/svd/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get SVD recommendations",
)
def recommend_svd(
    user_id: str = Field(..., description="User ID from the dataset"),
    k: int = Field(default=10, ge=1, le=100, description="Number of recommendations"),
) -> RecommendationResponse:
    """
    Get top-K product recommendations from the **SVD (Matrix Factorization)** model.

    The SVD model uses latent factors to predict user-item affinity scores.
    Candidate items are filtered to exclude those the user has already rated.

    **Example:**
    ```
    GET /recommend/svd/A1EXAMPLE123?k=5
    ```
    """
    user_idx   = _get_user_idx(user_id)
    candidates = _get_candidates(user_idx)

    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has interacted with all items. No recommendations available.",
        )

    ranked = _state["svd"].top_k(user_idx, candidates, k=k)

    return RecommendationResponse(
        user_id=user_id,
        model="SVD",
        k=k,
        recommendations=_format_recommendations(ranked),
    )


@app.get(
    "/recommend/ncf/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get NCF recommendations",
)
def recommend_ncf(
    user_id: str = Field(..., description="User ID from the dataset"),
    k: int = Field(default=10, ge=1, le=100, description="Number of recommendations"),
) -> RecommendationResponse:
    """
    Get top-K product recommendations from the **NCF (Neural Collaborative Filtering)** model.

    NCF uses a multi-layer perceptron to learn non-linear user-item interactions.
    Generally achieves higher NDCG@10 and MAP@10 than SVD, but with longer inference time.

    **Example:**
    ```
    GET /recommend/ncf/A1EXAMPLE123?k=5
    ```
    """
    user_idx   = _get_user_idx(user_id)
    candidates = _get_candidates(user_idx)

    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has interacted with all items. No recommendations available.",
        )

    ranked = _state["ncf"].top_k(user_idx, candidates, k=k)

    return RecommendationResponse(
        user_id=user_id,
        model="NCF",
        k=k,
        recommendations=_format_recommendations(ranked),
    )


@app.post(
    "/recommend/compare",
    response_model=CompareResponse,
    tags=["Recommendations"],
    summary="Compare SVD vs NCF side-by-side",
)
def recommend_compare(request: CompareRequest) -> CompareResponse:
    """
    Get recommendations from **both SVD and NCF** for the same user and compare them.

    Returns side-by-side lists with overlap count (how many items appear in both top-K).
    Useful for A/B testing or understanding model disagreement.

    **Example request:**
    ```json
    {
      "user_id": "A1EXAMPLE123",
      "k": 10
    }
    ```

    **Example response:**
    ```json
    {
      "user_id": "A1EXAMPLE123",
      "k": 10,
      "svd": [
        {"item_id": "B00XYZ", "score": 4.23},
        ...
      ],
      "ncf": [
        {"item_id": "B00ABC", "score": 0.89},
        ...
      ],
      "overlap": 6
    }
    ```
    """
    user_id = request.user_id
    k       = request.k

    # Validate user and get candidates
    user_idx   = _get_user_idx(user_id)
    candidates = _get_candidates(user_idx)

    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has interacted with all items. No recommendations available.",
        )

    # Score with both models
    svd_ranked = _state["svd"].top_k(user_idx, candidates, k=k)
    ncf_ranked = _state["ncf"].top_k(user_idx, candidates, k=k)

    svd_recs = _format_recommendations(svd_ranked)
    ncf_recs = _format_recommendations(ncf_ranked)

    # Compute overlap
    svd_items = {rec.item_id for rec in svd_recs}
    ncf_items = {rec.item_id for rec in ncf_recs}
    overlap   = len(svd_items & ncf_items)

    return CompareResponse(
        user_id=user_id,
        k=k,
        svd=svd_recs,
        ncf=ncf_recs,
        overlap=overlap,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Root redirect (optional)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirect root to interactive API docs."""
    return {
        "message": "Product Recommendation API",
        "docs": "/docs",
        "health": "/health",
    }
