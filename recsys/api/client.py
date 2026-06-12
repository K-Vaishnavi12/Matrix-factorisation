"""
Python client for the Recommendation API.

Usage Example
─────────────
    from recsys.api.client import RecommendationClient

    client = RecommendationClient("http://localhost:8000")

    # Health check
    status = client.health()
    print(f"Service status: {status['status']}")

    # Get recommendations
    svd_recs = client.recommend_svd("A1EXAMPLE123", k=10)
    ncf_recs = client.recommend_ncf("A1EXAMPLE123", k=10)

    # Compare both models
    comparison = client.compare("A1EXAMPLE123", k=10)
    print(f"Overlap: {comparison['overlap']} items")
"""

from typing import Any, Dict, List, Optional

import requests


class RecommendationAPIError(Exception):
    """Raised when the API returns an error response."""
    pass


class RecommendationClient:
    """
    Python client for the FastAPI recommendation service.

    Parameters
    ----------
    base_url : str
        Base URL of the API server (e.g., "http://localhost:8000")
    timeout : int
        Request timeout in seconds (default: 30)
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.session  = requests.Session()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Internal request handler with error checking."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            # Try to extract detail from JSON error response
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            raise RecommendationAPIError(
                f"API request failed: {error_detail}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RecommendationAPIError(
                f"Network error: {str(e)}"
            ) from e

    # ── Public API methods ────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns
        -------
        {
            "status": "ok",
            "models_loaded": True,
            "num_users": 5234,
            "num_items": 12456
        }
        """
        return self._request("GET", "/health")

    def recommend_svd(self, user_id: str, k: int = 10) -> Dict[str, Any]:
        """
        Get top-K recommendations from the SVD model.

        Parameters
        ----------
        user_id : str
            User ID from the dataset
        k : int
            Number of recommendations (1-100, default: 10)

        Returns
        -------
        {
            "user_id": "A1EXAMPLE123",
            "model": "SVD",
            "k": 10,
            "recommendations": [
                {"item_id": "B00ABC", "score": 4.67},
                ...
            ]
        }
        """
        return self._request("GET", f"/recommend/svd/{user_id}", params={"k": k})

    def recommend_ncf(self, user_id: str, k: int = 10) -> Dict[str, Any]:
        """
        Get top-K recommendations from the NCF model.

        Parameters
        ----------
        user_id : str
            User ID from the dataset
        k : int
            Number of recommendations (1-100, default: 10)

        Returns
        -------
        {
            "user_id": "A1EXAMPLE123",
            "model": "NCF",
            "k": 10,
            "recommendations": [
                {"item_id": "B00ABC", "score": 0.92},
                ...
            ]
        }
        """
        return self._request("GET", f"/recommend/ncf/{user_id}", params={"k": k})

    def compare(self, user_id: str, k: int = 10) -> Dict[str, Any]:
        """
        Get side-by-side recommendations from both SVD and NCF.

        Parameters
        ----------
        user_id : str
            User ID from the dataset
        k : int
            Number of recommendations (1-100, default: 10)

        Returns
        -------
        {
            "user_id": "A1EXAMPLE123",
            "k": 10,
            "svd": [...],
            "ncf": [...],
            "overlap": 6
        }
        """
        return self._request(
            "POST",
            "/recommend/compare",
            json={"user_id": user_id, "k": k},
        )

    # ── Convenience methods ───────────────────────────────────────────────

    def get_recommendations(
        self,
        user_id: str,
        model: str = "ncf",
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations from the specified model (convenience wrapper).

        Parameters
        ----------
        user_id : str
            User ID
        model : str
            "svd" or "ncf" (default: "ncf")
        k : int
            Number of recommendations

        Returns
        -------
        List of {"item_id": str, "score": float} dicts.
        """
        if model.lower() == "svd":
            resp = self.recommend_svd(user_id, k=k)
        elif model.lower() == "ncf":
            resp = self.recommend_ncf(user_id, k=k)
        else:
            raise ValueError(f"Unknown model '{model}'. Use 'svd' or 'ncf'.")
        return resp["recommendations"]

    def get_top_items(
        self,
        user_id: str,
        model: str = "ncf",
        k: int = 10,
    ) -> List[str]:
        """
        Get just the item IDs (no scores) from the specified model.

        Returns
        -------
        List of item_id strings, e.g. ["B00ABC123", "B00XYZ789", ...]
        """
        recs = self.get_recommendations(user_id, model=model, k=k)
        return [rec["item_id"] for rec in recs]


# ────────────────────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    client = RecommendationClient()

    # Quick health check
    try:
        health = client.health()
        print(f"✓ Service is {health['status']}")
        print(f"  Models loaded: {health['models_loaded']}")
        print(f"  Users: {health.get('num_users', 'n/a')}")
        print(f"  Items: {health.get('num_items', 'n/a')}")
    except RecommendationAPIError as e:
        print(f"✗ Service unreachable: {e}")
        sys.exit(1)

    # Demo with a hardcoded user (replace with real user_id from your data)
    print("\nExample: Get top-5 NCF recommendations for a sample user")
    print("(Replace 'A1EXAMPLE123' with a real user_id from artifacts/user2idx.pkl)\n")

    try:
        demo_user = "A1EXAMPLE123"
        recs = client.get_recommendations(demo_user, model="ncf", k=5)
        print(f"Top-5 NCF for {demo_user}:")
        for i, rec in enumerate(recs, start=1):
            print(f"  {i}. {rec['item_id']}  (score: {rec['score']})")
    except RecommendationAPIError as e:
        print(f"Error: {e}")
        print("\nTip: Get a valid user_id with:")
        print("  python -c \"import pickle; u=pickle.load(open('artifacts/user2idx.pkl','rb')); print(sorted(u.keys())[0])\"")
