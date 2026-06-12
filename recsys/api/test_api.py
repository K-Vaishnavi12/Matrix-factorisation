"""
Manual API test script — demonstrates all endpoints.

Prerequisites
─────────────
1. Train models:  python -m recsys.train
2. Start server:  uvicorn recsys.api.main:app --port 8000
3. Run tests:     python -m recsys.api.test_api

Or use the interactive docs at http://localhost:8000/docs
"""

import json
import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"


def print_section(title: str) -> None:
    """Pretty-print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def test_health() -> None:
    print_section("GET /health")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


def test_svd(user_id: str, k: int = 5) -> None:
    print_section(f"GET /recommend/svd/{user_id}?k={k}")
    resp = requests.get(f"{BASE_URL}/recommend/svd/{user_id}", params={"k": k})
    print(f"Status: {resp.status_code}")
    if resp.ok:
        data = resp.json()
        print(f"User:  {data['user_id']}")
        print(f"Model: {data['model']}")
        print(f"Top-{data['k']} Recommendations:")
        for i, rec in enumerate(data["recommendations"], start=1):
            print(f"  {i}. {rec['item_id']}  score={rec['score']}")
    else:
        print(resp.text)


def test_ncf(user_id: str, k: int = 5) -> None:
    print_section(f"GET /recommend/ncf/{user_id}?k={k}")
    resp = requests.get(f"{BASE_URL}/recommend/ncf/{user_id}", params={"k": k})
    print(f"Status: {resp.status_code}")
    if resp.ok:
        data = resp.json()
        print(f"User:  {data['user_id']}")
        print(f"Model: {data['model']}")
        print(f"Top-{data['k']} Recommendations:")
        for i, rec in enumerate(data["recommendations"], start=1):
            print(f"  {i}. {rec['item_id']}  score={rec['score']}")
    else:
        print(resp.text)


def test_compare(user_id: str, k: int = 5) -> None:
    print_section(f"POST /recommend/compare")
    payload = {"user_id": user_id, "k": k}
    resp = requests.post(
        f"{BASE_URL}/recommend/compare",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    print(f"Status: {resp.status_code}")
    if resp.ok:
        data = resp.json()
        print(f"User:    {data['user_id']}")
        print(f"K:       {data['k']}")
        print(f"Overlap: {data['overlap']} items appear in both lists\n")

        print("SVD Recommendations:")
        for i, rec in enumerate(data["svd"], start=1):
            print(f"  {i}. {rec['item_id']}  score={rec['score']}")

        print("\nNCF Recommendations:")
        for i, rec in enumerate(data["ncf"], start=1):
            print(f"  {i}. {rec['item_id']}  score={rec['score']}")
    else:
        print(resp.text)


def test_404() -> None:
    print_section("GET /recommend/svd/NONEXISTENT_USER (expect 404)")
    resp = requests.get(f"{BASE_URL}/recommend/svd/NONEXISTENT_USER")
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


def get_sample_user() -> str:
    """Extract a valid user_id from the training data."""
    artifacts = Path("artifacts")
    if not artifacts.exists():
        print("Error: ./artifacts/ not found. Run `python -m recsys.train` first.")
        sys.exit(1)

    import pickle
    with open(artifacts / "user2idx.pkl", "rb") as f:
        user2idx = pickle.load(f)

    # Return the first user (alphabetically)
    return sorted(user2idx.keys())[0]


def main() -> None:
    print(f"\nAPI Base URL: {BASE_URL}")
    print("Ensure the server is running:  uvicorn recsys.api.main:app --port 8000\n")

    # Get a real user_id from the training data
    sample_user = get_sample_user()
    print(f"Using sample user: {sample_user}\n")

    try:
        test_health()
        test_svd(sample_user, k=5)
        test_ncf(sample_user, k=5)
        test_compare(sample_user, k=5)
        test_404()

        print("\n" + "="*70)
        print("  All tests complete. Check the interactive docs at:")
        print(f"  {BASE_URL}/docs")
        print("="*70 + "\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ Connection failed. Is the server running?")
        print("   Start it with:  uvicorn recsys.api.main:app --port 8000\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
