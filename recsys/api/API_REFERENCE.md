# FastAPI Recommendation Service — Quick Reference

## Base URL

```
http://localhost:8000          (local development)
http://your-ec2-ip:8000        (production)
```

## Interactive Documentation

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## Endpoints

### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true,
  "num_users": 5234,
  "num_items": 12456
}
```

---

### 2. SVD Recommendations

```http
GET /recommend/svd/{user_id}?k=10
```

**Path Parameters:**
- `user_id` (string, required) — User ID from the Amazon Reviews dataset

**Query Parameters:**
- `k` (integer, optional) — Number of recommendations (1-100, default: 10)

**Example:**
```bash
curl "http://localhost:8000/recommend/svd/A1EXAMPLE123?k=5"
```

**Response:**
```json
{
  "user_id": "A1EXAMPLE123",
  "model": "SVD",
  "k": 5,
  "recommendations": [
    {"item_id": "B00ABC123", "score": 4.67},
    {"item_id": "B00XYZ789", "score": 4.55}
  ]
}
```

**Score interpretation:** Predicted rating on scale 1-5.

---

### 3. NCF Recommendations

```http
GET /recommend/ncf/{user_id}?k=10
```

**Path Parameters:**
- `user_id` (string, required) — User ID from the Amazon Reviews dataset

**Query Parameters:**
- `k` (integer, optional) — Number of recommendations (1-100, default: 10)

**Example:**
```bash
curl "http://localhost:8000/recommend/ncf/A1EXAMPLE123?k=5"
```

**Response:**
```json
{
  "user_id": "A1EXAMPLE123",
  "model": "NCF",
  "k": 5,
  "recommendations": [
    {"item_id": "B00ABC123", "score": 0.92},
    {"item_id": "B00XYZ789", "score": 0.89}
  ]
}
```

**Score interpretation:** Probability of relevance (0-1, higher is better).

---

### 4. Side-by-Side Comparison

```http
POST /recommend/compare
Content-Type: application/json
```

**Request Body:**
```json
{
  "user_id": "A1EXAMPLE123",
  "k": 10
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/recommend/compare" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "A1EXAMPLE123", "k": 5}'
```

**Response:**
```json
{
  "user_id": "A1EXAMPLE123",
  "k": 5,
  "svd": [
    {"item_id": "B00ABC123", "score": 4.67},
    {"item_id": "B00DEF456", "score": 4.55}
  ],
  "ncf": [
    {"item_id": "B00ABC123", "score": 0.92},
    {"item_id": "B00XYZ999", "score": 0.89}
  ],
  "overlap": 1
}
```

**Overlap:** Number of items appearing in both top-K lists.

---

## Error Responses

### 404 User Not Found

```json
{
  "detail": "User 'BADUSER' not found in the training data. The model was trained on 5,234 users."
}
```

### 400 No Candidates

```json
{
  "detail": "User has interacted with all items. No recommendations available."
}
```

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["query", "k"],
      "msg": "ensure this value is less than or equal to 100",
      "type": "value_error.number.not_le"
    }
  ]
}
```

---

## Model Details

| Model | Output Range | Interpretation | Training Time | Inference |
|-------|--------------|----------------|---------------|-----------|
| SVD   | 1.0 - 5.0    | Predicted rating | ~12s | Fast |
| NCF   | 0.0 - 1.0    | Relevance probability | ~45s | Moderate |

**Recommendation:** Use NCF for highest ranking quality (NDCG@10 ~0.49), use SVD for faster inference and simpler deployment.

---

## Design Notes

1. **Candidate Filtering**  
   Both endpoints automatically exclude items the user has already rated during training.

2. **User/Item Mappings**  
   Raw string IDs from the dataset are mapped to internal integer indices using `user2idx.pkl` and `item2idx.pkl` loaded at startup.

3. **Model Loading**  
   Models are loaded into memory once at application startup via the lifespan context manager. All requests share the same model instances (no per-request overhead).

4. **Score Differences**  
   SVD outputs predicted ratings (1-5 scale), NCF outputs probabilities (0-1 scale). Higher is better in both cases, but scores are NOT directly comparable across models.

5. **Concurrency**  
   FastAPI handles requests asynchronously. For production, run with multiple workers:
   ```bash
   uvicorn recsys.api.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

---

## Testing

### Automated Test Suite

```bash
# Start server in one terminal
uvicorn recsys.api.main:app --port 8000

# Run tests in another
python -m recsys.api.test_api
```

### Manual Testing

```bash
# Get a valid user ID
python -c "import pickle; u=pickle.load(open('artifacts/user2idx.pkl','rb')); print(sorted(u.keys())[0])"

# Use it in requests
USER_ID="<output_from_above>"
curl "http://localhost:8000/recommend/svd/$USER_ID?k=10"
```

---

## Production Deployment

See `README.md` § Deploy to AWS EC2 for full systemd setup.

Quick EC2 one-liner:
```bash
nohup uvicorn recsys.api.main:app --host 0.0.0.0 --port 8000 --workers 2 &
```

Remember to open port 8000 in your security group inbound rules.
