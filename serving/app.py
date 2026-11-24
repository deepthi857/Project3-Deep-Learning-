"""Week 4: FastAPI real-time recommendations.  Run: uvicorn serving.app:app --reload"""
import json
import time
import numpy as np
import redis
from fastapi import FastAPI

app = FastAPI(title="H&M Two-Tower Recommender")
r = redis.Redis(host="localhost", port=6379)
ITEM_EMB = np.load("artifacts/item_emb.npy")
ITEM_IDS = json.load(open("artifacts/item_ids.json"))


@app.get("/health")
def health():
    return {"status": "ok", "items": len(ITEM_IDS)}


@app.get("/recommend/{customer_id}")
def recommend(customer_id: str, k: int = 12):
    t0 = time.time()
    k = min(k, len(ITEM_IDS) - 1)
    raw = r.get(f"user_emb:{customer_id}")
    if raw is None:  # cold-start user -> popularity fallback
        popular = json.loads(r.get("popular") or "[]")
        return {"customer_id": customer_id, "source": "popular_fallback", "items": popular[:k]}
    u = np.frombuffer(raw, dtype=np.float32)
    scores = ITEM_EMB @ u
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    return {"customer_id": customer_id, "source": "two_tower",
            "items": [ITEM_IDS[i] for i in top],
            "latency_ms": round((time.time() - t0) * 1000, 2)}
