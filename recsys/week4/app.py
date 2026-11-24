"""Week 4 Day 22-24: FastAPI service with Redis cache.
Run: uvicorn recsys.week4.app:app --reload
"""
import json
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import redis
from fastapi import FastAPI, Query
from pydantic import BaseModel

CACHE_TTL = 300
S = {}


@asynccontextmanager
async def lifespan(app):
    S["r"] = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379)
    S["emb"] = np.load("artifacts/item_emb.npy")
    S["ids"] = json.load(open("artifacts/item_ids.json"))
    yield


app = FastAPI(title="H&M Two-Tower Recommender", lifespan=lifespan)


class RecResponse(BaseModel):
    customer_id: str
    source: str
    items: list[str]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "redis": bool(S["r"].ping()), "items": len(S["ids"])}


@app.get("/recommend/{customer_id}", response_model=RecResponse)
def recommend(customer_id: str, k: int = Query(12, ge=1, le=100)):
    t0, r = time.time(), S["r"]
    cache_key = f"rec:{customer_id}:{k}"
    cached = r.get(cache_key)
    if cached:
        return RecResponse(customer_id=customer_id, source="cache", items=json.loads(cached),
                           latency_ms=round((time.time() - t0) * 1000, 2))
    raw = r.get(f"user_emb:{customer_id}")
    if raw is None:                                  # cold-start user
        items = json.loads(r.get("popular") or "[]")[:k]
        source = "popular_fallback"
    else:
        scores = S["emb"] @ np.frombuffer(raw, dtype=np.float32)
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        items, source = [S["ids"][i] for i in top], "two_tower"
        r.setex(cache_key, CACHE_TTL, json.dumps(items))
    return RecResponse(customer_id=customer_id, source=source, items=items,
                       latency_ms=round((time.time() - t0) * 1000, 2))

