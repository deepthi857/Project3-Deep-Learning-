"""Week 4 Day 25: end-to-end test + latency. Needs Redis loaded.
Run: python -m recsys.week4.integration_test   (pip install httpx)
"""
import json
import statistics
import time
from fastapi.testclient import TestClient
from recsys.week4.app import app

known = json.load(open("artifacts/user_ids.json"))[0]
with TestClient(app) as c:
    assert c.get("/health").json()["status"] == "ok"

    r = c.get(f"/recommend/{known}", params={"k": 10}).json()
    assert r["source"] in ("two_tower", "cache") and len(r["items"]) == 10, r

    r = c.get("/recommend/not_a_real_user", params={"k": 5}).json()
    assert r["source"] == "popular_fallback" and len(r["items"]) == 5, r

    lat = []
    for i in range(100):
        t = time.time()
        c.get(f"/recommend/{known}", params={"k": 12 + i % 3})
        lat.append((time.time() - t) * 1000)
    print(f"p50={statistics.median(lat):.1f}ms  p95={sorted(lat)[94]:.1f}ms")
print("INTEGRATION TEST PASSED")

