"""Week 3 Day 16-17: integration check."""
import json
import redis

r = redis.Redis(host="localhost", port=6379)
assert r.ping(), "Redis not reachable"
n_users = sum(1 for _ in r.scan_iter("user_emb:*", count=5000))
print("user embeddings in Redis:", n_users)
ids = json.load(open("artifacts/user_ids.json"))
sample = r.get(f"user_emb:{ids[0]}")
print("bytes per embedding:", len(sample), "(expected dim*4)")
popular = json.loads(r.get("popular") or "[]")
assert len(popular) > 0, "popular fallback missing"
print("popular fallback size:", len(popular))
print("REDIS CHECK OK")


