"""Week 3: push user embeddings + popular fallback into Redis (feature store/cache)."""
import json
import numpy as np
import pandas as pd
import redis

r = redis.Redis(host="localhost", port=6379)
r.ping()
ids = json.load(open("artifacts/user_ids.json"))
emb = np.load("artifacts/user_emb.npy")

BATCH = 5000
pipe = r.pipeline(transaction=False)
for n, (cid, vec) in enumerate(zip(ids, emb), 1):
    pipe.set(f"user_emb:{cid}", vec.astype(np.float32).tobytes())
    if n % BATCH == 0:
        pipe.execute()
        print(f"{n}/{len(ids)} users loaded", flush=True)
pipe.execute()

popular = pd.read_parquet("data/features/popular").article_id.astype(str).tolist()
r.set("popular", json.dumps(popular))
print("loaded", len(ids), "users")
