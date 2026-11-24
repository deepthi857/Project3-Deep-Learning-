# Redis schema

| Key | Value | Notes |
|---|---|---|
| user_emb:{customer_id} | 64 float32 values as raw bytes (256 bytes) | written by serving/load_redis.py |
| popular | JSON list of the 100 most popular article ids | fallback for unknown users |
| rec:{customer_id}:{k} | JSON list of recommended article ids | cache, expires after 300 seconds |
