# Redis schema

| Key | Value | Notes |
|---|---|---|
| user_emb:{customer_id} | 64 float32 values as raw bytes (256 bytes) | written by serving/load_redis.py |
| popular | JSON list of the 100 most popular article ids | fallback for unknown users |
| rec:{customer_id}:{k} | JSON list of recommended article ids | cache, expires after 300 seconds |

## Redis Load Test

- Redis version: 7.0.15
- User embeddings loaded: 489,553
- Embedding dimension: 64
- Embedding size: 256 bytes per user
- Popular fallback entries: 100
- Integration check: PASSED

### Benchmark results

**Redis GET benchmark**
- Requests: 10,000
- Concurrent clients: 50
- Throughput: 107,526.88 requests/second
- Average latency: 0.241 ms
- P99 latency: 0.599 ms

**Actual embedding MGET test**
- User embedding keys tested: 10,000
- Successful reads: 10,000
- Total time: 0.6538 seconds
- Batch throughput: approximately 15,295 embeddings/second
- Embedding size: 256 bytes

Note: The Redis GET benchmark uses a small dummy payload. The MGET test reads actual 256-byte user embeddings in batches.
