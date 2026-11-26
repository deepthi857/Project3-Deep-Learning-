# Final Project Report (Item Pipeline, Evaluation & QA)

## 1. Data & Cold-Start Behavior
- Catalog items include both active/warm items and long-tail cold-start items with fewer than 5 purchases in the lookback window.
- Cold-start and out-of-vocabulary items route deterministically to the popular-items fallback so recommendations remain non-empty and fast.

## 2. Model vs. Popularity Baseline
- Evaluated across Top-K cutoffs (Top-1, Top-5, Top-10, Top-50, Top-100) and broken down by `product_type_name`.
- Combining learned item embeddings for covered items with popular-item fallback protects coverage on tail or cold requests.

## 3. Serving & Latency Verification
- Verified HTTP status codes, response schema, top-K bounds, and cold-user fallback via `tests/test_recommend_endpoint.py`.
- Measured end-to-end request latency (mean, p50, p95, p99) and throughput under concurrent load via `tests/latency_test.py`.
