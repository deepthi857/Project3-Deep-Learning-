"""Week 4: endpoint tests, latency benchmark, and final docs (Days 22-28).
Run generator:    python week4_serving_tests_docs.py
Run endpoint test: pytest tests/test_recommend_endpoint.py
Run latency test:  python tests/latency_test.py --url http://localhost:8000/recommend --n 100
"""
from pathlib import Path

TEST_DIR = Path("tests")
DOC_DIR = Path("docs/week4")
TEST_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, content: str):
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print("saved", path)


def main():
    endpoint_test = '''import os
import pytest
import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def test_recommend_valid_payload():
    payload = {"user_id": "00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdf6654", "k": 10}
    resp = requests.post(f"{BASE_URL}/recommend", json=payload, timeout=5)
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} {resp.text}"
    data = resp.json()
    assert isinstance(data, dict)
    recs = data.get("recommendations") or data.get("items") or []
    assert isinstance(recs, list)
    assert 0 < len(recs) <= 10


def test_recommend_cold_user_fallback():
    payload = {"user_id": "cold_or_unknown_user_000", "k": 5}
    resp = requests.post(f"{BASE_URL}/recommend", json=payload, timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    recs = data.get("recommendations") or data.get("items") or []
    assert isinstance(recs, list)
    assert len(recs) == 5


@pytest.mark.parametrize("bad_k", [0, -1])
def test_recommend_invalid_k(bad_k):
    payload = {"user_id": "test_user", "k": bad_k}
    resp = requests.post(f"{BASE_URL}/recommend", json=payload, timeout=5)
    assert resp.status_code in (400, 422)
'''
    write_file(TEST_DIR / "test_recommend_endpoint.py", endpoint_test)

    latency_test = '''import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


def call_endpoint(url: str, payload: dict, timeout: float = 10.0) -> tuple[float, bool]:
    start = time.perf_counter()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        ok = (r.status_code == 200)
    except Exception:
        ok = False
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, ok


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[idx], 2)


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for /recommend")
    parser.add_argument("--url", default="http://localhost:8000/recommend")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    payload = {"user_id": "bench_user_1", "k": args.k}
    latencies = []
    successes = 0

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(call_endpoint, args.url, payload) for _ in range(args.n)]
        for fut in as_completed(futures):
            ms, ok = fut.result()
            latencies.append(ms)
            if ok:
                successes += 1
    total_s = time.perf_counter() - wall_start

    print(f"Requests: {args.n}, Successes: {successes}/{args.n}, Duration: {total_s:.2f}s")
    print(f"Throughput: {args.n / max(total_s, 1e-6):.2f} req/s")
    print(f"Latency (ms) -> mean: {statistics.mean(latencies):.2f}, p50: {percentile(latencies, 50)}, "
          f"p95: {percentile(latencies, 95)}, p99: {percentile(latencies, 99)}")


if __name__ == "__main__":
    main()
'''
    write_file(TEST_DIR / "latency_test.py", latency_test)

    readme_md = '''# Two-Tower Recommendation System - Item, Evaluation & Serving Summary

## Overview
End-to-end item retrieval pipeline built on retail transaction and catalog data:
- **Item Data & Features (Week 1):** Catalog profiling, cleaning verification, cold-start item flags (<5 purchases), item recency, and weekly log-popularity (`item_pop_log`).
- **Item Tower & Baseline (Week 2):** Item feature spec audit, trainable parameter count, candidate set coverage, embedding sanity checks, and 4-week popularity baseline comparison.
- **Routing & Category Evaluation (Week 3):** Cold/unknown item fallback to popular items, routing unit tests, per-`product_type_name` evaluation, and consolidated results table.
- **Serving Tests & Final Review (Week 4):** Functional `/recommend` tests, p50/p95/p99 latency test script, and final project documentation.

## Key Commands
- Run Week 1 item checks: `python week1_item_checks.py`
- Run Week 2 item checks: `python week2_item_checks.py`
- Run Week 3 routing & evaluation: `python week3_item_eval_routing.py`
- Run routing unit tests: `python -m unittest discover -s tests -p "test_item_routing.py"`
- Run API tests: `pytest tests/test_recommend_endpoint.py`
- Run latency test: `python tests/latency_test.py --url http://localhost:8000/recommend --n 100`
'''
    write_file(DOC_DIR / "README_SUMMARY.md", readme_md)

    report_md = '''# Final Project Report (Item Pipeline, Evaluation & QA)

## 1. Data & Cold-Start Behavior
- Catalog items include both active/warm items and long-tail cold-start items with fewer than 5 purchases in the lookback window.
- Cold-start and out-of-vocabulary items route deterministically to the popular-items fallback so recommendations remain non-empty and fast.

## 2. Model vs. Popularity Baseline
- Evaluated across Top-K cutoffs (Top-1, Top-5, Top-10, Top-50, Top-100) and broken down by `product_type_name`.
- Combining learned item embeddings for covered items with popular-item fallback protects coverage on tail or cold requests.

## 3. Serving & Latency Verification
- Verified HTTP status codes, response schema, top-K bounds, and cold-user fallback via `tests/test_recommend_endpoint.py`.
- Measured end-to-end request latency (mean, p50, p95, p99) and throughput under concurrent load via `tests/latency_test.py`.
'''
    write_file(DOC_DIR / "PROJECT_REPORT.md", report_md)

    slides_md = '''# Final Presentation Outline

1. **Problem & Dataset:** Retail catalog & transaction retrieval challenge; time-based split and cold-start skew.
2. **Item Features & Tower:** `article_id`, category embeddings, `item_pop_log`, recency, and embedding export checks.
3. **Baseline vs. Two-Tower & Category Slice:** Top-K accuracy comparison and breakdown across top `product_type_name` groups.
4. **Cold-Start Routing & Redis Serving:** Fallback logic for cold/unknown items, Redis embedding lookup, and FastAPI `/recommend`.
5. **Testing, Latency & Lessons Learned:** Endpoint test suite, p50/p95/p99 latency results, and workflow automation takeaways.
'''
    write_file(DOC_DIR / "PRESENTATION_OUTLINE.md", slides_md)


if __name__ == "__main__":
    main()
