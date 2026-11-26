import argparse
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
