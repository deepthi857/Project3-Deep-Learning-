# Two-Tower Recommendation System - Item, Evaluation & Serving Summary

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
