# Final Presentation Outline

1. **Problem & Dataset:** Retail catalog & transaction retrieval challenge; time-based split and cold-start skew.
2. **Item Features & Tower:** `article_id`, category embeddings, `item_pop_log`, recency, and embedding export checks.
3. **Baseline vs. Two-Tower & Category Slice:** Top-K accuracy comparison and breakdown across top `product_type_name` groups.
4. **Cold-Start Routing & Redis Serving:** Fallback logic for cold/unknown items, Redis embedding lookup, and FastAPI `/recommend`.
5. **Testing, Latency & Lessons Learned:** Endpoint test suite, p50/p95/p99 latency results, and workflow automation takeaways.
