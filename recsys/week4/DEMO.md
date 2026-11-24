# Demo script

1. Check Redis (Memurai) is running: `Get-Service Memurai`
2. Check the data in Redis: `python -m recsys.week3.redis_check`
3. Start the API: `uvicorn recsys.week4.app:app`
4. Health check: `Invoke-RestMethod "http://localhost:8000/health"`
5. Known user (source = two_tower): pick an id from artifacts/user_ids.json and call `/recommend/<id>?k=5`
6. Repeat the same call (source = cache, served from Redis cache)
7. Unknown user (source = popular_fallback): `/recommend/new_user?k=5`
8. Show artifacts/metrics.json, recsys/week3/FINAL_MODEL.md and airflow/dag_pipeline.py
