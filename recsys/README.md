# Context-aware two-tower recommender (H&M)

Run order (from the project root, Python environment active):

1. Put transactions_train.csv, customers.csv and articles.csv in data/raw/
2. `spark-submit spark_jobs/spark_prep.py 12`
3. `spark-submit spark_jobs/features.py`
4. `python training/vocab.py`
5. Train on a GPU (Colab): `python -m recsys.week2.train_first_run --epochs 3`
6. Start Redis (Memurai on Windows, or docker compose -f recsys/week3/docker-compose.yml up -d)
7. `python serving/load_redis.py` then `python -m recsys.week3.redis_check`
8. `uvicorn recsys.week4.app:app` and test `/recommend/{customer_id}`
9. `python -m recsys.week4.integration_test`

Folders: recsys/week1 (checks), recsys/week2 (model and training), recsys/week3 (Redis, error analysis, final model), recsys/week4 (API, tests, demo), airflow/ (pipeline DAG).
