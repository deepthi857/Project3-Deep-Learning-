from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

PROJECT = Path(__file__).resolve().parents[1]
ML_PYTHON = "/home/shash/project3-venv311/bin/python"
SPARK_SUBMIT = "/home/shash/project3-venv311/bin/spark-submit"

default_args = {
    "owner": "project3-team",
    "depends_on_past": False,
}

with DAG(
    dag_id="hm_recommendation_pipeline",
    default_args=default_args,
    description="H&M recommendation pipeline: Spark, features, vocabulary, training and Redis",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["project3", "recommendations", "deep-learning"],
) as dag:

    spark_prep = BashOperator(
        task_id="spark_prep",
        bash_command=(
            f"cd {PROJECT} && "
            f"export PYTHONPATH=/home/shash/project3-venv311/lib/python3.11/site-packages && "
            f"export PYSPARK_PYTHON={ML_PYTHON} && "
            f"{SPARK_SUBMIT} spark_jobs/spark_prep.py 12"
        ),
    )

    features = BashOperator(
        task_id="features",
        bash_command=(
            f"cd {PROJECT} && "
	    f"export PYTHONPATH=/home/shash/project3-venv311/lib/python3.11/site-packages && "
	    f"export PYSPARK_PYTHON={ML_PYTHON} && "
	    f"{SPARK_SUBMIT} spark_jobs/spark_prep.py 12"
        ),
    )

    vocab = BashOperator(
        task_id="vocab",
        bash_command=(
            f"cd {PROJECT} && "
            f"mkdir -p artifacts && "
            f"{ML_PYTHON} training/vocab.py"
        ),
    )

    train = BashOperator(
        task_id="train",
        bash_command=(
            f"cd {PROJECT} && "
	    f"export PYTHONPATH={PROJECT} && "
            f"{ML_PYTHON} recsys/week2/train_first_run.py "
            "--epochs 1 --dim 64 --batch 128 "
            "--train-rows 50000 --val-rows 10000 --export"
        ),
    )

    load_redis = BashOperator(
        task_id="load_redis",
        bash_command=(
            f"cd {PROJECT} && "
            f"{ML_PYTHON} serving/load_redis.py"
        ),
    )

    spark_prep >> features >> vocab >> train >> load_redis
