import os
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TransactionDataInspection") \
    .master("local[2]") \
    .getOrCreate()

transaction_path = os.environ.get(
    "TRANSACTIONS_CSV",
    "/mnt/c/Users/shash/Desktop/transactions_train.csv/transactions_train.csv"
)

transactions = spark.read.csv(
    transaction_path,
    header=True,
    inferSchema=False
)

print("TRANSACTION DATASET SCHEMA")
transactions.printSchema()

print("FIRST 5 TRANSACTION RECORDS")
transactions.show(5, truncate=False)

spark.stop()
