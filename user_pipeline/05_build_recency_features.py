import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, max, datediff, lit, count
)

spark = SparkSession.builder \
    .appName("UserRecencyFeatures") \
    .master("local[2]") \
    .getOrCreate()

# Reference date for recency calculation
reference_date = "2020-09-22"
# Load transactions
transaction_path = os.environ.get(
    "TRANSACTIONS_CSV",
    "/mnt/c/Users/shash/Desktop/transactions_train.csv/transactions_train.csv"
)

transactions = spark.read.csv(
    transaction_path,
    header=True,
    inferSchema=False
)

# Convert transaction date into date format
transactions = transactions.withColumn(
    "t_dat",
    to_date(col("t_dat"), "yyyy-MM-dd")
)

# Calculate last purchase date and total transactions
user_features = transactions.groupBy("customer_id").agg(
    max("t_dat").alias("last_purchase_date"),
    count("*").alias("total_transactions")
)

# Calculate recency in days
user_features = user_features.withColumn(
    "recency_days",
    datediff(
        lit(reference_date).cast("date"),
        col("last_purchase_date")
    )
)

# Display results
print("USER TIME AND RECENCY FEATURES")
user_features.show(10, truncate=False)

# Save output
output_path = "data/processed/user_recency_features"

user_features.write \
    .mode("overwrite") \
    .parquet(output_path)

print("RECENCY FEATURES SAVED SUCCESSFULLY!")
print("Output location:", output_path)

spark.stop()
