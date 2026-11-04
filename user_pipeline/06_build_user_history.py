import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, row_number, collect_list, struct,
    sort_array, expr
)
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("BuildUserPurchaseHistory") \
    .master("local[2]") \
    .getOrCreate()

# Load transaction dataset
transaction_path = os.environ.get(
    "TRANSACTIONS_CSV",
    "/mnt/c/Users/shash/Desktop/transactions_train.csv/transactions_train.csv"
)

transactions = spark.read.csv(
    transaction_path,
    header=True,
    inferSchema=False
)

# Remove records with missing important values
transactions = transactions.filter(
    col("customer_id").isNotNull() &
    col("article_id").isNotNull() &
    col("t_dat").isNotNull()
)

# Define ordering: latest purchase first
window_spec = Window.partitionBy("customer_id").orderBy(
    col("t_dat").desc(),
    col("article_id").asc()
)

# Assign row numbers to purchases
ranked_transactions = transactions.withColumn(
    "row_num",
    row_number().over(window_spec)
)

# Keep only the latest 10 purchases
last_10 = ranked_transactions.filter(
    col("row_num") <= 10
)

# Collect the 10 article IDs for each customer in date order
user_history = last_10.groupBy("customer_id").agg(
    sort_array(
        collect_list(
            struct(
                col("row_num"),
                col("article_id")
            )
        )
    ).alias("history_struct")
)

# Extract article IDs only
user_history = user_history.withColumn(
    "last_10_items",
    expr("transform(history_struct, x -> x.article_id)")
).drop("history_struct")

# Display results
print("USER LAST-10-ITEMS HISTORY")
user_history.show(10, truncate=False)

# Save results
output_path = "data/processed/user_history"

user_history.write \
    .mode("overwrite") \
    .parquet(output_path)

print("USER HISTORY SAVED SUCCESSFULLY!")
print("Output location:", output_path)

spark.stop()
