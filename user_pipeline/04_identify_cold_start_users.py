import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

spark = SparkSession.builder \
    .appName("IdentifyColdStartUsers") \
    .master("local[2]") \
    .getOrCreate()

# Load cleaned customers
customers = spark.read.parquet(
    "data/processed/customers_cleaned"
)

# Load transaction data
transaction_path = os.environ.get(
    "TRANSACTIONS_CSV",
    "/mnt/c/Users/shash/Desktop/transactions_train.csv/transactions_train.csv"
)

transactions = spark.read.csv(
    transaction_path,
    header=True,
    inferSchema=False
)

# Get unique customers who have made purchases
active_customers = transactions.select(
    "customer_id"
).dropDuplicates()

# Find customers without any purchase history
cold_start_users = customers.join(
    active_customers,
    on="customer_id",
    how="left_anti"
)

# Add cold-start flag
cold_start_users = cold_start_users.withColumn(
    "is_cold_start",
    lit(1)
)

# Display results
print("COLD-START USERS")
cold_start_users.show(10, truncate=False)

print("TOTAL COLD-START USERS:", cold_start_users.count())

# Save results
output_path = "data/processed/cold_start_users"

cold_start_users.write \
    .mode("overwrite") \
    .parquet(output_path)

print("COLD-START USERS SAVED SUCCESSFULLY!")
print("Output location:", output_path)

spark.stop()
