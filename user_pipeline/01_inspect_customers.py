import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when


# Start Spark
spark = SparkSession.builder \
    .appName("CustomerDataInspection") \
    .master("local[2]") \
    .getOrCreate()

# Customer dataset path

customer_path = os.environ.get(
    "CUSTOMERS_CSV",
    "/mnt/c/Users/shash/Desktop/customers.csv/customers.csv"
)
# Load CSV
customers = spark.read.csv(
    customer_path,
    header=True,
    inferSchema=True
)

# Display basic information
print("CUSTOMER DATASET SCHEMA")
customers.printSchema()

print("TOTAL CUSTOMER RECORDS")
print(customers.count())

print("FIRST 5 CUSTOMER RECORDS")
customers.show(5, truncate=False)



print("\nMISSING VALUES IN EACH COLUMN")

missing_values = customers.select([
    count(
        when(col(c).isNull(), c)
    ).alias(c)
    for c in customers.columns
])

missing_values.show(truncate=False)


print("\nAGE STATISTICS")

customers.select("age").describe().show()

print("\nAGE DISTRIBUTION - FIRST 20 VALUES")

customers.groupBy("age") \
    .count() \
    .orderBy("age") \
    .show(20)


print("\nMEDIAN AGE")

median_age = customers.approxQuantile(
    "age",
    [0.5],
    0.001
)[0]

print("Median Age:", median_age)

spark.stop()
