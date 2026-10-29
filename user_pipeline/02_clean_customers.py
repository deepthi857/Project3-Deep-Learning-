import os


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, upper

spark = SparkSession.builder \
    .appName("CleanCustomerData") \
    .master("local[2]") \
    .getOrCreate()

customer_path = os.environ.get(
    "CUSTOMERS_CSV",
    "/mnt/c/Users/shash/Desktop/customers.csv/customers.csv"
)
customers = spark.read.csv(
    customer_path,
    header=True,
    inferSchema=True
)

# 1. Fill missing age using median
median_age = customers.approxQuantile("age", [0.5], 0.001)[0]

customers = customers.fillna({
    "age": int(median_age)
})

# 2. Handle missing membership status
customers = customers.withColumn(
    "club_member_status",
    upper(trim(col("club_member_status")))
)

customers = customers.fillna({
    "club_member_status": "UNKNOWN"
})

# 3. Handle missing fashion news frequency
customers = customers.withColumn(
    "fashion_news_frequency",
    upper(trim(col("fashion_news_frequency")))
)

customers = customers.fillna({
    "fashion_news_frequency": "UNKNOWN"
})

# Display cleaned data
print("CLEANED CUSTOMER DATA")
customers.show(10, truncate=False)

print("TOTAL CUSTOMERS:", customers.count())

# 4. Save cleaned customer dataset as Parquet
output_path = "data/processed/customers_cleaned"

customers.write \
    .mode("overwrite") \
    .parquet(output_path)

print("CLEANED CUSTOMER DATA SAVED SUCCESSFULLY!")
print("Output location:", output_path)

spark.stop()
