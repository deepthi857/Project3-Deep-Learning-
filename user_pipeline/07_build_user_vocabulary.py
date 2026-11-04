from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType

spark = SparkSession.builder \
    .appName("BuildUserVocabulary") \
    .master("local[2]") \
    .getOrCreate()

# Load cleaned customers
customers = spark.read.parquet(
    "data/processed/customers_cleaned"
)

# Select unique customer IDs
unique_users = customers.select(
    "customer_id"
).dropDuplicates()

# Sort users alphabetically and assign consecutive IDs
sorted_users = unique_users.rdd \
    .sortBy(lambda row: row["customer_id"])

indexed_users = sorted_users.zipWithIndex().map(
    lambda item: (item[0]["customer_id"], item[1])
)

# Create vocabulary DataFrame
schema = StructType([
    StructField("customer_id", StringType(), False),
    StructField("user_id", LongType(), False)
])

user_vocabulary = spark.createDataFrame(
    indexed_users,
    schema
)

# Display sample vocabulary
print("USER VOCABULARY")
user_vocabulary.show(10, truncate=False)

# Display total users
print("TOTAL USERS:", user_vocabulary.count())

# Save vocabulary
output_path = "data/processed/user_vocabulary"

user_vocabulary.write \
    .mode("overwrite") \
    .parquet(output_path)

print("USER VOCABULARY SAVED SUCCESSFULLY!")
print("Output location:", output_path)

spark.stop()
