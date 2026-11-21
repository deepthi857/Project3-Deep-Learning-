import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, DateType, StringType,
                               DoubleType, IntegerType)

RAW, OUT = "data/raw", "data/processed"
WEEKS = int(sys.argv[1]) if len(sys.argv) > 1 else 12

spark = (SparkSession.builder.appName("hm-prep")
         .config("spark.driver.memory", "8g").getOrCreate())

tx_schema = StructType([
    StructField("t_dat", DateType()), StructField("customer_id", StringType()),
    StructField("article_id", StringType()), StructField("price", DoubleType()),
    StructField("sales_channel_id", IntegerType())])
tx = spark.read.csv(f"{RAW}/transactions_train.csv", header=True, schema=tx_schema)
cust = spark.read.csv(f"{RAW}/customers.csv", header=True, inferSchema=True)
art = spark.read.csv(f"{RAW}/articles.csv", header=True)  
tx = tx.dropna(subset=["t_dat", "customer_id", "article_id"]).filter(F.col("price") > 0)
max_date = tx.agg(F.max("t_dat")).first()[0]
tx = tx.filter(F.col("t_dat") > F.date_sub(F.lit(max_date), WEEKS * 7))
median_age = cust.approxQuantile("age", [0.5], 0.01)[0]
cust = (cust
    .withColumn("age", F.coalesce("age", F.lit(median_age)))
    .withColumn("age_bucket",
        F.when(F.col("age") < 20, "<20").when(F.col("age") < 25, "20-24")
         .when(F.col("age") < 35, "25-34").when(F.col("age") < 45, "35-44")
         .when(F.col("age") < 55, "45-54").when(F.col("age") < 65, "55-64")
         .otherwise("65+"))
    .withColumn("club_member_status", F.upper(F.coalesce("club_member_status", F.lit("UNKNOWN"))))
    .withColumn("fashion_news_frequency", F.upper(F.coalesce("fashion_news_frequency", F.lit("NONE")))))
u_cnt = tx.groupBy("customer_id").count().withColumnRenamed("count", "u_cnt")
i_cnt = tx.groupBy("article_id").count().withColumnRenamed("count", "i_cnt")
cust = (cust.join(u_cnt, "customer_id", "left")
        .withColumn("is_cold_user", F.coalesce("u_cnt", F.lit(0)) < 3).drop("u_cnt")
        .select("customer_id", "age_bucket", "club_member_status",
                "fashion_news_frequency", "is_cold_user"))

keep = ["article_id", "product_type_name", "colour_group_name",
        "department_name", "garment_group_name"]
art = (art.select(*keep).fillna("unknown")
       .join(i_cnt, "article_id", "left")
       .withColumn("is_cold_item", F.coalesce("i_cnt", F.lit(0)) < 5).drop("i_cnt"))

tx.write.mode("overwrite").parquet(f"{OUT}/transactions")
cust.write.mode("overwrite").parquet(f"{OUT}/customers")
art.write.mode("overwrite").parquet(f"{OUT}/articles")
print("rows:", tx.count(), "| max_date:", max_date)
