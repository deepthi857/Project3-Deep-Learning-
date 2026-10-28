"""Week 1 (Day 4-6): contextual features (time, recency, popularity, history).
Run: spark-submit spark_jobs/features.py
Leakage-safe: history uses only PRIOR rows; popularity uses the PREVIOUS week.
"""
from pyspark.sql import SparkSession, functions as F, Window as W

P, OUT = "data/processed", "data/features"
TRAIN_SAMPLE_FRAC = 0.1   # lower/raise depending on your laptop RAM

spark = (SparkSession.builder.appName("hm-features")
         .config("spark.driver.memory", "8g").getOrCreate())
tx = spark.read.parquet(f"{P}/transactions")
cust = spark.read.parquet(f"{P}/customers")
art = spark.read.parquet(f"{P}/articles")
max_date = tx.agg(F.max("t_dat")).first()[0]

# ---- time context
tx = (tx.withColumn("dow", F.dayofweek("t_dat").cast("string"))
        .withColumn("month", F.month("t_dat").cast("string"))
        .withColumn("week", F.date_trunc("week", "t_dat").cast("date")))

# ---- user recency + last-10-item history (prior purchases only)
w = W.partitionBy("customer_id").orderBy("t_dat")
tx = (tx.withColumn("prev_date", F.lag("t_dat").over(w))
        .withColumn("days_since_last", F.coalesce(F.datediff("t_dat", "prev_date"), F.lit(0)))
        .withColumn("user_recency", F.log1p("days_since_last").cast("float"))
        .withColumn("hist_list", F.collect_list("article_id").over(w.rowsBetween(-10, -1)))
        .withColumn("history", F.when(F.size("hist_list") == 0, F.lit("PAD"))
                                 .otherwise(F.concat_ws(" ", "hist_list"))))

# ---- item popularity over time: purchases of the item in the previous week
pop = (tx.groupBy("article_id", "week").count()
         .withColumn("week", F.date_add("week", 7))
         .withColumnRenamed("count", "pop_prev_week"))
tx = (tx.join(pop, ["article_id", "week"], "left")
        .withColumn("item_pop_log", F.log1p(F.coalesce("pop_prev_week", F.lit(0))).cast("float")))

# ---- attach demographics + item metadata
cust_f = cust.select("customer_id", "age_bucket", "club_member_status", "fashion_news_frequency")
art_f = art.select("article_id", "product_type_name", "colour_group_name",
                   "department_name", "garment_group_name")
cols = ["customer_id", "article_id", "age_bucket", "club_member_status",
        "fashion_news_frequency", "dow", "month", "history", "user_recency",
        "product_type_name", "colour_group_name", "department_name",
        "garment_group_name", "item_pop_log", "t_dat"]
data = tx.join(cust_f, "customer_id", "left").join(art_f, "article_id", "left") \
         .fillna("unknown").select(*cols)

cutoff = F.date_sub(F.lit(max_date), 7)          # time-based split
train = data.filter(F.col("t_dat") < cutoff).drop("t_dat")
val = data.filter(F.col("t_dat") >= cutoff).drop("t_dat")
train.write.mode("overwrite").parquet(f"{OUT}/train")
train.sample(False, TRAIN_SAMPLE_FRAC, 42).write.mode("overwrite").parquet(f"{OUT}/train_sample")
val.write.mode("overwrite").parquet(f"{OUT}/val")

# ---- item table (candidate set) + popular fallback list
recent = (tx.filter(F.col("t_dat") > F.date_sub(F.lit(max_date), 7))
            .groupBy("article_id").count())
seen = tx.select("article_id").distinct()
items = (art_f.join(seen, "article_id")
         .join(recent, "article_id", "left")
         .withColumn("item_pop_log", F.log1p(F.coalesce("count", F.lit(0))).cast("float")))
items.drop("count").write.mode("overwrite").parquet(f"{OUT}/items")
(items.orderBy(F.desc(F.coalesce("count", F.lit(0)))).select("article_id").limit(100)
      .write.mode("overwrite").parquet(f"{OUT}/popular"))

# ---- user state for serving (last 10 items overall, "tomorrow" context)
last = (tx.groupBy("customer_id")
          .agg(F.sort_array(F.collect_list(F.struct("t_dat", "article_id"))).alias("s"),
               F.max("t_dat").alias("last_date")))
last = (last.withColumn("history", F.concat_ws(" ", F.slice(F.col("s.article_id"), -10, 10)))
            .withColumn("user_recency", F.log1p(F.datediff(F.lit(max_date), "last_date")).cast("float"))
            .withColumn("dow", F.dayofweek(F.date_add(F.lit(max_date), 1)).cast("string"))
            .withColumn("month", F.month(F.date_add(F.lit(max_date), 1)).cast("string"))
            .join(cust_f, "customer_id", "left").fillna("unknown")
            .select("customer_id", "age_bucket", "club_member_status",
                    "fashion_news_frequency", "dow", "month", "history", "user_recency"))
last.write.mode("overwrite").parquet(f"{OUT}/user_state")
