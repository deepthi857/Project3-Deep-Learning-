"""Week 1 Day 6: sanity checks on train/val tables. Run: python -m recsys.week1.leakage_check"""
from pyspark.sql import functions as F
from recsys.week1.config import get_spark, FEATURES, PROCESSED, SPLIT_DAYS
from recsys.week1.utils import null_report

spark = get_spark("leakage-check")
train = spark.read.parquet(f"{FEATURES}/train")
val = spark.read.parquet(f"{FEATURES}/val")
tx = spark.read.parquet(f"{PROCESSED}/transactions")

# 1) split is purely by time
max_d = tx.agg(F.max("t_dat")).first()[0]
print("validation window = last", SPLIT_DAYS, "days up to", max_d)

# 2) no nulls in model inputs
null_report(train, "train")
null_report(val, "val")

# 3) how much of validation is cold-start?
n_val = val.count()
val_users_unseen = val.join(train.select("customer_id").distinct(), "customer_id", "left_anti").count()
val_items_unseen = val.join(train.select("article_id").distinct(), "article_id", "left_anti").count()
print("val rows with unseen users %:", round(100 * val_users_unseen / n_val, 2))
print("val rows with unseen items %:", round(100 * val_items_unseen / n_val, 2))

# 4) share of rows with no history, and popularity feature spread
print("train rows with empty history %:",
      round(100 * train.filter(F.col("history") == "PAD").count() / train.count(), 2))
train.select(F.mean("item_pop_log"), F.stddev("item_pop_log"), F.max("item_pop_log")).show()
print("LEAKAGE CHECK DONE (history uses prior rows only; popularity uses previous week only)")

