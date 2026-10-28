"""Week 1 Day 2-3: data-quality gate. Run: python -m recsys.week1.data_quality_check"""
from pyspark.sql import functions as F
from recsys.week1.config import get_spark, PROCESSED
from recsys.week1.utils import null_report

spark = get_spark("dq-check")
tx = spark.read.parquet(f"{PROCESSED}/transactions")
cust = spark.read.parquet(f"{PROCESSED}/customers")
art = spark.read.parquet(f"{PROCESSED}/articles")

for name, df in [("transactions", tx), ("customers", cust), ("articles", art)]:
    print(name, "rows:", df.count())
    null_report(df, name)

d = tx.agg(F.min("t_dat"), F.max("t_dat")).first()
print("date range:", d[0], "->", d[1])
print("exact duplicate rows in tx:", tx.count() - tx.dropDuplicates().count())
print("cold users %:", round(100 * cust.filter("is_cold_user").count() / cust.count(), 2))
print("cold items %:", round(100 * art.filter("is_cold_item").count() / art.count(), 2))

assert tx.filter(F.col("price") <= 0).count() == 0, "non-positive prices found"
assert cust.filter(F.col("age_bucket").isNull()).count() == 0, "age_bucket has nulls"
print("DATA QUALITY OK")

