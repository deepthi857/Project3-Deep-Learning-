"""Week 1 Day 1: shared config used by every script."""
from pyspark.sql import SparkSession

RAW, PROCESSED, FEATURES, ARTIFACTS = "data/raw", "data/processed", "data/features", "artifacts"
WEEKS_TO_KEEP = 12          # time window kept from the 31M-row transactions file
SPLIT_DAYS = 7              # last 7 days = validation
COLD_USER_MIN = 3           # users with < 3 purchases are cold-start
COLD_ITEM_MIN = 5           # items with < 5 purchases are cold-start
TRAIN_SAMPLE_FRAC = 0.1


def get_spark(name="hm-recsys", mem="8g"):
    return (SparkSession.builder.appName(name)
            .config("spark.driver.memory", mem)
            .config("spark.sql.shuffle.partitions", "200").getOrCreate())

