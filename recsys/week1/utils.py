"""Week 1 Day 2: shared cleaning helpers for Person 1 and Person 2."""
from pyspark.sql import functions as F


def fill_unknown(df, cols, token="unknown"):
    return df.fillna({c: token for c in cols})


def age_bucket(col="age"):
    c = F.col(col)
    return (F.when(c < 20, "<20").when(c < 25, "20-24").when(c < 35, "25-34")
             .when(c < 45, "35-44").when(c < 55, "45-54").when(c < 65, "55-64")
             .otherwise("65+"))


def null_report(df, name):
    """Print null count per column."""
    exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    print(f"--- nulls in {name} ---")
    print(df.agg(*exprs).first().asDict())


def write_parquet(df, path):
    df.write.mode("overwrite").parquet(path)
    print("wrote", path)

