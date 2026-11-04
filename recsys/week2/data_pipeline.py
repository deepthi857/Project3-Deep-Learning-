"""Week 2 Day 10: tf.data pipeline."""
import pandas as pd
import tensorflow as tf
from training.model import USER_CATS, ITEM_CATS

F = "data/features"
UCOLS = ["customer_id", "history"] + USER_CATS
ICOLS = ["article_id"] + ITEM_CATS


def to_ds(df, str_cols, float_cols):
    d = {c: df[c].astype(str).values for c in str_cols}
    d.update({c: df[c].astype("float32").values for c in float_cols})
    return tf.data.Dataset.from_tensor_slices(d)


def pair_ds(df):
    return to_ds(df, UCOLS + ICOLS, ["user_recency", "item_pop_log"])


def load_all(batch=1024, val_rows=50_000, train_rows=None):
    train = pd.read_parquet(f"{F}/train_sample")
    if train_rows:
        train = train.sample(n=min(train_rows, len(train)), random_state=0)
    val = pd.read_parquet(f"{F}/val")
    val = val.sample(n=min(val_rows, len(val)), random_state=1)
    items = pd.read_parquet(f"{F}/items")
    train_ds = pair_ds(train).shuffle(100_000).batch(batch).prefetch(2)
    val_ds = pair_ds(val).batch(batch).prefetch(2)
    items_ds = to_ds(items, ICOLS, ["item_pop_log"])
    return train_ds, val_ds, items_ds, items

