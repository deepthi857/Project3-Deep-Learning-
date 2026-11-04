"""Week 2-3: train, evaluate, export embeddings.   Run: python training/train.py"""
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from model import UserTower, ItemTower, TwoTower, USER_CATS, ITEM_CATS

F, EPOCHS, BATCH = "data/features", 3, 1024
V = json.load(open("artifacts/vocab.json"))
UCOLS = ["customer_id", "history"] + USER_CATS
ICOLS = ["article_id"] + ITEM_CATS


def to_ds(df, str_cols, float_cols):
    d = {c: df[c].astype(str).values for c in str_cols}
    d.update({c: df[c].astype("float32").values for c in float_cols})
    return tf.data.Dataset.from_tensor_slices(d)


def pair_ds(df):
    return to_ds(df, UCOLS + ICOLS, ["user_recency", "item_pop_log"])


train = pd.read_parquet(f"{F}/train_sample")
val = pd.read_parquet(f"{F}/val")
val = val.sample(n=min(50_000, len(val)), random_state=1)
items = pd.read_parquet(f"{F}/items")
items_ds = to_ds(items, ICOLS, ["item_pop_log"])

user_model, item_model = UserTower(V), ItemTower(V)
model = TwoTower(user_model, item_model, items_ds)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

train_ds = pair_ds(train).shuffle(100_000).batch(BATCH).prefetch(2)
val_ds = pair_ds(val).batch(BATCH).prefetch(2)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
metrics = model.evaluate(val_ds, return_dict=True)
json.dump({k: float(v) for k, v in metrics.items()}, open("artifacts/metrics.json", "w"), indent=2)
print(metrics)

# ---- export embeddings for serving
item_emb = item_model.predict(items_ds.batch(2048))
np.save("artifacts/item_emb.npy", item_emb.astype("float32"))
json.dump(items.article_id.astype(str).tolist(), open("artifacts/item_ids.json", "w"))

users = pd.read_parquet(f"{F}/user_state")
users_ds = to_ds(users, UCOLS, ["user_recency"]).batch(4096)
np.save("artifacts/user_emb.npy", user_model.predict(users_ds).astype("float32"))
json.dump(users.customer_id.astype(str).tolist(), open("artifacts/user_ids.json", "w"))
