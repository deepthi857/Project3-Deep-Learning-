"""Week 2 Day 14: export L2-normalised user/item embeddings for serving."""
import json
import numpy as np
import pandas as pd
from recsys.week2.data_pipeline import to_ds, UCOLS, ICOLS

F, A = "data/features", "artifacts"


def _norm(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)).astype("float32")


def export(user_model, item_model, items_ds, items_df):
    np.save(f"{A}/item_emb.npy", _norm(item_model.predict(items_ds.batch(2048))))
    json.dump(items_df.article_id.astype(str).tolist(), open(f"{A}/item_ids.json", "w"))
    users = pd.read_parquet(f"{F}/user_state")
    u_ds = to_ds(users, UCOLS, ["user_recency"]).batch(4096)
    np.save(f"{A}/user_emb.npy", _norm(user_model.predict(u_ds)))
    json.dump(users.customer_id.astype(str).tolist(), open(f"{A}/user_ids.json", "w"))
    print("exported embeddings to", A)

