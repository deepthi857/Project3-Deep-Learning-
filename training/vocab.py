"""Week 1 Day 7: build vocabularies for embedding layers -> artifacts/vocab.json"""
import json
import pandas as pd

F = "data/features"
items = pd.read_parquet(f"{F}/items")
users = pd.read_parquet(f"{F}/user_state")
train = pd.read_parquet(f"{F}/train_sample", columns=[
    "age_bucket", "club_member_status", "fashion_news_frequency", "dow", "month"])

def uniq(*series):
    return sorted(set(pd.concat(series).astype(str)))

vocab = {
    "customer_id": users.customer_id.astype(str).unique().tolist(),
    "article_id": items.article_id.astype(str).unique().tolist(),
    "product_type_name": uniq(items.product_type_name),
    "colour_group_name": uniq(items.colour_group_name),
    "department_name": uniq(items.department_name),
    "garment_group_name": uniq(items.garment_group_name),
    "age_bucket": uniq(train.age_bucket, users.age_bucket),
    "club_member_status": uniq(train.club_member_status, users.club_member_status),
    "fashion_news_frequency": uniq(train.fashion_news_frequency, users.fashion_news_frequency),
    "dow": [str(i) for i in range(1, 8)],
    "month": [str(i) for i in range(1, 13)],
}
json.dump(vocab, open("artifacts/vocab.json", "w"))
print({k: len(v) for k, v in vocab.items()})
