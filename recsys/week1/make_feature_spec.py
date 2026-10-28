"""Week 1 Day 7: write artifacts/feature_spec.json from vocab.json."""
import json
from recsys.week1.config import ARTIFACTS

V = json.load(open(f"{ARTIFACTS}/vocab.json"))
spec = {"user_tower": {}, "item_tower": {}}
for f in ["customer_id", "age_bucket", "club_member_status", "fashion_news_frequency", "dow", "month"]:
    spec["user_tower"][f] = {"type": "categorical", "vocab_size": len(V[f]),
                             "embedding_dim": 64 if f == "customer_id" else 16}
spec["user_tower"]["history"] = {"type": "sequence(article_id)", "pooling": "mean", "length": 10}
spec["user_tower"]["user_recency"] = {"type": "numeric", "transform": "log1p(days)"}
for f in ["article_id", "product_type_name", "colour_group_name", "department_name", "garment_group_name"]:
    spec["item_tower"][f] = {"type": "categorical", "vocab_size": len(V[f]),
                             "embedding_dim": 64 if f == "article_id" else 16}
spec["item_tower"]["item_pop_log"] = {"type": "numeric", "transform": "log1p(prev-week purchases)"}
json.dump(spec, open(f"{ARTIFACTS}/feature_spec.json", "w"), indent=2)
print("feature_spec.json written")

