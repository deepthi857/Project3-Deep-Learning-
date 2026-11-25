"""Preliminary Week 3 age-group retrieval evaluation."""

import json
import numpy as np
import pandas as pd
from collections import defaultdict

ARTIFACTS = "artifacts"
VAL_FILE = "data/features/val"
BATCH_SIZE = 128
K_VALUES = (10, 50, 100)


def main():
    print("Loading validation data...", flush=True)
    val = pd.read_parquet(VAL_FILE)
    val["customer_id"] = val["customer_id"].astype(str)
    val["article_id"] = val["article_id"].astype(str)

    print("Loading embedding IDs...", flush=True)
    user_ids = json.load(open(f"{ARTIFACTS}/user_ids.json"))
    item_ids = json.load(open(f"{ARTIFACTS}/item_ids.json"))

    user_index = {uid: n for n, uid in enumerate(user_ids)}
    item_index = {iid: n for n, iid in enumerate(item_ids)}

    user_emb = np.load(f"{ARTIFACTS}/user_emb.npy", mmap_mode="r")
    item_emb = np.load(f"{ARTIFACTS}/item_emb.npy", mmap_mode="r")

    val["user_idx"] = val["customer_id"].map(user_index)
    val["item_idx"] = val["article_id"].map(item_index)

    val = val.dropna(subset=["user_idx", "item_idx"]).copy()
    val["user_idx"] = val["user_idx"].astype(int)
    val["item_idx"] = val["item_idx"].astype(int)

    # One representative age bucket per user.
    user_age = (
        val.groupby("customer_id")["age_bucket"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    # Unique user-item validation pairs avoid duplicate purchases
    # inflating the evaluation.
    pairs = val[
        ["customer_id", "user_idx", "item_idx"]
    ].drop_duplicates()

    results = defaultdict(lambda: {
        "interactions": 0,
        "hit10": 0,
        "hit50": 0,
        "hit100": 0,
    })

    total = len(pairs)
    print(f"Evaluating {total} unique user-item pairs...", flush=True)

    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs.iloc[start:start + BATCH_SIZE]

        uidx = batch["user_idx"].to_numpy()
        iidx = batch["item_idx"].to_numpy()

        # Dot product because embeddings are L2-normalised.
        scores = np.asarray(user_emb[uidx]) @ item_emb.T

        # Find top 100 candidates for each user.
        top100 = np.argpartition(
            scores, -100, axis=1
        )[:, -100:]

        top100_scores = np.take_along_axis(
            scores, top100, axis=1
        )

        order = np.argsort(top100_scores, axis=1)[:, ::-1]
        ranked = np.take_along_axis(top100, order, axis=1)

        for row, (_, record) in enumerate(batch.iterrows()):
            age = user_age[record["customer_id"]]
            target = record["item_idx"]
            ranking = ranked[row]

            hit = target in ranking

            results[age]["interactions"] += 1
            results[age]["hit10"] += int(hit and target in ranking[:10])
            results[age]["hit50"] += int(hit and target in ranking[:50])
            results[age]["hit100"] += int(hit)

        if (start + BATCH_SIZE) % 5000 < BATCH_SIZE:
            print(f"Processed {min(start + BATCH_SIZE, total)}/{total}",
                  flush=True)

    rows = []

    for age in sorted(results):
        r = results[age]
        n = r["interactions"]

        rows.append({
            "age_bucket": age,
            "interactions": n,
            "hit_rate_at_10": r["hit10"] / n,
            "hit_rate_at_50": r["hit50"] / n,
            "hit_rate_at_100": r["hit100"] / n,
        })

    output = pd.DataFrame(rows)

    output.to_csv(
        f"{ARTIFACTS}/age_group_metrics.csv",
        index=False,
    )

    output.to_json(
        f"{ARTIFACTS}/age_group_metrics.json",
        orient="records",
        indent=2,
    )

    print("\nAGE-GROUP EVALUATION RESULTS")
    print(output.to_string(index=False))
    print("\nSaved CSV and JSON results.")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
