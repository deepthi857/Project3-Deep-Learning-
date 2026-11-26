"""Week 2: item tower checks and evaluation (Days 8-14).
Run all days:  python week2_item_checks.py
Run one day:   python week2_item_checks.py 12
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("docs/week2")
OUT.mkdir(parents=True, exist_ok=True)


def report(name, lines):
    text = "\n".join(str(x) for x in lines)
    print(text)
    (OUT / name).write_text(text, encoding="utf-8")
    print("saved", OUT / name, "\n")


def baseline_top(tx):
    """100 most popular items in the 4 weeks before the validation week (same split as the model)."""
    cutoff = tx.t_dat.max() - pd.Timedelta(days=7)
    hist = tx[(tx.t_dat < cutoff) & (tx.t_dat >= cutoff - pd.Timedelta(days=28))]
    return hist.article_id.value_counts().index.tolist()[:100], tx[tx.t_dat >= cutoff]


def day8():  # item tower design table from the feature spec
    spec = json.load(open("artifacts/feature_spec.json"))["item_tower"]
    L = ["# Item tower design", "", "| Feature | Type | Vocabulary size | Embedding size |", "|---|---|---|---|"]
    L += [f"| {k} | {v['type']} | {v.get('vocab_size', '-')} | {v.get('embedding_dim', '-')} |"
          for k, v in spec.items()]
    L += ["", "All embeddings and the popularity value are concatenated, passed through a",
          "Dense(128, relu) layer and a Dense(64) output layer. The output is the item vector."]
    report("ITEM_TOWER_DESIGN.md", L)


def day9():  # trainable parameters of the item tower
    V = json.load(open("artifacts/vocab.json"))
    rows = [("article_id embedding", (len(V["article_id"]) + 1) * 64)]
    for c in ["product_type_name", "colour_group_name", "department_name", "garment_group_name"]:
        rows.append((f"{c} embedding", (len(V[c]) + 1) * 16))
    inp = 64 + 4 * 16 + 1                       # id + 4 categories + popularity value
    rows += [(f"Dense({inp} -> 128)", inp * 128 + 128), ("Dense(128 -> 64)", 128 * 64 + 64)]
    L = [f"{name:<32}{n:>12,}" for name, n in rows]
    report("w2_day9_item_tower_params.txt", L + [f"{'TOTAL':<32}{sum(n for _, n in rows):>12,}"])


def day10():  # candidate set and the best possible validation score
    items = pd.read_parquet("data/features/items")
    val = pd.read_parquet("data/features/val", columns=["article_id"]).article_id
    L = [f"candidate items: {len(items):,}",
         f"validation purchases whose item is a candidate: {val.isin(set(items.article_id)).mean() * 100:.2f}%",
         "(this is the best possible score, since the model can only recommend candidate items)",
         "", "candidates by garment group (top 10):", items.garment_group_name.value_counts().head(10).to_string()]
    report("w2_day10_candidate_set_report.txt", L)


def day11():  # sanity checks on the exported item vectors
    emb = np.load("artifacts/item_emb.npy")
    ids = json.load(open("artifacts/item_ids.json"))
    items = pd.read_parquet("data/features/items")
    n = np.linalg.norm(emb, axis=1)
    L = [f"item vectors: {emb.shape}, item ids: {len(ids):,}, item table rows: {len(items):,}",
         f"NaN values: {int(np.isnan(emb).sum())}",
         f"vector length: min {n.min():.3f}, mean {n.mean():.3f}, max {n.max():.3f} (about 1 means L2-normalised)"]
    assert emb.shape[0] == len(ids) == len(items), "sizes do not match"
    assert not np.isnan(emb).any(), "NaN values found"
    assert ids == items.article_id.astype(str).tolist(), "id order does not match the item table"
    report("w2_day11_embedding_sanity.txt", L + ["", "ITEM EMBEDDINGS OK"])


def day12():  # popularity baseline compared with the model
    tx = pd.read_parquet("data/processed/transactions", columns=["t_dat", "article_id"])
    tx["t_dat"] = pd.to_datetime(tx.t_dat)
    top, val = baseline_top(tx)
    m = json.load(open("artifacts/metrics.json"))
    L = [f"baseline: recommend the {len(top)} most popular items of the 4 weeks before the validation week",
         f"validation purchases: {len(val):,}", "",
         f"{'top-K':<8}{'popularity baseline':>22}{'two-tower model':>20}"]
    for k in (1, 5, 10, 50, 100):
        base = val.article_id.isin(top[:k]).mean()
        L.append(f"{k:<8}{base * 100:>21.2f}%{m[f'factorized_top_k/top_{k}_categorical_accuracy'] * 100:>19.2f}%")
    L += ["", "The model score uses a 50,000-row validation sample; the baseline uses all validation rows."]
    report("w2_day12_popularity_baseline.txt", L)


def day13():  # garment groups: share of purchases and baseline coverage
    tx = pd.read_parquet("data/processed/transactions", columns=["t_dat", "article_id"])
    a = pd.read_parquet("data/processed/articles", columns=["article_id", "garment_group_name"])
    tx["t_dat"] = pd.to_datetime(tx.t_dat)
    top, val = baseline_top(tx)
    val = val.merge(a, on="article_id", how="left")
    val["hit"] = val.article_id.isin(top)
    t = val.groupby("garment_group_name").agg(purchases=("hit", "size"), baseline_hit=("hit", "mean"))
    t["share_of_purchases_%"] = (t.purchases / t.purchases.sum() * 100).round(2)
    t["baseline_hit_%"] = (t.baseline_hit * 100).round(2)
    report("w2_day13_category_analysis.txt",
           ["validation purchases by garment group (popularity top-100 baseline):", "",
            t.sort_values("purchases", ascending=False)
            [["purchases", "share_of_purchases_%", "baseline_hit_%"]].head(12).to_string()])


def day14():  # similar items from the exported item vectors
    emb = np.load("artifacts/item_emb.npy")
    ids = json.load(open("artifacts/item_ids.json"))
    items = pd.read_parquet("data/features/items").set_index("article_id")
    cols = ["product_type_name", "colour_group_name", "garment_group_name"]
    L = []
    for i in np.random.default_rng(0).choice(len(ids), 3, replace=False):
        sim = emb @ emb[i]                      # vectors are L2-normalised, so this is cosine similarity
        L += ["", f"item {ids[i]}: " + ", ".join(items.loc[ids[i], cols].astype(str))]
        L += [f"   {sim[j]:.3f}  {ids[j]}: " + ", ".join(items.loc[ids[j], cols].astype(str))
              for j in np.argsort(-sim)[1:6]]
    report("w2_day14_similar_items.txt", L)


DAYS = {8: day8, 9: day9, 10: day10, 11: day11, 12: day12, 13: day13, 14: day14}

if __name__ == "__main__":
    for d in [int(x) for x in sys.argv[1:]] or list(DAYS):
        print(f"===== Week 2, Day {d} =====")
        DAYS[d]()
