"""Week 1: item-side data checks (Days 1-7).
Run all days:  python week1_item_checks.py
Run one day:   python week1_item_checks.py 3
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path("docs/week1")
OUT.mkdir(parents=True, exist_ok=True)
CATS = ["product_type_name", "colour_group_name", "department_name", "garment_group_name"]


def report(name, lines):
    text = "\n".join(str(x) for x in lines)
    print(text)
    (OUT / name).write_text(text, encoding="utf-8")
    print("saved", OUT / name, "\n")


def day1():  # load and profile the article catalog
    a = pd.read_parquet("data/processed/articles")
    L = [f"articles: {len(a):,} rows, {a.shape[1]} columns", "", "unique values per column:"]
    L += [f"  {c}: {a[c].nunique():,}" for c in a.columns]
    for c in ["product_type_name", "colour_group_name", "garment_group_name"]:
        L += ["", f"top 10 {c}:", a[c].value_counts().head(10).to_string()]
    report("w1_day1_articles_profile.txt", L)


def day2():  # check cleaning: no missing values, no duplicates
    a = pd.read_parquet("data/processed/articles")
    L = ["nulls per column:", a.isna().sum().to_string(), "",
         f"duplicate article_id: {a.article_id.duplicated().sum()}"]
    L += [f"'unknown' values in {c}: {(a[c] == 'unknown').sum():,}" for c in CATS]
    assert a.isna().sum().sum() == 0, "missing values found"
    assert not a.article_id.duplicated().any(), "duplicate article ids found"
    report("w1_day2_article_cleaning_report.txt", L + ["", "ARTICLE CLEANING OK"])


def day3():  # cold-start items (fewer than 5 purchases)
    tx = pd.read_parquet("data/processed/transactions", columns=["article_id"])
    a = pd.read_parquet("data/processed/articles", columns=["article_id", "is_cold_item"])
    cnt = tx.article_id.value_counts()
    a["n"] = a.article_id.map(cnt).fillna(0).astype(int)
    cold = a.n < 5
    L = [f"catalog items: {len(a):,}",
         f"items never bought in the window: {(a.n == 0).sum():,} ({(a.n == 0).mean() * 100:.2f}%)",
         f"cold items (fewer than 5 purchases): {cold.sum():,} ({cold.mean() * 100:.2f}%)",
         f"share of all purchases that involve cold items: {cnt[cnt < 5].sum() / cnt.sum() * 100:.2f}%",
         f"is_cold_item flag matches the count rule: {bool((cold == a.is_cold_item).all())}",
         "", "Handling: cold items still get a vector from their category, colour and department."]
    report("w1_day3_cold_items_report.txt", L)


def day4():  # item recency features
    tx = pd.read_parquet("data/processed/transactions", columns=["t_dat", "article_id"])
    tx["t_dat"] = pd.to_datetime(tx.t_dat)
    max_d = tx.t_dat.max()
    g = tx.groupby("article_id").t_dat.agg(["min", "max"])
    rec = pd.DataFrame({"days_since_first_sale": (max_d - g["min"]).dt.days,
                        "days_since_last_sale": (max_d - g["max"]).dt.days})
    Path("data/item_recency").mkdir(parents=True, exist_ok=True)
    rec.to_parquet("data/item_recency/item_recency.parquet")
    L = [f"latest date in data: {max_d.date()}",
         f"items with at least one sale: {len(rec):,}",
         f"first sale in the last 14 days: {(rec.days_since_first_sale < 14).sum():,}",
         f"not sold in the last 28 days: {(rec.days_since_last_sale >= 28).sum():,}",
         "", rec.describe().round(1).to_string()]
    report("w1_day4_item_recency_report.txt", L)


def day5():  # weekly item popularity
    tx = pd.read_parquet("data/processed/transactions", columns=["t_dat", "article_id"])
    a = pd.read_parquet("data/processed/articles",
                        columns=["article_id", "product_type_name", "colour_group_name"])
    tx["week"] = pd.to_datetime(tx.t_dat).dt.to_period("W").dt.start_time
    weekly = tx.groupby(["week", "article_id"]).size().rename("n").reset_index()
    last = weekly.week.max()
    prev = last - pd.Timedelta(days=7)
    top = weekly[weekly.week == last].nlargest(10, "n").merge(a, on="article_id", how="left")
    top["previous_week"] = top.article_id.map(
        weekly[weekly.week == prev].set_index("article_id").n).fillna(0).astype(int)
    L = [f"weeks in data: {weekly.week.nunique()}, latest week starts {last.date()}", "",
         "top 10 items in the latest week (with the previous week for comparison):",
         top[["article_id", "product_type_name", "colour_group_name", "n", "previous_week"]]
         .rename(columns={"n": "latest_week"}).to_string(index=False), "",
         "The model feature item_pop_log = log(1 + purchases in the previous week)."]
    report("w1_day5_item_popularity_report.txt", L)


def day6():  # validate the item feature table
    items = pd.read_parquet("data/features/items")
    L = [f"item table: {len(items):,} rows", "", "nulls per column:", items.isna().sum().to_string(),
         "", "item_pop_log summary:", items.item_pop_log.describe().round(3).to_string()]
    ids = set(items.article_id)
    for name in ["train_sample", "val"]:
        col = pd.read_parquet(f"data/features/{name}", columns=["article_id"]).article_id
        L.append(f"{name}: rows whose item is missing from the item table: {(~col.isin(ids)).mean() * 100:.2f}%")
    assert items.article_id.is_unique, "duplicate article ids in the item table"
    assert items.isna().sum().sum() == 0, "missing values in the item table"
    assert items.item_pop_log.between(0, 20).all(), "item_pop_log out of range"
    report("w1_day6_item_feature_validation.txt", L + ["", "ITEM FEATURES OK"])


def day7():  # check the item vocabularies
    V = json.load(open("artifacts/vocab.json"))
    items = pd.read_parquet("data/features/items")
    L = [f"{c}: vocabulary {len(V[c]):,} values, {items[c].nunique():,} unique in the item table"
         for c in ["article_id"] + CATS]
    val = pd.read_parquet("data/features/val", columns=["article_id"]).article_id
    L += ["", f"validation purchases whose item is in the vocabulary: {val.isin(set(V['article_id'])).mean() * 100:.2f}%",
          "Items outside the vocabulary map to a shared 'unknown' embedding."]
    report("w1_day7_item_vocab_report.txt", L)


DAYS = {1: day1, 2: day2, 3: day3, 4: day4, 5: day5, 6: day6, 7: day7}

if __name__ == "__main__":
    for d in [int(x) for x in sys.argv[1:]] or list(DAYS):
        print(f"===== Week 1, Day {d} =====")
        DAYS[d]()
