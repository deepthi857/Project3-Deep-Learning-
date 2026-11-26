"""Week 3: fallback, cold-start routing, and per-category evaluation (Days 15-21).
Run all steps:  python week3_item_checks.py
Run one step:   python week3_item_checks.py 15
Run unit tests: python -m unittest discover -s tests -p "test_item_routing.py"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("docs/week3")
OUT.mkdir(parents=True, exist_ok=True)
TEST_DIR = Path("tests")
TEST_DIR.mkdir(parents=True, exist_ok=True)


def report(name, lines):
    text = "\n".join(str(x) for x in lines)
    print(text)
    (OUT / name).write_text(text, encoding="utf-8")
    print("saved", OUT / name, "\n")


class ItemFallbackRouter:
    """Routes known active items via cosine similarity on item embeddings,
    and falls back to popular items for cold or unseen items.
    """

    def __init__(self, item_ids, emb, popular_ids, cold_ids=None):
        self.item_ids = [str(x) for x in item_ids]
        self.id_to_idx = {aid: i for i, aid in enumerate(self.item_ids)}
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.norm_emb = emb / norms
        self.popular_ids = [str(x) for x in popular_ids]
        self.cold_ids = {str(x) for x in (cold_ids or set())}

    def is_fallback_needed(self, item_id):
        aid = str(item_id)
        return (aid not in self.id_to_idx) or (aid in self.cold_ids)

    def recommend(self, seed_item_id, k=10):
        aid = str(seed_item_id)
        if self.is_fallback_needed(aid):
            return self.popular_ids[:k], "popular_fallback"

        idx = self.id_to_idx[aid]
        scores = self.norm_emb @ self.norm_emb[idx]
        scores[idx] = -np.inf
        top_idx = np.argsort(-scores)[:k]
        recs = [self.item_ids[i] for i in top_idx]
        if len(recs) < k:
            for pid in self.popular_ids:
                if pid not in recs and pid != aid:
                    recs.append(pid)
                if len(recs) == k:
                    break
        return recs[:k], "embedding_similarity"


def baseline_popular_and_val(tx):
    cutoff = tx.t_dat.max() - pd.Timedelta(days=7)
    hist = tx[(tx.t_dat < cutoff) & (tx.t_dat >= cutoff - pd.Timedelta(days=28))]
    pop_ids = [str(x) for x in hist.article_id.value_counts().index.tolist()]
    val = tx[tx.t_dat >= cutoff].copy()
    val["aid_str"] = val.article_id.astype(str)
    return pop_ids[:100], val


def day15_16():  # cold-start routing, popular fallback, and unit tests
    tx = pd.read_parquet("data/processed/transactions", columns=["t_dat", "article_id"])
    tx["t_dat"] = pd.to_datetime(tx.t_dat)
    pop_top100, _ = baseline_popular_and_val(tx)

    emb = np.load("artifacts/item_emb.npy")
    item_ids = json.load(open("artifacts/item_ids.json"))

    cold_ids = set()
    art_path = Path("data/processed/articles")
    if art_path.exists():
        art = pd.read_parquet(art_path, columns=["article_id", "is_cold_item"])
        cold_ids = set(art.loc[art.is_cold_item.astype(bool), "article_id"].astype(str))

    router = ItemFallbackRouter(item_ids, emb, pop_top100, cold_ids)
    warm_sample = next((aid for aid in item_ids if aid not in cold_ids), item_ids[0])
    cold_sample = next(iter(cold_ids), "COLD_SAMPLE")

    warm_recs, warm_route = router.recommend(warm_sample, k=5)
    cold_recs, cold_route = router.recommend(cold_sample, k=5)
    unk_recs, unk_route = router.recommend("UNKNOWN_ITEM", k=5)

    report("w3_day15_16_routing_check.txt", [
        f"warm item ({warm_sample}) -> route={warm_route}, top5={warm_recs}",
        f"cold item ({cold_sample}) -> route={cold_route}, top5={cold_recs}",
        f"unknown item (UNKNOWN_ITEM) -> route={unk_route}, top5={unk_recs}",
    ])

    unit_test = '''import unittest
import numpy as np
from week3_item_checks import ItemFallbackRouter


class TestItemFallbackRouter(unittest.TestCase):
    def setUp(self):
        self.ids = ["a1", "a2", "a3", "a4"]
        self.emb = np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [-1.0, 0.0],
        ], dtype=np.float32)
        self.pop = ["p1", "p2", "p3", "a2", "a3"]
        self.cold = {"a4"}
        self.router = ItemFallbackRouter(self.ids, self.emb, self.pop, self.cold)

    def test_warm_item_uses_embeddings(self):
        recs, route = self.router.recommend("a1", k=2)
        self.assertEqual(route, "embedding_similarity")
        self.assertEqual(recs[0], "a2")
        self.assertNotIn("a1", recs)

    def test_cold_item_uses_popular_fallback(self):
        recs, route = self.router.recommend("a4", k=3)
        self.assertEqual(route, "popular_fallback")
        self.assertEqual(recs, ["p1", "p2", "p3"])

    def test_unknown_item_uses_popular_fallback(self):
        recs, route = self.router.recommend("missing_id", k=2)
        self.assertEqual(route, "popular_fallback")
        self.assertEqual(recs, ["p1", "p2"])


if __name__ == "__main__":
    unittest.main()
'''
    (TEST_DIR / "test_item_routing.py").write_text(unit_test, encoding="utf-8")
    print("saved", TEST_DIR / "test_item_routing.py")


def day17_21():  # per-product-category evaluation and results table
    tx = pd.read_parquet("data/processed/transactions", columns=["t_dat", "article_id"])
    a = pd.read_parquet("data/processed/articles", columns=["article_id", "product_type_name"])
    tx["t_dat"] = pd.to_datetime(tx.t_dat)
    pop_top100, val = baseline_popular_and_val(tx)

    item_ids = json.load(open("artifacts/item_ids.json"))
    emb_set = set(item_ids)

    val = val.merge(a, on="article_id", how="left")
    val["product_type_name"] = val["product_type_name"].fillna("unknown")
    val["in_emb"] = val["aid_str"].isin(emb_set)
    val["base_top10"] = val["aid_str"].isin(set(pop_top100[:10]))
    val["base_top100"] = val["aid_str"].isin(set(pop_top100[:100]))

    cat_df = val.groupby("product_type_name").agg(
        val_purchases=("article_id", "count"),
        emb_cov_pct=("in_emb", lambda s: round(s.mean() * 100, 2)),
        base_top10_pct=("base_top10", lambda s: round(s.mean() * 100, 2)),
        base_top100_pct=("base_top100", lambda s: round(s.mean() * 100, 2)),
    ).sort_values("val_purchases", ascending=False)

    report("w3_day17_21_product_type_eval.txt", [
        "Top 15 product types in validation (baseline vs. embedding coverage):",
        cat_df.head(15).to_string()
    ])

    metrics_path = Path("artifacts/metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    rows = []
    for k in (1, 5, 10, 50, 100):
        base_k = val["aid_str"].isin(set(pop_top100[:k])).mean() * 100
        model_k = metrics.get(f"factorized_top_k/top_{k}_categorical_accuracy", np.nan) * 100
        delta_k = model_k - base_k if not np.isnan(model_k) else np.nan
        rows.append((k, base_k, model_k, delta_k))

    md = [
        "# Week 3 Results Table",
        "",
        "## Top-K Comparison: Popularity Baseline vs. Two-Tower Model",
        "",
        "| Top-K | Popularity Baseline (%) | Two-Tower Model (%) | Diff (pp) |",
        "|---:|---:|---:|---:|",
    ]
    for k, base_k, model_k, delta_k in rows:
        m_str = f"{model_k:.2f}" if not np.isnan(model_k) else "N/A"
        d_str = f"{delta_k:+.2f}" if not np.isnan(delta_k) else "N/A"
        md.append(f"| Top-{k} | {base_k:.2f} | {m_str} | {d_str} |")

    md += [
        "",
        "## Top 10 Product Types (Validation Breakdown)",
        "",
        "| Product Type | Val Purchases | Emb Coverage (%) | Baseline Top-10 (%) | Baseline Top-100 (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for ptype, r in cat_df.head(10).iterrows():
        md.append(
            f"| {ptype} | {int(r['val_purchases']):,} | {r['emb_cov_pct']:.2f} | "
            f"{r['base_top10_pct']:.2f} | {r['base_top100_pct']:.2f} |"
        )

    report("results_table.md", md)


STEPS = {15: day15_16, 16: day15_16, 17: day17_21, 18: day17_21, 19: day17_21, 20: day17_21, 21: day17_21}

if __name__ == "__main__":
    if not sys.argv[1:]:
        day15_16()
        day17_21()
    else:
        for arg in sys.argv[1:]:
            s = int(arg)
            if s in STEPS:
                STEPS[s]()
