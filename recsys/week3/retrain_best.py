"""Week 3 Day 18-20: small grid search, then final retrain + export.
Run: python -m recsys.week3.retrain_best
"""
import itertools
import json
from recsys.week2.train_first_run import run_training

GRID = {"dim": [32, 64], "lr": [1e-3, 3e-3]}
results = []
for dim, lr in itertools.product(GRID["dim"], GRID["lr"]):
    m = run_training(epochs=1, dim=dim, lr=lr, train_rows=200_000, do_export=False)
    key = [k for k in m if "top_100" in k and not k.startswith("val")]
    score = m[key[0]] if key else 0.0
    results.append({"dim": dim, "lr": lr, "top100": score})
    print("grid:", results[-1])

best = max(results, key=lambda r: r["top100"])
json.dump({"grid": results, "best": best}, open("artifacts/tuning.json", "w"), indent=2)
print("BEST:", best)
run_training(epochs=3, dim=best["dim"], lr=best["lr"], do_export=True)   # final model

