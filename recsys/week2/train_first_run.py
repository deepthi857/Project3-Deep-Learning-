import argparse
import json
import tensorflow as tf
from training.model import UserTower, ItemTower
from recsys.week2.two_tower import TwoTower
from recsys.week2.data_pipeline import load_all
from recsys.week2.export_embeddings import export


def run_training(epochs=3, dim=64, lr=1e-3, batch=1024, train_rows=None, do_export=True):
    V = json.load(open("artifacts/vocab.json"))
    train_ds, val_ds, items_ds, items_df = load_all(batch, train_rows=train_rows)
    user_model, item_model = UserTower(V, dim), ItemTower(V, dim)
    model = TwoTower(user_model, item_model, items_ds)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr))
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    metrics = {k: float(v) for k, v in model.evaluate(val_ds, return_dict=True).items()}
    json.dump(metrics, open("artifacts/metrics.json", "w"), indent=2)
    print(metrics)
    if do_export:
        export(user_model, item_model, items_ds, items_df)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=1024)
    a = p.parse_args()
    run_training(a.epochs, a.dim, a.lr, a.batch)

