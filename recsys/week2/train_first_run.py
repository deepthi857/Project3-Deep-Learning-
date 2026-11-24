import argparse
import json
import tensorflow as tf

from training.model import UserTower, ItemTower
from recsys.week2.two_tower import TwoTower
from recsys.week2.data_pipeline import load_all
from recsys.week2.export_embeddings import export


def run_training(
    epochs=1,
    dim=64,
    lr=1e-3,
    batch=256,
    train_rows=None,
    val_rows=50000,
    do_export=False,
):
    with open("artifacts/vocab.json") as f:
        vocab = json.load(f)

    train_ds, val_ds, items_ds, items_df = load_all(
        batch=batch,
        val_rows=val_rows,
        train_rows=train_rows,
    )

    user_model = UserTower(vocab, dim)
    item_model = ItemTower(vocab, dim)

    model = TwoTower(user_model, item_model, items_ds)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    metrics = {
        key: float(value)
        for key, value in model.evaluate(
            val_ds,
            return_dict=True,
        ).items()
    }

    metrics_file = f"artifacts/metrics_dim{dim}.json"

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDimension {dim} results:")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_file}")

    if do_export:
        export(user_model, item_model, items_ds, items_df)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--train-rows", type=int, default=None)
    parser.add_argument("--val-rows", type=int, default=50000)
    parser.add_argument("--export", action="store_true")

    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        dim=args.dim,
        lr=args.lr,
        batch=args.batch,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        do_export=args.export,
    )
