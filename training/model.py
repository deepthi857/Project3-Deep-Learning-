
"""Week 2: Context-aware two-tower recommendation model."""

import tensorflow as tf
import tensorflow_recommenders as tfrs

DIM = 64

USER_CATS = [
    "age_bucket",
    "club_member_status",
    "fashion_news_frequency",
    "dow",
    "month",
]

ITEM_CATS = [
    "product_type_name",
    "colour_group_name",
    "department_name",
    "garment_group_name",
]


def lookup_embed(vocab, dim):
    return tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=vocab,
            mask_token=None,
            num_oov_indices=1,
        ),
        tf.keras.layers.Embedding(
            input_dim=len(vocab) + 1,
            output_dim=dim,
        ),
    ])


def mlp(dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(dim),
    ])


class UserTower(tf.keras.Model):
    """Creates user embeddings from identity, demographics and history."""

    def __init__(self, v, dim=DIM):
        super().__init__()

        self.uid = lookup_embed(v["customer_id"], dim)

        self.cats = {
            c: lookup_embed(v[c], 16)
            for c in USER_CATS
        }

        self.hist_lookup = tf.keras.layers.StringLookup(
            vocabulary=v["article_id"],
            mask_token=None,
            num_oov_indices=1,
        )

        self.hist_emb = tf.keras.layers.Embedding(
            input_dim=len(v["article_id"]) + 1,
            output_dim=dim,
        )

        self.mlp = mlp(dim)

    def call(self, f):
        parts = [self.uid(f["customer_id"])]

        parts += [
            self.cats[c](f[c])
            for c in USER_CATS
        ]

        # Split purchase history into article IDs.
        tokens = tf.strings.split(f["history"])

        # Ignore the PAD placeholder.
        mask = tf.not_equal(tokens, "PAD")

        hist_ids = self.hist_lookup(tokens)
        hist = self.hist_emb(hist_ids)

        # Masked mean pooling.
        mask_float = tf.cast(mask, hist.dtype)
        mask_float = tf.expand_dims(mask_float, axis=-1)

        hist_sum = tf.reduce_sum(
            hist * mask_float,
            axis=1,
        )

        hist_count = tf.reduce_sum(
            mask_float,
            axis=1,
        )

        # Empty histories safely produce zero vectors.
        hist_mean = tf.math.divide_no_nan(
            hist_sum,
            hist_count,
        )

        parts.append(hist_mean)

        # Recency is already log-transformed by Spark.
        parts.append(
            tf.expand_dims(f["user_recency"], axis=1)
        )

        combined = tf.concat(parts, axis=1)

        return self.mlp(combined)


class ItemTower(tf.keras.Model):
    """Creates item embeddings from article identity and attributes."""

    def __init__(self, v, dim=DIM):
        super().__init__()

        self.aid = lookup_embed(v["article_id"], dim)

        self.cats = {
            c: lookup_embed(v[c], 16)
            for c in ITEM_CATS
        }

        self.mlp = mlp(dim)

    def call(self, f):
        parts = [self.aid(f["article_id"])]

        parts += [
            self.cats[c](f[c])
            for c in ITEM_CATS
        ]

        parts.append(
            tf.expand_dims(f["item_pop_log"], axis=1)
        )

        combined = tf.concat(parts, axis=1)

        return self.mlp(combined)


class TwoTower(tfrs.Model):
    """Two-tower retrieval model."""

    def __init__(self, user_model, item_model, candidates_ds):
        super().__init__()

        self.user_model = user_model
        self.item_model = item_model

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates_ds.batch(1024).map(
                    lambda x: tf.math.l2_normalize(
                        item_model(x),
                        axis=1,
                    )
                )
            )
        )

    def compute_loss(self, features, training=False):
        users = tf.math.l2_normalize(
            self.user_model(features),
            axis=1,
        )

        items = tf.math.l2_normalize(
            self.item_model(features),
            axis=1,
        )

        return self.task(
            users,
            items,
            compute_metrics=not training,
        )
