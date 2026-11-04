"""Week 2: two-tower model (TensorFlow Recommenders)."""
import tensorflow as tf
import tensorflow_recommenders as tfrs

DIM = 64
USER_CATS = ["age_bucket", "club_member_status", "fashion_news_frequency", "dow", "month"]
ITEM_CATS = ["product_type_name", "colour_group_name", "department_name", "garment_group_name"]


def lookup_embed(vocab, dim):
    return tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None),
        tf.keras.layers.Embedding(len(vocab) + 1, dim)])


def mlp(dim):
    return tf.keras.Sequential([tf.keras.layers.Dense(128, activation="relu"),
                                tf.keras.layers.Dense(dim)])


class UserTower(tf.keras.Model):
    def __init__(self, v, dim=DIM):
        super().__init__()
        self.uid = lookup_embed(v["customer_id"], dim)
        self.cats = {c: lookup_embed(v[c], 16) for c in USER_CATS}
        self.hist_lookup = tf.keras.layers.StringLookup(vocabulary=v["article_id"], mask_token=None)
        self.hist_emb = tf.keras.layers.Embedding(len(v["article_id"]) + 1, dim)
        self.mlp = mlp(dim)

    def call(self, f):
        parts = [self.uid(f["customer_id"])]
        parts += [self.cats[c](f[c]) for c in USER_CATS]
        hist = self.hist_emb(self.hist_lookup(tf.strings.split(f["history"])))  # [B, n, dim]
        parts.append(tf.reduce_mean(hist, axis=1))              # short-term intent
        parts.append(tf.expand_dims(f["user_recency"], 1))
        return self.mlp(tf.concat(parts, axis=1))


class ItemTower(tf.keras.Model):
    def __init__(self, v, dim=DIM):
        super().__init__()
        self.aid = lookup_embed(v["article_id"], dim)
        self.cats = {c: lookup_embed(v[c], 16) for c in ITEM_CATS}
        self.mlp = mlp(dim)

    def call(self, f):
        parts = [self.aid(f["article_id"])] + [self.cats[c](f[c]) for c in ITEM_CATS]
        parts.append(tf.expand_dims(f["item_pop_log"], 1))
        return self.mlp(tf.concat(parts, axis=1))


class TwoTower(tfrs.Model):
    def __init__(self, user_model, item_model, candidates_ds):
        super().__init__()
        self.user_model, self.item_model = user_model, item_model
        self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
            candidates=candidates_ds.batch(1024).map(item_model)))

    def compute_loss(self, features, training=False):
        u, i = self.user_model(features), self.item_model(features)
        return self.task(u, i, compute_metrics=not training)
