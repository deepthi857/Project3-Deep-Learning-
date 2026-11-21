import tensorflow as tf
import tensorflow_recommenders as tfrs


def l2(x):
    return tf.math.l2_normalize(x, axis=1)


class TwoTower(tfrs.Model):
    def __init__(self, user_model, item_model, candidates_ds, temperature=0.05):
        super().__init__()
        self.user_model, self.item_model = user_model, item_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates_ds.batch(1024).map(lambda x: l2(item_model(x)))),
            temperature=temperature)

    def compute_loss(self, features, training=False):
        u = l2(self.user_model(features))
        i = l2(self.item_model(features))
        return self.task(u, i, compute_metrics=not training)

