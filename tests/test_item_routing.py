import unittest
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
