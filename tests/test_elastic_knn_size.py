"""The vector search returns k hits, not Elasticsearch's default 10.

A top-level `knn` search returns `size` hits, and `size` defaults to 10. The
QA pipeline asks each vector leg for 30 candidates, so without an explicit
size every semantic leg silently saw a third of what it asked for.
"""

import unittest
from unittest.mock import patch

from backend.elastic import ELASTIC_CLIENT


class _FakeES:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        size = kwargs.get("size", 10)
        k = kwargs["knn"]["k"]
        return {"hits": {"hits": [{"_id": str(i), "_score": 1.0, "_source": {}} for i in range(min(size, k))]}}


class KnnSizeTests(unittest.TestCase):
    def test_knn_asks_for_k_hits(self):
        fake = _FakeES()
        with patch.object(type(ELASTIC_CLIENT), "client", new=fake):
            hits = ELASTIC_CLIENT.knn_search("articles", [0.1, 0.2], k=30, num_candidates=200)
        self.assertEqual(fake.calls[0]["size"], 30)
        self.assertEqual(len(hits), 30)


if __name__ == "__main__":
    unittest.main()
