import numpy as np

from src.retrievers.dense_index import VectorIndex


def test_saved_index_matches_direct_ranking(tmp_path):
    vectors = np.array([[1, 0], [0, 2], [-1, 1]], dtype=np.float32)
    queries = np.array([[3, 1], [-1, 4]], dtype=np.float32)
    expected = np.argsort(-(queries @ vectors.T), axis=1)
    ids = ["first", "second", "third"]
    index = VectorIndex.build(vectors, ids, "checkpoint")
    index.save(tmp_path)
    reloaded = VectorIndex.load(tmp_path, "checkpoint")
    np.testing.assert_array_equal(index.search(queries, 3), expected)
    np.testing.assert_array_equal(reloaded.search(queries, 10), expected)
    assert [[reloaded.passage_ids[i] for i in row] for row in expected] == [
        ["first", "second", "third"], ["second", "third", "first"],
    ]
