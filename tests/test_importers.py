import numpy as np
import pytest
import scipy.sparse as sp
from graphconstructor.importers import (
    from_ann,
    from_csr,
    from_dense,
    from_knn,
    from_pairwise,
)


def _csr(data, rows, cols, n):
    return sp.csr_matrix(
        (np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))),
        shape=(n, n),
    )


# ----------------- from_csr / from_dense -----------------
def test_from_csr_and_from_dense_basic_flags_and_symmetry():
    A = _csr([0.4, 0.9], [0, 1], [1, 0], 2)
    G1 = from_csr(A, directed=False, weighted=True)
    assert not G1.directed and G1.weighted
    assert (G1.adj != G1.adj.T).nnz == 0

    D = np.array([[0.0, 0.4], [0.9, 0.0]])
    G2 = from_dense(D, directed=False, weighted=True)
    assert (G1.adj != G2.adj).nnz == 0


# ----------------- from_knn -----------------
def test_from_knn_store_weights_true_and_false():
    ind = np.array([[1, 2], [0, 2], [0, 1]])
    dist = np.array([[0.2, 0.7], [0.3, 0.9], [0.4, 0.5]])

    Gw = from_knn(ind, dist, store_weights=True, directed=True)
    assert Gw.weighted
    assert pytest.approx(Gw.adj[0, 1]) == 0.2
    assert pytest.approx(Gw.adj[1, 0]) == 0.3

    Gw = from_knn(ind, dist, store_weights=True, directed=False)
    assert Gw.weighted
    assert pytest.approx(Gw.adj[0, 1]) == 0.3

    Gu = from_knn(ind, dist, store_weights=False, directed=False)
    assert not Gu.weighted
    assert np.allclose(Gu.adj.data, 1.0)


# ----------------- from_ann -----------------
class _PrecomputedIndex:
    def __init__(self, indices, distances):
        self.indices_ = indices
        self.distances_ = distances


class _QueryIndex:
    def __init__(self, indices, distances):
        self._ind = np.asarray(indices)
        self._dist = np.asarray(distances)

    def query(self, X, k: int):
        return self._ind[:, :k], self._dist[:, :k]


def test_from_ann_uses_cached_neighbors():
    ind = np.array([[1, 2], [0, 2], [0, 1]])
    dist = np.array([[0.2, 0.7], [0.3, 0.9], [0.4, 0.5]])
    ann = type("ANN", (), {"index": _PrecomputedIndex(ind, dist)})

    # directed
    G = from_ann(ann, query_data=None, k=2, store_weights=True, directed=True)
    assert pytest.approx(G.adj[0, 1]) == 0.2
    assert G.adj.shape == (3, 3)
    # undirected version
    G = from_ann(ann, query_data=None, k=2, store_weights=True, directed=False)
    assert pytest.approx(G.adj[0, 1]) == 0.3
    assert G.adj.shape == (3, 3)


def test_from_ann_queries_index_when_no_cache():
    ind = np.array([[2, 0, 1], [0, 1, 2]])
    dist = np.array([[0.9, 0.1, 0.2], [0.3, 0.4, 0.5]])
    ann = type("ANN", (), {"index": _QueryIndex(ind, dist)})

    G = from_ann(ann, query_data=np.zeros((2, 3)), k=2, store_weights=True)
    assert G.adj.shape == (3, 3)
    # structure sanity: at least some edges present
    assert G.adj.nnz > 0


# ----------------- from_pairwise -----------------
def test_from_pairwise_knn_strategy_distance_mode():
    # Small distance matrix; k=1 keeps the smallest off-diagonal per row
    D = np.array([
        [0.0, 0.2, 0.8, 0.3],
        [0.2, 0.0, 0.6, 0.4],
        [0.8, 0.6, 0.0, 0.5],
        [0.3, 0.4, 0.5, 0.0],
    ])
    G = from_pairwise(D, strategy=("knn", 1), mode="distance", directed=False, store_weights=True)
    # After sym 'max' on a symmetric input, weights match the distances
    assert pytest.approx(G.adj[0, 1]) == 0.2
    assert pytest.approx(G.adj[0, 3]) == 0.3


def test_from_pairwise_epsilon_strategy_similarity_mode_and_store_weights_false():
    S = np.array([
        [1.0, 0.8, 0.6],
        [0.7, 1.0, 0.9],
        [0.4, 0.85, 1.0],
    ])
    G = from_pairwise(S, strategy=("epsilon", 0.75), mode="similarity",
                      directed=False, store_weights=False)
    # edges above 0.75 kept with unit weights; sym 'max' mirrors presence
    assert G.adj[0, 1] == 1.0
    assert G.adj[1, 2] == 1.0
    assert G.adj[0, 2] == 0.0


# ----------------- flags and metadata passthrough -----------------
def test_importers_preserve_directed_flag_and_allow_metadata():
    D = np.array([
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.7],
        [0.0, 0.0, 0.0],
    ])
    meta = None  # can replace with a DataFrame; here we just test directed flag
    G = from_dense(D, directed=True, weighted=True, meta=meta)
    assert G.directed and G.weighted
