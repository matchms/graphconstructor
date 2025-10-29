import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from graphconstructor import Graph
from graphconstructor.operators import DoublyStochastic


def _csr(data, rows, cols, n):
    return sp.csr_matrix(
        (np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))),
        shape=(n, n),
    )


# ----------------- Positive dense matrix: converges to ~doubly stochastic -----------------
def test_doubly_stochastic_converges_on_positive_dense():
    # Strictly positive, symmetric 4x4 (undirected)
    M = np.array([
        [0.2, 0.8, 0.5, 0.3],
        [0.7, 0.1, 0.4, 0.6],
        [0.3, 0.9, 0.2, 0.5],
        [0.5, 0.2, 0.7, 0.4],
    ], dtype=float)
    # Zero the diagonal (typical adjacency semantics)
    np.fill_diagonal(M, 0.0)

    G0 = Graph.from_dense(M, directed=False, weighted=True, mode="similarity", sym_op="max")
    op = DoublyStochastic(tolerance=1e-6, max_iter=10_000)
    G = op.apply(G0)

    A = G.adj
    # No NaNs or infs
    assert np.isfinite(A.data).all()

    # All rows/cols have edges, so sums should be ~1
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    col_sums = np.asarray(A.sum(axis=0)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert np.allclose(col_sums, 1.0, atol=1e-6)

    # Flags preserved
    assert not G.directed and G.weighted


# ----------------- Sparse graph with isolates: zero rows/cols remain zero, others ~1 -----------------
def test_doubly_stochastic_sparse_with_isolates():
    # 5 nodes, node 4 is isolated
    A = _csr(
        data=[0.4, 0.6, 0.3, 0.7, 0.2, 0.5],
        rows=[0, 0, 1, 1, 2, 3],
        cols=[1, 2, 2, 3, 3, 2],
        n=5,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="similarity", sym_op="max")

    op = DoublyStochastic(tolerance=1e-6, max_iter=10_000)
    G = op.apply(G0)
    A2 = G.adj

    # No NaNs or infs used in storage
    assert np.isfinite(A2.data).all()

    row_sums = np.asarray(A2.sum(axis=1)).ravel()
    col_sums = np.asarray(A2.sum(axis=0)).ravel()

    # Indices with edges
    rows_with = (np.diff(A2.indptr) > 0)
    cols_with = (sp.csc_matrix(A2).indptr[1:] - sp.csc_matrix(A2).indptr[:-1]) > 0

    # Non-isolated rows/cols sum ~1
    if rows_with.any():
        assert np.allclose(row_sums[rows_with], 1.0, atol=1e-4)
    if cols_with.any():
        assert np.allclose(col_sums[cols_with], 1.0, atol=1e-4)  # more would be possible but needs higher bit floats

    # Isolated node (4) stays isolated (sum == 0)
    assert row_sums[4] == 0.0 and col_sums[4] == 0.0

    # Flags preserved
    assert not G.directed and G.weighted


# ----------------- Directed case: rows and cols ~1 for nonzero rows/cols -----------------
def test_doubly_stochastic_directed_graph_unsolvable():
    # Directed 4x4 with zeros on diagonal, not symmetric
    # This graph cannot be made doubly stochastic exactly!!!
    A = _csr(
        data=[0.5, 0.9, 0.4, 0.6, 0.3, 0.7],
        rows=[0, 0, 1, 2, 2, 3],
        cols=[1, 2, 2, 0, 3, 1],
        n=4,
    )
    G0 = Graph.from_csr(A, directed=True, weighted=True, mode="similarity")

    op = DoublyStochastic(tolerance=1e-6, max_iter=10_000)
    G = op.apply(G0)
    A2 = G.adj

    # No NaNs or infs
    assert np.isfinite(A2.data).all()

    expected_result = np.array([
        [0. , 0.5, 0.5, 0. ],
        [0. , 0. , 1. , 0. ],
        [0.5, 0. , 0. , 0.5],
        [0. , 1. , 0. , 0. ]
        ])
    assert np.allclose(A2.toarray(), expected_result, atol=1e-4)

    # Directed flag preserved
    assert G.directed and G.weighted


# ----------------- Error cases -----------------
def test_doubly_stochastic_rejects_negative_weights():
    A = _csr([-0.2, 0.5], [0, 1], [1, 0], 2)
    G0 = Graph.from_csr(A, directed=True, weighted=True, mode="similarity", sym_op="max")
    op = DoublyStochastic()
    with pytest.raises(ValueError, match="nonnegative"):
        op.apply(G0)


def test_doubly_stochastic_rejects_distances():
    A = _csr([0.2, 0.5], [0, 1], [1, 0], 2)
    G0 = Graph.from_csr(A, directed=True, weighted=True, mode="distance", sym_op="max")
    op = DoublyStochastic()
    with pytest.raises(ValueError, match="only supports modes"):
        op.apply(G0)


# ----------------- Trivial all-zero matrix: returned unchanged -----------------
def test_doubly_stochastic_all_zero_matrix_noop():
    A = sp.csr_matrix((4, 4), dtype=float)
    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="similarity", sym_op="max")
    op = DoublyStochastic()
    G = op.apply(G0)
    assert G.adj.nnz == 0
    assert G.adj.shape == (4, 4)
    assert not G.directed and G.weighted


# ----------------- Metadata preservation (copy) -----------------
def test_doubly_stochastic_preserves_flags_and_copies_metadata():
    meta = pd.DataFrame({"name": ["a", "b", "c"], "group": [1, 0, 1]})
    A = _csr([0.4, 0.6, 0.3], [0, 1, 2], [1, 2, 0], 3)
    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="similarity", meta=meta, sym_op="max")

    op = DoublyStochastic(tolerance=1e-6, max_iter=10_000, copy_meta=True)
    G = op.apply(G0)

    # Flags
    assert not G.directed and G.weighted

    # Metadata copied: changing original shouldn't affect result
    if G.meta is not None:
        meta.loc[0, "group"] = 999
        assert G.meta.loc[0, "group"] == 1
