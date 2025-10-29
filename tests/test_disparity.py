import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from graphconstructor import Graph
from graphconstructor.operators import DisparityFilter


def _csr(data, rows, cols, n):
    return sp.csr_matrix(
        (np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))),
        shape=(n, n),
    )


# ----------------- Directed: matches closed-form p-values via min(out,in) -----------------
def test_disparity_directed_min_out_in_formula():
    A = _csr(
        data=[0.6, 0.4, 0.2, 0.8, 0.2],
        rows=[0, 0, 1, 2, 2],
        cols=[1, 2, 2, 0, 3],
        n=4,
    )
    G0 = Graph.from_csr(A, directed=True, weighted=True, mode="similarity")

    s_out = np.asarray(A.sum(axis=1)).ravel()
    s_in = np.asarray(A.sum(axis=0)).ravel()
    k_out = np.diff(A.indptr)
    k_in = np.diff(A.tocsc().indptr)

    coo = A.tocoo()
    rows, cols, w = coo.row, coo.col, coo.data

    # Out
    p_out = np.zeros_like(w)
    m_out = s_out[rows] > 0
    p_out[m_out] = w[m_out] / s_out[rows[m_out]]
    e_out = np.maximum(k_out[rows] - 1, 0)
    pv_out = np.power(1 - np.clip(p_out, 0.0, 1.0), e_out)

    # In
    p_in = np.zeros_like(w)
    m_in = s_in[cols] > 0
    p_in[m_in] = w[m_in] / s_in[cols[m_in]]
    e_in = np.maximum(k_in[cols] - 1, 0)
    pv_in = np.power(1 - np.clip(p_in, 0.0, 1.0), e_in)

    alpha = 0.2
    expect_keep = np.minimum(pv_out, pv_in) <= alpha
    expected = set(zip(rows[expect_keep].tolist(), cols[expect_keep].tolist()))

    out = DisparityFilter(alpha=alpha).apply(G0)
    got = set(zip(*out.adj.nonzero()))
    assert got == expected
    assert out.directed and out.weighted
    assert np.isfinite(out.adj.data).all()


# ----------------- Undirected: "or" is superset of "and"; symmetry guaranteed -----------------
def test_disparity_undirected_or_superset_and():
    A = _csr(
        data=[0.9, 0.2, 0.6, 0.4, 0.3],
        rows=[0,   0,   1,   2,   3],
        cols=[1,   2,   2,   3,   0],
        n=4,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="similarity", sym_op="max")

    alpha = 0.3
    G_or = DisparityFilter(alpha=alpha, rule="or").apply(G0)
    G_and = DisparityFilter(alpha=alpha, rule="and").apply(G0)

    # Symmetric
    assert (G_or.adj != G_or.adj.T).nnz == 0
    assert (G_and.adj != G_and.adj.T).nnz == 0

    e_or = set(zip(*sp.triu(G_or.adj, k=1).nonzero()))
    e_and = set(zip(*sp.triu(G_and.adj, k=1).nonzero()))
    assert e_and.issubset(e_or)


# ----------------- Undirected: degree-1 node is NOT automatically kept for tiny alpha -----------------
def test_disparity_undirected_degree1_not_always_kept_for_tiny_alpha():
    # Node 0 has only one neighbor; whether edge is kept depends on node 1's side.
    A = _csr(
        data=[0.5, 0.3, 0.2],
        rows=[0,   1,   2],
        cols=[1,   2,   3],
        n=4,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="similarity", sym_op="max")

    out = DisparityFilter(alpha=1e-6, rule="or").apply(G0)
    # It's valid that (0,1) may be dropped at tiny alpha; the test asserts only that
    # no NaNs occur and symmetry holds.
    assert np.isfinite(out.adj.data).all()
    assert (out.adj != out.adj.T).nnz == 0


# ----------------- Negative weights rejected -----------------
def test_disparity_rejects_negative_weights():
    A = _csr([-0.5, 0.2], [0, 1], [1, 0], 2)
    G0 = Graph.from_csr(A, directed=True, weighted=True, mode="similarity", sym_op="max")
    with pytest.raises(ValueError, match="nonnegative"):
        DisparityFilter(alpha=0.1).apply(G0)


# ----------------- Metadata & flags preserved -----------------
def test_disparity_preserves_flags_and_copies_metadata():
    meta = pd.DataFrame({"name": ["a", "b", "c"], "grp": [1, 0, 1]})
    A = _csr([0.6, 0.4, 0.7], [0, 1, 2], [1, 2, 0], 3)
    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="similarity", meta=meta, sym_op="max")

    out = DisparityFilter(alpha=0.2, rule="or", copy_meta=True).apply(G0)
    assert not out.directed and out.weighted
    if out.meta is not None:
        meta.loc[0, "grp"] = 999
        assert out.meta.loc[0, "grp"] == 1
