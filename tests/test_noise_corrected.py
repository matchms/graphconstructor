import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from graphconstructor import Graph
from graphconstructor.operators.noise_corrected import NoiseCorrected


def _csr(data, rows, cols, n):
    return sp.csr_matrix(
        (np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))),
        shape=(n, n),
    )


# ----------------- Undirected: symmetry + monotonicity in delta -----------------
def test_nc_undirected_symmetry_and_monotonicity():
    # 5-node undirected, asymmetric input but Graph symmetrizes (max)
    A = _csr(
        data=[0.9, 0.1, 0.6, 0.4, 0.3, 0.05, 0.8],
        rows=[0,   0,   1,   2,   3,   3,    4],
        cols=[1,   2,   2,   3,   0,   4,    1],
        n=5,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, sym_op="max")

    # Larger delta -> sparser backbone
    G_lo = NoiseCorrected(delta=1.0).apply(G0)
    G_hi = NoiseCorrected(delta=2.5).apply(G0)

    # Symmetry
    assert (G_lo.adj != G_lo.adj.T).nnz == 0
    assert (G_hi.adj != G_hi.adj.T).nnz == 0

    e_lo = set(zip(*sp.triu(G_lo.adj, k=1).nonzero()))
    e_hi = set(zip(*sp.triu(G_hi.adj, k=1).nonzero()))
    assert e_hi.issubset(e_lo)


# ----------------- Directed: behaves and monotone in delta -----------------
def test_nc_directed_monotonicity_and_no_negatives():
    A = _csr(
        data=[3.0, 1.0, 4.0, 2.0, 0.5],
        rows=[0,   0,   1,   2,   3],
        cols=[1,   2,   2,   3,   0],
        n=4,
    )
    G0 = Graph.from_csr(A, directed=True, weighted=True)

    G1 = NoiseCorrected(delta=1.0).apply(G0)
    G2 = NoiseCorrected(delta=2.0).apply(G0)

    e1 = set(zip(*G1.adj.nonzero()))
    e2 = set(zip(*G2.adj.nonzero()))
    assert e2.issubset(e1)

    # No NaNs/Inf
    assert np.isfinite(G1.adj.data).all()
    assert np.isfinite(G2.adj.data).all()


# ----------------- Negative weights rejected -----------------
def test_nc_rejects_negative_weights():
    A = _csr([-0.1, 0.4], [0, 1], [1, 0], 2)
    G0 = Graph.from_csr(A, directed=True, weighted=True, sym_op="max")
    with pytest.raises(ValueError, match="nonnegative"):
        NoiseCorrected().apply(G0)


# ----------------- All-zero graph: noop -----------------
def test_nc_all_zero_noop():
    A = sp.csr_matrix((3, 3), dtype=float)
    G0 = Graph.from_csr(A, directed=False, weighted=True, sym_op="max")
    G = NoiseCorrected().apply(G0)
    assert G.adj.nnz == 0 and G.adj.shape == (3, 3)


# ----------------- Metadata & flags preserved -----------------
def test_nc_preserves_flags_and_copies_metadata():
    meta = pd.DataFrame({"name": ["a", "b", "c"], "grp": [1, 0, 1]})
    A = _csr([0.5, 0.3, 0.7], [0, 1, 2], [1, 2, 0], 3)
    G0 = Graph.from_csr(A, directed=False, weighted=True, meta=meta, sym_op="max")

    out = NoiseCorrected(delta=1.64, copy_meta=True).apply(G0)
    assert not out.directed and out.weighted
    if out.meta is not None:
        meta.loc[0, "grp"] = 999
        assert out.meta.loc[0, "grp"] == 1
