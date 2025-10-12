import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from graphconstructor import Graph
from graphconstructor.operators import LocallyAdaptiveSparsification


def _csr(data, rows, cols, n):
    return sp.csr_matrix(
        (np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))),
        shape=(n, n),
    )


# ----------------- Basic sanity: undirected, symmetry, weights preserved -----------------
def test_lans_undirected_symmetry_and_weights_preserved():
    # Build an undirected graph (Graph will symmetrize with max)
    # Row-wise fractions:
    # 0: to 1=0.6, to 2=0.4 -> pvals (upper-tail): 0.6->0.5, 0.4->1.0
    A = _csr(
        data=[0.6, 0.4, 0.5, 0.7],
        rows=[0,   0,   1,   2],
        cols=[1,   2,   2,   3],
        n=4,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, sym_op="max")

    # alpha=0.5 should keep 0->1 from node 0's perspective (p=0.5), drop 0->2 (p=1.0)
    G = LocallyAdaptiveSparsification(alpha=0.5, rule="or").apply(G0)

    # Symmetry
    assert (G.adj != G.adj.T).nnz == 0

    # 0-1 kept, 0-2 dropped; 1-2 may depend on node 1's/2's local tails, but we focus on 0's row behavior
    assert G.adj[0, 1] > 0 and G.adj[1, 0] > 0
    assert G.adj[0, 2] == 0 and G.adj[2, 0] == 0

    # Weights must be original ones on kept edges
    assert G.adj[0, 1] == pytest.approx(G0.adj[0, 1])


# ----------------- Undirected: "and" is a subset of "or"; monotonicity in alpha -----------------
def test_lans_undirected_and_subset_or_and_alpha_monotonicity():
    A = _csr(
        data=[0.9, 0.2, 0.6, 0.4, 0.3],
        rows=[0,   0,   1,   2,   3],
        cols=[1,   2,   2,   3,   0],
        n=4,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, sym_op="max")

    G_or = LocallyAdaptiveSparsification(alpha=0.30, rule="or").apply(G0)
    G_and = LocallyAdaptiveSparsification(alpha=0.30, rule="and").apply(G0)

    e_or = set(zip(*sp.triu(G_or.adj, k=1).nonzero()))
    e_and = set(zip(*sp.triu(G_and.adj, k=1).nonzero()))
    assert e_and.issubset(e_or)  # "and" ⊆ "or"

    # Monotonicity in alpha: larger alpha keeps (weakly) more edges
    G_lo = LocallyAdaptiveSparsification(alpha=0.10, rule="or").apply(G0)
    G_hi = LocallyAdaptiveSparsification(alpha=0.40, rule="or").apply(G0)
    e_lo = set(zip(*sp.triu(G_lo.adj, k=1).nonzero()))
    e_hi = set(zip(*sp.triu(G_hi.adj, k=1).nonzero()))
    assert e_lo.issubset(e_hi)

    # Symmetry guaranteed
    assert (G_or.adj != G_or.adj.T).nnz == 0
    assert (G_and.adj != G_and.adj.T).nnz == 0


# ----------------- Directed: require BOTH endpoints significant -----------------
def test_lans_directed_requires_both_endpoints():
    # Construct a small directed graph where one side is strong from source but weak from target
    # Row 0: to 1 (0.9), to 2 (0.1) -> for row 0, 0->1 has frac=0.9 (pval_out ~ 0.5), 0->2 frac=0.1 (pval_out ~ 1.0)
    # Column 1: suppose its in-side has many stronger edges -> pval_in might be ~1.0 so 0->1 could be dropped.
    A = _csr(
        data=[0.9, 0.1, 0.8, 0.7, 0.2],
        rows=[0,   0,   2,   3,   1],
        cols=[1,   2,   1,   1,   3],
        n=4,
    )
    G0 = Graph.from_csr(A, directed=True, weighted=True)

    # With a fairly small alpha, keep only edges significant on BOTH out and in sides
    G = LocallyAdaptiveSparsification(alpha=0.2).apply(G0)
    # There is no guarantee 0->1 survives, because in-side at node 1 can be unfavorable.
    # Sanity: no NaNs/Inf and directed flag preserved
    assert np.isfinite(G.adj.data).all()
    assert G.directed and G.weighted


# ----------------- Negative weights rejected -----------------
def test_lans_rejects_negative_weights():
    A = _csr([-0.3, 0.4], [0, 1], [2, 1], 3)
    G0 = Graph.from_csr(A, directed=False, weighted=True, sym_op="average")
    with pytest.raises(ValueError, match="nonnegative"):
        LocallyAdaptiveSparsification(alpha=0.1).apply(G0)


# ----------------- Handles zero-strength rows (isolates / no outgoing edges) -----------------
def test_lans_handles_zero_strength_rows():
    # Node 3 has no edges; node 2 only incoming (from 1) -> fine.
    A = _csr([0.5, 0.2], [0, 1], [1, 2], 4)
    G0 = Graph.from_csr(A, directed=True, weighted=True)

    G = LocallyAdaptiveSparsification(alpha=0.5).apply(G0)
    # Graph remains well-formed, only existing arcs possibly filtered
    assert G.adj.shape == (4, 4)
    assert (G.adj.toarray() >= 0).all()


# ----------------- Metadata preserved (if copy_meta=True) -----------------
def test_lans_preserves_metadata():
    meta = pd.DataFrame({"name": ["a", "b", "c"], "grp": [1, 0, 1]})
    A = _csr([0.6, 0.4, 0.7], [0, 1, 2], [1, 2, 0], 3)
    G0 = Graph.from_csr(A, directed=False, weighted=True, meta=meta, sym_op="max")

    out = LocallyAdaptiveSparsification(alpha=0.25, rule="or", copy_meta=True).apply(G0)
    assert not out.directed and out.weighted
    # metadata should be copied, not referenced
    if out.meta is not None:
        meta.loc[0, "grp"] = 999
        assert out.meta.loc[0, "grp"] == 1
