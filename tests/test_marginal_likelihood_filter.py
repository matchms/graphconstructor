import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import binom
from graphconstructor import Graph
from graphconstructor.operators import MarginalLikelihoodFilter


def _csr(data, rows, cols, n):
    return sp.csr_matrix(
        (np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))),
        shape=(n, n),
    )


# ----------------- Undirected: matches binomial tail exactly -----------------
def test_mlf_undirected_matches_binomial_tail():
    """
    Build a small undirected graph, compute exact binomial-tail p-values per edge,
    and assert the operator keeps exactly the edges with p <= alpha.
    """
    # Asymmetric input; Graph.from_csr(sym_op="max") will symmetrize.
    # Intended symmetric weights after sym: (0,1)=5, (0,2)=1, (1,2)=2, (2,3)=8
    A = _csr(
        data=[5, 1, 2, 8],
        rows=[0, 0, 1, 2],
        cols=[1, 2, 2, 3],
        n=4,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, sym_op="max")

    # Compute strengths and T from the symmetrized adjacency
    A_sym = G0.adj  # already symmetric, no self-loops
    k = np.asarray(A_sym.sum(axis=1)).ravel()
    T = 0.5 * float(k.sum())

    # Upper triangle COO to compute p-values for each undirected edge once
    Au = sp.triu(A_sym, k=1).tocoo()
    w = np.rint(Au.data).astype(int)
    ki = k[Au.row]
    kj = k[Au.col]
    p = (ki * kj) / (2.0 * (T ** 2))
    p = np.clip(p, 0.0, 1.0)
    n_trials = int(round(T))
    pvals = binom.sf(w - 1, n=n_trials, p=p)

    alpha = 0.05
    keep = pvals <= alpha
    expected_edges = set(zip(Au.row[keep].tolist(), Au.col[keep].tolist()))

    # Run operator
    out = MarginalLikelihoodFilter(alpha=alpha).apply(G0)
    A2 = out.adj
    # Extract kept undirected edges from result (upper triangle)
    kept_u = sp.triu(A2, k=1).tocoo()
    got_edges = set(zip(kept_u.row.tolist(), kept_u.col.tolist()))

    assert got_edges == expected_edges
    # Symmetry and flags preserved
    assert (A2 != A2.T).nnz == 0
    assert not out.directed and out.weighted


# ----------------- Directed: uses kout x kin -----------------
def test_mlf_directed_uses_out_in_degrees():
    """
    Build a directed graph where row sums != col sums to ensure we truly use kout x kin.
    Check exact binomial tail and the set of retained arcs.
    """
    # Directed arcs: 0->1 (4), 1->2 (1), 2->0 (3), 0->2 (2)
    A = _csr(
        data=[4, 1, 3, 2],
        rows=[0, 1, 2, 0],
        cols=[1, 2, 0, 2],
        n=3,
    )
    G0 = Graph.from_csr(A, directed=True, weighted=True)

    A_dir = G0.adj.tocsr()
    kout = np.asarray(A_dir.sum(axis=1)).ravel()
    kin = np.asarray(A_dir.sum(axis=0)).ravel()
    T = float(kout.sum())
    n_trials = int(round(T))

    coo = A_dir.tocoo()
    w = np.rint(coo.data).astype(int)
    p = (kout[coo.row] * kin[coo.col]) / (T ** 2)
    p = np.clip(p, 0.0, 1.0)
    pvals = binom.sf(w - 1, n=n_trials, p=p)

    alpha = 0.10
    keep = pvals <= alpha
    expected_arcs = set(zip(coo.row[keep].tolist(), coo.col[keep].tolist()))

    out = MarginalLikelihoodFilter(alpha=alpha).apply(G0)
    coo2 = out.adj.tocoo()
    got_arcs = set(zip(coo2.row.tolist(), coo2.col.tolist()))

    assert got_arcs == expected_arcs
    assert out.directed and out.weighted


# ----------------- Alpha monotonicity -----------------
def test_mlf_alpha_monotonicity():
    """
    For alpha1 < alpha2, the kept edge set at alpha1 must be a subset of the kept edge set at alpha2.
    """
    A = _csr(
        data=[5, 1, 2, 8, 3, 1],
        rows=[0, 0, 1, 2, 2, 3],
        cols=[1, 2, 2, 3, 0, 1],
        n=4,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True)

    G_small = MarginalLikelihoodFilter(alpha=0.01).apply(G0)
    G_large = MarginalLikelihoodFilter(alpha=0.2).apply(G0)

    e_small = set(zip(*sp.triu(G_small.adj, k=1).nonzero()))
    e_large = set(zip(*sp.triu(G_large.adj, k=1).nonzero()))
    assert e_small.issubset(e_large)


# ----------------- Degenerate T = 0 case -----------------
def test_mlf_handles_no_edges_T_zero():
    A = sp.csr_matrix((3, 3), dtype=float)  # all zeros
    G0 = Graph.from_csr(A, directed=False, weighted=True)
    out = MarginalLikelihoodFilter(alpha=0.05).apply(G0)
    assert out.adj.nnz == 0
    assert out.adj.shape == (3, 3)


# ----------------- Metadata preservation -----------------
def test_mlf_preserves_flags_and_copies_metadata_when_requested():
    meta = pd.DataFrame({"name": ["a", "b", "c"], "group": [1, 0, 1]})
    A = _csr([1, 2], [0, 1], [1, 2], 3)
    G0 = Graph.from_csr(A, directed=False, weighted=True, meta=meta)

    op = MarginalLikelihoodFilter(alpha=0.5, copy_meta=True)
    out = op.apply(G0)

    # Flags preserved
    assert not out.directed and out.weighted
    # Metadata copied (modifying original should not affect the result)
    if out.meta is not None:
        meta.loc[0, "group"] = 999
        assert out.meta.loc[0, "group"] == 1
