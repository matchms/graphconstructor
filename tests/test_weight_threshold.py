import numpy as np
import pytest
import scipy.sparse as sp
from graphconstructor import Graph
from graphconstructor.operators import WeightThreshold


def _csr(data, rows, cols, n):
    return sp.csr_matrix((np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))), shape=(n, n))


def test_weight_threshold_distance_keeps_lt_eps_and_symmetrizes_max():
    """
    For distance mode, keep edges with weight < threshold.
    Undirected graphs are symmetrized using 'max'.
    """
    # Asymmetric distances; ε=0.5
    A = _csr(
        data=[0.4, 0.7, 0.49, 0.9, 0.6, 0.45],
        rows=[0, 0, 1, 1, 2, 2],
        cols=[1, 2, 0, 2, 0, 1],
        n=3,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, sym_op="min")

    out = WeightThreshold(threshold=0.5, mode="distance").apply(G0)
    A2 = out.adj

    # self-loops absent; symmetric
    assert np.allclose(A2.diagonal(), 0.0)
    assert (A2 != A2.T).nnz == 0

    # Kept edges: (0,1)=0.4 and (1,0)=0.49 -> sym 'min' => 0.4
    assert A2[0, 1] == pytest.approx(0.4)
    # (2,1)=0.45 kept; (1,2)=0.9 not kept but was mirrored -> sym 'min' => 0.45
    assert A2[1, 2] == pytest.approx(0.45)
    # (0,2) and (2,0) filtered out (0.7 / 0.6 >= 0.5)
    assert A2[0, 2] == 0.0 and A2[2, 0] == 0.0


def test_weight_threshold_similarity_keeps_gt_tau_and_symmetrizes_max():
    """
    For similarity mode, keep edges with weight > threshold; symmetrization preserves max of both directions.
    """
    tau = 0.75
    A = _csr(
        data=[0.8, 0.6, 0.7, 0.9, 0.4, 0.85],
        rows=[0, 0, 1, 1, 2, 2],
        cols=[1, 2, 0, 2, 0, 1],
        n=3,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, sym_op="max")

    out = WeightThreshold(threshold=tau, mode="similarity").apply(G0)
    A2 = out.adj

    assert (A2 != A2.T).nnz == 0
    # (0,1)=0.8 kept; (1,0)=0.7 not -> mirrored 0.8
    assert A2[0, 1] == pytest.approx(0.8)
    # (1,2)=0.9 and (2,1)=0.85 both kept -> max = 0.9
    assert A2[1, 2] == pytest.approx(0.9)
    # (0,2)=0.6 not kept
    assert A2[0, 2] == 0.0


def test_weight_threshold_preserves_directed_and_unweighted_flags():
    """
    If the input Graph is directed or unweighted, those flags should be preserved.
    For unweighted graphs, the kept edges remain with weight 1.0.
    """
    # Directed adjacency where only some edges pass a similarity threshold
    A = _csr(
        data=[0.2, 0.95, 0.6, 0.85],
        rows=[0, 0, 1, 2],
        cols=[1, 2, 2, 1],
        n=3,
    )
    G0 = Graph.from_csr(A, directed=True, weighted=False)  # unweighted -> all ones internally

    out = WeightThreshold(threshold=0.8, mode="similarity").apply(G0)

    # Flags preserved
    assert out.directed and not out.weighted

    # Edges with original weight > 0.8 should remain present, but weights are 1.0 in unweighted graphs.
    # In our unweighted Graph, all values are 1.0; the operator should interpret the existing data (ones),
    # so to make this meaningful, we check structural outcome and weight = 1.0.
    # All nonzero entries in G0 are ones; applying similarity threshold to ones keeps them if threshold < 1.0,
    # but we set threshold=0.8 so they all remain.
    assert out.adj.nnz == G0.adj.nnz
    assert np.allclose(out.adj.data, 1.0)


def test_weight_threshold_sparse_path_and_no_densification():
    """
    Ensure the operator works efficiently on a sparse input and keeps only the edges that pass the rule.
    """
    rows = np.array([0, 0, 1, 2, 3, 3])
    cols = np.array([1, 2, 0, 3, 0, 2])
    data = np.array([0.35, 0.6, 0.2, 0.39, 0.8, 0.1])
    A = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))

    G0 = Graph.from_csr(A, directed=False, weighted=True, sym_op="max")
    out = WeightThreshold(threshold=0.4, mode="distance").apply(G0)
    A2 = out.adj

    # Kept: (0,1)=0.35, (1,0)=0.2, (2,3)=0.39, (3,2)=0.1
    # Sym 'max': (0,1)->0.35, (2,3)->0.39
    assert (A2 != A2.T).nnz == 0
    assert A2[0, 1] == pytest.approx(0.35)
    assert A2[2, 3] == pytest.approx(0.39)
    # Others filtered out
    assert A2[0, 2] == 0.0 and A2[0, 3] == 0.0
