import numpy as np
import pytest
import scipy.sparse as sp
from graphconstructor import Graph
from graphconstructor.operators import KNNSelector


def _csr(data, rows, cols, n):
    return sp.csr_matrix((np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))), shape=(n, n))


def test_knn_selector_distance_k2_undirected_symmetric_max():
    """
    For distance mode, the operator should keep the smallest k values per row.
    With directed=False (default in Graph.from_csr), the result is symmetrized using 'max',
    so an asymmetric pair (i->j, j->i) resolves to the larger of the two retained weights.
    """
    # 4x4 distances (smaller is closer). Provide at least 2 off-diagonals per row.
    # Row 0: to 1=0.2, 3=0.3, 2=0.8
    # Row 1: to 0=0.2, 3=0.4, 2=0.6
    # Row 2: to 3=0.5, 0=0.8, 1=0.6
    # Row 3: to 0=0.3, 1=0.4, 2=0.5
    A = _csr(
        data=[0.2, 0.8, 0.3,  0.2, 0.6, 0.4,  0.8, 0.6, 0.5,  0.3, 0.4, 0.5],
        rows=[0, 0, 0,        1, 1, 1,        2, 2, 2,        3, 3, 3],
        cols=[1, 2, 3,        0, 2, 3,        0, 1, 3,        0, 1, 2],
        n=4,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="similarity", sym_op="max")

    out = KNNSelector(k=2, mutual=False, mode="distance").apply(G0)
    A2 = out.adj

    # Still undirected & weighted; no self-loops
    assert not out.directed and out.weighted
    assert np.allclose(A2.diagonal(), 0.0)
    assert (A2 != A2.T).nnz == 0

    # Row-wise smallest-2 choices produce edges:
    # (0,1)=0.2 and (0,3)=0.3; symmetric counterpart for (1,0) is also 0.2, for (3,0) is 0.3
    assert A2[0, 1] == pytest.approx(0.2) and A2[1, 0] == pytest.approx(0.2)
    assert A2[0, 3] == pytest.approx(0.3) and A2[3, 0] == pytest.approx(0.3)
    # Pair (1,3)=0.4 kept; (2,3)=0.5 kept
    assert A2[1, 3] == pytest.approx(0.4) and A2[3, 1] == pytest.approx(0.4)
    assert A2[2, 3] == pytest.approx(0.5) and A2[3, 2] == pytest.approx(0.5)


def test_knn_selector_similarity_k1_mode_override_and_symmetry():
    """
    For similarity mode, keep largest k per row. Symmetrization with 'max' preserves the larger among both directions.
    """
    # Similarity scores
    A = _csr(
        data=[0.9, 0.2,   0.8, 0.3,   0.5, 0.4],
        rows=[0, 0,       1, 1,       2, 2],
        cols=[1, 2,       0, 2,       0, 1],
        n=3,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="similarity", sym_op="max")

    out = KNNSelector(k=1, mutual=False, mode="similarity").apply(G0)
    A2 = out.adj

    # largest-1 per row: 0->1 (0.9), 1->0 (0.8), 2->0 (0.5)
    # sym 'max' on (0,1) => 0.9, on (0,2) => 0.5
    assert (A2 != A2.T).nnz == 0
    assert A2[0, 1] == pytest.approx(0.9) and A2[1, 0] == pytest.approx(0.9)
    assert A2[0, 2] == pytest.approx(0.5) and A2[2, 0] == pytest.approx(0.5)


def test_knn_selector_mutual_true_and_mutual_k_limits_per_row():
    """
    mutual=True keeps only edges that are reciprocal within the top-mutual_k of both endpoints.
    After mutual filtering, each row is limited to k edges (in the original row order).
    """
    # Construct a case where k=1 but mutual_k=2 admits edges that are not both in top-1.
    # Row neighbor rankings (by weight ordering already present in CSR row order):
    # 0: [1 (0.2), 2 (0.9)]
    # 1: [2 (0.3), 0 (0.4)]
    # 2: [1 (0.5), 0 (0.6)]
    A = _csr(
        data=[0.2, 0.9,   0.3, 0.4,   0.5, 0.6],
        rows=[0, 0,       1, 1,       2, 2],
        cols=[1, 2,       2, 0,       1, 0],
        n=3,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="similarity", sym_op="max")

    out = KNNSelector(k=1, mutual=True, mutual_k=2, mode="distance").apply(G0)
    A2 = out.adj

    # With mutual_k=2, (0,1) is mutual within top-2, so it's accepted for node 0 (up to k=1)
    # For pair (1,2), both in each other's top-2; accepted for node 1 (k=1).
    # Symmetrization mirrors the weights using 'max' across directions:
    # - (0,1): max(0.2, 0.4) = 0.4 (since 1->0 is 0.4 in the input row 1)
    # - (1,2): max(0.3, 0.5) = 0.5
    assert (A2 != A2.T).nnz == 0
    assert A2[0, 1] == pytest.approx(0.4)
    assert A2[1, 2] == pytest.approx(0.5)
    # No (0,2)
    assert A2[0, 2] == 0.0 and A2[2, 0] == 0.0


def test_knn_selector_preserves_directed_flag_and_unit_weights_when_graph_is_unweighted():
    """
    If the input Graph is directed or unweighted, the operator should keep those flags intact.
    Unweighted graphs keep weight=1.0 on all retained edges.
    """
    # Directed CSR
    A = _csr(
        data=[2.0, 1.0,  3.0],
        rows=[0, 0,      1],
        cols=[1, 2,      2],
        n=3,
    )
    # Mark as unweighted => data coerced to 1.0 by Graph.from_csr
    G0 = Graph.from_csr(A, directed=True, weighted=False, mode="similarity")

    out = KNNSelector(k=1, mutual=False, mode="similarity").apply(G0)
    assert out.directed and not out.weighted
    # largest-1 per row: row0 keeps the edge to 1 (or 2) but since it's unweighted, weights are 1.0
    # For determinism with CSR row order: largest-1 will choose the largest data value among nonzeros,
    # but data are already 1.0; any kept edge must have weight 1.0.
    assert set(out.adj.nonzero()[0].tolist()) <= {0, 1}
    assert np.allclose(out.adj.data, 1.0)
