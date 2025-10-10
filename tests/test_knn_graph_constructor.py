import numpy as np
import networkx as nx
import pytest
import scipy.sparse as sp

from graphconstructor.constructors import KNNGraphConstructor, GraphConstructionConfig


# ----------------- constructor validation -----------------
def test_knn_ctor_validates_k_and_mode():
    with pytest.raises(TypeError, match="k must be positive"):
        KNNGraphConstructor(k=0)
    with pytest.raises(TypeError, match="k must be positive"):
        KNNGraphConstructor(k=-3)
    with pytest.raises(TypeError, match="mode must be 'distance' or 'similarity'"):
        KNNGraphConstructor(k=3, mode="weird")


# ----------------- from_matrix (dense, distance) -----------------
def test_from_matrix_dense_distance_k2_symmetric_result():
    # Symmetric distance matrix (diag=0), so row-wise kNN is reciprocal
    M = np.array([
        [0.0, 0.2, 0.8, 0.3],
        [0.2, 0.0, 0.6, 0.4],
        [0.8, 0.6, 0.0, 0.5],
        [0.3, 0.4, 0.5, 0.0],
    ])
    gc = KNNGraphConstructor(k=2, mode="distance", out="array")  # default config: symmetric=True, self_loops=False
    A = gc.from_matrix(M)

    # shape, float dtype, symmetric, and no self-loops
    assert isinstance(A, sp.csr_matrix)
    assert A.shape == (4, 4)
    assert A.dtype == float
    assert (A != A.T).nnz == 0
    assert np.allclose(A.diagonal(), 0.0)

    # For symmetric distance matrix and k=2, edges should connect 0 with {1,3}, 1 with {0,3}, etc.
    # And since matrix is symmetric, weights should match original distances.
    assert A[0, 1] == pytest.approx(0.2)
    assert A[0, 3] == pytest.approx(0.3)
    assert A[1, 3] == pytest.approx(0.4)
    assert A[2, 3] == pytest.approx(0.5)


# ----------------- from_matrix (sparse input) -----------------
def test_from_matrix_sparse_distance_k1_enough_offdiagonals():
    # Build 3x3 sparse with at least one off-diagonal per row
    # Distances (smaller = closer):
    # row0 -> {1:0.1, 2:0.7}; row1 -> {0:0.2, 2:0.4}; row2 -> {1:0.3, 0:0.9}
    rows = np.array([0, 0, 1, 1, 2, 2])
    cols = np.array([1, 2, 0, 2, 1, 0])
    data = np.array([0.1, 0.7, 0.2, 0.4, 0.3, 0.9])
    M = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))

    gc = KNNGraphConstructor(k=1, mode="distance", out="array")
    A = gc.from_matrix(M)

    # With k=1 (smallest distance):
    # row0 picks 1 (0.1), row1 picks 0 (0.2), row2 picks 1 (0.3).
    # Finalization makes it symmetric (max), no self-loops.
    assert (A != A.T).nnz == 0
    assert np.allclose(A.diagonal(), 0.0)
    # Edges present between (0,1) from rows 0 and 1 (min distances 0.1 and 0.2 -> sym 'max' => 0.2)
    assert A[0, 1] == pytest.approx(0.2)
    assert A[1, 0] == pytest.approx(0.2)
    # Edge (1,2) only chosen by row2 (0.3), so after sym it's still 0.3
    assert A[1, 2] == pytest.approx(0.3)
    assert A[2, 1] == pytest.approx(0.3)


# ----------------- from_matrix mode override -----------------
def test_from_matrix_mode_override_similarity_on_distance_instance():
    # Construct with mode="distance" but override call with mode="similarity"
    # Similarity matrix (larger = closer)
    S = np.array([
        [1.0, 0.9, 0.2],
        [0.8, 1.0, 0.3],
        [0.5, 0.4, 1.0],
    ])
    gc = KNNGraphConstructor(k=1, mode="distance", out="array")
    A = gc.from_matrix(S, mode="similarity")

    # For similarity, each row picks the largest off-diagonal:
    # row0->1 (0.9), row1->0 (0.8), row2->0 (0.5); finalize sym (max) keeps reciprocal (0<->1) at 0.9
    assert (A != A.T).nnz == 0
    assert A[0, 1] == pytest.approx(0.9)
    assert A[1, 0] == pytest.approx(0.9)
    # 2 connected to 0 (0.5) but not vice versa -> after sym (max) == 0.5
    assert A[2, 0] == pytest.approx(0.5)
    assert A[0, 2] == pytest.approx(0.5)


# ----------------- mutual filtering -----------------
def test_mutual_removes_non_reciprocal_and_preserves_weights_then_symmetrizes():
    # n=3, k=1. Non-mutual cycle: 0->1 (w=0.2), 1->2 (w=0.4), 2->0 (w=0.6)
    ind = np.array([[1], [2], [0]])
    dist = np.array([[0.2], [0.4], [0.6]])

    cfg = GraphConstructionConfig(symmetric=True, symmetrize_op="max", self_loops=False)
    gc = KNNGraphConstructor(k=1, mutual=True, mode="distance", out="array", config=cfg)
    A = gc.from_knn(ind, dist)

    # No pair is reciprocal -> all edges dropped; finalize keeps it zero & symmetric
    assert A.nnz == 0
    assert (A != A.T).nnz == 0
    assert np.allclose(A.toarray(), 0.0)

    # Now make a reciprocal pair with different weights in each direction
    ind2 = np.array([[1], [0], [0]])               # 0->1, 1->0, 2->0
    dist2 = np.array([[0.2], [0.35], [0.9]])       # weights differ on 0<->1
    A2 = gc.from_knn(ind2, dist2)

    # Mutual keeps 0<->1 edges with original directional weights, then _finalize(sym='max') uses max=0.35
    assert (A2 != A2.T).nnz == 0
    assert A2[0, 1] == pytest.approx(0.35)
    assert A2[1, 0] == pytest.approx(0.35)
    # 2->0 is not reciprocal -> dropped
    assert A2[2, 0] == 0.0 and A2[0, 2] == 0.0
    # Diagonal cleared
    assert np.allclose(A2.diagonal(), 0.0)


# ----------------- store_weights = False -----------------
def test_store_weights_false_sets_unit_weights():
    cfg = GraphConstructionConfig(symmetric=False, self_loops=False, store_weights=False)
    gc = KNNGraphConstructor(k=2, mutual=False, out="array", mode="distance", config=cfg)

    ind = np.array([[1, 2], [0, 2], [0, 1]])
    dist = np.array([[0.2, 0.9], [0.3, 0.4], [0.5, 0.6]])
    A = gc.from_knn(ind, dist)

    # All non-diagonal edges exist with weight 1.0 regardless of distances
    assert A[0, 1] == 1.0 and A[0, 2] == 1.0
    assert A[1, 0] == 1.0 and A[1, 2] == 1.0
    assert A[2, 0] == 1.0 and A[2, 1] == 1.0
    # Not forced symmetric in this config (but happens to be symmetric by construction here)
    assert np.allclose(A.diagonal(), 0.0)


def test_networkx_default_undirected_from_matrix_distance():
    # Symmetric distance matrix; k=2; default config => symmetric graph, no self-loops
    M = np.array([
        [0.0, 0.2, 0.8, 0.3],
        [0.2, 0.0, 0.6, 0.4],
        [0.8, 0.6, 0.0, 0.5],
        [0.3, 0.4, 0.5, 0.0],
    ])
    gc = KNNGraphConstructor(k=2, mode="distance")  # out defaults to "networkx"
    G = gc.from_matrix(M)

    # Type & undirected due to symmetric=True (default)
    assert isinstance(G, nx.Graph)
    assert not any(u == v for u, v in G.edges())  # no self-loops

    # Weights mirror the symmetrized CSR (max) which equals original distances for symmetric M
    assert pytest.approx(G[0][1]["weight"]) == 0.2
    assert pytest.approx(G[0][3]["weight"]) == 0.3
    assert pytest.approx(G[1][3]["weight"]) == 0.4
    assert pytest.approx(G[2][3]["weight"]) == 0.5


def test_networkx_directed_when_symmetric_false_and_unit_weights():
    # Asymmetric selection; store_weights=False => all weights become 1.0
    cfg = GraphConstructionConfig(symmetric=False, self_loops=False, store_weights=False)
    gc = KNNGraphConstructor(k=1, mode="distance", config=cfg)

    # n=2, each selects the other as its single neighbor
    ind = np.array([[1], [0]])
    dist = np.array([[0.2], [0.9]])
    G = gc.from_knn(ind, dist)

    # Directed graph due to symmetric=False
    assert isinstance(G, nx.DiGraph)
    # Two directed edges, unit weights
    assert G.has_edge(0, 1) and G.has_edge(1, 0)
    assert pytest.approx(G[0][1]["weight"]) == 1.0
    assert pytest.approx(G[1][0]["weight"]) == 1.0
    # No self-loops
    assert not G.has_edge(0, 0) and not G.has_edge(1, 1)


def test_networkx_mutual_k_allows_edges_beyond_top_k_but_limited_to_k():
    # k=1 but mutual_k=2: allow an edge if nodes are within each other's top-2
    # even when not both in top-1; keep at most 1 edge per node (k).
    gc = KNNGraphConstructor(k=1, mutual=True, mutual_k=2, mode="distance")

    # Indices have two columns (base_k=2) to let mutual_k=2 take effect
    # Row-wise neighbor lists (by rank):
    # 0: [1, 2]   -> prefers 1 first
    # 1: [2, 0]   -> prefers 2 first, but 0 is within top-2
    # 2: [1, 0]
    ind = np.array([
        [1, 2],
        [2, 0],
        [1, 0],
    ])
    # Distances for the chosen positions
    dist = np.array([
        [0.2, 0.9],  # 0->1 has weight 0.2
        [0.3, 0.4],  # 1->2 has weight 0.3; 1->0 would have 0.4 (not picked due to k=1)
        [0.5, 0.6],  # 2->1 has weight 0.5
    ])

    G = gc.from_knn(ind, dist)

    # Default symmetric=True -> undirected graph
    assert isinstance(G, nx.Graph)

    # (0,1) is accepted because 1 is in 0's top-2 AND 0 is in 1's top-2,
    # even though 1 does not *select* 0 in its top-1 (k=1).
    # Only 0->1 was added directionally; finalize(sym='max') mirrors to undirected with weight 0.2.
    assert G.has_edge(0, 1)
    assert pytest.approx(G[0][1]["weight"]) == 0.2

    # For (1,2): mutual within top-2 and 1's first choice is 2 -> edge weight 0.3 (no opposing larger weight)
    assert G.has_edge(1, 2)
    assert pytest.approx(G[1][2]["weight"]) == 0.5

    # (0,2) not mutual within top-2 in both directions for the *positions considered with k=1 per node*
    # and given earlier picks, so it should not be present
    assert not G.has_edge(0, 2)

    # Test again with symmetric=False to get a directed graph
    cfg = GraphConstructionConfig(symmetric=False)
    gc = KNNGraphConstructor(k=1, mutual=True, mutual_k=2, mode="distance", config=cfg)
    G = gc.from_knn(ind, dist)
    assert isinstance(G, nx.DiGraph)
    assert pytest.approx(G[1][2]["weight"]) == 0.3
    assert pytest.approx(G[2][1]["weight"]) == 0.5
