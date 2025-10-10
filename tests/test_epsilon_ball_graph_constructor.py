import numpy as np
import networkx as nx
import pytest
import scipy.sparse as sp

from graphconstructor.constructors import EpsilonBallGraphConstructor, GraphConstructionConfig


# ----------------- constructor validation -----------------
def test_eps_ctor_validates_mode():
    with pytest.raises(TypeError, match="mode must be 'distance' or 'similarity'"):
        EpsilonBallGraphConstructor(threshold=0.5, mode="odd")


# ----------------- from_matrix (dense, distance) -----------------
def test_from_matrix_dense_distance_threshold_and_symmetrize_max():
    # Asymmetric distances; ε = 0.5. Keep entries < 0.5.
    # Diagonal values (0.0) should be dropped by finalize (self_loops=False)
    M = np.array([
        [0.0, 0.4, 0.7],
        [0.49, 0.0, 0.9],
        [0.6, 0.45, 0.0],
    ], dtype=float)

    # default config: symmetric=True (max), self_loops=False, store_weights=True
    gc = EpsilonBallGraphConstructor(threshold=0.5, mode="distance", out="array")
    A = gc.from_matrix(M)

    assert isinstance(A, sp.csr_matrix) and A.shape == (3, 3)
    # self-loops removed
    assert np.allclose(A.diagonal(), 0.0)
    # symmetrized (max)
    assert (A != A.T).nnz == 0

    # Pairs that pass threshold (< 0.5):
    # (0,1)=0.4 and (1,0)=0.49 -> symmetric max is 0.49
    assert A[0, 1] == pytest.approx(0.49)
    assert A[1, 0] == pytest.approx(0.49)

    # (2,1)=0.45 (<0.5) but (1,2)=0.9 (not kept). After sym 'max', value mirrors to both sides = 0.45
    assert A[2, 1] == pytest.approx(0.45)
    assert A[1, 2] == pytest.approx(0.45)

    # (0,2)=0.7 and (2,0)=0.6 -> both >= 0.5 -> not kept
    assert A[0, 2] == 0.0 and A[2, 0] == 0.0


# ----------------- from_matrix (dense, similarity) -----------------
def test_from_matrix_dense_similarity_threshold_and_symmetrize_max():
    # Similarities; τ = 0.75. Keep entries > τ.
    S = np.array([
        [1.0, 0.8, 0.6],
        [0.7, 1.0, 0.9],
        [0.4, 0.85, 1.0],
    ], dtype=float)

    gc = EpsilonBallGraphConstructor(threshold=0.75, mode="similarity", out="array")
    A = gc.from_matrix(S)

    assert (A != A.T).nnz == 0
    assert np.allclose(A.diagonal(), 0.0)

    # (0,1)=0.8 kept; (1,0)=0.7 not kept -> sym 'max' -> 0.8 both ways
    assert A[0, 1] == pytest.approx(0.8) and A[1, 0] == pytest.approx(0.8)
    # (1,2)=0.9 and (2,1)=0.85 -> both kept -> max is 0.9 both ways
    assert A[1, 2] == pytest.approx(0.9) and A[2, 1] == pytest.approx(0.9)
    # (0,2)=0.6 below τ (not kept)
    assert A[0, 2] == 0.0 and A[2, 0] == 0.0


# ----------------- from_matrix (sparse input, distance) -----------------
def test_from_matrix_sparse_distance_no_densify_needed_behaviour():
    # 4x4 CSR with off-diagonals; ε=0.4
    rows = np.array([0, 0, 1, 2, 3, 3])
    cols = np.array([1, 2, 0, 3, 0, 2])
    data = np.array([0.35, 0.6, 0.2, 0.39, 0.8, 0.1])
    M = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))

    gc = EpsilonBallGraphConstructor(threshold=0.4, mode="distance", out="array")
    A = gc.from_matrix(M)

    # Kept edges: (0,1)=0.35, (1,0)=0.2, (2,3)=0.39, (3,2)=0.1
    # Sym 'max': (0,1) -> 0.35 vs 0.2 => 0.35; (2,3) -> 0.39 vs 0.1 => 0.39
    assert (A != A.T).nnz == 0
    assert A[0, 1] == pytest.approx(0.35) and A[1, 0] == pytest.approx(0.35)
    assert A[2, 3] == pytest.approx(0.39) and A[3, 2] == pytest.approx(0.39)
    # Others fail threshold or not present
    assert A[0, 2] == 0.0 and A[3, 0] == 0.0


# ----------------- per-call mode override on from_matrix -----------------
def test_from_matrix_mode_override_similarity_on_distance_instance():
    S = np.array([
        [1.0, 0.9, 0.3],
        [0.2, 1.0, 0.85],
        [0.7, 0.6, 1.0],
    ])
    gc = EpsilonBallGraphConstructor(threshold=0.8, mode="distance", out="array")
    A = gc.from_matrix(S, mode="similarity")

    # Keep > 0.8: (0,1)=0.9, (1,2)=0.85
    assert (A != A.T).nnz == 0
    assert A[0, 1] == pytest.approx(0.9) and A[1, 0] == pytest.approx(0.9)
    assert A[1, 2] == pytest.approx(0.85) and A[2, 1] == pytest.approx(0.85)


# ----------------- from_knn (distance) -----------------
def test_from_knn_distance_threshold_and_self_loops_removed():
    # n=3, k=3; ε=0.35 -> keep distances < 0.35
    # Include diagonal entries (i->i) and padded-like large values; the diagonal must be removed in finalize.
    indices = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [2, 1, 0],
    ])
    dist = np.array([
        [0.0, 0.3, 0.6],   # keep (0->1)
        [0.4, 0.0, 0.2],   # keep (1->2)
        [0.9, 0.25, 0.0],  # keep (2->1)
    ])
    gc = EpsilonBallGraphConstructor(threshold=0.35, mode="distance", out="array")
    A = gc.from_knn(indices, dist)

    # self-loops removed, symmetric (max)
    assert np.allclose(A.diagonal(), 0.0)
    assert (A != A.T).nnz == 0

    # Kept directed edges: 0->1 (0.3), 1->2 (0.2), 2->1 (0.25).
    # After sym 'max': (1,2) becomes 0.25 (max of 0.2 and 0.25).
    assert A[0, 1] == pytest.approx(0.3) and A[1, 0] == pytest.approx(0.3)
    assert A[1, 2] == pytest.approx(0.25) and A[2, 1] == pytest.approx(0.25)


# ----------------- from_knn (similarity) + store_weights=False -----------------
def test_from_knn_similarity_threshold_and_unit_weights_when_store_weights_false():
    # τ = 0.7 -> keep > 0.7
    indices = np.array([
        [1, 2],
        [2, 0],
        [0, 1],
    ])
    sims = np.array([
        [0.8, 0.4],  # keep 0->1
        [0.9, 0.6],  # keep 1->2
        [0.1, 0.95], # keep 2->1
    ])
    cfg = GraphConstructionConfig(store_weights=False, symmetric=True, self_loops=False)
    gc = EpsilonBallGraphConstructor(threshold=0.7, mode="similarity", out="array", config=cfg)
    A = gc.from_knn(indices, sims)

    # All kept edges must have weight 1.0 after store_weights=False
    assert A[0, 1] == 1.0 and A[1, 0] == 1.0   # sym 'max' mirrors presence
    # For (1,2) and (2,1), both directions present -> remains 1.0
    assert A[1, 2] == 1.0 and A[2, 1] == 1.0
    assert np.allclose(A.diagonal(), 0.0)
    assert (A != A.T).nnz == 0


def test_networkx_default_undirected_from_matrix_distance():
    # Asymmetric distances; ε = 0.5. Keep entries < 0.5.
    M = np.array([
        [0.0, 0.4, 0.7],
        [0.49, 0.0, 0.9],
        [0.6, 0.45, 0.0],
    ], dtype=float)

    # out defaults to "networkx", config defaults to symmetric=True, symmetrize_op="max"
    gc = EpsilonBallGraphConstructor(threshold=0.5, mode="distance")
    G = gc.from_matrix(M)

    assert isinstance(G, nx.Graph)  # undirected by default
    # no self-loops
    assert not any(u == v for u, v in G.edges())

    # (0,1)=0.4 and (1,0)=0.49 kept -> max sym => 0.49
    assert pytest.approx(G[0][1]["weight"]) == 0.49
    # (2,1)=0.45 kept; (1,2)=0.9 not kept -> mirrored as 0.45
    assert pytest.approx(G[1][2]["weight"]) == 0.45
    # (0,2) not kept
    assert not G.has_edge(0, 2)


def test_networkx_directed_when_symmetric_false_from_matrix_distance():
    # ε = 0.4. Build a tiny dense matrix where only certain directions pass.
    M = np.array([
        [0.0, 0.35, 0.6],
        [0.2,  0.0,  0.9],
        [0.7,  0.39, 0.0],
    ], dtype=float)
    cfg = GraphConstructionConfig(symmetric=False, self_loops=False)
    gc = EpsilonBallGraphConstructor(threshold=0.4, mode="distance", config=cfg)

    G = gc.from_matrix(M)
    assert isinstance(G, nx.DiGraph)  # directed due to symmetric=False

    # Edges that pass ε: 0->1 (0.35), 1->0 (0.2), 2->1 (0.39)
    assert G.has_edge(0, 1) and pytest.approx(G[0][1]["weight"]) == 0.35
    assert G.has_edge(1, 0) and pytest.approx(G[1][0]["weight"]) == 0.2
    assert G.has_edge(2, 1) and pytest.approx(G[2][1]["weight"]) == 0.39
    # No self-loops present
    assert not G.has_edge(0, 0) and not G.has_edge(1, 1) and not G.has_edge(2, 2)


def test_networkx_store_weights_false_similarity_from_knn_unit_weights():
    # τ = 0.75. Keep > τ. store_weights=False => all kept edges weight 1.0
    indices = np.array([
        [1, 2],
        [2, 0],
        [0, 1],
    ])
    sims = np.array([
        [0.8, 0.4],   # 0->1 kept
        [0.9, 0.6],   # 1->2 kept
        [0.1, 0.95],  # 2->1 kept
    ])
    cfg = GraphConstructionConfig(store_weights=False, symmetric=True, self_loops=False)
    gc = EpsilonBallGraphConstructor(threshold=0.75, mode="similarity", config=cfg)

    G = gc.from_knn(indices, sims)
    assert isinstance(G, nx.Graph)
    # All kept edges must be weight 1.0 after symmetrization
    assert G.has_edge(0, 1) and pytest.approx(G[0][1]["weight"]) == 1.0
    assert G.has_edge(1, 2) and pytest.approx(G[1][2]["weight"]) == 1.0
    # no self-loops
    assert not G.has_edge(0, 0) and not G.has_edge(1, 1) and not G.has_edge(2, 2)


def test_networkx_mode_override_on_from_matrix():
    # Construct instance with mode="distance" but call with mode="similarity"
    S = np.array([
        [1.0, 0.9, 0.3],
        [0.2, 1.0, 0.85],
        [0.7, 0.6, 1.0],
    ], dtype=float)

    gc = EpsilonBallGraphConstructor(threshold=0.8, mode="distance")  # out defaults to networkx
    G = gc.from_matrix(S, mode="similarity")

    assert isinstance(G, nx.Graph)
    # Keep > 0.8: (0,1)=0.9 and (1,2)=0.85; sym (max) keeps those weights
    assert G.has_edge(0, 1) and pytest.approx(G[0][1]["weight"]) == 0.9
    assert G.has_edge(1, 2) and pytest.approx(G[1][2]["weight"]) == 0.85


def test_networkx_sparse_input_path_distance():
    # Sparse 4x4; ε = 0.4
    rows = np.array([0, 0, 1, 2, 3, 3])
    cols = np.array([1, 2, 0, 3, 0, 2])
    data = np.array([0.35, 0.6, 0.2, 0.39, 0.8, 0.1])
    M = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))

    gc = EpsilonBallGraphConstructor(threshold=0.4, mode="distance")  # networkx out by default
    G = gc.from_matrix(M)

    assert isinstance(G, nx.Graph)
    # Kept edges (after sym 'max'):
    # (0,1): max(0.35, 0.2) = 0.35
    assert G.has_edge(0, 1) and pytest.approx(G[0][1]["weight"]) == 0.35
    # (2,3): max(0.39, 0.1) = 0.39
    assert G.has_edge(2, 3) and pytest.approx(G[2][3]["weight"]) == 0.39
    # Not present:
    assert not G.has_edge(0, 2) and not G.has_edge(0, 3)
