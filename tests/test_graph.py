import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from graphconstructor import Graph


try:
    import networkx as nx  # noqa: F401
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    import igraph as ig  # noqa: F401
    HAS_IG = True
except Exception:
    HAS_IG = False


def _csr(data, rows, cols, n):
    return sp.csr_matrix(
        (np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))),
        shape=(n, n),
    )


@pytest.fixture
def S_dense():
    """Simple 4-node dense similarity matrix."""
    return np.array([
        [0.0, 0.2, 0.8, 0.5],
        [0.2, 0.0, 0.6, 0.3],
        [0.8, 0.6, 0.0, 0.4],
        [0.5, 0.3, 0.4, 0.0],
    ], dtype=float)


@pytest.fixture
def S_csr():
    """Simple 4-node sparse similarity matrix."""
    return _csr(
        data=[0.2, 0.8, 0.5, 0.6, 0.3, 0.4],
        rows=[0, 0, 0, 1, 1, 2],
        cols=[1, 2, 3, 2, 3, 3],
        n=4,
    )


@pytest.fixture
def meta_df():
    """Simple metadata DataFrame for 4 nodes."""
    return pd.DataFrame({
        "name": ["node0", "node1", "node2", "node3"],
        "group": [0, 1, 0, 2],
    })

# ----------------- construction invariants -----------------
def test_from_csr_enforces_square_and_drops_self_loops_and_symmetrizes_max():
    # 3x3 with asymmetry + self-loops
    A = _csr(
        data=[1.0, 5.0, 3.0, 2.0, 7.0],
        rows=[0,   0,   1,   2,   1],
        cols=[0,   1,   0,   2,   2],
        n=3,
    )
    G = Graph.from_csr(A, directed=False, weighted=True, mode="distance", sym_op="max")

    # self-loops removed
    assert np.allclose(G.adj.diagonal(), 0.0)
    # symmetric (max(0->1=5, 1->0=3) => 5)
    assert (G.adj != G.adj.T).nnz == 0
    assert G.adj[0, 1] == pytest.approx(5.0)
    assert G.adj[1, 0] == pytest.approx(5.0)
    # flags
    assert not G.directed and G.weighted
    # Check mode
    assert G.mode == "distance"


def test_symmetrization_operations():
    """Test min, max, average symmetrization."""
    A = _csr([5.0, 2.0], [0, 1], [1, 0], 2)
    
    G_max = Graph.from_csr(A, directed=False, mode="distance", sym_op="max")
    assert G_max.adj[0, 1] == pytest.approx(5.0)
    
    G_min = Graph.from_csr(A, directed=False, mode="distance", sym_op="min")
    assert G_min.adj[0, 1] == pytest.approx(2.0)
    
    G_avg = Graph.from_csr(A, directed=False, mode="distance", sym_op="average")
    assert G_avg.adj[0, 1] == pytest.approx(3.5)
    
    with pytest.raises(ValueError, match="Unsupported symmetrization"):
        Graph.from_csr(A, directed=False, mode="distance", sym_op="invalid")


def test_from_csr_enforces_square_and_keep_selfloops():
    # 3x3 with asymmetry + self-loops
    A = _csr(
        data=[1.0, 5.0, 3.0, 2.0, 7.0],
        rows=[0,   0,   1,   2,   1],
        cols=[0,   1,   0,   2,   2],
        n=3,
    )
    G = Graph.from_csr(A, directed=True, weighted=True, mode="distance", sym_op="max")

    # self-loops removed
    assert np.allclose(G.adj.diagonal(), np.array([1., 0., 2.]))


def test_from_dense_enforces_square_and_keep_selfloops(S_dense):
    G_no_selfloops = Graph.from_dense(
        S_dense, directed=True, weighted=True, mode="distance", sym_op="max",
        ignore_selfloops=True
        )
    G_selfloops = Graph.from_dense(
        S_dense, directed=True, weighted=True, mode="distance", sym_op="max",
        ignore_selfloops=False
        )
    assert np.allclose(G_no_selfloops.adj.diagonal(), 0.0)
    assert np.allclose(G_selfloops.adj.diagonal(), S_dense.diagonal())


def test_from_csr_unweighted_forces_unit_weights():
    A = _csr([0.2, 0.8], [0, 1], [1, 0], 2)
    G = Graph.from_csr(A, directed=False, weighted=False, mode="distance")
    assert np.allclose(G.adj.data, 1.0)
    assert not G.weighted


def test_from_csr_metadata_alignment_and_names():
    A = _csr([1.0], [0], [1], 2)
    meta = pd.DataFrame({"name": ["a", "b"], "cls": [0, 1]})
    G = Graph.from_csr(A, directed=False, weighted=True, mode="distance", meta=meta)
    assert G.node_names == ["a", "b"]
    assert list(G.meta.columns) == ["name", "cls"]

    with pytest.raises(ValueError, match="meta has .* rows"):
        Graph.from_csr(A, directed=False, weighted=True, mode="distance", meta=meta.iloc[:1])


def test_from_edges_missing_weights():
    """Test that from_edges raises ValueError when weights are missing for weighted graph."""
    edges = np.array([[0, 1], [1, 2]])
    weights = None
    with pytest.raises(ValueError, match="weights must be provided"):
        _ = Graph.from_edges(n=3, edges=edges, mode="distance", weights=weights, directed=False, weighted=True)


def test_from_edges_invalid_indices():
    """Negative or out-of-bounds indices should raise error."""
    edges = np.array([[0, 1], [1, 5]])  # 5 >= n=3
    with pytest.raises((ValueError, IndexError)):
        Graph.from_edges(n=3, edges=edges, mode="distance", weights=[1.0, 1.0])
    
    edges = np.array([[-1, 1], [1, 2]])  # negative index
    with pytest.raises((ValueError, IndexError)):
        Graph.from_edges(n=3, edges=edges, mode="distance", weights=[1.0, 1.0])


def test_from_edges_and_from_dense_equivalence_to_from_csr():
    edges = np.array([[0, 1], [1, 2]])
    weights = np.array([2.0, 3.0])
    G1 = Graph.from_edges(n=3, edges=edges, mode="distance", weights=weights, directed=False, weighted=True)

    D = np.zeros((3, 3), float)
    D[0, 1] = 2.0
    D[1, 2] = 3.0
    G2 = Graph.from_dense(D, directed=False, weighted=True, mode="distance")

    assert (G1.adj != G2.adj).nnz == 0


# ----------------- properties -----------------
def test_n_nodes_n_edges_undirected_counts_upper_triangle():
    A = _csr([1, 1, 1, 1], [0, 0, 1, 2], [1, 2, 2, 2], 3)
    G = Graph.from_csr(A, directed=False, weighted=True, mode="distance")
    # After sym, edges are: (0,1), (0,2), (1,2) => 3 undirected edges
    assert G.n_nodes == 3
    assert G.n_edges == 3


def test_n_edges_directed_counts_arcs():
    A = _csr([1, 1, 1], [0, 1, 2], [1, 2, 0], 3)
    G = Graph.from_csr(A, directed=True, weighted=True, mode="distance")
    assert G.n_edges == 3


def test_has_self_loops_property_false_by_default():
    A = _csr([5.0], [0], [0], 2)
    G = Graph.from_csr(A, directed=False, weighted=True, mode="distance")
    assert not G.has_self_loops


# ----------------- editing -----------------
def test_drop_by_index_and_name_updates_adj_and_meta():
    A = _csr([1, 1, 1], [0, 0, 1], [1, 2, 2], 3)
    meta = pd.DataFrame({"name": ["a", "b", "c"], "cls": [0, 1, 1]})
    G = Graph.from_csr(A, directed=False, weighted=True, mode="distance", meta=meta)

    G2 = G.drop(["b"])  # drop name "b" (index 1)
    assert G2.n_nodes == 2
    assert list(G2.node_names) == ["a", "c"]
    # adjacency shrunk accordingly: original edges touching node 1 removed
    assert G2.adj.shape == (2, 2)
    assert G2.adj.nnz >= 0  # structural check


def test_drop_nonexistent_node():
    """Should raise KeyError/IndexError for invalid nodes."""
    A = _csr([1], [0], [1], 2)
    G = Graph.from_csr(A, mode="distance", directed=False)
    
    with pytest.raises(IndexError):
        G.drop([5])  # out of range
    
    with pytest.raises(KeyError):
        G.drop(["nonexistent"])  # name not found


def test_drop_empty_list():
    """Dropping empty list should return same graph."""
    A = _csr([1], [0], [1], 2)
    G = Graph.from_csr(A, directed=False, mode="distance")
    G2 = G.drop([])
    assert G2.n_nodes == G.n_nodes


def test_drop_single_value():
    """Test dropping a single int/str (not in list)."""
    A = _csr([1], [0], [1], 3)
    meta = pd.DataFrame({"name": ["a", "b", "c"]})
    G = Graph.from_csr(A, directed=False, mode="distance", meta=meta)
    
    G2 = G.drop(1)  # single int
    assert G2.n_nodes == 2
    
    G3 = G.drop("b")  # single string
    assert G3.n_nodes == 2


def test_sorted_by_permuted_order():
    A = _csr([1, 1], [0, 1], [1, 2], 3)
    meta = pd.DataFrame({"name": ["c", "a", "b"], "score": [3, 1, 2]})
    G = Graph.from_csr(A, directed=False, weighted=True, mode="distance", meta=meta)
    G2 = G.sorted_by("score")
    # names should be ordered by score ascending: a, b, c
    assert G2.node_names == ["a", "b", "c"]
    # adjacency must be permuted consistently
    assert G2.adj.shape == (3, 3)


def test_copy_creates_independent_graph():
    """Modifications to copy shouldn't affect original."""
    A = _csr([1], [0], [1], 2)
    meta = pd.DataFrame({"name": ["a", "b"]})
    G = Graph.from_csr(A, mode="distance", meta=meta)
    
    G2 = G.copy()
    G2.adj.data[0] = 999.0
    assert G.adj.data[0] != 999.0  # original unchanged
    
    G2.meta.iloc[0, 0] = "changed"
    assert G.meta.iloc[0, 0] == "a"  # original unchanged


# ----------------- utilities -----------------
def test_graph_is_connected_method():
    # Connected undirected graph
    A1 = _csr([1, 1, 1, 1], [0, 0, 1, 2], [1, 2, 2, 0], 3)
    G1 = Graph.from_csr(A1, directed=False, weighted=True, mode="distance")
    assert G1.is_connected()

    # Disconnected undirected graph
    A2 = _csr([1, 1], [0, 0], [1, 2], 4)
    G2 = Graph.from_csr(A2, directed=False, weighted=True, mode="distance")
    assert not G2.is_connected()

    # Strongly connected directed graph
    A3 = _csr([1, 1, 1, 1], [0, 1, 2, 2], [1, 2, 0, 1], 3)
    G3 = Graph.from_csr(A3, directed=True, weighted=True, mode="distance")
    assert G3.is_connected()

    # Not strongly connected directed graph
    A4 = _csr([1, 1], [0, 1], [1, 2], 3)
    G4 = Graph.from_csr(A4, directed=True, weighted=True, mode="distance")
    assert not G4.is_connected()


def test_graph_connected_components_method():
    # Undirected graph with 2 components
    A = _csr([1, 1, 1, 1], [0, 0, 3, 3], [1, 2, 3, 4], 5)
    G = Graph.from_csr(A, directed=False, weighted=True, mode="distance")
    n_components, labels = G.connected_components(return_labels=True)
    assert n_components == 2
    assert set(labels) == {0, 1}
    assert np.allclose(labels, np.array([0, 0, 0, 1, 1]))


def test_graph_degree_method_weighted_and_unweighted():
    A = _csr([2.0, 3.0, 4.0, 5.0], [0, 1, 3, 3], [1, 2, 2, 0], 4)
    G_weighted = Graph.from_csr(A, directed=False, weighted=True, mode="distance")
    deg_weighted = G_weighted.degree()
    assert np.allclose(deg_weighted, np.array([7.0, 5.0, 7.0, 9.0]))

    # ignore weights
    deg_weighted = G_weighted.degree(ignore_weights=True)
    assert np.allclose(deg_weighted, np.array([2, 2, 2, 2]))

    G_unweighted = Graph.from_csr(A, directed=False, weighted=False, mode="distance")
    deg_unweighted = G_unweighted.degree()
    assert np.allclose(deg_unweighted, np.array([2, 2, 2, 2]))


def test_graph_degree_method_selfloops_counted_twice_unweighted():
    S = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0]
    ])
    # undirected case
    G = Graph.from_dense(S, directed=False, weighted=True, ignore_selfloops=False, mode="distance", sym_op="max")
    deg = G.degree()
    assert np.allclose(deg, np.array([3, 2, 1]))  # node 0 and 1 have self-loop counted twice

    # directed case
    G = Graph.from_dense(S, directed=True, weighted=True, ignore_selfloops=False, mode="distance", sym_op="max")
    deg_out, deg_in = G.degree()
    assert np.allclose(deg_in, np.array([2, 1, 0]))
    assert np.allclose(deg_out, np.array([1, 1, 1]))

def test_graph_degree_method_selfloops_counted_twice_weighted():
    S = np.array([
        [1, 0, 0],
        [0, 2.5, 0],
        [0.25, 0, 0]
    ])
    # undirected case
    G = Graph.from_dense(S, directed=False, weighted=True, ignore_selfloops=False, mode="distance", sym_op="max")
    deg = G.degree()
    assert np.allclose(deg, np.array([2.25, 5, 0.25]))  # node 0 and 1 have self-loop counted twice

    # directed case
    G = Graph.from_dense(S, directed=True, weighted=True, ignore_selfloops=False, mode="distance", sym_op="max")
    deg_out, deg_in = G.degree()
    assert np.allclose(deg_in, np.array([1.25, 2.5, 0]))
    assert np.allclose(deg_out, np.array([1, 2.5, 0.25]))


# ----------------- exporters -----------------
@pytest.mark.skipif(not HAS_NX, reason="networkx not installed")
def test_to_networkx_types_and_node_attributes():
    A = _csr([0.5, 0.7], [0, 1], [1, 2], 3)
    meta = pd.DataFrame({"name": ["n0", "n1", "n2"], "cls": [0, 1, 1]})
    G = Graph.from_csr(A, directed=False, weighted=True, mode="distance", meta=meta)

    nxG = G.to_networkx()
    import networkx as nx
    assert isinstance(nxG, nx.Graph)
    # node attributes set
    assert nxG.nodes[0]["name"] == "n0"
    assert nxG.nodes[2]["cls"] == 1
    # weight on edge reflected
    assert pytest.approx(nxG[0][1]["weight"]) == G.adj[0, 1]


@pytest.mark.skipif(not HAS_IG, reason="python-igraph not installed")
def test_to_igraph_types_and_attributes():
    A = _csr([0.2, 0.9, 0.3], [0, 1, 2], [1, 2, 0], 3)
    meta = pd.DataFrame({"name": ["a", "b", "c"], "label": [10, 20, 30]})
    G = Graph.from_csr(A, directed=True, weighted=True, mode="distance", meta=meta)

    igG = G.to_igraph()
    import igraph as ig
    assert isinstance(igG, ig.Graph)
    assert igG.is_directed()
    # vertex attributes
    assert igG.vs["name"] == ["a", "b", "c"]
    assert igG.vs["label"] == [10, 20, 30]
    # edge weights exist
    assert "weight" in igG.es.attributes()


# ----------------- Distance/similarity conversion -----------------
def test_convert_mode_distance_to_similarity_and_back_dense(S_dense, meta_df):
    G = Graph.from_dense(
        S_dense, directed=False, weighted=True, mode="similarity", meta=meta_df
        )
    G_dist = G.convert_mode("distance")
    assert G_dist.mode == "distance"
    G_sim = G_dist.convert_mode("similarity")
    assert G_sim.mode == "similarity"
    assert np.allclose(G_sim.adj.toarray(), G.adj.toarray())
    # Metadata preserved
    assert G_sim.meta.equals(G.meta)


def test_convert_mode_distance_to_similarity_and_back_csr(S_csr, meta_df):
    G = Graph.from_csr(
        S_csr, directed=False, weighted=True, mode="similarity", meta=meta_df
        )
    G_dist = G.convert_mode("distance")
    assert G_dist.mode == "distance"
    G_sim = G_dist.convert_mode("similarity")
    assert G_sim.mode == "similarity"
    assert np.allclose(G_sim.adj.toarray(), G.adj.toarray())
    # Metadata preserved
    assert G_sim.meta.equals(G.meta)

    # Conversion to dense should lead to different values (in place of 0s)
    G_dense = Graph.from_dense(
        S_csr.toarray(), directed=False, weighted=True, mode="similarity", meta=meta_df
        )
    G_dist_dense = G_dense.convert_mode("distance")
    assert G_dense.adj.data.shape == G.adj.data.shape  # no change in dense conversion
    assert G_dist_dense.adj.data.shape == G_dist.adj.data.shape  # no change in dense conversion


# --------------- expliccit zero handling ---------------
def test_keep_explicit_zeros_in_symmetrization():
    D_csr =_csr(
        data=[0.2, 0.8, 0.5, 0.0, 0.3, 0.4],
        rows=[0, 0, 0, 1, 1, 2],
        cols=[1, 2, 3, 2, 3, 3],
        n=4,
    )
    G = Graph.from_csr(D_csr, mode="distance")
    assert np.allclose(G.adj.data, np.array([0.2, 0.8, 0.5, 0.2, 0. , 0.3, 0.8, 0.4, 0.5, 0.3, 0.4]))

    G_no_zeros = Graph.from_csr(D_csr, mode="distance", keep_explicit_zeros=False)
    assert np.allclose(G_no_zeros.adj.data, np.array([0.2, 0.8, 0.5, 0.2, 0.3, 0.8, 0.4, 0.5, 0.3, 0.4]))

    G_sim = Graph.from_csr(D_csr, mode="similarity")
    assert np.allclose(G_sim.adj.data, np.array([0.2, 0.8, 0.5, 0.2, 0.3, 0.8, 0.4, 0.5, 0.3, 0.4]))
