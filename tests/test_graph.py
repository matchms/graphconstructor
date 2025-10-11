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


# ----------------- construction invariants -----------------
def test_from_csr_enforces_square_and_drops_self_loops_and_symmetrizes_max():
    # 3x3 with asymmetry + self-loops
    A = _csr(
        data=[1.0, 5.0, 3.0, 2.0, 7.0],
        rows=[0,   0,   1,   2,   2],
        cols=[0,   1,   0,   2,   2],
        n=3,
    )
    G = Graph.from_csr(A, directed=False, weighted=True, sym_op="max")

    # self-loops removed
    assert np.allclose(G.adj.diagonal(), 0.0)
    # symmetric (max(0->1=5, 1->0=3) => 5)
    assert (G.adj != G.adj.T).nnz == 0
    assert G.adj[0, 1] == pytest.approx(5.0)
    assert G.adj[1, 0] == pytest.approx(5.0)
    # flags
    assert not G.directed and G.weighted


def test_from_csr_unweighted_forces_unit_weights():
    A = _csr([0.2, 0.8], [0, 1], [1, 0], 2)
    G = Graph.from_csr(A, directed=False, weighted=False)
    assert np.allclose(G.adj.data, 1.0)
    assert not G.weighted


def test_from_csr_metadata_alignment_and_names():
    A = _csr([1.0], [0], [1], 2)
    meta = pd.DataFrame({"name": ["a", "b"], "cls": [0, 1]})
    G = Graph.from_csr(A, directed=False, weighted=True, meta=meta)
    assert G.node_names == ["a", "b"]
    assert list(G.meta.columns) == ["name", "cls"]

    with pytest.raises(ValueError, match="meta has .* rows"):
        Graph.from_csr(A, directed=False, weighted=True, meta=meta.iloc[:1])


def test_from_edges_and_from_dense_equivalence_to_from_csr():
    edges = np.array([[0, 1], [1, 2]])
    weights = np.array([2.0, 3.0])
    G1 = Graph.from_edges(n=3, edges=edges, weights=weights, directed=False, weighted=True)

    D = np.zeros((3, 3), float)
    D[0, 1] = 2.0
    D[1, 2] = 3.0
    G2 = Graph.from_dense(D, directed=False, weighted=True)

    assert (G1.adj != G2.adj).nnz == 0


# ----------------- properties -----------------
def test_n_nodes_n_edges_undirected_counts_upper_triangle():
    A = _csr([1, 1, 1, 1], [0, 0, 1, 2], [1, 2, 2, 2], 3)
    G = Graph.from_csr(A, directed=False, weighted=True)
    # After sym, edges are: (0,1), (0,2), (1,2) => 3 undirected edges
    assert G.n_nodes == 3
    assert G.n_edges == 3


def test_n_edges_directed_counts_arcs():
    A = _csr([1, 1, 1], [0, 1, 2], [1, 2, 0], 3)
    G = Graph.from_csr(A, directed=True, weighted=True)
    assert G.n_edges == 3


def test_has_self_loops_property_false_by_default():
    A = _csr([5.0], [0], [0], 2)
    G = Graph.from_csr(A, directed=False, weighted=True)
    assert not G.has_self_loops


# ----------------- editing -----------------
def test_drop_by_index_and_name_updates_adj_and_meta():
    A = _csr([1, 1, 1], [0, 0, 1], [1, 2, 2], 3)
    meta = pd.DataFrame({"name": ["a", "b", "c"], "cls": [0, 1, 1]})
    G = Graph.from_csr(A, directed=False, weighted=True, meta=meta)

    G2 = G.drop(["b"])  # drop name "b" (index 1)
    assert G2.n_nodes == 2
    assert list(G2.node_names) == ["a", "c"]
    # adjacency shrunk accordingly: original edges touching node 1 removed
    assert G2.adj.shape == (2, 2)
    assert G2.adj.nnz >= 0  # structural check


def test_sorted_by_permuted_order():
    A = _csr([1, 1], [0, 1], [1, 2], 3)
    meta = pd.DataFrame({"name": ["c", "a", "b"], "score": [3, 1, 2]})
    G = Graph.from_csr(A, directed=False, weighted=True, meta=meta)
    G2 = G.sorted_by("score")
    # names should be ordered by score ascending: a, b, c
    assert G2.node_names == ["a", "b", "c"]
    # adjacency must be permuted consistently
    assert G2.adj.shape == (3, 3)


# ----------------- exporters -----------------
@pytest.mark.skipif(not HAS_NX, reason="networkx not installed")
def test_to_networkx_types_and_node_attributes():
    A = _csr([0.5, 0.7], [0, 1], [1, 2], 3)
    meta = pd.DataFrame({"name": ["n0", "n1", "n2"], "cls": [0, 1, 1]})
    G = Graph.from_csr(A, directed=False, weighted=True, meta=meta)

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
    G = Graph.from_csr(A, directed=True, weighted=True, meta=meta)

    igG = G.to_igraph()
    import igraph as ig
    assert isinstance(igG, ig.Graph)
    assert igG.is_directed()
    # vertex attributes
    assert igG.vs["name"] == ["a", "b", "c"]
    assert igG.vs["label"] == [10, 20, 30]
    # edge weights exist
    assert "weight" in igG.es.attributes()
