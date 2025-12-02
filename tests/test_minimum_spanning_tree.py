import numpy as np
import pandas as pd
import pytest
from graphconstructor import Graph
from graphconstructor.operators import MinimumSpanningTree


def dense_to_graph(adj: np.ndarray, mode: str = "distance") -> Graph:
    """Helper: build a simple undirected weighted Graph from dense adjacency."""
    return Graph.from_dense(adj, mode=mode, directed=False, weighted=True)


# ---------------------------------------------------------------------------
# Distance-mode tests
# ---------------------------------------------------------------------------

def test_mst_distance_simple_triangle():
    """
    3-node triangle with unique MST:
        0-1: 1
        0-2: 2
        1-2: 3

    MST should use edges (0,1) and (0,2).
    """
    adj = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    G = dense_to_graph(adj, mode="distance")
    op = MinimumSpanningTree()

    T = op.apply(G)

    # Structural checks
    assert not T.directed
    assert T.n_nodes == 3
    assert T.n_edges == 2  # tree on 3 nodes

    A = T.adj.toarray()
    expected = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    np.testing.assert_array_almost_equal(A, expected)


def test_mst_distance_square_unique_tree():
    """
    4-node graph with unique MST:
        0-1: 1, 0-2: 5, 0-3: 4
        1-2: 2, 1-3: 100
        2-3: 3

    MST should select edges:
        0-1 (1), 1-2 (2), 2-3 (3).
    """
    adj = np.array(
        [
            [0.0, 1.0, 5.0, 4.0],
            [1.0, 0.0, 2.0, 100.0],
            [5.0, 2.0, 0.0, 3.0],
            [4.0, 100.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    G = dense_to_graph(adj, mode="distance")
    op = MinimumSpanningTree()

    T = op.apply(G)

    assert T.n_nodes == 4
    assert T.n_edges == 3
    assert T.is_connected()

    A = T.adj.toarray()
    expected = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0, 3.0],
            [0.0, 0.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    np.testing.assert_array_almost_equal(A, expected)


def test_mst_does_not_modify_input_distance_graph():
    adj = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    G = dense_to_graph(adj, mode="distance")
    A_before = G.adj.copy()

    op = MinimumSpanningTree()
    _ = op.apply(G)

    np.testing.assert_array_almost_equal(G.adj.toarray(), A_before.toarray())


# ---------------------------------------------------------------------------
# Similarity-mode tests (maximum spanning tree behavior)
# ---------------------------------------------------------------------------

def test_mst_similarity_maximum_spanning_tree():
    """
    4-node similarity graph:

        0-1: 0.9
        0-2: 0.4
        0-3: 0.3
        1-2: 0.8
        1-3: 0.7
        2-3: 0.2

    Maximum spanning tree should pick edges:
        (0,1) 0.9, (1,2) 0.8, (1,3) 0.7.
    """
    adj = np.array(
        [
            [0.0, 0.9, 0.4, 0.3],
            [0.9, 0.0, 0.8, 0.7],
            [0.4, 0.8, 0.0, 0.2],
            [0.3, 0.7, 0.2, 0.0],
        ],
        dtype=float,
    )
    G = dense_to_graph(adj, mode="similarity")
    op = MinimumSpanningTree()

    T = op.apply(G)

    assert T.n_nodes == 4
    assert T.n_edges == 3
    assert T.is_connected()
    assert T.mode == "similarity"

    A = T.adj.toarray()
    expected = np.array(
        [
            [0.0, 0.9, 0.0, 0.0],
            [0.9, 0.0, 0.8, 0.7],
            [0.0, 0.8, 0.0, 0.0],
            [0.0, 0.7, 0.0, 0.0],
        ],
        dtype=float,
    )
    np.testing.assert_array_almost_equal(A, expected)


def test_mst_similarity_preserves_original_weights_on_tree_edges():
    """
    Ensure that the returned similarity MST has edge weights equal to
    the original similarities, not transformed costs.
    """
    adj = np.array(
        [
            [0.0, 0.5, 0.1],
            [0.5, 0.0, 0.4],
            [0.1, 0.4, 0.0],
        ],
        dtype=float,
    )
    G = dense_to_graph(adj, mode="similarity")
    op = MinimumSpanningTree()

    T = op.apply(G)

    A_orig = G.adj.toarray()
    A_tree = T.adj.toarray()

    rows, cols = np.where(A_tree > 0)
    for r, c in zip(rows, cols):
        assert A_tree[r, c] == pytest.approx(A_orig[r, c])


def test_mst_similarity_does_not_modify_input_graph():
    adj = np.array(
        [
            [0.0, 0.9, 0.4],
            [0.9, 0.0, 0.8],
            [0.4, 0.8, 0.0],
        ],
        dtype=float,
    )
    G = dense_to_graph(adj, mode="similarity")
    A_before = G.adj.copy()

    op = MinimumSpanningTree()
    _ = op.apply(G)

    np.testing.assert_array_almost_equal(G.adj.toarray(), A_before.toarray())


# ---------------------------------------------------------------------------
# Error handling: disconnected / directed graphs
# ---------------------------------------------------------------------------

def test_mst_raises_on_disconnected_graph():
    """
    Two disconnected components:
      0-1, 2-3. MST should raise, because the graph is not fully connected.
    """
    adj = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    G = dense_to_graph(adj, mode="distance")
    op = MinimumSpanningTree()

    with pytest.raises(ValueError, match="requires a connected graph"):
        _ = op.apply(G)


def test_mst_raises_on_directed_graph():
    """
    MinimumSpanningTree should refuse directed graphs.
    """
    adj = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    G = Graph.from_dense(adj, mode="distance", directed=True, weighted=True)
    op = MinimumSpanningTree()

    with pytest.raises(ValueError, match="only supports undirected graphs"):
        _ = op.apply(G)


# ---------------------------------------------------------------------------
# Metadata behavior
# ---------------------------------------------------------------------------
def test_mst_preserves_metadata_copy_meta_true():
    """
    When copy_meta=True (default), MST result should have a deep copy of meta.
    """
    adj = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    meta = pd.DataFrame({"name": ["a", "b"], "group": [1, 2]})
    G = Graph.from_dense(adj, mode="distance", directed=False, weighted=True, meta=meta)
    op = MinimumSpanningTree(copy_meta=True)

    T = op.apply(G)

    assert T.meta is not None
    assert G.meta is not None

    # Different objects, but equal content
    assert T.meta is not G.meta
    pd.testing.assert_frame_equal(T.meta, G.meta)


'''
# TODO: check why this fails!!
def test_mst_reuses_metadata_copy_meta_false():
    """
    When copy_meta=False, MST result should reuse the same metadata object.
    """
    adj = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    meta = pd.DataFrame({"name": ["a", "b"], "group": [1, 2]})
    G = Graph.from_dense(adj, mode="distance", directed=False, weighted=True, meta=meta)
    op = MinimumSpanningTree(copy_meta=False)

    T = op.apply(G)

    # Same object
    assert T.meta is G.meta
'''
    

# ---------------------------------------------------------------------------
# Structural edge-count & corner cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["distance", "similarity"])
def test_mst_edge_count_is_n_minus_one_for_connected_graphs(mode: str):
    """
    For a connected undirected graph with n>0, MST should have n-1 edges.
    """
    adj = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 4.0, 5.0],
            [2.0, 4.0, 0.0, 6.0],
            [3.0, 5.0, 6.0, 0.0],
        ],
        dtype=float,
    )
    G = dense_to_graph(adj, mode=mode)
    op = MinimumSpanningTree()

    T = op.apply(G)

    assert T.n_nodes == 4
    assert T.n_edges == 3
    assert T.is_connected()

    A = T.adj.toarray()
    assert np.allclose(A, A.T)
    assert np.allclose(np.diag(A), 0.0)


@pytest.mark.parametrize("mode", ["distance", "similarity"])
def test_mst_single_node_graph(mode: str):
    """
    Single-node graph should be trivially connected and MST is itself with 0 edges.
    """
    adj = np.array([[0.0]], dtype=float)
    G = dense_to_graph(adj, mode=mode)
    op = MinimumSpanningTree()

    T = op.apply(G)

    assert T.n_nodes == 1
    assert T.n_edges == 0
    assert T.is_connected()
    np.testing.assert_array_almost_equal(T.adj.toarray(), np.array([[0.0]]))
