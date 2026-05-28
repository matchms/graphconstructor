import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
from graphconstructor import Graph
from graphconstructor.operators import MetricDistanceFilter


def _csr(data, rows, cols, n):
    return sp.csr_matrix(
        (np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))),
        shape=(n, n),
    )


def simple_undirected_graph():
    A = _csr(
        data=[0.5, 0.5, 0.3, 0.3, 0.8, 0.8],
        rows=[0, 1, 0, 2, 1, 2],
        cols=[1, 0, 2, 0, 2, 1],
        n=3,
    )

    return Graph.from_csr(A, directed=False, weighted=True, mode="similarity")


def simple_directed_graph():
    A = _csr(
        data=[0.5, 0.5, 0.3],
        rows=[0, 0, 1],
        cols=[1, 2, 2],
        n=3,
    )

    return Graph.from_csr(A, directed=True, weighted=True, mode="similarity")


def test_basic_undirected_filtering():
    G0 = simple_undirected_graph()

    out = MetricDistanceFilter(distortion=False, verbose=False).apply(G0)

    assert isinstance(out, Graph)
    assert not out.directed
    assert out.weighted

    original_edges = G0.to_networkx().number_of_edges()
    result_edges = out.to_networkx().number_of_edges()
    assert result_edges <= original_edges


def test_undirected_filtering_distortion():
    G0 = simple_undirected_graph()

    out = MetricDistanceFilter(distortion=True, verbose=False).apply(G0)

    assert isinstance(out, tuple)
    assert len(out) == 2

    filtered_graph, svals = out
    assert isinstance(filtered_graph, Graph)
    assert isinstance(svals, dict)

    if svals:
        key = next(iter(svals.keys()))
        assert isinstance(key, tuple)
        assert len(key) == 2


def test_directed_graph_not_implemented():
    G0 = simple_directed_graph()
    with pytest.raises(NotImplementedError):
        MetricDistanceFilter().apply(G0)


def test_edge_removal_logic():
    G0 = simple_undirected_graph()
    out = MetricDistanceFilter().apply(G0)

    original_nx = G0.to_networkx()
    out_nx = out.to_networkx()

    assert out_nx.number_of_edges() <= original_nx.number_of_edges()

    if nx.is_connected(original_nx):
        assert nx.is_connected(out_nx)


def test_isolated_nodes():
    A = _csr(
        data=[0.5, 0.5],
        rows=[0, 1],
        cols=[1, 0],
        n=3,
    )
    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="distance")
    out = MetricDistanceFilter().apply(G0)

    assert out.to_networkx().number_of_nodes() == 3
    assert 2 in out.to_networkx().nodes()


def test_empty_graph():
    A = _csr(data=[], rows=[], cols=[], n=3)
    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="distance")

    out = MetricDistanceFilter().apply(G0)

    assert out.to_networkx().number_of_edges() == 0
    assert out.to_networkx().number_of_nodes() == 3


def test_distance_mode_removes_semimetric_edge():
    """In distance mode, an edge is removed if an indirect path is shorter."""
    A = _csr(
        data=[
            1.0, 1.0,  # 0 -- 1
            1.0, 1.0,  # 1 -- 2
            3.0, 3.0,  # 0 -- 2, longer than 0 -- 1 -- 2
        ],
        rows=[0, 1, 1, 2, 0, 2],
        cols=[1, 0, 2, 1, 2, 0],
        n=3,
    )

    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="distance")
    out = MetricDistanceFilter(mode="distance").apply(G0)
    out_nx = out.to_networkx()

    assert out_nx.has_edge(0, 1)
    assert out_nx.has_edge(1, 2)
    assert not out_nx.has_edge(0, 2)


def test_similarity_mode_converts_similarity_before_filtering():
    """
    In similarity mode, strong similarities should behave like short distances.

    The weak edge 0 -- 2 should be removed because 0 -- 1 -- 2 is the
    stronger / closer indirect connection.
    """
    A = _csr(
        data=[
            0.9, 0.9,  # 0 -- 1, strong similarity
            0.9, 0.9,  # 1 -- 2, strong similarity
            0.1, 0.1,  # 0 -- 2, weak similarity
        ],
        rows=[0, 1, 1, 2, 0, 2],
        cols=[1, 0, 2, 1, 2, 0],
        n=3,
    )

    G0 = Graph.from_csr(A, directed=False, weighted=True, mode="similarity")
    out = MetricDistanceFilter(mode="similarity").apply(G0)
    out_nx = out.to_networkx()

    assert out_nx.has_edge(0, 1)
    assert out_nx.has_edge(1, 2)
    assert not out_nx.has_edge(0, 2)


def test_similarity_and_distance_inputs_give_equivalent_backbone_when_consistent():
    """
    A similarity graph and its corresponding distance graph should produce
    the same backbone topology.
    """
    A_sim = _csr(
        data=[
            0.9, 0.9,
            0.9, 0.9,
            0.1, 0.1,
        ],
        rows=[0, 1, 1, 2, 0, 2],
        cols=[1, 0, 2, 1, 2, 0],
        n=3,
    )

    A_dist = _csr(
        data=[
            0.1, 0.1,
            0.1, 0.1,
            0.9, 0.9,
        ],
        rows=[0, 1, 1, 2, 0, 2],
        cols=[1, 0, 2, 1, 2, 0],
        n=3,
    )

    G_sim = Graph.from_csr(A_sim, directed=False, weighted=True, mode="similarity")
    G_dist = Graph.from_csr(A_dist, directed=False, weighted=True, mode="distance")

    out_sim = MetricDistanceFilter(mode="similarity").apply(G_sim).to_networkx()
    out_dist = MetricDistanceFilter(mode="distance").apply(G_dist).to_networkx()

    assert set(out_sim.edges()) == set(out_dist.edges())
    assert set(out_dist.edges()) == {(0, 1), (1, 2)}
