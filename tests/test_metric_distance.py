import numpy as np
import networkx as nx
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

def test_basic_undirected_filtering():
    G0 = simple_undirected_graph()

    out = MetricDistanceFilter(distortion=False, verbose=False).apply(G0)

    assert isinstance(out, Graph)
    assert out.directed == False
    assert out.weighted == True

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
    G0 = simple_undirected_graph()
    out = MetricDistanceFilter().apply(G0)
    assert out is None

def test_edge_removal_logic():
    G0 = simple_undirected_graph()
    out = MetricDistanceFilter().apply(G0)

    original_nx = G0.to_networkx()
    out_nx = G0.to_networkx()

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

