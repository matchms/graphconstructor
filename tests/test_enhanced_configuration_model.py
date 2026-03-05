import numpy as np
import pytest
import scipy.sparse as sp
from graphconstructor import Graph
from graphconstructor.operators import EnhancedConfigurationModelFilter


def _csr(*, data, rows, cols, n):
    """Small helper to build CSR adjacency."""
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def test_ecm_rejects_directed_graph():
    A = _csr(
        data=[4, 1, 3, 2],
        rows=[0, 1, 2, 0],
        cols=[1, 2, 0, 2],
        n=3,
    )
    G = Graph.from_csr(A, directed=True, weighted=True, mode="similarity")

    op = EnhancedConfigurationModelFilter()

    with pytest.raises(NotImplementedError):
        op.apply(G)


def test_ecm_output_graph_basic_invariants_undirected():
    # Make an undirected weighted graph
    # (store symmetric entries explicitly)
    A = _csr(
        data=[4, 4, 2, 2, 1, 1],
        rows=[0, 1, 0, 2, 1, 2],
        cols=[1, 0, 2, 0, 2, 1],
        n=3,
    )
    G = Graph.from_csr(A, directed=False, weighted=True, mode="similarity")

    np.random.seed(0)  # for deterministic init guess in your implementation
    op = EnhancedConfigurationModelFilter()
    Gp = op.apply(G)

    assert isinstance(Gp, Graph)
    assert Gp.n_nodes == G.n_nodes
    assert Gp.directed is False
    assert Gp.weighted is True
    assert Gp.mode == "similarity"

    Wp = Gp.adj.tocsr()

    # Diagonal must be zero (self-loops ignored)
    assert np.allclose(Wp.diagonal(), 0.0)

    # P-values should be in [0, 1] (allow tiny numerical noise)
    if Wp.nnz > 0:
        vals = Wp.data
        assert np.nanmin(vals) >= -1e-12
        assert np.nanmax(vals) <= 1.0 + 1e-12


def test_ecm_ignores_input_self_loops():
    # Create undirected graph with a self-loop on node 0
    A = _csr(
        data=[10, 4, 4],
        rows=[0, 0, 1],
        cols=[0, 1, 0],
        n=2,
    )
    # Mirror for undirected adjacency explicitly
    A = A + A.T

    G = Graph.from_csr(A.tocsr(), directed=False, weighted=True, mode="similarity")

    np.random.seed(0)
    op = EnhancedConfigurationModelFilter()
    Gp = op.apply(G)

    Wp = Gp.adj.tocsr()
    assert np.allclose(Wp.diagonal(), 0.0)


def test_ecm_only_modifies_existing_edges_sparsity_subset():
    """
    ECM p-values are computed only for observed edges (nonzeros of W),
    so the output should not introduce brand-new edges where input had none.
    (It may drop some, but shouldn't add new ones.)
    """
    A = _csr(
        data=[3, 3, 2, 2],
        rows=[0, 1, 1, 2],
        cols=[1, 0, 2, 1],
        n=3,
    )
    G = Graph.from_csr(A, directed=False, weighted=True, mode="similarity")

    np.random.seed(0)
    op = EnhancedConfigurationModelFilter()
    Gp = op.apply(G)

    Win = G.adj.tocsr()
    Wout = Gp.adj.tocsr()

    in_mask = (Win != 0).astype(np.int8)
    out_mask = (Wout != 0).astype(np.int8)

    # Output nonzeros must be a subset of input nonzeros:
    # i.e., (out_mask - in_mask) should have no positive entries
    diff = out_mask - in_mask
    assert diff.nnz == 0 or diff.max() <= 0


def test_ecm_reproducible_given_fixed_seed():
    A = _csr(
        data=[4, 4, 2, 2, 1, 1],
        rows=[0, 1, 0, 2, 1, 2],
        cols=[1, 0, 2, 0, 2, 1],
        n=3,
    )
    G = Graph.from_csr(A, directed=False, weighted=True, mode="similarity")
    op = EnhancedConfigurationModelFilter()

    np.random.seed(123)
    Gp1 = op.apply(G)
    W1 = Gp1.adj.tocsr()

    np.random.seed(123)
    Gp2 = op.apply(G)
    W2 = Gp2.adj.tocsr()

    # Compare sparse matrices exactly in structure and approximately in values.
    assert (W1 != 0).nnz == (W2 != 0).nnz
    assert np.array_equal(W1.indices, W2.indices)
    assert np.array_equal(W1.indptr, W2.indptr)
    assert np.allclose(W1.data, W2.data, rtol=1e-8, atol=1e-10)
