import numpy as np
import pytest
import scipy.sparse as sp
from graphconstructor import Graph
from graphconstructor.operators import EnhancedConfigurationModelFilter
from graphconstructor.operators.enhanced_configuration_model import _neg_log_likelihood, _neg_log_likelihood_grad


def _csr(*, data, rows, cols, n):
    """Small helper to build CSR adjacency."""
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def _central_diff_grad(f, z, h=1e-6):
    """Central-difference gradient for 1D numpy arrays."""
    g = np.zeros_like(z, dtype=np.float64)
    for i in range(z.size):
        zp = z.copy()
        zm = z.copy()
        zp[i] += h
        zm[i] -= h
        g[i] = (f(zp) - f(zm)) / (2.0 * h)
    return g


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


@pytest.mark.parametrize("N", [4, 6])
def test_neg_log_likelihood_grad_matches_finite_differences(N):
    rng = np.random.default_rng(0)

    # Make bounded parameters safely away from problematic edges
    x = rng.uniform(0.2, 2.0, size=N).astype(np.float64)     # x > 0
    y = rng.uniform(0.05, 0.95, size=N).astype(np.float64)   # 0 < y < 1

    # Dummy constraints (must be nonnegative; avoid zeros to keep logs happy)
    k = rng.integers(1, 5, size=N).astype(np.float64)
    s = rng.uniform(0.5, 5.0, size=N).astype(np.float64)

    # Wrap objective as function of concatenated z = [x, y]
    def f(z):
        xx = z[:N]
        yy = z[N:]
        return float(_neg_log_likelihood(xx, yy, k, s))

    z0 = np.concatenate([x, y])

    # Finite-diff reference
    g_fd = _central_diff_grad(f, z0, h=1e-6)

    # Analytic gradient under test
    gx, gy = _neg_log_likelihood_grad(x, y, k, s)
    g_an = np.concatenate([gx, gy])

    # Compare with reasonable tolerances (FD is noisy; tighten once stable)
    np.testing.assert_allclose(g_an, g_fd, rtol=1e-4, atol=1e-5)
