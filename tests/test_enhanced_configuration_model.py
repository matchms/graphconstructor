import numpy as np
import pytest
import scipy.sparse as sp
from graphconstructor import Graph
from graphconstructor.operators import EnhancedConfigurationModelFilter
from graphconstructor.operators.enhanced_configuration_model import _neg_log_likelihood, _neg_log_likelihood_grad


def _csr(*, data, rows, cols, n):
    """Small helper to build CSR adjacency."""
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def _finite_difference_gradient(fun, x, y, k, s, h=1e-7):
    """Central-difference gradient for x and y separately."""
    grad_x = np.empty_like(x, dtype=np.float64)
    grad_y = np.empty_like(y, dtype=np.float64)

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad_x[i] = (fun(x_plus, y, k, s) - fun(x_minus, y, k, s)) / (2.0 * h)

    for i in range(len(y)):
        y_plus = y.copy()
        y_minus = y.copy()
        y_plus[i] += h
        y_minus[i] -= h
        grad_y[i] = (fun(x, y_plus, k, s) - fun(x, y_minus, k, s)) / (2.0 * h)

    return grad_x, grad_y


@pytest.fixture
def small_undirected_graph():
    """A small symmetric weighted similarity graph."""
    A = _csr(
        data=[4, 4, 2, 2, 1, 1],
        rows=[0, 1, 0, 2, 1, 2],
        cols=[1, 0, 2, 0, 2, 1],
        n=3,
    )
    return Graph.from_csr(A, directed=False, weighted=True, mode="similarity")


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


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.5, 1.0])
def test_ecm_output_graph_basic_invariants_undirected(small_undirected_graph, alpha):
    np.random.seed(0)
    op = EnhancedConfigurationModelFilter(alpha=alpha)
    Gp = op.apply(small_undirected_graph)

    assert isinstance(Gp, Graph)
    assert Gp.n_nodes == small_undirected_graph.n_nodes
    assert Gp.directed is False
    assert Gp.weighted is True
    assert Gp.mode == "similarity"

    Wp = Gp.adj.tocsr()

    # No self-loops in output
    assert np.allclose(Wp.diagonal(), 0.0)

    # Output must remain symmetric
    assert (Wp != Wp.T).nnz == 0


def test_ecm_ignores_input_self_loops():
    A = _csr(
        data=[10, 4, 4],
        rows=[0, 0, 1],
        cols=[0, 1, 0],
        n=2,
    )
    A = A + A.T

    G = Graph.from_csr(A.tocsr(), directed=False, weighted=True, mode="similarity")

    np.random.seed(0)
    op = EnhancedConfigurationModelFilter()
    Gp = op.apply(G)

    Wp = Gp.adj.tocsr()
    assert np.allclose(Wp.diagonal(), 0.0)


def test_ecm_output_edges_are_subset_of_input_edges(small_undirected_graph):
    """
    Thresholding may remove observed edges, but must never introduce new ones.
    """
    np.random.seed(0)
    op = EnhancedConfigurationModelFilter(alpha=0.5)
    Gp = op.apply(small_undirected_graph)

    Win = small_undirected_graph.adj.tocsr()
    Wout = Gp.adj.tocsr()

    in_mask = (Win != 0).astype(np.int8)
    out_mask = (Wout != 0).astype(np.int8)

    diff = out_mask - in_mask
    assert diff.nnz == 0 or diff.max() <= 0


def test_ecm_alpha_one_keeps_all_existing_edges(small_undirected_graph):
    """
    Since ECM p-values should lie in [0, 1], alpha=1 should keep all observed
    off-diagonal edges.
    """
    np.random.seed(0)
    op = EnhancedConfigurationModelFilter(alpha=1.0)
    Gp = op.apply(small_undirected_graph)

    Win = small_undirected_graph.adj.tocsr().copy()
    Win = Win - sp.diags(Win.diagonal())
    Wout = Gp.adj.tocsr()

    in_mask = (Win != 0).astype(np.int8)
    out_mask = (Wout != 0).astype(np.int8)

    assert (in_mask != out_mask).nnz == 0


@pytest.mark.parametrize("alpha", [1e-6, 1e-3, 1e-2])
def test_ecm_smaller_alpha_cannot_increase_number_of_edges(small_undirected_graph, alpha):
    np.random.seed(0)
    G_loose = EnhancedConfigurationModelFilter(alpha=1.0).apply(small_undirected_graph)
    np.random.seed(0)
    G_strict = EnhancedConfigurationModelFilter(alpha=alpha).apply(small_undirected_graph)

    assert G_strict.adj.nnz <= G_loose.adj.nnz


def test_ecm_default_keeps_original_weights_for_retained_edges(small_undirected_graph):
    np.random.seed(0)
    op = EnhancedConfigurationModelFilter(alpha=1.0, replace_weights_by_p_values=False)
    Gp = op.apply(small_undirected_graph)

    Win = small_undirected_graph.adj.tocsr()
    Wout = Gp.adj.tocsr()

    # With alpha=1, all off-diagonal edges should remain, and the weights
    # should match the original weights.
    assert np.array_equal(Wout.indices, Win.indices)
    assert np.array_equal(Wout.indptr, Win.indptr)
    assert np.allclose(Wout.data, Win.data, rtol=1e-8, atol=1e-10)


def test_ecm_can_replace_weights_by_p_values(small_undirected_graph):
    np.random.seed(0)
    op = EnhancedConfigurationModelFilter(alpha=1.0, replace_weights_by_p_values=True)
    Gp = op.apply(small_undirected_graph)

    Wp = Gp.adj.tocsr()

    # Retained edge weights now represent p-values
    if Wp.nnz > 0:
        assert np.nanmin(Wp.data) >= -1e-12
        assert np.nanmax(Wp.data) <= 1.0 + 1e-12

    # In general these should differ from the original weights
    Win = small_undirected_graph.adj.tocsr()
    assert not np.allclose(Wp.data, Win.data)


def test_ecm_reproducible_given_fixed_seed(small_undirected_graph):
    op = EnhancedConfigurationModelFilter(alpha=0.5, replace_weights_by_p_values=True)

    np.random.seed(123)
    Gp1 = op.apply(small_undirected_graph)
    W1 = Gp1.adj.tocsr()

    np.random.seed(123)
    Gp2 = op.apply(small_undirected_graph)
    W2 = Gp2.adj.tocsr()

    assert np.array_equal(W1.indices, W2.indices)
    assert np.array_equal(W1.indptr, W2.indptr)
    assert np.allclose(W1.data, W2.data, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.1])
def test_ecm_rejects_invalid_alpha(small_undirected_graph, alpha):
    op = EnhancedConfigurationModelFilter(alpha=alpha)
    with pytest.raises(ValueError):
        op.apply(small_undirected_graph)


@pytest.mark.parametrize("n", [3, 5])
def test_neg_log_likelihood_grad_matches_finite_difference(n):
    """
    Check that the analytic gradient of the ECM negative log-likelihood
    matches a numerical central-difference approximation.
    """
    rng = np.random.default_rng(12345)

    x = rng.uniform(0.3, 2.0, size=n).astype(np.float64)
    y = rng.uniform(0.2, 0.8, size=n).astype(np.float64)
    k = rng.uniform(0.5, 5.0, size=n).astype(np.float64)
    s = rng.uniform(0.5, 5.0, size=n).astype(np.float64)

    grad_x_analytic, grad_y_analytic = _neg_log_likelihood_grad(x, y, k, s)
    grad_x_fd, grad_y_fd = _finite_difference_gradient(
        _neg_log_likelihood, x, y, k, s, h=1e-7
    )

    np.testing.assert_allclose(grad_x_analytic, grad_x_fd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(grad_y_analytic, grad_y_fd, rtol=1e-6, atol=1e-7)


def test_neg_log_likelihood_grad_returns_finite_arrays():
    x = np.array([0.7, 1.2, 1.8], dtype=np.float64)
    y = np.array([0.2, 0.4, 0.6], dtype=np.float64)
    k = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    s = np.array([1.5, 2.5, 3.5], dtype=np.float64)

    grad_x, grad_y = _neg_log_likelihood_grad(x, y, k, s)

    assert grad_x.shape == x.shape
    assert grad_y.shape == y.shape
    assert np.all(np.isfinite(grad_x))
    assert np.all(np.isfinite(grad_y))


@pytest.mark.parametrize("alphas", [
    [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
])
def test_ecm_retained_edges_monotone_in_alpha(small_undirected_graph, alphas):
    """
    Add mock-based tests to verify that the alpha thresholding logic in ECM's apply() method
    correctly retains edges based on the p-values computed from the optimization output.
    """
    counts = []
    for alpha in alphas:
        np.random.seed(123)
        Gp = EnhancedConfigurationModelFilter(alpha=alpha).apply(small_undirected_graph)
        # undirected graph stores both directions explicitly
        counts.append(Gp.adj.nnz)

    assert counts == sorted(counts)


def test_ecm_alpha_thresholding_exact_edge_counts(monkeypatch):
    A = _csr(
        data=[4, 4, 2, 2, 1, 1],
        rows=[0, 1, 0, 2, 1, 2],
        cols=[1, 0, 2, 0, 2, 1],
        n=3,
    )
    G = Graph.from_csr(A, directed=False, weighted=True, mode="similarity")

    # Lower-triangle edges of this graph are:
    # (1,0), (2,0), (2,1)
    fake_pvals = np.array([0.001, 0.02, 0.2], dtype=np.float64)

    def fake_pval_matrix_data(x, y, row, col, weights):
        assert len(weights) == 3
        return fake_pvals.copy()

    def fake_make_objective(*args, **kwargs):
        def fun(v):
            return 0.0
        def jac(v):
            return np.zeros_like(v)
        return fun, jac

    class FakeResult:
        success = True
        message = "ok"
        x = np.zeros(2 * G.n_nodes, dtype=np.float64)

    def fake_minimize(*args, **kwargs):
        return FakeResult()

    import graphconstructor.operators.enhanced_configuration_model as ecm_mod

    monkeypatch.setattr(ecm_mod, "_pval_matrix_data", fake_pval_matrix_data)
    monkeypatch.setattr(ecm_mod, "_make_objective", fake_make_objective)
    monkeypatch.setattr(ecm_mod.so, "minimize", fake_minimize)

    cases = [
        (0.0005, 0),  # keep none
        (0.01,   2),  # keep one undirected edge -> 2 stored entries
        (0.05,   4),  # keep two undirected edges -> 4 stored entries
        (0.5,    6),  # keep all three undirected edges -> 6 stored entries
    ]

    for alpha, expected_nnz in cases:
        op = EnhancedConfigurationModelFilter(alpha=alpha)
        Gp = op.apply(G)
        assert Gp.adj.nnz == expected_nnz
