from dataclasses import dataclass
import numpy as np
import scipy.optimize as so
import scipy.sparse as sp
from numba import njit
from ..graph import Graph
from .base import GraphOperator


EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# R <=> (0, inf) homeomorphisms
# ---------------------------------------------------------------------------

@njit(cache=True)
def _exp(x):
    return np.exp(x)

@njit(cache=True)
def _log(x):
    return np.log(x)

@njit(cache=True)
def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)  # avoids overflows for large x

@njit(cache=True)
def _softplus_inv(x):
    return np.log(np.expm1(x))  # avoids unstability near 0

R_to_zero_to_inf = [
    (_exp, _log),
    (_softplus, _softplus_inv),
]


# ---------------------------------------------------------------------------
# R <=> (0, 1) homeomorphisms
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@njit(cache=True)
def _sigmoid_inv(x):
    return -np.log(1.0 / x - 1.0)

@njit(cache=True)
def _shift_scale_arctan(x):
    return (1.0 / np.pi) * np.arctan(x) + 0.5

@njit(cache=True)
def _shift_scale_arctan_inv(x):
    return np.tan(np.pi * (x - 0.5))

R_to_zero_to_one = [
    (_sigmoid, _sigmoid_inv),
    (_shift_scale_arctan, _shift_scale_arctan_inv),
]


# ---------------------------------------------------------------------------
# Core numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _neg_log_likelihood(x, y, k, s):
    """
    Negative log-likelihood of the ECM model (upper-triangle sum).

    Parameters are already in bounded form (x > 0, 0 < y < 1).
    Implements formula (13) from https://arxiv.org/abs/1706.00230.
    """
    N = len(x)

    llhood = 0.0
    for i in range(N):
        llhood += k[i] * np.log(x[i])
        llhood += s[i] * np.log(y[i])

    # Sum log(t_ij) for i < j  (upper triangle = unique pairs)
    for i in range(N):
        for j in range(i):
            xx = x[i] * x[j]
            yy = y[i] * y[j]
            t = (1.0 - yy) / (1.0 - yy + xx * yy)
            llhood += np.log(t)

    return -llhood


@njit(cache=True)
def _neg_log_likelihood_grad(x, y, k, s):
    """
    Analytic gradient of the negative log-likelihood w.r.t. the *bounded*
    parameters (x, y). Derived by differentiating formula (13).
    Parameters should be in their bounded form (x > 0, 0 < y < 1) for correct gradients.
    """
    N = len(x)
    grad_x = np.empty(N, dtype=np.float64)
    grad_y = np.empty(N, dtype=np.float64)

    # Derivatives of the node-wise terms:
    #  - sum_i k_i log(x_i)  ->  -k_i / x_i
    #  - sum_i s_i log(y_i)  ->  -s_i / y_i
    for i in range(N):
        grad_x[i] = -k[i] / x[i]
        grad_y[i] = -s[i] / y[i]

    # Pair contributions for i>j
    for i in range(N):
        for j in range(i):
            yy = y[i] * y[j]
            xx = x[i] * x[j]
            D = 1.0 - yy + xx * yy

            # x-grad from +log(D)
            # d/dx_i log(D) = (yy * x_j)/D
            grad_x[i] += (yy * x[j]) / D
            grad_x[j] += (yy * x[i]) / D

            # y-grad from -log(1-yy) + log(D)
            # First part:
            #   d/dy_i [-log(1 - yy)] = y_j / (1 - yy)
            #   d/dy_j [-log(1 - yy)] = y_i / (1 - yy)
            d_minus_log1myy_dyi = y[j] / (1.0 - yy)
            d_minus_log1myy_dyj = y[i] / (1.0 - yy)

            # Second part:
            #   dD/dy_i = y_j (xx - 1)
            #   dD/dy_j = y_i (xx - 1)
            d_logD_dyi = y[j] * (xx - 1.0) / D
            d_logD_dyj = y[i] * (xx - 1.0) / D

            grad_y[i] += d_minus_log1myy_dyi + d_logD_dyi
            grad_y[j] += d_minus_log1myy_dyj + d_logD_dyj

    return grad_x, grad_y


@njit(cache=True)
def _pval_matrix_data(x, y, row, col, weights):
    """
    Compute p-values for each entry in the lower triangle of the weight matrix.

    p_val = p_ij * (y_i * y_j)^(w - 1)   [formula (15)]

    Returns an array of p-values aligned with (row, col, weights).
    Only entries where row > col (lower triangle) are meaningful; upper-triangle
    indices are skipped to avoid double work.
    """
    n = len(weights)
    out = np.empty(n)
    for k in range(n):
        i = row[k]
        j = col[k]
        w = weights[k]
        xx = x[i] * x[j]
        yy = y[i] * y[j]
        pij = xx * yy / (1.0 - yy + xx * yy)
        out[k] = pij * (y[i] * y[j]) ** (w - 1.0)
    return out


# ---------------------------------------------------------------------------
# Thin wrapper: reparameterised objective for scipy
# ---------------------------------------------------------------------------

def _make_objective(num_nodes, k, s, x_transform, x_inv_transform,
                    y_transform, y_inv_transform):
    """
    Return (fun, jac) callables in *unconstrained* space for scipy.optimize.

    The reparameterisation (transform_parameters) maps R^2N → (0,∞)^N × (0,1)^N,
    so no explicit bounds are needed by the solver — domain constraints are
    enforced implicitly, matching the design of the original MaxentGraph.solve().
    """

    def _transform(v):
        """Clip and transform unconstrained v to bounded (x, y) for the objective."""
        x_raw = v[:num_nodes]
        y_raw = v[num_nodes:]

        x = x_transform(x_raw)
        y = y_transform(y_raw)

        x = np.maximum(x, EPS)
        y = np.clip(y, EPS, 1.0 - EPS)

        return x, y

    def fun(v):
        x, y = _transform(v)
        return float(_neg_log_likelihood(x, y, k, s))

    def jac(v):
        x, y = _transform(v)
        gx_bounded, gy_bounded = _neg_log_likelihood_grad(x, y, k, s)

        # Chain rule: dL/dv = dL/dz * dz/dv
        # Approximates dz/dv numerically for the transform Jacobian diagonal,
        # using a small finite difference.
        h = np.sqrt(EPS)
        v_raw_x = v[:num_nodes]
        v_raw_y = v[num_nodes:]

        dx_dv = (x_transform(v_raw_x + h) - x_transform(v_raw_x - h)) / (2 * h)
        dy_dv = (y_transform(v_raw_y + h) - y_transform(v_raw_y - h)) / (2 * h)

        grad_v = np.concatenate([gx_bounded * dx_dv, gy_bounded * dy_dv])
        return grad_v

    return fun, jac


# ---------------------------------------------------------------------------
# Graph operator
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EnhancedConfigurationModelFilter(GraphOperator):
    """
    Filter an undirected weighted similarity graph using the Enhanced
    Configuration Model (ECM).

    This operator fits the ECM null model to the input graph and returns a new
    graph whose edge weights are the ECM p-values associated with the observed
    edge weights. The model preserves, in expectation, both the degree sequence
    and the strength sequence of the input graph.

    The implementation follows the maximum-likelihood formulation of the ECM
    for weighted undirected networks. Internally, it estimates the node-wise
    model parameters ``x`` and ``y`` by minimizing the negative log-likelihood
    in a reparameterized unconstrained space and then computes an ECM-based
    p-value for each observed edge.

    Parameters
    ----------
    x_transform_idx : int, default=0
        Index selecting the reparameterization used for the positive ECM
        parameters ``x``. The index refers to :data:`R_to_zero_to_inf`, whose
        entries map from the real line to the open interval ``(0, inf)``.

    y_transform_idx : int, default=0
        Index selecting the reparameterization used for the bounded ECM
        parameters ``y``. The index refers to :data:`R_to_zero_to_one`, whose
        entries map from the real line to the open interval ``(0, 1)``.

    Paper: https://arxiv.org/abs/1706.00230
    Code: https://gitlab.liris.cnrs.fr/coregraphie/aliplosone/-/blob/main/Backbones/ecm.py
    """
    supported_modes = ["similarity"]

    # Reparameterization choices for the ECM variables:
    # x: R -> (0, inf)
    # y: R -> (0, 1)
    x_transform_idx: int = 0
    y_transform_idx: int = 0

    def _directed(self, G: Graph) -> Graph:
        raise NotImplementedError(
            "EnhancedConfigurationModelFilter is defined only for undirected graphs."
        )

    def _undirected(self, G: Graph) -> Graph:
        W = G.adj.copy().tocsr()
        W -= sp.diags(W.diagonal())

        k = np.asarray((W > 0).sum(axis=1)).ravel().astype(np.float64)
        s = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)

        num_nodes = G.n_nodes

        x_transform, x_inv_transform = R_to_zero_to_inf[self.x_transform_idx]
        y_transform, y_inv_transform = R_to_zero_to_one[self.y_transform_idx]

       # ---- Initial guess (option 5 from original) ----------------------
        num_edges = k.sum() / 2.0
        x0_bounded = k / np.sqrt(max(num_edges, EPS))
        y0_bounded = np.random.random(num_nodes)

        # Clip to valid domain before applying inverse transform
        lower = np.full(2 * num_nodes, EPS)
        upper = np.concatenate([np.full(num_nodes, np.inf),
                                 np.full(num_nodes, 1.0 - EPS)])
        v0_bounded = np.clip(np.concatenate([x0_bounded, y0_bounded]),
                             lower, upper)

        v0 = np.concatenate([
            x_inv_transform(v0_bounded[:num_nodes]),
            y_inv_transform(v0_bounded[num_nodes:]),
        ])

        # ---- Objective + gradient in unconstrained space ---------
        fun, jac = _make_objective(
            num_nodes, k, s,
            x_transform, x_inv_transform,
            y_transform, y_inv_transform,
        )

        # ---- Optimize in reparameterized space -----------------------------
        # The optimizer works on unconstrained variables; valid ECM parameter
        # domains are enforced by the x/y transforms before each evaluation.
        res = so.minimize(
            fun=fun,
            jac=jac,
            x0=v0,
            method="L-BFGS-B",
        )

        if not res.success:
            import warnings
            warnings.warn(
                f"ECM optimisation did not converge: {res.message}",
                RuntimeWarning,
            )

        # ---- p-value matrix ----------------------------------------------
        x_opt = x_transform(res.x[:num_nodes])
        y_opt = y_transform(res.x[num_nodes:])

        # Work only on the lower triangle to avoid double computation
        W_lower = sp.tril(W, k=-1).tocoo()
        row = W_lower.row.astype(np.int64)
        col = W_lower.col.astype(np.int64)
        weights = W_lower.data

        pvals = _pval_matrix_data(x_opt, y_opt, row, col, weights)

        W_p = sp.csr_matrix(
            (pvals, (row, col)),
            shape=(num_nodes, num_nodes),
            dtype=np.float64,
        )
        # Symmetrise (ECM is undirected)
        W_p = W_p + W_p.T

        # Convert back to Graph
        return Graph.from_csr(
            W_p,
            mode="similarity",
            weighted=True,
            directed=False
        )
    
    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)
        if G.directed:
            return self._directed(G)
        else:
            return self._undirected(G)
