from dataclasses import dataclass
import numpy as np
import scipy.sparse
import scipy.optimize
from ..graph import Graph
from .base import GraphOperator
from jax import jit
import jax.numpy as jnp
import jax


# utils -----------------------------------------------------------------
EPS = np.finfo(float).eps

### R <=> (0, inf) homeomorphisms
@jit
def softplus_inv(x):
    return jnp.log(jnp.exp(x) - 1)

R_to_zero_to_inf = [(jit(jnp.exp), jit(jnp.log)), (jit(jax.nn.softplus), softplus_inv)]

### R <=> (0,1) homeomorphisms
@jit
def shift_scale_arctan(x):
    # scaled, shifted arctan
    return (1 / jnp.pi) * jnp.arctan(x) + 1 / 2


@jit
def shift_scale_arctan_inv(x):
    return jnp.tan(jnp.pi * (x - 1 / 2))


@jit
def sigmoid_inv(x):
    return -jnp.log(1 / x - 1)

R_to_zero_to_one = [
    (jit(jax.nn.sigmoid), sigmoid_inv),
    (shift_scale_arctan, shift_scale_arctan_inv),
]

# Enhanced Configuration Model Filter ------------------------------------------------

@dataclass(slots=True)
class EnhancedConfigurationModelFilter(GraphOperator):
    """
    Enhanced Configuration Model (ECM) filter for weighted, undirected
    similarity graphs.

    Paper: https://arxiv.org/abs/1706.00230
    Code: https://gitlab.liris.cnrs.fr/coregraphie/aliplosone/-/blob/main/Backbones/ecm.py?ref_type=heads
    """
    supported_modes = ["similarity"]

    @staticmethod
    def _transform_parameters(num_nodes, x_transform, y_transform, v):
        x = v[:num_nodes]
        y = v[num_nodes:]
        return np.concatenate((x_transform(x), y_transform(y)))

    @staticmethod
    def _neg_log_likelihood(num_nodes, k, s, x_transform, y_transform, v):
        # Formel (13)
        z = EnhancedConfigurationModelFilter._transform_parameters(num_nodes, x_transform, y_transform, v)

        x = z[:num_nodes]
        y = z[num_nodes:]

        llhood = 0.0
        llhood += np.sum(k * np.log(x))
        llhood += np.sum(s * np.log(y))

        xx = np.outer(x, x)
        yy = np.outer(y, y)

        t = (1 - yy) / (1 - yy + xx * yy)
        log_t = np.log(t)
        llhood += np.sum(log_t) - np.sum(np.tril(log_t))

        return -llhood
    
    def _directed(self, G: Graph) -> Graph:
        raise NotImplementedError(
            "EnhancedConfigurationModelFilter is defined only for undirected graphs."
        )

    def _undirected(self, G: Graph) -> Graph:
        W = G.adj.copy().tocsr()

        W -= scipy.sparse.diags(W.diagonal())

        k = (W > 0).sum(axis=1).A1.astype("float64")
        s = W.sum(axis=1).A1

        num_nodes = G.n_nodes

        x_transform, x_inv_transform = R_to_zero_to_inf[0]
        y_transform, y_inv_transform = R_to_zero_to_one[0]

        # -------- Initial Guess --------
        # initial guess option 5
        num_edges = np.sum(k) / 2
        v0 = np.concatenate([
            k / np.sqrt(num_edges + EPS),
            np.random.random(num_nodes)
        ])

        v0 = np.concatenate((
            x_inv_transform(v0[:num_nodes]),
            y_inv_transform(v0[num_nodes:])
        ))

        # -------- Bounds --------
        # bounds()
        lower_bounds = np.array([EPS] * (2 * num_nodes))
        upper_bounds = np.array([np.inf] * num_nodes + [1 - EPS] * num_nodes)
        bounds = scipy.optimize.Bounds(lower_bounds, upper_bounds)

        # -------- Optimierung --------
        # solve()
        res = scipy.optimize.minimize(
            fun=lambda v: np.array(self._neg_log_likelihood(num_nodes, k, s, x_transform, y_transform, v)),
            x0=np.array(v0),
            bounds=bounds,
            method="L-BFGS-B"
        )

        v_opt = res.x

        # -------- p-value Matrix --------
        # get_pval_matrix()
        # pij: Formel (9)
        # p_val: Formel (15)
        z = self._transform_parameters(num_nodes, x_transform, y_transform, v_opt)

        x = z[:num_nodes]
        y = z[num_nodes:]
        
        W_p = scipy.sparse.tril(W.copy()).tolil().astype(float)

        for i, j in zip(*W.nonzero()):
            w = W[i, j]
            xx_out = x[i] * x[j]
            yy_out = y[i] * y[j]

            pij = xx_out * yy_out / (1 - yy_out + xx_out * yy_out)
            p_val = pij * (y[i] * y[j]) ** (w - 1)

            W_p[i, j] = p_val

        W_p = W_p.tocsr()

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

