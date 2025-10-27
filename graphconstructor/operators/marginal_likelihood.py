from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.stats import binom
from ..graph import Graph
from .base import GraphOperator


@dataclass(slots=True)
class MarginalLikelihoodFilter(GraphOperator):
    """
    Marginal Likelihood Filter (MLF) as in:
    Dianati, N. (2016). "Unwinding the hairball graph: Pruning algorithms for weighted complex networks."
    Physical Review E 93, 012304.

    Keeps edge (i,j) iff p-value s_ij(w_ij) = P[σ_ij >= w_ij | degrees, T] <= alpha,
    under the null model that preserves node strengths on average.

    The original algorithm assumes integer weights (counts). If your graph has
    float weights, those are converted to integers by scaling and rounding.

    - Undirected case:
        p = (k_i * k_j) / (2 T^2),  where T = 0.5 * sum_i k_i   (strengths)
    - Directed case:
        p = (k_out_i * k_in_j) / (T^2),  where T = sum_i k_out_i

    Parameters
    ----------
    alpha : float
        Significance threshold. Keep edges with p-value <= alpha.
    float_scaling : float, default 20.0
        Edge weights are assumed to be (nonnegative) counts. If your Graph stores
        floats, then those will be converted by scaling and rounding to integers
        between 0 and float_scaling. Default is 20.
    assume_loopless : bool, default False
        If True, self-loops are excluded from testing. Graph already drops
        self-loops by construction; this flag is here for clarity and future loopless
        corrections. Currently it only ensures i != j in filtering.
    copy_meta : bool, default True
        If True, copies metadata frame; set to False to reuse reference.

    Notes
    -----
    - Numerical tail computation uses scipy.stats.binom.sf(w-1, T, p), which equals
      sum_{m >= w} C(T,m) p^m (1-p)^{T-m}.
    - For very large T, this is still efficient and stable compared to naive summation.
    - We only test existing edges (sparse support). No densification.
    - Undirected graphs are treated on the UPPER TRIANGLE only and mirrored back.
    """
    alpha: float
    float_scaling: float = 20.
    assume_loopless: bool = False
    copy_meta: bool = True
    supported_modes = ["similarity"]

    def _cast_weights_to_int(self, w: np.ndarray, max_weight=None) -> np.ndarray:
        """Map floats to integers by scaling and rounding."""
        if np.allclose(w, np.rint(w)):
            # Already integers
            return np.rint(w).astype(np.int32, copy=False)
        elif max_weight is not None:
            wi = np.round(self.float_scaling * w)
        else:
            raise ValueError("Invalid float_scaling option.")
        return wi.clip(min=0).astype(np.int32, copy=False)

    def _undirected_filter(self, G: Graph) -> Graph:
        A = G.adj.tocsr(copy=False)

        # strengths (row sums) — for symmetric undirected, row sum == strength
        k = np.asarray(A.sum(axis=1)).ravel()
        T = 0.5 * float(k.sum())  # total unit edges
        if T <= 0:
            # degenerate: no edges to keep
            return Graph.from_csr(A.copy() * 0.0, directed=False, weighted=G.weighted,
                                  mode=G.mode,
                                  meta=G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta,
                                  sym_op="max")

        # Work on upper triangle to avoid double-testing undirected pairs
        Au = sp.triu(A, k=1).tocoo()
        if Au.nnz == 0:
            return Graph.from_csr(A.copy() * 0.0, directed=False, weighted=G.weighted,
                                  mode=G.mode,
                                  meta=G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta,
                                  sym_op="max")

        # integerized realized weights
        w = self._cast_weights_to_int(Au.data, max_weight=Au.data.max())

        # p = k_i k_j / (2 T^2)
        ki = k[Au.row]
        kj = k[Au.col]
        p = (ki * kj) / (2.0 * (T ** 2))
        # clip to [0,1] numerically
        p = np.clip(p, 0.0, 1.0)

        # p-value s_ij(w_ij) = P[σ >= w_ij] = sf(w_ij - 1)
        # sf expects n=T (integer) and k integer, but T can be large float -> cast
        n = int(round(T))
        # To be conservative if rounding T: we can keep <wT as integer rounding; T comes from counts anyway.
        pvals = binom.sf(w - 1, n=n, p=p)

        keep = pvals <= self.alpha

        # Build filtered symmetric adjacency
        rows = Au.row[keep]
        cols = Au.col[keep]
        data = Au.data[keep] if G.weighted else np.ones(keep.sum(), dtype=float)

        # mirror to both (i,j) and (j,i)
        rows_full = np.concatenate([rows, cols])
        cols_full = np.concatenate([cols, rows])
        data_full = np.concatenate([data, data])

        A_f = sp.csr_matrix((data_full, (rows_full, cols_full)), shape=A.shape)
        return Graph.from_csr(A_f, directed=False, weighted=G.weighted,
                              mode=G.mode,
                              meta=G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta,
                              sym_op="max")

    def _directed_filter(self, G: Graph) -> Graph:
        A = G.adj.tocsr(copy=False)

        kout = np.asarray(A.sum(axis=1)).ravel()
        kin = np.asarray(A.sum(axis=0)).ravel()
        T = float(kout.sum())
        if T <= 0:
            return Graph.from_csr(A.copy() * 0.0, directed=True, weighted=G.weighted,
                                  mode=G.mode,
                                  meta=G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta)

        coo = A.tocoo()
        if coo.nnz == 0:
            return Graph.from_csr(A.copy() * 0.0, directed=True, weighted=G.weighted,
                                  mode=G.mode,
                                  meta=G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta)

        # Optionally exclude self-edges (currently Graph anyway drops them)
        if self.assume_loopless:
            mask = coo.row != coo.col
            coo = sp.coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=A.shape)

        w = self._cast_weights_to_int(coo.data, max_weight=coo.data.max())

        pi = kout[coo.row]
        pj = kin[coo.col]
        p = (pi * pj) / (T ** 2)
        p = np.clip(p, 0.0, 1.0)

        n = int(round(T))
        pvals = binom.sf(w - 1, n=n, p=p)
        keep = pvals <= self.alpha

        rows = coo.row[keep]
        cols = coo.col[keep]
        data = coo.data[keep] if G.weighted else np.ones(keep.sum(), dtype=float)

        A_f = sp.csr_matrix((data, (rows, cols)), shape=A.shape)
        return Graph.from_csr(A_f, directed=True, weighted=G.weighted,
                              mode=G.mode,
                              meta=G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta)

    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)
        if G.directed:
            return self._directed_filter(G)
        else:
            return self._undirected_filter(G)
