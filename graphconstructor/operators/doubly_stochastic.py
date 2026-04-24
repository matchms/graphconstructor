import warnings
from dataclasses import dataclass
import networkx as nx
import numpy as np
import scipy.sparse as sp
from ..graph import Graph
from .base import GraphOperator


@dataclass(slots=True)
class DoublyStochasticNormalize(GraphOperator):
    """
    Alternating row/column normalization (Sinkhorn-Knopp) to make the adjacency
    approximately doubly stochastic. Works with CSR without densifying.

    Convergence is guaranteed for strictly positive matrices (total support).
    For sparse graphs, zero rows/cols remain zero (scales stay at 1.0).

    References
    ----------
    - Coscia, M. (2025). "The Atlas for the Inspiring Network Scientist."
    - Sinkhorn, R. (1964). "A relationship between arbitrary positive matrices
      and doubly stochastic matrices." Ann. Math. Stat.

    Parameters
    ----------
    tolerance : float
        Band tolerance for the "both sums ~ 1" check, i.e. accept if each row/col
        sum is in [1 - tolerance, 1 + tolerance]. Default 1e-5.
    max_iter : int
        Maximum iterations. Default 10_000.
    copy_meta : bool
        Copy metadata frame if present. Default True.
    """

    tolerance: float = 1e-5
    max_iter: int = 10_000
    copy_meta: bool = True
    supported_modes = ["similarity"]

    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)
        A = G.adj.tocsr(copy=False)

        if A.shape[0] != A.shape[1]:
            raise TypeError("DoublyStochasticNormalize requires a square adjacency.")
        if (A.data < 0).any():
            raise ValueError("DoublyStochasticNormalize requires nonnegative weights.")

        n = A.shape[0]
        if A.nnz == 0:
            # trivial: nothing to normalize
            return Graph.from_csr(
                A.copy(),
                directed=G.directed,
                weighted=True,
                mode=G.mode,
                meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
                sym_op="max",
            )

        # Scaling vectors (1D)
        r = np.ones(n, dtype=float)
        c = np.ones(n, dtype=float)

        # Precompute nnz masks for convergence checks under sparsity
        row_has_edges = np.array(A.sum(axis=1) > 0).T[0]
        col_has_edges = np.array(A.sum(axis=0) > 0)[0]

        # Build transposed CSR once for repeated column-sum products
        A_T = A.T.tocsr(copy=False)

        min_thres = 1.0 - self.tolerance
        max_thres = 1.0 + self.tolerance

        # Parameter for stabilization
        MAX_FACTOR = 1e50

        converged = False
        for _ in range(self.max_iter):
            c[col_has_edges] = 1 / A_T.dot(r)[col_has_edges]
            r[row_has_edges] = 1 / A.dot(c)[row_has_edges]

            if np.any(np.abs(r) > MAX_FACTOR) or np.any(np.abs(c) > MAX_FACTOR):
                warnings.warn(
                    "DoublyStochasticNormalize stopped early because scaling factors "
                    "became very large. Result may not be doubly stochastic.",
                    RuntimeWarning,
                )
                break

            # Convergence check (band and tol)
            row_sums = r * (A.dot(c))
            col_sums = c * (A_T.dot(r))

            # Only check rows/cols that have edges (others stay 0 and are irrelevant)
            if row_has_edges.any():
                rows_ok = np.all(
                    (row_sums[row_has_edges] >= min_thres)
                    & (row_sums[row_has_edges] <= max_thres)
                )
            else:
                rows_ok = True

            if col_has_edges.any():
                cols_ok = np.all(
                    (col_sums[col_has_edges] >= min_thres)
                    & (col_sums[col_has_edges] <= max_thres)
                )
            else:
                cols_ok = True

            if rows_ok and cols_ok:
                converged = True
                break

        if not converged:
            warnings.warn(
                "DoublyStochasticNormalize did not converge within max_iter.",
                RuntimeWarning,
            )

        # Apply scaling once: A' = diag(r) * A * diag(c)  (CSR-friendly)
        A_scaled = A.copy()

        # row scaling
        A_scaled.data *= np.repeat(r, np.diff(A_scaled.indptr))

        # col scaling
        A_scaled.data *= c[A_scaled.indices]

        # TODO: For undirected graphs, Graph.from_csr symmetrizes A_scaled again.
        # This may change row/column sums after normalization. A future revision should
        # either use symmetric scaling or allow Graph construction without re-symmetrizing.
        return Graph.from_csr(
            A_scaled,
            directed=G.directed,
            weighted=True,
            mode=G.mode,
            meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
            sym_op="max",
        )


@dataclass(slots=True)
class DoublyStochasticBackbone(GraphOperator):
    """
    Backbone extraction based on doubly-stochastic edge scores.

    First applies DoublyStochasticNormalize, then sorts edges by normalized score
    and adds strongest edges until the graph becomes connected, or until no candidate
    edges remain. For disconnected inputs, the result is a strongest-edge forest
    over the available components.

    References
    ----------
    - Coscia, M. (2025). "The Atlas for the Inspiring Network Scientist."
    - Yassin, A. (2023). "An evaluation tool for backbone extraction techniques
      in weighted complex networks." Scientific Reports.

    Parameters
    ----------
    tolerance : float
        Band tolerance passed to DoublyStochasticNormalize. Default 1e-5.
    max_iter : int
        Maximum Sinkhorn iterations passed to DoublyStochasticNormalize.
        Default 10_000.
    copy_meta : bool
        Copy metadata frame if present. Default True.
    """

    tolerance: float = 1e-5
    max_iter: int = 10_000
    copy_meta: bool = True
    supported_modes = ["similarity"]

    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)

        if G.directed:
            raise NotImplementedError(
                "DoublyStochasticBackbone currently supports only undirected graphs."
            )

        normalized = DoublyStochasticNormalize(
            tolerance=self.tolerance,
            max_iter=self.max_iter,
            copy_meta=self.copy_meta,
        ).apply(G)

        A_scaled = normalized.adj.tocsr(copy=False)
        G_filtered = self._extract_undirected_backbone(A_scaled, G.n_nodes)

        G_csr = nx.to_scipy_sparse_array(
            G_filtered,
            nodelist=list(range(G.n_nodes)),
            weight="weight",
            format="csr",
        )

        return Graph.from_csr(
            G_csr,
            directed=G.directed,
            weighted=True,
            mode=G.mode,
            meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
            sym_op="max",
        )

    @staticmethod
    def _extract_undirected_backbone(A_scaled, n_nodes: int) -> nx.Graph:
        """
        Extract an undirected backbone from normalized edge scores.

        Uses only the upper triangle of the adjacency matrix so that each
        undirected edge is considered once.
        """
        G_filtered = nx.Graph()
        G_filtered.add_nodes_from(range(n_nodes))

        if n_nodes <= 1 or A_scaled.nnz == 0:
            return G_filtered

        # Use only one orientation per undirected edge and ignore self-loops.
        A_upper = sp.triu(A_scaled, k=1).tocoo()

        if A_upper.nnz == 0:
            return G_filtered

        rows = A_upper.row
        cols = A_upper.col
        vals = A_upper.data

        order = np.argsort(vals)[::-1]

        for idx in order:
            if nx.is_connected(G_filtered):  # TODO: For large graphs, this check could be expensive
                break

            G_filtered.add_edge(
                int(rows[idx]),
                int(cols[idx]),
                weight=float(vals[idx]),
            )

        return G_filtered
