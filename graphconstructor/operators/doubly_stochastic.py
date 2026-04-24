from dataclasses import dataclass
import networkx as nx
import numpy as np
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

        # The csc conversions above could be heavy; build once
        A_T = A.T.tocsr(copy=False)

        min_thres = 1.0 - self.tolerance
        max_thres = 1.0 + self.tolerance

        # Parameter for stabilization
        MAX_FACTOR = 1e50

        for _ in range(self.max_iter):
            c[col_has_edges] = 1 / A_T.dot(r)[col_has_edges]
            r[row_has_edges] = 1 / A.dot(c)[row_has_edges]

            if np.any(np.abs(r) > MAX_FACTOR) or np.any(np.abs(c) > MAX_FACTOR):
                break  # avoid overflow; close enough

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
                break

        # Apply scaling once: A' = diag(r) * A * diag(c)  (CSR-friendly)
        A_scaled = A.copy()

        # row scaling
        A_scaled.data *= np.repeat(r, np.diff(A_scaled.indptr))

        # col scaling
        A_scaled.data *= c[A_scaled.indices]

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
    and adds strongest edges until the graph becomes connected, following the
    behavior of the previous DoublyStochastic(backbone_method=True) branch.

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

        normalized = DoublyStochasticNormalize(
            tolerance=self.tolerance,
            max_iter=self.max_iter,
            copy_meta=self.copy_meta,
        ).apply(G)

        A_scaled = normalized.adj.tocsr(copy=False)

        i = 0
        rows, cols = A_scaled.nonzero()
        vals = A_scaled.data

        order = np.argsort(vals)[::-1]
        rows = rows[order]
        cols = cols[order]
        vals = vals[order]

        if not G.directed:
            G_filtered = nx.Graph()

            while (
                nx.number_connected_components(G_filtered) != 1
                or len(G_filtered) < A_scaled.shape[0]
                or not nx.is_connected(G_filtered)
            ):
                if i == len(rows) or G_filtered.number_of_nodes() == len(set(rows)):
                    break

                G_filtered.add_edge(rows[i], cols[i], weight=vals[i])
                i += 1

            # add isolated nodes
            G_filtered.add_nodes_from(range(G.n_nodes))

        else:
            raise NotImplementedError(
                "DoublyStochasticBackbone currently supports only undirected graphs."
            )

        G_csr = nx.to_scipy_sparse_array(
            G_filtered,
            nodelist=list(range(G.n_nodes)),
        )

        return Graph.from_csr(
            G_csr,
            directed=G.directed,
            weighted=True,
            mode=G.mode,
            meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
            sym_op="max",
        )
