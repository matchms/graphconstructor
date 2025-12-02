from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree
from graphconstructor import Graph
from graphconstructor.operators.base import GraphOperator


@dataclass(slots=True)
class MinimumSpanningTree(GraphOperator):
    """
    Minimum Spanning Tree (MST) operator.

    - For graphs with mode="distance", computes a standard minimum spanning tree
      (minimizing the sum of edge weights).

    - For graphs with mode="similarity", computes a maximum spanning tree by
      internally converting similarities to costs, but returns a graph whose
      edge weights are still the original similarities. That is, the resulting
      tree connects all nodes while preserving as much similarity as possible.

    Notes
    -----
    - Only supports undirected graphs (directed=False).
    - The input graph must be (weakly) connected; otherwise a ValueError
      is raised.
    - The result is an undirected Graph with n_nodes identical to the input,
      but with exactly n_nodes - 1 edges (if n_nodes > 0).
    """

    copy_meta: bool = True
    # Distance: classical MST; Similarity: maximum spanning tree
    supported_modes = ["distance", "similarity"]

    def _mst_from_distance_adj(self, A: sp.csr_matrix) -> sp.csr_matrix:
        """MST for distance graphs (minimize total weight)."""
        # SciPy returns a sparse matrix with only the tree edges (usually upper tri)
        mst = minimum_spanning_tree(A)
        return mst.tocsr()

    def _mst_for_similarity(self, A: sp.csr_matrix) -> sp.csr_matrix:
        """
        Maximum spanning tree for similarity graphs.

        We transform similarities w_ij to costs c_ij = max_w - w_ij, compute a
        minimum spanning tree on the costs, and then restore the original
        similarity weights on the selected edges.
        """
        coo = A.tocoo(copy=False)
        if coo.nnz == 0:
            # Degenerate: no edges
            return A.copy() * 0.0

        data = coo.data
        max_w = data.max()
        # Larger similarity -> smaller cost
        cost = (max_w - data).astype(float, copy=False)

        cost_adj = sp.csr_matrix((cost, (coo.row, coo.col)), shape=A.shape)
        mst_cost = minimum_spanning_tree(cost_adj).tocoo()

        if mst_cost.nnz == 0:
            # Should not happen for a connected graph, but be defensive
            return A.copy() * 0.0

        # Build a mask adjacency for tree edges
        mask = sp.csr_matrix(
            (np.ones_like(mst_cost.data, dtype=float),
             (mst_cost.row, mst_cost.col)),
            shape=A.shape,
        )
        # Keep original similarities on the tree edges only
        tree = A.multiply(mask)
        return tree.tocsr()

    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)

        if G.directed:
            raise ValueError(
                "MinimumSpanningTree only supports undirected graphs "
                "(directed=False)."
            )

        # Check that MST is well-defined: graph must be connected
        if not G.is_connected():
            raise ValueError(
                "MinimumSpanningTree requires a connected graph. "
                "The input graph has more than one connected component."
            )

        A = G.adj.tocsr(copy=False)

        if G.mode == "distance":
            mst_adj = self._mst_from_distance_adj(A)
        elif G.mode == "similarity":
            mst_adj = self._mst_for_similarity(A)
        else:
            # Should be covered by _check_mode_supported, but keep explicit
            raise ValueError(f"Unsupported graph mode '{G.mode}'.")

        # For undirected graphs, SciPy's MST is usually upper-triangular; letting
        # Graph.from_csr(sym_op='max') symmetrize is enough.
        return Graph.from_csr(
            mst_adj,
            directed=False,
            weighted=G.weighted,
            mode=G.mode,
            meta=G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta,
            sym_op="max",
            ignore_selfloops=G.ignore_selfloops,
            keep_explicit_zeros=G.keep_explicit_zeros,
        )
