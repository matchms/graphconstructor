from dataclasses import dataclass
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

    def _mst_distance_adj(self, A: sp.csr_matrix) -> sp.csr_matrix:
        """MST for distance graphs (minimize total weight)."""
        # SciPy returns a sparse matrix with only the tree edges (usually upper tri)
        mst = minimum_spanning_tree(A)
        return mst.tocsr()

    def _mst_similarity(self, A: sp.csr_matrix) -> sp.csr_matrix:
        """
        Maximum spanning tree for similarity graphs.

        We transform similarities w_ij to costs c_ij = max_w - w_ij, compute a
        minimum spanning tree on the costs, and then restore the original
        similarity weights on the selected edges.
        """
        if A.nnz == 0:
            # No edges -> return empty adjacency
            return A.copy() * 0.0

        # Negative weights: minimizing sum(-w) == maximizing sum(w)
        cost_adj = (-A).tocsr()
        mst_cost = minimum_spanning_tree(cost_adj)
        mst_cost = mst_cost.tocsr()

        # Mask of selected edges (positions where mst_cost is non-zero)
        mask = mst_cost.astype(bool).astype(float)

        # Recover original similarities on those edges
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
            mst_adj = self._mst_distance_adj(A)
        if G.mode == "similarity":
            mst_adj = self._mst_similarity(A)

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
