from dataclasses import dataclass
from typing import Literal
import numpy as np
import scipy.sparse as sp
from ..graph import Graph
from .base import GraphOperator


UndirectedRule = Literal["or", "and"]


@dataclass(slots=True)
class LocallyAdaptiveSparsification(GraphOperator):
    """
    Implementation of "Locally Adaptive Network Sparsification" (LANS)
    Foti, Hughes, Rockmore (Nonparametric Sparsification of Complex Multiscale Networks).

    For each node i, fractional weights p_ij = w_ij / s_i (s_i = row sum).
    Local upper-tail p-value at i for neighbor j:
        pval_i(j) = (# neighbors u with p_iu >= p_ij) / k_i
    Keep an edge if it is locally significant.

    Directed: keep i->j if BOTH endpoints deem it significant:
        min(pval_out(i->j), pval_in(i->j)) <= alpha
        (implemented as elementwise AND of boolean masks)
    Undirected: combine endpoints via rule:
        "or"  -> keep if pval_i(j) <= alpha OR pval_j(i) <= alpha
        "and" -> keep only if both are <= alpha

    Notes
    -----
    - Nonparametric: no distributional assumptions; purely rank/ECDF based per node.
    - Complexity ~ sum_i O(k_i log k_i) due to per-row sort for ECDF.
    - Requires nonnegative weights. Self-loops are assumed absent by Graph construction.

    Parameters
    ----------
    alpha : float
        Significance level in (0,1].
    rule : {"or","and"}
        Combination rule for UNDIRECTED graphs. Ignored for directed graphs.
    copy_meta : bool
        If True, copy metadata frame onto the result graph.

    """
    alpha: float = 0.05
    rule: UndirectedRule = "or"
    copy_meta: bool = True
    supported_modes = ["similarity"]

    # ---- helpers ----
    @staticmethod
    def _row_upper_tail_pvals_csr(A: sp.csr_matrix) -> np.ndarray:
        """
        For each edge in CSR, compute upper-tail empirical p-value at the row's node:
            pval = (# neighbors with fractional >= current) / k_row
        Returns an array of shape (A.nnz,), aligned with A.data order.
        Rows with zero sum -> fractional weights 0; pvals become 1 (drop unless alpha>=1).
        """
        n = A.shape[0]
        data = A.data
        indptr = A.indptr

        # out-strengths s_i and degrees k_i
        s_out = np.asarray(A.sum(axis=1)).ravel()
        #k_out = np.diff(indptr)

        pvals = np.empty_like(data, dtype=float)

        # iterate rows (per-row ECDF)
        for i in range(n):
            start, end = indptr[i], indptr[i + 1]
            if start == end:
                continue
            w = data[start:end]
            s = s_out[i]
            k = end - start
            if s <= 0:
                # no strength -> all fractional weights 0, upper-tail pval = 1 for all
                pvals[start:end] = 1.0
                continue

            frac = w / s  # fractional weights for row i
            # sort ascending once; upper-tail count for x is k - left_index_ge(x)
            asc = np.sort(frac, kind="mergesort")  # stable sort helps ties (optional)
            left_idx = np.searchsorted(asc, frac, side="left")
            count_ge = k - left_idx
            pvals[start:end] = count_ge / k

        return pvals

    @staticmethod
    def _mask_from_pvals(A: sp.csr_matrix, pvals: np.ndarray, alpha: float) -> sp.csr_matrix:
        """
        Build a sparse boolean mask (CSR) with ones at positions where pvals<=alpha.
        """
        coo = A.tocoo()
        keep = pvals <= alpha
        if not np.any(keep):
            return sp.csr_matrix(A.shape, dtype=bool)
        rows = coo.row[keep]
        cols = coo.col[keep]
        data = np.ones(keep.sum(), dtype=bool)
        return sp.csr_matrix((data, (rows, cols)), shape=A.shape, dtype=bool)

    def _apply_directed(self, G: Graph) -> Graph:
        A = G.adj.tocsr(copy=False)
        if (A.data < 0).any():
            raise ValueError("LANS requires nonnegative weights.")
        if A.nnz == 0:
            return Graph.from_csr(
                A.copy(), directed=True, weighted=G.weighted,
                meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta)
            )

        # Out-side mask on A
        p_out = self._row_upper_tail_pvals_csr(A)
        M_out = self._mask_from_pvals(A, p_out, self.alpha)          # boolean CSR in A-coordinates

        # In-side mask: do row-ECDF on A^T, then transpose back to align with A
        AT = A.T.tocsr(copy=False)
        p_in_T = self._row_upper_tail_pvals_csr(AT)
        M_in_T = self._mask_from_pvals(AT, p_in_T, self.alpha)       # boolean CSR in AT-coordinates
        M_in = M_in_T.T                                              # back to A-coordinates

        # Keep arcs where both endpoints significant
        K = M_out.multiply(M_in)                                     # elementwise AND (1*1=1)
        A_kept = A.multiply(K.astype(float))                         # preserve original weights

        return Graph.from_csr(
            A_kept, directed=True, weighted=G.weighted,
            mode=G.mode,
            meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta)
        )

    def _apply_undirected(self, G: Graph) -> Graph:
        A = G.adj.tocsr(copy=False)
        if (A.data < 0).any():
            raise ValueError("LANS requires nonnegative weights.")
        if A.nnz == 0:
            return Graph.from_csr(
                A.copy(), directed=False, weighted=G.weighted,
                mode=G.mode,
                meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
                sym_op="max",
            )

        # Endpoint-local masks (using the same A since rows represent each endpoint's neighborhood)
        p_src = self._row_upper_tail_pvals_csr(A)
        M_src = self._mask_from_pvals(A, p_src, self.alpha)          # boolean CSR

        if self.rule == "or":
            K = M_src.maximum(M_src.T)                               # OR via elementwise max on booleans
        elif self.rule == "and":
            K = M_src.multiply(M_src.T)                              # AND via elementwise product
        else:
            raise ValueError("rule must be 'or' or 'and'.")

        # Apply mask, then ensure symmetry (though K is already symmetric by construction)
        A_kept = A.multiply(K.astype(float)).maximum(
            A.multiply(K.astype(float)).T
        )

        return Graph.from_csr(
            A_kept, directed=False, weighted=G.weighted,
            mode=G.mode,
            meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
            sym_op="max",
        )

    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)
        if G.directed:
            return self._apply_directed(G)
        return self._apply_undirected(G)
