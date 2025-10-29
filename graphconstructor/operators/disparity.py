from dataclasses import dataclass
from typing import Literal
import numpy as np
import scipy.sparse as sp
from ..graph import Graph
from .base import GraphOperator


UndirectedRule = Literal["or", "and"]


@dataclass(slots=True)
class DisparityFilter(GraphOperator):
    """
    Disparity Filter backbone (Serrano, Boguñá, Vespignani, 2009).
    Works on nonnegative, real-valued weights; no integer casting needed.

    Input requirements:
    - mode: "similarity" (higher weights = stronger connections)
    - weights: continuous, non-negative

    Undirected:
        P = A / rowSums(A)
        pval_undirected = min( (1-P)^(k_row-1), (1-P^T)^(k_col-1) )
        Keep if pval_undirected <= alpha

    Directed:
        outp = A / rowSums(A);   outval = (1 - outp)^(k_out - 1)
        inp  = A / colSums(A);   inval = (1 - inp )^(k_in  - 1)
        pval_directed = min(outval, inval)
        Keep if pval_directed <= alpha

    Parameters
    ----------
    alpha : float
        Significance level (0,1]; smaller -> sparser backbone.
    rule : {"or","and"}
        For undirected graphs only. If "or" (default), keep if either endpoint
        finds the edge significant (this matches the R backbone code).
        If "and", require both endpoints to be significant (stricter).
    copy_meta : bool
        Copy metadata (True) or keep reference (False).
    """
    alpha: float = 0.05
    rule: UndirectedRule = "or"
    copy_meta: bool = True
    supported_modes = ["similarity"]

    def _undirected(self, G: Graph) -> Graph:
        A = G.adj.tocsr(copy=False)
        if (A.data < 0).any():
            raise ValueError("DisparityFilter requires nonnegative weights.")
        if A.nnz == 0:
            return Graph.from_csr(A.copy(), directed=False, weighted=G.weighted,
                                  meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
                                  mode="similarity",
                                  sym_op="max")

        # strengths and degrees (row-wise)
        strength = np.asarray(A.sum(axis=1)).ravel()
        degree = np.diff(A.indptr)  # counts of nonzeros per row

        coo = A.tocoo()
        rows = coo.row
        cols = coo.col
        w = coo.data

        # p_ij from i's perspective
        p_src = np.zeros_like(w)
        mask_src = strength[rows] > 0
        p_src[mask_src] = w[mask_src] / strength[rows[mask_src]]
        k_src = np.maximum(degree[rows] - 1, 0)
        pval_src = np.power(1.0 - np.clip(p_src, 0.0, 1.0), k_src)

        # p_ji from j's perspective (i.e., using column's row stats)
        # For undirected, we use the same row-wise formulation on the transposed roles.
        strength_T = np.asarray(A.sum(axis=1)).ravel()  # same as strength; we need values indexed by 'cols'
        degree_T = degree
        p_tgt = np.zeros_like(w)
        mask_tgt = strength_T[cols] > 0
        p_tgt[mask_tgt] = w[mask_tgt] / strength_T[cols[mask_tgt]]
        k_tgt = np.maximum(degree_T[cols] - 1, 0)
        pval_tgt = np.power(1.0 - np.clip(p_tgt, 0.0, 1.0), k_tgt)

        if self.rule == "or":
            keep = (pval_src <= self.alpha) | (pval_tgt <= self.alpha)
        elif self.rule == "and":
            keep = (pval_src <= self.alpha) & (pval_tgt <= self.alpha)
        else:
            raise ValueError("rule must be 'or' or 'and'")

        A_f = sp.csr_matrix((w[keep], (rows[keep], cols[keep])), shape=A.shape)
        # Symmetrize to be safe (weights preserved as in input)
        A_f = A_f.maximum(A_f.T)
        return Graph.from_csr(A_f, directed=False, weighted=G.weighted,
                              mode=G.mode,
                              meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
                              sym_op="max")

    def _directed(self, G: Graph) -> Graph:
        A = G.adj.tocsr(copy=False)
        if (A.data < 0).any():
            raise ValueError("DisparityFilter requires nonnegative weights.")
        if A.nnz == 0:
            return Graph.from_csr(A.copy(), directed=True, weighted=G.weighted,
                                  meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta))


        s_out = np.asarray(A.sum(axis=1)).ravel()
        s_in = np.asarray(A.sum(axis=0)).ravel()
        k_out = np.diff(A.indptr)
        k_in = np.diff(A.tocsc().indptr)

        coo = A.tocoo()
        rows = coo.row
        cols = coo.col
        w = coo.data

        # out side
        p_out = np.zeros_like(w)
        mask_out = s_out[rows] > 0
        p_out[mask_out] = w[mask_out] / s_out[rows[mask_out]]
        exp_out = np.maximum(k_out[rows] - 1, 0)
        pval_out = np.power(1.0 - np.clip(p_out, 0.0, 1.0), exp_out)

        # in side
        p_in = np.zeros_like(w)
        mask_in = s_in[cols] > 0
        p_in[mask_in] = w[mask_in] / s_in[cols[mask_in]]
        exp_in = np.maximum(k_in[cols] - 1, 0)
        pval_in = np.power(1.0 - np.clip(p_in, 0.0, 1.0), exp_in)

        keep = np.minimum(pval_out, pval_in) <= self.alpha
        A_f = sp.csr_matrix((w[keep], (rows[keep], cols[keep])), shape=A.shape)
        return Graph.from_csr(A_f, directed=True, weighted=G.weighted,
                              mode=G.mode,
                              meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta))

    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)
        if G.directed:
            return self._directed(G)
        return self._undirected(G)
