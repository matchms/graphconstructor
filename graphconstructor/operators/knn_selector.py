from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import scipy.sparse as sp
from ..graph import Graph
from ..utils import _topk_per_row_sparse
from .base import GraphOperator


Mode = Literal["distance", "similarity"]

@dataclass(slots=True)
class KNNSelector(GraphOperator):
    """Construct a graph where each node is at max connected to its k nearest neighbors.

    Parameters
    ----------
    k
        Number of neighbors per node.
    mutual
        If True, keep only edges that are reciprocal in the kNN relation.
    mutual_k
        If `mutual=True`, mutuality is tested against the top-`mutual_k`
        neighbor lists of the two endpoints. Then, for each node, at most
        the first `k` of those mutually-confirmed neighbors (preserving the
        original neighbor order) are kept as edges. If `None`, defaults to `k`.
    mode
        Interpret inputs as "distance" (smaller = closer) or "similarity"
        (larger = closer).
    """
    k: int
    mutual: bool = False
    mutual_k: Optional[int] = None
    mode: Mode = "distance"
    supported_modes = ["similarity", "distance"]

    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)
        csr = G.adj.tocsr(copy=False)
        n = csr.shape[0]
        largest = (self.mode == "similarity")
        eff_k = max(self.k, self.mutual_k or self.k)

        # pick top-k per row on the chosen criterion
        ind, vals = _topk_per_row_sparse(csr, eff_k, largest=largest)
        rows = np.repeat(np.arange(n), eff_k)
        cols = ind.reshape(-1)
        mask_valid = cols >= 0
        rows, cols = rows[mask_valid], cols[mask_valid]
        w = vals.reshape(-1)[mask_valid]
        A = sp.csr_matrix((w, (rows, cols)), shape=(n, n))

        if self.mutual:
            mk = self.mutual_k or self.k
            # within top-mk --> mutual check
            # Build boolean mask: keep i->j if j in top-mk(i) and i in top-mk(j)
            mk = int(min(mk, eff_k))
            #top_i = ind[:, :mk]
            # membership test via a sparse “candidate” matrix
            Cand = sp.csr_matrix((np.ones_like(rows, dtype=np.float32), (rows, cols)), shape=(n, n))
            # Keep only edges where (i,j) and (j,i) both exist in Cand restricted to top-mk:
            # Efficient approximation: intersect with transpose sign()
            A = A.multiply(Cand.T.sign())
            # After intersection we still want to limit to k per row
            ind2, val2 = _topk_per_row_sparse(A, self.k, largest=largest)
            rows = np.repeat(np.arange(n), self.k)
            cols = ind2.reshape(-1)
            valid = cols >= 0
            rows, cols = rows[valid], cols[valid]
            w = val2.reshape(-1)[valid]
            A = sp.csr_matrix((w, (rows, cols)), shape=(n, n))

        return Graph.from_csr(
            A,
            directed=G.directed,
            weighted=G.weighted,
            mode=self.mode,
            meta=G.meta.copy() if G.meta is not None else None,
            sym_op="max" if not G.directed else "max",
        )
