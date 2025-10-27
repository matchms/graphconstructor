from dataclasses import dataclass
from typing import Literal
import numpy as np
import scipy.sparse as sp
from ..graph import Graph
from .base import GraphOperator


Mode = Literal["distance", "similarity"]

@dataclass(slots=True)
class WeightThreshold(GraphOperator):
    """Keep edges which correspond to weights above/below a threshold.
    Weights can be interpreted as similarity (then >= threshold) or as distances (then <= threshold).

    Parameters
    ----------
    threshold
        Depending on the set mode, edges with weights above/below this threshold are kept.
    mode
        "distance" or "similarity".
    """
    threshold: float
    mode: Mode = "distance"
    supported_modes = ["similarity", "distance"]

    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)
        csr = G.adj.tocsr(copy=False)
        coo = csr.tocoo()
        keep = (coo.data <= self.threshold) if self.mode == "distance" else (coo.data >= self.threshold)
        rows, cols = coo.row[keep], coo.col[keep]
        w = coo.data[keep] if G.weighted else np.ones(keep.sum(), dtype=float)
        A = sp.csr_matrix((w, (rows, cols)), shape=csr.shape)
        return Graph.from_csr(A, directed=G.directed, weighted=G.weighted,
                              mode=self.mode,
                              meta=None if G.meta is None else G.meta.copy(), sym_op="max")
