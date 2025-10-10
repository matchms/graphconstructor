from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix
import scipy.sparse as sp

from ..types import MatrixMode, CSRMatrix
from . import BaseGraphConstructor, GraphConstructionConfig
from ..utils import (
    _coerce_knn_inputs,
    _threshold_mask,
    _as_csr_square,
)


class EpsilonBallGraphConstructor(BaseGraphConstructor):
    """Connect all pairs (i, j) where distance < ε or similarity > τ.

    Parameters
    ----------
    threshold
        ε for distance mode, or τ for similarity mode.
    mode
        "distance" or "similarity".

    Notes
    -----
    - :meth:`from_ann` in the base class delegates to :meth:`from_knn` after
      querying top-`k` neighbors. Set `k` high enough to avoid missing edges.
    """

    def __init__(self, threshold: float, mode: MatrixMode = "distance",
                 config: Optional[GraphConstructionConfig] = None) -> None:
        super().__init__(config)
        if mode not in ("distance", "similarity"):
            raise TypeError("mode must be 'distance' or 'similarity'.")
        self.threshold = float(threshold)
        self.mode = mode

    def from_matrix(self, matrix: NDArray | spmatrix, *, mode: MatrixMode = None) -> CSRMatrix:
        mode = self.mode if mode is None else mode
        csr, n = _as_csr_square(matrix)
        # Apply threshold without densifying
        data = csr.data
        if mode == "distance":
            keep = data < self.threshold
        else:
            keep = data > self.threshold
        rows, cols = sp.find(csr)[0:2]  # slower for huge matrices; but we reuse CSR internals below TODO:check and optimize?
        # More efficient: reconstruct from row slices
        pruned = csr.copy()
        pruned.data = pruned.data[keep]
        # Easiest is to build from COO of kept mask
        coo = csr.tocoo()
        mask = keep
        rows = coo.row[mask]
        cols = coo.col[mask]
        weights = coo.data[mask]
        if not self.config.store_weights:
            weights = np.ones_like(weights, dtype=float)
        A = csr_matrix((weights, (rows, cols)), shape=(n, n))
        return self._finalize(A)

    def from_knn(self, indices: NDArray, distances: NDArray) -> CSRMatrix:
        ind, dist = _coerce_knn_inputs(indices, distances)
        n = ind.shape[0]
        mask = _threshold_mask(dist, self.threshold, self.mode)
        rows = np.repeat(np.arange(n), ind.shape[1])[mask.ravel()]
        cols = ind.ravel()[mask.ravel()]
        weights = dist.ravel()[mask.ravel()]
        if not self.config.store_weights:
            weights = np.ones_like(weights, dtype=float)
        A = csr_matrix((weights, (rows, cols)), shape=(n, n))
        return self._finalize(A)
