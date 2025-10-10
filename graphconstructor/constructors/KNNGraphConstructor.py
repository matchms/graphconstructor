from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from ..types import MatrixMode, CSRMatrix
from . import BaseGraphConstructor, GraphConstructionConfig
from ..utils import (
    _coerce_knn_inputs,
    _csr_from_edges,
    _knn_from_matrix,
)


class KNNGraphConstructor(BaseGraphConstructor):
    """Construct a graph by connecting each node to its k nearest neighbors.

    Parameters
    ----------
    k
        Number of neighbors per node.
    mutual
        If True, keep only edges that are reciprocal in the kNN relation.
    mode
        Interpret inputs as "distance" (smaller = closer) or "similarity"
        (larger = closer).
    """

    def __init__(self, k: int, mutual: bool = False, mode: MatrixMode = "distance",
                 config: Optional[GraphConstructionConfig] = None) -> None:
        super().__init__(config)
        if k <= 0:
            raise TypeError("k must be positive.")
        if mode not in ("distance", "similarity"):
            raise TypeError("mode must be 'distance' or 'similarity'.")
        self.k = int(k)
        self.mutual = bool(mutual)
        self.mode = mode

    def from_matrix(self, matrix: NDArray | spmatrix, *, mode: MatrixMode = None) -> CSRMatrix:
        mode = self.mode if mode is None else mode
        ind, dist = _knn_from_matrix(matrix, self.k, mode=mode)
        return self.from_knn(ind, dist)

    def from_knn(self, indices: NDArray, distances: NDArray) -> CSRMatrix:
        ind, dist = _coerce_knn_inputs(indices, distances)
        n = ind.shape[0]
        rows = np.repeat(np.arange(n), ind.shape[1])
        cols = ind.reshape(-1)
        weights = dist.reshape(-1)
        if not self.config.store_weights:
            weights = np.ones_like(weights, dtype=float)
        A = _csr_from_edges(n, rows, cols, weights)
        if self.mutual:
            # Keep only edges present in both directions; preserves weights from A
            A = A.multiply(A.T.sign())
        return self._finalize(A)
