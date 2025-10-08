from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .types import MatrixMode, CSRMatrix
from .adapters import MatrixInput, KNNInput, ANNInput
from .utils import (
    _make_symmetric_csr,
    _coerce_knn_inputs,
    _threshold_mask,
    _csr_from_edges,
)


@dataclass(slots=True)
class GraphConstructionConfig:
    symmetric: bool = True
    symmetrize_op: Literal["max", "min", "average"] = "max"
    store_weights: bool = True  # False -> unweighted (all weights = 1)
    self_loops: bool = False


class BaseGraphConstructor:
    def __init__(self, config: Optional[GraphConstructionConfig] = None) -> None:
        self.config = config or GraphConstructionConfig()

    def _finalize(self, A: CSRMatrix) -> CSRMatrix:
        if not self.config.self_loops:
            A.setdiag(0)
            A.eliminate_zeros()
        if self.config.symmetric:
            A = _make_symmetric_csr(A, option=self.config.symmetrize_op)
        return A.astype(float, copy=False)


class KNNGraphConstructor(BaseGraphConstructor):
    """Construct a graph by connecting each node to its k nearest neighbors.

    Supports mutual edges mode (keep only edges i->j where j is in i's neighbors
    **and** i is in j's neighbors) and accepts (a) a distance/similarity matrix,
    (b) precomputed kNN arrays, or (c) an ANN index.
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

    # ----- Entry points for the three starting points -----
    def from_matrix(self, matrix: NDArray) -> CSRMatrix:
        M = np.asarray(matrix, dtype=float)
        n = M.shape[0]
        # For similarity, larger = closer -> use argsort descending
        if self.mode == "distance":
            nn_idx = np.argpartition(M, kth=self.k, axis=1)[:, : self.k]
            nn_dist = np.take_along_axis(M, nn_idx, axis=1)
        else:  # similarity
            nn_idx = np.argpartition(-M, kth=self.k, axis=1)[:, : self.k]
            nn_dist = np.take_along_axis(M, nn_idx, axis=1)
        return self._build_from_knn_arrays(nn_idx, nn_dist, n)

    def from_knn(self, indices: NDArray, distances: NDArray) -> CSRMatrix:
        ind, dist = _coerce_knn_inputs(indices, distances)
        n = ind.shape[0]
        return self._build_from_knn_arrays(ind, dist, n)

    def from_ann(self, ann: ANNInput) -> CSRMatrix:
        idx = ann.index
        # Prefer precomputed neighbor graph if present
        if hasattr(idx, "indices_") and getattr(idx, "indices_") is not None:
            ind = np.asarray(getattr(idx, "indices_"))[:, : self.k]
            dist = np.asarray(getattr(idx, "distances_"))[:, : self.k]
            n = ind.shape[0]
            return self._build_from_knn_arrays(ind, dist, n)
        # Otherwise query using provided query_data or the index itself if it supports it
        if ann.query_data is None:
            raise TypeError(
                "ANNInput requires query_data when the index has no precomputed neighbors.")
        ind, dist = idx.query(ann.query_data, k=self.k)
        n = ind.shape[0]
        return self._build_from_knn_arrays(ind, dist, n)

    # ----- Core -----
    def _build_from_knn_arrays(self, indices: NDArray, distances: NDArray, n: int) -> CSRMatrix:
        rows = np.repeat(np.arange(n), indices.shape[1])
        cols = indices.reshape(-1)
        weights = distances.reshape(-1)
        if not self.config.store_weights:
            weights = np.ones_like(weights, dtype=float)
        A = _csr_from_edges(n, rows, cols, weights)
        if self.mutual:
            # Keep only edges present in both directions; use Hadamard product of boolean structure
            B = A.multiply(A.T.sign())  # zeroes non-mutual edges; preserves weights from A
            A = B
        return self._finalize(A)


class EpsilonBallGraphConstructor(BaseGraphConstructor):
    """Connect all pairs (i, j) where distance < ε or similarity > τ.

    Works from (a) a square matrix, or (b) precomputed kNN arrays (filtered by threshold).
    """

    def __init__(self, threshold: float, mode: MatrixMode = "distance",
                 config: Optional[GraphConstructionConfig] = None) -> None:
        super().__init__(config)
        if threshold <= 0 and mode == "distance":
            # allow zero if explicitly desired but negative makes no sense
            pass
        self.threshold = float(threshold)
        if mode not in ("distance", "similarity"):
            raise TypeError("mode must be 'distance' or 'similarity'.")
        self.mode = mode

    def from_matrix(self, matrix: NDArray) -> CSRMatrix:
        M = np.asarray(matrix, dtype=float)
        n = M.shape[0]
        mask = _threshold_mask(M, self.threshold, self.mode)
        rows, cols = mask.nonzero()
        weights = M[rows, cols]
        if not self.config.store_weights:
            weights = np.ones_like(weights, dtype=float)
        A = csr_matrix((weights, (rows, cols)), shape=(n, n))
        return self._finalize(A)

    def from_knn(self, indices: NDArray, distances: NDArray, n: Optional[int] = None) -> CSRMatrix:
        ind, dist = _coerce_knn_inputs(indices, distances)
        n = int(n if n is not None else ind.shape[0])
        mask = _threshold_mask(dist, self.threshold, self.mode)
        rows = np.repeat(np.arange(ind.shape[0]), ind.shape[1])[mask.ravel()]
        cols = ind.ravel()[mask.ravel()]
        weights = dist.ravel()[mask.ravel()]
        if not self.config.store_weights:
            weights = np.ones_like(weights, dtype=float)
        A = csr_matrix((weights, (rows, cols)), shape=(n, n))
        return self._finalize(A)
