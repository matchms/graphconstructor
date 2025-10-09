from dataclasses import dataclass
from typing import Optional, Literal
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix
import scipy.sparse as sp

from .types import MatrixMode, CSRMatrix
from .adapters import MatrixInput, KNNInput, ANNInput
from .utils import (
    _make_symmetric_csr,
    _coerce_knn_inputs,
    _threshold_mask,
    _csr_from_edges,
    _as_csr_square,
    _knn_from_matrix,
)


@dataclass(slots=True)
class GraphConstructionConfig:
    """Common configuration for all graph constructors.

    Parameters
    ----------
    symmetric
        If True, symmetrize the resulting adjacency using `symmetrize_op`.
    symmetrize_op
        Strategy to combine reciprocal edges when `symmetric=True`.
        One of {"max", "min", "average"}.
    store_weights
        If False, store all edges with weight 1.0 regardless of input values.
    self_loops
        If False, diagonal entries are cleared before returning.
    """

    symmetric: bool = True
    symmetrize_op: Literal["max", "min", "average"] = "max"
    store_weights: bool = True
    self_loops: bool = False


class BaseGraphConstructor(ABC):
    """Abstract base for graph constructors.

    Subclasses must implement :meth:`from_matrix` and :meth:`from_knn`.
    This base class also provides a generic :meth:`from_ann` that converts
    an ANN index to `(indices, distances)` and then delegates to
    :meth:`from_knn`.
    """

    def __init__(self, config: Optional[GraphConstructionConfig] = None) -> None:
        self.config = config or GraphConstructionConfig()

    def _finalize(self, A: CSRMatrix) -> CSRMatrix:
        if not self.config.self_loops:
            A.setdiag(0)
            A.eliminate_zeros()
        if self.config.symmetric:
            A = _make_symmetric_csr(A, option=self.config.symmetrize_op)
        return A.astype(float, copy=False)

    @abstractmethod
    def from_matrix(self, matrix: NDArray | spmatrix, *, mode: MatrixMode = "distance") -> CSRMatrix:
        """Build the graph from a square distance/similarity matrix.

        Implementations **must** support both dense NumPy arrays and SciPy sparse
        matrices (ideally CSR/CSC). Implementations MUST NOT densify sparse inputs.
        """

    @abstractmethod
    def from_knn(self, indices: NDArray, distances: NDArray) -> CSRMatrix:
        """Build the graph from precomputed kNN arrays of shape (n, k)."""

    def from_ann(self, ann: ANNInput, *, k: Optional[int] = None) -> CSRMatrix:
        """Build the graph by querying an approximate nearest-neighbor index.

        The default implementation extracts `(indices, distances)` from the
        ANN object if available (attributes `indices_`, `distances_`), otherwise
        it performs a query with the requested `k` and delegates to
        :meth:`from_knn`.

        Notes
        -----
        - For epsilon-ball graphs this path relies on a kNN truncation and may
          miss neighbors outside the top-`k`. Choose a sufficiently large `k`
          when using ANN with epsilon-ball construction.
        """
        idx = ann.index
        if hasattr(idx, "indices_") and getattr(idx, "indices_") is not None:
            ind = np.asarray(getattr(idx, "indices_"))
            dist = np.asarray(getattr(idx, "distances_"))
            if k is not None:
                ind = ind[:, :k]
                dist = dist[:, :k]
            return self.from_knn(ind, dist)
        if ann.query_data is None:
            raise TypeError(
                "ANNInput requires query_data when the index has no precomputed neighbors.")
        if k is None:
            raise TypeError("from_ann requires parameter k when querying the index.")
        ind, dist = idx.query(ann.query_data, k=k)
        return self.from_knn(ind, dist)


# ---------------------------------------------------------------------------
# Concrete constructors
# ---------------------------------------------------------------------------

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
        rows, cols = sp.find(csr)[0:2]  # slower for huge matrices; but we reuse CSR internals below
        # More efficient: reconstruct from row slices
        pruned = csr.copy()
        pruned.data = pruned.data[keep]
        # We must also shrink indices/indptr accordingly; easiest is to build from COO of kept mask
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
