from typing import Literal, Optional, Union
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from ..types import CSRMatrix, MatrixMode
from ..utils import (
    _coerce_knn_inputs,
    _csr_from_edges,
    _knn_from_matrix,
)
from . import BaseGraphConstructor, GraphConstructionConfig


NXOut = Union[nx.Graph, nx.DiGraph]
OutType = Literal["networkx", "array"]


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
    out
        Output type: "networkx" (default) returns a NetworkX graph; "array"
        returns a SciPy CSR adjacency matrix.
    mutual_k
        If `mutual=True`, mutuality is tested against the top-`mutual_k`
        neighbor lists of the two endpoints. Then, for each node, at most
        the first `k` of those mutually-confirmed neighbors (preserving the
        original neighbor order) are kept as edges. If `None`, defaults to `k`.
    """

    def __init__(
        self,
        k: int,
        mutual: bool = False,
        mode: MatrixMode = "distance",
        *,
        out: OutType = "networkx",
        mutual_k: Optional[int] = None,
        config: Optional[GraphConstructionConfig] = None,
    ) -> None:
        super().__init__(config)
        if k <= 0:
            raise TypeError("k must be positive.")
        if mode not in ("distance", "similarity"):
            raise TypeError("mode must be 'distance' or 'similarity'.")
        if out not in ("networkx", "array"):
            raise TypeError("out must be 'networkx' or 'array'.")
        if mutual_k is not None and mutual_k < k:
            raise TypeError("mutual_k must be >= k if provided.")

        self.k = int(k)
        self.mutual = bool(mutual)
        self.mode = mode
        self.out: OutType = out
        self.mutual_k: Optional[int] = int(mutual_k) if mutual_k is not None else None

    def _effective_k_for_selection(self) -> int:
        """k used when *computing* candidate neighbors (need enough for mutual_k)."""
        mk = self.mutual_k if self.mutual_k is not None else self.k
        return max(self.k, mk)

    def from_matrix(
        self,
        matrix: NDArray | spmatrix,
        *,
        mode: MatrixMode = None,
    ) -> Union[CSRMatrix, NXOut]:
        mode = self.mode if mode is None else mode
        base_k = self._effective_k_for_selection()
        ind, dist = _knn_from_matrix(matrix, base_k, mode=mode)
        return self.from_knn(ind, dist)

    def from_knn(self, indices: NDArray, distances: NDArray) -> Union[CSRMatrix, NXOut]:
        ind, dist = _coerce_knn_inputs(indices, distances)
        n, base_k = ind.shape

        if not self.mutual:
            # original simple construction
            rows = np.repeat(np.arange(n), base_k)
            cols = ind.reshape(-1)
            weights = dist.reshape(-1)
            if not self.config.store_weights:
                weights = np.ones_like(weights, dtype=float)
            A = _csr_from_edges(n, rows, cols, weights)
            return self._to_output(self._finalize(A))

        # --- mutual=True path with mutual_k support ---
        mk = self.mutual_k if self.mutual_k is not None else self.k
        mk = min(mk, base_k)  # don't exceed available columns

        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []

        # For each node, scan its neighbor list in order and keep up to k
        for i in range(n):
            added = 0
            # Preload top-mk of the *other* lists lazily when needed
            for j_pos in range(base_k):
                if added >= self.k:
                    break
                j = int(ind[i, j_pos])
                if j < 0 or j >= n:
                    continue
                # Mutuality: i in top-mk of j AND j in top-mk of i
                # (the latter is guaranteed since we're iterating top base_k of i;
                # but ensure j_pos < mk if we want j to be within i's top mk too)
                if j_pos >= mk:
                    # j is not within i's top-mk neighbors
                    continue
                # Is i within j's top-mk neighbors?
                # Note: j may have padded -1s; we ignore negatives.
                top_j = ind[j, :mk]
                if i not in top_j:
                    continue

                w = float(dist[i, j_pos]) if self.config.store_weights else 1.0
                rows.append(i)
                cols.append(j)
                weights.append(w)
                added += 1

        A = _csr_from_edges(n, np.array(rows), np.array(cols), np.array(weights, dtype=float))
        return self._to_output(self._finalize(A))
