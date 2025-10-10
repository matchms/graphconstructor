from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Union
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from ..adapters import ANNInput
from ..types import CSRMatrix, MatrixMode
from ..utils import _make_symmetric_csr


NXOut = Union[nx.Graph, nx.DiGraph]
OutType = Literal["networkx", "array"]


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

    def _to_output(self, A: CSRMatrix) -> Union[CSRMatrix, NXOut]:
        """Convert the finalized CSR matrix to the requested output type."""
        if self.out == "array":
            return A
        # networkx output
        if self.config.symmetric:
            # Undirected graph
            G = nx.from_scipy_sparse_array(A, parallel_edges=False, create_using=nx.Graph)
        else:
            # Directed graph
            G = nx.from_scipy_sparse_array(A, parallel_edges=False, create_using=nx.DiGraph)
        return G

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
