from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .types import MatrixMode
from .utils import _to_numpy, _validate_square_matrix


@dataclass
class MatrixInput:
    """Input for graph constructors that take a dense matrix."""
    matrix: NDArray
    mode: MatrixMode  # "distance" or "similarity"

    def __post_init__(self) -> None:
        M = _to_numpy(self.matrix)
        _validate_square_matrix(M)
        self.matrix = M.astype(float, copy=False)


@dataclass
class KNNInput:
    """Input for graph constructors that take KNN graphs."""
    indices: NDArray[np.int_]
    distances: NDArray

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=int)
        self.distances = np.asarray(self.distances, dtype=float)
        if self.indices.shape != self.distances.shape or self.indices.ndim != 2:
            raise TypeError("indices and distances must both be (n, k).")


@dataclass
class ANNInput:
    """A fitted ANN index, e.g., pynndescent.NNDescent"""
    index: object
    # Optionally, a query set to build edges from (defaults to the index's training set)
    query_data: NDArray | None = None
