from __future__ import annotations
from typing import Literal, Protocol
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix


MatrixMode = Literal["distance", "similarity"]
CSRMatrix = csr_matrix

class ANNLike(Protocol):
    """Protocol for ANN indexes we can query for neighbors.

    The minimal surface we rely on mirrors PyNNDescent and similar libraries.
    """

    def query(self, X: ArrayLike, k: int) -> tuple[NDArray[np.int_], NDArray[np.floating]]:  # indices, distances
        ...

    # Optional attributes commonly present on fitted ANN indexes
    indices_: NDArray[np.int_] | None
    distances_: NDArray[np.floating] | None
