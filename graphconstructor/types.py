from __future__ import annotations
from typing import Literal, Optional, Protocol, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix


MatrixMode = Literal["distance", "similarity"]
CSRMatrix = csr_matrix

class ANNLike(Protocol):
    """Protocol for ANN indexes we can query for neighbors.

    The minimal surface we rely on mirrors PyNNDescent and similar libraries.
    """

    def query(self, X: ArrayLike, k: int) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:  # indices, distances
        ...

    # Optional attributes commonly present on fitted ANN indexes
    indices_: Optional[NDArray[np.int_]]
    distances_: Optional[NDArray[np.floating]]
