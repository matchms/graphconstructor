from typing import Optional
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .core import (
    GraphConstructionConfig,
    KNNGraphConstructor,
    EpsilonBallGraphConstructor,
)
from .types import MatrixMode
from .adapters import MatrixInput, KNNInput, ANNInput

# ---------- Functional convenience API ----------

def build_knn_graph(
    *,
    # One of the following must be provided:
    matrix: Optional[NDArray] = None,
    mode: MatrixMode = "distance",
    indices: Optional[NDArray] = None,
    distances: Optional[NDArray] = None,
    ann: Optional[ANNInput] = None,
    # Parameters
    k: int,
    mutual: bool = False,
    config: Optional[GraphConstructionConfig] = None,
) -> csr_matrix:
    """Build a kNN graph from one of the supported inputs.

    Exactly one of (matrix) or (indices & distances) or (ann) should be provided.
    """
    ctor = KNNGraphConstructor(k=k, mutual=mutual, mode=mode, config=config)

    if matrix is not None:
        return ctor.from_matrix(matrix)
    if indices is not None and distances is not None:
        return ctor.from_knn(indices=indices, distances=distances)
    if ann is not None:
        return ctor.from_ann(ann)
    raise ValueError("Provide either matrix, (indices & distances), or ann.")


def build_epsilon_ball_graph(
    *,
    # One of the following must be provided:
    matrix: Optional[NDArray] = None,
    mode: MatrixMode = "distance",
    indices: Optional[NDArray] = None,
    distances: Optional[NDArray] = None,
    n: Optional[int] = None,
    # Parameter
    threshold: float,
    config: Optional[GraphConstructionConfig] = None,
) -> csr_matrix:
    """Build an ε-ball graph from a matrix or filtered kNN results.

    If using (indices, distances), provide `n` if the global number of nodes differs
    from `indices.shape[0]` (e.g., when kNN computed on a subset).
    """
    ctor = EpsilonBallGraphConstructor(threshold=threshold, mode=mode, config=config)

    if matrix is not None:
        return ctor.from_matrix(matrix)
    if indices is not None and distances is not None:
        return ctor.from_knn(indices=indices, distances=distances, n=n)
    raise ValueError("Provide either matrix or (indices & distances).")
