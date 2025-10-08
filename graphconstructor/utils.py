from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix

from .types import MatrixMode


def _validate_square_matrix(M: np.ndarray) -> None:
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise TypeError("Matrix must be square (n x n).")


def _to_numpy(array) -> np.ndarray:
    return array if isinstance(array, np.ndarray) else np.asarray(array)


def _make_symmetric_csr(A: csr_matrix, option: str = "max") -> csr_matrix:
    if option == "max":
        return A.maximum(A.T)
    if option == "min":
        return A.minimum(A.T)
    if option == "average":
        return (A + A.T) * 0.5
    raise ValueError("Unsupported option for symmetrization.")


def _coerce_knn_inputs(indices, distances) -> Tuple[np.ndarray, np.ndarray]:
    ind = _to_numpy(indices)
    dist = _to_numpy(distances)
    if ind.shape != dist.shape:
        raise TypeError("indices and distances must have the same shape (n, k).")
    if ind.ndim != 2:
        raise TypeError("indices/distances must be 2D arrays (n, k).")
    return ind, dist


def _threshold_mask(values: np.ndarray, threshold: float, mode: MatrixMode) -> np.ndarray:
    if mode == "distance":
        return (values < threshold)
    return (values > threshold)


def _csr_from_edges(n: int, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray) -> csr_matrix:
    return csr_matrix((weights, (rows, cols)), shape=(n, n))
