from typing import Tuple
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse, spmatrix
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


def _as_csr_square(M: NDArray | spmatrix) -> Tuple[sp.csr_matrix, int]:
    """Return (CSR, n) for a square matrix without densifying.

    If `M` is dense, convert to CSR. If `M` is sparse, convert format to CSR
    (without touching data). Raises TypeError for non-square matrices.
    """
    if issparse(M):
        csr = M.tocsr(copy=False)
        n, m = csr.shape
        if n != m:
            raise TypeError("Matrix must be square (n x n).")
        return csr, n
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise TypeError("Matrix must be square (n x n).")
    return sp.csr_matrix(arr), arr.shape[0]


def _topk_per_row_sparse(csr: sp.csr_matrix, k: int, *, largest: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, values) of top-k entries per row from CSR matrix.

    This operates strictly on the row's nonzeros without densifying.
    Diagonal entries are ignored.
    """
    n = csr.shape[0]
    ind = np.empty((n, k), dtype=int)
    vals = np.empty((n, k), dtype=float)
    for i in range(n):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        cols = csr.indices[start:end]
        data = csr.data[start:end]
        # drop diagonal if present
        mask = cols != i
        cols = cols[mask]
        data = data[mask]
        if cols.size == 0:
            ind[i, :] = -1
            vals[i, :] = np.inf if not largest else -np.inf
            continue
        if cols.size <= k:
            order = np.arange(cols.size)
        else:
            # choose smallest or largest k according to `largest`
            if largest:
                order = np.argpartition(-data, kth=k-1)[:k]
            else:
                order = np.argpartition(data, kth=k-1)[:k]
        chosen_cols = cols[order]
        chosen_vals = data[order]
        # If fewer than k, pad with placeholders
        if chosen_cols.size < k:
            pad = k - chosen_cols.size
            chosen_cols = np.pad(chosen_cols, (0, pad), constant_values=-1)
            filler = (-np.inf if largest else np.inf)
            chosen_vals = np.pad(chosen_vals, (0, pad), constant_values=filler)
        ind[i, :] = chosen_cols[:k]
        vals[i, :] = chosen_vals[:k]
    return ind, vals


def _knn_from_matrix(M: NDArray | spmatrix, k: int, *, mode: MatrixMode) -> Tuple[np.ndarray, np.ndarray]:
    """Compute kNN (indices, values) from a square distance/similarity matrix.

    Supports dense and sparse inputs without densifying sparse matrices.
    For `mode="distance"` the *smallest* values are selected; for
    `mode="similarity"` the *largest* values are selected. The diagonal is
    ignored.
    """
    if issparse(M):
        csr, n = _as_csr_square(M)
        largest = (mode == "similarity")
        ind, val = _topk_per_row_sparse(csr, k, largest=largest)
        return ind, val
    arr = np.asarray(M, dtype=float)

    # mask diagonal with +/- inf so it never gets picked
    arr = arr.copy()
    if mode == "distance":
        np.fill_diagonal(arr, np.inf)
        nn_idx = np.argpartition(arr, kth=k-1, axis=1)[:, :k]
    else:
        np.fill_diagonal(arr, -np.inf)
        nn_idx = np.argpartition(-arr, kth=k-1, axis=1)[:, :k]
    nn_val = np.take_along_axis(arr, nn_idx, axis=1)
    return nn_idx, nn_val
