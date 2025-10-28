from typing import Callable, Literal, Tuple, Union
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse, spmatrix
from .types import MatrixMode


# Type aliases for clarity
Mode = Literal["distance", "similarity"]
ConversionMethod = Union[
    Literal["reciprocal", "negative", "exp", "gaussian"],
    Callable[[np.ndarray], np.ndarray]
    ]


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


def _drop_diagonal(A: sp.csr_matrix) -> sp.csr_matrix:
    # Remove diagonal entries without touching other zeros in csr matrix
    coo = A.tocoo()
    mask = coo.row != coo.col
    return sp.csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=A.shape)


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


def convert_weights(
    weights: np.ndarray,
    source_mode: Mode,
    target_mode: Mode,
    method: ConversionMethod = "reciprocal",
    **kwargs
) -> np.ndarray:
    """
    Convert weight values between distance and similarity representations.
    
    Parameters
    ----------
    weights : np.ndarray
        Array of weight values to convert.
    source_mode : {"distance", "similarity"}
        The current interpretation of weights.
    target_mode : {"distance", "similarity"}
        The desired interpretation of weights.
    method : str or callable, default="reciprocal"
        Conversion method to use. Built-in options:
        
        - "reciprocal": w_new = 1 / w_old (handles zeros with small epsilon)
        - "negative": w_new = -w_old (useful for optimization contexts)
        - "exp": w_new = exp(-w_old) (distance to similarity only)
        - "gaussian": w_new = exp(-w_old^2 / (2*sigma^2)) (distance to similarity only)
        
        Or provide a custom callable: f(weights) -> converted_weights
    **kwargs
        Additional arguments passed to conversion functions:
        
        - epsilon : float, default=1e-10
            Small value added to denominators to avoid division by zero.
        - sigma : float, default=1.0
            Bandwidth parameter for "gaussian" method.
    """
    if source_mode == target_mode:
        raise ValueError(
            f"source_mode and target_mode are both '{source_mode}'. "
            "No conversion needed."
        )
    
    if source_mode not in ("distance", "similarity"):
        raise ValueError(f"Invalid source_mode: '{source_mode}'")
    if target_mode not in ("distance", "similarity"):
        raise ValueError(f"Invalid target_mode: '{target_mode}'")
    
    # Handle custom callable
    if callable(method):
        return method(weights)
    
    # Handle built-in methods
    epsilon = kwargs.get("epsilon", 1e-10)

    if method == "reciprocal":
        # Works in both directions
        return 1.0 / (weights + epsilon)

    elif method == "negative":
        # Works in both directions
        return -weights

    elif method == "exp":
        # Only distance -> similarity
        if source_mode != "distance":
            raise ValueError(
                "Method 'exp' only supports distance -> similarity conversion"
            )
        return np.exp(-weights)

    elif method == "gaussian":
        # Only distance -> similarity
        if source_mode != "distance":
            raise ValueError(
                "Method 'gaussian' only supports distance -> similarity conversion"
            )
        sigma = kwargs.get("sigma", 1.0)
        return np.exp(-weights**2 / (2 * sigma**2))

    else:
        raise ValueError(
            f"Unknown conversion method: '{method}'. "
            "Use 'reciprocal', 'negative', 'exp', 'gaussian', or provide a callable."
        )


def convert_adjacency_mode(
    adj: csr_matrix,
    source_mode: Mode,
    target_mode: Mode,
    method: ConversionMethod = "reciprocal",
    inplace: bool = False,
    **kwargs
) -> csr_matrix:
    """
    Convert a sparse adjacency matrix between distance and similarity modes.
    
    Only the non-zero elements (edges) are converted. Zero elements (non-edges)
    remain zero.
    
    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Sparse adjacency matrix to convert.
    source_mode : {"distance", "similarity"}
        Current interpretation of edge weights.
    target_mode : {"distance", "similarity"}
        Desired interpretation of edge weights.
    method : str or callable, default="reciprocal"
        Conversion method. See `convert_weights` for options.
    inplace : bool, default=False
        If True, modify the matrix in place (faster, but mutates input).
        If False, create a copy before converting.
    **kwargs
        Additional arguments passed to conversion function.
    """
    if not issparse(adj):
        raise TypeError("adj must be a sparse matrix")
    
    if source_mode == target_mode:
       raise ValueError(
            f"source_mode and target_mode are both '{source_mode}'. "
            "No conversion needed."
        )
    
    # Work with CSR format for efficiency
    if not isinstance(adj, csr_matrix):
        adj = adj.tocsr()
        inplace = False  # Can't modify in place after format conversion
    
    if not inplace:
        adj = adj.copy()
    
    # Convert only the non-zero edge weights
    adj.data = convert_weights(
        adj.data,
        source_mode=source_mode,
        target_mode=target_mode,
        method=method,
        **kwargs
    )
    
    return adj
