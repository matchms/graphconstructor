from typing import Literal
import numpy as np
import scipy.sparse as sp
from .graph import Graph
from .utils import _coerce_knn_inputs


Mode = Literal["distance", "similarity"]

def from_dense(arr, mode, *, directed=False, weighted=True, meta=None, sym_op="max") -> Graph:
    return Graph.from_dense(arr, directed=directed, weighted=weighted, mode=mode, meta=meta, sym_op=sym_op)


def from_csr(adj, mode, *, directed=False, weighted=True, meta=None, sym_op="max") -> Graph:
    return Graph.from_csr(adj, directed=directed, weighted=weighted, mode=mode, meta=meta, sym_op=sym_op)


def from_knn(indices, distances, *, store_weights=True, directed=False, meta=None, sym_op="max") -> Graph:
    ind, dist = _coerce_knn_inputs(indices, distances)
    n_query, k = ind.shape

    # Rows correspond to query points [0..n_query-1]
    rows = np.repeat(np.arange(n_query), k)
    cols = ind.reshape(-1)
    weights = dist.reshape(-1)
    if not store_weights:
        weights = np.ones_like(weights, dtype=float)

    # Filter out padded neighbors (-1) if present
    valid = cols >= 0
    rows, cols, weights = rows[valid], cols[valid], weights[valid]

    # Infer full graph size from neighbor ids
    n_full = _infer_n_from_indices(ind)
    A = sp.csr_matrix((weights, (rows, cols)), shape=(n_full, n_full))
    return Graph.from_csr(A, directed=directed, weighted=store_weights, mode="distance",
                          meta=meta, sym_op=sym_op)


def from_ann(ann, query_data, k: int, *, store_weights=True, directed=False, meta=None, sym_op="max") -> Graph:
    idx = ann.index if hasattr(ann, "index") else ann
    if hasattr(idx, "indices_") and getattr(idx, "indices_") is not None:
        ind = np.asarray(getattr(idx, "indices_"))[:, :k]
        dist = np.asarray(getattr(idx, "distances_"))[:, :k]
    else:
        if query_data is None:
            raise TypeError("from_ann requires query_data when index has no cached neighbors.")
        ind, dist = idx.query(query_data, k=k)
    return from_knn(ind, dist, store_weights=store_weights, directed=directed,
                    meta=meta, sym_op=sym_op)


# helper functions ---------------------------------------------

def _infer_n_from_indices(ind: np.ndarray) -> int:
    """Infer full graph size from neighbor indices and number of query rows."""
    if ind.size == 0:
        return ind.shape[0]
    # ignore padding like -1
    max_idx = np.max(ind[ind >= 0]) if np.any(ind >= 0) else -1
    return max(ind.shape[0], int(max_idx) + 1)
