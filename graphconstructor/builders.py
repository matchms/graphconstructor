from typing import Optional
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix

from .constructors import (
    GraphConstructionConfig,
    KNNGraphConstructor,
    EpsilonBallGraphConstructor,
)
from .types import MatrixMode
from .adapters import ANNInput


# ---------- Functional convenience API ----------

def build_knn_graph(
    *,
    # Exactly one of the following input groups must be provided:
    matrix: Optional[NDArray | spmatrix] = None,
    mode: MatrixMode = "distance",
    indices: Optional[NDArray] = None,
    distances: Optional[NDArray] = None,
    ann: Optional[ANNInput] = None,
    ann_k: Optional[int] = None,
    # Parameters
    k: int,
    mutual: bool = False,
    config: Optional[GraphConstructionConfig] = None,
) -> csr_matrix:
    """Build a k-nearest neighbors (kNN) graph from one supported input form.

    This convenience wrapper constructs a :class:`~graphconstructor.core.KNNGraphConstructor`
    and dispatches to the appropriate builder based on the provided inputs.

    Parameters
    ----------
    matrix : ndarray | spmatrix, optional
        Square distance/similarity matrix of shape ``(n, n)``. Dense NumPy arrays
        and SciPy sparse matrices (e.g., CSR/CSC) are supported. For ``mode='distance'``,
        smaller values indicate closer neighbors; for ``mode='similarity'``, larger
        values indicate closer neighbors.
    mode : {"distance", "similarity"}, default="distance"
        Interpretation of the values in ``matrix`` (or subsequent computations).
    indices : ndarray, optional
        Precomputed neighbor indices of shape ``(n, k)``.
    distances : ndarray, optional
        Precomputed neighbor distances/similarities of shape ``(n, k)`` matching
        ``indices``. Required if ``indices`` is provided.
    ann : ANNInput, optional
        Approximate nearest neighbor (ANN) handle. If the underlying ANN object
        exposes ``indices_`` / ``distances_``, those are used; otherwise a query
        is issued using ``ann_k``.
    ann_k : int, optional
        Number of neighbors to request from the ANN when a live query is needed
        (i.e., when ``indices_``/``distances_`` are not present). If omitted and
        a query is required, a :class:`~graphconstructor.exceptions.InvalidInput`
        is raised.
    k : int
        Number of neighbors per node to connect in the kNN graph. When building
        from ``matrix`` or ``ann`` this also controls the truncation to ``k``
        neighbors per row.
    mutual : bool, default=False
        If ``True``, keep only reciprocal edges (i→j and j→i both present among
        their respective k nearest neighbors).
    config : GraphConstructionConfig, optional
        Global graph-construction options (symmetry policy, self-loops, etc.).

    Returns
    -------
    scipy.sparse.csr_matrix
        CSR adjacency matrix of shape ``(n, n)`` with weights taken from the
        corresponding distances/similarities, unless ``config.store_weights`` is
        ``False`` (in which case all weights are 1.0).

    Raises
    ------
    ValueError
        If not exactly one input group is provided.
    graphconstructor.exceptions.InvalidInput
        For invalid shapes, missing parameters for ANN queries, or non-square matrices.
    """
    ctor = KNNGraphConstructor(k=k, mutual=mutual, mode=mode, config=config)

    _check_input(matrix, indices, distances, ann)

    if matrix is not None:
        return ctor.from_matrix(matrix)
    if indices is not None and distances is not None:
        return ctor.from_knn(indices=indices, distances=distances)
    # ann is not None
    return ctor.from_ann(ann, k=ann_k)


def build_epsilon_ball_graph(
    *,
    # Exactly one of the following input groups must be provided:
    matrix: Optional[NDArray | spmatrix] = None,
    mode: MatrixMode = "distance",
    indices: Optional[NDArray] = None,
    distances: Optional[NDArray] = None,
    n: Optional[int] = None,
    ann: Optional[ANNInput] = None,
    ann_k: Optional[int] = None,
    # Parameter
    threshold: float,
    config: Optional[GraphConstructionConfig] = None,
) -> csr_matrix:
    """Build an ε-ball (or similarity-τ) graph from one supported input form.

    For ``mode='distance'``, connect (i, j) if ``distance(i, j) < threshold``.
    For ``mode='similarity'``, connect if ``similarity(i, j) > threshold``.

    Parameters
    ----------
    matrix : ndarray | spmatrix, optional
        Square distance/similarity matrix of shape ``(n, n)``. Dense and SciPy
        sparse matrices are supported without densification.
    mode : {"distance", "similarity"}, default="distance"
        Interpretation of values in ``matrix`` or provided distances/similarities.
    indices : ndarray, optional
        Precomputed neighbor indices of shape ``(m, k)`` obtained from a kNN routine.
        When using this path, only candidate pairs in these kNN lists are tested
        against ``threshold``.
    distances : ndarray, optional
        Distances/similarities of shape ``(m, k)`` to match ``indices``.
    n : int, optional
        Global number of nodes for the output graph if ``indices.shape[0]`` is not
        equal to ``n`` (e.g., when kNN was computed on a subset or query set). If
        omitted, defaults to ``indices.shape[0]``.
    ann : ANNInput, optional
        Approximate nearest neighbor handle. A kNN candidate set is taken from
        ``indices_``/``distances_`` if available, otherwise a query with ``ann_k``
        is executed; edges are then filtered by ``threshold``. **Note:** because
        ε-ball graphs can include neighbors outside the top-``k``, choose a large
        enough ``ann_k`` to avoid missing edges.
    ann_k : int, optional
        Number of neighbors to request from the ANN when a live query is required.
    threshold : float
        ε (for distances) or τ (for similarities).
    config : GraphConstructionConfig, optional
        Global graph-construction options (symmetry policy, self-loops, etc.).

    Returns
    -------
    scipy.sparse.csr_matrix
        CSR adjacency matrix of shape ``(n, n)``.

    Raises
    ------
    ValueError
        If not exactly one input group is provided.
    graphconstructor.exceptions.InvalidInput
        For invalid shapes, missing parameters for ANN queries, or non-square matrices.
    """
    ctor = EpsilonBallGraphConstructor(threshold=threshold, mode=mode, config=config)

    _check_input(matrix, indices, distances, ann)

    if matrix is not None:
        return ctor.from_matrix(matrix)
    if indices is not None and distances is not None:
        return ctor.from_knn(indices=indices, distances=distances, n=n)

    # For ε-ball via ANN we query k candidates then threshold.
    return ctor.from_ann(ann, k=ann_k)


# Basic helpers ---------------------------------

def _check_input(matrix, indices, distances, ann):
    """Helper to ensure exactly one input group is provided."""
    provided = sum([
        matrix is not None,
        (indices is not None and distances is not None),
        ann is not None,
    ])
    if provided != 1:
        raise ValueError("Provide exactly one of: matrix, (indices & distances), or ann.")
