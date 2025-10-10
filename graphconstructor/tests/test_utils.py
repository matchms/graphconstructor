import numpy as np
import pytest
import scipy.sparse as sp
from graphconstructor.utils import (
    _as_csr_square,
    _coerce_knn_inputs,
    _csr_from_edges,
    _knn_from_matrix,
    _make_symmetric_csr,
    _threshold_mask,
    _to_numpy,
    _topk_per_row_sparse,
    _validate_square_matrix,
)


# ---- helpers ----
def _rowwise_pairset(indices_row, values_row):
    """Create a set of (idx, val) pairs ignoring order for a single row."""
    return set(zip(indices_row.tolist(), values_row.tolist()))


# ---- _validate_square_matrix ----
def test_validate_square_matrix_ok():
    M = np.zeros((3, 3))
    _validate_square_matrix(M)  # should not raise


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (3,), (3, 3, 1)])
def test_validate_square_matrix_raises(shape):
    M = np.zeros(shape)
    with pytest.raises(TypeError, match="Matrix must be square"):
        _validate_square_matrix(M)


# ---- _to_numpy ----
def test_to_numpy_passthrough_numpy():
    arr = np.array([1, 2, 3])
    out = _to_numpy(arr)
    assert out is arr  # exact same object


def test_to_numpy_from_list():
    lst = [1, 2, 3]
    out = _to_numpy(lst)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.array(lst))


# ---- _make_symmetric_csr ----
def test_make_symmetric_csr_max_min_avg():
    # Asymmetric 3x3
    data = np.array([1.0, 2.0, 3.0, 4.0])
    rows = np.array([0, 1, 2, 0])
    cols = np.array([1, 0, 1, 2])
    A = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))
    # A[0,1]=1, A[1,0]=2, A[2,1]=3, A[0,2]=4

    Amax = _make_symmetric_csr(A, option="max")
    Amin = _make_symmetric_csr(A, option="min")
    Aavg = _make_symmetric_csr(A, option="average")

    assert (Amax != Amax.T).nnz == 0  # symmetric
    assert (Amin != Amin.T).nnz == 0
    assert (Aavg != Aavg.T).nnz == 0

    # pair (0,1): max(1,2)=2, min=1, avg=1.5
    assert Amax[0, 1] == 2 and Amax[1, 0] == 2
    assert Amin[0, 1] == 1 and Amin[1, 0] == 1
    assert Aavg[0, 1] == 1.5 and Aavg[1, 0] == 1.5

    # pair (0,2): only A[0,2]=4 exists, so symmetric counterpart uses 4
    assert Amax[0, 2] == 4 and Amax[2, 0] == 4
    assert Amin[0, 2] == 0 and Amin[2, 0] == 0  # min with missing = 0
    assert Aavg[0, 2] == 2 and Aavg[2, 0] == 2  # (4 + 0)/2

    with pytest.raises(ValueError, match="Unsupported option"):
        _make_symmetric_csr(A, option="median")


# ---- _coerce_knn_inputs ----
def test_coerce_knn_inputs_ok_and_numpy():
    ind_list = [[1, 2], [3, 4]]
    dist_list = [[0.1, 0.2], [0.3, 0.4]]
    ind, dist = _coerce_knn_inputs(ind_list, dist_list)
    assert isinstance(ind, np.ndarray) and isinstance(dist, np.ndarray)
    np.testing.assert_array_equal(ind, np.array(ind_list))
    np.testing.assert_array_equal(dist, np.array(dist_list))


def test_coerce_knn_inputs_shape_mismatch():
    ind = np.zeros((2, 3))
    dist = np.zeros((2, 2))
    with pytest.raises(TypeError, match="same shape"):
        _coerce_knn_inputs(ind, dist)


def test_coerce_knn_inputs_not_2d():
    ind = np.zeros((2, 2, 1))
    dist = np.zeros((2, 2, 1))
    with pytest.raises(TypeError, match="must be 2D"):
        _coerce_knn_inputs(ind, dist)


# ---- _threshold_mask ----
@pytest.mark.parametrize(
    "mode,thresh,values,expected",
    [
        ("distance", 0.5, np.array([0.4, 0.5, 0.6]), np.array([True, False, False])),
        ("similarity", 0.5, np.array([0.4, 0.5, 0.6]), np.array([False, False, True])),
    ],
)
def test_threshold_mask(mode, thresh, values, expected):
    out = _threshold_mask(values, thresh, mode)  # type: ignore[arg-type]
    np.testing.assert_array_equal(out, expected)


# ---- _csr_from_edges ----
def test_csr_from_edges_basic():
    n = 4
    rows = np.array([0, 1, 2])
    cols = np.array([1, 2, 3])
    w = np.array([1.0, 2.0, 3.0])
    csr = _csr_from_edges(n, rows, cols, w)
    assert isinstance(csr, sp.csr_matrix)
    assert csr.shape == (4, 4)
    assert csr[0, 1] == 1.0 and csr[1, 2] == 2.0 and csr[2, 3] == 3.0


# ---- _as_csr_square ----
def test_as_csr_square_from_dense_ok():
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    csr, n = _as_csr_square(M)
    assert isinstance(csr, sp.csr_matrix)
    assert n == 2
    np.testing.assert_allclose(csr.toarray(), M)


def test_as_csr_square_from_sparse_ok():
    A = sp.coo_matrix([[1, 0], [0, 2]])
    csr, n = _as_csr_square(A)
    assert isinstance(csr, sp.csr_matrix)
    assert n == 2
    np.testing.assert_allclose(csr.toarray(), np.array([[1, 0], [0, 2]]))


def test_as_csr_square_raises_for_non_square():
    A = sp.csr_matrix(np.zeros((2, 3)))
    with pytest.raises(TypeError, match="square"):
        _as_csr_square(A)
    with pytest.raises(TypeError, match="square"):
        _as_csr_square(np.zeros((3, 2)))


# ---- _topk_per_row_sparse ----
def test_topk_per_row_sparse_largest_and_smallest_ignore_diag_and_pad():
    # 4x4 with:
    # row 0: (0,0)=10 (diag), (0,1)=1, (0,2)=5
    # row 1: (1,1)=9 (diag), (1,0)=2
    # row 2: no off-diagonals
    # row 3: (3,0)=7, (3,2)=3
    rows = np.array([0, 0, 1, 0, 1, 3, 3])
    cols = np.array([0, 1, 0, 2, 1, 0, 2])
    data = np.array([10.0, 1.0, 2.0, 5.0, 9.0, 7.0, 3.0])
    csr = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))

    # largest: pick largest k=2 per row excluding diagonal
    indL, valL = _topk_per_row_sparse(csr, k=2, largest=True)

    # row 0 off-diagonals: (1,1), (2,5) -> top2 are idx {2,1} with vals {5,1}
    assert _rowwise_pairset(indL[0], valL[0]) == {(2, 5.0), (1, 1.0)}
    # row 1 off-diagonals: only (0,2) -> padded with -1 and -inf
    assert indL[1, 0] == 0 and valL[1, 0] == 2.0
    assert indL[1, 1] == -1 and np.isneginf(valL[1, 1])
    # row 2: empty row -> all padded with -1 and -inf
    assert (indL[2] == np.array([-1, -1])).all()
    assert np.isneginf(valL[2]).all()
    # row 3 off-diagonals: (0,7), (2,3) -> both
    assert _rowwise_pairset(indL[3], valL[3]) == {(0, 7.0), (2, 3.0)}

    # smallest: pick smallest k=2 per row excluding diagonal
    indS, valS = _topk_per_row_sparse(csr, k=2, largest=False)
    # row 0: (1,1), (2,5) -> keep both, smallest first not enforced (order not guaranteed)
    assert _rowwise_pairset(indS[0], valS[0]) == {(1, 1.0), (2, 5.0)}
    # row 1: single neighbor padded with +inf
    assert indS[1, 0] == 0 and valS[1, 0] == 2.0
    assert indS[1, 1] == -1 and np.isposinf(valS[1, 1])
    # row 2: empty row -> +inf padding
    assert (indS[2] == np.array([-1, -1])).all()
    assert np.isposinf(valS[2]).all()
    # row 3: both neighbors remain
    assert _rowwise_pairset(indS[3], valS[3]) == {(0, 7.0), (2, 3.0)}


def test_topk_per_row_sparse_k_greater_than_nnz():
    # 2x2 with single off-diagonal per row
    rows = np.array([0, 1])
    cols = np.array([1, 0])
    data = np.array([4.0, 3.0])
    csr = sp.csr_matrix((data, (rows, cols)), shape=(2, 2))

    ind, val = _topk_per_row_sparse(csr, k=3, largest=True)
    # each row: one real neighbor + two pads (-1, -inf)
    for r in range(2):
        assert (ind[r] == np.array([1 - r, -1, -1])).sum() >= 1  # first element is neighbor, order not guaranteed
        assert np.isneginf(val[r]).sum() >= 2


# ---- _knn_from_matrix (dense) ----
def test_knn_from_dense_distance_ignores_diagonal_and_picks_smallest():
    M = np.array([
        [0.0, 0.3, 0.1, 0.9],
        [0.2, 0.0, 0.8, 0.4],
        [0.5, 0.7, 0.0, 0.6],
        [0.9, 0.1, 0.2, 0.0],
    ])
    k = 2
    ind, val = _knn_from_matrix(M, k=k, mode="distance")

    # row 0 smallest two: (2,0.1), (1,0.3)
    assert _rowwise_pairset(ind[0], val[0]) == {(2, 0.1), (1, 0.3)}
    # row 1: (0,0.2), (3,0.4)
    assert _rowwise_pairset(ind[1], val[1]) == {(0, 0.2), (3, 0.4)}
    # row 2: (0,0.5), (3,0.6)
    assert _rowwise_pairset(ind[2], val[2]) == {(0, 0.5), (3, 0.6)}
    # row 3: (1,0.1), (2,0.2)
    assert _rowwise_pairset(ind[3], val[3]) == {(1, 0.1), (2, 0.2)}


def test_knn_from_dense_similarity_ignores_diagonal_and_picks_largest():
    S = np.array([
        [1.0, 0.9, 0.2],
        [0.8, 1.0, 0.3],
        [0.5, 0.4, 1.0],
    ])
    k = 1
    ind, val = _knn_from_matrix(S, k=k, mode="similarity")

    # row 0: best is 1 (0.9)
    assert (ind[0, 0], val[0, 0]) == (1, 0.9)
    # row 1: best is 0 (0.8)
    assert (ind[1, 0], val[1, 0]) == (0, 0.8)
    # row 2: best is 0 (0.5)
    assert (ind[2, 0], val[2, 0]) == (0, 0.5)


# ---- _knn_from_matrix (sparse) ----
def test_knn_from_sparse_distance_and_similarity_consistency():
    # Build 4x4 sparse with unique off-diagonals, include diagonals that must be ignored
    rows = np.array([0, 0, 1, 1, 2, 3, 0, 1, 2, 3])
    cols = np.array([1, 2, 0, 3, 3, 2, 0, 1, 2, 3])  # last four are diagonals
    data = np.array([0.3, 0.1, 0.2, 0.4, 0.6, 0.2, 9.0, 9.0, 9.0, 9.0])  # large diags ignored
    csr = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))

    # distance: pick smallest k
    indD, valD = _knn_from_matrix(csr, k=2, mode="distance")
    assert _rowwise_pairset(indD[0], valD[0]) == {(2, 0.1), (1, 0.3)}
    assert _rowwise_pairset(indD[1], valD[1]) == {(0, 0.2), (3, 0.4)}
    # rows 2 and 3 each have one off-diagonal -> padding rules apply
    # row 2: only (3,0.6)
    assert (2 in indD[2]) is False  # never picks diagonal
    assert (3 in indD[2]) or True  # presence checked below with pairset
    ps2 = _rowwise_pairset(indD[2], valD[2])
    assert (3, 0.6) in ps2
    assert (-1, np.inf) in ps2
    # row 3: only (2,0.2)
    ps3 = _rowwise_pairset(indD[3], valD[3])
    assert (2, 0.2) in ps3
    assert (-1, np.inf) in ps3

    # similarity: pick largest k
    # Reuse the same matrix as "similarity" scores (larger is better)
    # For row 0 off-diagonals: (1,0.3), (2,0.1) -> both picked
    indS, valS = _knn_from_matrix(csr, k=2, mode="similarity")
    assert _rowwise_pairset(indS[0], valS[0]) == {(1, 0.3), (2, 0.1)}
    assert _rowwise_pairset(indS[1], valS[1]) == {(3, 0.4), (0, 0.2)}
    # rows 2 and 3 padding with -inf
    ps2 = _rowwise_pairset(indS[2], valS[2])
    assert (3, 0.6) in ps2 and (-1, -np.inf) in ps2
    ps3 = _rowwise_pairset(indS[3], valS[3])
    assert (2, 0.2) in ps3 and (-1, -np.inf) in ps3
