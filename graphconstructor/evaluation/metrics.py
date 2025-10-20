from typing import Literal, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from graphconstructor import Graph


EdgeWeightMode = Literal["distance", "similarity", "unweighted"]
CorrKind = Literal["spearman", "pearson"]


def edge_jaccard(
    G1: Graph,
    G2: Graph,
    *,
    weighted: bool = False,
) -> float:
    """
    Jaccard similarity between the edge sets of two graphs built on the same node set.

    Parameters
    ----------
    G1, G2 : Graph
        Graphs to compare. Must have the same shape (same node set).
        If undirected, edges are interpreted as unordered pairs; we only count each once.
        If directed, edges are arcs; (i,j) and (j,i) are distinct.
    weighted : bool, default False
        - If False: binary Jaccard on presence/absence.
          J = |E1 ∩ E2| / |E1 ∪ E2|
        - If True : "generalized" (Tanimoto) Jaccard on weights:
          J = sum_{(i,j)∈E1∪E2} min(w1,w2) / sum_{(i,j)∈E1∪E2} max(w1,w2)
          Missing weights are treated as 0.

    Returns
    -------
    float in [0,1]
    """
    if G1.adj.shape != G2.adj.shape:
        raise ValueError("Graphs must have the same number of nodes to compare.")

    A = G1.adj
    B = G2.adj

    # For undirected, operate on upper triangles to avoid double counting.
    if not G1.directed and not G2.directed:
        A_use = sp.triu(A, k=1).tocsr()
        B_use = sp.triu(B, k=1).tocsr()
    else:
        A_use = A.tocsr()
        B_use = B.tocsr()

    if not weighted:
        # Binary Jaccard: use set operations via sparse structure.
        A_bin = A_use.sign()
        B_bin = B_use.sign()
        inter = A_bin.multiply(B_bin).nnz
        # union = nnz(A) + nnz(B) - nnz(intersection)
        union = A_bin.nnz + B_bin.nnz - inter
        if union == 0:
            return 1.0  # both empty => identical
        return inter / union

    # Weighted generalized Jaccard (Tanimoto)
    # Align on union index set: use COO to concatenate & coalesce.
    Acoo = A_use.tocoo()
    Bcoo = B_use.tocoo()
    # Build dict of weights for quick min/max accumulation
    # For large graphs, coalescing via CSR/CSC is faster: do elementwise ops.
    # min(A,B) and max(A,B) can be computed as:
    #   min = 0.5*(A+B - |A-B|), max = 0.5*(A+B + |A-B|)
    # But abs() on sparse is not directly supported; emulate using elementwise operations.
    # We'll construct union coordinates and fetch values.
    keys_A = np.stack([Acoo.row, Acoo.col], axis=1)
    keys_B = np.stack([Bcoo.row, Bcoo.col], axis=1)

    # Lexicographic order to merge
    def _lex_order(k):
        return np.lexsort((k[:, 1], k[:, 0]))

    oA = _lex_order(keys_A) if keys_A.size else np.array([], dtype=int)
    oB = _lex_order(keys_B) if keys_B.size else np.array([], dtype=int)

    keys_A = keys_A[oA] if keys_A.size else keys_A
    keys_B = keys_B[oB] if keys_B.size else keys_B
    vals_A = Acoo.data[oA] if Acoo.nnz else np.array([], dtype=float)
    vals_B = Bcoo.data[oB] if Bcoo.nnz else np.array([], dtype=float)

    iA = iB = 0
    num = 0.0
    den = 0.0
    nA = keys_A.shape[0]
    nB = keys_B.shape[0]

    while iA < nA or iB < nB:
        if iB >= nB or (iA < nA and (keys_A[iA, 0] < keys_B[iB, 0] or
                                     (keys_A[iA, 0] == keys_B[iB, 0] and keys_A[iA, 1] < keys_B[iB, 1]))):
            # key in A only
            w1 = float(vals_A[iA])
            num += 0.0
            den += w1
            iA += 1
        elif iA >= nA or (keys_B[iB, 0] < keys_A[iA, 0] or
                          (keys_B[iB, 0] == keys_A[iA, 0] and keys_B[iB, 1] < keys_A[iA, 1])):
            # key in B only
            w2 = float(vals_B[iB])
            num += 0.0
            den += w2
            iB += 1
        else:
            # shared
            w1 = float(vals_A[iA]); w2 = float(vals_B[iB])
            num += min(w1, w2)
            den += max(w1, w2)
            iA += 1; iB += 1

    if den == 0.0:
        return 1.0  # both empty => identical
    return num / den


def shortest_path_metric_correlation(
    G: Graph,
    M: Union[np.ndarray, sp.spmatrix],
    *,
    metric_mode: Literal["distance", "similarity"] = "distance",
    edge_weight_mode: EdgeWeightMode = "distance",
    sample_pairs: Optional[int] = None,
    correlation: CorrKind = "spearman",
    random_state: Optional[int] = None,
    similarity_eps: float = 1e-12,
) -> Tuple[float, float, int]:
    """
    Correlate graph shortest-path distances d_G(i,j) with the original metric M(i,j).

    Parameters
    ----------
    G : Graph
        Backbone graph (weighted or unweighted).
    M : array-like (n x n) dense or sparse
        Original metric between nodes. If metric_mode="similarity", M is similarity (larger=closer).
        If metric_mode="distance", M is distance (smaller=closer).
    metric_mode : {"distance","similarity"}, default "distance"
        How to interpret M.
    edge_weight_mode : {"distance","similarity","unweighted"}, default "distance"
        How to interpret G.adj weights for shortest-path cost:
        - "distance": use weights as path costs directly.
        - "similarity": use cost = 1 / (w + similarity_eps) on nonzeros.
        - "unweighted": cost of every present edge = 1.0.
    sample_pairs : int, optional
        If provided, randomly sample this many unordered pairs (i<j) from the giant component
        to estimate correlation. Otherwise, use all pairs (upper triangle), which is O(n^2).
    correlation : {"spearman","pearson"}, default "spearman"
        Which correlation to compute between flattened vectors.
    random_state : int, optional
        Seed for pair sampling reproducibility.
    similarity_eps : float, default 1e-12
        Stabilizer when inverting similarities (both for M and for edge weights if needed).

    Returns
    -------
    (rho, pval, n_pairs_used)
        rho : correlation coefficient
        pval: two-sided p-value from scipy.stats
        n_pairs_used: number of pairs used in the calculation

    Notes
    -----
    - Infinite shortest-path distances (disconnected pairs) and diagonal entries are excluded.
    - For undirected graphs, pairs are taken with i<j.
    - If M is similarity, it is converted to distances via 1/(M+eps) elementwise.
    """
    n = G.n_nodes
    if M.shape[0] != n or M.shape[1] != n:
        raise ValueError("M must be an n x n matrix matching the graph size.")

    # Build edge cost matrix for shortest paths
    if edge_weight_mode == "unweighted":
        cost = G.adj.copy().astype(float)
        cost.data[:] = 1.0
    elif edge_weight_mode == "distance":
        cost = G.adj.copy().astype(float)
    elif edge_weight_mode == "similarity":
        cost = G.adj.copy().astype(float)
        # cost = 1 / (w + eps) on existing edges
        cost.data = 1.0 / (cost.data + similarity_eps)
    else:
        raise ValueError("edge_weight_mode must be 'distance', 'similarity', or 'unweighted'.")

    # Shortest paths on the (possibly directed) graph
    dG = shortest_path(
        csgraph=cost,
        directed=G.directed,
        return_predecessors=False,
        unweighted=False,
        overwrite=False,
    )
    # Exclude diagonal and infs later

    # Convert M to distances array
    if sp.issparse(M):
        M_arr = M.toarray()
    else:
        M_arr = np.asarray(M, dtype=float)

    if metric_mode == "distance":
        D = M_arr
    elif metric_mode == "similarity":
        D = 1.0 / (M_arr + similarity_eps)
    else:
        raise ValueError("metric_mode must be 'distance' or 'similarity'.")

    # We'll compare only i<j (undirected sense) to avoid duplicate pairs, regardless of G.directed.
    iu, ju = np.triu_indices(n, k=1)

    # Mask out infinite graph distances
    mask = np.isfinite(dG[iu, ju])
    iu = iu[mask]; ju = ju[mask]
    if iu.size == 0:
        raise ValueError("No finite shortest paths between any node pairs; graph may be disconnected.")

    # Optional sampling
    if sample_pairs is not None and iu.size > sample_pairs:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(iu.size, size=sample_pairs, replace=False)
        iu = iu[sel]; ju = ju[sel]

    x = D[iu, ju].ravel()
    y = dG[iu, ju].ravel()

    # If there are NaNs in M, drop those pairs
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]; y = y[valid]
    if x.size < 2:
        raise ValueError("Not enough valid pairs to compute correlation.")

    if correlation == "spearman":
        rho, p = stats.spearmanr(x, y)
    elif correlation == "pearson":
        rho, p = stats.pearsonr(x, y)
    else:
        raise ValueError("correlation must be 'spearman' or 'pearson'.")

    return float(rho), float(p), int(x.size)
