from dataclasses import dataclass
from typing import Iterable, Literal, Sequence
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


SymOp = Literal["max", "min", "average"]


@dataclass(slots=True)
class Graph:
    """Sparse (CSR) graph with optional node metadata.

    - `adj`: CSR adjacency of shape (n, n)
    - `directed`: True if directed, else undirected (stored symmetric)
    - `weighted`: True if edge weights are meaningful; if False, all edges are 1.0
    - `meta`: pandas DataFrame with n rows (optional). May have a 'name' column.
    """
    adj: sp.csr_matrix
    directed: bool
    weighted: bool
    meta: pd.DataFrame | None = None

    # -------- Construction helpers --------
    @staticmethod
    def _ensure_csr(M: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
        if sp.issparse(M):
            return M.tocsr(copy=False)
        arr = np.asarray(M, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise TypeError("Adjacency must be square (n x n).")
        return sp.csr_matrix(arr)

    @staticmethod
    def _symmetrize(A: sp.csr_matrix, how: SymOp = "max") -> sp.csr_matrix:
        if how == "max":
            return A.maximum(A.T)
        if how == "min":
            return A.minimum(A.T)
        if how == "average":
            return (A + A.T) * 0.5
        raise ValueError("Unsupported symmetrization op. Use 'max', 'min', or 'average'.")

    @classmethod
    def from_csr(
        cls,
        adj: sp.spmatrix | np.ndarray,
        *,
        directed: bool = False,
        weighted: bool = True,
        meta: pd.DataFrame | None = None,
        sym_op: SymOp = "max",
        copy: bool = False,
    ) -> "Graph":
        A = cls._ensure_csr(adj)
        if not weighted:
            A = A.copy() if (sp.issparse(adj) and not copy) else (A if copy else A)
            A.data[:] = 1.0
        if not directed:
            A = cls._symmetrize(A, how=sym_op)
        # drop self-loops by default (TODO: maybe add flag later?)
        if A.diagonal().any():
            A = A.tolil(copy=False)
            A.setdiag(0)
            A = A.tocsr(copy=False)
            A.eliminate_zeros()

        n = A.shape[0]
        if meta is not None:
            if len(meta) != n:
                raise ValueError(f"meta has {len(meta)} rows but adjacency is {n}x{n}.")
            meta = meta.reset_index(drop=True)
        return cls(adj=A.astype(float, copy=False), directed=directed, weighted=weighted, meta=meta)

    @classmethod
    def from_dense(
        cls,
        adj: np.ndarray,
        **kwargs,
    ) -> "Graph":
        return cls.from_csr(adj, **kwargs)

    @classmethod
    def from_edges(
        cls,
        n: int,
        edges: Sequence[tuple[int, int]] | np.ndarray,
        weights: Sequence[float] | np.ndarray | None = None,
        *,
        directed: bool = False,
        weighted: bool = True,
        meta: pd.DataFrame | None = None,
        sym_op: SymOp = "max",
    ) -> "Graph":
        """Build from an edge list. For undirected=True, we symmetrize later."""
        if isinstance(edges, np.ndarray):
            if edges.ndim != 2 or edges.shape[1] != 2:
                raise TypeError("edges ndarray must be shape (m, 2).")
            rows = edges[:, 0].astype(int, copy=False)
            cols = edges[:, 1].astype(int, copy=False)
        else:
            rows, cols = map(np.asarray, zip(*edges)) if edges else (np.array([], int), np.array([], int))

        if weights is None:
            data = np.ones_like(rows, dtype=float)
            # edges w/o weights → unweighted
            weighted_eff = False if not weighted else False if weights is None else True
        else:
            data = np.asarray(weights, dtype=float)
            if data.shape[0] != rows.shape[0]:
                raise ValueError("weights length must match number of edges.")
            weighted_eff = weighted

        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        # if user declared weighted=False, force ones
        if not weighted:
            A.data[:] = 1.0
        return cls.from_csr(A, directed=directed, weighted=weighted_eff, meta=meta, sym_op=sym_op)

    # -------- Core properties --------
    @property
    def n_nodes(self) -> int:
        return self.adj.shape[0]

    @property
    def n_edges(self) -> int:
        if self.directed:
            # exclude diagonal, count every stored arc
            diag_nnz = int(np.count_nonzero(self.adj.diagonal()))
            return int(self.adj.nnz - diag_nnz)
        # undirected: count upper triangle only
        return int(sp.triu(self.adj, k=1).nnz)

    @property
    def has_self_loops(self) -> bool:
        return bool(self.adj.diagonal().any())

    @property
    def node_names(self) -> list[str] | list[int]:
        if self.meta is not None and "name" in self.meta.columns:
            return self.meta["name"].tolist()
        return list(range(self.n_nodes))

    # -------- Editing --------
    def drop(self, nodes: Iterable[int | str]) -> "Graph":
        """Drop nodes by index or name. Returns a *new* Graph (immutable style)."""
        if not nodes:
            return self
        if isinstance(nodes, (int, str)):
            nodes = [nodes]

        to_drop_idx: set[int] = set()
        if self.meta is not None and "name" in self.meta.columns:
            name_to_idx = {name: i for i, name in enumerate(self.meta["name"].tolist())}
        else:
            name_to_idx = {}

        for x in nodes:
            if isinstance(x, str):
                if x not in name_to_idx:
                    raise KeyError(f"Node name '{x}' not found.")
                to_drop_idx.add(name_to_idx[x])
            else:
                if x < 0 or x >= self.n_nodes:
                    raise IndexError(f"Node index {x} out of range [0, {self.n_nodes}).")
                to_drop_idx.add(int(x))

        if not to_drop_idx:
            return self

        keep_mask = np.ones(self.n_nodes, dtype=bool)
        keep_mask[list(to_drop_idx)] = False

        A2 = self.adj[keep_mask][:, keep_mask].tocsr(copy=False)
        meta2 = self.meta.loc[keep_mask].reset_index(drop=True) if self.meta is not None else None
        return Graph(adj=A2, directed=self.directed, weighted=self.weighted, meta=meta2)

    # -------- Exporters --------
    def to_networkx(self):
        """Return nx.Graph or nx.DiGraph with node attributes from metadata (if any)."""
        try:
            import networkx as nx  # lazy import
        except Exception as e:
            raise ImportError("networkx is required for to_networkx().") from e

        create_using = nx.DiGraph if self.directed else nx.Graph
        G = nx.from_scipy_sparse_array(self.adj, create_using=create_using)
        # attach node attributes from meta
        if self.meta is not None:
            for col in self.meta.columns:
                nx.set_node_attributes(G, {i: self.meta.iloc[i, self.meta.columns.get_loc(col)]
                                           for i in range(self.n_nodes)}, name=col)
        return G

    def to_igraph(self):
        """Return igraph.Graph with 'weight' edge attribute (if weighted) and node attributes from metadata."""
        try:
            import igraph as ig  # python-igraph
        except Exception as e:
            raise ImportError("python-igraph is required for to_igraph().") from e

        # Build edge list and weights
        coo = self.adj.tocoo()
        # remove self loops to match usual semantics
        mask = coo.row != coo.col
        rows = coo.row[mask]
        cols = coo.col[mask]
        weights = coo.data[mask] if self.weighted else np.ones(mask.sum(), dtype=float)

        g = ig.Graph(n=self.n_nodes, directed=self.directed)
        g.add_edges(list(zip(rows.tolist(), cols.tolist())))
        if self.weighted:
            g.es["weight"] = weights.tolist()
        else:
            g.es["weight"] = [1.0] * len(rows)

        # node attributes
        if self.meta is not None:
            for col in self.meta.columns:
                g.vs[col] = self.meta[col].tolist()
        return g

    # -------- Utilities --------
    def copy(self) -> "Graph":
        return Graph(
            adj=self.adj.copy(),
            directed=self.directed,
            weighted=self.weighted,
            meta=None if self.meta is None else self.meta.copy(),
        )

    def sorted_by(self, col: str) -> "Graph":
        """Return a new graph with nodes permuted by ascending meta[col]."""
        if self.meta is None or col not in self.meta.columns:
            raise KeyError(f"Column '{col}' not found in metadata.")
        order = np.argsort(self.meta[col].to_numpy())
        P = sp.identity(self.n_nodes, format="csr")[order]  # permutation matrix as row selector
        A2 = (P @ self.adj @ P.T).tocsr(copy=False)
        meta2 = self.meta.iloc[order].reset_index(drop=True)
        return Graph(adj=A2, directed=self.directed, weighted=self.weighted, meta=meta2)

    def degree(self, ignore_weights: bool = False) -> np.ndarray:
        """Return (out-)degree for directed, degree for undirected. For weighted graphs sum of weights.
        
        Parameters
        ----------
        ignore_weights
            If True, count number of edges only (treat as unweighted).
            Default is False.
        """
        if self.weighted and not ignore_weights:
            deg = np.asarray(self.adj.sum(axis=1)).ravel()
        else:
            # count nonzeros per row
            deg = np.diff(self.adj.indptr).astype(float)
        if not self.directed:
            return deg
        return deg  # could also return (out_degree, in_degree) if desired
    
    def is_connected(self) -> bool:
        """Return True if the graph is connected (undirected) or strongly connected (directed)."""
        n_components = connected_components(
            self.adj,
            directed=self.directed,
            connection="strong" if self.directed else "weak",
            return_labels=False
            )
        return n_components == 1

    def connected_components(self, return_labels: bool = False) -> bool:
        """Return the number of connected components or labels per node.
        For directed graphs, strongly connected components are returned.
        
        Parameters
        ----------
        return_labels
            If True, return an array of component labels per node instead of the number of components.
            Default is False.
        """
        return connected_components(
            self.adj,
            directed=self.directed,
            connection="strong" if self.directed else "weak",
            return_labels=return_labels
            )
