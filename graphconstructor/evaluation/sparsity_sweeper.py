from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from graphconstructor import Graph
from graphconstructor.operators import GraphOperator


Monotonicity = Literal["increasing", "decreasing", "unknown"]


@dataclass(slots=True)
class BackboneSpec:
    """
    Specification for sweeping a backbone operator over a scalar parameter.

    Parameters
    ----------
    name : str
        A human-readable name for the curve/legend.
    build : Callable[[float], GraphOperator]
        A factory that returns an initialized operator when called with the sweep parameter.
        Example: lambda alpha: DisparityFilter(alpha=alpha)
    grid : Iterable[float]
        The parameter values to sweep (in desired order).
    monotonic : {"increasing","decreasing","unknown"}, default "unknown"
        Expected relationship between parameter and *number of edges* in the output graph,
        used for interpolation to a target |E|. If unknown, we'll fall back to nearest grid value.
    meta : dict, optional
        Arbitrary extra info to keep with the spec (not used by the sweeper).
    """
    name: str
    build: Callable[[float], GraphOperator]
    grid: Iterable[float]
    monotonic: Monotonicity = "unknown"
    meta: Optional[Dict[str, Any]] = None


class SparsitySweeper:
    """
    Run parameter sweeps for multiple operators on a fixed input Graph,
    collect edge counts, compute overlap, and plot curves.

    Notes
    -----
    - Assumes each operator's `apply(Graph)` returns a new `Graph`.
    - Edge count is computed as:
        - directed: nnz
        - undirected: nnz/2 (since adjacency is symmetric)
    - No graph densification involved; we trust operators to be sparse-friendly.
    """

    def __init__(self, specs: List[BackboneSpec]) -> None:
        if not specs:
            raise ValueError("At least one BackboneSpec is required.")
        self.specs = specs

    @staticmethod
    def _edge_count(G: Graph) -> int:
        nnz = int(G.adj.nnz)
        if G.directed:
            return nnz
        return nnz // 2

    def sweep(self, G: Graph) -> Dict[str, pd.DataFrame]:
        """
        Run the sweep for all specs on the same input Graph.

        Returns
        -------
        dict name -> DataFrame with columns:
            ["param", "n_edges", "directed", "weighted"]
        """
        results: Dict[str, pd.DataFrame] = {}
        for spec in self.specs:
            rows = []
            for p in spec.grid:
                op = spec.build(float(p))
                Gp = op.apply(G)
                rows.append(
                    {
                        "param": float(p),
                        "n_edges": self._edge_count(Gp),
                        "directed": bool(Gp.directed),
                        "weighted": bool(Gp.weighted),
                    }
                )
            df = pd.DataFrame(rows).sort_values("param").reset_index(drop=True)
            results[spec.name] = df
        return results

    @staticmethod
    def overlap_range(results: Dict[str, pd.DataFrame]) -> Optional[Tuple[int, int]]:
        """
        Compute the intersection of attainable |E| ranges across all methods.

        Returns
        -------
        (emin, emax) if overlap exists, otherwise None.
        """
        if not results:
            return None
        mins = []
        maxs = []
        for df in results.values():
            if df.empty:
                return None
            mins.append(int(df["n_edges"].min()))
            maxs.append(int(df["n_edges"].max()))
        lo = max(mins)
        hi = min(maxs)
        if lo <= hi:
            return (lo, hi)
        return None

    def interpolate_param_for_edges(
        self, df: pd.DataFrame, target_edges: int, monotonic: Monotonicity
    ) -> float:
        """
        Given a (param, n_edges) DataFrame for one method and a target |E|,
        return an interpolated parameter value.

        Assumes n_edges is monotonic in param when `monotonic` is not "unknown".
        Otherwise returns the param of the closest observed n_edges in df.

        Linear interpolation is performed in param-vs-edges space.

        Returns
        -------
        float param_estimate
        """
        # Fast path: if unknown monotonicity, choose nearest neighbor in |E|
        if monotonic == "unknown" or df.shape[0] < 2:
            idx = (df["n_edges"] - target_edges).abs().argmin()
            return float(df.loc[idx, "param"])

        # Ensure sorting in the direction matching monotonicity
        if monotonic == "increasing":
            dsorted = df.sort_values("param")
        else:
            dsorted = df.sort_values("param", ascending=False)

        # Locate the sandwich around target_edges
        n = dsorted["n_edges"].to_numpy()
        p = dsorted["param"].to_numpy()

        # If outside range, clamp to boundary param
        if target_edges <= n.min():
            return float(p[n.argmin()])
        if target_edges >= n.max():
            return float(p[n.argmax()])

        # Find first index where n_edges >= target (assuming increasing with param)
        # For decreasing, we sorted descending so n still "increases" along p array.
        idx = np.searchsorted(n, target_edges, side="left")
        i0 = max(idx - 1, 0)
        i1 = min(idx, len(n) - 1)

        if i0 == i1:
            return float(p[i0])

        # Linear interpolation in (n_edges -> param)
        n0, n1 = float(n[i0]), float(n[i1])
        p0, p1 = float(p[i0]), float(p[i1])

        # avoid division by zero
        if abs(n1 - n0) < 1e-12:
            return p0

        t = (target_edges - n0) / (n1 - n0)
        return p0 + t * (p1 - p0)

    def plot_sweeps(
        self,
        results: Dict[str, pd.DataFrame],
        figsize: Tuple[float, float] = (7.5, 5.0),
        sharey: bool = False,
        title: Optional[str] = "Edge-count sweeps per method",
    ) -> plt.Figure:
        """
        Plot one subplot per method: x = |E|, y = parameter value.
        Shade the vertical band corresponding to the overlap range if present.
        """
        n_methods = len(results)
        fig, axes = plt.subplots(
            n_methods, 1, figsize=figsize, sharex=True, sharey=sharey
        )
        if n_methods == 1:
            axes = [axes]

        overlap = self.overlap_range(results)

        for ax, (name, df) in zip(axes, results.items()):
            ax.plot(df["n_edges"].to_numpy(), df["param"].to_numpy(), marker="o")
            ax.set_ylabel(f"{name}\nparam", rotation=0, ha="right", va="center")
            ax.grid(True, alpha=0.3)

            if overlap is not None:
                lo, hi = overlap
                ax.axvspan(lo, hi, alpha=0.15)

        axes[-1].set_xlabel("|E| (number of edges)")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        return fig
