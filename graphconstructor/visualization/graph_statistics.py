from typing import Iterable, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_degree_distribution(
    graph,
    *,
    x_scale: str = "log",
    y_scale: str = "log",
    ax: Optional[plt.Axes] = None,
    normalize: bool = True,
    include_zero_degree: bool = False,
    label: Optional[str] = None,
    marker: str = "o",
    markersize: float = 5.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the degree distribution p(k) vs k for a single graph.

    Parameters
    ----------
    graph
        Object exposing `.degree()` -> 1-D numpy array of node degrees.
    x_scale, y_scale
        Matplotlib scales ("linear", "log", "symlog", etc.). Defaults "log" / "log".
    ax
        Optional axes to draw on. If None, a new figure and axes are created.
    normalize
        If True, plot probability mass p(k) = count(k) / N. If False, plot raw counts.
    include_zero_degree
        If True, include k=0 in the plot (useful if not using log scale).
        Note: when using log scale, k=0 cannot be shown and will be dropped.
    label
        Optional label for legend.
    marker
        Matplotlib marker style for the scatter points.
    markersize
        Size of scatter markers.

    Returns
    -------
    (fig, ax)
        The Matplotlib figure and axes used for plotting.

    Notes
    -----
    - When `x_scale` or `y_scale` is "log", any k=0 or p(k)=0 entries are removed
      to avoid invalid values on a log axis.
    """
    degrees = graph.degree(ignore_weights=True)
    if degrees.ndim != 1:
        raise ValueError("graph.degree(ignore_weights=True) must return a 1-D array of degrees.")
    if degrees.size == 0:
        # Create empty plot but still return fig/ax for consistency
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.set_xlabel("Degree k")
        ax.set_ylabel("p(k)" if normalize else "count(k)")
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_title(getattr(graph, "name", "Degree distribution (empty graph)"))
        return fig, ax

    if np.any(degrees < 0):
        raise ValueError("Degrees must be nonnegative.")

    # Compute counts per unique degree efficiently
    # Using np.bincount for speed (requires nonnegative ints)
    if not np.issubdtype(degrees.dtype, np.integer):
        # allow float degrees that are whole numbers
        if np.all(np.mod(degrees, 1) == 0):
            degrees = degrees.astype(int)
        else:
            raise ValueError("Degrees must be integers for degree distributions.")

    counts = np.bincount(degrees)  # index = k, value = count(k)
    freqs = counts.astype(float)

    if normalize:
        total = freqs.sum()
        if total == 0:
            raise ValueError("No nodes with valid degrees found.")
        pk = freqs / total
    else:
        pk = freqs

    # Handle zero-degree inclusion/exclusion
    if not include_zero_degree or x_scale == "log":
        ks = np.nonzero(counts)[0]
        pk = pk[ks]
    else:
        ks = np.arange(len(counts))

    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(ks, pk, marker=marker, s=markersize**2, label=label)
    ax.set_xlabel("Degree k")
    ax.set_ylabel("p(k)" if normalize else "count(k)")
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    title_default = getattr(graph, "name", None)
    if title_default:
        ax.set_title(f"Degree distribution: {title_default}")
    else:
        ax.set_title("Degree distribution")

    if label is not None:
        ax.legend()

    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    return fig, ax


def plot_degree_distributions_grid(
    graphs: Iterable,
    *,
    ncols: int = 3,
    x_scale: str = "log",
    y_scale: str = "log",
    normalize: bool = True,
    include_zero_degree: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    tight_layout: bool = True,
    sharex: bool = False,
    sharey: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot a grid of degree distribution plots for multiple graphs.

    Parameters
    ----------
    graphs
        Iterable of Graph-like objects exposing `.degree() -> np.ndarray`.
    ncols
        Number of columns in the grid.
    x_scale, y_scale
        Axis scales for all subplots (defaults to "log").
    normalize
        If True, plot probability mass p(k). If False, raw counts.
    include_zero_degree
        Whether to include k=0 (will be dropped automatically on log x).
    figsize
        Optional figure size. If None, inferred from grid size.
    tight_layout
        Whether to call `fig.tight_layout()` at the end.
    sharex, sharey
        Whether to share x/y axes across subplots.

    Returns
    -------
    (fig, axes)
        Figure and 2D ndarray of Axes (some entries may be unused if the grid
        is larger than the number of graphs).
    """
    graphs = list(graphs)
    n = len(graphs)
    if n == 0:
        raise ValueError("`graphs` must contain at least one graph.")

    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        # heuristic: wider for more columns, taller for more rows
        figsize = (4 * ncols, 3.2 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey)

    # Ensure axes is 2D array for consistent indexing
    if isinstance(axes, plt.Axes):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)

    # Plot each graph
    for i, graph in enumerate(graphs):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        label = getattr(graph, "label", None)  # optional custom label field
        plot_degree_distribution(
            graph,
            x_scale=x_scale,
            y_scale=y_scale,
            ax=ax,
            normalize=normalize,
            include_zero_degree=include_zero_degree,
            label=label,
        )
        # Prefer a clean, concise title per subplot
        title = getattr(graph, "name", None) or f"Graph {i+1}"
        ax.set_title(title)

    # Hide any unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    if tight_layout:
        fig.tight_layout()

    return fig, axes
