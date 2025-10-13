import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.patches import Patch
from graphconstructor import Graph


def plot_graph_by_class(
        G: Graph,
        class_attr: str = None,
        pos=None,
        cmap_name: str = "tab20",
        default_color="teal",
        with_labels: bool = True):
    """
    Color nodes by the categorical attribute stored on each node (e.g., node['cf_class']).

    Parameters
    ----------
    G : Graph
        Graph whose nodes carry the class attribute.
    class_attr : str
        Node attribute name with the class label (default: 'cf_class').
    pos : dict or None
        Optional positions dict; if None, uses nx.spring_layout.
    cmap_name : str
        Matplotlib categorical colormap (e.g., 'tab20', 'tab10', 'Set3').
    default_color : str
        Color for nodes missing the class attribute.
    with_labels : bool
        Draw node labels.
    """
    nxG = G.to_networkx()

    # Collect class labels for nodes (in node order)
    node_list = list(nxG.nodes())
    if class_attr:
        node_classes = [nxG.nodes[n].get(class_attr, None) for n in node_list]

        # Stable set of unique classes (preserve first-seen order, skip None)
        unique_classes = [c for c in dict.fromkeys(node_classes) if c is not None]
    
        # Map classes -> colors
        if unique_classes:
            cmap = cm.get_cmap(cmap_name, len(unique_classes))
            class_to_color = {c: cmap(i) for i, c in enumerate(unique_classes)}
        else:
            class_to_color = {}
    
        node_colors = [class_to_color.get(c, default_color) for c in node_classes]
    else:
        node_colors = default_color
        unique_classes = False
    
    if G.weighted:
        edge_weights = [d.get("weight", 1.0) for _, _, d in nxG.edges(data=True)]
        # Scale edge widths for visibility; tweak as needed
        edge_widths = [0.5 + 5.0 * (w / max(edge_weights)) for w in edge_weights]

    # --- Node sizes (optional): use degree for a bit of visual structure ---
    degrees = dict(nxG.degree())
    # Scale size gently: 200 for degree 0, 200*(1+sqrt(deg)) otherwise
    node_sizes = [200.0 * (1.0 + np.sqrt(degrees.get(n, 0))) for n in nxG.nodes()]

    # Layout
    if pos is None:
        pos = nx.spring_layout(nxG, seed=42)

    # Figure size similar to your original heuristic
    size = (len(node_list) ** 0.5)
    fig, ax = plt.subplots(figsize=(size, size))

    nx.draw(
        nxG,
        pos=pos,
        ax=ax,
        with_labels=with_labels,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="gray" if not G.weighted else [cm.gray(w/max(edge_weights)) for w in edge_weights],
        width=1.0 if not G.weighted else edge_widths,
        alpha=0.85,
        linewidths=0.5,
        font_size=8,
    )

    # Legend
    if unique_classes:
        handles = [Patch(facecolor=class_to_color[c], edgecolor="none", label=str(c)) for c in unique_classes]
        ax.legend(handles=handles, title=class_attr, loc="best", frameon=True)

    ax.set_axis_off()
    fig.tight_layout()
    plt.show()
    return fig, ax
