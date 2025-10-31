import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm, colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from graphconstructor import Graph


def plot_graph_by_feature(
        G: Graph,
        color_attribute: str = None,
        attribute_type: str = "categorical",
        pos=None,
        cmap_name: str = "tab20",
        default_color="teal",
        with_labels: bool = True):
    """
    Color nodes by the selected attribute stored on each node (e.g., node['cf_class']).
    
    Parameters
    ----------
    G : Graph
        Graph whose nodes carry the class attribute.
    color_attribute : str
        Node attribute name with the class label (default: 'cf_class').
    attribute_type : str
        'categorical' or 'continuous'. This will determine the legend used.
    pos : dict or None
        Optional positions dict; if None, uses nx.spring_layout.
    cmap_name : str
        Matplotlib colormap (e.g., 'tab20' for categorical, 'viridis' for continuous).
    default_color : str
        Color for nodes missing the class attribute.
    with_labels : bool
        Draw node labels.
    """
    if attribute_type not in {"categorical", "continuous"}:
        raise ValueError("attribute_type must be 'categorical' or 'continuous'")
    
    nxG = G.to_networkx()
    node_list = list(nxG.nodes())
    
    # Initialize variables for continuous colorbar
    norm = None
    cmap_continuous = None
    
    if color_attribute:
        node_features = [nxG.nodes[n].get(color_attribute, None) for n in node_list]
        
        if attribute_type == "categorical":
            # Stable set of unique classes (preserve first-seen order, skip None)
            unique_classes = [c for c in dict.fromkeys(node_features) if c is not None]
            unique_classes.sort()
        
            # Map classes -> colors
            if unique_classes:
                cmap = colormaps.get_cmap(cmap_name, len(unique_classes))
                class_to_color = {c: cmap(i) for i, c in enumerate(unique_classes)}
            else:
                class_to_color = {}
        
            node_colors = [class_to_color.get(c, default_color) for c in node_features]
            
        elif attribute_type == "continuous":
            # Filter out None values to find min/max
            valid_values = [v for v in node_features if v is not None]
            
            if valid_values:
                # Convert to numeric (in case they aren't already)
                try:
                    valid_values = [float(v) for v in valid_values]
                    vmin, vmax = min(valid_values), max(valid_values)
                    
                    # Create normalization and colormap for continuous scale
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    cmap_continuous = colormaps.get_cmap(cmap_name)
                    
                    # Map node values to colors
                    node_colors = []
                    for val in node_features:
                        if val is not None:
                            try:
                                node_colors.append(cmap_continuous(norm(float(val))))
                            except (ValueError, TypeError):
                                node_colors.append(default_color)
                        else:
                            node_colors.append(default_color)
                    
                    unique_classes = True  # Flag to indicate we have valid data
                except (ValueError, TypeError):
                    # Fall back to default color if conversion fails
                    node_colors = [default_color] * len(node_list)
                    unique_classes = False
            else:
                node_colors = [default_color] * len(node_list)
                unique_classes = False
    else:
        node_colors = default_color
        unique_classes = False
    
    # Handle edge weights
    if G.weighted:
        edge_weights = [d.get("weight", 1.0) for _, _, d in nxG.edges(data=True)]
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [0.5 + 5.0 * (w / max_weight) for w in edge_weights]
            edge_colors = [cm.gray(w/max_weight) for w in edge_weights]
        else:
            edge_widths = 1.0
            edge_colors = "gray"
    else:
        edge_widths = 1.0
        edge_colors = "gray"
    
    # Node sizes based on degree
    degrees = dict(nxG.degree())
    node_sizes = [200.0 * (1.0 + np.sqrt(degrees.get(n, 0))) for n in nxG.nodes()]
    
    # Layout
    if pos is None:
        pos = nx.spring_layout(nxG, seed=42)
    
    # Figure size
    size = (len(node_list) ** 0.5)
    fig, ax = plt.subplots(figsize=(size, size))
    
    # Draw the graph
    nx.draw(
        nxG,
        pos=pos,
        ax=ax,
        with_labels=with_labels,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.85,
        linewidths=0.5,
        font_size=8,
    )
    
    # Legend or Colorbar
    if unique_classes:
        if attribute_type == "categorical":
            # Categorical legend with patches
            handles = [Patch(facecolor=class_to_color[c], edgecolor="none", label=str(c)) 
                      for c in unique_classes]
            ax.legend(handles=handles, title=color_attribute, loc="best", frameon=True)
        
        elif attribute_type == "continuous" and norm is not None and cmap_continuous is not None:
            # Continuous colorbar
            sm = cm.ScalarMappable(cmap=cmap_continuous, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(color_attribute, rotation=270, labelpad=15)
    
    ax.set_axis_off()
    fig.tight_layout()
    plt.show()
    
    return fig, ax
