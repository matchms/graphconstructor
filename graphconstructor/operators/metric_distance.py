from dataclasses import dataclass
from typing import Literal
import networkx as nx
from distanceclosure.dijkstra import single_source_dijkstra_path_length
from networkx.algorithms.shortest_paths.weighted import _weight_function
from ..graph import Graph
from .base import GraphOperator


Mode = Literal["distance", "similarity"]


@dataclass(slots=True)
class MetricDistanceFilter(GraphOperator):
    """
    Metric Distance Backbone filter for undirected weighted similarity or distance graphs.

    The method follows the distance backbone approach described in:
    Simas, T., Correia, R.B., & Rocha, L.M. (2021).
    "The distance backbone of complex networks."
    Journal of Complex Networks, 9(6), cnab021.  https://doi.org/10.1093/comnet/cnab021

    Parameters
    ----------
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    distortion : bool, optional
        Whether to compute and return distortion values, by default False
    verbose : bool, optional
        Prints statements as it computes, by default False
    """

    weight: str = "weight"
    distortion: bool = False
    verbose: bool = False
    mode: Mode = "distance"
    supported_modes = ["distance", "similarity"]

    @staticmethod
    def _compute_distortions(D: GraphOperator, B, weight="weight", disjunction=sum):
        G = D.copy()

        G.remove_edges_from(B.edges())
        weight_function = _weight_function(B, weight)

        svals = {}
        for u in G.nodes():
            metric_dist = single_source_dijkstra_path_length(
                B, source=u, weight_function=weight_function, disjunction=disjunction
            )
            for v in G.neighbors(u):
                svals[(u, v)] = G[u][v][weight] / metric_dist[v]

        return svals

    def _directed_filter(self, G: Graph) -> Graph:
        raise NotImplementedError("MetricDistanceFilter is defined only for undirected graphs.")

    def _undirected_filter(self, D):
        disjunction = sum

        # The backbone algorithm is defined for distances.
        if D.mode == "distance":
            D_distance = D
        else:
            D_distance = D.convert_mode("distance")

        D_nx = D_distance.to_networkx()
        G = D_nx.copy()
        weight_function = _weight_function(G, self.weight)

        if self.verbose:
            total = G.number_of_nodes()
            i = 0

        for u, _ in sorted(G.degree(weight=self.weight), key=lambda x: x[1]):
            if self.verbose:
                i += 1
                per = i / total
                print(f"Backbone: Dijkstra: {i} of {total} ({per:.2%})")

            metric_dist = single_source_dijkstra_path_length(
                G, source=u, weight_function=weight_function, disjunction=disjunction
            )
            for v in list(G.neighbors(u)):
                if metric_dist[v] < G[u][v][self.weight]:
                    G.remove_edge(u, v)

        sparse_adj = nx.to_scipy_sparse_array(G, weight=self.weight)

        filtered_graph = Graph(
            sparse_adj,
            directed=False,
            weighted=True,
            mode="distance",
            metadata=None if D.metadata is None else D.metadata.copy(),
        )

        # Optional output conversion.
        if self.mode == "similarity":
            filtered_graph = filtered_graph.convert_mode("similarity")

        if self.distortion:
            svals = self._compute_distortions(D_nx, G, weight=self.weight, disjunction=disjunction)
            return filtered_graph, svals
        else:
            return filtered_graph

    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)
        if G.directed:
            return self._directed_filter(G)
        else:
            return self._undirected_filter(G)
