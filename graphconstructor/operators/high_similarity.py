from dataclasses import dataclass
from typing import Literal
import networkx as nx
from ..graph import Graph
from .base import GraphOperator


Method = Literal["PA", "LP"]


@dataclass(slots=True)
class HighSimilarityFilter(GraphOperator):
    k: float = 0.5
    method: Method = "PA"
    supported_modes = ["similarity"]

    @staticmethod
    def _calculate_edge_similarities(G, method):
        """Calculate similarities for all edges in graph G using similarity function S.

        Parameters
        ----------
        G NetworkX graph
        """
        # for each edge (u, v) in E do:
        # temprarily remove edge (u, v) from G
        # calculate similarity S(u, v)
        # restore (u, v) to G
        # assign S(u, v) to edge (u, v)
        # return G
        epsilon = 0.01
        graph = G.to_networkx()
        A = nx.to_numpy_array(graph)
        A2 = A @ A
        A3 = A2 @ A
        for u, v in graph.edges():
            graph.remove_edge(u, v)
            if method == "PA":
                s = nx.preferential_attachment(graph, [(u, v)])
                p = next(iter(s))[2]
            elif method == "LP":
                # https://www.sciencedirect.com/science/article/pii/S0378437120300856?via%3Dihub
                p = A2[u, v] + epsilon * A3[u, v]

            graph.add_edge(u, v, similarity=p)

        return graph

    def _directed_filter(self, G: Graph) -> Graph:
        pass

    def _undirected_filter(self, G):
        """Select top k% edges from graph G"""
        # initialize E <- empty set
        # sort edges E in descending order based on S(u, v)
        # select top k% edges from E to form sorted E'
        # E' <- selected edges
        # return E'
        n_nodes = G.n_nodes
        G = self._calculate_edge_similarities(G, self.method)

        edge_similarities = {}
        for u, v, data in sorted(G.edges(data=True), key=lambda x: x[2]["similarity"]):
            edge_similarities[(u, v)] = data["similarity"]

        select_count = int(len(edge_similarities) * self.k)

        selected_edges = dict(list(edge_similarities.items())[:select_count])
        new_graph = nx.Graph()
        new_graph.add_weighted_edges_from((u, v, w) for (u, v), w in selected_edges.items())
        new_graph.add_nodes_from(range(n_nodes))

        new_graph = nx.to_scipy_sparse_array(new_graph)
        new_graph = Graph.from_dense(new_graph, mode="similarity")
        return new_graph

    def apply(self, G):
        self._check_mode_supported(G)
        if G.directed:
            return self._directed_filter(G)
        else:
            return self._undirected_filter(G)
