from dataclasses import dataclass
from ..graph import Graph
from .base import GraphOperator
from distanceclosure.dijkstra import single_source_dijkstra_path_length
from networkx.algorithms.shortest_paths.weighted import _weight_function
import networkx as nx
from typing import Literal

Mode = Literal["distance", "similarity"]

@dataclass(slots=True)
class MetricDistanceFilter(GraphOperator):
    """
    Metric Distance Backbone Filter for similarity graphs.
    Code: https://github.com/CASCI-lab/distanceclosure/blob/master/distanceclosure/backbone.py

    Parameters
    ----------
    weight : str, optional
        Edge property containing distance values, by default 'weight'
    distortion : bool, optional
        Whether to compute and return distortion values, by default False
    verbose : bool, optional
        Prints statements as it computes, by default False
    """
    weight: str = 'weight'
    distortion: bool = False
    verbose: bool = False
    mode: Mode = "distance"
    supported_modes = ["distance", "similarity"]

    @staticmethod
    def _compute_distortions(D: GraphOperator, B, weight='weight', disjunction=sum):
        G = D.copy()
        
        G.remove_edges_from(B.edges())
        weight_function = _weight_function(B, weight)

        svals = dict()        
        for u in G.nodes():
            metric_dist = single_source_dijkstra_path_length(B, source=u, weight_function=weight_function, disjunction=disjunction)
            for v in G.neighbors(u):
                svals[(u, v)] = G[u][v][weight]/metric_dist[v]
        
        return svals
        
    def _directed_filter(self, G: Graph) -> Graph:
        raise NotImplementedError(
            "MetricDistanceFilter is defined only for undirected graphs."
        )
        
    def _undirected_filter(self, D):          
        disjunction = sum
        
        D = D.to_networkx()
        G = D.copy()
        weight_function = _weight_function(G, self.weight)
        
        if self.verbose:
            total = G.number_of_nodes()
            i = 0
        
        for u, _ in sorted(G.degree(weight=self.weight), key=lambda x: x[1]):
            if self.verbose:
                i += 1
                per = i/total
                print("Backbone: Dijkstra: {i:d} of {total:d} ({per:.2%})".format(i=i, total=total, per=per))
            
            metric_dist = single_source_dijkstra_path_length(G, source=u, weight_function=weight_function, disjunction=disjunction)
            for v in list(G.neighbors(u)):
                if metric_dist[v] < G[u][v][self.weight]:
                    G.remove_edge(u, v)
        

        sparse_adj = nx.to_scipy_sparse_array(G)
        if self.distortion:
            svals = self._compute_distortions(D, G, weight=self.weight, disjunction=disjunction)         
            return Graph(sparse_adj, False, True, self.mode), svals
        else:
            return Graph(sparse_adj, False, True, self.mode)
    
    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)
        if G.directed:
            return self._directed_filter(G)
        else:
            return self._undirected_filter(G)