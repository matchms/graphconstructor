from dataclasses import dataclass
from ..graph import Graph
from .base import GraphOperator
from distanceclosure.dijkstra import single_source_dijkstra_path_length
from networkx.algorithms.shortest_paths.weighted import _weight_function
from heapq import heappush, heappop
from itertools import count
import networkx as nx
from typing import Literal

def single_source_dijkstra_path_length(G, source, weight_function, paths=None, disjunction=sum):
    """Uses (a custom) Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for path.

    weight_function: function
        Function with (u, v, data) input that returns that edges weight

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node.
        If None, paths are not stored.

    disjunction: function (default=sum)
        Whether to sum paths or use the max value.
        Use `sum` for metric and `max` for ultrametric.

    Returns
    -------
    distance : dictionary
        A mapping from node to shortest distance to that node from one
        of the source nodes.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Note
    -----
    The optional predecessor and path dictionaries can be accessed by
    the caller through the original paths objects passed
    as arguments. No need to explicitly return paths.

    """
    G_succ = G._succ if G.is_directed() else G._adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not in G")
    seen[source] = 0
    push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        for u, e in G_succ[v].items():
            cost = weight_function(v, u, e)
            if cost is None:
                continue
            vu_dist = disjunction([dist[v], cost])
            if u in dist:
                u_dist = dist[u]
                if vu_dist < u_dist:
                    raise ValueError("Contradictory paths found:", "negative weights?")
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
    return dist

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
        pass
        
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