![GitHub License](https://img.shields.io/github/license/matchms/graphconstructor)
[![PyPI](https://img.shields.io/pypi/v/graphconstructor?color=teal)](https://pypi.org/project/graphconstructor/)
![[GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/matchms/graphconstructor/CI_build_and_matrix_tests.yml?branch=main)](https://img.shields.io/github/actions/workflow/status/matchms/graphconstructor/CI_build_and_matix_tests.yml
)


<img src="/docs/graphconstructor_logo.png" alt="Alt Text" width="600">

# graphconstructor
Fast, NumPy/SciPy-centric tools to **build and refine large sparse graphs from distances/similarities**.
Use one of the provided **importers** to *get* a first graph from a distance/similarity array, kNN results, or ANN indices.
This will usually be followed by a custom combination of one or multiple **operators** that will transform the graph, typically in the form of *sparsification* (also termed *backboning* or *pruning*).

`graphconstructor` further provides
* Optional exporters to **NetworkX** / **python-igraph** for using their powerful graph analysis and layouting methods!
* Very basic graph visualizations (for more fancy options --> export the graph and use one of the available tools for graph visualization such as [Gephi](https://gephi.org/), [Cytoscape](https://cytoscape.org/), or others.

---

## Key elements of `graphconstructor`:

* **Graph** class (`graphconstructor.Graph`)
  Central graph class in graphconstructor. The actual graph is stored as a sparse adjacency matrix `graph.adj` and can represent a **directed** or **undirected** graph (either as a **weighted** or **unweighted** graph).
  A `graph` object also contains node metadata at `graph.metadata` in the form of a pandas DataFrame.

  * Editing: `drop(...)`, `sorted_by(...)`
  * Exporters: `to_networkx()`, `to_igraph()`

* **Importers** (`graphconstructor.importers`)
  Functions to construct a first graph from various import formats. This is only meant as a first step in the full "graph construction" process and will usually be followed by one or multiple **operator steps**.  
  All importers will return a `graphconstructor.Graph` object.

  * `from_csr`, `from_dense`
  * `from_knn(indices, distances, ...)`
  * `from_ann(ann, query_data, k, ...)` (supports cached neighbors or `.query`)

* **Operators** (`graphconstructor.operators`)
  The `operators` are the central algorithms for graph construction from similarity or distance metrics.
  Starting from a similarity or distance-based graph with (usually) far too many edges for many purposes (e.g., further analysis or graph visualization), `graphconstructor` provides a range of different methods to sparsify the graph.  
  All operators will take a `graphconstructor.Graph` object as input (as well as optional parameters) and will then also return a (modified) `graphconstructor.Graph` object.

  * `KNNSelector(k, mutual=False, mutual_k=None, mode="distance"|"similarity")`  
    **k-Nearest Neighrbor (KNN)** based edge selections. This will keep only top-*k* neighbors per node. Optionally, it requires **mutual** edges using top-`mutual_k`.
  * `WeightThreshold(threshold, mode="distance"|"similarity")`  
    Basic (or "naive") sparsification algorithm that simply applies a **global weight threshold**. Only edges with weight `< threshold` (distance) or `> threshold` (similarity) will be kept.
  * `DoublyStochastic(tolerance=1e-5, max_iter=10000)`  
    **Sinkhorn–Knopp** alternating row/column normalization to make the adjacency (approximately) **doubly stochastic** without densifying (CSR-only). Can be useful as a normalization step before backboning/thresholding.  
    Ref: Sinkhorn (1964); discussed in Coscia, "The Atlas for the Inspiring Network Scientist" (2025).
  * `DisparityFilter(alpha=0.05, rule="or"|"and")`  
    **Disparity Filter** algorithm for graphs with continuous weights. Tests each edge against a node-wise null (Dirichlet/Beta split of strength). Undirected edges can be kept if either (“or”, default) or both (“and”) endpoints deem them significant.  
    Ref: Serrano, Boguñá, Vespignani, "Extracting the multiscale backbone of complex weighted networks", PNAS 2009.
  * `LocallyAdaptiveSparsification(alpha=0.05, rule="or"|"and")`  
    Implementation of the **Locally Adaptive Network Sparsification (LANS)** algorithm, which does not assume any particular null model. Instead, the distribution of similarity weights is used to determine and then retain statistically significant edges.  
    Ref: Foti, Hughes, Rockmore, "Nonparametric Sparsification of Complex Multiscale Networks", 2011, https://doi.org/10.1371/journal.pone.0016431
  * `MarginalLikelihoodFilter(alpha, float_scaling=20, assume_loopless=False)`  
    Dianati’s **Marginal Likelihood Filter (MLF)** for integer weights. Uses configuration-like binomial null preserving strengths on average; computes upper-tail p-values and keeps edges with ($p \le \alpha$). Supports float → integer casting strategies.  
    Ref: Dianati, "Unwinding the hairball graph: Pruning algorithms for weighted complex networks", Phys. Rev. E 2016, https://link.aps.org/doi/10.1103/PhysRevE.93.012304
  * `NoiseCorrected(delta=1.64, derivative="constant"|"full")`  
    **Noise-Corrected (NC) backbone**. Computes symmetric lift relative to a pairwise null, estimates variance via a binomial model with **Beta** prior (Bayesian shrinkage), and keeps edges exceeding ( $\delta$ ) standard deviations. `derivative="full"` matches the paper’s delta-method including ($d\kappa/dn$); `"constant"` is a        simpler, fast variant.  
    Ref: Coscia & Neffke, "Network Backboning with Noisy Data", 2017, https://ieeexplore.ieee.org/document/7929996

---

## Installation

```bash
pip install graphconstructor
```

---

## Quickstart

### 1) Build a graph (importers)

```python
import numpy as np
from graphconstructor.importers import from_dense  # or use other options, e.g. from_knn, from_ann

# Symmetric distance matrix (example)
D = np.random.rand(100, 100) ** 0.5
D = (D + D.T) / 2
np.fill_diagonal(D, 0.0)

# Import (from dense array)
G0 = from_dense(D, directed=False, mode="distance")
```

### 2) Refine a graph (operators)
Some operators can work on both distance and similarity based graphs, such as kNN and a simple weight thresholding method:

```python
from graphconstructor.operators import KNNSelector, WeightThreshold

# Keep only the top-10 mutual neighbors (mutuality checked within top-20)
G_refined = KNNSelector(k=5, mutual=True, mutual_k=20, mode="distance").apply(G0)

# Prune weak edges (keep distance < 0.3)
G_pruned = WeightThreshold(threshold=0.3, mode="distance").apply(G_refined)
```

Other operators require particular either similarity or distance values as weights. In such cases it can be necessary to switch from distance to similarity measures (or vice versa).

```python
from graphconstructor.operators import NoiseCorrected

# There are various method for distance/similarity conversions (including using a custom function)
G0_sim = G0.convert_mode("similarity", method="exp")

# Once converted to similarities, other operators become available 
G_refined = NoiseCorrected().apply(G0_sim)
```


### 3) Export when needed

```python
nx_graph = G_pruned.to_networkx()   # nx.Graph or nx.DiGraph
ig_graph = G_pruned.to_igraph()     # igraph.Graph
```
