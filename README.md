# graphconstructor

Fast, NumPy/SciPy-centric tools to **build and refine large sparse graphs** from distances/similarities, kNN results, or ANN indices — without paying the conversion tax of pure-Python graph libraries.

* Core storage: **SciPy CSR** adjacency
* Node attributes: **pandas.DataFrame** (one row per node)
* Clean separation of **importers** (how you *get* a graph) and **operators** (how you *transform* it)
* Optional exporters to **NetworkX** / **python-igraph** for using their powerful graph analysis and layouting methods!

---

## Features

* **Graph** class (`graphconstructor.graph.Graph`)

  * CSR adjacency, `directed` / `weighted` flags
  * Node metadata as a `DataFrame` (optional `"name"` column)
  * Editing: `drop(...)`, `sorted_by(...)`
  * Exporters: `to_networkx()`, `to_igraph()`

* **Importers** (`graphconstructor.importers`)

  * `from_csr`, `from_dense`
  * `from_knn(indices, distances, ...)`
  * `from_ann(ann, query_data, k, ...)` (supports cached neighbors or `.query`)

* **Operators** (`graphconstructor.operators`)

  * `KNNSelector(k, mutual=False, mutual_k=None, mode="distance"|"similarity")`
    Keep top-*k* neighbors per node; optionally require **mutual** edges using top-`mutual_k`.
  * `WeightThreshold(threshold, mode="distance"|"similarity")`
    Keep edges with weight `< ε` (distance) or `> τ` (similarity).

Operators are **pure**: they take a `Graph`, return a new `Graph` (no densification).

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
G0 = from_dense(D, directed=False)
```

### 2) Refine a graph (operators)

```python
from graphconstructor.operators import KNNSelector, WeightThreshold

# Keep only the top-10 mutual neighbors (mutuality checked within top-20)
G_refined = KNNSelector(k=5, mutual=True, mutual_k=20, mode="distance").apply(G0)

# Prune weak edges (keep distance < 0.3)
G_pruned = WeightThreshold(threshold=0.3, mode="distance").apply(G_refined)
```

### 3) Export when needed

```python
nx_graph = G_pruned.to_networkx()   # nx.Graph or nx.DiGraph
ig_graph = G_pruned.to_igraph()     # igraph.Graph
```
