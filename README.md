# graphconstructor

Thie library is made to build sparse graphs from:
- Distance or similarity matrices (e.g., numpy arrays)
- precomputed kNN results `(distances, indices)`
- ANN indexes (e.g., PyNNDescent)

## General design

We distinguish `graph constructors` which will build a graph based on the given similarity- or distance-based input.

`similarity matrix` --> `graph constructor` --> `graph pruner` and/or `graph expander` ?

### Graph Constructors:
- kNN graph with optional mutual links and directed/undirected modes
- ε-ball neighborhood (distance < ε or similarity > τ)


### Quickstart
```python
import numpy as np
from graphconstructor import build_knn_graph, build_epsilon_ball_graph

# Create dummy distance matrix
D = np.random.rand(10, 10)
D = (D + D.T)/2
np.fill_diagonal(D, 0.0)

# Construct graph from distance matrix
G1 = build_epsilon_ball_graph(matrix=D, mode="distance", threshold=0.5)
G2 = build_knn_graph(matrix=D, mode="distance", k=5, mutual=True)
```
