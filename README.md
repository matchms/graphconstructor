# graphconstructor

Build sparse graphs from:
- distance or similarity matrices
- precomputed kNN results `(distances, indices)`
- ANN indexes (e.g., PyNNDescent)

### Graph Constructors:
- kNN graph with optional mutual links and directed/undirected modes
- ε-ball neighborhood (distance < ε or similarity > τ)


### Quickstart
```python
import numpy as np
from graphconstructor import build_knn_graph, build_epsilon_ball_graph

# From a distance matrix
D = np.random.rand(10, 10); D = (D + D.T)/2; np.fill_diagonal(D, 0.0)
G1 = build_epsilon_ball_graph(matrix=D, mode="distance", threshold=0.5)
G2 = build_knn_graph(matrix=D, k=5, mutual=True)
```