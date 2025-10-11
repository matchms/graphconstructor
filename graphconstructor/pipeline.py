from dataclasses import dataclass
from .graph import Graph
from .operators import GraphOperator


@dataclass(slots=True)
class Pipeline:
    """Basic first pipeline class to chain graph operators.
    """
    operators: tuple[GraphOperator, ...]
    def apply(self, G: Graph) -> Graph:
        for op in self.operators:
            G = op.apply(G)
        return G

    def then(self, op: GraphOperator) -> "Pipeline":
        return Pipeline(self.operators + (op,))
