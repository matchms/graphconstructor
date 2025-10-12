from abc import ABC, abstractmethod
from ..graph import Graph


class GraphOperator(ABC):
    """Pure transform: Graph -> Graph."""
    @abstractmethod
    def apply(self, G: Graph) -> Graph: ...
