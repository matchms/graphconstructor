from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..graph import Graph


class GraphOperator(ABC):
    """Pure transform: Graph -> Graph."""
    @abstractmethod
    def apply(self, G: Graph) -> Graph: ...

@dataclass(slots=True)
class InPlaceConfig:
    """If you ever want to offer in-place variants later."""
    allow_inplace: bool = False
