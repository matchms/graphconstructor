from abc import ABC, abstractmethod
from ..graph import Graph


class GraphOperator(ABC):
    """Base class for graph operators."""
    supported_modes = []  # Specify supported modes from ["distance", "similarity"]

    """Pure transform: Graph -> Graph."""
    @abstractmethod
    def apply(self, G: Graph) -> Graph: ...

    def _check_mode_supported(self, G: Graph):
        if G.mode not in self.supported_modes:
            raise ValueError(f"{self.__class__.__name__} only supports modes: {self.supported_modes}, got {G.mode}")
        