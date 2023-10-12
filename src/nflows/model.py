from typing import List, Tuple
from .flows import Flow


class Model:
    def __init__(self, flows: List[Flow]):
        # Check that all flows have the same shapes
        self._shape = flows[0].shape
        for flow in flows:
            assert isinstance(flow, Flow)
            assert flow.shape == self._shape

        self._flows = flows

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @property
    def flows(self) -> List[Flow]:
        return self._flows

    def forward(self, X: np.ndarray) -> np.ndarray:
        for flow in self.flows:
            X = flow.forward(X)
        return X

    def invert(self, Y: np.ndarray) -> np.ndarray:
        for flow in self.flows:
            Y = flow.invert(Y)
        return Y

    def generate(self, n: int) -> np.ndarray:
        Y = np.random.normal(size=(n, *self.shape[1:])).astype(DTYPE)
        return self.invert(Y)
