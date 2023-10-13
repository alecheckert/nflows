from typing import List, Tuple
from .constants import DTYPE, EPSILON
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
        """Transform some observed data points *X* into their
        latent representations *Y*."""
        for flow in self.flows:
            X = flow.forward(X)
        return X

    def invert(self, Y: np.ndarray) -> np.ndarray:
        """Given some data points from the latent space *Y*,
        transform into the data space *X*."""
        for flow in self.flows:
            Y = flow.invert(Y)
        return Y

    def generate(self, n: int) -> np.ndarray:
        """Simulate *n* data points."""
        Y = np.random.normal(size=(n, *self.shape[1:])).astype(DTYPE)
        return self.invert(Y)

    def backward(self, X: np.ndarray) -> np.ndarray:
        """Run a full forward and backward propagation on a set
        of data points.

        Parameters
        ----------
        X   :   ndarray of shape (n, ...) where *n* is the number
                of data points.

        Returns
        -------
        0:  ndarray of shape *X.shape*, latent representation of
            each input datum;
        1:  ndarray of shape (n,), value of the loss for each
            input datum;
        ...
        """
        if len(X.shape) == 1:
            X = X[None, :]
        n = X.shape[0]
        d = X.shape[1]

        # Forward evaluation
        Y = X.copy()
        for flow in self.flows:
            Y = flow.forward(Y)

        # Negative log likelihood w.r.t. to unit Gaussian
        # in latent space
        L = 0.5 * (Y**2).sum(axis=1) + (d/2)*np.log(2*np.pi)

        # Partial derivatives of the loss w.r.t. each element
        # of the latent vector
        dloss_dX = Y.copy()

        # Backpropagate through all parameters
        for flow in self.flows[::-1]:
            dloss_dX, detjac, dlogdetjac_dX = flow.backward(X, dloss_dX)
            loss -= np.log(np.abs(detjac) + EPSILON)

        # TODO:
        # - return value
        # - store gradients w.r.t. parameters of each Flow
        # - in backprop, use the correct *X* (should be input to that Flow)
