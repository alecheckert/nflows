import numpy as np
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
            X, _ = flow.forward(X)
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

        # Loss for each datum. For a normalizing flow the
        # loss comprises two parts: (1) the negative log
        # Jacobian determinants for each transformation and
        # (2) the negative log likelihood of the latent space
        # representation. We accumulate the factors for (1)
        # during forward evaluation.
        L = np.zeros(n, dtype=X.dtype)

        # Keep track of intermediate values for each layer
        values = {0: X}

        # Forward evaluation
        Y = X.copy()
        for i, flow in enumerate(self.flows):
            Y, dJ = flow.forward(Y)
            values[i + 1] = Y
            L -= np.log(np.abs(dJ) + EPSILON)

        # Component of loss due to negative log likelihood of
        # latent space representation
        L += 0.5 * (Y**2).sum(axis=1) + (d / 2) * np.log(2 * np.pi)

        # Partial derivatives of the loss w.r.t. each element
        # of the latent vector
        dL_dX = Y.copy()

        # Backpropagate through all parameters
        dL_dpars = {}
        for i in list(range(len(self.flows)))[::-1]:
            flow = self.flows[i]
            dL_dX, _, dL_dpars_i = flow.backward(values[i], dL_dX)
            dL_dpars[i] = dL_dpars_i

        return L, dL_dX, dL_dpars
