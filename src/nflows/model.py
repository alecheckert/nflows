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

        # Map model parameters into indices of a linear array
        n_parameters = 0
        self.parameter_map = {}
        for i, flow in enumerate(flows):
            for k, v in flow.parameters.items():
                self.parameter_map[(i, k)] = (n_parameters, n_parameters + v.size)
                n_parameters += v.size
        self._n_parameters = n_parameters
        self._flows = flows

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @property
    def flows(self) -> List[Flow]:
        return self._flows

    @property
    def n_parameters(self) -> int:
        return self._n_parameters

    def get_parameters(self, flat: bool = False) -> np.ndarray:
        """Return all model parameters.

        Parameters
        ----------
        flat    :   return all parameters as a single linear array
                    rather than a dict. Makes a copy of parameters.

        Returns
        -------
        if flat:
            1D ndarray of shape (self.n_parameters,), copies of
            all model parameters
        else:
            dict keyed by (flow index, parameter name); values are
            references to the underlying parameters
        """
        if flat:
            params = np.zeros(self.n_parameters, dtype=DTYPE)
            for (i, k), (s0, s1) in self.parameter_map.items():
                params[s0:s1] = self.flows[i].parameters[k]
            return params
        else:
            return {
                (i, k): self.flows[i].parameters[k] for (i, k) in self.parameter_map
            }

    def set_parameters(self, params: np.ndarray):
        """Set all parameters of this Model. Input should be
        a linear array of length *self.n_parameters*."""
        assert params.shape == (self.n_parameters,)
        for (i, k), (s0, s1) in self.parameter_map.items():
            self.flows[i].parameters[k][:] = params[s0:s1]

    def split_parameters(self, params: np.ndarray) -> dict:
        """Utility for turning a 1D ndarray over model parameters
        or their gradients into a dict keyed by
        (flow_idx, parameter_name)."""
        assert params.shape == (self.n_parameters,)
        return {
            (i, k): params[s0:s1].copy()
            for (i, k), (s0, s1) in self.parameter_map.items()
        }

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Transform some observed data points *X* into their
        latent representations *Y*."""
        for flow in self.flows:
            X, _ = flow.forward(X)
        return X

    def invert(self, Y: np.ndarray) -> np.ndarray:
        """Given some data points from the latent space *Y*,
        transform into the data space *X*."""
        for flow in self.flows[::-1]:
            Y = flow.invert(Y)
        return Y

    def generate(self, n: int) -> np.ndarray:
        """Simulate *n* data points."""
        Y = np.random.normal(size=(n, *self.shape[1:])).astype(DTYPE)
        return self.invert(Y)

    def backward(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the derivatives of all parameters in this Model
        with respect to the mean negative log likelihood of all data
        points.

        For a normalizing flow, this can be expressed

            -log p(X) = -log p(Y) - log|detJ1| - ... - log|detJN|

        where X are the observed data points, Y is their latent space
        representation, and J1, ..., JN are the Jacobians for each
        (forward) transformation in the Model.

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
        dL_dpars = np.zeros(self.n_parameters, dtype=X.dtype)
        for i in list(range(len(self.flows)))[::-1]:
            flow = self.flows[i]
            dL_dX, _, dL_dpars_i = flow.backward(values[i], dL_dX)
            for k in dL_dpars_i:
                s0, s1 = self.parameter_map[(i, k)]
                dL_dpars[s0:s1] = dL_dpars_i[k]

        return L, dL_dX, dL_dpars
