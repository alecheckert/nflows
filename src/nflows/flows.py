"""Implementations of flow layers."""
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple
import numpy as np
from .constants import DTYPE, EPSILON


class Flow(ABC):
    """Represents an invertible operation Y = f(X)."""

    @classmethod
    @abstractmethod
    def from_parameters(cls, shape: tuple, params: np.ndarray):
        """Make an instance of this Flow class from a shape and
        a linear array of parameters. Useful for deserialization."""

    @property
    @abstractmethod
    def shape(self) -> Tuple[int]:
        """Shape of the inputs and outputs to this model.
        First dimension is assumed to encode index in
        batch (e.g. None)."""

    @property
    @abstractmethod
    def parameters(self) -> OrderedDict:
        """Get a dict of all model parameters, keyed by
        parameter name."""

    @abstractmethod
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray]:
        """Transform samples of random variable X into
        samples of random variable Y.

        Parameters
        ----------
        X   :   input to this Flow; first dimension is assumed
                to encode different samples

        Returns
        -------
        0:  Y, output of this flow
        1:  Jacobian determinant for each sample; ndarray of
            shape (X.shape[0],)
        """

    @abstractmethod
    def invert(self, Y: np.ndarray) -> np.ndarray:
        """Transform samples of random variable Y into
        samples of random variable X."""

    @abstractmethod
    def backward(
        self, X: np.ndarray, dL_dY: np.ndarray, normalize: bool = True
    ) -> tuple:
        """Given partial derivatives of a scalar loss with
        respect to the output of this Flow, propagate these
        partial derivatives to the Flow's input and its
        parameters.

        Parameters
        ---------
        X           :   input to this Flow
        dL_dY       :   partial derivatives of a scalar loss
                        with respect to each output element
                        of this loss (shape *X.shape*)
        normalize   :   actually normalize the propagated
                        partial derivatives using the Jac.
                        determinant. If False, we only
                        backpropagate through the output of
                        the Flow, not through its determinant.
                        Useful for tests.

        Returns
        -------
        0:  partial derivatives of loss with respect to each
            element in input X (shape *X.shape*);
        1:  partial derivatives of log absolute Jacobian
            determinant with respect to each element in the
            input (shape *X.shape*);
        2:  dict keyed by parameter name, partial derivatives
            of loss with respect to each parameter (gradient)
        """


class ScalarFlow(Flow):
    def __init__(self, d: int, mean=None, scale=None):
        self.d = d
        if mean is None:
            mean = np.zeros(d, dtype=DTYPE)
        if scale is None:
            scale = np.ones(d, dtype=DTYPE)
        assert len(mean.shape) == len(scale.shape) == 1
        assert mean.shape == scale.shape == (d,)
        self.mean = mean
        self.scale = scale

    @classmethod
    def from_parameters(cls, shape: tuple, params: np.ndarray):
        assert isinstance(shape, tuple)
        assert isinstance(params, np.ndarray)
        assert len(params.shape) == 1
        assert len(shape) == 1
        d = shape[0]
        assert params.shape == (2 * d,)
        mean = params[:d]
        scale = params[d:]
        return cls(shape[0], mean=mean, scale=scale)

    @property
    def shape(self) -> Tuple[int]:
        return (None, self.d)

    @property
    def parameters(self) -> OrderedDict:
        return OrderedDict(mean=self.mean, scale=self.scale)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray]:
        assert len(X.shape) == 2
        assert X.shape[1] == self.d
        Y = X * self.scale + self.mean
        detjac = np.full(X.shape[0], np.prod(self.scale), dtype=X.dtype)
        return Y, detjac

    def invert(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.mean) / (self.scale + EPSILON)

    def backward(
        self, X: np.ndarray, dL_dY: np.ndarray, normalize: bool = True
    ) -> tuple:
        assert len(X.shape) == len(dL_dY.shape) == 2
        assert X.shape[1] == dL_dY.shape[1] == self.d
        a = self.scale
        b = self.mean
        dL_dX = dL_dY * a
        dlogdetjac_dX = np.zeros_like(X)
        # if normalize:  # meaningless for ScalarFlow
        #     dL_dX -= dlogdetjac_dX
        dL_dpars = {}
        dL_dpars["mean"] = dL_dY.mean(axis=0)
        dL_dpars["scale"] = (dL_dY * X).mean(axis=0) - 1 / (a + np.sign(a) * EPSILON)
        return dL_dX, dlogdetjac_dX, dL_dpars


class PlanarFlow(Flow):
    def __init__(self, d: int, w=None, v=None, b=None):
        self.d = d
        if w is None:
            w = np.random.normal(scale=np.sqrt(1 / d), size=d).astype(DTYPE)
        if v is None:
            v = np.random.normal(scale=np.sqrt(1 / d), size=d).astype(DTYPE)
        if b is None:
            b = np.array([0.0])
        assert len(w.shape) == len(v.shape) == len(b.shape) == 1
        assert w.shape == v.shape == (d,)
        assert b.shape == (1,)
        self.w = w
        self.v = v
        self.b = b

    @classmethod
    def from_parameters(cls, shape: tuple, params: np.ndarray):
        assert isinstance(shape, tuple)
        assert isinstance(params, np.ndarray)
        assert len(shape) == 1
        assert len(params.shape) == 1
        d = shape[0]
        assert params.shape == (2 * d + 1,)
        w = params[:d]
        v = params[d : 2 * d]
        b = params[2 * d :]
        return cls(d, w=w, v=v, b=b)

    @property
    def shape(self) -> Tuple[int]:
        return (None, self.d)

    @property
    def parameters(self) -> OrderedDict:
        return OrderedDict(w=self.w, v=self.v, b=self.b)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray]:
        assert len(X.shape) == 2
        assert X.shape[1] == self.d
        w = self.w
        v = self.v
        b = self.b
        tanha = np.tanh(X @ w + b)
        Y = X + tanha[:, None] * v
        wv = w @ v
        dtanha_da = 1 - tanha**2
        detjac = 1 + wv * dtanha_da
        return Y, detjac

    def invert(self, Y: np.ndarray) -> np.ndarray:
        assert len(Y.shape) == 2
        assert Y.shape[1] == self.d
        n = Y.shape[0]
        alpha = np.zeros(n, dtype=Y.dtype)
        converged = np.zeros(n, dtype=bool)
        w = self.w
        v = self.v
        b = self.b
        wv = w @ v
        assert wv > -1  # invertibility criterion
        wy = Y @ w
        while not converged.all():
            sigma = np.tanh(alpha + b)
            dsigma_dalpha = 1.0 - sigma**2
            dg_dalpha = wv * dsigma_dalpha + 1.0
            g = alpha + wv * sigma - wy
            update = -g / (2 * dg_dalpha)
            converged = np.abs(update) < 1e-6
            alpha += update
        X = Y - np.tanh(alpha + b)[:, None] * v
        return X

    def backward(
        self, X: np.ndarray, dL_dY: np.ndarray, normalize: bool = True
    ) -> tuple:
        assert len(X.shape) == len(dL_dY.shape) == 2
        assert X.shape[1] == dL_dY.shape[1] == self.d
        w = self.w
        v = self.v
        b = self.b

        # Gradient of loss with respect to inputs to this Flow
        a = X @ w + b
        tanha = np.tanh(a)
        dtanha_da = 1 - tanha**2
        ddtanha_dda = -2 * tanha * dtanha_da
        dyv = dL_dY @ v
        dL_dX = dL_dY + (dtanha_da * dyv)[:, None] * w

        # Jacobian determinant
        wv = w @ v
        detjac = 1 + wv * dtanha_da

        # Gradient of loss with respect to log Jacobian determinant
        dlogdetjac_dX = ((wv * ddtanha_dda) / (detjac + EPSILON))[:, None] * w
        if normalize:
            dL_dX -= dlogdetjac_dX

        # Gradient of loss with respect to all parameters of this Flow
        dL_dpars = {}
        invdetjac = 1 / (detjac + np.sign(detjac) * EPSILON)
        dL_dpars["w"] = (
            ((dL_dY @ v) * dtanha_da)[:, None] * X
            - (
                (dtanha_da * invdetjac)[:, None] * v
                + wv * (ddtanha_dda * invdetjac)[:, None] * X
            )
        ).mean(axis=0)
        dL_dpars["v"] = (
            tanha[:, None] * dL_dY - (dtanha_da * invdetjac)[:, None] * w
        ).mean(axis=0)
        dL_dpars["b"] = ((dL_dY @ v) * dtanha_da - wv * ddtanha_dda * invdetjac)[
            :, None
        ].mean(axis=0)

        return dL_dX, dlogdetjac_dX, dL_dpars


FLOWS = {f.__name__: f for f in [ScalarFlow, PlanarFlow]}
