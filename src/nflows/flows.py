"""Implementations of flow layers."""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from .constants import DTYPE, EPSILON


class Flow(ABC):
    """Represents an invertible operation Y = f(X)."""

    @property
    @abstractmethod
    def shape(self) -> Tuple[int]:
        """Shape of the inputs and outputs to this model.
        First dimension is assumed to encode index in
        batch (e.g. None)."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """Get a dict of all model parameters, keyed by
        parameter name."""

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Transform samples of random variable X into
        samples of random variable Y."""

    @abstractmethod
    def invert(self, Y: np.ndarray) -> np.ndarray:
        """Transform samples of random variable Y into
        samples of random variable X."""

    @abstractmethod
    def backward(self, X: np.ndarray, dL_dY: np.ndarray) -> np.ndarray:
        """Given partial derivatives of a scalar loss with
        respect to the output of this Flow, evaluate the
        partial derivatives of the loss with respect to the
        input of this Flow."""


class LinearFlow(Flow):
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

    @property
    def shape(self) -> Tuple[int]:
        return (None, self.d)

    @property
    def parameters(self) -> dict:
        return {"mean": self.mean, "scale": self.scale}

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert len(X.shape) == 2
        assert X.shape[1] == self.d
        return X * self.scale + self.mean

    def invert(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.mean) / (self.scale + EPSILON)

    def backward(self, X: np.ndarray, dL_dY: np.ndarray) -> np.ndarray:
        assert len(X.shape) == len(dL_dY.shape) == 2
        assert X.shape[1] == dL_dY.shape[1] == self.d
        a = self.scale
        b = self.mean
        dL_dX = dL_dY * a
        return dL_dX


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

    @property
    def shape(self) -> Tuple[int]:
        return (None, self.d)

    @property
    def parameters(self) -> Tuple[np.ndarray]:
        return {"w": self.w, "v": self.v, "b": self.b}

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert len(X.shape) == 2
        assert X.shape[1] == self.d
        w = self.w
        v = self.v
        b = self.b
        Y = X + np.tanh(X @ w + b)[:, None] * v
        return Y

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

    def backward(self, X: np.ndarray, dL_dY: np.ndarray) -> np.ndarray:
        assert len(X.shape) == len(dL_dY.shape) == 2
        assert X.shape[1] == dL_dY.shape[1] == self.d
        w = self.w
        v = self.v
        b = self.b
        a = X @ w + b
        tanha = np.tanh(a)
        dtanha_da = 1 - tanha**2
        dyv = dL_dY @ v
        dL_dX = dL_dY + (dtanha_da * dyv)[:, None] * w

        # # Contribution from log Jacobian determinant
        # vw = v @ w
        # ddtanha_dda = -2 * tanha * dtanha_da
        # detJ = 1 + dtanha_da * vw
        # dlogdetjac_dX = (-vw * ddtanha_dda / (detJ + EPSILON))[:, None] * w
        # dL_dX += dlogdetjac_dX

        return dL_dX

    def detjac(self, X: np.ndarray) -> np.ndarray:
        """Compute the determinant of the Jacobian for each datum,
        returning an 1D ndarray of shape (X.shape[0],)."""
        assert len(X.shape) == 2
        assert X.shape[1] == self.d
        w = self.w
        v = self.v
        b = self.b
        wv = w @ v
        return 1 + wv * (1 - np.tanh(X @ w + b) ** 2)

    def logdetjac(self, X: np.ndarray) -> np.ndarray:
        return np.log(np.abs(self.detjac(X)) + EPSILON)

    def dlogdetjac_dX(self, X: np.ndarray) -> np.ndarray:
        """Partial derivatives of the log Jacobian determinant
        with respect to each element of *X*."""
        assert len(X.shape) == 2
        assert X.shape[1] == self.d
        w = self.w
        v = self.v
        b = self.b
        vw = v @ w
        tanha = np.tanh(X @ w + b)
        dtanha_da = 1 - tanha**2
        ddtanha_dda = -2 * tanha * dtanha_da
        detjac = 1 + dtanha_da * vw
        return (vw * ddtanha_dda / (detjac + EPSILON))[:, None] * w


FLOWS = {f.__name__: f for f in [LinearFlow, PlanarFlow]}
