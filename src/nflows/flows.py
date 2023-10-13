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
    def backward(
        self, X: np.ndarray, dL_dY: np.ndarray, normalize: bool = True
    ) -> Tuple[np.ndarray]:
        """Given partial derivatives of a scalar loss with
        respect to the output of this Flow, propagate these
        partial derivatives to the Flow's input.

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
        1:  Jacobian determinant for each element in the input
            (shape (X.shape[0],));
        2:  partial derivatives of log absolute Jacobian
            determinant with respect to each element in the
            input (shape *X.shape*);
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

    def backward(
        self, X: np.ndarray, dL_dY: np.ndarray, normalize: bool = True
    ) -> Tuple[np.ndarray]:
        assert len(X.shape) == len(dL_dY.shape) == 2
        assert X.shape[1] == dL_dY.shape[1] == self.d
        a = self.scale
        b = self.mean
        dL_dX = dL_dY * a
        detjac = np.full(X.shape[0], np.prod(self.scale), dtype=X.dtype)
        dlogdetjac_dX = np.zeros_like(X)
        # if normalize:  # meaningless for ScalarFlow
        #     dL_dX -= dlogdetjac_dX
        return dL_dX, detjac, dlogdetjac_dX


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

    def backward(
        self, X: np.ndarray, dL_dY: np.ndarray, normalize: bool = True
    ) -> Tuple[np.ndarray]:
        assert len(X.shape) == len(dL_dY.shape) == 2
        assert X.shape[1] == dL_dY.shape[1] == self.d
        w = self.w
        v = self.v
        b = self.b
        a = X @ w + b
        tanha = np.tanh(a)
        dtanha_da = 1 - tanha**2
        ddtanha_dda = -2 * tanha * dtanha_da
        dyv = dL_dY @ v
        dL_dX = dL_dY + (dtanha_da * dyv)[:, None] * w
        wv = w @ v
        detjac = 1 + wv * dtanha_da
        dlogdetjac_dX = ((wv * ddtanha_dda) / (detjac + EPSILON))[:, None] * w
        if normalize:
            dL_dX -= dlogdetjac_dX
        return dL_dX, detjac, dlogdetjac_dX


FLOWS = {f.__name__: f for f in [ScalarFlow, PlanarFlow]}
