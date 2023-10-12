"""Implementations of flow layers."""
from abc import ABC, abstractmethod
import numpy as np


DTYPE = np.float32


class Flow(ABC):
    """Represents an invertible operation Y = f(X)."""

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Transform samples of random variable X into
        samples of random variable Y."""

    @abstractmethod
    def invert(self, Y: np.ndarray) -> np.ndarray:
        """Transform samples of random variable Y into
        samples of random variable X."""


class PlanarFlow(Flow):
    def __init__(self, n: int, w=None, v=None, b=None):
        self.n = n
        if w is None:
            w = np.random.normal(scale=np.sqrt(1 / n), size=n).astype(DTYPE)
        if v is None:
            v = np.random.normal(scale=np.sqrt(1 / n), size=n).astype(DTYPE)
        if b is None:
            b = 0.0
        assert len(w.shape) == len(v.shape) == 1
        assert w.shape == v.shape
        self.w = w
        self.v = v
        self.b = b

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert len(X.shape) == 2
        assert X.shape[1] == self.n
        w = self.w
        v = self.v
        b = self.b
        Y = X + np.tanh(X @ w + b)[:, None] * v
        return Y

    def invert(self, Y: np.ndarray) -> np.ndarray:
        assert len(Y.shape) == 2
        assert Y.shape[1] == self.n
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
